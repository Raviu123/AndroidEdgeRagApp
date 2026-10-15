// JNI bridge between LlamaNative (Kotlin) and llama.cpp.
//
// All text crosses the boundary as UTF-8 byte arrays: JNI's String helpers use "modified UTF-8",
// which corrupts characters outside the BMP (emoji, many math symbols) and aborts under CheckJNI
// when generation stops in the middle of a multi-byte character.

#include <jni.h>
#include <string>
#include <vector>
#include <atomic>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <unistd.h>
#include <android/log.h>
#include "llama.h"
#include "ggml-backend.h"

#define TAG "LlamaJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

namespace {

// Timings of the most recent nativeInference call (read back with nativeLastStats).
struct InferenceStats {
    int64_t prompt_tokens = 0;
    int64_t reused_tokens = 0;   // prompt prefix kept from the previous request's KV cache
    int64_t prefill_ms = 0;      // processing of the new (non-reused) prompt tokens
    int64_t first_token_ms = 0;  // call start -> first generated token sampled
    int64_t decode_ms = 0;       // first token sampled -> generation finished
    int64_t generated_tokens = 0;
};

struct LlamaModelContext {
    llama_model *model = nullptr;
    llama_context *ctx = nullptr;
    int threads_decode = 4;
    int threads_batch = 4;
    std::atomic<bool> cancel{false};
    // Exactly the tokens currently held in sequence 0 of the KV cache (prompt + decoded output).
    std::vector<llama_token> cached_tokens;
    InferenceStats stats;
};

int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void reset_cache(LlamaModelContext *mc) {
    llama_memory_clear(llama_get_memory(mc->ctx), true);
    mc->cached_tokens.clear();
}

// Decodes tokens[from, to) at their own positions; logits only for the last token when [want_last_logits].
// Returns llama_decode's result code (0 = success).
int decode_range(llama_context *ctx, const std::vector<llama_token> &tokens, int from, int to, bool want_last_logits) {
    const int n_batch = 512;
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    int rc = 0;
    for (int i = from; i < to && rc == 0; i += n_batch) {
        const int n_eval = std::min(to - i, n_batch);
        batch.n_tokens = n_eval;
        for (int j = 0; j < n_eval; ++j) {
            batch.token[j] = tokens[i + j];
            batch.pos[j] = i + j;
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j] = want_last_logits && (i + j == to - 1);
        }
        rc = llama_decode(ctx, batch);
    }
    llama_batch_free(batch);
    return rc;
}

struct EmbeddingContext {
    llama_model *model = nullptr;
    llama_context *ctx = nullptr;
    int n_ctx = 512;
};

std::once_flag g_backend_once;
std::mutex g_backend_mutex;
std::string g_cpu_variant;

void throw_java(JNIEnv *env, const char *message) {
    jclass cls = env->FindClass("java/lang/IllegalStateException");
    if (cls) env->ThrowNew(cls, message);
}

std::string to_std_string(JNIEnv *env, jbyteArray bytes) {
    if (!bytes) return {};
    const jsize len = env->GetArrayLength(bytes);
    std::string out(static_cast<size_t>(len), '\0');
    if (len > 0) env->GetByteArrayRegion(bytes, 0, len, reinterpret_cast<jbyte *>(&out[0]));
    return out;
}

jbyteArray to_byte_array(JNIEnv *env, const std::string &s) {
    jbyteArray arr = env->NewByteArray(static_cast<jsize>(s.size()));
    if (arr && !s.empty()) env->SetByteArrayRegion(arr, 0, static_cast<jsize>(s.size()), reinterpret_cast<const jbyte *>(s.data()));
    return arr;
}

// Length of [s] without a trailing incomplete UTF-8 sequence (generation can stop between the bytes of one character).
size_t complete_utf8_length(const std::string &s) {
    size_t i = s.size();
    int continuation = 0;
    while (i > 0 && continuation < 4) {
        const unsigned char c = static_cast<unsigned char>(s[i - 1]);
        if ((c & 0xC0) == 0x80) { continuation++; i--; continue; }
        int expected = 1;
        if ((c & 0xE0) == 0xC0) expected = 2;
        else if ((c & 0xF0) == 0xE0) expected = 3;
        else if ((c & 0xF8) == 0xF0) expected = 4;
        return (expected != continuation + 1) ? i - 1 : s.size();
    }
    return s.size();
}

std::vector<llama_token> tokenize(const llama_vocab *vocab, const std::string &text, bool add_special, bool parse_special) {
    int n = -llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()), nullptr, 0, add_special, parse_special);
    if (n <= 0) return {};
    std::vector<llama_token> tokens(static_cast<size_t>(n));
    const int written = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()), tokens.data(), n, add_special, parse_special);
    if (written < 0) return {};
    tokens.resize(static_cast<size_t>(written));
    return tokens;
}

bool abort_callback(void *data) {
    auto *mc = static_cast<LlamaModelContext *>(data);
    return mc != nullptr && mc->cancel.load();
}

LlamaModelContext *from_handle(jlong handle) { return reinterpret_cast<LlamaModelContext *>(handle); }
EmbeddingContext *embedding_from_handle(jlong handle) { return reinterpret_cast<EmbeddingContext *>(handle); }

#ifdef LLAMA_JNI_BACKEND_DL
bool try_load_backend(const std::string &dir, const std::string &file) {
    if (!dir.empty()) {
        const std::string full = dir + "/" + file;
        if (access(full.c_str(), R_OK) == 0 && ggml_backend_load(full.c_str()) != nullptr) return true;
    }
    // Libraries kept inside the APK (not extracted) are still resolvable by name.
    return ggml_backend_load(file.c_str()) != nullptr;
}
#endif

// Registers the CPU backend exactly once. With dynamic backends the best variant the CPU supports
// is chosen (ggml checks the CPU features of each candidate); safe mode forces the baseline variant.
void ensure_backends(const std::string &lib_dir, bool safe_mode) {
    std::lock_guard<std::mutex> lock(g_backend_mutex);
    if (!g_cpu_variant.empty()) return;
#ifdef LLAMA_JNI_BACKEND_DL
    static const char *variants[] = {
        "android_armv9.2_2", "android_armv9.2_1", "android_armv9.0_1", "android_armv8.6_1",
        "android_armv8.2_2", "android_armv8.2_1", "android_armv8.0_1"
    };
    for (const char *variant : variants) {
        const std::string name(variant);
        if (safe_mode && name != "android_armv8.0_1") continue;
        if (try_load_backend(lib_dir, "libggml-cpu-" + name + ".so")) {
            g_cpu_variant = name;
            break;
        }
    }
    if (g_cpu_variant.empty()) {
        ggml_backend_load_all_from_path(lib_dir.empty() ? nullptr : lib_dir.c_str());
        g_cpu_variant = ggml_backend_dev_count() > 0 ? "auto" : "";
    }
#else
    g_cpu_variant = "static";
#endif
    std::call_once(g_backend_once, [] { llama_backend_init(); });
    LOGI("CPU backend: %s (safe mode: %d)", g_cpu_variant.empty() ? "none" : g_cpu_variant.c_str(), safe_mode ? 1 : 0);
}

} // namespace

#define JNI_FN(name) Java_com_siddu_offlinedocai_ai_llm_LlamaNative_##name

extern "C" JNIEXPORT jstring JNICALL
JNI_FN(nativeInitBackends)(JNIEnv *env, jclass, jstring lib_dir, jboolean safe_mode) {
    const char *dir = lib_dir ? env->GetStringUTFChars(lib_dir, nullptr) : nullptr;
    ensure_backends(dir ? std::string(dir) : std::string(), safe_mode);
    if (dir) env->ReleaseStringUTFChars(lib_dir, dir);
    return env->NewStringUTF(g_cpu_variant.c_str());
}

extern "C" JNIEXPORT jlong JNICALL
JNI_FN(nativeInit)(JNIEnv *env, jclass, jbyteArray model_path, jint context_size, jint thread_count) {
    if (g_cpu_variant.empty()) {
        LOGE("No CPU backend available");
        return 0;
    }
    const std::string path = to_std_string(env, model_path);
    llama_model_params model_params = llama_model_default_params();
    llama_model *model = llama_model_load_from_file(path.c_str(), model_params);
    if (!model) {
        LOGE("Failed to load model: %s", path.c_str());
        return 0;
    }

    // Never ask for more context than the model was trained with.
    const int train_ctx = llama_model_n_ctx_train(model);
    const int n_ctx = (train_ctx > 0) ? std::min<int>(context_size, train_ctx) : context_size;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = static_cast<uint32_t>(n_ctx);
    ctx_params.n_batch = 512;
    ctx_params.n_ubatch = 256;
    ctx_params.n_threads = thread_count;
    ctx_params.n_threads_batch = thread_count;

    llama_context *ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        LOGE("Failed to create llama context (n_ctx=%d)", n_ctx);
        llama_model_free(model);
        return 0;
    }

    auto *mc = new LlamaModelContext();
    mc->model = model;
    mc->ctx = ctx;
    mc->threads_decode = thread_count;
    mc->threads_batch = thread_count;
    llama_set_abort_callback(ctx, abort_callback, mc);

    char desc[128] = {0};
    llama_model_desc(model, desc, sizeof(desc));
    LOGI("Model loaded: %s, n_ctx=%u (train %d), threads=%d, cpu=%s", desc, llama_n_ctx(ctx), train_ctx, thread_count, g_cpu_variant.c_str());
    return reinterpret_cast<jlong>(mc);
}

extern "C" JNIEXPORT void JNICALL
JNI_FN(nativeFree)(JNIEnv *, jclass, jlong handle) {
    LlamaModelContext *mc = from_handle(handle);
    if (!mc) return;
    if (mc->ctx) llama_free(mc->ctx);
    if (mc->model) llama_model_free(mc->model);
    delete mc;
}

extern "C" JNIEXPORT jint JNICALL
JNI_FN(nativeGetContextSize)(JNIEnv *, jclass, jlong handle) {
    LlamaModelContext *mc = from_handle(handle);
    return (mc && mc->ctx) ? static_cast<jint>(llama_n_ctx(mc->ctx)) : 0;
}

extern "C" JNIEXPORT jint JNICALL
JNI_FN(nativeCountTokens)(JNIEnv *env, jclass, jlong handle, jbyteArray text, jboolean is_prompt) {
    LlamaModelContext *mc = from_handle(handle);
    if (!mc || !mc->model) return -1;
    const std::string s = to_std_string(env, text);
    // Prompts are counted exactly as they will be tokenized for inference (BOS + parsed control tokens).
    return static_cast<jint>(tokenize(llama_model_get_vocab(mc->model), s, is_prompt, is_prompt).size());
}

extern "C" JNIEXPORT jbyteArray JNICALL
JNI_FN(nativeGetMetadata)(JNIEnv *env, jclass, jlong handle, jstring key) {
    LlamaModelContext *mc = from_handle(handle);
    if (!mc || !mc->model) return nullptr;
    const char *k = env->GetStringUTFChars(key, nullptr);
    std::string value;
    if (std::string(k) == "tokenizer.chat_template") {
        const char *tmpl = llama_model_chat_template(mc->model, nullptr);
        if (tmpl) value = tmpl;
    } else {
        std::vector<char> buf(256);
        int n = llama_model_meta_val_str(mc->model, k, buf.data(), buf.size());
        if (n >= static_cast<int>(buf.size())) {
            buf.resize(static_cast<size_t>(n) + 1);
            n = llama_model_meta_val_str(mc->model, k, buf.data(), buf.size());
        }
        if (n >= 0) value.assign(buf.data(), static_cast<size_t>(n));
    }
    env->ReleaseStringUTFChars(key, k);
    if (value.empty()) return nullptr;
    return to_byte_array(env, value);
}

// Formats a conversation with the model's own chat template (llama_chat_apply_template supports the
// common families: ChatML/Qwen, Llama 3, Gemma, Phi-3, Mistral, DeepSeek, ...).
// Returns null when the model has no template or llama.cpp does not recognise it.
extern "C" JNIEXPORT jbyteArray JNICALL
JNI_FN(nativeApplyChatTemplate)(JNIEnv *env, jclass, jlong handle, jobjectArray roles, jobjectArray contents, jboolean add_assistant) {
    LlamaModelContext *mc = from_handle(handle);
    if (!mc || !mc->model) return nullptr;
    const char *tmpl = llama_model_chat_template(mc->model, nullptr);
    if (!tmpl) return nullptr;

    const jsize n = env->GetArrayLength(roles);
    std::vector<std::string> role_store(static_cast<size_t>(n));
    std::vector<std::string> content_store(static_cast<size_t>(n));
    std::vector<llama_chat_message> messages(static_cast<size_t>(n));
    size_t total = 0;
    for (jsize i = 0; i < n; ++i) {
        auto role = static_cast<jstring>(env->GetObjectArrayElement(roles, i));
        const char *r = env->GetStringUTFChars(role, nullptr);
        role_store[i] = r;
        env->ReleaseStringUTFChars(role, r);
        env->DeleteLocalRef(role);
        auto content = static_cast<jbyteArray>(env->GetObjectArrayElement(contents, i));
        content_store[i] = to_std_string(env, content);
        env->DeleteLocalRef(content);
        total += content_store[i].size();
    }
    for (jsize i = 0; i < n; ++i) {
        messages[i].role = role_store[i].c_str();
        messages[i].content = content_store[i].c_str();
    }

    std::vector<char> buf(total * 2 + 512);
    int32_t len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), add_assistant, buf.data(), static_cast<int32_t>(buf.size()));
    if (len < 0) return nullptr;
    if (len > static_cast<int32_t>(buf.size())) {
        buf.resize(static_cast<size_t>(len));
        len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), add_assistant, buf.data(), static_cast<int32_t>(buf.size()));
        if (len < 0) return nullptr;
    }
    return to_byte_array(env, std::string(buf.data(), static_cast<size_t>(len)));
}

// Clears the cancel flag. Called by Kotlin when a new request takes ownership of the model, never by
// nativeInference itself (a cancel issued just before inference starts must not be lost).
extern "C" JNIEXPORT void JNICALL
JNI_FN(nativeResetCancel)(JNIEnv *, jclass, jlong handle) {
    LlamaModelContext *mc = from_handle(handle);
    if (mc) mc->cancel.store(false);
}

extern "C" JNIEXPORT void JNICALL
JNI_FN(nativeCancel)(JNIEnv *, jclass, jlong handle) {
    LlamaModelContext *mc = from_handle(handle);
    if (mc) mc->cancel.store(true);
}

// Threads for single-token generation (memory-bound) and for prompt processing (compute-bound).
// Takes effect on the next decode; no reload needed.
extern "C" JNIEXPORT void JNICALL
JNI_FN(nativeSetThreads)(JNIEnv *, jclass, jlong handle, jint threads_decode, jint threads_batch) {
    LlamaModelContext *mc = from_handle(handle);
    if (!mc || !mc->ctx) return;
    mc->threads_decode = std::max(1, static_cast<int>(threads_decode));
    mc->threads_batch = std::max(1, static_cast<int>(threads_batch));
    llama_set_n_threads(mc->ctx, mc->threads_decode, mc->threads_batch);
}

// [prompt_tokens, reused_tokens, prefill_ms, first_token_ms, decode_ms, generated_tokens, threads_decode, threads_batch]
extern "C" JNIEXPORT jlongArray JNICALL
JNI_FN(nativeLastStats)(JNIEnv *env, jclass, jlong handle) {
    LlamaModelContext *mc = from_handle(handle);
    if (!mc) return nullptr;
    const InferenceStats &s = mc->stats;
    const jlong values[8] = {s.prompt_tokens, s.reused_tokens, s.prefill_ms, s.first_token_ms, s.decode_ms, s.generated_tokens,
                             mc->threads_decode, mc->threads_batch};
    jlongArray out = env->NewLongArray(8);
    if (out) env->SetLongArrayRegion(out, 0, 8, values);
    return out;
}

// Micro-benchmark used to pick thread counts: processes [prompt_tokens] synthetic tokens with
// [threads_batch] threads, then generates [gen_tokens] tokens one at a time with [threads_decode].
// Returns [prefill_ms, decode_ms], or null if it failed or was cancelled. Leaves the KV cache empty
// and restores the previous thread settings.
extern "C" JNIEXPORT jlongArray JNICALL
JNI_FN(nativeBenchmark)(JNIEnv *env, jclass, jlong handle, jint prompt_tokens, jint gen_tokens, jint threads_decode, jint threads_batch) {
    LlamaModelContext *mc = from_handle(handle);
    if (!mc || !mc->ctx || !mc->model || mc->cancel.load()) return nullptr;
    llama_context *ctx = mc->ctx;
    const llama_vocab *vocab = llama_model_get_vocab(mc->model);
    const int n_ctx = static_cast<int>(llama_n_ctx(ctx));
    const int n_prompt = std::max(8, std::min(static_cast<int>(prompt_tokens), n_ctx / 2));
    const int n_gen = std::max(0, std::min(static_cast<int>(gen_tokens), n_ctx / 4));

    // Ordinary text so the benchmark exercises the same kernels as real prompts.
    std::string text;
    std::vector<llama_token> tokens;
    while (static_cast<int>(tokens.size()) < n_prompt) {
        text += "The committee reviewed the annual report, compared the results with the previous year and "
                "recorded every recommendation, deadline and responsible person in the final summary. ";
        tokens = tokenize(vocab, text, true, false);
        if (tokens.empty()) return nullptr;
    }
    tokens.resize(static_cast<size_t>(n_prompt));

    reset_cache(mc);
    llama_set_n_threads(ctx, std::max(1, static_cast<int>(threads_decode)), std::max(1, static_cast<int>(threads_batch)));

    const int64_t t0 = now_ms();
    int rc = decode_range(ctx, tokens, 0, n_prompt, true);
    const int64_t t1 = now_ms();
    int64_t decode_ms = 0;
    if (rc == 0 && n_gen > 0) {
        llama_batch batch = llama_batch_init(1, 0, 1);
        llama_token token = tokens.back();
        for (int i = 0; i < n_gen && rc == 0 && !mc->cancel.load(); ++i) {
            batch.n_tokens = 1;
            batch.token[0] = token;
            batch.pos[0] = n_prompt + i;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = true;
            rc = llama_decode(ctx, batch);
        }
        llama_batch_free(batch);
        decode_ms = now_ms() - t1;
    }

    reset_cache(mc);
    llama_set_n_threads(ctx, mc->threads_decode, mc->threads_batch);
    if (rc != 0 || mc->cancel.load()) return nullptr;

    const jlong values[2] = {t1 - t0, decode_ms};
    jlongArray out = env->NewLongArray(2);
    if (out) env->SetLongArrayRegion(out, 0, 2, values);
    return out;
}

// Generates a completion for a fully formatted prompt. [listener] (may be null) receives the complete
// UTF-8 text produced so far every few tokens, for streaming to the UI.
extern "C" JNIEXPORT jbyteArray JNICALL
JNI_FN(nativeInference)(JNIEnv *env, jclass, jlong handle, jbyteArray prompt_bytes,
                        jint max_tokens, jfloat temperature, jfloat top_p, jfloat repeat_penalty,
                        jint seed, jobject listener) {
    LlamaModelContext *mc = from_handle(handle);
    if (!mc || !mc->ctx || !mc->model) {
        throw_java(env, "No model is loaded");
        return nullptr;
    }
    if (mc->cancel.load()) return to_byte_array(env, "");

    jmethodID on_text = nullptr;
    if (listener) {
        jclass cls = env->GetObjectClass(listener);
        on_text = env->GetMethodID(cls, "onText", "([B)V");
        env->DeleteLocalRef(cls);
        if (!on_text) {
            env->ExceptionClear();
            listener = nullptr;
        }
    }

    llama_context *ctx = mc->ctx;
    const llama_vocab *vocab = llama_model_get_vocab(mc->model);
    const std::string prompt = to_std_string(env, prompt_bytes);
    const int64_t call_start = now_ms();
    mc->stats = InferenceStats();

    // add_special: let the tokenizer add BOS when the model expects it (templates never include it).
    // parse_special: the prompt's template markers must become control tokens. Untrusted document and
    // user text is neutralised in Kotlin (PromptBuilder.sanitizeUntrusted) before it reaches this point.
    std::vector<llama_token> tokens = tokenize(vocab, prompt, true, true);
    const int n_prompt = static_cast<int>(tokens.size());
    const int n_ctx = static_cast<int>(llama_n_ctx(ctx));
    if (n_prompt == 0) {
        throw_java(env, "Prompt tokenization failed");
        return nullptr;
    }
    if (n_prompt >= n_ctx) {
        throw_java(env, "Prompt does not fit in the model context window");
        return nullptr;
    }

    // Prefix reuse: requests share the system prompt and template, so keep the KV entries of the longest
    // common prefix with what is already cached and process only the rest. At least the last prompt
    // token is always re-processed so fresh logits exist. Any doubt -> fall back to an empty cache.
    llama_memory_t mem = llama_get_memory(ctx);
    int n_keep = 0;
    {
        const int limit = std::min(static_cast<int>(mc->cached_tokens.size()), n_prompt - 1);
        while (n_keep < limit && mc->cached_tokens[n_keep] == tokens[n_keep]) n_keep++;
    }
    if (n_keep > 0) {
        const bool removed = llama_memory_seq_rm(mem, 0, n_keep, -1);
        // A cache that dropped early positions (e.g. a pruned sliding window) cannot serve this prefix.
        if (!removed || llama_memory_seq_pos_min(mem, 0) > 0 || llama_memory_seq_pos_max(mem, 0) != n_keep - 1) n_keep = 0;
    }
    if (n_keep == 0) {
        reset_cache(mc);
    } else {
        mc->cached_tokens.resize(static_cast<size_t>(n_keep));
    }
    mc->stats.prompt_tokens = n_prompt;
    mc->stats.reused_tokens = n_keep;

    const int64_t prefill_start = now_ms();
    const int prefill_rc = decode_range(ctx, tokens, n_keep, n_prompt, true);
    mc->stats.prefill_ms = now_ms() - prefill_start;
    if (prefill_rc != 0) {
        reset_cache(mc); // partially processed batches may remain after an abort
        if (mc->cancel.load()) return to_byte_array(env, "");
        throw_java(env, "llama_decode failed during prompt processing");
        return nullptr;
    }
    mc->cached_tokens = tokens;
    llama_batch batch = llama_batch_init(1, 0, 1);

    llama_sampler *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (repeat_penalty > 1.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_penalties(llama_vocab_n_tokens(vocab), 64, repeat_penalty, 0.0f, 0.0f));
    }
    if (temperature <= 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed < 0 ? LLAMA_DEFAULT_SEED : static_cast<uint32_t>(seed)));
    }

    std::string output;
    int n_generated = 0;
    int since_emit = 0;
    bool cache_valid = true;
    int64_t first_token_at = 0;
    while (n_generated < max_tokens && !mc->cancel.load()) {
        if (n_prompt + n_generated >= n_ctx) {
            LOGI("Context window full after %d generated tokens", n_generated);
            break;
        }
        const llama_token token = llama_sampler_sample(smpl, ctx, -1);
        llama_sampler_accept(smpl, token);
        if (first_token_at == 0) {
            first_token_at = now_ms();
            mc->stats.first_token_ms = first_token_at - call_start;
        }

        // End of turn, or any other control token (e.g. a hallucinated "<|im_start|>" for a new turn).
        if (llama_vocab_is_eog(vocab, token) || llama_vocab_is_control(vocab, token)) break;

        char piece[256];
        int n_chars = llama_token_to_piece(vocab, token, piece, sizeof(piece), 0, false);
        if (n_chars < 0) {
            std::vector<char> big(static_cast<size_t>(-n_chars));
            n_chars = llama_token_to_piece(vocab, token, big.data(), static_cast<int32_t>(big.size()), 0, false);
            if (n_chars > 0) output.append(big.data(), static_cast<size_t>(n_chars));
        } else if (n_chars > 0) {
            output.append(piece, static_cast<size_t>(n_chars));
        }

        // The first piece is emitted at once (honest time-to-first-token in the UI), then every 4 tokens.
        if (listener && (n_generated == 0 || ++since_emit >= 4)) {
            since_emit = 0;
            jbyteArray partial = to_byte_array(env, output.substr(0, complete_utf8_length(output)));
            env->CallVoidMethod(listener, on_text, partial);
            env->DeleteLocalRef(partial);
            if (env->ExceptionCheck()) {
                env->ExceptionClear();
                listener = nullptr;
            }
        }

        batch.n_tokens = 1;
        batch.token[0] = token;
        batch.pos[0] = n_prompt + n_generated;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = true;
        n_generated++;
        if (llama_decode(ctx, batch) != 0) {
            if (!mc->cancel.load()) LOGE("llama_decode failed during generation");
            cache_valid = false;
            break;
        }
        mc->cached_tokens.push_back(token);
    }

    llama_sampler_free(smpl);
    llama_batch_free(batch);
    if (!cache_valid || mc->cancel.load()) reset_cache(mc);
    mc->stats.generated_tokens = n_generated;
    mc->stats.decode_ms = first_token_at > 0 ? now_ms() - first_token_at : 0;
    if (first_token_at == 0) mc->stats.first_token_ms = now_ms() - call_start;
    output.resize(complete_utf8_length(output));
    LOGI("Generation finished: prompt=%d (reused %d) tokens, prefill=%lld ms, generated=%d tokens in %lld ms, threads=%d/%d, cancelled=%d",
         n_prompt, n_keep, static_cast<long long>(mc->stats.prefill_ms), n_generated, static_cast<long long>(mc->stats.decode_ms),
         mc->threads_decode, mc->threads_batch, mc->cancel.load() ? 1 : 0);
    return to_byte_array(env, output);
}

// -------------------------------------------------------------------------------------------------
// Sentence embeddings (BERT-family GGUF models such as all-MiniLM-L6-v2)
// -------------------------------------------------------------------------------------------------

extern "C" JNIEXPORT jlong JNICALL
JNI_FN(nativeEmbedInit)(JNIEnv *env, jclass, jbyteArray model_path, jint context_size, jint thread_count) {
    if (g_cpu_variant.empty()) {
        LOGE("No CPU backend available");
        return 0;
    }
    const std::string path = to_std_string(env, model_path);
    llama_model_params model_params = llama_model_default_params();
    llama_model *model = llama_model_load_from_file(path.c_str(), model_params);
    if (!model) {
        LOGE("Failed to load embedding model: %s", path.c_str());
        return 0;
    }
    const int train_ctx = llama_model_n_ctx_train(model);
    const int n_ctx = (train_ctx > 0) ? std::min<int>(context_size, train_ctx) : context_size;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.embeddings = true;
    ctx_params.pooling_type = LLAMA_POOLING_TYPE_MEAN;
    ctx_params.n_ctx = static_cast<uint32_t>(n_ctx);
    // Non-causal (encoder) models must see the whole input in one micro-batch.
    ctx_params.n_batch = static_cast<uint32_t>(n_ctx);
    ctx_params.n_ubatch = static_cast<uint32_t>(n_ctx);
    ctx_params.n_threads = thread_count;
    ctx_params.n_threads_batch = thread_count;

    llama_context *ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        LOGE("Failed to create embedding context");
        llama_model_free(model);
        return 0;
    }
    auto *ec = new EmbeddingContext();
    ec->model = model;
    ec->ctx = ctx;
    ec->n_ctx = n_ctx;
    LOGI("Embedding model loaded: dim=%d n_ctx=%d cpu=%s", llama_model_n_embd(model), n_ctx, g_cpu_variant.c_str());
    return reinterpret_cast<jlong>(ec);
}

extern "C" JNIEXPORT void JNICALL
JNI_FN(nativeEmbedFree)(JNIEnv *, jclass, jlong handle) {
    EmbeddingContext *ec = embedding_from_handle(handle);
    if (!ec) return;
    if (ec->ctx) llama_free(ec->ctx);
    if (ec->model) llama_model_free(ec->model);
    delete ec;
}

extern "C" JNIEXPORT jint JNICALL
JNI_FN(nativeEmbedDim)(JNIEnv *, jclass, jlong handle) {
    EmbeddingContext *ec = embedding_from_handle(handle);
    return (ec && ec->model) ? static_cast<jint>(llama_model_n_embd(ec->model)) : 0;
}

/** Mean-pooled (not normalised) sentence embedding of [text], or null on failure. */
extern "C" JNIEXPORT jfloatArray JNICALL
JNI_FN(nativeEmbed)(JNIEnv *env, jclass, jlong handle, jbyteArray text_bytes) {
    EmbeddingContext *ec = embedding_from_handle(handle);
    if (!ec || !ec->ctx) return nullptr;
    const llama_vocab *vocab = llama_model_get_vocab(ec->model);
    const std::string text = to_std_string(env, text_bytes);

    std::vector<llama_token> tokens = tokenize(vocab, text, true, false);
    if (tokens.empty()) return nullptr;
    if (static_cast<int>(tokens.size()) > ec->n_ctx) {
        // Keep [CLS] + leading content and re-append the separator, like the reference tokenizer's truncation.
        const llama_token sep = llama_vocab_sep(vocab);
        tokens.resize(static_cast<size_t>(ec->n_ctx));
        if (sep != LLAMA_TOKEN_NULL) tokens.back() = sep;
    }

    llama_memory_clear(llama_get_memory(ec->ctx), true);
    const int n = static_cast<int>(tokens.size());
    llama_batch batch = llama_batch_init(n, 0, 1);
    batch.n_tokens = n;
    for (int i = 0; i < n; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = true;
    }
    const bool encoder_only = llama_model_has_encoder(ec->model) && !llama_model_has_decoder(ec->model);
    const int rc = encoder_only ? llama_encode(ec->ctx, batch) : llama_decode(ec->ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        LOGE("Embedding forward pass failed (rc=%d)", rc);
        return nullptr;
    }
    const float *emb = llama_get_embeddings_seq(ec->ctx, 0);
    if (!emb) return nullptr;
    const int dim = llama_model_n_embd(ec->model);
    jfloatArray out = env->NewFloatArray(dim);
    if (out) env->SetFloatArrayRegion(out, 0, dim, emb);
    return out;
}
