#include <jni.h>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <android/log.h>
#include "llama.h"

#define TAG "LlamaJNI"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

struct LlamaModelContext {
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    int thread_count = 4;
};

std::atomic<bool> g_cancel_generation(false);

// Helper to add tokens to batch
void batch_add(llama_batch & batch, llama_token id, llama_pos pos, const std::vector<llama_seq_id> & seq_ids, bool logits) {
    int i = batch.n_tokens;
    batch.token[i] = id;
    batch.pos[i] = pos;
    batch.n_seq_id[i] = seq_ids.size();
    for (size_t j = 0; j < seq_ids.size(); ++j) {
        batch.seq_id[i][j] = seq_ids[j];
    }
    batch.logits[i] = logits;
    batch.n_tokens++;
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_offlinedocai_ai_llm_LlamaRunner_nativeInit(JNIEnv* env, jobject thiz, jstring model_path, jint context_size, jint thread_count) {
    const char* path = env->GetStringUTFChars(model_path, nullptr);
    if (!path) {
        LOGE("Model path is null");
        return 0;
    }

    FILE* fp = fopen(path, "rb");
    if (!fp) {
        LOGE("Cannot open model file: %s", path);
        env->ReleaseStringUTFChars(model_path, path);
        return 0;
    }
    fseek(fp, 0, SEEK_END);
    long long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    char magic[5] = {0};
    size_t rb = fread(magic, 1, 4, fp);
    fclose(fp);

    if (rb < 4 || std::string(magic, 4) != "GGUF") {
        LOGE("File at %s is not a valid GGUF file! Header magic: '%s'", path, magic);
        env->ReleaseStringUTFChars(model_path, path);
        return 0;
    }

    std::string path_str(path);
    std::string filename = path_str;
    size_t last_slash = path_str.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        filename = path_str.substr(last_slash + 1);
    }

    __android_log_print(ANDROID_LOG_INFO, "MODEL_VERIFICATION", "==================================================");
    __android_log_print(ANDROID_LOG_INFO, "MODEL_VERIFICATION", "MODEL_PATH=%s", path);
    __android_log_print(ANDROID_LOG_INFO, "MODEL_VERIFICATION", "MODEL_FILENAME=%s", filename.c_str());
    __android_log_print(ANDROID_LOG_INFO, "MODEL_VERIFICATION", "MODEL_FILE_SIZE=%lld", file_size);
    __android_log_print(ANDROID_LOG_INFO, "MODEL_VERIFICATION", "GGUF_MAGIC_VERIFIED=true");

    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    llama_model* model = llama_model_load_from_file(path, model_params);
    env->ReleaseStringUTFChars(model_path, path);

    if (!model) {
        LOGE("Failed to load model from file: %s", path_str.c_str());
        return 0;
    }

    char desc[128] = {0};
    llama_model_desc(model, desc, sizeof(desc));
    uint64_t n_params = llama_model_n_params(model);

    __android_log_print(ANDROID_LOG_INFO, "MODEL_VERIFICATION", "MODEL_ARCHITECTURE=%s", desc);
    __android_log_print(ANDROID_LOG_INFO, "MODEL_VERIFICATION", "MODEL_PARAMETER_COUNT=%llu", (unsigned long long)n_params);
    __android_log_print(ANDROID_LOG_INFO, "MODEL_VERIFICATION", "ENGINE_STATE=GGUF_NATIVE");
    __android_log_print(ANDROID_LOG_INFO, "MODEL_VERIFICATION", "==================================================");

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = context_size;
    ctx_params.n_batch = 512;
    ctx_params.n_ubatch = 256;
    ctx_params.n_threads = thread_count;
    ctx_params.n_threads_batch = thread_count;

    llama_context* ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        LOGE("Failed to create llama context");
        llama_model_free(model);
        return 0;
    }

    LlamaModelContext* model_ctx = new LlamaModelContext();
    model_ctx->model = model;
    model_ctx->ctx = ctx;
    model_ctx->thread_count = thread_count;

    LOGD("Model and context initialized successfully");
    return reinterpret_cast<jlong>(model_ctx);
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_offlinedocai_ai_llm_LlamaRunner_nativeFree(JNIEnv* env, jobject thiz, jlong model_context) {
    LlamaModelContext* model_ctx = reinterpret_cast<LlamaModelContext*>(model_context);
    if (model_ctx) {
        if (model_ctx->ctx) {
            llama_free(model_ctx->ctx);
        }
        if (model_ctx->model) {
            llama_model_free(model_ctx->model);
        }
        delete model_ctx;
        LOGD("Model and context freed successfully");
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_offlinedocai_ai_llm_LlamaRunner_nativeInference(JNIEnv* env, jobject thiz, jlong model_context, jstring prompt, jint max_tokens, jfloat temperature) {
    auto t_entry = std::chrono::steady_clock::now();
    g_cancel_generation.store(false);
    __android_log_print(ANDROID_LOG_INFO, "CHAT_PERF", "JNI_ENTRY");
    __android_log_print(ANDROID_LOG_INFO, "CHAT_TRACE", "CHAT_TRACE: NATIVE_INFERENCE_ENTER");

    long long sampler_create_ms = 0;
    long long sampler_sample_ms = 0;
    long long sampler_accept_ms = 0;
    long long stop_check_ms = 0;
    long long string_conv_ms = 0;
    int decode_calls = 0;
    int sample_calls = 0;
    int jni_callbacks = 0;

    LlamaModelContext* model_ctx = reinterpret_cast<LlamaModelContext*>(model_context);
    
    // MODEL_ALREADY_LOADED_CHECK
    __android_log_print(ANDROID_LOG_INFO, "CHAT_PERF", "MODEL_ALREADY_LOADED_CHECK");
    if (!model_ctx || !model_ctx->ctx || !model_ctx->model) {
        LOGE("Invalid model context");
        return env->NewStringUTF("Error: Invalid model context");
    }

    const char* prompt_text = env->GetStringUTFChars(prompt, nullptr);
    if (!prompt_text) {
        LOGE("Prompt is null");
        return env->NewStringUTF("Error: Prompt is null");
    }

    std::string prompt_str(prompt_text);
    env->ReleaseStringUTFChars(prompt, prompt_text);

    llama_model* model = model_ctx->model;
    llama_context* ctx = model_ctx->ctx;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 1024;
    ctx_params.n_batch = 512;
    ctx_params.n_ubatch = 256;
    ctx_params.n_threads = model_ctx->thread_count;
    ctx_params.n_threads_batch = model_ctx->thread_count;

    if (!ctx) {
        ctx = llama_init_from_model(model, ctx_params);
        if (!ctx) {
            LOGE("Failed to create llama context");
            return env->NewStringUTF("Error: Failed to create context");
        }
        model_ctx->ctx = ctx;
    } else {
        llama_memory_clear(llama_get_memory(ctx), true);
    }

    const struct llama_vocab* vocab = llama_model_get_vocab(model);

    // PROMPT_TOKENIZATION_START
    auto t_tok_start = std::chrono::steady_clock::now();
    __android_log_print(ANDROID_LOG_INFO, "CHAT_PERF", "PROMPT_TOKENIZATION_START");

    std::vector<llama_token> tokens(prompt_str.length() + 4);
    int n_tokens = llama_tokenize(vocab, prompt_str.c_str(), prompt_str.length(), tokens.data(), tokens.size(), false, true);
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab, prompt_str.c_str(), prompt_str.length(), tokens.data(), tokens.size(), false, true);
    }
    tokens.resize(n_tokens);

    // PROMPT_TOKENIZATION_END
    auto t_tok_end = std::chrono::steady_clock::now();
    __android_log_print(ANDROID_LOG_INFO, "CHAT_PERF", "PROMPT_TOKENIZATION_END");
    long long prompt_tok_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_tok_end - t_tok_start).count();
    __android_log_print(ANDROID_LOG_INFO, "CHAT_PERF", "PROMPT_TOKENIZATION_MS = %lld", prompt_tok_ms);

    if (n_tokens <= 0) {
        LOGE("Tokenization failed");
        return env->NewStringUTF("Error: Tokenization failed");
    }

    // PROMPT_EVAL_START
    auto t_eval_start = std::chrono::steady_clock::now();
    __android_log_print(ANDROID_LOG_INFO, "CHAT_PERF", "PROMPT_EVAL_START");

    // Prefill decoding batch
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        batch_add(batch, tokens[i], i, { 0 }, i == n_tokens - 1);
    }

    decode_calls++;
    __android_log_print(ANDROID_LOG_INFO, "CHAT_PERF", "PREFILL_DECODE_BEGIN");
    __android_log_print(ANDROID_LOG_INFO, "CHAT_PERF", "PREFILL_TOKEN_COUNT=%d", n_tokens);
    __android_log_print(ANDROID_LOG_INFO, "CHAT_PERF", "PREFILL_N_BATCH=512");
    __android_log_print(ANDROID_LOG_INFO, "CHAT_PERF", "PREFILL_N_UBATCH=512");
    __android_log_print(ANDROID_LOG_INFO, "CHAT_PERF", "PREFILL_N_CTX=%d", llama_n_ctx(ctx));
    __android_log_print(ANDROID_LOG_INFO, "CHAT_PERF", "PREFILL_THREADS=%d", model_ctx->thread_count);
    __android_log_print(ANDROID_LOG_INFO, "CHAT_PERF", "PREFILL_THREADS_BATCH=%d", model_ctx->thread_count);

    std::atomic<bool> decode_finished(false);
    
    // Check for cancellation
    if (g_cancel_generation.load()) {
        __android_log_print(ANDROID_LOG_INFO, "CHAT_PERF", "GENERATION_CANCELLED_BEFORE_START");
        return env->NewStringUTF("");
    }

    std::thread watchdog([&decode_finished]() {
        for (int i = 0; i < 3000; ++i) { // 300 seconds (5 min) safe timeout for emulator
            if (decode_finished.load()) {
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (!decode_finished.load()) {
            __android_log_print(ANDROID_LOG_ERROR, "CHAT_PERF", "PREFILL_TIMEOUT");
            __android_log_print(ANDROID_LOG_ERROR, "CHAT_PERF", "PREFILL_STUCK=true");
            g_cancel_generation.store(true); // force cancel
        }
    });

    __android_log_print(ANDROID_LOG_INFO, "CHAT_TRACE", "CHAT_TRACE: BEFORE_PREFILL_DECODE");
    const int32_t decode_res = llama_decode(ctx, batch);
    decode_finished.store(true);
    __android_log_print(ANDROID_LOG_INFO, "CHAT_TRACE", "CHAT_TRACE: AFTER_PREFILL_DECODE result=%d", decode_res);

    if (watchdog.joinable()) {
        watchdog.join();
    }

    __android_log_print(ANDROID_LOG_INFO, "CHAT_PERF", "PREFILL_DECODE_RETURN=%d", decode_res);

    if (decode_res != 0) {
        LOGE("llama_decode failed in prefill with result %d", decode_res);
        llama_batch_free(batch);
        return env->NewStringUTF("Error: llama_decode failed during prefill");
    }

    // PROMPT_EVAL_END
    auto t_eval_end = std::chrono::steady_clock::now();
    __android_log_print(ANDROID_LOG_INFO, "CHAT_PERF", "PROMPT_EVAL_END");
    long long prompt_eval_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_eval_end - t_eval_start).count();
    __android_log_print(ANDROID_LOG_INFO, "CHAT_PERF", "PROMPT_EVAL_MS = %lld", prompt_eval_ms);

    // FIRST_DECODE_START
    auto t_first_decode_start = std::chrono::steady_clock::now();
    __android_log_print(ANDROID_LOG_INFO, "CHAT_PERF", "FIRST_DECODE_START");

    // Initialize sampler
    auto t_sampler_start = std::chrono::steady_clock::now();
    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    llama_sampler* smpl = llama_sampler_chain_init(sparams);
    if (temperature <= 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9f, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(1234));
    }
    auto t_sampler_end = std::chrono::steady_clock::now();
    sampler_create_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_sampler_end - t_sampler_start).count();

    std::string generated_text = "";
    auto t_s_start = std::chrono::steady_clock::now();
    llama_token curr_token = llama_sampler_sample(smpl, ctx, -1);
    auto t_s_end = std::chrono::steady_clock::now();
    sampler_sample_ms += std::chrono::duration_cast<std::chrono::milliseconds>(t_s_end - t_s_start).count();
    sample_calls++;

    auto t_a_start = std::chrono::steady_clock::now();
    llama_sampler_accept(smpl, curr_token);
    auto t_a_end = std::chrono::steady_clock::now();
    sampler_accept_ms += std::chrono::duration_cast<std::chrono::milliseconds>(t_a_end - t_a_start).count();

    // FIRST_TOKEN
    auto t_first_token = std::chrono::steady_clock::now();
    __android_log_print(ANDROID_LOG_INFO, "CHAT_PERF", "FIRST_TOKEN");
    long long ttft_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_first_token - t_entry).count();
    __android_log_print(ANDROID_LOG_INFO, "CHAT_PERF", "TTFT_MS = %lld", ttft_ms);

    llama_batch_free(batch);
    batch = llama_batch_init(1, 0, 1);

    // GENERATION_START
    auto t_gen_start = std::chrono::steady_clock::now();
    __android_log_print(ANDROID_LOG_INFO, "CHAT_PERF", "GENERATION_START");
    __android_log_print(ANDROID_LOG_INFO, "CHAT_TRACE", "CHAT_TRACE: BEFORE_GENERATION_LOOP");

    int tokens_generated = 0;
    while (tokens_generated < max_tokens) {
        if (g_cancel_generation.load()) {
            __android_log_print(ANDROID_LOG_INFO, "CHAT_PERF", "GENERATION_CANCELLED");
            break;
        }

        auto t_stop_start = std::chrono::steady_clock::now();
        bool is_eos = llama_vocab_is_eog(vocab, curr_token);
        auto t_stop_end = std::chrono::steady_clock::now();
        stop_check_ms += std::chrono::duration_cast<std::chrono::milliseconds>(t_stop_end - t_stop_start).count();

        if (is_eos) {
            break;
        }

        char token_piece[256];
        int n_chars = llama_token_to_piece(vocab, curr_token, token_piece, sizeof(token_piece), 0, true);
        if (n_chars > 0) {
            std::string piece(token_piece, n_chars);
            if (piece.find("<|im_end|>") != std::string::npos || 
                piece.find("<|endoftext|>") != std::string::npos ||
                piece.find("<|eot_id|>") != std::string::npos ||
                piece.find("<|end_of_text|>") != std::string::npos) {
                break;
            }
            generated_text.append(piece);
        }

        batch.n_tokens = 0;
        batch_add(batch, curr_token, n_tokens + tokens_generated, { 0 }, true);

        decode_calls++;
        if (llama_decode(ctx, batch) != 0) {
            LOGE("llama_decode failed during generation");
            break;
        }

        auto t_s_start2 = std::chrono::steady_clock::now();
        curr_token = llama_sampler_sample(smpl, ctx, -1);
        auto t_s_end2 = std::chrono::steady_clock::now();
        sampler_sample_ms += std::chrono::duration_cast<std::chrono::milliseconds>(t_s_end2 - t_s_start2).count();
        sample_calls++;

        auto t_a_start2 = std::chrono::steady_clock::now();
        llama_sampler_accept(smpl, curr_token);
        auto t_a_end2 = std::chrono::steady_clock::now();
        sampler_accept_ms += std::chrono::duration_cast<std::chrono::milliseconds>(t_a_end2 - t_a_start2).count();

        tokens_generated++;

        // EACH_TOKEN_GENERATION
        __android_log_print(ANDROID_LOG_INFO, "CHAT_PERF", "EACH_TOKEN_GENERATION");
        if (tokens_generated % 4 == 0) {
            __android_log_print(ANDROID_LOG_INFO, "CHAT_TRACE", "CHAT_TRACE: GENERATED_TOKENS=%d", tokens_generated);
        }
    }

    // GENERATION_END
    auto t_gen_end = std::chrono::steady_clock::now();
    __android_log_print(ANDROID_LOG_INFO, "CHAT_PERF", "GENERATION_END");
    __android_log_print(ANDROID_LOG_INFO, "CHAT_TRACE", "CHAT_TRACE: NATIVE_GENERATION_COMPLETE tokens=%d", tokens_generated);
    __android_log_print(ANDROID_LOG_INFO, "CHAT_TRACE", "CHAT_TRACE: NATIVE_RESULT_LENGTH=%zu", generated_text.length());
    long long gen_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_gen_end - t_gen_start).count();
    __android_log_print(ANDROID_LOG_INFO, "CHAT_PERF", "GENERATION_MS = %lld", gen_ms);

    llama_sampler_free(smpl);
    llama_batch_free(batch);

    double tok_per_sec = gen_ms > 0 ? (double)tokens_generated / ((double)gen_ms / 1000.0) : 0.0;

    auto t_conv_start = std::chrono::steady_clock::now();
    __android_log_print(ANDROID_LOG_INFO, "CHAT_TRACE", "CHAT_TRACE: JNI_RETURN_BEGIN");
    jstring result = env->NewStringUTF(generated_text.c_str());
    __android_log_print(ANDROID_LOG_INFO, "CHAT_TRACE", "CHAT_TRACE: JNI_RETURN_END");
    auto t_conv_end = std::chrono::steady_clock::now();
    string_conv_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_conv_end - t_conv_start).count();

    // Logs for configuration and sizes
    __android_log_print(ANDROID_LOG_INFO, "CHAT_PERF", "PROMPT_TOKENS = %d", n_tokens);
    __android_log_print(ANDROID_LOG_INFO, "CHAT_PERF", "OUTPUT_TOKENS = %d", tokens_generated);
    __android_log_print(ANDROID_LOG_INFO, "CHAT_PERF", "N_CTX = %d", llama_n_ctx(ctx));
    __android_log_print(ANDROID_LOG_INFO, "CHAT_PERF", "N_BATCH = 512");
    __android_log_print(ANDROID_LOG_INFO, "CHAT_PERF", "N_UBATCH = 512");

    // Output all detailed JNI counters
    __android_log_print(ANDROID_LOG_INFO, "CHAT_PERF", "DECODE_CALLS=%d", decode_calls);
    __android_log_print(ANDROID_LOG_INFO, "CHAT_PERF", "SAMPLE_CALLS=%d", sample_calls);
    __android_log_print(ANDROID_LOG_INFO, "CHAT_PERF", "SAMPLER_CREATE_MS=%lld", sampler_create_ms);
    __android_log_print(ANDROID_LOG_INFO, "CHAT_PERF", "SAMPLER_SAMPLE_MS=%lld", sampler_sample_ms);
    __android_log_print(ANDROID_LOG_INFO, "CHAT_PERF", "SAMPLER_ACCEPT_MS=%lld", sampler_accept_ms);
    __android_log_print(ANDROID_LOG_INFO, "CHAT_PERF", "JNI_CALLBACKS=%d", jni_callbacks);
    __android_log_print(ANDROID_LOG_INFO, "CHAT_PERF", "STRING_CONVERSION_MS=%lld", string_conv_ms);
    __android_log_print(ANDROID_LOG_INFO, "CHAT_PERF", "STOP_CHECK_MS=%lld", stop_check_ms);
    __android_log_print(ANDROID_LOG_INFO, "CHAT_PERF", "TOTAL_GENERATION_MS=%lld", gen_ms);
    __android_log_print(ANDROID_LOG_INFO, "CHAT_PERF", "TOKENS_PER_SEC=%.2f", tok_per_sec);

    auto t_return = std::chrono::steady_clock::now();
    long long total_native_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_return - t_entry).count();
    __android_log_print(ANDROID_LOG_INFO, "CHAT_PERF", "JNI_RETURN");
    __android_log_print(ANDROID_LOG_INFO, "CHAT_PERF", "TOTAL_NATIVE_INFERENCE_MS = %lld", total_native_ms);

    return result;
}

// Active Package JNI Wrappers for com.siddu.offlinedocai.ai.llm.LlamaRunner
extern "C" JNIEXPORT jlong JNICALL
Java_com_siddu_offlinedocai_ai_llm_LlamaRunner_nativeInit(JNIEnv* env, jobject thiz, jstring model_path, jint context_size, jint thread_count) {
    return Java_com_example_offlinedocai_ai_llm_LlamaRunner_nativeInit(env, thiz, model_path, context_size, thread_count);
}

extern "C" JNIEXPORT void JNICALL
Java_com_siddu_offlinedocai_ai_llm_LlamaRunner_nativeFree(JNIEnv* env, jobject thiz, jlong model_context) {
    Java_com_example_offlinedocai_ai_llm_LlamaRunner_nativeFree(env, thiz, model_context);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_siddu_offlinedocai_ai_llm_LlamaRunner_nativeInference(JNIEnv* env, jobject thiz, jlong model_context, jstring prompt, jint max_tokens, jfloat temperature) {
    return Java_com_example_offlinedocai_ai_llm_LlamaRunner_nativeInference(env, thiz, model_context, prompt, max_tokens, temperature);
}

extern "C" JNIEXPORT void JNICALL
Java_com_siddu_offlinedocai_ai_llm_LlamaRunner_nativeCancel(JNIEnv* env, jobject thiz) {
    g_cancel_generation.store(true);
    __android_log_print(ANDROID_LOG_INFO, "CHAT_PERF", "NATIVE_CANCEL_TRIGGERED");
}

