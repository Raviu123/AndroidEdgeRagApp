import java.net.URI
import java.security.MessageDigest

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
    id("org.jetbrains.kotlin.kapt")
}

val bundledEmbeddingSha256 = "afdb6f1a0e45b715d0bb9b11772f032c399babd23bfc31fed1c170afc848bdb1"
val bundledRerankerSha256 = "e9d8ebf845c413e981c175bfe49a3bfa9b3dcce2a3ba54875ee5df5a58639fbe"

android {
    namespace = "com.siddu.offlinedocai"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.siddu.offlinedocai"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        // Re-extracts the bundled embedding model whenever the asset changes (marker = its SHA-256).
        buildConfigField("String", "EMBEDDING_ASSET_SHA256", "\"$bundledEmbeddingSha256\"")
        buildConfigField("String", "RERANKER_ASSET_SHA256", "\"$bundledRerankerSha256\"")
        ndk {
            abiFilters.addAll(listOf("arm64-v8a", "x86_64"))
        }
        externalNativeBuild {
            cmake {
                cppFlags += listOf("-O3", "-fexceptions", "-frtti")
                cFlags += listOf("-O3")
            }
        }
    }

    ndkVersion = "26.1.10909125"

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }
    // NOTE: composeOptions.kotlinCompilerExtensionVersion is NOT needed
    // when using the org.jetbrains.kotlin.plugin.compose plugin (Kotlin 2.0+)
    packaging {
        resources {
            excludes += setOf("META-INF/LICENSE.md", "META-INF/LICENSE-notice.md")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }
    kotlinOptions {
        jvmTarget = "1.8"
    }
    testOptions {
        unitTests.isReturnDefaultValues = true
    }
}

kapt {
    arguments {
        // Exported Room schemas (app/schemas) make migrations reviewable and testable.
        arg("room.schemaLocation", "$projectDir/schemas")
    }
}

// The bundled ONNX models are not committed to git (binary, ~23 MB each). Fetch them on a fresh clone
// and verify their SHA-256 so each bundled asset is always exactly the model the code expects.
fun sha256Of(f: File): String {
    val digest = MessageDigest.getInstance("SHA-256")
    f.inputStream().use { input ->
        val buffer = ByteArray(1 shl 16)
        while (true) {
            val read = input.read(buffer)
            if (read < 0) break
            digest.update(buffer, 0, read)
        }
    }
    return digest.digest().joinToString("") { b -> "%02x".format(b) }
}

fun registerBundledModel(taskName: String, label: String, target: File, url: String, sha256: String) =
    tasks.register(taskName) {
        description = "Downloads and verifies the bundled $label ONNX model."
        outputs.file(target)
        doLast {
            if (target.exists() && sha256Of(target) == sha256) return@doLast
            logger.lifecycle("Downloading $label from $url")
            target.parentFile.mkdirs()
            val tmp = File(target.parentFile, target.name + ".part")
            URI(url).toURL().openStream().use { input ->
                tmp.outputStream().use { output -> input.copyTo(output) }
            }
            val actual = sha256Of(tmp)
            if (actual != sha256) {
                tmp.delete()
                throw GradleException("$label checksum mismatch: expected $sha256, got $actual")
            }
            if (target.exists()) target.delete()
            if (!tmp.renameTo(target)) throw GradleException("Could not move $tmp into place")
        }
    }

val ensureEmbeddingModel = registerBundledModel(
    "ensureEmbeddingModel", "all-MiniLM-L6-v2 embedding model",
    file("src/main/assets/models/embedding/all-MiniLM-L6-v2.onnx"),
    "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model_quantized.onnx",
    bundledEmbeddingSha256
)

// Cross-encoder that re-scores retrieved passages against the question (same uncased BERT vocabulary as the embedder).
val ensureRerankerModel = registerBundledModel(
    "ensureRerankerModel", "ms-marco-MiniLM-L-6-v2 reranker",
    file("src/main/assets/models/reranker/ms-marco-MiniLM-L-6-v2.onnx"),
    "https://huggingface.co/Xenova/ms-marco-MiniLM-L-6-v2/resolve/main/onnx/model_quantized.onnx",
    bundledRerankerSha256
)

tasks.named("preBuild") { dependsOn(ensureEmbeddingModel, ensureRerankerModel) }

dependencies {
    // Core AndroidX
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.core:core-splashscreen:1.0.1")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.2")
    implementation("androidx.activity:activity-compose:1.9.0")

    // Compose UI
    implementation("androidx.compose.ui:ui:1.6.8")
    implementation("androidx.compose.ui:ui-tooling-preview:1.6.8")
    implementation("androidx.compose.material3:material3:1.2.1")
    implementation("androidx.compose.material:material-icons-extended:1.6.8")
    implementation("androidx.navigation:navigation-compose:2.7.7")

    // Room (local DB)
    implementation("androidx.room:room-runtime:2.6.1")
    kapt("androidx.room:room-compiler:2.6.1")
    implementation("androidx.room:room-ktx:2.6.1")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.0")

    // PDF & DOCX parsing (Android compatible)
    implementation("com.tom-roush:pdfbox-android:2.0.27.0")
    implementation("org.apache.poi:poi-ooxml:5.2.3")

    // Background model downloads (foreground worker, resumable)
    implementation("androidx.work:work-runtime-ktx:2.9.1")

    // ONNX Runtime Android for real local embeddings
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.17.1")

    // Debug tooling
    debugImplementation("androidx.compose.ui:ui-tooling:1.6.8")
    debugImplementation("androidx.compose.ui:ui-test-manifest:1.6.8")

    // Testing
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20240303")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test:runner:1.5.2")
    androidTestImplementation("androidx.test:rules:1.5.0")
}
