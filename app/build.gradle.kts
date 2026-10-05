plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
    id("org.jetbrains.kotlin.kapt")
}

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
        ndk {
            abiFilters.addAll(listOf("arm64-v8a", "x86_64"))
        }
        externalNativeBuild {
            cmake {
                cppFlags += listOf("-O3", "-fexceptions", "-frtti")
                cFlags += listOf("-O3")
                arguments += listOf(
                    "-DGGML_AVX=ON",
                    "-DGGML_AVX2=ON",
                    "-DGGML_FMA=ON",
                    "-DGGML_F16C=ON",
                    "-DGGML_SSE42=ON"
                )
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
