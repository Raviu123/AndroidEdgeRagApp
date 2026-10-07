cmake_minimum_required(VERSION 3.18)

project(llama_jni_project LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configure llama.cpp to compile only core library
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Build tests" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Build examples" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Build server" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "Build tools" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "Disable curl" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "Disable OpenMP" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "Never tune for the build machine" FORCE)

if (${ANDROID_ABI} STREQUAL "arm64-v8a")
    # Phones differ widely in CPU features (dotprod, fp16, i8mm, SVE...). Build every Android CPU variant
    # of the ggml CPU backend as its own library; llama_jni picks the best one the device supports at
    # runtime (or the baseline one in compatibility mode). A single baseline build would be 2-4x slower,
    # while a single dotprod build crashes on some chips.
    set(GGML_BACKEND_DL ON CACHE BOOL "Load ggml backends dynamically" FORCE)
    set(GGML_CPU_ALL_VARIANTS ON CACHE BOOL "Build all CPU variants" FORCE)
    add_compile_options(-O3)
elseif (${ANDROID_ABI} STREQUAL "x86_64")
    # Emulator only.
    set(GGML_AVX ON CACHE BOOL "Enable AVX" FORCE)
    set(GGML_AVX2 ON CACHE BOOL "Enable AVX2" FORCE)
    set(GGML_FMA ON CACHE BOOL "Enable FMA" FORCE)
    set(GGML_F16C ON CACHE BOOL "Enable F16C" FORCE)
    set(GGML_SSE42 ON CACHE BOOL "Enable SSE 4.2" FORCE)
    add_compile_options(-O3 -mavx -mavx2 -mfma -mf16c -msse4.2)
else()
    add_compile_options(-O3)
endif()

# Download and compile llama.cpp from official repo
include(FetchContent)
FetchContent_Declare(
    llamacpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    # Pinned: the JNI layer depends on this exact llama.cpp API (sampler / chat-template signatures).
    # Bump deliberately and re-test; never track `master`.
    GIT_TAG        e117148a41d8e9bedb72e4c6c3f003ab0fe7f857
)

FetchContent_MakeAvailable(llamacpp)

# Define our JNI library
add_library(llama_jni SHARED
    llama_jni.cpp
)

if (GGML_BACKEND_DL)
    target_compile_definitions(llama_jni PRIVATE LLAMA_JNI_BACKEND_DL=1)
endif()

# Find Android log library
find_library(log-lib log)

# Link JNI wrapper with llama.cpp libraries and Android log
target_link_libraries(llama_jni
    PRIVATE
    llama
    ggml
    ${log-lib}
)
