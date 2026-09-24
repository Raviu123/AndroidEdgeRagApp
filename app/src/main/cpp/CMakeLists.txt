cmake_minimum_required(VERSION 3.18)

project(llama_jni_project LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configure llama.cpp to compile only core library
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Build tests" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Build examples" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Build server" FORCE)

# Download and compile llama.cpp from official repo
include(FetchContent)
FetchContent_Declare(
    llamacpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG        master
    GIT_SHALLOW    ON
)

FetchContent_MakeAvailable(llamacpp)

# Define our JNI library
add_library(llama_jni SHARED
    llama_jni.cpp
)

# Find Android log library
find_library(log-lib log)

# Link JNI wrapper with llama.cpp libraries and Android log
target_link_libraries(llama_jni
    PRIVATE
    llama
    ${log-lib}
)
