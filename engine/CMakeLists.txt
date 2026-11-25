cmake_minimum_required(VERSION 3.20)
project(indextts-server VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# MSVC: treat source files as UTF-8
if(MSVC)
    add_compile_options(/utf-8)
endif()

# ===========================================================================
# Options
# ===========================================================================
option(USE_CUDA "Enable CUDA/TensorRT support via ONNX Runtime" OFF)

# ===========================================================================
# Dependencies
# ===========================================================================

# --- ONNX Runtime (pre-built SDK) ---
# Default to local third_party copy; override with -DONNXRUNTIME_DIR=...
if(NOT DEFINED ONNXRUNTIME_DIR)
    # Auto-detect: third_party/onnxruntime-win-x64-*
    file(GLOB _ORT_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime-*")
    list(LENGTH _ORT_DIRS _ORT_COUNT)
    if(_ORT_COUNT GREATER 0)
        list(GET _ORT_DIRS 0 ONNXRUNTIME_DIR)
        message(STATUS "Auto-detected ONNXRuntime: ${ONNXRUNTIME_DIR}")
    else()
        message(FATAL_ERROR "ONNXRUNTIME_DIR not set and no onnxruntime-* found in third_party/.\n"
                "Download from https://github.com/microsoft/onnxruntime/releases")
    endif()
endif()

find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    PATHS "${ONNXRUNTIME_DIR}/lib"
    NO_DEFAULT_PATH
)
if(NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "Could not find onnxruntime library in ${ONNXRUNTIME_DIR}/lib")
endif()
message(STATUS "ONNXRuntime library: ${ONNXRUNTIME_LIB}")

# --- SentencePiece (built as subdirectory) ---
set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "" FORCE)
set(SPM_USE_BUILTIN_PROTOBUF ON CACHE BOOL "" FORCE)
add_subdirectory(third_party/sentencepiece_src EXCLUDE_FROM_ALL)

# --- nlohmann/json + cpp-httplib (FetchContent) ---
include(FetchContent)

FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE
)

FetchContent_Declare(
    httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG v0.15.3
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(json httplib)

# ===========================================================================
# Library: indextts-core
# ===========================================================================

add_library(indextts-core STATIC
    src/mel_extractor.cpp
    src/text_processor.cpp
    src/indextts_engine.cpp
    src/api_server.cpp
)

target_include_directories(indextts-core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${ONNXRUNTIME_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/sentencepiece_src/src
)

target_link_libraries(indextts-core PUBLIC
    ${ONNXRUNTIME_LIB}
    sentencepiece-static
    nlohmann_json::nlohmann_json
    httplib::httplib
)

if(USE_CUDA)
    target_compile_definitions(indextts-core PUBLIC INDEXTTS_USE_CUDA)
endif()

# ===========================================================================
# Executable: indextts
# ===========================================================================

add_executable(indextts src/main.cpp)
target_link_libraries(indextts PRIVATE indextts-core)

# ===========================================================================
# Install
# ===========================================================================

install(TARGETS indextts DESTINATION bin)

# Copy ONNXRuntime DLL to output directory (Windows)
if(WIN32)
    file(GLOB ORT_DLLS "${ONNXRUNTIME_DIR}/lib/*.dll")
    foreach(dll ${ORT_DLLS})
        add_custom_command(TARGET indextts POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${dll}" "$<TARGET_FILE_DIR:indextts>"
        )
    endforeach()
endif()
