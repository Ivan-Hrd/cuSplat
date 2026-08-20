cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

include(fetch_rapids.cmake)

include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-find)

# Generate cubin for this specific GPU. Must precede project().
set(CMAKE_CUDA_ARCHITECTURES "NATIVE")
rapids_cuda_init_architectures(cuSplat)

project(cuSplat LANGUAGES CXX CUDA)

rapids_cmake_build_type(Release)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(RAFT_NVTX ON)

rapids_cpm_init()
include(cmake/thirdparty/get_cccl.cmake)
include(cmake/thirdparty/get_rmm.cmake)
include(cmake/thirdparty/get_raft.cmake)

find_package(benchmark REQUIRED)

# Sources shared by both executables, compiled once
add_library(cusplat_core OBJECT
        src/rasterizer.cu
        src/projection.cu
        src/tiling.cu)
target_include_directories(cusplat_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(cusplat_core PUBLIC rmm::rmm raft::raft)

foreach(_target snapshot bench)
    add_executable(${_target})
    target_link_libraries(${_target} PRIVATE cusplat_core benchmark::benchmark)
endforeach()

add_subdirectory(src)