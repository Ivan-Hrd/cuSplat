cmake_minimum_required(VERSION 3.21.2)

project(cuSplat LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

enable_language(CUDA)
find_package(benchmark REQUIRED)

set(RAFT_NVTX ON)
include(fetch_rapids.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-find)

rapids_cuda_init_architectures(cuSplat)

rapids_cpm_init()
include(cmake/thirdparty/get_cccl.cmake)
include(cmake/thirdparty/get_rmm.cmake)
include(cmake/thirdparty/get_raft.cmake)

# Add fconcept for lambdas
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconcepts")
# Generate cubin for this specific GPU
set(CMAKE_CUDA_ARCHITECTURES "native")
# Set fpic for Google Benchmark
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


# SNAPSHOT
add_executable(snapshot
        include/gaussian.hpp
        include/camera.hpp
        src/rasterizer.cu
        include/utils.hpp
        src/projection.cu
        include/tiling.hpp
        src/tiling.cu)
target_include_directories(snapshot PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(snapshot PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-U__cpp_lib_assume_aligned>
)
target_link_libraries(snapshot
        PRIVATE
        rmm::rmm
        raft::raft
        benchmark::benchmark
)
# --

# bench
add_executable(bench
        include/gaussian.hpp
        include/camera.hpp
        src/rasterizer.cu
        include/utils.hpp
        src/projection.cu
        include/tiling.hpp
        src/tiling.cu)
target_include_directories(bench PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(bench PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-U__cpp_lib_assume_aligned>
)
target_link_libraries(bench
        PRIVATE
        rmm::rmm
        raft::raft
        benchmark::benchmark
)
# --

add_subdirectory(src)
