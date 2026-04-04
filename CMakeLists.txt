# © 2026 Marwane KADOUCI and Ivan HUARD. All rights reserved. See LICENSE at project root for terms.

cmake_minimum_required(VERSION 3.21.2)

project(rt-core-gaussian-splatting LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(main
        include/gaussian.hpp
        include/camera.hpp
        src/frustrumCulling.cpp
        src/rasterizer.cpp
        include/utils.hpp
        src/projection.cpp
        include/tiling.hpp
        src/tiling.cpp)
target_include_directories(main PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(src)
