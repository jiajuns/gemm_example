set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(gemm_example LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

find_package(CUDA)
if (CUDA_FOUND)
    message("CUDA found")
else()
    message("CUDA not found, doing something alternatively")
endif()

add_executable(gemm_example src/main.cu)
target_include_directories(gemm_example PRIVATE
                           ${CUDA_INCLUDE_DIRS}
                           ${CUDA_CUBLAS_DIRS}
                           ${PROJECT_SOURCE_DIR})
target_link_libraries(gemm_example PRIVATE -lcudart -lcuda -lcublas)
