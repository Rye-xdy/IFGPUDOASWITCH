cmake_minimum_required(VERSION 3.18)
project(CUDA_2d_interferometer CUDA CXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 查找 CUDA 包
find_package(CUDAToolkit REQUIRED)

# 添加可执行文件
add_executable(music src/uca_2d_interferometer.cu)
target_compile_options(music PRIVATE -lineinfo)

# 链接库 (链接 cuSOLVER 和 cuBLAS)
target_link_libraries(music PRIVATE CUDA::cusolver CUDA::cublas)