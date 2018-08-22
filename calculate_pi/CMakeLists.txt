cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(calculate_pi)

find_package(OpenMP REQUIRED)
add_executable(calculate_pi main.cpp)
target_link_libraries(calculate_pi PRIVATE OpenMP::OpenMP_CXX)
