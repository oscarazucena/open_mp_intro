cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(hello_world)

find_package(OpenMP REQUIRED)
add_executable(hello_world main.cpp)
target_link_libraries(hello_world PRIVATE OpenMP::OpenMP_CXX)
