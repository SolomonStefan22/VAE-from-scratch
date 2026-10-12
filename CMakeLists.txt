cmake_minimum_required(VERSION 3.16)
project(vae CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_SOURCE_DIR}/src)

add_executable(train_ae  examples/train_ae.cpp)
add_executable(train_vae examples/train_vae.cpp)
add_executable(train_classifier examples/train_classifier.cpp)
add_executable(evaluate examples/evaluate.cpp)
add_executable(show_mnist examples/show_mnist.cpp)