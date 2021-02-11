# minimum requirement
cmake_minimum_required(VERSION 3.5)

# solution
project(sampleSolution)

# 3rd party variables
set(THIRD_PARTY_DIR "${CMAKE_BINARY_DIR}/third-party")
file(MAKE_DIRECTORY ${THIRD_PARTY_DIR}/include)
##  cache dir
set(CACHE_DIR "${CMAKE_SOURCE_DIR}/.cache")
file(MAKE_DIRECTORY ${CACHE_DIR})
# include gtest settings
include("cmake/gtest.cmake")

SET(TARGET_NAME sampleTarget)
SET(TEST_NAME sampleTest)

find_package(CUDA REQUIRED)

# build TARGET_NAME from source code
cuda_cxx_test(${TEST_NAME} main.cpp)
target_link_libraries(${TEST_NAME} ${CUDA_LIBRARIES})
target_include_directories(${TEST_NAME}   PUBLIC ${THIRD_PARTY_DIR}/include)
