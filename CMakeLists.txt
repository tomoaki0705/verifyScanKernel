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
cuda_add_executable(${TEST_NAME} main.cpp kernel.cu)
target_link_libraries(${TEST_NAME} gtest ${CMAKE_THREAD_LIBS_INIT}  ${CUDA_LIBRARIES})
set_property(TARGET ${TEST_NAME} APPEND PROPERTY INCLUDE_DIRECTORIES ${GTEST_INCLUDE_DIRS} ${THIRD_PARTY_DIR}/include)
add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME} "--gtest_break_on_failure")
