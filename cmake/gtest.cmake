include(ExternalProject)
set(GTEST_MD5_EXPECTED "ad6868782b5952b7476a7c1c72d5a714")
set(GTEST_LOCAL_ARCHIVE_FILE "release-1.8.1.zip")
set(GTEST_LOCAL_COPY "${CACHE_DIR}/${GTEST_LOCAL_ARCHIVE_FILE}")

if(EXISTS ${GTEST_LOCAL_COPY})
    file(MD5 ${GTEST_LOCAL_COPY} GTEST_HASH)
    if(NOT ${GTEST_HASH} STREQUAL ${GTEST_MD5_EXPECTED})
        message(
            FATAL_ERROR
            "gtest MD5 hash match error : ${GTEST_LOCAL_ARCHIVE_FILE} (expected: ${GTEST_MD5_EXPECTED} actual:${GTEST_HASH})"
            )
    else()
        message(STATUS "gtest found in local")
    endif()
else()
    message(STATUS "gtest not found in local. Downloading from github")
    file(DOWNLOAD
         "https://github.com/google/googletest/archive/${GTEST_LOCAL_ARCHIVE_FILE}"
         ${GTEST_LOCAL_COPY}
         EXPECTED_MD5 ${GTEST_MD5_EXPECTED})
endif()

ExternalProject_Add(gtest_ext
    URL ${GTEST_LOCAL_COPY}
    URL_MD5 ${GTEST_MD5_EXPECTED}
    BINARY_DIR "${THIRD_PARTY_DIR}/gtest-build"
    SOURCE_DIR "${THIRD_PARTY_DIR}/gtest-src"
    CMAKE_ARGS "${gtest_cmake_args}"
        "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
        "-DCMAKE_INSTALL_PREFIX=${THIRD_PARTY_DIR}/"
        "-DBUILD_GMOCK=OFF"
)
set(GTEST_INCLUDE_DIRS
    "${THIRD_PARTY_DIR}/include"
)

if(WIN32)
    set(GTEST_LIBRARY_FILENAME "gtest")
    set(LIBRARY_EXT "lib")
endif(WIN32)
if(UNIX)
    set(GTEST_LIBRARY_FILENAME "libgtest")
    set(LIBRARY_EXT "a")
endif(UNIX)
add_library(gtest STATIC IMPORTED)
set_target_properties(gtest PROPERTIES
    IMPORTED_LOCATION "${THIRD_PARTY_DIR}/lib/${GTEST_LIBRARY_FILENAME}.${LIBRARY_EXT}"
    IMPORTED_LOCATION_DEBUG "${THIRD_PARTY_DIR}/lib/${GTEST_LIBRARY_FILENAME}d.${LIBRARY_EXT}"
    INTERFACE_INCLUDE_DIRECTORIES "${THIRD_PARTY_DIR}/include"
)
add_dependencies(gtest gtest_ext)


enable_testing()

find_package(Threads)

function(cxx_test name sources)
    add_executable(${name} ${sources})
    target_link_libraries(${name} ${ARGN} gtest ${CMAKE_THREAD_LIBS_INIT})
    set_property(TARGET ${name} APPEND PROPERTY INCLUDE_DIRECTORIES
                 ${GTEST_INCLUDE_DIRS}
                 )
    # Working directory: where the dlls are installed.
    add_test(NAME ${name} 
             COMMAND ${name} "--gtest_break_on_failure")
endfunction()

function(cuda_cxx_test name sources)
    cuda_add_executable(${name} ${sources})
    target_link_libraries(${name} ${ARGN} gtest ${CMAKE_THREAD_LIBS_INIT})
    set_property(TARGET ${name} APPEND PROPERTY INCLUDE_DIRECTORIES
                 ${GTEST_INCLUDE_DIRS}
                 )
    # Working directory: where the dlls are installed.
    add_test(NAME ${name} 
             COMMAND ${name} "--gtest_break_on_failure")
endfunction()
