# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# CMake configuration for A2F C Wrapper Library

cmake_minimum_required(VERSION 3.24)

project(a2f-wrapper LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Source files
set(A2F_WRAPPER_SOURCES
    src/a2f_wrapper.cpp
)

# Create shared library
add_library(a2f_wrapper SHARED ${A2F_WRAPPER_SOURCES})

# Include directories
target_include_directories(a2f_wrapper
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link against audio2x library (contains all SDK functionality including audio2emotion)
target_link_libraries(a2f_wrapper
    PRIVATE
        audio2x
        CUDA::cudart
)

# Export macro for Windows DLL
target_compile_definitions(a2f_wrapper
    PRIVATE
        A2F_WRAPPER_EXPORTS
)

# Compiler flags
if(UNIX)
    target_compile_options(a2f_wrapper PRIVATE
        -fvisibility=hidden
    )
    target_link_options(a2f_wrapper PRIVATE
        "LINKER:-Bsymbolic"
    )
endif()

# Set library properties
set_target_properties(a2f_wrapper PROPERTIES
    VERSION 1.0.0
    SOVERSION 1
    OUTPUT_NAME "a2f_wrapper"
)

# Install rules
install(TARGETS a2f_wrapper
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES include/a2f_wrapper.h
    DESTINATION include
)

# Copy to output directory alongside other SDK outputs
add_custom_command(TARGET a2f_wrapper POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:a2f_wrapper>
        ${CMAKE_BINARY_DIR}/audio2x-sdk/lib/$<TARGET_FILE_NAME:a2f_wrapper>
    COMMENT "Copying a2f_wrapper to audio2x-sdk output directory"
)
