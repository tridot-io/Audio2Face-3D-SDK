set(A2X_COMMON_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

add_subdirectory(source/audio2x)

# Tests are optional (skip in Docker builds)
if(BUILD_TESTING AND NOT DOCKER_BUILD)
    add_subdirectory(tests)
endif()
