set(A2E_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

add_subdirectory(source/audio2emotion-core)

# Tests and benchmarks are optional (skip in Docker builds)
if(BUILD_TESTING AND NOT DOCKER_BUILD)
    add_subdirectory(tests)
    add_subdirectory(source/benchmarks)
endif()

if(BUILD_SAMPLES AND NOT DOCKER_BUILD)
    add_subdirectory(source/samples)
endif()
