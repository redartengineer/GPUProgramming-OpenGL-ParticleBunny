cmake_minimum_required(VERSION 3.20)

project(
    ParticleBunny
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# RTX 4070 = compute capability 8.9
set(CMAKE_CUDA_ARCHITECTURES 89)

include(FetchContent)

# ------------------------------------------------------------
# GLFW
# ------------------------------------------------------------

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
)

# ------------------------------------------------------------
# GLM
# ------------------------------------------------------------

FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 1.0.3
)

# ------------------------------------------------------------
# GLAD
# ------------------------------------------------------------

FetchContent_Declare(
    glad
    GIT_REPOSITORY https://github.com/Dav1dde/glad.git
    GIT_TAG glad2
)

FetchContent_MakeAvailable(
    glfw
    glm
)

FetchContent_GetProperties(glad)

if(NOT glad_POPULATED)

    FetchContent_Populate(glad)

    add_subdirectory(
        ${glad_SOURCE_DIR}/cmake
        ${glad_BINARY_DIR}
    )

endif()

glad_add_library(
    glad_gl_core_43
    REPRODUCIBLE
    API gl:core=4.3
)

# ------------------------------------------------------------
# Executable
# ------------------------------------------------------------

add_executable(
    ParticleBunny

    src/main.cc
    src/ply_loader.cc
    src/particle_sim.cu
)

# ------------------------------------------------------------
# Include directories
# ------------------------------------------------------------

target_include_directories(
    ParticleBunny
    PRIVATE

    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# ------------------------------------------------------------
# Libraries
# ------------------------------------------------------------

target_link_libraries(
    ParticleBunny
    PRIVATE

    glfw
    glad_gl_core_43
    glm::glm
)

# ------------------------------------------------------------
# CUDA settings
# ------------------------------------------------------------

set_target_properties(
    ParticleBunny
    PROPERTIES

    CUDA_SEPARABLE_COMPILATION ON
)

# ------------------------------------------------------------
# Copy runtime assets
# ------------------------------------------------------------

add_custom_command(
    TARGET ParticleBunny
    POST_BUILD

    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders
            $<TARGET_FILE_DIR:ParticleBunny>/shaders

    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/Assets
            $<TARGET_FILE_DIR:ParticleBunny>/Assets
)