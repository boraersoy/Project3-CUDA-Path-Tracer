cmake_minimum_required(VERSION 3.28)

project(cis565_path_tracer LANGUAGES CXX CUDA)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Load GPU architectures list
include(${CMAKE_MODULE_PATH}/CUDAComputesList.cmake)

# Mark .cu files automatically as CUDA
file(GLOB_RECURSE ALL_CU_FILES RELATIVE ${CMAKE_SOURCE_DIR} "*.cu")
if(ALL_CU_FILES)
  message(STATUS "CUDA files detected: ${ALL_CU_FILES}")
  set_source_files_properties(${ALL_CU_FILES} PROPERTIES LANGUAGE CUDA)
endif()

########################################
# CUDA Toolkit
########################################
find_package(CUDAToolkit REQUIRED)

########################################
# OpenGL + GLFW + GLEW
########################################
find_package(OpenGL REQUIRED)

if(UNIX)
    find_package(glfw3 REQUIRED)
    find_package(GLEW REQUIRED)
    set(LIBRARIES glfw ${GLEW_LIBRARIES} ${OPENGL_gl_LIBRARY})

else()
    set(EXTERNAL "external")

    set(GLFW_ROOT_DIR ${EXTERNAL})
    set(GLFW_USE_STATIC_LIBS ON)
    find_package(GLFW REQUIRED)

    set(GLEW_ROOT_DIR ${EXTERNAL})
    set(GLEW_USE_STATIC_LIBS ON)
    find_package(GLEW REQUIRED)

    include_directories(${GLEW_INCLUDE_DIR} ${GLFW_INCLUDE_DIR})
    add_definitions(${GLEW_DEFINITIONS})

    set(LIBRARIES ${GLEW_LIBRARY} ${GLFW_LIBRARY} ${OPENGL_LIBRARY})
endif()

# GLM
set(GLM_ROOT_DIR "external")
find_package(GLM REQUIRED)
include_directories(${GLM_INCLUDE_DIRS})

########################################
# Sources
########################################
set(headers
    src/main.h
    src/image.h
    src/interactions.h
    src/intersections.h
    src/glslUtility.hpp
    src/pathtrace.h
    src/scene.h
    src/sceneStructs.h
    src/preview.h
    src/utilities.h
    src/ImGui/imconfig.h
    src/ImGui/imgui.h
    src/ImGui/imgui_impl_glfw.h
    src/ImGui/imgui_impl_opengl3.h
    src/ImGui/imgui_internal.h
    src/ImGui/imstb_rectpack.h
    src/ImGui/imstb_textedit.h
    src/ImGui/imstb_truetype.h
    src/tiny_obj_loader.h
    src/objloader.h
    src/octree.h
    src/bvh.h
    src/denoise.h
)

set(sources
    src/main.cpp
    src/stb.cpp
    src/image.cpp
    src/glslUtility.cpp
    src/pathtrace.cu
    src/scene.cpp
    src/preview.cpp
    src/utilities.cpp
    src/ImGui/imgui.cpp
    src/ImGui/imgui_demo.cpp
    src/ImGui/imgui_draw.cpp
    src/ImGui/imgui_impl_glfw.cpp
    src/ImGui/imgui_impl_opengl3.cpp
    src/ImGui/imgui_tables.cpp
    src/ImGui/imgui_widgets.cpp
    src/objloader.cpp
    src/octree.cpp
    src/bvh.cpp
    src/denoise.cpp
)

list(SORT headers)
list(SORT sources)

source_group(Headers FILES ${headers})
source_group(Sources FILES ${sources})

########################################
# Build
########################################
add_executable(${CMAKE_PROJECT_NAME} ${sources} ${headers})


set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_link_libraries(${CMAKE_PROJECT_NAME}
    ${LIBRARIES}
    CUDA::cudart
)
