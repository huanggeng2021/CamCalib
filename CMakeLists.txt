cmake_minimum_required(VERSION 3.16)

project(CamCalib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc highgui)

file(GLOB_RECURSE CAMCALIB_SOURCES CONFIGURE_DEPENDS
    ${PROJECT_SOURCE_DIR}/src/*.cpp
)

add_library(camcalib ${CAMCALIB_SOURCES})

target_include_directories(camcalib
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(camcalib
    PUBLIC
        ${OpenCV_LIBS}
)

add_executable(camcalib_app
    app/main.cpp
)

target_link_libraries(camcalib_app
    PRIVATE
        camcalib
        ${OpenCV_LIBS}
)
