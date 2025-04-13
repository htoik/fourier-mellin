cmake_minimum_required(VERSION 3.27)

include(FetchContent)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.1
)

FetchContent_MakeAvailable(pybind11)
