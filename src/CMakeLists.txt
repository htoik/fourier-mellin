cmake_minimum_required(VERSION 3.27)

find_package(OpenCV REQUIRED)

set(SOURCES
    fourier_mellin.cpp
    utilities.cpp
    transform.cpp
    log_polar_map.cpp
    image_filter.cpp
)

add_library(fourier-mellin-library STATIC ${SOURCES})
target_include_directories(fourier-mellin-library PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(fourier-mellin-library ${OpenCV_LIBS})

add_definitions(-DMODULE_NAME=${MODULE_NAME})
pybind11_add_module(${MODULE_NAME} fourier_mellin_module.cpp ${SOURCES})
target_link_libraries(${MODULE_NAME} PRIVATE ${OpenCV_LIBS})
install(TARGETS ${MODULE_NAME} DESTINATION .)
