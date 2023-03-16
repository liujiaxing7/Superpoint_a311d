cmake_minimum_required(VERSION 3.10)
project(Superpoint_a311d LANGUAGES CXX C)

set(CMAKE_C_STANDARD 99)

find_package(OpenCV REQUIRED)

set(NPU_INCLUDE_DIR ${npu_DIR}/include)
set(NPU_LIBRARY_DIR ${npu_DIR}/drivers_64)
set(NPU_LIBRARIES -L${NPU_LIBRARY_DIR}-lOpenVX -lOpenVXU -lArchModelSw -lCLC -lGAL -lNNArchPerf -lVSC -lovxlib)
set(THIRD_PARTY_LIBRARIES -L"/lib/aarch64-linux-gnu"  -lanl -lc -ldl -lgcc_s -lm -lrt -lstdc++)
set(SDK_LIBRARIES -L"/usr/lib/aarch64-linux-gnu" -ljpeg) # jpeg9.0

function (find_source_file SRC_FILES arg)
    set(file_list "")
    message(STATUS "[*] searching file: ${arg};${ARGN}")

    list(LENGTH ARGV  argv_len)
    set(i 1)
    while( i LESS ${argv_len})
        list(GET ARGV ${i} argv_value)
        foreach(file_path ${SRC_DIR})
            file(GLOB file ${file_path}/${argv_value})
            set(file_list ${file_list} ${file})
        endforeach()

        math(EXPR i "${i} + 1")
    endwhile()

    set(${SRC_FILES} ${file_list} PARENT_SCOPE)
endfunction ()

set(ROOT_PATH ${CMAKE_CURRENT_LIST_DIR})
set(SRC_DIR ${ROOT_PATH} ${ROOT_PATH})

set(INCLUDE_DIR ${SRC_DIR})
include_directories(${INCLUDE_DIR} ${NPU_INCLUDE_DIR} "/usr/include")
include_directories(/usr/include)

include_directories(/usr/share/npu/sdk/include)
execute_process(COMMAND cp ${CMAKE_CURRENT_LIST_DIR}/network_binary.nb ${CMAKE_BINARY_DIR})
set(SRC_FILES "./")
find_source_file(SRC_FILES  "*.h" "*.cpp"  "*.cc" "*.hpp")

add_executable(${PROJECT_NAME} ${SRC_FILES})
#add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${NPU_LIBRARIES} ${THIRD_PARTY_LIBRARIES} ${SDK_LIBRARIES})
