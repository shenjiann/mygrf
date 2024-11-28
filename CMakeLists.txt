cmake_minimum_required(VERSION 3.27)
project(mygrf)

## ======================================================================================##
## Compiler flags
## ======================================================================================##
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2")
if(MSVC)
  # Decrease warning level for verbose MSVC
  add_compile_options(/W3)
endif()

## ======================================================================================##
## Linker flags
## ======================================================================================##
if(NOT MSVC)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")
endif()

## ======================================================================================##
## Subdirectories and source files
## ======================================================================================##
file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp 
                          ${CMAKE_CURRENT_SOURCE_DIR}/test/*.cpp)
file(GLOB_RECURSE HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/third_party/*.h")
foreach(HEADER ${HEADER_FILES})
    get_filename_component(DIR ${HEADER} DIRECTORY)
    include_directories(${DIR})
endforeach()
include_directories(${CMAKE_SOURCE_DIR}/src/third_party)
include_directories(${CMAKE_SOURCE_DIR}/src/src)
include_directories(${CMAKE_SOURCE_DIR}/src/src/analysis)
include_directories(${CMAKE_SOURCE_DIR}/src/src/commons)
include_directories(${CMAKE_SOURCE_DIR}/src/src/forest)
include_directories(${CMAKE_SOURCE_DIR}/src/src/prediction)
include_directories(${CMAKE_SOURCE_DIR}/src/src/relabeling)
include_directories(${CMAKE_SOURCE_DIR}/src/src/sampling)
include_directories(${CMAKE_SOURCE_DIR}/src/src/splitting)
include_directories(${CMAKE_SOURCE_DIR}/src/src/tree)

## ======================================================================================##
## Debug and release targets
## ======================================================================================##
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
ADD_CUSTOM_TARGET(debug
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Debug"
  )
ADD_CUSTOM_TARGET(release
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Release"
  )

## ======================================================================================##
## Executable
## ======================================================================================##
add_executable(output ${SOURCES} main.cpp main_test.cpp)