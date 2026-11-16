cmake_minimum_required(VERSION 3.15)
project(histopresto LANGUAGES C)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# named libkernel so that Python does not mistake the library for an extension module called _kernel
add_library(kernel SHARED src/histopresto/kernel.c)
# no fast-math and no fused multiply-add: the arithmetic then matches the JavaScript operation by operation
target_compile_options(kernel PRIVATE -O3 -ffp-contract=off)
target_link_libraries(kernel PRIVATE m)

install(TARGETS kernel LIBRARY DESTINATION histopresto)
