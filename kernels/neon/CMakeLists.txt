add_compile_options(-march=armv8-a)
add_library(neon_kernels
        helloNeon.cpp
        memcmpNeon.cpp)