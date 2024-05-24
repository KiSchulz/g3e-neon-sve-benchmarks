add_compile_options(-march=armv8-a)
add_library(neon_kernels
        hello.cpp
        memcmp.cpp
        nBody_step.cpp
        intersectP.cpp
        accumulate.cpp)