add_compile_options(-march=armv8-a+sve)
add_library(sve_kernels
        hello.cpp
        memcmp.cpp
        nBody_step.cpp
        intersectP.cpp
        accumulate.cpp)