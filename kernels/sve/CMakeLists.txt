add_compile_options(-march=armv8-a+sve)
add_library(sve_kernels helloSVE.cpp)