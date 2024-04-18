add_executable(benchmark_driver main.cpp
        driver_common.h
        memcmp_kernel.h
        ../test_driver/hello_test.cpp)
target_link_libraries(benchmark_driver benchmark::benchmark libpfm.so neon_kernels sve_kernels)