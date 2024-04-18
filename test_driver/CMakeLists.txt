enable_testing()

add_executable(test_driver
                hello_test.cpp)

include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

target_link_libraries(test_driver gtest gtest_main neon_kernels sve_kernels)

gtest_discover_tests(test_driver)
