# add_compile_options(-mcpu=neoverse-v1 -mtune=neoverse-v1)
add_subdirectory(reference)
add_subdirectory(neon)
add_subdirectory(sve)