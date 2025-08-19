cmake_minimum_required(VERSION 3.15)
project(NeuralNet-C)

# Sources
add_executable(cnn
    src/main.c
    src/fileHandling.c
    src/helpers.c
)

# Option to enable CPU-specific optimisations (on by default)
option(USE_NATIVE_CPU "Enable -march=native optimisations for GCC/Clang" ON)

# Optimisations
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)

target_compile_options(cnn PRIVATE
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANG_AND_ID:C,Clang,GNU>>:$<IF:$<BOOL:${USE_NATIVE_CPU}>,-O3 -flto -ffast-math -march=native,-O3 -flto -ffast-math>>
    $<$<AND:$<CONFIG:Release>,$<C_COMPILER_ID:MSVC>>:/O2 /GL /fp:fast /DNDEBUG /permissive->
)

# Strip symbols on GCC/Clang
target_link_options(cnn PRIVATE
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANG_AND_ID:C,Clang,GNU>>:-s>
    $<$<AND:$<CONFIG:Release>,$<C_COMPILER_ID:MSVC>>:/LTCG>
)

# Cross-platform library linkage
include(CheckLibraryExists)
    if (NOT MSVC)
    check_library_exists(m sqrt "" HAVE_LIBM)

    if (HAVE_LIBM)
        target_link_libraries(cnn PRIVATE m)
    endif()
endif()

# Resource installation
install(TARGETS cnn RUNTIME DESTINATION .)
install(FILES config.cfg DESTINATION .)
install(FILES README.md DESTINATION .)
install(DIRECTORY data/ DESTINATION data)