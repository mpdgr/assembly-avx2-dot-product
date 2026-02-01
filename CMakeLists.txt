cmake_minimum_required(VERSION 3.31)
project(dot C)
set(CMAKE_C_STANDARD 11)
enable_language(ASM_MASM)

add_executable(dot main.c asm/dot_basic.asm asm/dot_ymm.asm)

if (MSVC)
    target_compile_options(dot PRIVATE
            $<$<CONFIG:Debug>:/Od /Zi>
            $<$<CONFIG:Release>:/O2>
    )
else()
    target_compile_options(dot PRIVATE
            $<$<CONFIG:Debug>:-O0 -g>
            $<$<CONFIG:Release>:-O3>
    )
endif()


# intel MKL
if(DEFINED ENV{MKLROOT})
    if(EXISTS "$ENV{MKLROOT}/include/mkl_cblas.h")
        target_include_directories(dot PRIVATE "$ENV{MKLROOT}/include")

        if(MSVC)
            target_link_libraries(dot PRIVATE
                    "$ENV{MKLROOT}/lib/mkl_intel_lp64.lib"
                    "$ENV{MKLROOT}/lib/mkl_sequential.lib"
                    "$ENV{MKLROOT}/lib/mkl_core.lib"
            )
        else() # GCC/Clang
            target_link_libraries(dot PRIVATE
                    "$ENV{MKLROOT}/lib/libmkl_intel_lp64.a"
                    "$ENV{MKLROOT}/lib/libmkl_sequential.a"
                    "$ENV{MKLROOT}/lib/libmkl_core.a"
                    -lpthread -lm -ldl
            )
        endif()
    endif()
endif()