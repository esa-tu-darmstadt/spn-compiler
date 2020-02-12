#
# This file is part of the SPNC project.
# Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
#
function(add_bitcode_library)
    cmake_parse_arguments(BITCODE_ARGS "" "LIBNAME" "SOURCES" ${ARGN})
    if(NOT BITCODE_ARGS_LIBNAME)
        message(FATAL_ERROR "You must provide a name for the bitcode library.")
    endif(NOT BITCODE_ARGS_LIBNAME)

    if(NOT DEFINED LLVM_TOOLS_BINARY_DIR)
        message(FATAL_ERROR "Expecting definition of LLVM_TOOLS_BINARY_DIR!")
    endif(NOT DEFINED LLVM_TOOLS_BINARY_DIR)

    LIST(APPEND CMAKE_PROGRAM_PATH "${LLVM_TOOLS_BINARY_DIR}")

    find_program(CLANG clang "${LLVM_TOOLS_BINARY_DIR}")
    if(NOT CLANG)
        message(FATAL_ERROR "Did not find clang executable!")
    endif(NOT CLANG)
    message(STATUS "${BITCODE_ARGS_LIBNAME} - Using clang: ${CLANG}")

    find_program(LLVM_LINK llvm-link "${LLVM_TOOLS_BINARY_DIR}")
    if(NOT LLVM_LINK)
        message(FATAL_ERROR "Did not find llvm-link executable!")
    endif(NOT LLVM_LINK)
    message(STATUS "${BITCODE_ARGS_LIBNAME} - Using llvm-link: ${LLVM_LINK}")

    foreach(src ${BITCODE_ARGS_SOURCES})
        get_filename_component(path ${src} REALPATH)
        get_filename_component(name ${src} NAME)
        set(output_dir "${CMAKE_CURRENT_BINARY_DIR}/${BITCODE_ARGS_LIBNAME}")
        file(MAKE_DIRECTORY "${output_dir}")
        set(bc_file "${output_dir}/${name}.bc")
        add_custom_command(OUTPUT "${bc_file}"
                COMMAND ${CLANG} -emit-llvm ${path} -c -o "${bc_file}"
                DEPENDS "${path}")
        list(APPEND bc_files ${bc_file})
    endforeach(src)
    set(bc_lib "${output_dir}/${BITCODE_ARGS_LIBNAME}.bc")
    add_custom_command(OUTPUT "${bc_lib}" COMMAND ${LLVM_LINK} ${bc_files} -o "${bc_lib}" DEPENDS ${bc_files})
    add_custom_target(${BITCODE_ARGS_LIBNAME} ALL DEPENDS ${bc_lib})
endfunction(add_bitcode_library)