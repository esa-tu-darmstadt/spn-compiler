# ==============================================================================
#  This file is part of the SPNC project under the Apache License v2.0 by the
#  Embedded Systems and Applications Group, TU Darmstadt.
#  For the full copyright and license information, please view the LICENSE
#  file that was distributed with this source code.
#  SPDX-License-Identifier: Apache-2.0
# ==============================================================================

macro(doxygen_setup)
    option(SPNC_BUILD_DOC "Build Doxygen documentation" ON)

    if (${SPNC_BUILD_DOC})
        find_package(Doxygen REQUIRED dot)
        if (${DOXYGEN_FOUND})
            message(STATUS "Found Doxygen ${DOXYGEN_VERSION}")
            if (${DOXYGEN_VERSION} VERSION_GREATER_EQUAL "1.8.15")
                set(DOXY_LAYOUT_FILE ${PROJECT_SOURCE_DIR}/docs/doxy-layout-1_8_15.xml)
            else (${DOXYGEN_VERSION} VERSION_GREATER_EQUAL "1.8.15")
                set(DOXY_LAYOUT_FILE ${PROJECT_SOURCE_DIR}/docs/doxy-layout-1_8_13.xml)
            endif (${DOXYGEN_VERSION} VERSION_GREATER_EQUAL "1.8.15")
        else (${DOXYGEN_FOUND})
            message(STATUS "Doxygen required to generate documentation!")
        endif (${DOXYGEN_FOUND})
    else (${SPNC_BUILD_DOC})
        set(DOXYGEN_FOUND false)
    endif (${SPNC_BUILD_DOC})
    add_custom_target(doxygen-all)
endmacro(doxygen_setup)

function(doxygen_toplevel)

    if (${SPNC_BUILD_DOC} AND ${DOXYGEN_FOUND})
        set(DOXYGEN_USE_MDFILE_AS_MAINPAGE ${PROJECT_SOURCE_DIR}/docs/index.md)
        set(DOXYGEN_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/doc)

        doxygen_add_docs(doxygen-toplevel
                ${PROJECT_SOURCE_DIR}/docs
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

        add_dependencies(doxygen-toplevel doxygen-all)
        add_custom_target(doxygen-doc ALL)
        add_dependencies(doxygen-doc doxygen-toplevel)
    endif (${SPNC_BUILD_DOC} AND ${DOXYGEN_FOUND})
endfunction(doxygen_toplevel)

function(doxygen_doc)
    set(oneValueArgs TARGET_NAME)
    set(multiValueArgs SRC_DIRECTORIES DEPENDS EXCLUDES)
    cmake_parse_arguments(DOXY "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (${SPNC_BUILD_DOC} AND ${DOXYGEN_FOUND})
        set(DOXY_BASE_DIR ${PROJECT_BINARY_DIR}/doc)
        set(DOXYGEN_OUTPUT_DIRECTORY ${DOXY_BASE_DIR}/${DOXY_TARGET_NAME})
        message(STATUS "Doxygen layout: ${DOXY_LAYOUT_FILE}")
        set(DOXYGEN_LAYOUT_FILE ${DOXY_LAYOUT_FILE})
        set(DOXYGEN_EXCLUDE ${DOXY_EXCLUDES})
        # Generate own tag-file for use by targets depending on this one.
        set(DOXYGEN_GENERATE_TAGFILE ${DOXYGEN_OUTPUT_DIRECTORY}/${DOXY_TARGET_NAME}.tag)
        # Fill DOXYGEN_TAGFILES with the tag-files of the dependencies.
        foreach (target ${DOXY_DEPENDS})
            list(APPEND tagfiles "${DOXY_BASE_DIR}/${target}/${target}.tag=${DOXY_BASE_DIR}/${target}/html")
            list(APPEND dependencies "DOC_${target}")
            # Add dependencies to ensure compilation happens before doxygen.
            add_dependencies(DOC_${target} ${DOXY_TARGET_NAME})
        endforeach (target)
        message(STATUS "TAGFILES: ${tagfiles}")
        set(DOXYGEN_TAGFILES ${tagfiles})
        # Generate Doxygen
        doxygen_add_docs("DOC_${DOXY_TARGET_NAME}"
                ${DOXY_SRC_DIRECTORIES}
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
        # Specify dependency on the Doxygen generation of the dependencies.
        if (dependencies)
            message(STATUS "DEPENDENCIES: ${dependencies}")
            add_dependencies("DOC_${DOXY_TARGET_NAME}" ${dependencies})
        endif (dependencies)
        add_dependencies("DOC_${DOXY_TARGET_NAME}" ${DOXY_TARGET_NAME})
        add_dependencies(doxygen-all "DOC_${DOXY_TARGET_NAME}")
    endif (${SPNC_BUILD_DOC} AND ${DOXYGEN_FOUND})
    # Add a dependency to the "doxygen-all" target to trigger execution of this target.
endfunction(doxygen_doc)