add_library(spnc SHARED src/Driver.cpp
                        src/graph-ir/GraphIRNode.cpp
                        src/graph-ir/InputVar.cpp
                        src/graph-ir/Histogram.cpp
                        src/graph-ir/WeightedSum.cpp
                        src/graph-ir/Sum.cpp
                        src/graph-ir/Product.cpp
                        src/json/Parser.cpp
                        src/transform/BaseVisitor.cpp)

target_include_directories(spnc
                            PUBLIC
                            $<INSTALL_INTERFACE:include>
                            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>

                            PRIVATE
                            ${CMAKE_CURRENT_SOURCE_DIR}/src)