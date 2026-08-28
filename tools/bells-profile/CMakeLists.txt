set(TARGET llama-bells-profile)
add_executable(${TARGET} bells-profile.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()

set(SELFTEST llama-bells-selftest)
add_executable(${SELFTEST} bells-selftest.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../../src/llama-bells.cpp)
target_include_directories(${SELFTEST} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
target_link_libraries(${SELFTEST} PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${SELFTEST} PRIVATE cxx_std_17)
