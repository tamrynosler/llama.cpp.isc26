# llama-orchestrator: data-parallel replica pool + scheduler, linked by the tools

set(TARGET llama-orchestrator)

add_library(${TARGET} orchestrator.cpp)
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC llama-common llama ${CMAKE_THREAD_LIBS_INIT})

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} LIBRARY)
endif()

# optional S5 smoke test (not installed); enable with -DLLAMA_BUILD_TESTS=ON
if (LLAMA_BUILD_TESTS)
    set(TARGET llama-orchestrator-smoke)
    add_executable(${TARGET} smoke-test.cpp)
    target_link_libraries(${TARGET} PRIVATE llama-orchestrator llama-common llama ${CMAKE_THREAD_LIBS_INIT})
    target_compile_features(${TARGET} PRIVATE cxx_std_17)
endif()
