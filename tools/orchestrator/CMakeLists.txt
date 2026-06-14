# llama-orchestrator: data-parallel replica pool + scheduler, linked by the tools

set(TARGET llama-orchestrator)

# Optional cross-node coordination layer (cluster_link). OFF by default: when off,
# no MPI source is compiled and the single-node build is byte-identical. When on,
# the MPI impl (cluster_mpi.cpp) is compiled into the lib (HPC-X / Open MPI etc.).
option(LLAMA_ORCH_MPI "orchestrator: build the MPI cross-node cluster_link" OFF)

set(ORCH_SOURCES orchestrator.cpp)
if (LLAMA_ORCH_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
    list(APPEND ORCH_SOURCES cluster_mpi.cpp)
endif()

add_library(${TARGET} ${ORCH_SOURCES})
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC llama-common llama ${CMAKE_THREAD_LIBS_INIT})

if (LLAMA_ORCH_MPI)
    target_link_libraries(${TARGET} PUBLIC MPI::MPI_CXX)
    target_compile_definitions(${TARGET} PUBLIC LLAMA_ORCH_MPI)
endif()

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

# optional cluster_link loopback test (not installed); needs both the MPI impl
# and the test flag. Run with: mpirun -n 2 ./bin/llama-orchestrator-cluster-test
if (LLAMA_BUILD_TESTS AND LLAMA_ORCH_MPI)
    set(TARGET llama-orchestrator-cluster-test)
    add_executable(${TARGET} cluster-test.cpp)
    target_link_libraries(${TARGET} PRIVATE llama-orchestrator MPI::MPI_CXX)
    target_compile_features(${TARGET} PRIVATE cxx_std_17)
endif()
