// cluster_link loopback test: exercise init / barrier / gather / broadcast with
// N MPI ranks on one host. Not part of the real tools - built only with
// -DLLAMA_BUILD_TESTS=ON and -DLLAMA_ORCH_MPI=ON. Run with:
//   mpirun -n 2 ./build-cpu/bin/llama-orchestrator-cluster-test
#include "cluster.h"

#include <cstdio>
#include <string>
#include <vector>

// tiny assert that prints which rank failed and returns a nonzero exit code
#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "[cluster-test] FAIL (rank %d): %s\n", rank, msg); \
            return 1;                                                          \
        }                                                                      \
    } while (0)

int main(int argc, char ** argv) {
    auto link = cluster_link::init(&argc, &argv);
    if (!link) {
        fprintf(stderr, "[cluster-test] FAIL: cluster_link::init returned null\n");
        return 1;
    }

    const int rank = link->rank();
    const int size = link->size();

    CHECK(size >= 1, "size must be >= 1");
    CHECK(rank >= 0 && rank < size, "rank out of range");
    CHECK(link->is_head() == (rank == 0), "is_head must mean rank==0");

    // barrier: all ranks must reach this point before any proceeds.
    link->barrier();

    // gather: each rank contributes a distinct, variable-length blob; the head
    // must receive exactly `size` blobs in rank order.
    const std::string mine = "rank=" + std::to_string(rank) +
                             std::string((size_t) rank, '.'); // varies length per rank
    std::vector<std::string> all = link->gather(mine);

    if (link->is_head()) {
        CHECK((int) all.size() == size, "head must receive `size` blobs");
        for (int i = 0; i < size; ++i) {
            const std::string want = "rank=" + std::to_string(i) +
                                     std::string((size_t) i, '.');
            CHECK(all[i] == want, "gathered blob out of order or corrupted");
        }
    } else {
        CHECK(all.empty(), "non-head must receive an empty vector");
    }

    // broadcast: rank 0's blob must reach every rank verbatim.
    const std::string bcast = link->broadcast(link->is_head() ? "hello-from-head" : "");
    CHECK(bcast == "hello-from-head", "broadcast did not deliver the head's blob");

    link->barrier();
    if (link->is_head()) {
        printf("[cluster-test] PASS (%d ranks: init/barrier/gather/broadcast)\n", size);
    }
    return 0;
}
