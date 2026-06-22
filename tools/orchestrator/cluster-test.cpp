// cluster_link loopback test: exercise init / barrier / gather / broadcast with
// N MPI ranks on one host. Not part of the real tools - built only with
// -DLLAMA_BUILD_TESTS=ON and -DLLAMA_ORCH_MPI=ON. Run with:
//   mpirun -n 2 ./build-cpu/bin/llama-orchestrator-cluster-test
#include "cluster.h"

#include <cstdio>
#include <sstream>
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

    // distributed work counter: every rank claims indices until the shared counter is exhausted. the
    // union of all claimed indices must be exactly {0..TOTAL-1}, each exactly once (no dup, no gap) -
    // this is the cross-node work-stealing invariant.
    {
        const int TOTAL = 1000;
        link->counter_begin();
        link->barrier();

        std::vector<int> mine_idx;
        for (;;) {
            const long long g = link->claim(1);
            if (g >= TOTAL) {
                break;
            }
            mine_idx.push_back((int) g);
        }
        link->counter_end();

        std::string blob;
        for (int v : mine_idx) {
            blob += std::to_string(v) + " ";
        }
        std::vector<std::string> claimed = link->gather(blob);

        if (link->is_head()) {
            std::vector<int> seen(TOTAL, 0);
            int total_claimed = 0;
            for (const std::string & b : claimed) {
                std::stringstream ss(b);
                int v;
                while (ss >> v) {
                    CHECK(v >= 0 && v < TOTAL, "counter: claimed index out of range");
                    seen[v]++;
                    total_claimed++;
                }
            }
            CHECK(total_claimed == TOTAL, "counter: total claimed != TOTAL (gap or overrun)");
            for (int i = 0; i < TOTAL; ++i) {
                CHECK(seen[i] == 1, "counter: an index was not claimed exactly once");
            }
        }
    }

    link->barrier();
    if (link->is_head()) {
        printf("[cluster-test] PASS (%d ranks: init/barrier/gather/broadcast/counter)\n", size);
    }
    return 0;
}
