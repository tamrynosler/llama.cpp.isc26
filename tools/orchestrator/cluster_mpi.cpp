// tools/orchestrator/cluster_mpi.cpp
// MPI implementation of cluster_link. Compiled into llama-orchestrator only when
// LLAMA_ORCH_MPI=ON (see CMakeLists.txt). The interface (cluster.h) is transport-
// neutral; this is the only impl built today.
//
// Threading contract: all collectives run on the MAIN thread, outside the pool's
// worker region, so we request MPI_THREAD_FUNNELED (MPI calls only from the
// thread that called MPI_Init). We never pass GPU pointers through MPI - blobs
// are small host byte buffers (per-config stats) - so CUDA-aware MPI is not
// needed.
#include "cluster.h"

#include <mpi.h>

#include <cstdio>
#include <stdexcept>

// ---- impl ------------------------------------------------------------------

struct cluster_link::impl {
    int  rank        = 0;
    int  size        = 1;
    bool we_init_mpi = false; // true if WE called MPI_Init (so WE finalize)

    // distributed work counter (counter_begin/claim/counter_end). the int64 lives on rank 0; every
    // rank reaches it via passive-target RMA (one shared lock epoch open for the counter's lifetime).
    MPI_Win counter_win = MPI_WIN_NULL;
    int64_t counter     = 0;
};

// ---- lifecycle -------------------------------------------------------------

std::unique_ptr<cluster_link> cluster_link::init(int * argc, char *** argv) {
    // unique_ptr<impl> first so we clean up on any throw below.
    std::unique_ptr<cluster_link> link(new cluster_link());

    int already = 0;
    if (MPI_Initialized(&already) != MPI_SUCCESS) {
        return nullptr;
    }

    if (!already) {
        int provided = MPI_THREAD_SINGLE;
        if (MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) {
            return nullptr;
        }
        link->p->we_init_mpi = true;
        if (provided < MPI_THREAD_FUNNELED) {
            // We only ever call collectives from the main thread, so this is a
            // warning, not a hard failure - but surface it: a runtime below
            // FUNNELED is misconfigured for our use.
            fprintf(stderr,
                    "[cluster] warning: MPI granted thread level %d < FUNNELED (%d); "
                    "collectives must stay on the main thread\n",
                    provided, MPI_THREAD_FUNNELED);
        }
    }

    if (MPI_Comm_rank(MPI_COMM_WORLD, &link->p->rank) != MPI_SUCCESS ||
        MPI_Comm_size(MPI_COMM_WORLD, &link->p->size) != MPI_SUCCESS) {
        return nullptr;
    }

    return link;
}

cluster_link::cluster_link() : p(new impl()) {}

cluster_link::~cluster_link() {
    // Only finalize if we initialized MPI and nobody finalized it already.
    if (p && p->we_init_mpi) {
        int finalized = 0;
        if (MPI_Finalized(&finalized) == MPI_SUCCESS && !finalized) {
            MPI_Finalize();
        }
    }
}

// ---- topology --------------------------------------------------------------

int  cluster_link::rank()    const { return p->rank; }
int  cluster_link::size()    const { return p->size; }
bool cluster_link::is_head() const { return p->rank == 0; }

// ---- collectives -----------------------------------------------------------

void cluster_link::barrier() {
    MPI_Barrier(MPI_COMM_WORLD);
}

std::vector<std::string> cluster_link::gather(const std::string & blob) {
    const int n = p->size;

    // Step 1: gather each rank's blob length to the head.
    const int my_len = (int) blob.size();
    std::vector<int> lengths(is_head() ? n : 0, 0);
    MPI_Gather(&my_len, 1, MPI_INT,
               is_head() ? lengths.data() : nullptr, 1, MPI_INT,
               0, MPI_COMM_WORLD);

    // Step 2: gather the variable-length bytes (MPI_Gatherv) to the head.
    std::vector<int> displs;
    std::vector<char> recvbuf;
    int total = 0;
    if (is_head()) {
        displs.resize(n, 0);
        for (int i = 0; i < n; ++i) {
            displs[i] = total;
            total    += lengths[i];
        }
        recvbuf.resize((size_t) total);
    }

    MPI_Gatherv(blob.data(), my_len, MPI_CHAR,
                is_head() ? recvbuf.data() : nullptr,
                is_head() ? lengths.data() : nullptr,
                is_head() ? displs.data()  : nullptr,
                MPI_CHAR, 0, MPI_COMM_WORLD);

    // Non-head ranks get an empty vector (by contract).
    std::vector<std::string> out;
    if (is_head()) {
        out.reserve(n);
        for (int i = 0; i < n; ++i) {
            out.emplace_back(recvbuf.data() + displs[i], (size_t) lengths[i]);
        }
    }
    return out;
}

std::string cluster_link::broadcast(const std::string & blob) {
    // Broadcast the length first, then the bytes, from rank 0.
    int len = is_head() ? (int) blob.size() : 0;
    MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::string out = is_head() ? blob : std::string((size_t) len, '\0');
    if (len > 0) {
        MPI_Bcast(out.data(), len, MPI_CHAR, 0, MPI_COMM_WORLD);
    }
    return out;
}

// ---- distributed work counter ---------------------------------------------

void cluster_link::counter_begin() {
    // rank 0 backs the counter (size 8); other ranks expose a zero-size window but still target rank 0.
    p->counter = 0;
    const MPI_Aint bytes = is_head() ? (MPI_Aint) sizeof(int64_t) : 0;
    MPI_Win_create(&p->counter, bytes, sizeof(int64_t),
                   MPI_INFO_NULL, MPI_COMM_WORLD, &p->counter_win);
    // one passive-target epoch open for the whole run; claim() uses MPI_Win_flush to complete each op.
    MPI_Win_lock_all(0, p->counter_win);
}

int64_t cluster_link::claim(int n) {
    int64_t increment = n;
    int64_t result    = 0;
    // atomic fetch-and-add on rank 0's counter; result is the pre-increment value (first reserved index).
    MPI_Fetch_and_op(&increment, &result, MPI_INT64_T, 0, /*disp=*/0, MPI_SUM, p->counter_win);
    MPI_Win_flush(0, p->counter_win); // ensure the RMA completed before we read `result`
    return result;
}

void cluster_link::counter_end() {
    if (p->counter_win != MPI_WIN_NULL) {
        MPI_Win_unlock_all(p->counter_win);
        MPI_Win_free(&p->counter_win);
        p->counter_win = MPI_WIN_NULL;
    }
}
