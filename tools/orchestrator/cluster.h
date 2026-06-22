// tools/orchestrator/cluster.h
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// cluster_link - cross-node coordination for the data-parallel orchestrator.
// Mirrors orchestrator_pool one level up: the pool coordinates N GPU replicas
// in ONE process; cluster_link coordinates N nodes, each running its own pool.
// Inference stays 100% node-local - this layer carries only a start barrier +
// an end-of-config stats reduce (kilobytes, infrequent). Never put per-token or
// per-step traffic here; that node-local property is what preserves the measured
// node scaling.
//
// Transport is chosen at BUILD time: exactly one impl .cpp is compiled
// (cluster_mpi.cpp when LLAMA_ORCH_MPI=ON). The driver/bench code never sees the
// transport. All collectives are called from the MAIN thread only, OUTSIDE the
// pool's worker region (the MPI_THREAD_FUNNELED contract).
struct cluster_link {
    // Initialize the cluster runtime once at process start (MPI_Init consumes
    // argc/argv). A non-cluster launch (no mpirun/srun) still succeeds with
    // size()==1, rank()==0, so --cluster degrades cleanly to a local run.
    // Returns nullptr if the runtime fails to initialize.
    static std::unique_ptr<cluster_link> init(int * argc, char *** argv);

    ~cluster_link();                                         // finalizes the runtime
    cluster_link(const cluster_link &)             = delete; // one runtime per process
    cluster_link & operator=(const cluster_link &) = delete;

    int  rank() const;    // 0 .. size()-1
    int  size() const;    // number of ranks (nodes)
    bool is_head() const; // rank() == 0

    // Block until all ranks arrive.
    void barrier();

    // Every rank contributes one opaque blob; rank 0 returns all blobs in rank
    // order (size() entries). Non-head ranks return an empty vector. Variable
    // lengths allowed (MPI_Gatherv under the hood).
    std::vector<std::string> gather(const std::string & blob);

    // rank 0's blob copied to every rank; returns the head's blob on all ranks.
    // Declared now so the contract is stable; first consumer is Phase 2.
    std::string broadcast(const std::string & blob);

    // ---- distributed work counter (cross-node dynamic load balancing) ----
    // A single monotonic counter hosted on rank 0, for pulling global work indices on demand.
    // counter_begin()/counter_end() are COLLECTIVE (all ranks, paired). claim(n) atomically reserves
    // the next `n` indices and returns the first reserved index (the value BEFORE the increment); the
    // caller stops once the returned index is >= its known total. Like the collectives above, these
    // are called from the MAIN thread only (FUNNELED). Coarse-grained - one call per work item/batch,
    // never per token/step - so the node-local property that preserves the measured scaling holds.
    void    counter_begin();
    int64_t claim(int n);
    void    counter_end();

private:
    cluster_link();
    struct impl;
    std::unique_ptr<impl> p;
};
