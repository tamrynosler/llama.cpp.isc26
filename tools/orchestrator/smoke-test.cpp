// S5 smoke test: load N replicas on CPU, check the accessors, tear down cleanly.
// not part of the real tools - built only with -DLLAMA_BUILD_TESTS=ON.
#include "orchestrator.h"

#include "common.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [n_replicas]\n", argv[0]);
        return 1;
    }

    common_init();        // set up logging so the orchestrator's messages show
    llama_backend_init(); // global backend init (matches every tool)

    common_params params;
    params.model.path   = argv[1];
    params.n_gpu_layers = 0; // CPU-only smoke test

    const int n = (argc >= 3) ? atoi(argv[2]) : 2;
    std::vector<int> devices(n, 0); // all replicas on "device 0" (CPU here)

    printf("\n[smoke] building %d replica(s) from '%s'\n", n, params.model.path.c_str());

    auto pool = orchestrator_make_pool(params, devices);
    if (!pool) {
        fprintf(stderr, "[smoke] FAIL: orchestrator_make_pool returned null\n");
        return 1;
    }

    bool ok = true;

    printf("[smoke] size() = %d (expected %d)\n", pool->size(), n);
    ok &= (pool->size() == n);

    for (int i = 0; i < pool->size(); ++i) {
        orchestrator_replica_info info = pool->at(i);
        printf("[smoke] at(%d): index=%d device=%d\n", i, info.index, info.device);
        ok &= (info.index == i);
    }

    const llama_model * m = pool->model();
    if (m) {
        const llama_vocab * vocab = llama_model_get_vocab(m);
        printf("[smoke] model() ok, vocab size = %d\n", llama_vocab_n_tokens(vocab));
    } else {
        printf("[smoke] FAIL: model() returned null\n");
        ok = false;
    }

    printf("[smoke] tearing down pool...\n");
    pool.reset(); // free all replicas now; should be clean

    llama_backend_free();

    printf("[smoke] %s\n\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
