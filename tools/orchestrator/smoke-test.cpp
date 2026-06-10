// S5 smoke test: load N replicas on CPU, check the accessors, tear down cleanly.
// not part of the real tools - built only with -DLLAMA_BUILD_TESTS=ON.
#include "orchestrator.h"

#include "common.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

// parse a comma-separated device list like "0,1,2,3" into {0,1,2,3}
static std::vector<int> parse_devices(const std::string & s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) {
            out.push_back(std::stoi(tok));
        }
    }
    return out;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr,
                "usage: %s <model.gguf> [devices] [n_gpu_layers]\n"
                "  devices:      comma-separated GPU indices (default \"0,0\" = 2 replicas on GPU 0)\n"
                "  n_gpu_layers: layers to offload per replica (default 99 = all; 0 = keep on CPU)\n",
                argv[0]);
        return 1;
    }

    common_init();        // set up logging so the orchestrator's messages show
    llama_backend_init(); // global backend init (matches every tool)

    common_params params;
    params.model.path   = argv[1];
    params.n_gpu_layers = (argc >= 4) ? atoi(argv[3]) : 99; // offload all by default

    std::vector<int> devices = parse_devices((argc >= 3) ? argv[2] : "0,0");
    const int n = (int) devices.size();
    if (n == 0) {
        fprintf(stderr, "[smoke] FAIL: empty device list\n");
        return 1;
    }

    printf("\n[smoke] building %d replica(s) from '%s' (n_gpu_layers=%d)\n",
           n, params.model.path.c_str(), params.n_gpu_layers);

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
        printf("[smoke] at(%d): index=%d device=%d (expected device=%d)\n",
               i, info.index, info.device, devices[i]);
        ok &= (info.index == i);
        ok &= (info.device == devices[i]);
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
