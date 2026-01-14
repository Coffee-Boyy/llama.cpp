#include <benchmark/benchmark.h>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-metal.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

namespace {

constexpr int kGraphNodes = 256;
constexpr size_t kTensorOverheadCount = 128;

struct ggml_context_deleter {
    void operator()(ggml_context * ctx) const {
        if (ctx) {
            ggml_free(ctx);
        }
    }
};

struct MetalBackendHolder {
    ggml_backend_t backend = nullptr;

    MetalBackendHolder() {
        ggml_backend_load_all();
        backend = ggml_backend_init_by_name("Metal", nullptr);
        if (!backend) {
            std::fprintf(stderr, "ggml-metal-microbench: failed to init Metal backend\n");
            std::abort();
        }
    }

    ~MetalBackendHolder() {
        if (backend) {
            ggml_backend_free(backend);
        }
    }
};

ggml_backend_t get_metal_backend() {
    static MetalBackendHolder holder;
    return holder.backend;
}

std::unique_ptr<ggml_context, ggml_context_deleter> make_context(size_t tensor_count = kTensorOverheadCount) {
    ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead() * tensor_count + ggml_graph_overhead_custom(kGraphNodes, false),
        /* .mem_base   = */ nullptr,
        /* .no_alloc   = */ true,
    };
    return std::unique_ptr<ggml_context, ggml_context_deleter>(ggml_init(params));
}

void fill_tensor_random(ggml_tensor * t) {
    static std::mt19937 rng(42);

    switch (t->type) {
        case GGML_TYPE_F32: {
            std::vector<float> data(ggml_nelements(t));
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (float & v : data) {
                v = dist(rng);
            }
            ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(float));
        } break;
        case GGML_TYPE_F16: {
            std::vector<ggml_fp16_t> data(ggml_nelements(t));
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (ggml_fp16_t & v : data) {
                v = ggml_fp32_to_fp16(dist(rng));
            }
            ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(ggml_fp16_t));
        } break;
        case GGML_TYPE_I32: {
            std::vector<int32_t> data(ggml_nelements(t));
            for (size_t i = 0; i < data.size(); ++i) {
                data[i] = static_cast<int32_t>(i);
            }
            ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(int32_t));
        } break;
        default: {
            std::vector<uint8_t> data(ggml_nbytes(t));
            std::uniform_int_distribution<int> dist(0, 255);
            for (uint8_t & v : data) {
                v = static_cast<uint8_t>(dist(rng));
            }
            ggml_backend_tensor_set(t, data.data(), 0, data.size());
        } break;
    }
}

struct BenchGraph {
    std::unique_ptr<ggml_context, ggml_context_deleter> ctx;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * out = nullptr;

    BenchGraph() = default;
    BenchGraph(const BenchGraph &) = delete;
    BenchGraph & operator=(const BenchGraph &) = delete;
    BenchGraph(BenchGraph && other) noexcept
        : ctx(std::move(other.ctx)),
          buffer(other.buffer),
          graph(other.graph),
          out(other.out) {
        other.buffer = nullptr;
        other.graph = nullptr;
        other.out = nullptr;
    }
    BenchGraph & operator=(BenchGraph && other) noexcept {
        if (this != &other) {
            if (buffer) {
                ggml_backend_buffer_free(buffer);
            }
            ctx = std::move(other.ctx);
            buffer = other.buffer;
            graph = other.graph;
            out = other.out;
            other.buffer = nullptr;
            other.graph = nullptr;
            other.out = nullptr;
        }
        return *this;
    }

    ~BenchGraph() {
        if (buffer) {
            ggml_backend_buffer_free(buffer);
        }
    }
};

void run_benchmark(benchmark::State & state, BenchGraph & bench) {
    ggml_backend_t backend = get_metal_backend();

    if (!ggml_backend_supports_op(backend, bench.out)) {
        state.SkipWithError("Metal backend does not support op");
        return;
    }

    ggml_status status = ggml_backend_graph_compute(backend, bench.graph);
    if (status != GGML_STATUS_SUCCESS) {
        state.SkipWithError("warmup failed");
        return;
    }
    ggml_backend_synchronize(backend);

    for (auto _ : state) {
        ggml_status st = ggml_backend_graph_compute(backend, bench.graph);
        if (st != GGML_STATUS_SUCCESS) {
            state.SkipWithError("compute failed");
            break;
        }
        ggml_backend_synchronize(backend);
    }
}

BenchGraph build_mul_mat_f16(int64_t m, int64_t k, int64_t n) {
    BenchGraph bench;
    bench.ctx = make_context();

    ggml_tensor * w = ggml_new_tensor_2d(bench.ctx.get(), GGML_TYPE_F16, k, m);
    ggml_tensor * x = ggml_new_tensor_2d(bench.ctx.get(), GGML_TYPE_F16, k, n);
    bench.out = ggml_mul_mat(bench.ctx.get(), w, x);

    bench.buffer = ggml_backend_alloc_ctx_tensors(bench.ctx.get(), get_metal_backend());
    fill_tensor_random(w);
    fill_tensor_random(x);

    bench.graph = ggml_new_graph_custom(bench.ctx.get(), kGraphNodes, false);
    ggml_build_forward_expand(bench.graph, bench.out);
    return bench;
}

BenchGraph build_mul_mat_q4_0(int64_t m, int64_t k, int64_t n) {
    BenchGraph bench;
    bench.ctx = make_context();

    ggml_tensor * w = ggml_new_tensor_2d(bench.ctx.get(), GGML_TYPE_Q4_0, k, m);
    ggml_tensor * x = ggml_new_tensor_2d(bench.ctx.get(), GGML_TYPE_F32, k, n);
    bench.out = ggml_mul_mat(bench.ctx.get(), w, x);

    bench.buffer = ggml_backend_alloc_ctx_tensors(bench.ctx.get(), get_metal_backend());
    fill_tensor_random(w);
    fill_tensor_random(x);

    bench.graph = ggml_new_graph_custom(bench.ctx.get(), kGraphNodes, false);
    ggml_build_forward_expand(bench.graph, bench.out);
    return bench;
}

BenchGraph build_rms_norm(int64_t n_embd) {
    BenchGraph bench;
    bench.ctx = make_context();

    ggml_tensor * x = ggml_new_tensor_2d(bench.ctx.get(), GGML_TYPE_F32, n_embd, 1);
    bench.out = ggml_rms_norm(bench.ctx.get(), x, 1e-5f);

    bench.buffer = ggml_backend_alloc_ctx_tensors(bench.ctx.get(), get_metal_backend());
    fill_tensor_random(x);

    bench.graph = ggml_new_graph_custom(bench.ctx.get(), kGraphNodes, false);
    ggml_build_forward_expand(bench.graph, bench.out);
    return bench;
}

BenchGraph build_rope(int64_t n_dims, int64_t n_heads, int64_t n_tokens) {
    BenchGraph bench;
    bench.ctx = make_context();

    ggml_tensor * x = ggml_new_tensor_3d(bench.ctx.get(), GGML_TYPE_F32, n_dims, n_heads, n_tokens);
    ggml_tensor * pos = ggml_new_tensor_1d(bench.ctx.get(), GGML_TYPE_I32, n_tokens);
    bench.out = ggml_rope_ext(
        bench.ctx.get(), x, pos, nullptr,
        static_cast<int>(n_dims), GGML_ROPE_TYPE_NEOX, static_cast<int>(n_tokens),
        10000.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f);

    bench.buffer = ggml_backend_alloc_ctx_tensors(bench.ctx.get(), get_metal_backend());
    fill_tensor_random(x);
    fill_tensor_random(pos);

    bench.graph = ggml_new_graph_custom(bench.ctx.get(), kGraphNodes, false);
    ggml_build_forward_expand(bench.graph, bench.out);
    return bench;
}

BenchGraph build_softmax(int64_t n_kv, int64_t n_heads) {
    BenchGraph bench;
    bench.ctx = make_context();

    ggml_tensor * x = ggml_new_tensor_2d(bench.ctx.get(), GGML_TYPE_F32, n_kv, n_heads);
    bench.out = ggml_soft_max(bench.ctx.get(), x);

    bench.buffer = ggml_backend_alloc_ctx_tensors(bench.ctx.get(), get_metal_backend());
    fill_tensor_random(x);

    bench.graph = ggml_new_graph_custom(bench.ctx.get(), kGraphNodes, false);
    ggml_build_forward_expand(bench.graph, bench.out);
    return bench;
}

BenchGraph build_flash_attn(int64_t head_dim, int64_t n_heads, int64_t n_heads_kv, int64_t n_kv, int64_t n_batch) {
    BenchGraph bench;
    bench.ctx = make_context();

    ggml_tensor * q = ggml_new_tensor_4d(bench.ctx.get(), GGML_TYPE_F32, head_dim, n_batch, n_heads, 1);
    ggml_tensor * k = ggml_new_tensor_4d(bench.ctx.get(), GGML_TYPE_F32, head_dim, n_kv, n_heads_kv, 1);
    ggml_tensor * v = ggml_new_tensor_4d(bench.ctx.get(), GGML_TYPE_F32, head_dim, n_kv, n_heads_kv, 1);
    ggml_tensor * mask = ggml_new_tensor_4d(bench.ctx.get(), GGML_TYPE_F16, n_kv, n_batch, 1, 1);

    bench.out = ggml_flash_attn_ext(bench.ctx.get(), q, k, v, mask, 1.0f / std::sqrt(static_cast<float>(head_dim)), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(bench.out, GGML_PREC_DEFAULT);

    bench.buffer = ggml_backend_alloc_ctx_tensors(bench.ctx.get(), get_metal_backend());
    fill_tensor_random(q);
    fill_tensor_random(k);
    fill_tensor_random(v);
    fill_tensor_random(mask);

    bench.graph = ggml_new_graph_custom(bench.ctx.get(), kGraphNodes, false);
    ggml_build_forward_expand(bench.graph, bench.out);
    return bench;
}

} // namespace

static void BM_MulMatF16(benchmark::State & state) {
    const int64_t m = 4096;
    const int64_t k = 4096;
    const int64_t n = 1;
    auto bench = build_mul_mat_f16(m, k, n);
    run_benchmark(state, bench);
}
BENCHMARK(BM_MulMatF16);

static void BM_MulMatQ4_0(benchmark::State & state) {
    const int64_t m = 4096;
    const int64_t k = 4096;
    const int64_t n = 1;
    auto bench = build_mul_mat_q4_0(m, k, n);
    run_benchmark(state, bench);
}
BENCHMARK(BM_MulMatQ4_0);

static void BM_RmsNorm(benchmark::State & state) {
    auto bench = build_rms_norm(4096);
    run_benchmark(state, bench);
}
BENCHMARK(BM_RmsNorm);

static void BM_Rope(benchmark::State & state) {
    auto bench = build_rope(128, 32, 1);
    run_benchmark(state, bench);
}
BENCHMARK(BM_Rope);

static void BM_Softmax(benchmark::State & state) {
    auto bench = build_softmax(128, 32);
    run_benchmark(state, bench);
}
BENCHMARK(BM_Softmax);

static void BM_FlashAttn(benchmark::State & state) {
    auto bench = build_flash_attn(128, 32, 8, 128, 1);
    run_benchmark(state, bench);
}
BENCHMARK(BM_FlashAttn);
