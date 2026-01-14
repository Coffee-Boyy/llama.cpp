#include <benchmark/benchmark.h>

#include "ggml.h"
#include "ggml-backend.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <algorithm>
#include <string>
#include <string_view>
#include <vector>

namespace {
bool has_prefix(const std::string_view value, const std::string_view prefix) {
    return value.size() >= prefix.size() && value.substr(0, prefix.size()) == prefix;
}

bool has_flag(const std::vector<std::string> & args, const std::string_view flag) {
    for (size_t i = 0; i < args.size(); ++i) {
        const std::string & arg = args[i];
        if (arg == flag) {
            return true;
        }
        if (has_prefix(arg, flag) && arg.size() > flag.size() && arg[flag.size()] == '=') {
            return true;
        }
    }
    return false;
}

int parse_flag_int(const std::string & value, int fallback) {
    try {
        size_t offset = 0;
        int parsed = std::stoi(value, &offset, 10);
        if (offset == value.size()) {
            return parsed;
        }
    } catch (const std::exception &) {
    }
    return fallback;
}

bool parse_flag_string(const std::string & value, std::string & out) {
    if (!value.empty()) {
        out = value;
        return true;
    }
    return false;
}


constexpr int kGraphNodes = 256;
constexpr size_t kTensorOverheadCount = 128;

struct ggml_context_deleter {
    void operator()(ggml_context * ctx) const {
        if (ctx) {
            ggml_free(ctx);
        }
    }
};

std::string g_backend_name = "CPU";

struct BackendHolder {
    ggml_backend_t backend = nullptr;

    BackendHolder() {
        ggml_backend_load_all();
        backend = ggml_backend_init_by_name(g_backend_name.c_str(), nullptr);
        if (!backend) {
            std::fprintf(stderr, "ggml-microbench: failed to init backend '%s'\n", g_backend_name.c_str());
            std::abort();
        }
    }

    ~BackendHolder() {
        if (backend) {
            ggml_backend_free(backend);
        }
    }
};

ggml_backend_t get_backend() {
    static BackendHolder holder;
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

void fill_tensor_pattern(ggml_tensor * t) {
    switch (t->type) {
        case GGML_TYPE_F32: {
            std::vector<float> data(ggml_nelements(t));
            for (size_t i = 0; i < data.size(); ++i) {
                data[i] = static_cast<float>((i % 4096) / 2048.0f - 1.0f);
            }
            ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(float));
        } break;
        case GGML_TYPE_F16: {
            std::vector<ggml_fp16_t> data(ggml_nelements(t));
            for (size_t i = 0; i < data.size(); ++i) {
                float value = static_cast<float>((i % 4096) / 2048.0f - 1.0f);
                data[i] = ggml_fp32_to_fp16(value);
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
            for (size_t i = 0; i < data.size(); ++i) {
                data[i] = static_cast<uint8_t>(i & 0xff);
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
    ggml_backend_t backend = get_backend();

    if (!ggml_backend_supports_op(backend, bench.out)) {
        state.SkipWithError("backend does not support op");
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

    bench.buffer = ggml_backend_alloc_ctx_tensors(bench.ctx.get(), get_backend());
    fill_tensor_pattern(w);
    fill_tensor_pattern(x);

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

    bench.buffer = ggml_backend_alloc_ctx_tensors(bench.ctx.get(), get_backend());
    fill_tensor_pattern(w);
    fill_tensor_pattern(x);

    bench.graph = ggml_new_graph_custom(bench.ctx.get(), kGraphNodes, false);
    ggml_build_forward_expand(bench.graph, bench.out);
    return bench;
}

BenchGraph build_rms_norm(int64_t n_embd) {
    BenchGraph bench;
    bench.ctx = make_context();

    ggml_tensor * x = ggml_new_tensor_2d(bench.ctx.get(), GGML_TYPE_F32, n_embd, 1);
    bench.out = ggml_rms_norm(bench.ctx.get(), x, 1e-5f);

    bench.buffer = ggml_backend_alloc_ctx_tensors(bench.ctx.get(), get_backend());
    fill_tensor_pattern(x);

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

    bench.buffer = ggml_backend_alloc_ctx_tensors(bench.ctx.get(), get_backend());
    fill_tensor_pattern(x);
    fill_tensor_pattern(pos);

    bench.graph = ggml_new_graph_custom(bench.ctx.get(), kGraphNodes, false);
    ggml_build_forward_expand(bench.graph, bench.out);
    return bench;
}

BenchGraph build_softmax(int64_t n_kv, int64_t n_heads) {
    BenchGraph bench;
    bench.ctx = make_context();

    ggml_tensor * x = ggml_new_tensor_2d(bench.ctx.get(), GGML_TYPE_F32, n_kv, n_heads);
    bench.out = ggml_soft_max(bench.ctx.get(), x);

    bench.buffer = ggml_backend_alloc_ctx_tensors(bench.ctx.get(), get_backend());
    fill_tensor_pattern(x);

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

    bench.buffer = ggml_backend_alloc_ctx_tensors(bench.ctx.get(), get_backend());
    fill_tensor_pattern(q);
    fill_tensor_pattern(k);
    fill_tensor_pattern(v);
    fill_tensor_pattern(mask);

    bench.graph = ggml_new_graph_custom(bench.ctx.get(), kGraphNodes, false);
    ggml_build_forward_expand(bench.graph, bench.out);
    return bench;
}

void apply_mul_mat_args(benchmark::internal::Benchmark * benchmark) {
    const std::vector<int64_t> sizes = { 1024, 2048, 4096 };
    const std::vector<int64_t> batch = { 1, 4 };
    for (int64_t m : sizes) {
        for (int64_t k : sizes) {
            for (int64_t n : batch) {
                benchmark->Args({m, k, n});
            }
        }
    }
}

void apply_rope_args(benchmark::internal::Benchmark * benchmark) {
    const std::vector<int64_t> dims = { 64, 128, 256 };
    const std::vector<int64_t> heads = { 8, 16, 32 };
    const std::vector<int64_t> tokens = { 1, 16, 128 };
    for (int64_t n_dims : dims) {
        for (int64_t n_heads : heads) {
            for (int64_t n_tokens : tokens) {
                benchmark->Args({n_dims, n_heads, n_tokens});
            }
        }
    }
}

void apply_flash_attn_args(benchmark::internal::Benchmark * benchmark) {
    const std::vector<int64_t> head_dim = { 64, 128 };
    const std::vector<int64_t> head_counts = { 8, 16, 32 };
    const std::vector<int64_t> kv_tokens = { 64, 128 };
    const std::vector<int64_t> batches = { 1, 4 };
    for (int64_t dim : head_dim) {
        for (int64_t n_heads : head_counts) {
            for (int64_t n_heads_kv : { std::max<int64_t>(1, n_heads / 4), std::max<int64_t>(1, n_heads / 2) }) {
                for (int64_t n_kv : kv_tokens) {
                    for (int64_t n_batch : batches) {
                        benchmark->Args({dim, n_heads, n_heads_kv, n_kv, n_batch});
                    }
                }
            }
        }
    }
}

} // namespace

static void BM_MulMatF16(benchmark::State & state) {
    const int64_t m = state.range(0);
    const int64_t k = state.range(1);
    const int64_t n = state.range(2);
    auto bench = build_mul_mat_f16(m, k, n);
    run_benchmark(state, bench);
}
BENCHMARK(BM_MulMatF16)->Apply(apply_mul_mat_args);

static void BM_MulMatQ4_0(benchmark::State & state) {
    const int64_t m = state.range(0);
    const int64_t k = state.range(1);
    const int64_t n = state.range(2);
    auto bench = build_mul_mat_q4_0(m, k, n);
    run_benchmark(state, bench);
}
BENCHMARK(BM_MulMatQ4_0)->Apply(apply_mul_mat_args);

static void BM_RmsNorm(benchmark::State & state) {
    auto bench = build_rms_norm(state.range(0));
    run_benchmark(state, bench);
}
BENCHMARK(BM_RmsNorm)->RangeMultiplier(2)->Range(512, 8192);

static void BM_Rope(benchmark::State & state) {
    auto bench = build_rope(state.range(0), state.range(1), state.range(2));
    run_benchmark(state, bench);
}
BENCHMARK(BM_Rope)->Apply(apply_rope_args);

static void BM_Softmax(benchmark::State & state) {
    auto bench = build_softmax(state.range(0), state.range(1));
    run_benchmark(state, bench);
}
BENCHMARK(BM_Softmax)->RangeMultiplier(2)->Ranges({{64, 1024}, {8, 32}});

static void BM_FlashAttn(benchmark::State & state) {
    auto bench = build_flash_attn(state.range(0), state.range(1), state.range(2), state.range(3), state.range(4));
    run_benchmark(state, bench);
}
BENCHMARK(BM_FlashAttn)->Apply(apply_flash_attn_args);

int main(int argc, char ** argv) {
    int ggml_repetitions = -1;
    bool ggml_aggregates_only = false;
    std::string ggml_backend = g_backend_name;

    std::vector<std::string> args;
    args.reserve(static_cast<size_t>(argc));
    for (int i = 0; i < argc; ++i) {
        std::string arg(argv[i]);
        if (has_prefix(arg, "--ggml_repetitions=")) {
            ggml_repetitions = parse_flag_int(arg.substr(std::string("--ggml_repetitions=").size()), ggml_repetitions);
            continue;
        }
        if (arg == "--ggml_repetitions" && i + 1 < argc) {
            ggml_repetitions = parse_flag_int(argv[++i], ggml_repetitions);
            continue;
        }
        if (arg == "--ggml_report_aggregates_only") {
            ggml_aggregates_only = true;
            continue;
        }
        if (has_prefix(arg, "--ggml_backend=")) {
            if (parse_flag_string(arg.substr(std::string("--ggml_backend=").size()), ggml_backend)) {
                g_backend_name = ggml_backend;
            }
            continue;
        }
        if (arg == "--ggml_backend" && i + 1 < argc) {
            if (parse_flag_string(argv[++i], ggml_backend)) {
                g_backend_name = ggml_backend;
            }
            continue;
        }
        args.emplace_back(std::move(arg));
    }

    if (ggml_repetitions > 0 && !has_flag(args, "--benchmark_repetitions")) {
        args.emplace_back("--benchmark_repetitions=" + std::to_string(ggml_repetitions));
    }
    if (ggml_aggregates_only && !has_flag(args, "--benchmark_report_aggregates_only")) {
        args.emplace_back("--benchmark_report_aggregates_only=true");
        args.emplace_back("--benchmark_display_aggregates_only=true");
    }

    std::vector<char *> argv_out;
    argv_out.reserve(args.size());
    for (std::string & arg : args) {
        argv_out.push_back(arg.data());
    }
    int argc_out = static_cast<int>(argv_out.size());

    benchmark::Initialize(&argc_out, argv_out.data());
    if (benchmark::ReportUnrecognizedArguments(argc_out, argv_out.data())) {
        return 1;
    }

    if (ggml_repetitions > 0) {
        benchmark::AddCustomContext("ggml_repetitions", std::to_string(ggml_repetitions));
    }
    if (ggml_aggregates_only) {
        benchmark::AddCustomContext("ggml_report_aggregates_only", "true");
    }
    benchmark::AddCustomContext("ggml_backend", g_backend_name);

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
