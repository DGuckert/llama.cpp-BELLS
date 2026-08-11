// Proves the core BELLS graph mechanism: running mul_mat_id over a small VRAM-resident
// expert cache, with routing ids rewritten through a slot table, gives exactly the same
// result as running it over the full expert tensor.
//
//   reference: mul_mat_id(experts[n_embd, n_ff, n_expert], x, ids)
//   bells:     mul_mat_id(cache  [n_embd, n_ff, n_slot  ], x, get_rows(slot_of, ids))
//
// The remap needs no new op: ggml_get_rows preserves I32, so slot_of[expert] -> slot is a
// plain gather that runs inside the graph, where the routing ids are actually produced.
//
// Invariant under test: every selected expert is resident. The runtime is responsible for
// guaranteeing that before the expert matmul executes; a miss would index a stale slot.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "llama-bells.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <vector>

static const int N_EMBD   = 64;
static const int N_FF     = 32;
static const int N_EXPERT = 16;
static const int N_SLOT   = 6;   // VRAM cache holds only this many experts
static const int N_USED   = 2;
static const int N_TOKEN  = 5;

struct build_result {
    ggml_tensor * out;
    ggml_tensor * remapped;
};

static build_result build(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x,
                          ggml_tensor * ids, ggml_tensor * slot_of) {
    build_result r = { nullptr, nullptr };

    ggml_tensor * use = ids;

    if (slot_of) {
        // [K, n_tokens] -> [K*n_tokens, 1] so get_rows broadcasting lines up without a repeat
        ggml_tensor * flat = ggml_reshape_2d(ctx, ids, ids->ne[0]*ids->ne[1], 1);
        ggml_tensor * tbl  = ggml_reshape_3d(ctx, slot_of, 1, slot_of->ne[0], 1);

        ggml_tensor * slots = ggml_get_rows(ctx, tbl, flat);           // [1, K*n_tokens] I32
        slots = ggml_reshape_2d(ctx, slots, ids->ne[0], ids->ne[1]);   // [K, n_tokens]

        r.remapped = slots;
        use = slots;
    }

    r.out = ggml_mul_mat_id(ctx, w, x, use);

    return r;
}

// Does a slot_of write issued from the scheduler's eval callback, part-way through the
// graph, get observed by a get_rows further down that same graph?
//
// If yes, the residency correction step is nearly free: hook ffn_moe_topk-<il>, patch any
// missing experts into the cache, rewrite slot_of, and let execution continue. If no, the
// graph must be split per MoE layer and we pay a sync every layer of every token.
struct midgraph_data {
    ggml_tensor *        slot_of = nullptr;
    std::vector<int32_t> updated;
    bool                 fired   = false;
};

static bool midgraph_cb(ggml_tensor * t, bool ask, void * user_data) {
    auto * d = (midgraph_data *) user_data;

    const bool match = strcmp(t->name, "bells_marker") == 0;

    if (ask) {
        return match;
    }

    if (match) {
        ggml_backend_tensor_set(d->slot_of, d->updated.data(), 0, ggml_nbytes(d->slot_of));
        d->fired = true;
    }

    return true;
}

static bool test_midgraph(ggml_backend_t backend) {
    printf("\n-- mid-graph slot_of update --\n");

    const int K = 2;
    const int T = 4;

    std::vector<int32_t> ids(K*T);
    for (int i = 0; i < K*T; ++i) {
        ids[i] = i % N_EXPERT;
    }

    std::vector<int32_t> initial(N_EXPERT);
    std::vector<int32_t> updated(N_EXPERT);
    for (int e = 0; e < N_EXPERT; ++e) {
        initial[e] = e;
        updated[e] = N_EXPERT - 1 - e;
    }

    ggml_init_params ip = { ggml_tensor_overhead()*16, nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * id_t = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, K, T);
    ggml_tensor * slot = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N_EXPERT);
    ggml_set_input(id_t);
    ggml_set_input(slot);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        printf("FAIL: input allocation failed\n");
        return false;
    }

    ggml_backend_tensor_set(id_t, ids.data(),     0, ggml_nbytes(id_t));
    ggml_backend_tensor_set(slot, initial.data(), 0, ggml_nbytes(slot));

    ggml_init_params gp = { ggml_tensor_overhead()*32 + ggml_graph_overhead(), nullptr, true };
    ggml_context * gctx = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph(gctx);

    // stands in for ffn_moe_topk: a real computed node the callback can fire on, and an
    // ancestor of the gather so ordering is guaranteed by the dependency
    ggml_tensor * marker = ggml_cont(gctx, id_t);
    ggml_set_name(marker, "bells_marker");

    ggml_tensor * flat  = ggml_reshape_2d(gctx, marker, K*T, 1);
    ggml_tensor * tbl   = ggml_reshape_3d(gctx, slot, 1, N_EXPERT, 1);
    ggml_tensor * slots = ggml_get_rows(gctx, tbl, flat);

    ggml_build_forward_expand(gf, slots);

    midgraph_data d;
    d.slot_of = slot;
    d.updated = updated;

    // the scheduler requires a CPU backend last in the list
    const bool is_cpu = ggml_backend_dev_type(ggml_backend_get_device(backend)) ==
                        GGML_BACKEND_DEVICE_TYPE_CPU;

    ggml_backend_t cpu = is_cpu ? nullptr : ggml_backend_cpu_init();

    ggml_backend_t backends[2] = { backend, cpu };
    const int n_backends = is_cpu ? 1 : 2;

    ggml_backend_sched_t sched =
        ggml_backend_sched_new(backends, nullptr, n_backends, GGML_DEFAULT_GRAPH_SIZE, false, false);
    ggml_backend_sched_set_eval_callback(sched, midgraph_cb, &d);

    const bool ok_compute = ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS;

    std::vector<int32_t> out(ggml_nelements(slots));
    if (ok_compute) {
        ggml_backend_tensor_get(slots, out.data(), 0, ggml_nbytes(slots));
    }

    int n_updated = 0;
    int n_initial = 0;
    for (size_t i = 0; i < out.size(); ++i) {
        if (out[i] == updated[ids[i]]) n_updated++;
        if (out[i] == initial[ids[i]]) n_initial++;
    }

    const bool observed = ok_compute && d.fired && n_updated == (int) out.size();

    printf("callback fired: %s\n", d.fired ? "yes" : "no");
    printf("gather saw updated table: %d/%zu, stale table: %d/%zu\n",
           n_updated, out.size(), n_initial, out.size());
    printf("%s\n", observed
        ? "PASS: mid-graph update is visible, correction can reuse the eval callback"
        : "INFO: mid-graph update NOT visible, graph must be split per MoE layer");

    ggml_backend_sched_free(sched);
    if (cpu) {
        ggml_backend_free(cpu);
    }
    ggml_free(gctx);
    ggml_free(ctx);
    ggml_backend_buffer_free(buf);

    return observed;
}

static void test_bandwidth(ggml_backend_t backend, ggml_backend_dev_t dev) {
    printf("\n-- host to device bandwidth --\n");

    const size_t SIZE = 256ull*1024*1024;
    const int    REPS = 8;

    ggml_init_params ip = { ggml_tensor_overhead()*4, nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * dst = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, SIZE/sizeof(float));
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        printf("SKIP: could not allocate %zu MiB on device\n", SIZE/1024/1024);
        ggml_free(ctx);
        return;
    }

    auto bench = [&](const char * label, void * src) {
        ggml_backend_synchronize(backend);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < REPS; ++i) {
            ggml_backend_tensor_set(dst, src, 0, SIZE);
        }
        ggml_backend_synchronize(backend);
        auto t1 = std::chrono::high_resolution_clock::now();

        const double secs = std::chrono::duration<double>(t1 - t0).count();
        const double gbs  = (double) SIZE*REPS/secs/1e9;
        printf("%-10s %6.2f GB/s\n", label, gbs);
        return gbs;
    };

    std::vector<char> pageable(SIZE, 1);
    const double bw_pageable = bench("pageable", pageable.data());

    double bw_pinned = 0.0;
    ggml_backend_buffer_type_t hbt = ggml_backend_dev_host_buffer_type(dev);
    ggml_backend_buffer_t hbuf = nullptr;

    if (hbt) {
        hbuf = ggml_backend_buft_alloc_buffer(hbt, SIZE);
        if (hbuf) {
            void * pinned = ggml_backend_buffer_get_base(hbuf);
            memset(pinned, 1, SIZE);
            bw_pinned = bench("pinned", pinned);
        }
    }
    if (bw_pinned == 0.0) {
        printf("pinned     unavailable\n");
    }

    const double bw = std::max(bw_pageable, bw_pinned);

    // what that buys per token, at the miss rates we measured
    struct scenario { const char * name; int layers; double mb_expert; double miss; };
    const scenario cases[] = {
        { "Qwen3-30B-A3B @1x", 48, 2.65, 0.441 },
        { "Qwen3-30B-A3B @2x", 48, 2.65, 0.257 },
        { "Qwen3-30B-A3B @4x", 48, 2.65, 0.128 },
    };

    printf("\n%-20s %10s %10s %10s\n", "scenario", "MB/token", "ms/token", "max t/s");
    for (const auto & c : cases) {
        const double mb = c.layers*8*c.miss*c.mb_expert;
        const double ms = mb/1024.0/bw*1000.0;
        printf("%-20s %10.1f %10.1f %10.1f\n", c.name, mb, ms, 1000.0/ms);
    }
    printf("(transfer only, assuming perfect overlap with compute)\n");

    if (hbuf) {
        ggml_backend_buffer_free(hbuf);
    }
    ggml_free(ctx);
    ggml_backend_buffer_free(buf);
}

// Hammers the residency manager against randomised routing and asserts the two things the
// graph cannot survive without: an injective expert->slot mapping, and full residency of
// every requested expert after ensure().
static bool test_cache() {
    printf("\n-- cache residency invariants --\n");

    const uint32_t n_layer  = 4;
    const uint32_t n_expert = 128;
    const uint32_t n_used   = 8;

    std::mt19937 rng(99);

    bool ok = true;
    uint64_t checked = 0;

    for (uint32_t n_slot : { 8u, 16u, 32u, 64u }) {
        bells_cache cache;
        cache.init(n_layer, n_expert, n_slot);

        bells_predictor none;

        std::vector<bells_copy> copies;
        std::vector<int32_t> want(n_used);
        std::vector<int32_t> pool(n_expert);
        for (uint32_t e = 0; e < n_expert; ++e) {
            pool[e] = e;
        }

        for (int step = 0; step < 4000 && ok; ++step) {
            const uint32_t il = step % n_layer;

            std::shuffle(pool.begin(), pool.end(), rng);
            for (uint32_t k = 0; k < n_used; ++k) {
                want[k] = pool[k];
            }

            copies.clear();
            if (!cache.ensure(il, want.data(), n_used, copies)) {
                printf("FAIL: ensure() failed with n_slot=%u >= n_used=%u\n", n_slot, n_used);
                ok = false;
                break;
            }

            const std::vector<int32_t> & tbl = cache.slot_table(il);

            // every requested expert must now be resident
            for (uint32_t k = 0; k < n_used; ++k) {
                if (tbl[want[k]] < 0 || (uint32_t) tbl[want[k]] >= n_slot) {
                    printf("FAIL: expert %d not resident after ensure (slot %d)\n",
                           want[k], tbl[want[k]]);
                    ok = false;
                    break;
                }
            }

            // and no two experts may share a slot
            std::vector<int32_t> owner(n_slot, -1);
            for (uint32_t e = 0; e < n_expert; ++e) {
                const int32_t s = tbl[e];
                if (s < 0) {
                    continue;
                }
                if (owner[s] >= 0) {
                    printf("FAIL: slot %d claimed by experts %d and %u\n", s, owner[s], e);
                    ok = false;
                    break;
                }
                owner[s] = e;
            }

            checked++;
        }

        if (ok) {
            const double miss = 100.0*cache.n_miss()/std::max<uint64_t>(1, cache.n_hit() + cache.n_miss());
            printf("n_slot %-3u ok, miss rate %.1f%% (random routing, so ~worst case)\n", n_slot, miss);
        }
    }

    // a request larger than the cache must be refused, not silently corrupt the mapping
    {
        bells_cache small;
        small.init(1, 32, 4);

        const int32_t want[6] = { 1, 2, 3, 4, 5, 6 };
        std::vector<bells_copy> copies;

        if (small.ensure(0, want, 6, copies)) {
            printf("FAIL: ensure() accepted 6 experts into 4 slots\n");
            ok = false;
        } else {
            printf("oversized request correctly refused\n");
        }
    }

    printf("%s (%llu states checked)\n", ok ? "PASS" : "FAIL", (unsigned long long) checked);

    return ok;
}

// The whole runtime loop, in miniature: routing arrives, the residency manager decides what
// must move, real weights are copied into a real VRAM cache, the slot table is uploaded, and
// mul_mat_id runs over the cache. Compared every step against the full expert stack on the
// same backend, so any difference is a real bug and not backend float drift.
static bool test_runtime(ggml_backend_t backend) {
    printf("\n-- end to end cache runtime --\n");

    const int64_t n_embd   = 64;
    const int64_t n_ff     = 32;
    const int64_t n_expert = 64;
    const uint32_t n_slot  = 16;
    const int64_t n_used   = 4;
    const int64_t n_tok    = 2;
    const int      steps   = 60;

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> full((size_t) n_embd*n_ff*n_expert);
    for (auto & v : full) {
        v = dist(rng);
    }

    // source expert stack, host resident, exactly as it would be when offloaded
    ggml_init_params hp = { ggml_tensor_overhead()*8, nullptr, true };
    ggml_context * hctx = ggml_init(hp);
    ggml_tensor * src = ggml_new_tensor_3d(hctx, GGML_TYPE_F32, n_embd, n_ff, n_expert);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_buffer_t hbuf = ggml_backend_alloc_ctx_tensors(hctx, cpu);
    ggml_backend_tensor_set(src, full.data(), 0, ggml_nbytes(src));

    // reference copy on the compute backend, plus the inputs
    ggml_init_params dp = { ggml_tensor_overhead()*8, nullptr, true };
    ggml_context * dctx = ggml_init(dp);
    ggml_tensor * ref  = ggml_new_tensor_3d(dctx, GGML_TYPE_F32, n_embd, n_ff, n_expert);
    ggml_tensor * x    = ggml_new_tensor_3d(dctx, GGML_TYPE_F32, n_embd, 1, n_tok);
    ggml_tensor * id_t = ggml_new_tensor_2d(dctx, GGML_TYPE_I32, n_used, n_tok);
    ggml_set_input(x);
    ggml_set_input(id_t);
    ggml_backend_buffer_t dbuf = ggml_backend_alloc_ctx_tensors(dctx, backend);
    ggml_backend_tensor_set(ref, full.data(), 0, ggml_nbytes(ref));

    std::vector<float> x_data((size_t) n_embd*n_tok);
    for (auto & v : x_data) {
        v = dist(rng);
    }
    ggml_backend_tensor_set(x, x_data.data(), 0, ggml_nbytes(x));

    bells_tensors tensors;
    std::vector<bells_tensors::layer_src> srcs(1);
    srcs[0].gate = src;
    srcs[0].il   = 0;

    if (!tensors.init(ggml_backend_get_default_buffer_type(backend), srcs, n_slot)) {
        printf("FAIL: cache allocation failed\n");
        return false;
    }

    printf("cache: %u slots of %lld experts, %.1f KiB VRAM, %zu B per expert\n",
           n_slot, (long long) n_expert, tensors.vram_bytes()/1024.0, tensors.bytes_per_expert());

    bells_cache cache;
    cache.init(1, (uint32_t) n_expert, n_slot);

    std::vector<int32_t> ids((size_t) n_used*n_tok);
    std::vector<int32_t> pool(n_expert);
    for (int64_t e = 0; e < n_expert; ++e) {
        pool[e] = (int32_t) e;
    }

    std::vector<bells_copy> copies;
    uint64_t n_copied = 0;
    double   max_diff = 0.0;
    bool     ok       = true;

    for (int step = 0; step < steps && ok; ++step) {
        for (int64_t t = 0; t < n_tok; ++t) {
            std::shuffle(pool.begin(), pool.end(), rng);
            for (int64_t k = 0; k < n_used; ++k) {
                ids[t*n_used + k] = pool[k];
            }
        }

        copies.clear();
        if (!cache.ensure(0, ids.data(), ids.size(), copies)) {
            printf("FAIL: ensure() refused %zu experts into %u slots\n", ids.size(), n_slot);
            ok = false;
            break;
        }

        for (const auto & c : copies) {
            tensors.copy_expert(0, c.expert, c.slot);
        }
        n_copied += copies.size();

        tensors.upload_slots(0, cache.slot_table(0));
        ggml_backend_tensor_set(id_t, ids.data(), 0, ggml_nbytes(id_t));

        std::vector<float> out[2];

        for (int pass = 0; pass < 2; ++pass) {
            ggml_init_params gp = { ggml_tensor_overhead()*32 + ggml_graph_overhead(), nullptr, true };
            ggml_context * gctx = ggml_init(gp);
            ggml_cgraph * gf = ggml_new_graph(gctx);

            build_result r = pass == 0
                ? build(gctx, ref, x, id_t, nullptr)
                : build(gctx, tensors.gate(0), x, id_t, tensors.slots(0));

            ggml_build_forward_expand(gf, r.out);

            ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            ggml_gallocr_alloc_graph(alloc, gf);

            if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
                printf("FAIL: compute failed at step %d pass %d\n", step, pass);
                ok = false;
            }

            out[pass].resize(ggml_nelements(r.out));
            ggml_backend_tensor_get(r.out, out[pass].data(), 0, ggml_nbytes(r.out));

            ggml_gallocr_free(alloc);
            ggml_free(gctx);
        }

        for (size_t i = 0; i < out[0].size() && ok; ++i) {
            max_diff = std::max(max_diff, (double) std::fabs(out[0][i] - out[1][i]));
        }
    }

    ok = ok && max_diff == 0.0;

    printf("%d steps, %llu experts copied, max abs diff %g\n",
           steps, (unsigned long long) n_copied, max_diff);
    printf("cache hit rate %.1f%% (random routing)\n",
           100.0*cache.n_hit()/std::max<uint64_t>(1, cache.n_hit() + cache.n_miss()));
    printf("%s\n", ok ? "PASS: runtime loop matches the full expert stack exactly"
                      : "FAIL: runtime loop diverges");

    tensors.free();
    ggml_free(dctx);
    ggml_free(hctx);
    ggml_backend_buffer_free(dbuf);
    ggml_backend_buffer_free(hbuf);
    ggml_backend_free(cpu);

    return ok;
}

static bool test_predictor(const char * path) {
    printf("\n-- predictor table --\n");

    bells_predictor p;
    if (!p.load(path)) {
        printf("FAIL: could not load %s\n", path);
        return false;
    }

    // a known token and an id far outside the table must both return usable rankings
    const int32_t * a = p.predict(0, 0);
    const int32_t * b = p.predict(1 << 30, 0);

    if (!a || !b) {
        printf("FAIL: null prediction\n");
        return false;
    }

    printf("n_take %u, layer 0 top-8 for token 0:", p.n_take());
    for (uint32_t i = 0; i < std::min(8u, p.n_take()); ++i) {
        printf(" %d", a[i]);
    }
    printf("\nunseen token falls back to prior:");
    for (uint32_t i = 0; i < std::min(8u, p.n_take()); ++i) {
        printf(" %d", b[i]);
    }
    printf("\nPASS\n");

    return true;
}

// Replays a real routing trace through the shipped C++ cache and predictor. The Python
// harness measured the policy; this measures the implementation, so a gap between them
// means the C++ does not do what was analysed.
static bool test_replay(const char * trace_path, const char * table_path) {
    printf("\n-- trace replay through the real cache --\n");

    std::ifstream f(trace_path, std::ios::binary);
    if (!f) {
        printf("SKIP: cannot open %s\n", trace_path);
        return true;
    }

    char magic[8] = { 0 };
    f.read(magic, 8);
    if (memcmp(magic, "BELLSTR1", 8) != 0) {
        printf("SKIP: %s is not a BELLS trace\n", trace_path);
        return true;
    }

    uint32_t version = 0, n_layer = 0, n_expert = 0, n_used = 0, n_vocab = 0;
    uint64_t n_rec = 0;

    f.read((char *) &version,  4);
    f.read((char *) &n_layer,  4);
    f.read((char *) &n_expert, 4);
    f.read((char *) &n_used,   4);
    f.read((char *) &n_vocab,  4);
    f.read((char *) &n_rec,    8);

    std::vector<uint32_t> layer_ids(n_layer);
    f.read((char *) layer_ids.data(), (std::streamsize) n_layer*4);

    bells_predictor pred;
    const bool have_pred = pred.load(table_path);

    printf("%llu records, %u layers, %u experts, %u used, predictor %s\n",
           (unsigned long long) n_rec, n_layer, n_expert, n_used,
           have_pred ? "loaded" : "MISSING");

    for (uint32_t mult : { 1u, 2u, 4u }) {
        const uint32_t n_slot = std::min(mult*n_used, n_expert);

        // header is magic(8) + 5 u32 + u64 = 36, then the layer id list
        f.clear();
        f.seekg(36 + (std::streamoff) n_layer*4);

        bells_cache cache;
        cache.init(n_layer, n_expert, n_slot);

        std::vector<bells_copy> copies;
        std::vector<int32_t>    want(n_used);
        std::vector<uint16_t>   row((size_t) n_layer*n_used);

        bool ok = true;
        uint64_t n_prefetched = 0;
        uint64_t n_lookup     = 0;
        uint64_t n_in_table   = 0;
        uint64_t n_recall     = 0;
        uint64_t n_want       = 0;

        for (uint64_t r = 0; r < n_rec && ok; ++r) {
            uint32_t token = 0, pos = 0;
            uint8_t  gen = 0, pad[3];

            f.read((char *) &token, 4);
            f.read((char *) &pos,   4);
            f.read((char *) &gen,   1);
            f.read((char *) pad,    3);
            f.read((char *) row.data(), (std::streamsize) row.size()*2);

            if (!f) {
                break;
            }

            for (uint32_t l = 0; l < n_layer; ++l) {
                // prefetch on the token id, which is known before any layer runs
                if (have_pred) {
                    const int32_t * p = pred.predict((int32_t) token, l);
                    if (p) {
                        const uint32_t take = std::min(pred.n_take(), n_slot);
                        copies.clear();
                        cache.prefetch(layer_ids[l], p, take, copies);
                        n_prefetched += copies.size();
                    }
                }

                for (uint32_t k = 0; k < n_used; ++k) {
                    want[k] = row[(size_t) l*n_used + k];
                }

                // direct recall of the prediction, independent of any cache behaviour
                if (have_pred) {
                    const int32_t * p = pred.predict((int32_t) token, l);
                    const uint32_t take = std::min(pred.n_take(), n_slot);
                    n_lookup++;
                    n_in_table += pred.in_table((int32_t) token) ? 1 : 0;
                    for (uint32_t k = 0; k < n_used; ++k) {
                        for (uint32_t j = 0; j < take; ++j) {
                            if (p[j] == want[k]) {
                                n_recall++;
                                break;
                            }
                        }
                    }
                    n_want += n_used;
                }

                copies.clear();
                if (!cache.ensure(layer_ids[l], want.data(), n_used, copies)) {
                    printf("FAIL: ensure refused at slot budget %u\n", n_slot);
                    ok = false;
                    break;
                }
            }
        }

        if (!ok) {
            return false;
        }

        const uint64_t tot = cache.n_hit() + cache.n_miss();
        printf("n_slot %-3u (%2ux used)  hit %5.1f%%  recall %5.1f%%  in-table %5.1f%%  prefetched %llu\n",
               n_slot, mult, 100.0*cache.n_hit()/std::max<uint64_t>(1, tot),
               100.0*n_recall/std::max<uint64_t>(1, n_want),
               100.0*n_in_table/std::max<uint64_t>(1, n_lookup),
               (unsigned long long) n_prefetched);
    }

    printf("PASS (in-sample: table was built from this trace, so held-out is lower)\n");

    return true;
}

int main(int argc, char ** argv) {
    ggml_backend_t     backend = nullptr;
    ggml_backend_dev_t device  = nullptr;

    for (size_t i = 0; i < ggml_backend_dev_count() && !backend; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            backend = ggml_backend_dev_init(dev, nullptr);
            device  = dev;
        }
    }

    if (!backend) {
        backend = ggml_backend_cpu_init();
        device  = ggml_backend_get_device(backend);
    }

    printf("backend: %s\n", ggml_backend_name(backend));

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // pick which experts live in the cache, and where
    std::vector<int32_t> resident;
    for (int e = 0; e < N_EXPERT && (int) resident.size() < N_SLOT; e += 2) {
        resident.push_back(e);
    }

    std::vector<int32_t> slot_of(N_EXPERT, -1);
    for (size_t s = 0; s < resident.size(); ++s) {
        slot_of[resident[s]] = (int32_t) s;
    }

    // route only to resident experts, which is the invariant the runtime enforces.
    // experts within a token must be distinct: top-k never repeats one, and the CUDA
    // mul_mat_id kernel takes only the first match per token, so duplicates underfill it.
    std::vector<int32_t> ids(N_USED*N_TOKEN);
    for (int t = 0; t < N_TOKEN; ++t) {
        std::vector<int32_t> pick = resident;
        std::shuffle(pick.begin(), pick.end(), rng);
        for (int k = 0; k < N_USED; ++k) {
            ids[t*N_USED + k] = pick[k];
        }
    }

    std::vector<float> w_data((size_t) N_EMBD*N_FF*N_EXPERT);
    for (auto & v : w_data) {
        v = dist(rng);
    }

    // b is [n_embd, 1, n_tokens] and broadcasts across the expert slots, matching how
    // build_moe_ffn calls mul_mat_id
    std::vector<float> x_data((size_t) N_EMBD*N_TOKEN);
    for (auto & v : x_data) {
        v = dist(rng);
    }

    // the cache is a gather of the resident experts out of the full tensor
    std::vector<float> c_data((size_t) N_EMBD*N_FF*N_SLOT);
    for (size_t s = 0; s < resident.size(); ++s) {
        memcpy(&c_data[(size_t) s*N_EMBD*N_FF],
               &w_data[(size_t) resident[s]*N_EMBD*N_FF],
               (size_t) N_EMBD*N_FF*sizeof(float));
    }

    ggml_init_params ip = { ggml_tensor_overhead()*64 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * w    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, N_EMBD, N_FF, N_EXPERT);
    ggml_tensor * c    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, N_EMBD, N_FF, N_SLOT);
    ggml_tensor * x    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, N_EMBD, 1, N_TOKEN);
    ggml_tensor * id_t = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N_USED, N_TOKEN);
    ggml_tensor * slot = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N_EXPERT);

    ggml_set_input(w); ggml_set_input(c); ggml_set_input(x);
    ggml_set_input(id_t); ggml_set_input(slot);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        printf("FAIL: could not allocate tensors\n");
        return 1;
    }

    ggml_backend_tensor_set(w,    w_data.data(),  0, ggml_nbytes(w));
    ggml_backend_tensor_set(c,    c_data.data(),  0, ggml_nbytes(c));
    ggml_backend_tensor_set(x,    x_data.data(),  0, ggml_nbytes(x));
    ggml_backend_tensor_set(id_t, ids.data(),     0, ggml_nbytes(id_t));
    ggml_backend_tensor_set(slot, slot_of.data(), 0, ggml_nbytes(slot));

    std::vector<float> ref;
    std::vector<float> got;

    for (int pass = 0; pass < 2; ++pass) {
        ggml_init_params gp = { ggml_tensor_overhead()*64 + ggml_graph_overhead(), nullptr, true };
        ggml_context * gctx = ggml_init(gp);

        ggml_cgraph * gf = ggml_new_graph(gctx);

        build_result r = pass == 0
            ? build(gctx, w, x, id_t, nullptr)
            : build(gctx, c, x, id_t, slot);

        ggml_build_forward_expand(gf, r.out);

        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(alloc, gf)) {
            printf("FAIL: graph allocation failed on pass %d\n", pass);
            return 1;
        }

        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            printf("FAIL: compute failed on pass %d\n", pass);
            return 1;
        }

        std::vector<float> & dst = pass == 0 ? ref : got;
        dst.resize(ggml_nelements(r.out));
        ggml_backend_tensor_get(r.out, dst.data(), 0, ggml_nbytes(r.out));

        if (pass == 1 && r.remapped) {
            std::vector<int32_t> rm(ggml_nelements(r.remapped));
            ggml_backend_tensor_get(r.remapped, rm.data(), 0, ggml_nbytes(r.remapped));

            bool ok = true;
            for (size_t i = 0; i < rm.size(); ++i) {
                if (rm[i] != slot_of[ids[i]]) {
                    printf("FAIL: remap[%zu] = %d, expected %d\n", i, rm[i], slot_of[ids[i]]);
                    ok = false;
                    break;
                }
            }
            if (ok) {
                printf("remap: ok (%zu ids rewritten through the slot table in-graph)\n", rm.size());
            }
        }

        ggml_gallocr_free(alloc);
        ggml_free(gctx);
    }

    double max_diff = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        max_diff = std::max(max_diff, (double) std::fabs(ref[i] - got[i]));
    }

    printf("experts %d, cache slots %d (%.0f%% resident), %zu outputs\n",
           N_EXPERT, N_SLOT, 100.0*N_SLOT/N_EXPERT, ref.size());
    printf("max abs diff vs full-tensor reference: %g\n", max_diff);

    const bool pass = max_diff == 0.0;
    printf("%s\n", pass ? "PASS: cache path is bit-identical" : "FAIL: outputs diverge");

    ggml_free(ctx);
    ggml_backend_buffer_free(buf);

    test_midgraph(backend);
    const bool cache_ok   = test_cache();
    const bool runtime_ok = test_runtime(backend);

    bool pred_ok = true;
    if (argc > 1) {
        pred_ok = test_predictor(argv[1]);
    }
    if (argc > 2) {
        pred_ok = test_replay(argv[2], argv[1]) && pred_ok;
    }

    test_bandwidth(backend, device);

    ggml_backend_free(backend);

    return (pass && cache_ok && runtime_ok && pred_ok) ? 0 : 1;
}
