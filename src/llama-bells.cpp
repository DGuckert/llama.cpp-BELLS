#include "llama-bells.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>

void bells_cache::init(uint32_t n_layer, uint32_t n_expert, uint32_t n_slot) {
    n_layer_  = n_layer;
    n_expert_ = n_expert;
    n_slot_   = std::min(n_slot, n_expert);

    layers_.assign(n_layer, layer());

    for (auto & l : layers_) {
        l.expert_slot.assign(n_expert_, -1);
        l.slot_expert.assign(n_slot_,   -1);
        l.last_used.assign(n_slot_,      0);
    }

    clock_  = 0;
    n_hit_  = 0;
    n_miss_ = 0;
}

void bells_cache::reset() {
    for (auto & l : layers_) {
        std::fill(l.expert_slot.begin(), l.expert_slot.end(), -1);
        std::fill(l.slot_expert.begin(), l.slot_expert.end(), -1);
        std::fill(l.last_used.begin(),   l.last_used.end(),    0);
    }

    clock_ = 0;
}

int32_t bells_cache::victim(layer & l, const int32_t * keep, size_t n_keep) const {
    int32_t best     = -1;
    int64_t best_age = 0;

    for (uint32_t s = 0; s < n_slot_; ++s) {
        const int32_t held = l.slot_expert[s];

        if (held < 0) {
            return (int32_t) s;
        }

        bool pinned = false;
        for (size_t k = 0; k < n_keep; ++k) {
            if (keep[k] == held) {
                pinned = true;
                break;
            }
        }
        if (pinned) {
            continue;
        }

        if (best < 0 || l.last_used[s] < best_age) {
            best     = (int32_t) s;
            best_age = l.last_used[s];
        }
    }

    return best;
}

bool bells_cache::ensure(uint32_t il, const int32_t * experts, size_t n, std::vector<bells_copy> & out) {
    if (!enabled() || il >= n_layer_) {
        return true;
    }

    layer & l = layers_[il];

    clock_++;

    for (size_t i = 0; i < n; ++i) {
        const int32_t e = experts[i];
        if (e < 0 || (uint32_t) e >= n_expert_) {
            continue;
        }

        if (l.expert_slot[e] >= 0) {
            n_hit_++;
            l.last_used[l.expert_slot[e]] = clock_;
            continue;
        }

        n_miss_++;

        const int32_t s = victim(l, experts, n);
        if (s < 0) {
            // more distinct experts requested than the cache can hold
            return false;
        }

        const int32_t evicted = l.slot_expert[s];
        if (evicted >= 0) {
            l.expert_slot[evicted] = -1;
        }

        l.slot_expert[s] = e;
        l.expert_slot[e] = s;
        l.last_used[s]   = clock_;

        out.push_back({ e, s });
    }

    return true;
}

//
// cache tensors
//

const bells_tensors::entry & bells_tensors::empty() {
    static const entry e;
    return e;
}

static ggml_tensor * bells_make_slice(ggml_context * ctx, ggml_tensor * src, uint32_t n_slot) {
    if (!src) {
        return nullptr;
    }

    // same layout and quant type as the full expert stack, just fewer experts
    return ggml_new_tensor_3d(ctx, src->type, src->ne[0], src->ne[1], n_slot);
}

bool bells_tensors::init(ggml_backend_buffer_type_t buft, const std::vector<layer_src> & srcs,
                         uint32_t n_slot, ggml_backend_t backend) {
    free();

    backend_ = backend;

    if (srcs.empty() || n_slot == 0) {
        return false;
    }

    int32_t max_il = -1;
    for (const auto & s : srcs) {
        max_il = std::max(max_il, s.il);
    }

    index_.assign(max_il + 1, -1);
    entries_.clear();
    entries_.reserve(srcs.size());

    n_slot_ = n_slot;

    // one spare slot beyond the cache, kept zeroed, so a non-resident expert can be routed
    // somewhere harmless rather than out of bounds
    const uint32_t n_alloc = n_slot + 1;

    // 5 tensors per layer at most, plus overhead
    ggml_init_params ip = { ggml_tensor_overhead()*srcs.size()*8, nullptr, true };
    ctx_ = ggml_init(ip);
    if (!ctx_) {
        return false;
    }

    for (const auto & s : srcs) {
        entry e;
        e.src = s;

        e.gate    = bells_make_slice(ctx_, s.gate,    n_alloc);
        e.up      = bells_make_slice(ctx_, s.up,      n_alloc);
        e.down    = bells_make_slice(ctx_, s.down,    n_alloc);
        e.gate_up = bells_make_slice(ctx_, s.gate_up, n_alloc);

        ggml_tensor * any = s.gate ? s.gate : (s.gate_up ? s.gate_up : s.up);
        if (!any) {
            ggml_free(ctx_);
            ctx_ = nullptr;
            return false;
        }

        e.slots = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, any->ne[2]);
        ggml_set_input(e.slots);

        index_[s.il] = (int32_t) entries_.size();
        entries_.push_back(e);
        layer_ids_.push_back(s.il);
    }

    buffer_ = ggml_backend_alloc_ctx_tensors_from_buft(ctx_, buft);
    if (!buffer_) {
        ggml_free(ctx_);
        ctx_ = nullptr;
        return false;
    }

    vram_bytes_ = ggml_backend_buffer_get_size(buffer_);

    bytes_per_expert_ = 0;
    const entry & first = entries_.front();
    for (ggml_tensor * t : { first.gate, first.up, first.down, first.gate_up }) {
        if (t) {
            bytes_per_expert_ += ggml_nbytes(t)/t->ne[2];
        }
    }

    // zero the spare slot in every layer
    {
        std::vector<char> zeros(bytes_per_expert_, 0);

        for (auto & e : entries_) {
            for (ggml_tensor * t : { e.gate, e.up, e.down, e.gate_up }) {
                if (!t) {
                    continue;
                }
                const size_t stride = ggml_nbytes(t)/t->ne[2];
                ggml_backend_tensor_set(t, zeros.data(), (size_t) n_slot*stride, stride);
            }
        }
    }

    return true;
}

void bells_tensors::free() {
    if (buffer_) {
        ggml_backend_buffer_free(buffer_);
        buffer_ = nullptr;
    }
    if (ctx_) {
        ggml_free(ctx_);
        ctx_ = nullptr;
    }

    entries_.clear();
    index_.clear();
    layer_ids_.clear();

    vram_bytes_       = 0;
    bytes_per_expert_ = 0;
}

void bells_tensors::copy_one(ggml_tensor * dst, ggml_tensor * src, int32_t expert, int32_t slot) const {
    if (!dst || !src) {
        return;
    }

    const size_t stride = ggml_nbytes(src)/src->ne[2];

    // the source expert stack is host resident, which is the whole premise of offloading
    const char * base = (const char *) src->data + (size_t) expert*stride;

    if (backend_) {
        ggml_backend_tensor_set_async(backend_, dst, base, (size_t) slot*stride, stride);
    } else {
        ggml_backend_tensor_set(dst, base, (size_t) slot*stride, stride);
    }
}

void bells_tensors::copy_expert(uint32_t il, int32_t expert, int32_t slot) {
    if (!has(il)) {
        return;
    }

    entry & e = get_mut(il);

    copy_one(e.gate,    e.src.gate,    expert, slot);
    copy_one(e.up,      e.src.up,      expert, slot);
    copy_one(e.down,    e.src.down,    expert, slot);
    copy_one(e.gate_up, e.src.gate_up, expert, slot);
}

void bells_tensors::upload_slots(uint32_t il, const std::vector<int32_t> & table) {
    if (!has(il)) {
        return;
    }

    ggml_tensor * t = get(il).slots;
    if (!t) {
        return;
    }

    const size_t n = std::min<size_t>(table.size(), (size_t) t->ne[0]);

    // NOTE: cannot go async here. table is a caller-owned vector that may be rewritten
    // before an async copy drains, so the write has to complete before returning.
    ggml_backend_tensor_set(t, table.data(), 0, n*sizeof(int32_t));
}

//
// runtime
//

bool bells_runtime::init(const bells_params & params,
                         ggml_backend_buffer_type_t buft,
                         const std::vector<bells_tensors::layer_src> & srcs,
                         uint32_t n_expert,
                         uint32_t n_expert_used,
                         ggml_backend_t backend) {
    free();

    params_ = params;

    if (!params.enabled || srcs.empty() || n_expert == 0) {
        return false;
    }

    // copy_expert reads straight out of the source tensors, so they must be allocated and host
    // resident. Checked first, before anything is sized, printed or allocated, because
    // common_fit_params probes memory by building a throwaway context over a model loaded with
    // no_alloc. That model has no tensor data, and BELLS used to build a full second cache
    // against it - briefly doubling VRAM use and pushing tight configurations into OOM, which
    // is why the docs had to tell people to pass -fit off.
    for (const auto & s : srcs) {
        for (ggml_tensor * t : { s.gate, s.up, s.down, s.gate_up }) {
            if (!t) {
                continue;
            }

            if (!t->buffer || !t->data) {
                // the memory-fit probe. Stay silent: the real context initialises BELLS
                // properly a moment later, and a warning here only looks like a failure.
                return false;
            }

            if (!ggml_backend_buffer_is_host(t->buffer)) {
                fprintf(stderr, "%s: layer %d expert tensors are not host resident, "
                                "BELLS needs them on the CPU (try --cpu-moe)\n", __func__, s.il);
                return false;
            }
        }
    }

    // bytes one expert occupies, summed over whatever projections this arch uses
    size_t per_expert = 0;
    for (ggml_tensor * t : { srcs[0].gate, srcs[0].up, srcs[0].down, srcs[0].gate_up }) {
        if (t && t->ne[2] > 0) {
            per_expert += ggml_nbytes(t)/t->ne[2];
        }
    }

    if (per_expert == 0) {
        fprintf(stderr, "%s: could not determine expert size\n", __func__);
        return false;
    }

    const size_t working_set = (size_t) srcs.size()*std::max(1u, n_expert_used)*per_expert;

    uint32_t n_slot = params.n_slot;

    if (n_slot == 0) {
        // Size from what the device actually has spare. Leave headroom: over-allocating the
        // cache starves the context and compute buffers and measurably slows decode, so more
        // cache is not monotonically better.
        size_t dev_free = 0, dev_total = 0;
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (dev) {
            ggml_backend_dev_memory(dev, &dev_free, &dev_total);
        }

        // A third of the free VRAM, floored at 1 GiB. This is a heuristic tuned on one model
        // (Qwen3-Next-80B on a 6 GB card, where 3.6 GiB was free at this point and the best
        // cache was ~2.5 GiB). Being greedy loses badly - sizing to 81 slots measured 25%
        // slower than 48 because the context and compute buffers got squeezed - so the
        // default errs small. Pass --bells-slots N to override.
        const size_t headroom = std::max<size_t>(1024ull*1024*1024, dev_free/3);
        const size_t budget   = dev_free > headroom ? dev_free - headroom : 0;

        n_slot = (uint32_t) std::min<size_t>(n_expert, budget/(per_expert*srcs.size()));

        fprintf(stderr, "%s: auto-sizing from %.1f GiB free, %.1f GiB headroom -> %u slots\n",
                __func__, dev_free/1024.0/1024.0/1024.0,
                headroom/1024.0/1024.0/1024.0, n_slot);
    }

    n_slot = std::min(n_slot, n_expert);

    // A cache below the per-token working set cannot hold even one token's experts, so every
    // access misses and BELLS is strictly worse than not caching at all.
    if (n_slot < n_expert_used) {
        fprintf(stderr, "%s: %u slots is below n_expert_used (%u), BELLS cannot help here\n",
                __func__, n_slot, n_expert_used);
        return false;
    }

    // The fit verdict.
    //
    // A high ratio does NOT mean BELLS will help - GPT-OSS-120B measured 6.1x slower at 4x -
    // so this only says whether a cache is physically worth attempting. A low ratio does
    // reliably mean it will not: 1.2x measured 0.83x against --cpu-moe.
    const double ratio = (double) n_slot/std::max(1u, n_expert_used);
    {
        const char * verdict =
            ratio >= 4.0 ? "enough to try, but measure - a high ratio does not predict a speedup" :
            ratio >= 2.0 ? "workable, measure against --cpu-moe" :
                           "marginal, likely slower than --cpu-moe";

        fprintf(stderr, "%s: working set %.2f GiB (%zu layers x %u experts x %.2f MiB), "
                        "cache holds %.1fx it -> %s\n",
                __func__, working_set/1024.0/1024.0/1024.0, srcs.size(), n_expert_used,
                per_expert/1024.0/1024.0, ratio, verdict);
    }

    // Refuse rather than warn. Below 1.5x the cache thrashes and BELLS is a measured loss -
    // 1.2x on a 6 GB card gave 0.83x - and it still costs its VRAM the whole time, which
    // squeezes the compute buffers even for the ubatches it declines to serve. Printing
    // "poor fit, expect a slowdown" and enabling anyway just meant users ate the slowdown
    // without reading the line.
    //
    // Auto-sizing lands here mostly because the KV cache got the VRAM first, so say that: the
    // actionable fix is a shorter context, not a different flag. An explicit --bells-slots is
    // taken as deliberate and honoured, since research needs to be able to measure bad configs.
    if (ratio < 1.5) {
        if (params.n_slot == 0) {
            fprintf(stderr,
                    "%s: only %u slots fit (%.1fx the working set), which is slower than plain "
                    "--cpu-moe. BELLS disabled.\n"
                    "%s: the expert cache and the KV cache come out of the same VRAM and the KV "
                    "cache is allocated first, so a shorter -c leaves room for a bigger cache. "
                    "Pass --bells-slots N to force it anyway.\n",
                    __func__, n_slot, ratio, __func__);
            return false;
        }

        fprintf(stderr, "%s: %u slots is only %.1fx the working set and measured slower than "
                        "--cpu-moe at this ratio; continuing because it was requested "
                        "explicitly\n", __func__, n_slot, ratio);
    }

    if (!tensors_.init(buft, srcs, n_slot, backend)) {
        fprintf(stderr, "%s: failed to allocate %u expert slots per layer\n", __func__, n_slot);
        return false;
    }

    uint32_t n_layer = 0;
    for (const auto & s : srcs) {
        n_layer = std::max(n_layer, (uint32_t) s.il + 1);
    }

    cache_.init(n_layer, n_expert, n_slot);

    if (params_.max_tokens == 0) {
        // n tokens may request up to n * n_expert_used distinct experts; anything beyond what
        // the cache holds could fail ensure() mid-graph, where there is no way to recover.
        params_.max_tokens = std::max(1u, n_slot/std::max(1u, n_expert_used));
    }

    params_.n_slot = n_slot;
    ready_         = true;
    n_copied_      = 0;

    fprintf(stderr, "%s: %u slots/layer of %u experts, %.1f MiB VRAM, serves ubatch <= %u%s\n",
            __func__, n_slot, n_expert, tensors_.vram_bytes()/1024.0/1024.0,
            params_.max_tokens,
            params_.passive ? "  [PASSIVE: cache allocated but unused, research mode]" : "");

    return true;
}

void bells_runtime::free() {
    const uint64_t tot = cache_.n_hit() + cache_.n_miss();
    if (tot > 0) {
        fprintf(stderr, "%s: hit %.1f%% (%llu of %llu), %llu experts copied, %.2f GiB moved\n",
                __func__, 100.0*cache_.n_hit()/tot,
                (unsigned long long) cache_.n_hit(), (unsigned long long) tot,
                (unsigned long long) n_copied_, bytes_moved()/1024.0/1024.0/1024.0);
    }

    // Where the per-layer cost actually goes. readback and upload are paid on every layer of
    // every token regardless of hit rate; only copy scales with misses. If readback+upload
    // dominates, no amount of cache tuning helps and the design needs to stop round-tripping
    // through the host.
    if (n_layer_calls_ > 0) {
        const double per = 1.0/(double) n_layer_calls_;
        fprintf(stderr, "%s: per layer-call: readback %.1f us, copy %.1f us, upload %.1f us "
                        "(%llu calls, %.1f ms total)\n",
                __func__, us_readback_*per, us_copy_*per, us_upload_*per,
                (unsigned long long) n_layer_calls_,
                (us_readback_ + us_copy_ + us_upload_)/1000.0);
    }

    tensors_.free();
    cache_.reset();

    ready_    = false;
    n_copied_ = 0;
}

void bells_runtime::begin_ubatch(int64_t n_tokens) {
    active_now_ = active(n_tokens);
}

bool bells_runtime::on_routing(uint32_t il, const int32_t * experts, size_t n) {
    // prefill bypasses the cache in the graph, so it must not touch residency here either
    if (!ready_ || !active_now_) {
        return true;
    }

    n_layer_calls_++;

    // Every expert this layer asked for must be resident before the matmul reads the slot
    // table, so this is a demand fetch with no lead time to hide it behind.
    copies_.clear();
    if (!cache_.ensure(il, experts, n, copies_)) {
        return false;
    }

    const auto t_copy0 = std::chrono::steady_clock::now();

    // No parallel prefault here, though it is tempting. It belonged to the prefetch path, where
    // one call covered every layer's admissions for the whole token. On the demand path it
    // would run per layer - 48x per token on Qwen3-Next, spawning a thread pool each time - and
    // measuring it that way cost more than it saved: 1.01/1.14/1.12x against --cpu-moe at
    // concurrency 1-4 became 0.88/0.94/0.93x. Its only win was GPT-OSS-120B, a configuration
    // BELLS loses at anyway, so it never converted a loss into a win. See RESULTS.md.
    for (const auto & c : copies_) {
        tensors_.copy_expert(il, c.expert, c.slot);
    }
    n_copied_ += copies_.size();

    const auto t_copy1 = std::chrono::steady_clock::now();

    tensors_.upload_slots(il, cache_.slot_table(il));

    const auto t_up1 = std::chrono::steady_clock::now();

    us_copy_   += std::chrono::duration_cast<std::chrono::microseconds>(t_copy1 - t_copy0).count();
    us_upload_ += std::chrono::duration_cast<std::chrono::microseconds>(t_up1  - t_copy1).count();

    return true;
}
