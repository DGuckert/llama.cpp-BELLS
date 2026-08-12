#include "llama-bells.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <thread>

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

void bells_cache::prefetch(uint32_t il, const int32_t * experts, size_t n, std::vector<bells_copy> & out) {
    if (!enabled() || il >= n_layer_) {
        return;
    }

    layer & l = layers_[il];

    clock_++;

    for (size_t i = 0; i < n; ++i) {
        const int32_t e = experts[i];
        if (e < 0 || (uint32_t) e >= n_expert_) {
            continue;
        }

        if (l.expert_slot[e] >= 0) {
            l.last_used[l.expert_slot[e]] = clock_;
            continue;
        }

        const int32_t s = victim(l, experts, n);
        if (s < 0) {
            // everything resident is wanted by this same prefetch, nothing to give up
            break;
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

void bells_tensors::prefault(const std::vector<std::pair<uint32_t, int32_t>> & experts, int n_threads) const {
    if (experts.empty() || n_threads < 1) {
        return;
    }

    n_threads = std::min<int>(n_threads, (int) experts.size());

    // reading one byte per page is enough to fault it in; volatile keeps it from being elided
    auto worker = [&](int id) {
        for (size_t i = id; i < experts.size(); i += n_threads) {
            const uint32_t il = experts[i].first;
            const int32_t   e = experts[i].second;

            if (!has(il)) {
                continue;
            }

            const entry & ent = get(il);

            for (ggml_tensor * src : { ent.src.gate, ent.src.up, ent.src.down, ent.src.gate_up }) {
                if (!src || !src->data) {
                    continue;
                }

                const size_t stride = ggml_nbytes(src)/src->ne[2];
                const volatile char * p = (const char *) src->data + (size_t) e*stride;

                for (size_t off = 0; off < stride; off += 4096) {
                    (void) p[off];
                }
            }
        }
    };

    std::vector<std::thread> pool;
    pool.reserve(n_threads - 1);

    for (int t = 1; t < n_threads; ++t) {
        pool.emplace_back(worker, t);
    }
    worker(0);

    for (auto & t : pool) {
        t.join();
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
// predictor
//

bool bells_predictor::load(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }

    char magic[8] = { 0 };
    f.read(magic, 8);
    if (memcmp(magic, "BELLSPR1", 8) != 0) {
        return false;
    }

    uint32_t version  = 0;
    uint32_t n_token  = 0;

    f.read((char *) &version,  sizeof(version));
    f.read((char *) &n_layer_, sizeof(n_layer_));
    f.read((char *) &n_take_,  sizeof(n_take_));
    f.read((char *) &n_token,  sizeof(n_token));

    if (version != 1 || n_layer_ == 0 || n_take_ == 0) {
        n_layer_ = 0;
        return false;
    }

    tokens_.resize(n_token);
    fallback_.resize((size_t) n_layer_*n_take_);
    ranked_.resize((size_t) n_token*n_layer_*n_take_);

    f.read((char *) tokens_.data(),   tokens_.size()  *sizeof(int32_t));
    f.read((char *) fallback_.data(), fallback_.size()*sizeof(int32_t));
    f.read((char *) ranked_.data(),   ranked_.size()  *sizeof(int32_t));

    if (!f) {
        n_layer_ = 0;
        return false;
    }

    return true;
}

bool bells_predictor::in_table(int32_t token) const {
    const auto it = std::lower_bound(tokens_.begin(), tokens_.end(), token);
    return it != tokens_.end() && *it == token;
}

const int32_t * bells_predictor::predict(int32_t token, uint32_t il) const {
    if (!enabled() || il >= n_layer_) {
        return nullptr;
    }

    const auto it = std::lower_bound(tokens_.begin(), tokens_.end(), token);
    if (it == tokens_.end() || *it != token) {
        return fallback_.data() + (size_t) il*n_take_;
    }

    const size_t idx = it - tokens_.begin();

    return ranked_.data() + ((size_t) idx*n_layer_ + il)*n_take_;
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

    // The fit verdict. This one ratio decided every result we measured.
    {
        // Thresholds calibrated against measured outcomes on a 6 GB card, not guessed:
        //   GPT-OSS-120B  1.0x -> 1.59x slower
        //   Qwen3-30B-A3B 2.2x -> 1.52x faster
        //   Qwen3-Next80B 4.7x -> 1.18x faster
        // The useful threshold sits near 2x. An earlier draft demanded 8x and would have
        // told users to skip configurations that measure as the biggest wins we have.
        const double ratio = (double) n_slot/std::max(1u, n_expert_used);
        const char * verdict =
            ratio >= 4.0 ? "good fit"     :
            ratio >= 2.0 ? "workable"     :
            ratio >= 1.5 ? "marginal, may be slower than --cpu-moe" :
                           "poor fit, expect a slowdown";

        fprintf(stderr, "%s: working set %.2f GiB (%zu layers x %u experts x %.2f MiB), "
                        "cache holds %.1fx it -> %s\n",
                __func__, working_set/1024.0/1024.0/1024.0, srcs.size(), n_expert_used,
                per_expert/1024.0/1024.0, ratio, verdict);
    }

    // copy_expert reads straight out of the source tensors, so they must stay host resident.
    // That is the normal state when experts are offloaded, but check rather than assume.
    for (const auto & s : srcs) {
        for (ggml_tensor * t : { s.gate, s.up, s.down, s.gate_up }) {
            if (t && t->buffer && !ggml_backend_buffer_is_host(t->buffer)) {
                fprintf(stderr, "%s: layer %d expert tensors are not host resident, "
                                "BELLS needs them on the CPU (try --cpu-moe)\n", __func__, s.il);
                return false;
            }
        }
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

    if (!params.table.empty() && !predictor_.load(params.table)) {
        fprintf(stderr, "%s: could not load predictor table '%s', "
                        "running demand-only\n", __func__, params.table.c_str());
    }

    if (params_.n_prefetch == 0) {
        params_.n_prefetch = std::max(1u, n_slot/2);
    }
    params_.n_prefetch = std::min(params_.n_prefetch, n_slot);

    if (params_.max_tokens == 0) {
        // n tokens may request up to n * n_expert_used distinct experts; anything beyond what
        // the cache holds could fail ensure() mid-graph, where there is no way to recover.
        params_.max_tokens = std::max(1u, n_slot/std::max(1u, n_expert_used));
    }

    params_.n_slot = n_slot;
    ready_         = true;
    n_copied_      = 0;

    fprintf(stderr, "%s: %u slots/layer of %u experts, %.1f MiB VRAM, serves ubatch <= %u, "
                    "predictor %s\n",
            __func__, n_slot, n_expert, tensors_.vram_bytes()/1024.0/1024.0,
            params_.max_tokens, predictor_.enabled() ? "on" : "off");

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

    tensors_.free();
    cache_.reset();

    ready_    = false;
    token_    = -1;
    n_copied_ = 0;
}

void bells_runtime::begin_ubatch(int32_t token, int64_t n_tokens) {
    token_      = token;
    active_now_ = active(n_tokens);

    if (!active_now_ || !predictor_.enabled() || token < 0) {
        return;
    }

    // Prefetch every layer here, before the graph runs. Doing this inside on_routing would
    // be pointless: by then the router has already said which experts it wants, so there is
    // no lead time left to hide anything behind.
    // Decide every layer's admissions first, fault all the source pages in parallel, then
    // copy. Faulting inside the copy loop leaves the drive at queue depth 1.
    pending_.clear();
    faults_.clear();

    for (int32_t il : tensors_.layers()) {
        const int32_t * p = predictor_.predict(token, (uint32_t) il);
        if (!p) {
            continue;
        }

        const uint32_t take = std::min(predictor_.n_take(), params_.n_prefetch);

        copies_.clear();
        cache_.prefetch((uint32_t) il, p, take, copies_);

        for (const auto & c : copies_) {
            pending_.push_back({ (uint32_t) il, c });
            faults_.emplace_back((uint32_t) il, c.expert);
        }
    }

    if (params_.n_fault_threads > 0) {
        tensors_.prefault(faults_, params_.n_fault_threads);
    }

    for (const auto & pc : pending_) {
        tensors_.copy_expert(pc.il, pc.copy.expert, pc.copy.slot);
    }
    n_copied_ += pending_.size();

    for (int32_t il : tensors_.layers()) {

        // In drop_missing mode nothing later fixes up the table, so publish it now with every
        // non-resident expert pointed at the zero slot. The graph then runs start to finish
        // without a single host round-trip.
        if (params_.drop_missing) {
            table_ = cache_.slot_table((uint32_t) il);

            const int32_t zero = tensors_.zero_slot();
            for (auto & s : table_) {
                if (s < 0) {
                    s = zero;
                }
            }

            tensors_.upload_slots((uint32_t) il, table_);
        }
    }
}

bool bells_runtime::on_routing(uint32_t il, const int32_t * experts, size_t n) {
    // prefill bypasses the cache in the graph, so it must not touch residency here either
    if (!ready_ || !active_now_ || params_.drop_missing) {
        return true;
    }

    // prefetching already happened in begin_ubatch; all that is left is to correct whatever
    // the prediction missed, which must be resident before the matmul reads the slot table
    copies_.clear();
    if (!cache_.ensure(il, experts, n, copies_)) {
        return false;
    }

    for (const auto & c : copies_) {
        tensors_.copy_expert(il, c.expert, c.slot);
    }
    n_copied_ += copies_.size();

    tensors_.upload_slots(il, cache_.slot_table(il));

    return true;
}
