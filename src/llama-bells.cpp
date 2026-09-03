#include "llama-bells.h"

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX          // windows.h defines min/max macros that break std::min/std::max
#  include <windows.h>
#  include <memoryapi.h>
#  include <psapi.h>
#  pragma comment(lib, "psapi.lib")
#elif defined(__linux__)
#  include <sys/mman.h>
#  include <unistd.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

void bells_cache::init(uint32_t n_layer, uint32_t n_expert, uint32_t n_slot) {
    n_layer_  = n_layer;
    n_expert_ = n_expert;
    n_slot_   = std::min(n_slot, n_expert);

    layers_.assign(n_layer, layer());

    for (auto & l : layers_) {
        l.expert_slot.assign(n_expert_, -1);
        l.slot_expert.assign(n_slot_,   -1);
        l.last_used.assign(n_slot_,      0);
        l.pinned.assign(n_slot_,         0);
    }

    clock_    = 0;
    n_hit_    = 0;
    n_miss_   = 0;
    n_pinned_ = 0;
}

void bells_cache::reset() {
    for (auto & l : layers_) {
        std::fill(l.expert_slot.begin(), l.expert_slot.end(), -1);
        std::fill(l.slot_expert.begin(), l.slot_expert.end(), -1);
        std::fill(l.last_used.begin(),   l.last_used.end(),    0);
    }

    clock_ = 0;
}

void bells_cache::pin_experts(uint32_t il, const std::vector<int32_t> & experts,
                              std::vector<bells_copy> & out) {
    if (il >= layers_.size()) {
        return;
    }
    layer & l = layers_[il];

    uint32_t s = 0;
    for (int32_t e : experts) {
        if (s >= n_slot_) {
            break;
        }
        if (e < 0 || (uint32_t) e >= n_expert_) {
            continue;
        }
        if (l.expert_slot[e] >= 0) {
            continue;   // already seated
        }

        // evict whatever is here; at init nothing is
        const int32_t held = l.slot_expert[s];
        if (held >= 0) {
            l.expert_slot[held] = -1;
        }

        l.slot_expert[s] = e;
        l.expert_slot[e] = (int32_t) s;
        l.last_used[s]   = ++clock_;
        l.pinned[s]      = 1;

        out.push_back({ e, (int32_t) s });
        s++;
    }

    n_pinned_ = std::max(n_pinned_, s);
}

int32_t bells_cache::victim(layer & l, const int32_t * keep, size_t n_keep,
                            const std::vector<uint8_t> * protect) const {
    int32_t best     = -1;
    int64_t best_age = 0;

    // Second choice, used only when everything unprotected is spoken for. Keeping it means a
    // wrong prediction costs a worse eviction rather than a failed ensure().
    int32_t fallback     = -1;
    int64_t fallback_age = 0;

    for (uint32_t s = 0; s < n_slot_; ++s) {
        const int32_t held = l.slot_expert[s];

        if (held < 0) {
            return (int32_t) s;
        }

        // Permanently seated by --pin-experts. Skipped before anything else is considered, so
        // the measured hot set cannot be cycled out by a single cold token.
        if (!l.pinned.empty() && l.pinned[s]) {
            continue;
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

        const bool wanted_soon = protect && (size_t) held < protect->size() && (*protect)[held];

        if (wanted_soon) {
            if (fallback < 0 || l.last_used[s] < fallback_age) {
                fallback     = (int32_t) s;
                fallback_age = l.last_used[s];
            }
            continue;
        }

        if (best < 0 || l.last_used[s] < best_age) {
            best     = (int32_t) s;
            best_age = l.last_used[s];
        }
    }

    return best >= 0 ? best : fallback;
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

        // Only the other members of this prefetch are protected. What the token will actually
        // want is unknown here, so a wrong guess can evict something needed - that cost is real
        // and is why the confidence threshold matters.
        const int32_t s = victim(l, experts, n);
        if (s < 0) {
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

bool bells_cache::ensure(uint32_t il, const int32_t * experts, size_t n,
                         std::vector<bells_copy> & out, const std::vector<uint8_t> * protect) {
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

        const int32_t s = victim(l, experts, n, protect);
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
                         uint32_t n_slot, ggml_backend_t backend, ggml_backend_t copy_backend) {
    free();

    backend_      = backend;
    copy_backend_ = copy_backend;

    if (copy_backend_) {
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        copy_event_ = dev ? ggml_backend_event_new(dev) : nullptr;
        if (!copy_event_) {
            // without an event there is no way to order the two streams, and guessing would
            // mean the graph reading half-written experts
            copy_backend_ = nullptr;
        }
    }

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

    // Group the layers by the device they run on. A cache slice has to live where the matmul
    // that reads it runs, so with a layer split across cards each device needs its own context
    // and buffer. One device is the ordinary case and falls out of this unchanged.
    std::vector<ggml_backend_buffer_type_t> bufts;
    std::vector<std::vector<size_t>>        by_buft;   // indices into srcs

    for (size_t i = 0; i < srcs.size(); ++i) {
        ggml_backend_buffer_type_t b = srcs[i].buft ? srcs[i].buft : buft;
        if (!b) {
            return false;
        }

        size_t g = 0;
        while (g < bufts.size() && bufts[g] != b) {
            g++;
        }
        if (g == bufts.size()) {
            bufts.push_back(b);
            by_buft.emplace_back();
        }
        by_buft[g].push_back(i);
    }

    // The second stream orders one copy backend against one compute backend with a single event.
    // Across devices that is no longer a well-defined pairing, and it has never been worth
    // anything anyway - measured 1.005x, because the per-layer readback is the real barrier.
    if (bufts.size() > 1 && copy_backend_) {
        if (copy_event_) {
            ggml_backend_event_free(copy_event_);
            copy_event_ = nullptr;
        }
        copy_backend_ = nullptr;
    }

    vram_bytes_ = 0;

    for (size_t g = 0; g < bufts.size(); ++g) {
        // 5 tensors per layer at most, plus overhead
        ggml_init_params ip = { ggml_tensor_overhead()*by_buft[g].size()*8, nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        if (!ctx) {
            free();
            return false;
        }
        ctxs_.push_back(ctx);

        for (size_t i : by_buft[g]) {
            const layer_src & s = srcs[i];

            entry e;
            e.src     = s;
            e.backend = s.backend ? s.backend : backend;

            e.gate    = bells_make_slice(ctx, s.gate,    n_alloc);
            e.up      = bells_make_slice(ctx, s.up,      n_alloc);
            e.down    = bells_make_slice(ctx, s.down,    n_alloc);
            e.gate_up = bells_make_slice(ctx, s.gate_up, n_alloc);

            ggml_tensor * any = s.gate ? s.gate : (s.gate_up ? s.gate_up : s.up);
            if (!any) {
                free();
                return false;
            }

            e.slots = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, any->ne[2]);
            ggml_set_input(e.slots);

            index_[s.il] = (int32_t) entries_.size();
            entries_.push_back(e);
            layer_ids_.push_back(s.il);
        }

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, bufts[g]);
        if (!buf) {
            free();
            return false;
        }
        buffers_.push_back(buf);

        vram_bytes_ += ggml_backend_buffer_get_size(buf);
    }

    // layer_ids_ must stay ordered by layer for callers that walk it as a schedule; grouping by
    // device above interleaves it when the split is not contiguous.
    std::sort(layer_ids_.begin(), layer_ids_.end());

    bytes_per_expert_ = 0;
    const entry & first = entries_.front();
    for (ggml_tensor * t : { first.gate, first.up, first.down, first.gate_up }) {
        if (t) {
            bytes_per_expert_ += ggml_nbytes(t)/t->ne[2];
        }
    }

    // Zero every slot, not just the spare one.
    //
    // Only slot n_slot was being initialised, so slots 0..n_slot-1 held whatever cudaMalloc
    // returned until a copy happened to fill them. A read of a slot before its first copy
    // dequantises uninitialised VRAM as q4_K/q5_K, which yields inf, and the accumulation then
    // yields NaN. Zeroed weights are the safe value: a zeroed quant block contributes nothing,
    // so a premature read degrades that expert's contribution instead of poisoning the pass.
    //
    // This is a latent defect, not the cause of the ///// corruption that prompted the search -
    // that was a host hardware fault, and this fix measured 1 clean run in 40 against a 0-in-46
    // baseline, i.e. no effect. Kept because reading never-written memory is wrong regardless.
    {
        std::vector<char> zeros(bytes_per_expert_, 0);

        for (auto & e : entries_) {
            for (ggml_tensor * t : { e.gate, e.up, e.down, e.gate_up }) {
                if (!t) {
                    continue;
                }
                const size_t stride = ggml_nbytes(t)/t->ne[2];
                for (int64_t s = 0; s < t->ne[2]; ++s) {
                    ggml_backend_tensor_set(t, zeros.data(), (size_t) s*stride, stride);
                }
            }
        }
    }

    return true;
}

// Advise the OS to fault these ranges in and keep them. Advisory and asynchronous on both
// platforms: the call queues the reads and returns, so pages arrive ahead of demand rather than
// stalling a matmul. Failure is ignored by design - this is a hint, not a guarantee.
// Filter candidates down to pages that are NOT already resident, hint those, and report how
// many were missing. The count drives the two-sided backoff in warm_lookahead.
//
// Both halves are batched. QueryWorkingSetEx and PrefetchVirtualMemory each accept an array, and
// calling them per-expert cost more than the faults being saved: 18.50 tok/s against a 19.14
// baseline, purely in query overhead. Skipping resident pages is what makes this a hint rather
// than an I/O storm - without it, measured 6.56 against the same 19.14.
static size_t bells_warm_ranges(const std::vector<std::pair<void *, size_t>> & ranges) {
    if (ranges.empty()) {
        return 0;
    }

#if defined(_WIN32)
    std::vector<PSAPI_WORKING_SET_EX_INFORMATION> info(ranges.size());
    for (size_t i = 0; i < ranges.size(); ++i) {
        info[i].VirtualAddress = ranges[i].first;
    }
    if (!QueryWorkingSetEx(GetCurrentProcess(), info.data(),
                           (DWORD) (info.size()*sizeof(info[0])))) {
        return 0;   // cannot tell what is resident: do nothing rather than guess
    }

    std::vector<WIN32_MEMORY_RANGE_ENTRY> ents;
    for (size_t i = 0; i < ranges.size(); ++i) {
        if (!info[i].VirtualAttributes.Valid) {
            WIN32_MEMORY_RANGE_ENTRY e;
            e.VirtualAddress = ranges[i].first;
            e.NumberOfBytes  = ranges[i].second;
            ents.push_back(e);
        }
    }
    if (ents.empty()) {
        return 0;   // common case once warm: one query, no I/O
    }
    PrefetchVirtualMemory(GetCurrentProcess(), ents.size(), ents.data(), 0);
    return ents.size();

#elif defined(__linux__)
    const size_t pg = (size_t) sysconf(_SC_PAGESIZE);
    size_t hinted = 0;
    for (const auto & r : ranges) {
        unsigned char vec = 0;
        void * base = (void *) ((uintptr_t) r.first & ~(uintptr_t)(pg - 1));
        if (mincore(base, pg, &vec) == 0 && (vec & 1)) {
            continue;
        }
        madvise(r.first, r.second, MADV_WILLNEED);
        hinted++;
    }
    return hinted;

#else
    (void) ranges;
    return 0;
#endif
}

void bells_tensors::warm_init(uint32_t n_layer, uint32_t n_expert) {
    warm_keep_ = 0;
    if (const char * s = getenv("BELLS_MMAP_WARM")) {
        const int v = atoi(s);
        if (v > 0) {
            warm_keep_ = (uint32_t) v;
        }
    }
    if (!warm_keep_ || n_expert == 0) {
        return;
    }

    warm_n_expert_ = n_expert;
    warm_used_.assign((size_t) n_layer * n_expert, 0);
    warm_clock_    = 0;

    fprintf(stderr, "%s: keeping the %u hottest experts per layer warm in the page cache\n",
            __func__, warm_keep_);
}

void bells_tensors::warm_note(uint32_t il, const int32_t * experts, size_t n) {
    if (!warm_keep_ || warm_n_expert_ == 0) {
        return;
    }
    std::lock_guard<std::mutex> lk(warm_mu_);
    warm_clock_++;
    for (size_t i = 0; i < n; ++i) {
        const int32_t e = experts[i];
        if (e < 0 || (uint32_t) e >= warm_n_expert_) {
            continue;
        }
        const size_t idx = (size_t) il*warm_n_expert_ + (size_t) e;
        if (idx < warm_used_.size()) {
            warm_used_[idx] = warm_clock_;
        }
    }
}

void bells_tensors::warm_lookahead(uint32_t il_next) {
    if (!warm_keep_ || warm_n_expert_ == 0 || entries_.empty()) {
        return;
    }

    // Nothing useful to do lately - stay out of the way. See the two-sided gate below.
    if (warm_skip_ > 0) {
        warm_skip_--;
        return;
    }

    // Only the next couple of layers, and only the hottest few experts in each: a small request
    // with real lead time. A full sweep asks for more than RAM holds and evicts what it is trying
    // to help - measured 0.63 tok/s against a 1.25 baseline before this was bounded.
    std::vector<std::pair<void *, size_t>> ranges;
    std::vector<std::pair<uint64_t, uint32_t>> rank;

    for (uint32_t ahead = 0; ahead < 2; ++ahead) {
        const uint32_t il = il_next + ahead;
        if (!has(il)) {
            continue;
        }
        const entry & e = get(il);

        rank.clear();
        {
            std::lock_guard<std::mutex> lk(warm_mu_);
            const size_t base = (size_t) il*warm_n_expert_;
            if (base + warm_n_expert_ > warm_used_.size()) {
                continue;
            }
            for (uint32_t x = 0; x < warm_n_expert_; ++x) {
                if (warm_used_[base + x] > 0) {
                    rank.emplace_back(warm_used_[base + x], x);
                }
            }
        }
        if (rank.empty()) {
            continue;
        }

        const size_t keep = std::min<size_t>(warm_keep_, rank.size());
        std::partial_sort(rank.begin(), rank.begin() + keep, rank.end(),
                          [](const std::pair<uint64_t, uint32_t> & a,
                             const std::pair<uint64_t, uint32_t> & b) { return a.first > b.first; });

        ggml_tensor * srcs[4] = { e.src.gate, e.src.up, e.src.down, e.src.gate_up };
        for (int k = 0; k < 4; ++k) {
            ggml_tensor * s = srcs[k];
            if (!s || !s->data || s->ne[2] <= 0) {
                continue;
            }
            const size_t stride = ggml_nbytes(s)/s->ne[2];
            for (size_t r = 0; r < keep; ++r) {
                ranges.emplace_back((char *) s->data + (size_t) rank[r].second*stride, stride);
            }
        }
    }

    const size_t candidates = ranges.size();
    const size_t missing    = bells_warm_ranges(ranges);

    warm_sweeps_++;
    warm_hinted_ += missing;

    // Back off at BOTH ends. Nothing missing means the page cache is coping; most missing means we
    // are cold and the demand path already owns the storage queue, so speculation only pushes the
    // blocking reads back. Measured: a one-sided gate cost the cold ramp 4.26 -> 2.24 tok/s.
    const bool idle      = (missing == 0);
    const bool saturated = (candidates > 0 && missing*2 > candidates);

    if (idle || saturated) {
        warm_backoff_ = std::min<uint32_t>(warm_backoff_*2, 256);
    } else {
        warm_backoff_ = 1;
    }
    warm_skip_ = warm_backoff_ - 1;
}

void bells_tensors::free() {
    if (copy_event_) {
        ggml_backend_event_synchronize(copy_event_);
        ggml_backend_event_free(copy_event_);
        copy_event_ = nullptr;
    }
    copy_backend_   = nullptr;
    copies_pending_ = false;

    for (ggml_backend_buffer_t b : buffers_) {
        if (b) {
            ggml_backend_buffer_free(b);
        }
    }
    buffers_.clear();

    for (ggml_context * c : ctxs_) {
        if (c) {
            ggml_free(c);
        }
    }
    ctxs_.clear();

    entries_.clear();
    index_.clear();
    layer_ids_.clear();

    vram_bytes_       = 0;
    bytes_per_expert_ = 0;
}

void bells_tensors::copy_one(ggml_tensor * dst, ggml_tensor * src, int32_t expert, int32_t slot,
                             ggml_backend_t backend) {
    if (!dst || !src) {
        return;
    }

    const size_t stride = ggml_nbytes(src)/src->ne[2];

    // the source expert stack is host resident, which is the whole premise of offloading
    const char * base = (const char *) src->data + (size_t) expert*stride;

    // BELLS_SYNC_COPY=1 forces the fully blocking copy. Diagnostic for the sm_86 corruption:
    // if garbage survives a synchronous copy then the defect is in the data or the indexing,
    // not in the ordering between the transfer and the matmul that reads it.
    static const bool sync_copy = [] {
        const char * s = getenv("BELLS_SYNC_COPY");
        return s && s[0] && s[0] != '0';
    }();

    if (sync_copy) {
        ggml_backend_tensor_set(dst, base, (size_t) slot*stride, stride);
        return;
    }

    // copy_backend_ is only ever set in the single-device case; see init(). Otherwise the copy
    // has to be issued against the device the destination actually lives on.
    if (copy_backend_) {
        // second stream: this can genuinely overlap the graph, which is the entire point
        ggml_backend_tensor_set_async(copy_backend_, dst, base, (size_t) slot*stride, stride);
        copies_pending_ = true;
    } else if (backend) {
        ggml_backend_tensor_set_async(backend, dst, base, (size_t) slot*stride, stride);
    } else {
        ggml_backend_tensor_set(dst, base, (size_t) slot*stride, stride);
    }
}

void bells_tensors::sync_copies() {
    if (!copies_pending_ || !copy_backend_ || !copy_event_ || !backend_) {
        return;
    }

    // Stream-level ordering, not a host stall: the compute stream is told to wait for the
    // copies, and the host returns immediately.
    ggml_backend_event_record(copy_event_, copy_backend_);
    ggml_backend_event_wait(backend_, copy_event_);

    copies_pending_ = false;
}

void bells_tensors::copy_expert(uint32_t il, int32_t expert, int32_t slot) {
    if (!has(il)) {
        return;
    }

    entry & e = get_mut(il);

    copy_one(e.gate,    e.src.gate,    expert, slot, e.backend);
    copy_one(e.up,      e.src.up,      expert, slot, e.backend);
    copy_one(e.down,    e.src.down,    expert, slot, e.backend);
    copy_one(e.gate_up, e.src.gate_up, expert, slot, e.backend);

    // BELLS_VERIFY_COPY=1 reads each slot back and compares it against the host source. This is
    // the question the sm_86 corruption turns on: if the bytes match, the copy is fine and the
    // defect is in how the matmul reads the cache; if they differ, the transfer itself is wrong.
    // Slow and synchronising - diagnostic only.
    static const bool verify = [] {
        const char * s = getenv("BELLS_VERIFY_COPY");
        return s && s[0] && s[0] != '0';
    }();

    if (!verify) {
        return;
    }

    static int n_checked = 0;
    static int n_bad     = 0;
    if (n_checked >= 64) {
        return;
    }

    struct { ggml_tensor * dst; ggml_tensor * src; const char * name; } pairs[] = {
        { e.gate,    e.src.gate,    "gate"    },
        { e.up,      e.src.up,      "up"      },
        { e.down,    e.src.down,    "down"    },
        { e.gate_up, e.src.gate_up, "gate_up" },
    };

    // copy_one derives the per-expert stride from the SOURCE and uses it to offset into the
    // DESTINATION. If the cache tensor strides differently the copy lands at the wrong offset,
    // and a readback using the same source stride would not notice. Compare them once.
    static bool geometry_logged = false;
    if (!geometry_logged) {
        geometry_logged = true;
        for (const auto & p : pairs) {
            if (!p.dst || !p.src) {
                continue;
            }
            const size_t s_stride = ggml_nbytes(p.src)/p.src->ne[2];
            const size_t d_stride = ggml_nbytes(p.dst)/p.dst->ne[2];
            fprintf(stderr, "%s: geom %s: src type=%s ne=[%lld,%lld,%lld] nb1=%zu nb2=%zu stride=%zu | "
                            "dst type=%s ne=[%lld,%lld,%lld] nb1=%zu nb2=%zu stride=%zu%s\n",
                    __func__, p.name,
                    ggml_type_name(p.src->type),
                    (long long) p.src->ne[0], (long long) p.src->ne[1], (long long) p.src->ne[2],
                    p.src->nb[1], p.src->nb[2], s_stride,
                    ggml_type_name(p.dst->type),
                    (long long) p.dst->ne[0], (long long) p.dst->ne[1], (long long) p.dst->ne[2],
                    p.dst->nb[1], p.dst->nb[2], d_stride,
                    s_stride == d_stride ? "" : "   <<< STRIDE MISMATCH");
        }
    }

    std::vector<uint8_t> got;

    for (const auto & p : pairs) {
        if (!p.dst || !p.src) {
            continue;
        }

        const size_t stride = ggml_nbytes(p.src)/p.src->ne[2];
        const uint8_t * want = (const uint8_t *) p.src->data + (size_t) expert*stride;

        got.resize(stride);
        ggml_backend_tensor_get(p.dst, got.data(), (size_t) slot*stride, stride);

        if (memcmp(got.data(), want, stride) != 0) {
            size_t first = 0;
            size_t ndiff = 0;
            for (size_t i = 0; i < stride; ++i) {
                if (got[i] != want[i]) {
                    if (ndiff == 0) {
                        first = i;
                    }
                    ndiff++;
                }
            }
            n_bad++;
            fprintf(stderr, "%s: MISMATCH layer %u expert %d slot %d %s: %zu/%zu bytes differ, "
                            "first at %zu (host 0x%02x, vram 0x%02x)\n",
                    __func__, il, expert, slot, p.name, ndiff, stride, first,
                    want[first], got[first]);
        }

        n_checked++;
    }

    if (n_checked >= 64) {
        fprintf(stderr, "%s: copy verification done - %d checks, %d mismatched\n",
                __func__, n_checked, n_bad);
    }
}

void bells_tensors::sync_compute() {
    if (backend_) {
        ggml_backend_synchronize(backend_);
    }
}

bool bells_tensors::slot_matches_expert(uint32_t il, int32_t expert, int32_t slot) {
    if (!has(il) || expert < 0 || slot < 0) {
        return false;
    }

    entry & e = get_mut(il);

    // one tensor is enough to catch a mismatched slot, and gate is always present
    ggml_tensor * dst = e.gate;
    ggml_tensor * src = e.src.gate;
    if (!dst || !src) {
        return true;
    }

    const size_t stride = ggml_nbytes(src)/src->ne[2];

    // compare a prefix rather than the whole expert: a wrong slot differs almost everywhere,
    // and reading 2 MB per expert per token would dominate the run
    const size_t n_cmp = std::min<size_t>(stride, 4096);

    std::vector<uint8_t> got(n_cmp);
    ggml_backend_tensor_get(dst, got.data(), (size_t) slot*stride, n_cmp);

    const uint8_t * want = (const uint8_t *) src->data + (size_t) expert*stride;

    return memcmp(got.data(), want, n_cmp) == 0;
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

    // Redirect non-resident experts to the spare zero slot instead of uploading -1.
    //
    // init() allocates n_slot+1 rows and keeps the last one zeroed for exactly this case, and
    // zero_slot() names it, but nothing ever used it - a -1 went to the graph verbatim. What the
    // matmul then does with it depends on the kernel: mul_mat_vec_q casts the id to unsigned and
    // offsets src0 by it, so -1 reads far out of bounds.
    //
    // Latent, like the zeroing above: measured 1 clean run in 40 against a 0-in-46 baseline, so it
    // is not what caused the ///// corruption (a host DMA fault was). Correct anyway.
    slot_scratch_.assign(table.begin(), table.begin() + n);
    for (size_t i = 0; i < n; ++i) {
        if (slot_scratch_[i] < 0 || (uint32_t) slot_scratch_[i] > n_slot_) {
            slot_scratch_[i] = (int32_t) n_slot_;
        }
    }

    // NOTE: cannot go async here. slot_scratch_ is reused on the next call, so the write has to
    // complete before returning.
    ggml_backend_tensor_set(t, slot_scratch_.data(), 0, n*sizeof(int32_t));

    // BELLS_TRACE_EVAL=1 also reports slot-table uploads. A cache that allocates and
    // substitutes correctly still produces garbage if this never runs: the table stays
    // zero, every expert remaps to slot 0, and the matmuls index it happily.
    {
        static const bool tr = [] {
            const char * s = getenv("BELLS_TRACE_EVAL");
            return s && s[0] && s[0] != '0';
        }();
        static int n_up = 0;
        if (tr && n_up < 10) {
            n_up++;
            fprintf(stderr, "upload_slots: il=%u n=%zu table[0..5]=%d,%d,%d,%d,%d,%d\n",
                    il, n,
                    n > 0 ? table[0] : -1, n > 1 ? table[1] : -1, n > 2 ? table[2] : -1,
                    n > 3 ? table[3] : -1, n > 4 ? table[4] : -1, n > 5 ? table[5] : -1);
            fflush(stderr);
        }
    }

    // BELLS_VERIFY_SLOTS=1 reads the table back. The copies were proven byte-correct on sm_86
    // while the output was still garbage, so the remaining suspect is the mapping the matmul
    // indexes through. Diagnostic only.
    static const bool verify = [] {
        const char * s = getenv("BELLS_VERIFY_SLOTS");
        return s && s[0] && s[0] != '0';
    }();

    if (!verify) {
        return;
    }

    static int n_layers_checked = 0;
    if (n_layers_checked >= 8) {
        return;
    }
    n_layers_checked++;

    std::vector<int32_t> got(n);
    ggml_backend_tensor_get(t, got.data(), 0, n*sizeof(int32_t));

    size_t n_diff = 0;
    size_t n_oob  = 0;
    for (size_t i = 0; i < n; ++i) {
        if (got[i] != table[i]) {
            n_diff++;
        }
        if (got[i] < 0 || (uint32_t) got[i] > n_slot_) {
            n_oob++;
        }
    }

    fprintf(stderr, "%s: layer %u slots: tensor=%p data=%p %zu entries, %zu readback mismatches, "
                    "%zu out of range (n_slot=%u), first 8: %d %d %d %d %d %d %d %d\n",
            __func__, il, (void *) t, (void *) t->data, n, n_diff, n_oob, n_slot_,
            n > 0 ? got[0] : -1, n > 1 ? got[1] : -1, n > 2 ? got[2] : -1, n > 3 ? got[3] : -1,
            n > 4 ? got[4] : -1, n > 5 ? got[5] : -1, n > 6 ? got[6] : -1, n > 7 ? got[7] : -1);
}

//
// confidence table
//

bool bells_conf::load(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }

    char magic[8] = { 0 };
    f.read(magic, 8);
    if (memcmp(magic, "BELLSCF1", 8) != 0) {
        return false;
    }

    uint32_t version = 0;
    uint64_t n_token = 0;

    f.read((char *) &version,   sizeof(version));
    f.read((char *) &n_layer_,  sizeof(n_layer_));
    f.read((char *) &n_expert_, sizeof(n_expert_));
    f.read((char *) &max_k_,    sizeof(max_k_));
    f.read((char *) &n_token,   sizeof(n_token));

    if (version != 1 || n_layer_ == 0 || max_k_ == 0 || n_token == 0) {
        n_layer_ = 0;
        return false;
    }

    std::vector<uint32_t> layer_ids(n_layer_);
    f.read((char *) layer_ids.data(), (std::streamsize) n_layer_*sizeof(uint32_t));

    uint32_t max_il = 0;
    for (uint32_t id : layer_ids) {
        max_il = std::max(max_il, id);
    }
    il_to_row_.assign(max_il + 1, -1);
    for (uint32_t r = 0; r < n_layer_; ++r) {
        il_to_row_[layer_ids[r]] = (int32_t) r;
    }

    tokens_.resize(n_token);
    f.read((char *) tokens_.data(), (std::streamsize) n_token*sizeof(int32_t));

    const size_t per = (size_t) n_layer_*max_k_;

    ex_.resize(n_token*per);
    cf_.resize(n_token*per);

    // written as int32[max_k] then float[max_k] per (token, layer)
    for (size_t t = 0; t < n_token; ++t) {
        for (uint32_t l = 0; l < n_layer_; ++l) {
            const size_t o = t*per + (size_t) l*max_k_;
            f.read((char *) (ex_.data() + o), max_k_*sizeof(int32_t));
            f.read((char *) (cf_.data() + o), max_k_*sizeof(float));
        }
    }

    gex_.resize(per);
    gcf_.resize(per);
    for (uint32_t l = 0; l < n_layer_; ++l) {
        const size_t o = (size_t) l*max_k_;
        f.read((char *) (gex_.data() + o), max_k_*sizeof(int32_t));
        f.read((char *) (gcf_.data() + o), max_k_*sizeof(float));
    }

    if (!f) {
        n_layer_ = 0;
        return false;
    }

    return true;
}

const int32_t * bells_conf::predict(int32_t token, uint32_t il, const float ** conf) const {
    if (!enabled() || il >= il_to_row_.size() || il_to_row_[il] < 0) {
        return nullptr;
    }

    const size_t row = (size_t) il_to_row_[il];
    const size_t per = (size_t) n_layer_*max_k_;

    const auto it = std::lower_bound(tokens_.begin(), tokens_.end(), token);
    if (it == tokens_.end() || *it != token) {
        *conf = gcf_.data() + row*max_k_;
        return gex_.data() + row*max_k_;
    }

    const size_t ti = it - tokens_.begin();
    *conf = cf_.data() + ti*per + row*max_k_;
    return ex_.data() + ti*per + row*max_k_;
}

//
// runtime
//

bool bells_runtime::init(const bells_params & params,
                         ggml_backend_buffer_type_t buft,
                         const std::vector<bells_tensors::layer_src> & srcs_all,
                         uint32_t n_expert,
                         uint32_t n_expert_used,
                         ggml_backend_t backend,
                         ggml_backend_t copy_backend) {
    free();

    params_ = params;
    n_expert_used_ = n_expert_used;

    if (!params.enabled || srcs_all.empty() || n_expert == 0) {
        return false;
    }

    // copy_expert reads straight out of the source tensors, so they must be allocated and host
    // resident. Checked first, before anything is sized, printed or allocated, because
    // common_fit_params probes memory by building a throwaway context over a model loaded with
    // no_alloc. That model has no tensor data, and BELLS used to build a full second cache
    // against it - briefly doubling VRAM use and pushing tight configurations into OOM, which
    // is why the docs had to tell people to pass -fit off.
    // Skip layers whose experts already live on the device rather than refusing to run.
    //
    // This check used to return false for the whole model on the first non-host-resident layer,
    // which made BELLS and -ot mutually exclusive. They are complementary: -ot puts a few layers'
    // experts permanently in VRAM, which is worth a lot to prefill, while BELLS caches the rest,
    // which is worth a lot to decode. Measured separately on Qwen3.6-35B at 262k, -ot 8 layers
    // gave 482 tok/s prefill against 331, and BELLS gave 31.8 tok/s decode against 27.5. There is
    // no reason those cannot compose - a layer already resident simply does not need caching.
    std::vector<bells_tensors::layer_src> host_srcs;
    host_srcs.reserve(srcs_all.size());

    uint32_t n_skipped = 0;

    for (const auto & s : srcs_all) {
        bool host_resident = true;

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
                host_resident = false;
                break;
            }
        }

        if (host_resident) {
            host_srcs.push_back(s);
        } else {
            n_skipped++;
        }
    }

    if (host_srcs.empty()) {
        fprintf(stderr, "%s: no expert tensors are host resident, nothing to cache "
                        "(try --cpu-moe)\n", __func__);
        return false;
    }

    if (n_skipped > 0) {
        fprintf(stderr, "%s: %u layer(s) already resident on the device, caching the other %zu\n",
                __func__, n_skipped, host_srcs.size());
    }

    // everything below sizes and allocates against the layers actually being cached
    const std::vector<bells_tensors::layer_src> & srcs = host_srcs;

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

    // Free VRAM on the *tightest* device, expressed as though every layer lived there.
    //
    // With the layers split across cards, each device holds only its own share, so the slot
    // count that fits is set by whichever device has the least room per layer it carries - not
    // by the total across all of them. Scaling that back up to a whole-model figure lets the
    // sizing arithmetic below stay as written.
    size_t dev_free = 0, dev_total = 0;
    {
        std::vector<ggml_backend_dev_t> devs;
        std::vector<size_t>             n_layer_on;

        for (const auto & s : srcs) {
            ggml_backend_dev_t d = ggml_backend_buft_get_device(s.buft ? s.buft : buft);
            size_t i = 0;
            while (i < devs.size() && devs[i] != d) {
                i++;
            }
            if (i == devs.size()) {
                devs.push_back(d);
                n_layer_on.push_back(0);
            }
            n_layer_on[i]++;
        }

        double tightest_free = 0.0, tightest_total = 0.0;

        for (size_t i = 0; i < devs.size(); ++i) {
            size_t f = 0, t = 0;
            if (devs[i]) {
                ggml_backend_dev_memory(devs[i], &f, &t);
            }

            // per-layer room on this device, scaled to the whole model
            const double share = (double) srcs.size()/std::max<size_t>(1, n_layer_on[i]);
            const double f_eq  = (double) f*share;
            const double t_eq  = (double) t*share;

            if (i == 0 || f_eq < tightest_free) {
                tightest_free  = f_eq;
                tightest_total = t_eq;
            }
        }

        dev_free  = (size_t) tightest_free;
        dev_total = (size_t) tightest_total;

        if (devs.size() > 1) {
            fprintf(stderr, "%s: cache split over %zu devices, sizing from the tightest\n",
                    __func__, devs.size());
        }
    }

    if (n_slot == 0) {
        // A third of the free VRAM, floored at 1 GiB of headroom.
        //
        // Deliberately conservative, and NOT tuned. It was originally fitted to one model on a
        // 6 GB card - a configuration since measured at 0.94-0.97x, i.e. one where BELLS does
        // not help at all - so it is a safe starting point rather than an optimum.
        //
        // It errs small because oversizing is punished much harder than undersizing. The cache
        // and the compute buffers come out of the same VRAM: a 19.9 GB cache on a 24 GB card
        // took the 235B from 1.81x to 1.02x, and 22.4 GB took it to 0.73x. An 81-slot cache on
        // a 6 GB card measured 25% slower than 48.
        //
        // Not re-tuned on the newer data because doing that honestly needs a slot sweep per
        // model per card, and every static rule this project has fitted to four models has had
        // to be retracted. A sweep beats this; --bells-slots N takes the result.
        // BELLS_VRAM_MB overrides the headroom rule with an explicit cache budget. The rule
        // below is calibrated for large cards: a flat 1 GiB floor is a sixth of a 6 GB card, so
        // it dominates whenever free VRAM drops under 3 GiB, which on a small card is most of
        // the time. Sweeping bytes is also more portable than sweeping slots, because
        // slots -> bytes depends on the model's expert size.
        size_t budget;
        size_t headroom = 0;

        if (const char * vm = getenv("BELLS_VRAM_MB")) {
            const long mb = atol(vm);
            budget = mb > 0 ? (size_t) mb*1024*1024 : 0;
            if (budget > dev_free) {
                fprintf(stderr, "%s: BELLS_VRAM_MB=%ld exceeds %.2f GiB free, clamping\n",
                        __func__, mb, dev_free/1024.0/1024.0/1024.0);
                budget = dev_free;
            }
        } else {
            headroom = std::max<size_t>(1024ull*1024*1024, dev_free/3);
            budget   = dev_free > headroom ? dev_free - headroom : 0;
        }

        n_slot = (uint32_t) std::min<size_t>(n_expert, budget/(per_expert*srcs.size()));

        fprintf(stderr, "%s: auto-sizing from %.1f GiB free, %.1f GiB headroom -> %u slots "
                        "(conservative, sweep --bells-slots to beat it)\n",
                __func__, dev_free/1024.0/1024.0/1024.0,
                headroom/1024.0/1024.0/1024.0, n_slot);

        // What other budgets would buy, so a sweep can be aimed rather than guessed. Capacity
        // is worth more than prediction on this workload: measured LRU hit rate goes 61.4% at
        // 32 slots to 78.1% at 64, while Belady (perfect prediction) at 32 is only 78.3%. So
        // doubling the cache matches a perfect oracle at the smaller size.
        {
            const size_t per_slot = per_expert*srcs.size();
            fprintf(stderr, "%s: budget -> slots:", __func__);
            for (size_t mb : { 512ull, 1024ull, 2048ull, 3072ull, 4096ull }) {
                const size_t b = mb*1024*1024;
                if (b <= dev_free) {
                    fprintf(stderr, "  %zuMB=%zu", mb, std::min<size_t>(n_expert, b/per_slot));
                }
            }
            fprintf(stderr, "   (BELLS_VRAM_MB=N to pick one)\n");
        }

        // Auto-sizing is not merely suboptimal when it comes out thin - it can be slower than not
        // using BELLS at all, because the per-layer readback is charged on every layer of every
        // token whether or not the cache saves anything. Measured on Qwen3-30B-A3B Q4_K_M on a
        // 6 GB card: --bells picked 17 slots and ran at 11.07 tok/s against an 11.37 tok/s
        // --cpu-moe baseline, while an explicit 26 slots reached 13.75. Below roughly 3x
        // n_expert_used there is not enough reuse to pay for the round-trip, so say so.
        if (n_expert_used > 0 && n_slot < 3*n_expert_used) {
            fprintf(stderr, "%s: WARNING %u slots is only %.1fx n_expert_used (%u). BELLS is "
                            "likely to be SLOWER than plain --cpu-moe at this size. Raise "
                            "--bells-slots, or drop BELLS and try partial expert offload "
                            "(-ot 'blk\\.(0|1|...)\\.ffn_.*_exps\\.weight=CPU') instead.\n",
                    __func__, n_slot, (double) n_slot/n_expert_used, n_expert_used);
        }
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

    if (!tensors_.init(buft, srcs, n_slot, backend, copy_backend)) {
        fprintf(stderr, "%s: failed to allocate %u expert slots per layer\n", __func__, n_slot);
        return false;
    }

    uint32_t n_layer = 0;
    for (const auto & s : srcs) {
        n_layer = std::max(n_layer, (uint32_t) s.il + 1);
    }

    cache_.init(n_layer, n_expert, n_slot);

    // --pin-experts: seat the measured hot set permanently.
    //
    // Dynamic admission at this cache size does not work: 64 slots of 512 experts measured
    // 5.06 tok/s against a 5.62 baseline, because a cold token evicts an expert that the next
    // few tokens want back. The offline histogram says those same 64 slots can cover 53.9% of
    // routing if chosen by frequency and left alone, against 12.6% for the equivalent VRAM spent
    // on whole layers via -ot.
    // A static layer's table is written once and never updated, so there has to be something
    // meaningful to write. A zeroed table maps every expert to slot 0, which yields confident
    // nonsense rather than an error - refuse instead.
    if (params_.refresh > 1 && params_.pin_file.empty()) {
        fprintf(stderr, "%s: --bells-refresh %u needs --pin-experts: layers that are never "
                        "observed require a permanent slot table\n", __func__, params_.refresh);
        return false;
    }

    uint32_t n_pinned = 0;
    if (!params_.pin_file.empty()) {
        std::vector<std::vector<std::pair<uint64_t, int32_t>>> hot(n_layer);

        FILE * f = fopen(params_.pin_file.c_str(), "r");
        if (!f) {
            fprintf(stderr, "%s: cannot read --pin-experts file %s\n", __func__, params_.pin_file.c_str());
        } else {
            char line[256];
            bool first = true;
            uint64_t n_rows = 0;
            while (fgets(line, sizeof(line), f)) {
                if (first) {           // header
                    first = false;
                    if (line[0] < '0' || line[0] > '9') {
                        continue;
                    }
                }
                unsigned il = 0, e = 0;
                unsigned long long cnt = 0;
                double wsum = 0.0;

                // Rank by summed routing weight when the column is present. Dropping an expert
                // removes its weighted contribution, so weight is what should be minimised;
                // occurrence count is the fallback for CSVs written before this existed.
                const int got = sscanf(line, "%u,%u,%llu,%lf", &il, &e, &cnt, &wsum);
                if (got >= 3 && il < n_layer && e < n_expert) {
                    const uint64_t key = (got >= 4 && wsum > 0.0)
                                       ? (uint64_t) (wsum*1000.0)
                                       : (uint64_t) cnt;
                    hot[il].emplace_back(key, (int32_t) e);
                    n_rows++;
                }
            }
            fclose(f);

            // Keep some slots dynamic. The residency invariant still has to hold: a token can
            // route to n_expert_used experts that are not in the pinned set, and ensure() has no
            // way to fail safely mid-graph.
            uint32_t reserve = params_.pin_reserve ? params_.pin_reserve
                                                   : std::max(4u*n_expert_used, 16u);
            if (reserve >= n_slot) {
                reserve = std::max(1u, n_slot/2);
            }
            // Dynamic layers keep a reserve so LRU can still admit what a token actually routes
            // to. Static layers never get another chance to admit anything, so every slot they
            // have should hold a measured-hot expert.
            const uint32_t n_pin_dynamic = n_slot - reserve;
            const uint32_t n_pin_static  = n_slot;

            // only layers the cache actually holds
            std::vector<uint8_t> is_moe(n_layer, 0);
            for (const auto & s : srcs) {
                if (s.il >= 0 && (uint32_t) s.il < n_layer) {
                    is_moe[s.il] = 1;
                }
            }

            std::vector<bells_copy> pin_copies;
            uint64_t n_copies = 0;
            for (uint32_t il = 0; il < n_layer; ++il) {
                if (!is_moe[il] || hot[il].empty()) {
                    continue;
                }
                auto & v = hot[il];
                const uint32_t n_pin = is_static_layer(il) ? n_pin_static : n_pin_dynamic;
                const size_t keep = std::min<size_t>(n_pin, v.size());
                std::partial_sort(v.begin(), v.begin() + keep, v.end(),
                                  [](const std::pair<uint64_t, int32_t> & a,
                                     const std::pair<uint64_t, int32_t> & b) {
                                      return a.first > b.first;
                                  });

                std::vector<int32_t> take;
                take.reserve(keep);
                for (size_t i = 0; i < keep; ++i) {
                    take.push_back(v[i].second);
                }

                pin_copies.clear();
                cache_.pin_experts(il, take, pin_copies);
                for (const auto & pc : pin_copies) {
                    tensors_.copy_expert(il, pc.expert, pc.slot);
                    n_copies++;
                }
                tensors_.upload_slots(il, cache_.slot_table(il));
            }
            tensors_.sync_copies();

            // Only dynamic layers are ever ensure()d, so only THEIR free slots bound the ubatch.
            // cache_.n_pinned() is the max across all layers, which for a fully-pinned static
            // layer is n_slot - using that made the bound nonsense.
            n_pinned = std::min(n_pin_dynamic, n_slot);
            uint32_t n_static = 0, n_dyn = 0;
            for (uint32_t il = 0; il < n_layer; ++il) {
                if (!is_moe[il]) {
                    continue;
                }
                if (is_static_layer(il)) { n_static++; } else { n_dyn++; }
            }
            fprintf(stderr, "%s: pinned up to %u of %u slots/layer from %llu rows "
                            "(%llu experts copied); %u layers dynamic, %u layers static "
                            "(never observed, no graph split)\n",
                    __func__, n_pinned, n_slot, (unsigned long long) n_rows,
                    (unsigned long long) n_copies, n_dyn, n_static);
        }
    }

    if (params_.max_tokens == 0) {
        // n tokens may request up to n * n_expert_used distinct experts; anything beyond what
        // the cache holds could fail ensure() mid-graph, where there is no way to recover.
        //
        // Pinned slots cannot absorb a miss, so the bound is over the free pool only.
        //
        // The previous fallback returned n_slot when the cache was fully pinned - the opposite of
        // the truth. It let BELLS accept 9-token ubatches while a dynamic layer had 32 free slots
        // against a worst case of 9*8 = 72 distinct experts, and the resulting ensure() failure
        // faulted in CUDA. Zero free slots means single-token only, not unlimited.
        const uint32_t free_slots = n_slot > n_pinned ? n_slot - n_pinned : 0;
        params_.max_tokens = std::max(1u, free_slots/std::max(1u, n_expert_used));
    }

    // Confidence-gated prefetch, off unless a table is supplied. BELLS_CONF sets the threshold;
    // 0.9 is where offline measurement put precision at 73.9%.
    if (const char * tp = getenv("BELLS_TABLE")) {
        if (tp[0]) {
            if (const char * ct = getenv("BELLS_CONF")) {
                conf_thresh_ = (float) atof(ct);
            }
            if (conf_.load(tp)) {
                // Prefetch is now opt-in, so lookahead eviction can be measured without it.
                // It has never produced a speedup: copies share the compute stream, so moving
                // a transfer earlier does not let it overlap anything.
                if (const char * pf = getenv("BELLS_PREFETCH")) {
                    if (pf[0] && pf[0] != '0') {
                        pf_pending_.assign(n_layer, 0);
                        pf_start();
                        fprintf(stderr, "%s: confidence prefetch on, threshold %.2f\n",
                                __func__, conf_thresh_);
                    }
                }

                // Lookahead eviction: protect what the next K tokens are predicted to want.
                if (const char * la = getenv("BELLS_LOOKAHEAD")) {
                    const uint32_t k = (uint32_t) atoi(la);
                    const char * fp  = getenv("BELLS_FUTURE");

                    if (k > 0 && fp && fp[0]) {
                        std::ifstream ff(fp, std::ios::binary);
                        if (ff) {
                            ff.seekg(0, std::ios::end);
                            const size_t n_tok = (size_t) ff.tellg()/sizeof(int32_t);
                            ff.seekg(0, std::ios::beg);
                            future_.resize(n_tok);
                            ff.read((char *) future_.data(), (std::streamsize) n_tok*sizeof(int32_t));
                        }

                        if (!future_.empty()) {
                            lookahead_ = k;
                            protect_.assign(n_layer, std::vector<uint8_t>(n_expert, 0));
                            fprintf(stderr, "%s: lookahead eviction on, K=%u, %zu future tokens\n",
                                    __func__, k, future_.size());
                        } else {
                            fprintf(stderr, "%s: could not read future tokens from '%s'\n",
                                    __func__, fp);
                        }
                    }
                }

                fprintf(stderr, "%s: confidence table %s loaded\n", __func__, tp);
            } else {
                fprintf(stderr, "%s: could not load confidence table '%s', prefetch off\n",
                        __func__, tp);
            }
        }
    }

    params_.n_slot = n_slot;
    ready_         = true;
    n_copied_      = 0;
    n_prefetched_  = 0;
    n_pf_used_     = 0;

    const double vram_gib  = tensors_.vram_bytes()/1024.0/1024.0/1024.0;
    const double vram_frac = dev_total > 0 ? (double) tensors_.vram_bytes()/dev_total : 0.0;

    fprintf(stderr, "%s: %u slots/layer of %u experts, %.2f GiB VRAM (%.0f%% of the card), "
                    "serves ubatch <= %u%s\n",
            __func__, n_slot, n_expert, vram_gib, 100.0*vram_frac,
            params_.max_tokens,
            params_.passive ? "  [PASSIVE: cache allocated but unused, research mode]" : "");

    // Measured danger zone. The cache competes with the compute buffers, and past roughly
    // four fifths of the card that competition dominates: on a 24 GB card the 235B fell to
    // 1.02x at 19.9 GB of cache (83%) and 0.73x at 22.4 GB (93%), having peaked at 17.4 GB.
    // Only a warning, since the exact threshold is model dependent and this is one data point.
    if (vram_frac > 0.75) {
        fprintf(stderr, "%s: that is most of the card, and past ~80%% the cache starts starving "
                        "the compute buffers - measured 1.02x at 83%% and 0.73x at 93%% on one "
                        "model. Try fewer slots if it is slow.\n", __func__);
    }

    return true;
}

void bells_runtime::free() {
    pf_stop();

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
    if (n_prefetched_ > 0) {
        fprintf(stderr, "%s: prefetched %llu experts, %.2f GiB speculative\n", __func__,
                (unsigned long long) n_prefetched_,
                n_prefetched_*(double) tensors_.bytes_per_expert()/1024.0/1024.0/1024.0);
    }

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

void bells_runtime::pf_start() {
    if (pf_enabled_) {
        return;
    }

    pf_quit_    = false;
    pf_enabled_ = true;

    pf_thread_ = std::thread([this]() {
        for (;;) {
            pf_job job;
            {
                std::unique_lock<std::mutex> lk(pf_mutex_);
                pf_cv_.wait(lk, [this]() { return pf_quit_ || !pf_queue_.empty(); });
                if (pf_quit_ && pf_queue_.empty()) {
                    return;
                }
                job = pf_queue_.front();
                pf_queue_.erase(pf_queue_.begin());
            }

            // The only work handed to this thread. Cache bookkeeping already happened on the
            // main thread, so this touches nothing bells_cache owns.
            tensors_.copy_expert(job.il, job.copy.expert, job.copy.slot);

            {
                std::lock_guard<std::mutex> lk(pf_mutex_);
                if (job.il < pf_pending_.size() && pf_pending_[job.il] > 0) {
                    pf_pending_[job.il]--;
                }
            }
            pf_done_.notify_all();
        }
    });
}

void bells_runtime::pf_stop() {
    if (!pf_enabled_) {
        return;
    }
    {
        std::lock_guard<std::mutex> lk(pf_mutex_);
        pf_quit_ = true;
    }
    pf_cv_.notify_all();
    if (pf_thread_.joinable()) {
        pf_thread_.join();
    }
    pf_enabled_ = false;
    pf_queue_.clear();
}

void bells_runtime::pf_submit(uint32_t il, const bells_copy & c) {
    {
        std::lock_guard<std::mutex> lk(pf_mutex_);
        if (il >= pf_pending_.size()) {
            pf_pending_.resize(il + 1, 0);
        }
        pf_pending_[il]++;
        pf_queue_.push_back({ il, c });
    }
    pf_cv_.notify_one();
}

void bells_runtime::pf_drain(uint32_t il) {
    std::unique_lock<std::mutex> lk(pf_mutex_);
    pf_done_.wait(lk, [this, il]() {
        return il >= pf_pending_.size() || pf_pending_[il] == 0;
    });
}

void bells_runtime::build_protect(int32_t pos) {
    // Union of the experts the next K tokens are predicted to want, per layer. Confidence is
    // deliberately not thresholded here: for eviction a weak signal is still better than the
    // nothing LRU has, and a wrong entry only costs a slightly worse victim choice.
    for (int32_t il : tensors_.layers()) {
        if ((size_t) il >= protect_.size()) {
            continue;
        }
        std::fill(protect_[il].begin(), protect_[il].end(), (uint8_t) 0);
    }

    for (uint32_t k = 1; k <= lookahead_; ++k) {
        const size_t p = (size_t) pos + k;
        if (p >= future_.size()) {
            break;
        }

        const int32_t tok = future_[p];

        for (int32_t il : tensors_.layers()) {
            if ((size_t) il >= protect_.size()) {
                continue;
            }

            const float * cf = nullptr;
            const int32_t * ex = conf_.predict(tok, (uint32_t) il, &cf);
            if (!ex) {
                continue;
            }

            for (uint32_t j = 0; j < conf_.max_k(); ++j) {
                if (ex[j] < 0) {
                    break;
                }
                if ((size_t) ex[j] < protect_[il].size()) {
                    protect_[il][ex[j]] = 1;
                    n_protect_hits_++;
                }
            }
        }
    }
}

void bells_runtime::begin_ubatch(int32_t token, int32_t pos, int64_t n_tokens) {
    // drives the --bells-refresh rotation; counted per ubatch, which is per token during decode
    n_tok_seen_++;

    active_now_ = active(n_tokens);

    if (!active_now_ || !conf_.enabled() || token < 0) {
        return;
    }

    if (lookahead_ > 0 && pos >= 0 && !future_.empty()) {
        build_protect(pos);
    }

    if (!pf_enabled_) {
        return;   // lookahead eviction only; prefetch is a separate opt-in
    }

    // Prefetch every layer here, before the graph runs. The token id is known now and layer 47's
    // experts are not needed for another 47 layers, so there is ample lead time - the reason the
    // original predictor failed was bandwidth, not warning.
    //
    // Only candidates above the confidence threshold are fetched. That is the whole difference
    // from the removed design, which took the top N regardless and therefore moved a superset.
    copies_.clear();

    for (int32_t il : tensors_.layers()) {
        const float * cf = nullptr;
        const int32_t * ex = conf_.predict(token, (uint32_t) il, &cf);
        if (!ex || !cf) {
            continue;
        }

        int32_t cand[32];
        uint32_t n = 0;
        for (uint32_t k = 0; k < conf_.max_k() && n < 32; ++k) {
            if (ex[k] < 0) {
                break;
            }
            if (cf[k] >= conf_thresh_) {
                cand[n++] = ex[k];
            }
        }

        if (n == 0) {
            continue;
        }

        copies_.clear();
        cache_.prefetch((uint32_t) il, cand, n, copies_);

        // Hand the bytes to the worker and move on. Doing this inline is what made prefetch a
        // serial prologue: a copy from pageable memory blocks the caller while the driver
        // stages it, so the token could not start until every layer's guesses had landed.
        for (const auto & c : copies_) {
            pf_submit((uint32_t) il, c);
        }

        n_copied_     += copies_.size();
        n_prefetched_ += copies_.size();
    }
}

bool bells_runtime::on_routing(uint32_t il, const int32_t * experts, size_t n) {
    // prefill bypasses the cache in the graph, so it must not touch residency here either
    if (!ready_ || !active_now_) {
        return true;
    }

    n_layer_calls_++;

    // Report the breakdown periodically, not only from free().
    //
    // free() runs from the destructor, and a force-kill skips destructors - which is why these
    // numbers had never actually been seen despite being collected all along. They are the ones
    // that decide whether cache tuning can help at all: readback and upload are paid every token
    // regardless of hit rate, only copy scales with misses.
    // 2000, not 20000: --bells-refresh N cuts layer-calls per token by N, so a 20000 threshold
    // needed 5000 tokens to fire at refresh 8 and the hit rate went unreported for every arm that
    // mattered.
    if ((n_layer_calls_ % 2000) == 0) {
        const double per = 1.0/(double) n_layer_calls_;
        fprintf(stderr, "bells_timing: per layer-call: readback %.1f us, copy %.1f us, "
                        "upload %.1f us, hit %.1f%% (%llu calls)\n",
                us_readback_*per, us_copy_*per, us_upload_*per,
                100.0*(double) cache_.n_hit()/std::max<uint64_t>(1, cache_.n_hit() + cache_.n_miss()),
                (unsigned long long) n_layer_calls_);
        fflush(stderr);
    }

    // Any speculative copy into this layer must have landed before we reuse its slots or read
    // them. This is the only synchronisation the background copier needs, and it is why cache
    // bookkeeping was kept on this thread.
    if (pf_enabled_) {
        pf_drain(il);
    }

    // Every expert this layer asked for must be resident before the matmul reads the slot
    // table, so this is a demand fetch with no lead time to hide it behind.
    const std::vector<uint8_t> * protect =
        (lookahead_ > 0 && il < protect_.size()) ? &protect_[il] : nullptr;

    copies_.clear();
    if (!cache_.ensure(il, experts, n, copies_, protect)) {
        return false;
    }

    // BELLS_SYNC_EVICT=1: before reusing a slot, wait for kernels already queued on the compute
    // backend. Eviction overwrites memory an in-flight matmul may still be reading, and that is
    // not covered by stream ordering when the write is issued off the compute stream. Suspected
    // cause of the sm_86 corruption, which only appears once the cache starts reusing slots.
    static const bool sync_evict = [] {
        const char * s = getenv("BELLS_SYNC_EVICT");
        return s && s[0] && s[0] != '0';
    }();

    if (sync_evict && !copies_.empty()) {
        tensors_.sync_compute();
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

    // Make the compute stream wait for this layer's copies before anything reads the slots.
    // With one stream this is a no-op and the ordering was implicit; with two it is the only
    // thing standing between the graph and a half-written expert.
    tensors_.sync_copies();

    tensors_.upload_slots(il, cache_.slot_table(il));

    // BELLS_MMAP_WARM: record routing, then hint the next layers while this one computes.
    if (tensors_.warm_enabled()) {
        tensors_.warm_note(il, experts, n);
        tensors_.warm_lookahead(il + 1);
    }


    // BELLS_VERIFY_RESIDENT=1 checks the whole invariant the matmul depends on, at the moment it
    // is about to run: for every expert this token routed to, the slot the table points at must
    // hold that expert's bytes. Copies verify individually and ids verify valid, yet output still
    // degrades once eviction starts - so this tests the combination rather than the parts.
    static const bool verify_res = [] {
        const char * s = getenv("BELLS_VERIFY_RESIDENT");
        return s && s[0] && s[0] != '0';
    }();

    if (verify_res) {
        static int n_reported = 0;
        static long n_tok     = 0;
        n_tok++;

        if (n_reported < 6) {
            const std::vector<int32_t> & tbl = cache_.slot_table(il);
            size_t bad = 0;
            int32_t bad_e = -1, bad_s = -1;

            for (size_t i = 0; i < n && bad == 0; ++i) {
                const int32_t e = experts[i];
                if (e < 0 || (size_t) e >= tbl.size()) {
                    continue;
                }
                const int32_t s = tbl[e];
                if (s < 0) {
                    bad++; bad_e = e; bad_s = s;
                    break;
                }
                if (!tensors_.slot_matches_expert(il, e, s)) {
                    bad++; bad_e = e; bad_s = s;
                }
            }

            if (bad > 0) {
                n_reported++;
                fprintf(stderr, "%s: *** RESIDENCY VIOLATION call %ld layer %u: expert %d -> slot %d "
                                "does not hold that expert's data\n",
                        __func__, n_tok, il, bad_e, bad_s);
            }
        }
    }

    const auto t_up1 = std::chrono::steady_clock::now();

    us_copy_   += std::chrono::duration_cast<std::chrono::microseconds>(t_copy1 - t_copy0).count();
    us_upload_ += std::chrono::duration_cast<std::chrono::microseconds>(t_up1  - t_copy1).count();

    return true;
}


//
// --cold-tensors
//

// Push a range out of the working set. Advisory on both platforms and safe on clean file-backed
// pages: the worst outcome is a re-fault from the file the pages came from.
static void bells_cold_range(void * addr, size_t bytes) {
#if defined(_WIN32)
    // VirtualUnlock on pages that were never locked returns FALSE with ERROR_NOT_LOCKED, but it
    // still removes them from the working set - that side effect is the whole point here, so the
    // return value is deliberately ignored. Chunked because the ranges run to tens of GB.
    const size_t chunk = (size_t) 256*1024*1024;
    for (size_t off = 0; off < bytes; off += chunk) {
        VirtualUnlock((char *) addr + off, std::min(chunk, bytes - off));
    }
#elif defined(__linux__)
  #if defined(MADV_COLD)
    // preferred: moves pages to the inactive list, so they are reclaimed first but not dropped
    if (madvise(addr, bytes, MADV_COLD) == 0) {
        return;
    }
  #endif
    madvise(addr, bytes, MADV_DONTNEED);
#else
    (void) addr; (void) bytes;
#endif
}

namespace {

class bells_cold {
public:
    void start(const std::vector<std::pair<void *, size_t>> & ranges, uint64_t bytes,
               const char * what) {
        stop();
        if (ranges.empty()) {
            return;
        }
        ranges_ = ranges;
        quit_   = false;
        sweeps_ = 0;
        thread_ = std::thread([this] { loop(); });

        fprintf(stderr, "%s: holding %.2f GiB of %s out of the working set\n",
                __func__, bytes/1024.0/1024.0/1024.0, what);
    }

    void stop() {
        if (!thread_.joinable()) {
            return;
        }
        quit_ = true;
        thread_.join();
        if (sweeps_ > 0) {
            fprintf(stderr, "bells_cold: %llu sweeps\n", (unsigned long long) sweeps_);
        }
        ranges_.clear();
    }

    ~bells_cold() { stop(); }

private:
    void loop() {
        // Slow and steady. The pages we are pushing out get faulted back in by ordinary use, so
        // the sweep only has to run often enough to stay ahead of that - fast enough and it burns
        // CPU evicting pages that are about to be read again.
        while (!quit_) {
            for (const auto & r : ranges_) {
                if (quit_) {
                    break;
                }
                bells_cold_range(r.first, r.second);
            }
            sweeps_++;
            for (int i = 0; i < 20 && !quit_; ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
    }

    std::vector<std::pair<void *, size_t>> ranges_;
    std::thread       thread_;
    std::atomic<bool> quit_{false};
    uint64_t          sweeps_ = 0;
};

bells_cold g_cold;

} // namespace

void bells_cold_start(const std::vector<std::pair<void *, size_t>> & ranges,
                      uint64_t bytes, const char * what) {
    g_cold.start(ranges, bytes, what);
}

void bells_cold_stop() {
    g_cold.stop();
}


//
// --moe-prefetch
//

void moe_prefetch::init(uint32_t n_layer, uint32_t n_expert, uint32_t keep) {
    keep_     = keep;
    n_expert_ = n_expert;
    layers_.assign(n_layer, layer{});
    used_.assign((size_t) n_layer * n_expert, 0);
    clock_   = 0;
    skip_    = 0;
    backoff_ = 1;
}

void moe_prefetch::add_layer(uint32_t il, ggml_tensor * gate, ggml_tensor * up,
                             ggml_tensor * down, ggml_tensor * gate_up) {
    if (il >= layers_.size()) {
        return;
    }
    layers_[il].t[0] = gate;
    layers_[il].t[1] = up;
    layers_[il].t[2] = down;
    layers_[il].t[3] = gate_up;
}

void moe_prefetch::note(uint32_t il, const int32_t * experts, size_t n) {
    if (!keep_ || n_expert_ == 0) {
        return;
    }
    std::lock_guard<std::mutex> lk(mu_);
    clock_++;
    for (size_t i = 0; i < n; ++i) {
        const int32_t e = experts[i];
        if (e < 0 || (uint32_t) e >= n_expert_) {
            continue;
        }
        const size_t idx = (size_t) il*n_expert_ + (size_t) e;
        if (idx < used_.size()) {
            used_[idx] = clock_;
        }
    }
}

void moe_prefetch::lookahead(uint32_t il_next) {
    if (!keep_ || n_expert_ == 0 || layers_.empty()) {
        return;
    }
    if (skip_ > 0) {
        skip_--;
        return;
    }

    std::vector<std::pair<void *, size_t>> ranges;
    std::vector<std::pair<uint64_t, uint32_t>> rank;

    // two layers of lookahead: enough lead time to cover a large read, small enough that the
    // request stays specific. Asking for more than the page cache can hold is how the first
    // version of this evicted the pages it was trying to help.
    for (uint32_t ahead = 0; ahead < 2; ++ahead) {
        const uint32_t il = il_next + ahead;
        if (il >= layers_.size()) {
            break;
        }

        rank.clear();
        {
            std::lock_guard<std::mutex> lk(mu_);
            const size_t base = (size_t) il*n_expert_;
            if (base + n_expert_ > used_.size()) {
                continue;
            }
            for (uint32_t x = 0; x < n_expert_; ++x) {
                if (used_[base + x] > 0) {
                    rank.emplace_back(used_[base + x], x);
                }
            }
        }
        if (rank.empty()) {
            continue;
        }

        const size_t keep = std::min<size_t>(keep_, rank.size());
        std::partial_sort(rank.begin(), rank.begin() + keep, rank.end(),
                          [](const std::pair<uint64_t, uint32_t> & a,
                             const std::pair<uint64_t, uint32_t> & b) {
                              return a.first > b.first;
                          });

        for (int k = 0; k < 4; ++k) {
            ggml_tensor * s = layers_[il].t[k];
            if (!s || !s->data || s->ne[2] <= 0) {
                continue;
            }
            // ne[2] indexes the expert, so one expert is a single contiguous extent of this
            // size - the whole point of the exercise. Asking for it in one request is what
            // turns ~31 readahead-sized faults into one.
            const size_t stride = ggml_nbytes(s)/s->ne[2];
            for (size_t r = 0; r < keep; ++r) {
                ranges.emplace_back((char *) s->data + (size_t) rank[r].second*stride, stride);
            }
        }
    }

    const size_t missing    = bells_warm_ranges(ranges);
    const size_t candidates = ranges.size();

    sweeps_++;
    hinted_ += missing;

    const bool idle      = (missing == 0);
    const bool saturated = (candidates > 0 && missing*2 > candidates);
    if (idle || saturated) {
        backoff_ = std::min<uint32_t>(backoff_*2, 256);
    } else {
        backoff_ = 1;
    }
    skip_ = backoff_ - 1;
}

void moe_prefetch::report() const {
    std::lock_guard<std::mutex> lk(mu_);
    if (!keep_) {
        return;
    }
    // Zero sweeps is a result, not a non-event: it means the hook never fired, which is a
    // different failure from "fired and found everything resident".
    fprintf(stderr, "moe_prefetch: %llu sweeps, %llu extents hinted (%.1f per sweep)\n",
            (unsigned long long) sweeps_, (unsigned long long) hinted_,
            sweeps_ ? (double) hinted_/(double) sweeps_ : 0.0);
    fflush(stderr);
}


//
// --moe-stats
//

void moe_stats::init(uint32_t n_layer, uint32_t n_expert, const char * path) {
    if (!path || !*path) {
        return;
    }
    n_layer_  = n_layer;
    n_expert_ = n_expert;
    count_.assign((size_t) n_layer * n_expert, 0);
    weight_.assign((size_t) n_layer * n_expert, 0.0);
    last_ids_.assign(n_layer, {});
    n_events_ = 0;
    path_      = path;
    last_dump_ = 0;

    fprintf(stderr, "moe_stats: counting expert usage over %u layers x %u experts -> %s\n",
            n_layer, n_expert, path);
    fflush(stderr);
}

void moe_stats::note(uint32_t il, const int32_t * ids, size_t n) {
    if (path_.empty() || n_expert_ == 0) {
        return;
    }
    std::lock_guard<std::mutex> lk(mu_);

    // Stash for note_weights. The graph emits ffn_moe_topk before ffn_moe_weights_norm for the
    // same layer, and both carry one entry per (expert_used, token) in the same order, so the
    // pairing is positional and matched on length.
    if (il < last_ids_.size()) {
        last_ids_[il].assign(ids, ids + n);
    }
    for (size_t i = 0; i < n; ++i) {
        const int32_t e = ids[i];
        if (e < 0 || (uint32_t) e >= n_expert_) {
            continue;
        }
        const size_t idx = (size_t) il*n_expert_ + (size_t) e;
        if (idx < count_.size()) {
            count_[idx]++;
            n_events_++;
        }
    }

    // Snapshot periodically rather than trusting the destructor. A force-kill skips destructors
    // entirely, which is how the expert cache's own stats went missing earlier in this project -
    // the run completes, the numbers do not survive it.
    if (n_events_ - last_dump_ >= 100000) {
        last_dump_ = n_events_;
        write_locked();
    }
}

void moe_stats::note_weights(uint32_t il, const float * w, size_t n) {
    if (path_.empty() || n_expert_ == 0 || il >= n_layer_) {
        return;
    }
    std::lock_guard<std::mutex> lk(mu_);

    // Pair only when the two tensors describe the same set of picks. A length mismatch means the
    // topk node arrived at a different width and the positional pairing would be meaningless.
    if (il >= last_ids_.size() || last_ids_[il].size() != n) {
        return;
    }
    for (size_t i = 0; i < n; ++i) {
        const int32_t e = last_ids_[il][i];
        if (e < 0 || (uint32_t) e >= n_expert_) {
            continue;
        }
        const size_t idx = (size_t) il*n_expert_ + (size_t) e;
        if (idx < weight_.size()) {
            weight_[idx] += (double) w[i];
        }
    }
}

void moe_stats::dump() {
    std::lock_guard<std::mutex> lk(mu_);
    write_locked();
}

// caller holds mu_
void moe_stats::write_locked() {
    if (path_.empty()) {
        return;
    }

    FILE * f = fopen(path_.c_str(), "w");
    if (!f) {
        fprintf(stderr, "moe_stats: cannot write %s\n", path_.c_str());
        return;
    }
    fprintf(f, "layer,expert,count,weight\n");
    for (uint32_t il = 0; il < n_layer_; ++il) {
        for (uint32_t e = 0; e < n_expert_; ++e) {
            const size_t   idx = (size_t) il*n_expert_ + e;
            const uint64_t v   = count_[idx];
            if (v) {
                fprintf(f, "%u,%u,%llu,%.6f\n", il, e, (unsigned long long) v,
                        idx < weight_.size() ? weight_[idx] : 0.0);
            }
        }
    }
    fclose(f);
    fprintf(stderr, "moe_stats: %llu routing events written to %s\n",
            (unsigned long long) n_events_, path_.c_str());
    fflush(stderr);
}
