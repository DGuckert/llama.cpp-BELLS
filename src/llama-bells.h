#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <vector>

// BELLS: a per-layer VRAM expert cache.
//
// Each MoE layer keeps n_slot experts resident out of n_expert. Routing ids are rewritten
// through slot_table() inside the graph, so mul_mat_id runs over the small cache tensor
// instead of the full expert stack.
//
// Two invariants the graph depends on, both enforced here:
//   - the expert -> slot mapping is injective. Two experts sharing a slot would produce
//     duplicate ids within a token, which the CUDA mul_mat_id kernel cannot represent.
//   - after ensure(), every requested expert is resident. A miss leaves slot -1, which
//     indexes out of bounds. There is no safe sentinel.

struct bells_copy {
    int32_t expert;
    int32_t slot;
};

class bells_cache {
public:
    void init(uint32_t n_layer, uint32_t n_expert, uint32_t n_slot);

    bool enabled() const { return n_slot_ > 0; }

    uint32_t n_slot()   const { return n_slot_;   }
    uint32_t n_expert() const { return n_expert_; }

    // expert -> slot for layer il, -1 where not resident. Uploaded to the graph each token.
    const std::vector<int32_t> & slot_table(uint32_t il) const { return layers_[il].expert_slot; }

    // Mandatory: after this every expert in `experts` is resident, or it returns false
    // because the request exceeds n_slot and cannot be satisfied at all.
    bool ensure(uint32_t il, const int32_t * experts, size_t n, std::vector<bells_copy> & out);

    void reset();

    uint64_t n_hit()  const { return n_hit_;  }
    uint64_t n_miss() const { return n_miss_; }

private:
    struct layer {
        std::vector<int32_t> expert_slot;  // n_expert, -1 if not resident
        std::vector<int32_t> slot_expert;  // n_slot,   -1 if empty
        std::vector<int64_t> last_used;    // n_slot
    };

    // slot to evict, preferring empty, then least recently used, skipping anything pinned
    int32_t victim(layer & l, const int32_t * keep, size_t n_keep) const;

    uint32_t n_layer_  = 0;
    uint32_t n_expert_ = 0;
    uint32_t n_slot_   = 0;

    int64_t clock_ = 0;

    uint64_t n_hit_  = 0;
    uint64_t n_miss_ = 0;

    std::vector<layer> layers_;
};

// Owns the VRAM cache tensors and moves expert weights into them.
//
// The cache tensors are persistent, not graph tensors: they outlive any single graph and
// are referenced by build_moe_ffn the same way the full expert tensors would be. Source
// weights are read straight out of the model's (host-resident) expert tensors, so a copy
// is a strided read of one expert followed by a write into its slot.
class bells_tensors {
public:
    // one entry per MoE layer; any of the three may be null if the arch does not use it
    struct layer_src {
        ggml_tensor * gate    = nullptr;
        ggml_tensor * up      = nullptr;
        ggml_tensor * down    = nullptr;
        ggml_tensor * gate_up = nullptr;
        int32_t       il      = -1;
    };

    // `backend`, when given, lets writes go out asynchronously on the compute stream instead
    // of stalling the pipeline. Ordering against the graph still holds because it is the
    // same stream the graph runs on.
    ~bells_tensors() { free(); }

    bool init(ggml_backend_buffer_type_t buft, const std::vector<layer_src> & srcs,
              uint32_t n_slot, ggml_backend_t backend = nullptr);

    void free();

    bool ready() const { return buffer_ != nullptr; }

    size_t vram_bytes() const { return vram_bytes_; }

    // cache tensors for the layer, indexed by model layer id
    ggml_tensor * gate(uint32_t il)    const { return get(il).gate;    }
    ggml_tensor * up(uint32_t il)      const { return get(il).up;      }
    ggml_tensor * down(uint32_t il)    const { return get(il).down;    }
    ggml_tensor * gate_up(uint32_t il) const { return get(il).gate_up; }
    ggml_tensor * slots(uint32_t il)   const { return get(il).slots;   }

    bool has(uint32_t il) const { return il < index_.size() && index_[il] >= 0; }

    // model layer ids that carry experts, in order
    const std::vector<int32_t> & layers() const { return layer_ids_; }

    // copy one expert from the model tensors into its slot
    void copy_expert(uint32_t il, int32_t expert, int32_t slot);

    // push the current expert->slot mapping to the device
    void upload_slots(uint32_t il, const std::vector<int32_t> & table);

    // Index of the spare slot holding zeros. Routing a non-resident expert here makes it
    // contribute nothing instead of indexing out of bounds, which is what lets the graph
    // run without stopping to ask the host which experts were selected.
    int32_t zero_slot() const { return (int32_t) n_slot_; }

    // bytes moved per expert, for budgeting
    size_t bytes_per_expert() const { return bytes_per_expert_; }

private:
    struct entry {
        ggml_tensor * gate    = nullptr;
        ggml_tensor * up      = nullptr;
        ggml_tensor * down    = nullptr;
        ggml_tensor * gate_up = nullptr;
        ggml_tensor * slots   = nullptr;
        layer_src     src;
    };

    static const entry & empty();

    const entry & get(uint32_t il) const {
        return has(il) ? entries_[index_[il]] : empty();
    }

    entry & get_mut(uint32_t il) { return entries_[index_[il]]; }

    void copy_one(ggml_tensor * dst, ggml_tensor * src, int32_t expert, int32_t slot) const;

    ggml_context          *      ctx_     = nullptr;
    ggml_backend_buffer_t        buffer_  = nullptr;
    ggml_backend_t               backend_ = nullptr;

    std::vector<entry>   entries_;
    std::vector<int32_t> index_;      // model layer -> entries_ index, -1 if not MoE
    std::vector<int32_t> layer_ids_;  // model layer ids carrying experts

    size_t   vram_bytes_       = 0;
    size_t   bytes_per_expert_ = 0;
    uint32_t n_slot_           = 0;
};

// There was a bells_predictor here: a token id -> per-layer expert ranking, counted over a
// routing trace and used to prefetch before layer 0 ran. It worked, in the sense that it
// predicted 68-77% of the expert set against LRU's 35%, and it lost every benchmark it was
// in. Hit rate was the wrong objective. Demand loading moves exactly the experts a token
// needs; a predictor moves a superset, and on a bandwidth-bound path the extra traffic costs
// more than the extra hits save.
//
// It is deleted rather than left switched off because a flag that never helps is a trap for
// whoever reads this next. tools/bells-profile/bells_predict.py still scores prediction
// against LRU and Belady offline, which is where that question belongs. Git history has the
// runtime if anyone wants to revisit it with a cheaper transfer path.
//
// A drop_missing mode went with it: a non-resident expert contributed nothing instead of being
// fetched, which removed the per-layer host sync and was the fastest thing this project ever
// measured. It was also a broken model - perplexity 52.97 against a baseline of 2.03, with
// generations collapsing into repetition loops. It could not have survived anyway, since
// prefetch was the only way an expert ever became resident in that mode.

struct bells_params {
    bool        enabled    = false;
    uint32_t    n_slot     = 0;   // experts resident per layer
    // Largest ubatch BELLS will serve. 0 means derive it from the cache size.
    //
    // n tokens can request up to n * n_expert_used distinct experts, and a graph already under
    // way cannot fall back if they do not fit, so the safe bound is n_slot / n_expert_used.
    // Prefill batches are far larger than that and bypass the cache, which is correct: a
    // 512-token batch touches nearly every expert and there is no hot set to exploit.
    //
    // This matters for llama-server, which defaults to n_parallel 4. With max_tokens pinned
    // at 1, BELLS silently did nothing as soon as two requests decoded together.
    uint32_t    max_tokens = 0;

    // Research only. Allocate the cache and keep taking the per-layer graph split and id
    // readback, but leave the matmuls pointing at the full expert stack and copy nothing.
    //
    // Exists to decompose a result nothing else explains: Mixtral with every expert resident
    // hits 100%, copies nothing, and still measures 0.80x. Passive isolates what that costs -
    // VRAM occupancy plus the graph split plus the readback - from anything the cache does.
    //   baseline vs passive = the fixed price of the mechanism
    //   passive vs BELLS    = what the cache is actually worth
    bool        passive = false;
};

// Ties the pieces together for the inference path.
//
// Deliberately knows nothing about llama_model: the caller hands it the expert tensors, so
// this stays unit-testable and free of circular includes.
class bells_runtime {
public:
    bool init(const bells_params & params,
              ggml_backend_buffer_type_t buft,
              const std::vector<bells_tensors::layer_src> & srcs,
              uint32_t n_expert,
              uint32_t n_expert_used,
              ggml_backend_t backend = nullptr);

    ~bells_runtime() { free(); }

    void free();

    bool ready() const { return ready_; }

    // BELLS only serves decode. A prefill ubatch touches nearly every expert, so there is no
    // hot set to exploit and a cache would only thrash.
    //
    // False in passive mode, so the graph keeps using the full expert stack while everything
    // else - the allocation, the split, the readback - still happens.
    bool active(int64_t n_tokens) const {
        return ready_ && !params_.passive && n_tokens <= (int64_t) params_.max_tokens;
    }

    bool passive() const { return params_.passive; }

    const bells_tensors & tensors() const { return tensors_; }

    // Called once per ubatch before it is computed. n_tokens decides whether this ubatch uses
    // the cache at all, and gates on_routing so prefill does not churn it.
    void begin_ubatch(int64_t n_tokens);

    // called once layer il's router has produced its selection; guarantees residency
    bool on_routing(uint32_t il, const int32_t * experts, size_t n);

    uint64_t n_hit()       const { return cache_.n_hit();  }
    uint64_t n_miss()      const { return cache_.n_miss(); }
    uint64_t n_copied()    const { return n_copied_;       }
    uint64_t bytes_moved() const { return n_copied_*(uint64_t) tensors_.bytes_per_expert(); }

    // Cost accounting for the per-layer host round trip, in microseconds.
    //
    // Mixtral holding every expert in VRAM - 100% hit, zero copies - still measured 0.80x, so
    // something other than transfers bounds this design. These separate the three candidates:
    // reading the router's ids back to the host, copying missing experts in, and uploading the
    // slot table. Only the copies scale with miss rate; the other two are paid every layer of
    // every token no matter how good the cache is.
    void add_readback_us(uint64_t us) { us_readback_ += us; }

    uint64_t us_readback() const { return us_readback_; }
    uint64_t us_copy()     const { return us_copy_;     }
    uint64_t us_upload()   const { return us_upload_;   }
    uint64_t n_layer_calls() const { return n_layer_calls_; }

private:
    bells_params  params_;
    bells_cache   cache_;
    bells_tensors tensors_;

    std::vector<bells_copy> copies_;

    bool     ready_      = false;
    bool     active_now_ = false;
    uint64_t n_copied_   = 0;

    uint64_t us_readback_   = 0;
    uint64_t us_copy_       = 0;
    uint64_t us_upload_     = 0;
    uint64_t n_layer_calls_ = 0;
};
