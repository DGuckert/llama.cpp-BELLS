#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <string>
#include <utility>
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

    // Speculative: admit as many of `experts` as fit, never evicting anything in `experts`.
    // Misses here are harmless, they just cost a later ensure().
    void prefetch(uint32_t il, const int32_t * experts, size_t n, std::vector<bells_copy> & out);

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

    // Touch the source pages for a set of experts across worker threads.
    //
    // When the model is larger than RAM these reads come off disk, and faulting them one at
    // a time leaves the NVMe at queue depth 1, which costs most of its throughput. Issuing
    // them in parallel first means the later copies hit warm pages.
    void prefault(const std::vector<std::pair<uint32_t, int32_t>> & experts, int n_threads) const;

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

// token id -> per-layer expert ranking, built by counting over a routing trace.
//
// Measured on OLMoE and Qwen3, the token id alone determines roughly 68-77% of the expert
// set, which is enough lead time to prefetch: the id is known before layer 0 runs.
class bells_predictor {
public:
    bool load(const std::string & path);

    bool enabled() const { return n_layer_ > 0; }

    uint32_t n_take() const { return n_take_; }

    // ranked experts for (token, layer); falls back to the global prior for unseen tokens
    const int32_t * predict(int32_t token, uint32_t il) const;

    bool in_table(int32_t token) const;

private:
    uint32_t n_layer_ = 0;
    uint32_t n_take_  = 0;

    std::vector<int32_t> tokens_;   // sorted, for binary search
    std::vector<int32_t> ranked_;   // n_token * n_layer * n_take
    std::vector<int32_t> fallback_; // n_layer * n_take, global prior
};

struct bells_params {
    bool        enabled    = false;
    uint32_t    n_slot     = 0;   // experts resident per layer
    uint32_t    n_prefetch = 0;   // per-layer prefetch budget, 0 = n_slot/2
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

    // When true, an expert that is not resident contributes nothing instead of being fetched
    // on demand. This removes the per-layer host round-trip entirely, which measurements show
    // is the dominant cost, at the price of exact output.
    bool        drop_missing = false;

    // Threads used to fault source pages in before copying. Only matters when the model is
    // larger than RAM; 0 disables. Exists to give the NVMe enough queue depth.
    int         n_fault_threads = 8;
    std::string table;            // predictor table, optional
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
    bool active(int64_t n_tokens) const {
        return ready_ && n_tokens <= (int64_t) params_.max_tokens;
    }

    const bells_tensors & tensors() const { return tensors_; }

    // Called once per ubatch before it is computed. The token id is known before layer 0
    // runs, which is the source of the prefetch lead time. n_tokens decides whether this
    // ubatch uses the cache at all, and gates on_routing so prefill does not churn it.
    void begin_ubatch(int32_t token, int64_t n_tokens);

    // called once layer il's router has produced its selection; guarantees residency.
    // Not used in drop_missing mode, where nothing needs correcting.
    bool on_routing(uint32_t il, const int32_t * experts, size_t n);

    // whether the graph needs the residency correction callback at all
    bool needs_correction() const { return ready_ && !params_.drop_missing; }

    uint64_t n_hit()       const { return cache_.n_hit();  }
    uint64_t n_miss()      const { return cache_.n_miss(); }
    uint64_t n_copied()    const { return n_copied_;       }
    uint64_t bytes_moved() const { return n_copied_*(uint64_t) tensors_.bytes_per_expert(); }

private:
    bells_params    params_;
    bells_cache     cache_;
    bells_tensors   tensors_;
    bells_predictor predictor_;

    struct pending_copy {
        uint32_t   il;
        bells_copy copy;
    };

    std::vector<bells_copy>   copies_;
    std::vector<pending_copy> pending_;
    std::vector<std::pair<uint32_t, int32_t>> faults_;
    std::vector<int32_t>      table_;   // slot table with misses pointed at the zero slot

    int32_t  token_      = -1;
    bool     ready_      = false;
    bool     active_now_ = false;
    uint64_t n_copied_   = 0;
};
