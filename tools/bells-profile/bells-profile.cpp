#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>     // exp/log in the perplexity path; MSVC pulls these in transitively, gcc does not
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

// Records the per-layer expert selection trace, then brackets what an expert cache can do:
//
//   LRU    - reactive policy, no knowledge of the future. The floor.
//   Belady - evicts the expert whose next use is furthest away, i.e. perfect prediction.
//            Unreachable in practice, but no predictor can beat it. The ceiling.
//
// The gap between them is the headroom a predictor is competing for. If Belady is already
// poor at an affordable capacity, no predictor rescues the design.
//
// The router tags two tensors per layer inside build_moe_ffn:
//   ffn_moe_argsort-<il> : [n_expert, n_tokens] I32, full descending ranking
//   ffn_moe_topk-<il>    : [n_expert_used, n_tokens] I32, a view of the first k columns
// topk is the authoritative selection but is a non-contiguous view, and archs that pass a
// precomputed selection never emit an argsort node. Tally both, prefer topk.

static const double BELLS_CACHE_FRACTIONS[] = { 0.125, 0.25, 0.50, 0.75 };
static const size_t BELLS_N_CACHE = sizeof(BELLS_CACHE_FRACTIONS)/sizeof(BELLS_CACHE_FRACTIONS[0]);

struct bells_source {
    std::vector<uint64_t> counts;
    std::vector<int32_t>  trace;
    uint64_t              n_routed = 0;

    // distinct experts touched by one ubatch: a cache must hold all of them at once, so this
    // bounds what tiering can do while processing a batch
    std::vector<uint32_t> seen_stamp;
    uint32_t              stamp        = 0;
    uint64_t              sum_distinct = 0;
    uint64_t              sum_rows     = 0;
    uint64_t              n_ubatch     = 0;
};

struct bells_layer {
    bells_source topk;
    bells_source argsort;
};

struct bells_profiler {
    std::map<int, bells_layer> layers;

    int64_t n_expert           = 0;
    int64_t n_expert_used_hint = 0;

    std::vector<int32_t> row;

    // per-token record, in the same order the layer traces are appended
    std::vector<int32_t> tok_id;
    std::vector<int32_t> tok_pos;
    std::vector<uint8_t> tok_generated;
};

static bool bells_parse_tagged(const char * name, const char * prefix, int & il) {
    const size_t n = strlen(prefix);
    if (strncmp(name, prefix, n) != 0) {
        return false;
    }

    // the suffix must be digits and nothing else: ggml_cont names its output
    // "ffn_moe_topk-<il> (cont)", which a plain prefix test would also accept and double count
    const char * suffix = name + n;
    if (*suffix == '\0') {
        return false;
    }

    for (const char * s = suffix; *s; ++s) {
        if (*s < '0' || *s > '9') {
            return false;
        }
    }

    il = atoi(suffix);

    return true;
}

// reads the first k entries of each row, honoring nb[1] so that views work too
static void bells_tally(const ggml_tensor * t, int64_t k, int64_t n_expert,
                        std::vector<int32_t> & row, bells_source & src) {
    k = std::min(k, t->ne[0]);
    if (k <= 0) {
        return;
    }

    if (src.seen_stamp.empty()) {
        src.seen_stamp.assign(n_expert, 0);
    }

    const int64_t n_rows = ggml_nrows(t);

    row.resize(k);

    src.stamp++;
    uint64_t n_distinct = 0;

    for (int64_t i = 0; i < n_rows; ++i) {
        ggml_backend_tensor_get(const_cast<ggml_tensor *>(t), row.data(), i*t->nb[1], k*sizeof(int32_t));

        for (int64_t j = 0; j < k; ++j) {
            const int32_t e = row[j];
            if (e < 0 || e >= n_expert) {
                continue;
            }

            if ((size_t) e >= src.counts.size()) {
                src.counts.resize(e + 1, 0);
            }
            src.counts[e]++;
            src.n_routed++;
            src.trace.push_back(e);

            if (src.seen_stamp[e] != src.stamp) {
                src.seen_stamp[e] = src.stamp;
                n_distinct++;
            }
        }
    }

    src.sum_distinct += n_distinct;
    src.sum_rows     += n_rows;
    src.n_ubatch++;
}

static bool bells_cb_eval(ggml_tensor * t, bool ask, void * user_data) {
    auto * prof = (bells_profiler *) user_data;

    int il = -1;

    const bool is_topk    = bells_parse_tagged(t->name, "ffn_moe_topk-", il);
    const bool is_argsort = !is_topk && bells_parse_tagged(t->name, "ffn_moe_argsort-", il);

    if (ask) {
        return (is_topk || is_argsort) && t->type == GGML_TYPE_I32;
    }

    if (!is_topk && !is_argsort) {
        return true;
    }

    auto & layer = prof->layers[il];

    if (is_topk) {
        bells_tally(t, t->ne[0], prof->n_expert, prof->row, layer.topk);
    } else {
        bells_tally(t, prof->n_expert_used_hint, prof->n_expert, prof->row, layer.argsort);
    }

    return true;
}

// hit rate of a reactive LRU cache of the given capacity
static double bells_sim_lru(const std::vector<int32_t> & trace, int64_t n_expert, size_t cap) {
    if (trace.empty() || cap == 0) {
        return 0.0;
    }

    std::vector<uint64_t> last_used(n_expert, 0);
    std::vector<char>     resident(n_expert, 0);
    std::vector<int32_t>  live;

    live.reserve(cap);

    uint64_t n_hit = 0;

    for (size_t i = 0; i < trace.size(); ++i) {
        const int32_t e = trace[i];

        if (resident[e]) {
            n_hit++;
            last_used[e] = i;
            continue;
        }

        if (live.size() >= cap) {
            size_t   victim_at = 0;
            uint64_t oldest    = UINT64_MAX;
            for (size_t j = 0; j < live.size(); ++j) {
                if (last_used[live[j]] < oldest) {
                    oldest    = last_used[live[j]];
                    victim_at = j;
                }
            }
            resident[live[victim_at]] = 0;
            live[victim_at] = live.back();
            live.pop_back();
        }

        resident[e]  = 1;
        last_used[e] = i;
        live.push_back(e);
    }

    return (double) n_hit/(double) trace.size();
}

// hit rate of Belady's optimal policy: evict whichever resident expert is needed furthest ahead
static double bells_sim_opt(const std::vector<int32_t> & trace, int64_t n_expert, size_t cap) {
    if (trace.empty() || cap == 0) {
        return 0.0;
    }

    std::vector<std::vector<uint32_t>> occ(n_expert);
    for (size_t i = 0; i < trace.size(); ++i) {
        occ[trace[i]].push_back((uint32_t) i);
    }

    std::vector<size_t>  ptr(n_expert, 0);
    std::vector<char>    resident(n_expert, 0);
    std::vector<int32_t> live;

    live.reserve(cap);

    auto next_use = [&](int32_t x, size_t i) -> uint64_t {
        while (ptr[x] < occ[x].size() && occ[x][ptr[x]] <= i) {
            ptr[x]++;
        }
        return ptr[x] < occ[x].size() ? (uint64_t) occ[x][ptr[x]] : UINT64_MAX;
    };

    uint64_t n_hit = 0;

    for (size_t i = 0; i < trace.size(); ++i) {
        const int32_t e = trace[i];

        next_use(e, i);

        if (resident[e]) {
            n_hit++;
            continue;
        }

        if (live.size() >= cap) {
            size_t   victim_at = 0;
            uint64_t furthest  = 0;
            for (size_t j = 0; j < live.size(); ++j) {
                const uint64_t nu = next_use(live[j], i);
                if (nu >= furthest) {
                    furthest  = nu;
                    victim_at = j;
                }
            }
            resident[live[victim_at]] = 0;
            live[victim_at] = live.back();
            live.pop_back();
        }

        resident[e] = 1;
        live.push_back(e);
    }

    return (double) n_hit/(double) trace.size();
}

// Writes the training set: one record per token, holding the token id and the experts every MoE
// layer selected for it. Layout is fixed-size so it maps straight onto a numpy array.
//
//   magic "BELLSTR1", u32 version, u32 n_layer_moe, u32 n_expert, u32 n_expert_used,
//   u32 n_vocab, u64 n_records, u32 layer_ids[n_layer_moe]
//   then per record: u32 token_id, u32 pos, u8 is_generated, u8 pad[3],
//                    u16 experts[n_layer_moe*n_expert_used]
static bool bells_write_trace(const std::string & path, const bells_profiler & prof,
                              int32_t n_expert, int32_t n_expert_used, int32_t n_vocab) {
    std::vector<int>                    layer_ids;
    std::vector<const std::vector<int32_t> *> traces;

    const size_t n_records = prof.tok_id.size();

    for (const auto & kv : prof.layers) {
        const bells_source & src = kv.second.topk.n_routed > 0 ? kv.second.topk : kv.second.argsort;

        if (src.n_routed == 0) {
            continue;
        }

        if (src.trace.size() != n_records*(size_t) n_expert_used) {
            LOG_WRN("%s: layer %d has %zu trace entries, expected %zu, excluding it from the dump\n",
                    __func__, kv.first, src.trace.size(), n_records*(size_t) n_expert_used);
            continue;
        }

        layer_ids.push_back(kv.first);
        traces.push_back(&src.trace);
    }

    if (layer_ids.empty() || n_records == 0) {
        LOG_ERR("%s: nothing to write\n", __func__);
        return false;
    }

    std::ofstream f(path, std::ios::binary);
    if (!f) {
        LOG_ERR("%s: failed to open '%s'\n", __func__, path.c_str());
        return false;
    }

    const uint32_t n_layer_moe = (uint32_t) layer_ids.size();
    const uint32_t version     = 1;
    const uint64_t n_rec64     = n_records;

    f.write("BELLSTR1", 8);
    f.write((const char *) &version,       sizeof(version));
    f.write((const char *) &n_layer_moe,   sizeof(n_layer_moe));
    f.write((const char *) &n_expert,      sizeof(n_expert));
    f.write((const char *) &n_expert_used, sizeof(n_expert_used));
    f.write((const char *) &n_vocab,       sizeof(n_vocab));
    f.write((const char *) &n_rec64,       sizeof(n_rec64));

    for (int il : layer_ids) {
        const uint32_t v = (uint32_t) il;
        f.write((const char *) &v, sizeof(v));
    }

    std::vector<uint16_t> experts((size_t) n_layer_moe*n_expert_used);

    for (size_t i = 0; i < n_records; ++i) {
        const uint32_t id  = (uint32_t) prof.tok_id[i];
        const uint32_t pos = (uint32_t) prof.tok_pos[i];
        const uint8_t  gen = prof.tok_generated[i];
        const uint8_t  pad[3] = { 0, 0, 0 };

        f.write((const char *) &id,  sizeof(id));
        f.write((const char *) &pos, sizeof(pos));
        f.write((const char *) &gen, sizeof(gen));
        f.write((const char *) pad,  sizeof(pad));

        for (uint32_t l = 0; l < n_layer_moe; ++l) {
            const std::vector<int32_t> & tr = *traces[l];
            for (int32_t j = 0; j < n_expert_used; ++j) {
                experts[(size_t) l*n_expert_used + j] = (uint16_t) tr[i*(size_t) n_expert_used + j];
            }
        }

        f.write((const char *) experts.data(), experts.size()*sizeof(uint16_t));
    }

    f.close();

    LOG_INF("%s: wrote %s (%zu records, %u MoE layers)\n", __func__, path.c_str(), n_records, n_layer_moe);

    return true;
}

// number of experts, ranked hottest first, needed to cover the given fraction of routing decisions
static size_t bells_coverage(const std::vector<uint64_t> & sorted, uint64_t total, double frac) {
    if (total == 0) {
        return 0;
    }

    const double target = frac*(double) total;

    double acc = 0.0;
    for (size_t i = 0; i < sorted.size(); ++i) {
        acc += (double) sorted[i];
        if (acc >= target) {
            return i + 1;
        }
    }

    return sorted.size();
}

int main(int argc, char ** argv) {
    common_params params;

    params.out_file = "bells-profile.json";
    params.n_ctx    = 512;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_IMATRIX)) {
        return 1;
    }

    common_init();

    llama_backend_init();
    llama_numa_init(params.numa);

    bells_profiler prof;

    params.cb_eval           = bells_cb_eval;
    params.cb_eval_user_data = &prof;
    params.warmup            = false;

    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s: failed to init\n", __func__);
        return 1;
    }

    const int32_t n_expert      = llama_model_n_expert(model);
    const int32_t n_expert_used = llama_model_n_expert_used(model);

    if (n_expert == 0) {
        LOG_ERR("%s: '%s' is not a MoE model, nothing to profile\n", __func__, params.model.path.c_str());
        return 1;
    }

    prof.n_expert           = n_expert;
    prof.n_expert_used_hint = n_expert_used;

    LOG_INF("%s: n_expert = %d, n_expert_used = %d, n_layer = %d\n",
            __func__, n_expert, n_expert_used, llama_model_n_layer(model));

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true, params.parse_special);

    const int n_ctx = llama_n_ctx(ctx);

    if ((int) tokens.size() < n_ctx) {
        LOG_ERR("%s: need at least %d tokens, input tokenizes to %d\n", __func__, n_ctx, (int) tokens.size());
        return 1;
    }

    const int n_chunk_max = tokens.size() / n_ctx;
    const int n_chunk     = params.n_chunks < 0 ? n_chunk_max : std::min(params.n_chunks, n_chunk_max);

    LOG_INF("%s: profiling %d chunks of %d tokens\n", __func__, n_chunk, n_ctx);

    llama_batch batch = llama_batch_init(n_ctx, 0, 1);

    uint64_t n_tokens_seen = 0;

    // every token must be an output, otherwise the final layer skips the non-output tokens and its
    // trace no longer lines up with the other layers
    for (int i = 0; i < n_chunk; ++i) {
        llama_memory_clear(llama_get_memory(ctx), true);

        common_batch_clear(batch);

        for (int j = 0; j < n_ctx; ++j) {
            const llama_token id = tokens[i*n_ctx + j];

            common_batch_add(batch, id, j, { 0 }, true);

            prof.tok_id.push_back(id);
            prof.tok_pos.push_back(j);
            prof.tok_generated.push_back(0);
        }

        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("%s: llama_decode failed on chunk %d\n", __func__, i);
            llama_batch_free(batch);
            return 1;
        }

        n_tokens_seen += n_ctx;

        LOG_INF("%s: chunk %d/%d\n", __func__, i + 1, n_chunk);
    }

    // decode-regime samples: prime with a short seed, then step one token at a time
    if (params.n_predict > 0) {
        const int32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
        const int     n_seed   = std::min(n_ctx/2, (int) tokens.size());
        const int     n_gen    = std::min(params.n_predict, n_ctx - n_seed);

        LOG_INF("%s: generating %d tokens after a %d token seed\n", __func__, n_gen, n_seed);

        int64_t t_gen_us = 0;

        llama_memory_clear(llama_get_memory(ctx), true);

        common_batch_clear(batch);
        for (int j = 0; j < n_seed; ++j) {
            common_batch_add(batch, tokens[j], j, { 0 }, true);
        }

        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("%s: llama_decode failed on the generation seed\n", __func__);
            llama_batch_free(batch);
            return 1;
        }

        for (int j = 0; j < n_seed; ++j) {
            prof.tok_id.push_back(tokens[j]);
            prof.tok_pos.push_back(j);
            prof.tok_generated.push_back(0);
        }

        n_tokens_seen += n_seed;

        int last_out = n_seed - 1;

        for (int i = 0; i < n_gen; ++i) {
            const float * logits = llama_get_logits_ith(ctx, last_out);
            if (logits == nullptr) {
                LOG_ERR("%s: no logits available at step %d\n", __func__, i);
                break;
            }

            llama_token best = 0;
            for (int32_t v = 1; v < n_vocab; ++v) {
                if (logits[v] > logits[best]) {
                    best = v;
                }
            }

            const int pos = n_seed + i;

            common_batch_clear(batch);
            common_batch_add(batch, best, pos, { 0 }, true);

            const int64_t t0 = ggml_time_us();

            if (llama_decode(ctx, batch) != 0) {
                LOG_ERR("%s: llama_decode failed at generation step %d\n", __func__, i);
                break;
            }

            t_gen_us += ggml_time_us() - t0;

            prof.tok_id.push_back(best);
            prof.tok_pos.push_back(pos);
            prof.tok_generated.push_back(1);

            n_tokens_seen++;
            last_out = 0;
        }

        if (t_gen_us > 0) {
            LOG_INF("%s: decode %.2f ms/token, %.2f tok/s (%d tokens)\n", __func__,
                    t_gen_us/1000.0/n_gen, 1e6*n_gen/(double) t_gen_us, n_gen);
        }

        // Degeneration guard. A generation that collapses into a loop touches almost no
        // distinct experts, so the cache hits ~100% and the timing above looks excellent
        // precisely because the model broke. Refuse to let that pass unremarked.
        {
            std::vector<llama_token> gen;
            for (size_t i = 0; i < prof.tok_id.size(); ++i) {
                if (prof.tok_generated[i]) {
                    gen.push_back(prof.tok_id[i]);
                }
            }

            std::vector<llama_token> sorted = gen;
            std::sort(sorted.begin(), sorted.end());
            const size_t n_uniq = std::unique(sorted.begin(), sorted.end()) - sorted.begin();

            size_t longest = gen.empty() ? 0 : 1;
            size_t run     = longest;
            for (size_t i = 1; i < gen.size(); ++i) {
                run = gen[i] == gen[i-1] ? run + 1 : 1;
                longest = std::max(longest, run);
            }

            const double frac = gen.empty() ? 0.0 : (double) n_uniq/gen.size();

            LOG_INF("%s: generated %zu tokens, %zu unique (%.1f%%), longest repeat %zu\n",
                    __func__, gen.size(), n_uniq, 100.0*frac, longest);

            if (!gen.empty() && (frac < 0.25 || longest > 8)) {
                LOG_WRN("%s: OUTPUT LOOKS DEGENERATE. Any speedup above is unreliable: a\n",
                        __func__);
                LOG_WRN("%s: looping model reuses the same few experts and inflates cache hits.\n",
                        __func__);
            }
        }
    }

    // Teacher-forced perplexity through the decode path.
    //
    // llama-perplexity runs large batches, which BELLS deliberately bypasses, so it cannot
    // measure the cache at all. Feeding known tokens one at a time keeps ubatch size 1, which
    // is exactly the regime BELLS serves, and scoring the real next token avoids the trap of
    // judging quality from greedy samples that diverge for numerical reasons alone.
    if (params.compute_ppl && (int) tokens.size() > n_ctx) {
        const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        const int     n_seed  = std::min(n_ctx/2, (int) tokens.size());
        const int     n_score = std::min(256, (int) tokens.size() - n_seed - 1);

        LOG_INF("%s: scoring %d tokens one at a time for perplexity\n", __func__, n_score);

        llama_memory_clear(llama_get_memory(ctx), true);

        common_batch_clear(batch);
        for (int j = 0; j < n_seed; ++j) {
            common_batch_add(batch, tokens[j], j, { 0 }, true);
        }

        double nll = 0.0;
        int    n_ok = 0;

        if (llama_decode(ctx, batch) == 0) {
            int last_out = n_seed - 1;

            for (int i = 0; i < n_score; ++i) {
                const float * logits = llama_get_logits_ith(ctx, last_out);
                if (logits == nullptr) {
                    break;
                }

                const llama_token next = tokens[n_seed + i];

                float max_l = logits[0];
                for (int32_t v = 1; v < n_vocab; ++v) {
                    max_l = std::max(max_l, logits[v]);
                }

                double sum = 0.0;
                for (int32_t v = 0; v < n_vocab; ++v) {
                    sum += exp((double) (logits[v] - max_l));
                }

                nll += -((double) (logits[next] - max_l) - log(sum));
                n_ok++;

                common_batch_clear(batch);
                common_batch_add(batch, next, n_seed + i, { 0 }, true);

                if (llama_decode(ctx, batch) != 0) {
                    break;
                }
                last_out = 0;
            }
        }

        if (n_ok > 0) {
            LOG_INF("%s: perplexity %.4f over %d tokens (ubatch 1, BELLS active)\n",
                    __func__, exp(nll/n_ok), n_ok);
        }
    }

    llama_batch_free(batch);

    size_t caps[BELLS_N_CACHE];
    for (size_t c = 0; c < BELLS_N_CACHE; ++c) {
        size_t cap = (size_t) (BELLS_CACHE_FRACTIONS[c]*(double) n_expert + 0.5);
        cap = std::max(cap, (size_t) n_expert_used);
        cap = std::min(cap, (size_t) n_expert);
        caps[c] = cap;
    }

    std::ofstream out(params.out_file);
    if (!out) {
        LOG_ERR("%s: failed to open '%s' for writing\n", __func__, params.out_file.c_str());
        return 1;
    }

    out << "{\n";
    out << "  \"model\": \"" << params.model.path << "\",\n";
    out << "  \"n_expert\": " << n_expert << ",\n";
    out << "  \"n_expert_used\": " << n_expert_used << ",\n";
    out << "  \"n_tokens\": " << n_tokens_seen << ",\n";
    out << "  \"layers\": [\n";

    LOG_INF("\n");
    LOG_INF("%s: hot-set size needed to cover a share of all routing decisions,\n", __func__);
    LOG_INF("%s: and distinct experts touched by one ubatch\n", __func__);
    LOG_INF("%s: %5s %12s %7s %7s %7s %7s %10s %9s\n", __func__,
            "layer", "routed", "50%", "80%", "95%", "99%", "tok/ubatch", "distinct");

    bool first = true;

    std::vector<double> agg_lru(BELLS_N_CACHE, 0.0);
    std::vector<double> agg_opt(BELLS_N_CACHE, 0.0);
    size_t n_layers_simulated = 0;

    for (auto & kv : prof.layers) {
        const int il = kv.first;

        const bool use_topk = kv.second.topk.n_routed > 0;
        bells_source & src  = use_topk ? kv.second.topk : kv.second.argsort;

        if (src.n_routed == 0) {
            continue;
        }

        std::vector<uint64_t> padded = src.counts;
        padded.resize(std::max<size_t>(padded.size(), (size_t) n_expert), 0);

        std::vector<uint64_t> sorted = padded;
        std::sort(sorted.begin(), sorted.end(), std::greater<uint64_t>());

        const double avg_rows     = src.n_ubatch ? (double) src.sum_rows/(double) src.n_ubatch : 0.0;
        const double avg_distinct = src.n_ubatch ? (double) src.sum_distinct/(double) src.n_ubatch : 0.0;

        LOG_INF("%s: %5d %12" PRIu64 " %7zu %7zu %7zu %7zu %10.0f %9.1f\n", __func__, il, src.n_routed,
                bells_coverage(sorted, src.n_routed, 0.50),
                bells_coverage(sorted, src.n_routed, 0.80),
                bells_coverage(sorted, src.n_routed, 0.95),
                bells_coverage(sorted, src.n_routed, 0.99),
                avg_rows, avg_distinct);

        double lru[BELLS_N_CACHE];
        double opt[BELLS_N_CACHE];

        for (size_t c = 0; c < BELLS_N_CACHE; ++c) {
            lru[c] = bells_sim_lru(src.trace, n_expert, caps[c]);
            opt[c] = bells_sim_opt(src.trace, n_expert, caps[c]);

            agg_lru[c] += lru[c];
            agg_opt[c] += opt[c];
        }
        n_layers_simulated++;

        if (!first) {
            out << ",\n";
        }
        first = false;

        out << "    { \"layer\": " << il
            << ", \"source\": \"" << (use_topk ? "topk" : "argsort")
            << "\", \"n_routed\": " << src.n_routed
            << ", \"distinct_per_ubatch\": " << avg_distinct
            << ", \"cache\": [";

        for (size_t c = 0; c < BELLS_N_CACHE; ++c) {
            if (c) {
                out << ", ";
            }
            out << "{ \"capacity\": " << caps[c]
                << ", \"lru\": " << lru[c]
                << ", \"belady\": " << opt[c] << " }";
        }

        out << "], \"counts\": [";

        for (size_t e = 0; e < padded.size(); ++e) {
            if (e) {
                out << ", ";
            }
            out << padded[e];
        }

        out << "] }";
    }

    out << "\n  ]\n}\n";
    out.close();

    LOG_INF("\n");
    LOG_INF("%s: expert cache hit rate, averaged over layers\n", __func__);
    LOG_INF("%s: LRU is reactive (no prediction), Belady assumes perfect prediction\n", __func__);
    LOG_INF("%s: %10s %10s %10s %10s\n", __func__, "capacity", "LRU", "Belady", "headroom");

    for (size_t c = 0; c < BELLS_N_CACHE; ++c) {
        if (n_layers_simulated == 0) {
            break;
        }
        const double l = 100.0*agg_lru[c]/(double) n_layers_simulated;
        const double o = 100.0*agg_opt[c]/(double) n_layers_simulated;
        LOG_INF("%s: %10zu %9.1f%% %9.1f%% %9.1f%%\n", __func__, caps[c], l, o, o - l);
    }

    LOG_INF("\n");
    LOG_INF("%s: wrote %s\n", __func__, params.out_file.c_str());

    {
        std::string trace_path = params.out_file;
        const size_t dot = trace_path.find_last_of('.');
        if (dot != std::string::npos) {
            trace_path.erase(dot);
        }
        trace_path += ".trace.bin";

        bells_write_trace(trace_path, prof, n_expert, n_expert_used,
                          llama_vocab_n_tokens(llama_model_get_vocab(model)));
    }

    llama_backend_free();

    return 0;
}
