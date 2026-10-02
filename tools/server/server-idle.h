#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

struct idle_profile {
    int32_t threads = 0;
    int32_t decay = -1; // -1 means no BELLS cache, zero means LRU
    bool operator==(const idle_profile & other) const {
        return threads == other.threads && decay == other.decay;
    }
};

struct idle_measurement {
    double tps = 0, low = 0, high = 0, p99_ms = 0;
    static idle_measurement summarize(std::vector<double> values) {
        idle_measurement result;
        if (values.empty() || std::any_of(values.begin(), values.end(), [](double v) { return !std::isfinite(v) || v <= 0; })) return result;
        result.tps = values.size()/std::accumulate(values.begin(), values.end(), 0.0);
        std::sort(values.begin(), values.end());
        const size_t tail = (values.size() + 99)/100;
        result.low = tail/std::accumulate(values.end() - tail, values.end(), 0.0);
        result.high = tail/std::accumulate(values.begin(), values.begin() + tail, 0.0);
        result.p99_ms = values[(values.size()*99 + 99)/100 - 1]*1000;
        return result;
    }
};

// Two distinct workloads, each in incumbent/candidate/candidate/incumbent order.
inline bool idle_is_improvement(const std::array<std::vector<double>, 8> & runs) {
    for (size_t offset : { size_t(0), size_t(4) }) {
        const size_t count = runs[offset].size();
        if (count < 128) return false;
        for (size_t i = 0; i < 4; ++i) if (runs[offset + i].size() != count) return false;
        auto base = runs[offset], trial = runs[offset + 1];
        base.insert(base.end(), runs[offset + 3].begin(), runs[offset + 3].end());
        trial.insert(trial.end(), runs[offset + 2].begin(), runs[offset + 2].end());
        const auto a = idle_measurement::summarize(base), b = idle_measurement::summarize(trial);
        if (!a.tps || !b.tps || b.tps < a.tps*1.03 || b.low < a.low ||
            b.high < a.high*0.99 || b.p99_ms > a.p99_ms*1.01) return false;
        // Both orderings must agree; one anomalous baseline must not select a winner.
        if (idle_measurement::summarize(runs[offset + 1]).tps < idle_measurement::summarize(runs[offset]).tps*1.01 ||
            idle_measurement::summarize(runs[offset + 2]).tps < idle_measurement::summarize(runs[offset + 3]).tps*1.01) return false;
    }
    return true;
}

struct server_idle_tuner {
    bool enabled = false, running = false, adopted = false;
    uint64_t activity = 0, attempted_activity = UINT64_MAX;
    int64_t last_busy = 0;
    int threads_batch = 0;
    int warm = 64, measured = 512;
    size_t candidate = 0, run = 0, position = 0;
    idle_profile accepted;
    std::vector<idle_profile> candidates;
    std::array<std::vector<int32_t>, 2> tokens;
    std::array<std::vector<uint64_t>, 2> reference;
    std::array<std::vector<double>, 8> times;
    std::string state = "disabled";
    std::string last_decision;
    uint64_t completed_trials = 0, interruptions = 0;
};
