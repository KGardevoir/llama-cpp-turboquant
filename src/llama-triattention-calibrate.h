#pragma once

// TriAttention offline calibration — integrated llama.cpp path
//
// Captures pre-RoPE Q vectors during llama_decode() via the "Qcur_pre_rope"
// graph callback hook, accumulates per-(layer, head, freq) statistics, and
// writes a .triattention binary calibration file on context destruction (or
// when explicitly flushed with triattention_calibrate_write).
//
// Usage:
//   llama-cli -m model.gguf --triattention-calibrate out.triattention -f corpus.txt
//
// The resulting file can be passed to --triattention-stats for inference.

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct ggml_tensor;

struct triattention_calibrate_state {
    std::string output_path;

    uint32_t n_layers;
    uint32_t n_attn_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t freq_count;   // head_dim / 2
    double   rope_theta;
    uint32_t rope_style;   // 0 = half (LLaMA/Qwen), 1 = interleaved (GPT-NeoX)
    char     model_name[256];

    // Running sums indexed as [layer * n_attn_heads * freq_count + head * freq_count + freq]
    std::vector<double> sum_q_real;
    std::vector<double> sum_q_imag;
    std::vector<double> sum_q_abs;   // E[|q_f|] accumulated
    uint64_t            n_tokens;    // total tokens seen across all batches

    // Qcur tensors registered by graph_get_cb for the current batch.
    // Key: layer index (il).  Cleared on graph rebuild, updated by cb.
    std::unordered_map<int, ggml_tensor *> pending_q;

    bool written;  // guards against double-write
};

// Allocate and initialise a calibration state from model metadata.
triattention_calibrate_state * triattention_calibrate_create(
    const char * output_path,
    uint32_t     n_layers,
    uint32_t     n_attn_heads,
    uint32_t     n_kv_heads,
    uint32_t     head_dim,
    double       rope_theta,
    uint32_t     rope_style,
    const char * model_name);

// Read back pending Qcur tensors from device memory and accumulate statistics.
// Called after each successful graph_compute() in process_ubatch().
void triattention_calibrate_process_batch(triattention_calibrate_state * state);

// Compute means from accumulators and write the .triattention binary file.
// Safe to call more than once; subsequent calls are no-ops.
// Returns true on success.
bool triattention_calibrate_write(triattention_calibrate_state * state);

// Free all memory.  Safe to call with nullptr.
void triattention_calibrate_free(triattention_calibrate_state * state);
