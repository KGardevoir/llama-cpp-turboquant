#include "llama-triattention-calibrate.h"
#include "llama-triattention.h"   // TRIATTENTION_MAGIC, TRIATTENTION_VERSION

#include "ggml-backend.h"
#include "ggml.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>

triattention_calibrate_state * triattention_calibrate_create(
        const char * output_path,
        uint32_t     n_layers,
        uint32_t     n_attn_heads,
        uint32_t     n_kv_heads,
        uint32_t     head_dim,
        double       rope_theta,
        uint32_t     rope_style,
        const char * model_name) {

    if (!output_path || output_path[0] == '\0') {
        return nullptr;
    }
    if (head_dim < 2 || n_layers == 0 || n_attn_heads == 0) {
        fprintf(stderr, "[TriAttention calibrate] ERROR: invalid model dimensions "
                "(layers=%u heads=%u head_dim=%u)\n", n_layers, n_attn_heads, head_dim);
        return nullptr;
    }

    auto * s = new triattention_calibrate_state();
    s->output_path  = output_path;
    s->n_layers     = n_layers;
    s->n_attn_heads = n_attn_heads;
    s->n_kv_heads   = n_kv_heads;
    s->head_dim     = head_dim;
    s->freq_count   = head_dim / 2;
    s->rope_theta   = rope_theta;
    s->rope_style   = rope_style;
    s->n_tokens     = 0;
    s->written      = false;

    strncpy(s->model_name, model_name ? model_name : "", sizeof(s->model_name) - 1);
    s->model_name[sizeof(s->model_name) - 1] = '\0';

    size_t total = (size_t)n_layers * n_attn_heads * s->freq_count;
    s->sum_q_real.assign(total, 0.0);
    s->sum_q_imag.assign(total, 0.0);
    s->sum_q_abs .assign(total, 0.0);

    fprintf(stderr, "[TriAttention calibrate] Started: output=%s layers=%u heads=%u "
            "head_dim=%u rope_theta=%.1f rope_style=%u\n",
            output_path, n_layers, n_attn_heads, head_dim, rope_theta, rope_style);

    return s;
}

void triattention_calibrate_process_batch(triattention_calibrate_state * state) {
    if (!state || state->pending_q.empty()) {
        return;
    }

    // All layers share the same token count for this batch.
    uint32_t n_batch = (uint32_t)state->pending_q.begin()->second->ne[2];
    if (n_batch == 0) {
        return;
    }

    const uint32_t fc = state->freq_count;

    for (auto & [il, cur] : state->pending_q) {
        if (il < 0 || (uint32_t)il >= state->n_layers) {
            continue;
        }

        const uint32_t ne0 = (uint32_t)cur->ne[0];  // head_dim
        const uint32_t ne1 = (uint32_t)cur->ne[1];  // n_head
        const uint32_t ne2 = (uint32_t)cur->ne[2];  // n_tokens

        // Skip layers with mismatched geometry (e.g. cross-attention, MoE router layers).
        if (ne0 != state->head_dim || ne1 != state->n_attn_heads || ne2 == 0) {
            continue;
        }

        const size_t nb1 = cur->nb[1];  // bytes per head
        const size_t nb2 = cur->nb[2];  // bytes per token (may include K/V gap for fused QKV)

        // Use ggml_nbytes() so we never exceed the tensor's logical span.
        // For contiguous tensors this equals nb2*ne2; for fused-QKV views
        // nb2 is wider than the Q row so nb2*ne2 would exceed ggml_nbytes()
        // and trigger an assertion.  Stride-based indexing below still works
        // correctly because the inter-token gaps are included in nb2.
        const size_t raw_bytes = ggml_nbytes(cur);
        std::vector<uint8_t> raw(raw_bytes);
        ggml_backend_tensor_get(cur, raw.data(), 0, raw_bytes);

        const size_t layer_base = (size_t)il * state->n_attn_heads * fc;

        for (uint32_t t = 0; t < ne2; t++) {
            for (uint32_t h = 0; h < ne1; h++) {
                const float * hd = reinterpret_cast<const float *>(
                    raw.data() + (size_t)t * nb2 + (size_t)h * nb1);

                const size_t head_base = layer_base + (size_t)h * fc;

                for (uint32_t f = 0; f < fc; f++) {
                    float re, im;
                    if (state->rope_style == 0) {
                        // Half layout: q[0..fc-1] = real, q[fc..head_dim-1] = imag
                        re = hd[f];
                        im = hd[f + fc];
                    } else {
                        // Interleaved layout: q[2f] = real, q[2f+1] = imag
                        re = hd[2 * f];
                        im = hd[2 * f + 1];
                    }
                    state->sum_q_real[head_base + f] += (double)re;
                    state->sum_q_imag[head_base + f] += (double)im;
                    state->sum_q_abs [head_base + f] += (double)sqrtf(re * re + im * im);
                }
            }
        }
    }

    state->n_tokens += n_batch;
}

bool triattention_calibrate_write(triattention_calibrate_state * state) {
    if (!state) {
        return false;
    }
    if (state->written) {
        return true;
    }
    fprintf(stderr, "[TriAttention calibrate] write called: output=%s tokens=%llu\n",
            state->output_path.c_str(), (unsigned long long)state->n_tokens);
    if (state->n_tokens == 0) {
        fprintf(stderr, "[TriAttention calibrate] ERROR: no tokens accumulated — "
                "run with -p or -f to provide calibration text\n");
        return false;
    }

    FILE * f = fopen(state->output_path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "[TriAttention calibrate] ERROR: cannot open %s for writing\n",
                state->output_path.c_str());
        return false;
    }

    const uint32_t n_layers  = state->n_layers;
    const uint32_t n_heads   = state->n_attn_heads;
    const uint32_t fc        = state->freq_count;
    const uint32_t n_sampled = n_layers * n_heads;  // all (layer, head) pairs

    // ---- header ----
    const uint32_t magic   = TRIATTENTION_MAGIC;
    const uint32_t version = TRIATTENTION_VERSION;
    fwrite(&magic,              sizeof(uint32_t), 1, f);
    fwrite(&version,            sizeof(uint32_t), 1, f);
    fwrite(&state->head_dim,    sizeof(uint32_t), 1, f);
    fwrite(&n_layers,           sizeof(uint32_t), 1, f);
    fwrite(&n_heads,            sizeof(uint32_t), 1, f);
    fwrite(&state->n_kv_heads,  sizeof(uint32_t), 1, f);
    fwrite(&state->rope_theta,  sizeof(double),   1, f);
    fwrite(&state->rope_style,  sizeof(uint32_t), 1, f);
    fwrite(&n_sampled,          sizeof(uint32_t), 1, f);
    fwrite(&fc,                 sizeof(uint32_t), 1, f);

    const uint32_t name_len = (uint32_t)(strlen(state->model_name) + 1);
    fwrite(&name_len,           sizeof(uint32_t), 1, f);
    fwrite(state->model_name,   1, name_len,       f);

    // ---- per-head stats ----
    const double inv_n = 1.0 / (double)state->n_tokens;
    std::vector<float> buf(fc);

    for (uint32_t il = 0; il < n_layers; il++) {
        for (uint32_t h = 0; h < n_heads; h++) {
            const uint32_t layer_idx = il;
            const uint32_t head_idx  = h;
            fwrite(&layer_idx, sizeof(uint32_t), 1, f);
            fwrite(&head_idx,  sizeof(uint32_t), 1, f);

            const size_t base = (size_t)il * n_heads * fc + (size_t)h * fc;

            // q_mean_real  Re(E[q_f])
            for (uint32_t ff = 0; ff < fc; ff++) {
                buf[ff] = (float)(state->sum_q_real[base + ff] * inv_n);
            }
            fwrite(buf.data(), sizeof(float), fc, f);

            // q_mean_imag  Im(E[q_f])
            for (uint32_t ff = 0; ff < fc; ff++) {
                buf[ff] = (float)(state->sum_q_imag[base + ff] * inv_n);
            }
            fwrite(buf.data(), sizeof(float), fc, f);

            // q_abs_mean  E[|q_f|]
            for (uint32_t ff = 0; ff < fc; ff++) {
                buf[ff] = (float)(state->sum_q_abs[base + ff] * inv_n);
            }
            fwrite(buf.data(), sizeof(float), fc, f);

            // r_f  ||E[q_f]|| / E[|q_f|]  (validation field, not used at inference)
            for (uint32_t ff = 0; ff < fc; ff++) {
                const double re      = state->sum_q_real[base + ff] * inv_n;
                const double im      = state->sum_q_imag[base + ff] * inv_n;
                const double abs_mean = state->sum_q_abs[base + ff] * inv_n;
                buf[ff] = (abs_mean > 0.0) ? (float)(sqrt(re * re + im * im) / abs_mean) : 0.0f;
            }
            fwrite(buf.data(), sizeof(float), fc, f);
        }
    }

    fclose(f);
    state->written = true;

    fprintf(stderr, "[TriAttention calibrate] Wrote %s "
            "(layers=%u heads=%u kv_heads=%u head_dim=%u tokens=%llu)\n",
            state->output_path.c_str(),
            n_layers, n_heads, state->n_kv_heads, state->head_dim,
            (unsigned long long)state->n_tokens);

    return true;
}

void triattention_calibrate_free(triattention_calibrate_state * state) {
    delete state;
}
