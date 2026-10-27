---
layout: distill
title: "Memory Attention"
description: Replacing attention's value projection with a token-indexed memory table
tags: attention memory language-models
giscus_comments: false
date: 2026-10-27
featured: false
# thumbnail: assets/img/memory-attention/cover.png

authors:
  - name: Jiale Kang
    url: https://joluck.github.io/

toc:
  - name: TLDR
  - name: The Odd One Out
  - name: Memory Was Already in the Building
  - name: Memory Attention, Precisely
    subsections:
      - name: Values from keys and memory
      - name: The output splits in two
      - name: Fold the norm, drop the matmul
  - name: The Ablations Are the Story
  - name: Results
    subsections:
      - name: Language modeling
      - name: Retrieval
      - name: Efficiency
  - name: Offload, and the Recall Extension
  - name: Implementation
  - name: What This Is Not
  - name: Closing Thoughts
  - name: Citation
---

<!-- COVER (optional) — upload assets/img/memory-attention/cover.png, then remove the Liquid comment wrapper below. -->
{% comment %}
{% include figure.liquid loading="eager" path="assets/img/memory-attention/cover.png" width="55%" class="rounded mx-auto d-block" %}
{% endcomment %}

<style>
/* wide result tables scroll sideways on phones instead of being cut off */
@media (max-width: 768px) { d-article table { display: block; overflow-x: auto; } }
</style>

Attention is a soft lookup. Queries ask, keys answer, and values are what gets carried back. The first two get the architectural attention: their interaction is where context mixing happens, and most efficiency tricks in the book are about making that interaction cheaper. The value projection is the quiet third member of the trio — a dense matrix that turns the same hidden state into the payload.

That symmetry is suspicious. Queries and keys are irreducibly contextual: the same word must be able to look for different things in different sentences. Values are different. A value vector is what gets averaged and written back into the residual stream, and a large part of it is plausibly about the token itself — the entity, the verb, the name — content that is the same every time the token appears.

The question behind Memory Attention is whether that part can come from a table instead of a projection. Each layer learns one vector per vocabulary token. The value becomes the contextual key plus the retrieved row, and the value projection disappears.

<div style="background:rgba(59,130,246,.10); border-radius:10px; padding:0.5rem 1.5rem; margin:1rem 0;">

$$
\mathbf{V} = \mathbf{K} + \operatorname{Norm}\left(\mathbf{E}[\mathbf{s}]\right)
$$

</div>

That is the whole architectural change. The rest of this post is about why it works, where it does not, and what it costs.

[Paper](https://arxiv.org/abs/2609.28399) · [Code](https://github.com/Joluck/memory-attention) · [Project page and demo](https://joluck.github.io/memory-attention/)

## TLDR

- **The change.** Drop the value projection; build values as $\mathbf{V} = \mathbf{K} + \operatorname{Norm}(\mathbf{E}[\mathbf{s}])$, with one table per layer, indexed by token ID. Keys, RoPE, and the attention operator are untouched.
- **Why it can work.** The output splits into $\mathbf{A}\mathbf{K} + \mathbf{A}\mathbf{M}$: contextual retrieval plus attention-weighted memory. The table is context-free; the retrieval is not.
- **The inference trick.** Row-wise normalization folds into the table, so value construction is one gather and one add. Lookup addresses depend only on token IDs, so the table can live in CPU memory and be prefetched.
- **The evidence.** Under matched token budgets: +0.71 average for MHA at 1,024 hidden, +1.48 for the gated variant, +1.16 at 2,048; lower LAMBADA and WikiText perplexity; all nine NIAH cells improve. With offload, GPU parameter storage falls below the standard baseline.
- **The catch.** MA adds parameters — 1.6–3× at 1,024 hidden — so the comparison is token-matched, not parameter-matched. Offload is a prototype, and the recall extension is analysis only.

## The Odd One Out

Write standard attention out and the three projections look interchangeable:

$$
\mathbf{Q} = \mathbf{X}\mathbf{W}_Q,\quad
\mathbf{K} = \mathbf{X}\mathbf{W}_K,\quad
\mathbf{V} = \mathbf{X}\mathbf{W}_V,
$$

$$
\operatorname{Attn}(\mathbf{X}) = \operatorname{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^{\top}}{\sqrt{d_h}} + \mathbf{C}\right)\mathbf{V}.
$$

They take the same input and, in the MHA case, have the same shape and asymptotic cost. But they enter the computation at different places. Queries and keys meet inside the softmax; their inner product decides which positions matter. Values never meet queries or keys until the weights are fixed. They are payload: weighted, summed, projected back.

That division of labor is why value embeddings work: keep the projected payload, add a learned table vector for the token, and give the model extra token-specific capacity without widening the block. The projection stays; the table supplements it.

Memory Attention asks how far that can go. If the key is already a contextual representation of the token, can it serve as the contextual half of the value? Then the value projection collapses into a table lookup plus an addition.

Two things have to be true for this to be a good idea. The key has to carry enough context to be a useful value, and the table has to add something the key does not already have. The ablations test both. The first is essential: give memory the whole job and the model gets worse than the baseline (40.34 versus 41.11). The gain only appears when the two are added.

## Memory Was Already in the Building

Adding token-indexed memory to transformers is not new. The variants differ in where the memory enters and how it is combined:

- **Value embeddings** add a learned table vector to the projected value, usually with a scalar mix. The projection stays.
- **Per-Layer Embeddings (PLE)** ([Gemma 3n](https://ai.google.dev/gemma/docs/gemma-3n)) add per-layer token embeddings, gated into the block.
- **Engram** ([Cheng et al., 2026](https://arxiv.org/abs/2601.07372)) indexes memory by local n-grams rather than single tokens, and gates the retrieved content into the residual stream.
- **Bank of Values (BoV)** ([He et al., 2026](https://arxiv.org/abs/2606.02780)) replaces value projections in the last third of layers with context-free token tables.
- **DeepEmbedAttention (DEA)** ([Bo, 2021](https://github.com/BlinkDL/RWKV-LM)) uses token-indexed embeddings to modulate keys and values multiplicatively; its author also suggested replacing key and value computation with direct lookups.

All of these add memory. MA asks whether memory can replace the value projection. The distinction sounds like a matter of wording until you see the ablation: letting memory stand in for the whole value scores 40.34, below the 41.11 baseline. The gain comes from the combination, $\mathbf{V} = \mathbf{K} + \mathbf{M}$. So MA is not "memory instead of values." It is "keys as the contextual half of values, memory as the token-specific half."

The rest of the design follows from making that combination work: normalize the memory rows, keep position out of the value path, and put the memory where attention aggregates rather than where it matches.

## Memory Attention, Precisely

Let $\mathbf{X} \in \mathbb{R}^{T \times d}$ be the hidden states and $\mathbf{s} = (s_1, \ldots, s_T)$ the token IDs. Let $H_{\mathrm{KV}}$ be the number of key/value heads, $d_h$ the head dimension, and $d_v = H_{\mathrm{KV}} d_h$. Each layer owns a table $\mathbf{E} \in \mathbb{R}^{N \times d_v}$, where $N$ is the vocabulary size.

$$
\mathbf{M} = \operatorname{Norm}(\mathbf{E}[\mathbf{s}]), \qquad
\mathbf{Q} = \mathbf{X}\mathbf{W}_Q, \qquad
\mathbf{K} = \mathbf{X}\mathbf{W}_K, \qquad
\mathbf{V} = \mathbf{K} + \mathbf{M}.
$$

$$
\mathbf{Q}^{R} = \operatorname{RoPE}(\mathbf{Q}), \qquad
\mathbf{K}^{R} = \operatorname{RoPE}(\mathbf{K}),
$$

$$
\mathbf{O} = \operatorname{softmax}\left(\frac{\mathbf{Q}^{R}(\mathbf{K}^{R})^{\top}}{\sqrt{d_h}} + \mathbf{C}\right)\mathbf{V}.
$$

$\operatorname{Norm}$ is RMSNorm applied per token and per KV head. The table is per layer, so the same token gets a different vector at every depth. Everything else is stock attention: the same softmax, the same causal mask, the same FlashAttention call.

Three details matter more than they look:

1. **Values use the key before RoPE**, and in the reference code before QK-norm as well. Position helps decide where to look; it is less obviously useful in what gets carried. The ablation puts a number on this.
2. **The memory rows are normalized.** The learned table's scale is otherwise free to drift, and with it the scale of the value.
3. **The table is indexed by token, not position.** That is what makes the lookup independent of sequence position and prefetchable.

### Values from keys and memory

Rewrite the value as a shared projection plus a bias:

$$
\mathbf{V} = \underbrace{\mathbf{X}\mathbf{W}_K}_{\text{shared contextual projection}} + \underbrace{\operatorname{Norm}(\mathbf{E}[\mathbf{s}])}_{\text{token-dependent bias}}.
$$

This is key–value weight sharing with a token-dependent bias. Unlike a usual bias, it is not shared across tokens. Unlike a usual embedding, it is not added to the input or the output; it enters exactly where the value would have. The key projection is now trained for two jobs: to be compared with queries, and to be part of the payload. The bet is that these two jobs are compatible.

One way to see why the two jobs are compatible: the key is already a contextual summary of the token. Reusing it as the contextual part of the value avoids learning a second summary of the same thing. The table then supplies a stable, token-specific component that does not have to be recovered sharply from the hidden state every time.

<!-- FIGURE 1 — upload assets/img/memory-attention/fig1-overview.png, then remove the Liquid comment wrapper below. -->
{% comment %}
{% include figure.liquid loading="eager" path="assets/img/memory-attention/fig1-overview.png" title="Three value paths" caption="Standard attention, Memory Attention, and MA-Offload. MA replaces the dense value projection with a lookup and an addition; MA-Offload keeps the folded table in CPU memory and prefetches rows for the next layer group. RoPE and other components are omitted." class="img-fluid rounded z-depth-1" %}
{% endcomment %}

### The output splits in two

Because attention is linear in $\mathbf{V}$, the output splits:

$$
\mathbf{O} = \mathbf{A}(\mathbf{K} + \mathbf{M}) = \underbrace{\mathbf{A}\mathbf{K}}_{\text{contextual retrieval}} + \underbrace{\mathbf{A}\mathbf{M}}_{\text{memory retrieval}}.
$$

For a query at position $t$, the memory term is

$$
\mathbf{o}^{\text{mem}}_t = \sum_{j \le t} a_{tj}\mathbf{m}_{s_j},
$$

where $a_{tj}$ depends on the query and the contextual keys. The memory vector for a given token is fixed within a layer, but its contribution is not: the same row can be read with different weights in different contexts. **The table is context-free; the retrieval is not.**

This is also the cleanest way to distinguish MA from context-free value tables. In BoV, selected layers use a value that depends only on the token. In MA, every value is context-dependent, because $\mathbf{K}$ is; the table only supplies the part of the value that is not.

### Fold the norm, drop the matmul

At inference the normalization is row-wise, so it can be applied once to the table:

$$
\bar{\mathbf{E}}[i] = \operatorname{Norm}(\mathbf{E}[i]), \qquad
\mathbf{V} = \mathbf{K} + \bar{\mathbf{E}}[\mathbf{s}].
$$

Online value construction is then a gather and an add. The dense $d \times d_v$ matrix multiply is gone.

| Per layer | Standard | Memory Attention |
|---|---|---|
| Value construction | $\mathbf{X}\mathbf{W}_V$ | $\mathbf{K} + \bar{\mathbf{E}}[\mathbf{s}]$ |
| Trainable parameters | $L d d_v$ | $L (N d_v + p_{\text{norm}})$ |
| Online value-construction FLOPs | $\approx 2 L S d d_v$ | $L S d_v$ |

<div class="caption" style="color: #666666; margin-top: 1px;">$L$ replaced layers, $S$ newly processed tokens. Lookup overhead and memory traffic are excluded; inference costs cover new tokens only.</div>

The trade is visible in the table: parameter count goes up when $N > d$, while online arithmetic goes down. At 32k vocabulary and 2,048 hidden, the tables alone are $24 \times 32{,}000 \times 2{,}048 \approx 1.57$B parameters — 3,000 MiB in BF16. A table that large is not something you want to keep on the GPU if you can avoid it. That is the motivation for offloading, and the reason the next question is not just "does it work?" but "where does it live?"

## The Ablations Are the Story

Before the headline numbers, the ablations. They use 24 layers, 1,024 hidden, about 10B tokens, and a batch of roughly 0.5M tokens. The baseline is a 336M-parameter MQA-style model; the memory variants add about 190M parameters. Bold marks the best value in each column.

**Memory is a complement, not a replacement.**

| Configuration | Params | LAMBADA PPL ↓ | WikiText PPL ↓ | Avg ↑ |
|---|---|---|---|---|
| $\mathbf{V} = \mathbf{X}\mathbf{W}$ (standard) | 336M | 50.38 | 31.84 | 41.11 |
| $\mathbf{V} = \mathbf{M}$ | 526M | 58.36 | 32.94 | 40.34 |
| $\mathbf{V} = (1-\lambda)\mathbf{X}\mathbf{W} + \lambda\mathbf{E}[i]$ | 532M | 47.87 | 31.12 | 40.99 |
| $\mathbf{V} = \mathbf{X}\mathbf{W} + \mathbf{M}$ | 532M | **45.68** | 29.89 | 41.28 |
| $\mathbf{V} = \mathbf{K} + \mathbf{M}$ | 526M | 45.93 | **29.82** | **41.38** |

Replacing values with memory alone is worse than the baseline: 40.34 versus 41.11, and LAMBADA perplexity jumps to 58.36. Interpolating between the projection and the table recovers little (40.99). Addition works better (41.28), and reusing the key is best (41.38), with the lowest WikiText perplexity. The ordering says memory is useful as a complement to input-dependent values, not as a substitute for them.

**Placement matters as much as integration.** These rows share keys and values ($\mathbf{K} = \mathbf{V}$), so the memory is the only difference.

| Memory added to | Params | LAMBADA PPL ↓ | WikiText PPL ↓ | Avg ↑ |
|---|---|---|---|---|
| Hidden states $\mathbf{H} + \mathbf{M}$ | 526M | 50.22 | 30.68 | 41.14 |
| Queries $\mathbf{Q} + \mathbf{M}$ | 526M | 59.69 | 33.17 | 40.55 |
| Keys $\mathbf{K} + \mathbf{M}$ | 526M | 58.23 | 32.63 | 40.35 |
| Values $\mathbf{V} + \mathbf{M}$ | 526M | **45.93** | **29.82** | **41.38** |
| Outputs $\mathbf{O} + \mathbf{M}$ | 526M | 71.83 | 34.47 | 39.83 |

The pattern is consistent: memory helps where attention aggregates content, not where it matches queries to keys, and not at the output. Injecting into queries or keys hurts the matching signal. Injecting into the output is the worst of the five, with LAMBADA perplexity rising to 71.83. Hidden-state injection is competitive (41.14), which makes sense — it is another way to add capacity to the block — but the value path is the cleanest place.

**Normalize the rows, keep position out of the value path.**

| Configuration | Params | LAMBADA PPL ↓ | WikiText PPL ↓ | Avg ↑ |
|---|---|---|---|---|
| $\mathbf{V} = \mathbf{K} + \mathbf{E}[i]$ | 526M | 54.46 | 32.60 | 40.85 |
| $\mathbf{V} = \mathbf{K}^{R} + \operatorname{Norm}(\mathbf{E}[i])$ | 526M | 51.38 | 30.79 | 40.61 |
| $\mathbf{V} = \mathbf{K} + \operatorname{Norm}(\mathbf{E}[i])$ | 526M | **45.93** | **29.82** | **41.38** |

Two independent choices, both large. Without normalization, LAMBADA perplexity is 54.46; with it, 45.93. Sharing the RoPE-transformed key is worse than sharing the pre-RoPE key (40.61 versus 41.38), even when the memory is normalized. The reading: control the scale of the memory contribution, and do not rotate content that is going to be averaged. Position is for matching, not for the payload.

**It stacks with AttnRes.** Adding attention residuals improves both models. Standard+AttnRes reaches 41.00 with LAMBADA perplexity 39.59; MA+AttnRes reaches 41.46 with LAMBADA 38.37 and WikiText 28.88, the best perplexities in the ablation tables. The average for Standard+AttnRes is slightly below Standard (41.11), a useful reminder that perplexity and the downstream average do not always move together at this scale. For MA, they do.

## Results

All models are trained on FineWeb-10BT with 24 layers, RoPE, RMSNorm, and a gated MLP, using Flash Linear Attention (FLA) on H800s. The 1,024-hidden models see 10B tokens at a 2,048 context; the 2,048-hidden model sees 20B. Each Standard/MA pair shares the training budget and seed. Evaluation is zero-shot with lm-evaluation-harness.

### Language modeling

| Backbone | Params (Std → MA) | Standard avg ↑ | MA avg ↑ | Δ |
|---|---|---|---|---|
| MHA · 1,024 | 373M → 1,135M | 40.68 | 41.39 | +0.71 |
| GQA · 1,024 | 349M → 729M | 40.90 | 41.51 | +0.61 |
| MQA · 1,024 | 336M → 526M | 41.11 | 41.34 | +0.23 |
| Gated · 1,024 | 398M → 1,160M | 41.40 | 42.88 | +1.48 |
| MHA · 2,048 | 1,364M → 2,836M | 49.06 | 50.22 | +1.16 |

MA wins every pair. The largest gain is the gated variant (+1.48); the smallest is MQA (+0.23), which also has the smallest table. Perplexity moves more consistently than the average: at 1,024, MHA's WikiText perplexity falls from 31.55 to 28.64, and GQA's LAMBADA perplexity from 50.46 to 43.68; at 2,048, LAMBADA goes from 18.59 to 18.15 and WikiText from 21.52 to 20.79.

The average is not uniformly better task by task. At 2,048, MA gains 3.0 points on OpenBookQA and 2.65 on ARC-Challenge but loses a little on ARC-Easy and PIQA. The headline is an unweighted mean of seven accuracies, and it is worth looking at the spread.

Two caveats. First, these are token-matched comparisons: MA has 1.6–3× the parameters at 1,024 hidden. Some of the gain is capacity, and the paper says so. Second, the budget is small by modern standards: the dense backbones are 336M–1.36B parameters, with another 1.5B parameters in the largest memory tables. Treat the numbers as evidence about a trade, not as a scaling law.

### Retrieval

Single-needle needle-in-a-haystack (NIAH), trained at a 2,048 context and evaluated at 1,024, 2,048, and 4,096 tokens. The last is twice the training window.

| Length | Model | niah_single_1 | niah_single_2 | niah_single_3 | Mean |
|---|---|---|---|---|---|
| 1K | Standard | 99.8 | 85.6 | 62.4 | 82.6 |
| 1K | MA | 100.0 | 94.0 | 98.2 | **97.4** |
| 2K | Standard | 84.4 | 91.4 | 70.4 | 82.1 |
| 2K | MA | 99.4 | 100.0 | 91.0 | **96.8** |
| 4K† | Standard | 41.4 | 23.2 | 13.0 | 25.9 |
| 4K† | MA | 51.0 | 45.4 | 29.4 | **41.9** |

<div class="caption" style="color: #666666; margin-top: 1px;">† Extrapolation beyond the 2K training context. Higher is better; bold marks the better mean.</div>

All nine cells improve. Within the training window, the mean goes from 82.6% to 97.4% at 1K and from 82.1% to 96.8% at 2K. At 4K, both models degrade — 25.9% versus 41.9% — but MA degrades less. The biggest single jump is niah_single_3 at 1K, from 62.4% to 98.2%.

This is the result that makes the mechanism plausible: a token-indexed table gives a rare entity a stable representation that does not have to be reconstructed from a context-averaged hidden state, while attention still decides when to read it. But it is evidence within and just beyond the training window, not a long-context result.

### Efficiency

Training token efficiency is 1.42× at 1,024 hidden and 1.16× at 2,048 at matched loss — 29.6% and 13.8% fewer tokens to reach the selected loss levels. That is a statement about tokens, not wall-clock time; the extra parameters and optimizer state are real costs.

| Configuration | Prefill (ms) | Decode (ms) | GPU params (M) | GPU storage (MiB) | CPU storage (MiB) |
|---|---|---|---|---|---|
| Standard | 90.970 | 17.636 | 1,364.3 | 2,602.4 | 0 |
| MA | 88.202 | 17.788 | 2,836.5 | 5,410.4 | 0 |
| MA-Offload | 90.686 | 17.189 | 1,263.6 | 2,410.4 | 3,000.0 |

At the reference point — batch 8, 2,048-token prefill, 2,048-token decode history, BF16, one H800 — MA's prefill is 3.0% faster and decode is 0.9% slower. The offload variant is 0.3% faster in prefill and 2.5% faster in decode. These are small differences at one operating point, not a speedup story.

The storage story is the interesting one. MA-Offload keeps 2,410 MiB of parameters on the GPU: 192 MiB less than Standard (7.4%) and 55.5% less than GPU-resident MA, while total parameters are 2.08× Standard. The price is 3,000 MiB of CPU memory and the transfer path.

<!-- FIGURE 2 — upload assets/img/memory-attention/fig2-efficiency.png, then remove the Liquid comment wrapper below. -->
{% comment %}
{% include figure.liquid loading="lazy" path="assets/img/memory-attention/fig2-efficiency.png" title="Token efficiency and latency" caption="Token efficiency and forward-pass latency for the 2,048-hidden model. MA reaches the matched loss with 13.8% fewer training tokens; at the reference operating point, latency is roughly flat while the table moves to CPU memory." class="img-fluid rounded z-depth-1" %}
{% endcomment %}

**Scope.** Forward pass only, including embeddings, final norm, and head; excluding KV cache, activations, buffers, and token-ID transfers. The paper's Figure 3 sweeps batch size and context length. The timings are operating-point measurements, not throughput claims.

## Offload, and the Recall Extension

**Prefetch is the whole point.** Token IDs are known before the hidden states. For layer $\ell$, the rows $\mathbf{E}[s_t]$ can be gathered while layer $\ell - 1$ is still computing. The prototype folds the normalization into the table, keeps it in pinned CPU memory, and runs a single worker thread that gathers the rows for a group of layers. A separate CUDA stream copies the group to the GPU; group-level events make the consumer wait only for the group it needs. Prefill groups one layer at a time; decoding groups four, with a prefetch depth of four.

The benchmark still keeps a conventional KV cache. Offloading changes where the parameter table lives, not what attention reads. That is why GPU storage drops but the total parameter count does not.

Caveats: this is a prototype, not part of the causal LM implementation. The transfer buffers still occupy GPU memory; CPU memory bandwidth and PCIe become the new limits; and the measured benefit is specific to the operating point. The paper calls it a feasibility result, which is the right framing.

**Reconstructing values instead of caching them.** Because the value is additive, it is also invertible: keep the content keys and the token IDs, and the values can be rebuilt.

$$
\mathbf{V}_{1:T} = \mathbf{K}_{1:T} + \bar{\mathbf{E}}[\mathbf{s}_{1:T}].
$$

A conventional BF16 KV cache stores both keys and values, $2 b_c L B T d_v$ bytes. Keeping one key representation and the token IDs, then reconstructing values on demand, halves that component — to $b_c L B T d_v$. The cost is reconstruction: about $L B R d_v$ additions per decoding step, where $R$ is the number of historical positions reconstructed ($R = T$ for full causal attention, less for sliding-window), plus the positional transform.

The positional transform is the awkward part. Attention scoring uses rotated keys. If you cache pre-RoPE keys, you must rotate them when scoring; if you cache rotated keys, you must invert the rotation to recover the content key for the value. Either way there is extra work, and if you keep both representations the cache saving shrinks.

This is the part of the paper I find most interesting and least settled. It turns the memory table into a compression scheme for the KV cache: store the ingredients, re-derive the content. The open questions are practical — gather throughput, overlap, numerical drift, and whether the rotation bookkeeping is cheap enough. There are no measurements for it in the paper, by design.

{% details What the 50% does not include %}

The 50% compares one key representation against a key-plus-value cache. It excludes token IDs, position metadata, reconstruction buffers, and any second copy of keys kept for scoring. The paper states this explicitly. Treat 50% as a bound on the cache component, not on total memory.

{% enddetails %}

## Implementation

The reference implementation is small. The layer is a normal attention block with three changes: no `v_proj`, an embedding table, and a per-head RMSNorm on the retrieved rows. It uses FlashAttention for the attention itself, so the kernel is unchanged. A sketch:

```python
class MemoryAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, vocab_size, qk_norm=False):
        super().__init__()
        self.head_dim = hidden_size // num_heads
        self.kv_dim = num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.kv_dim, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.m_proj = nn.Embedding(vocab_size, self.kv_dim)      # one per layer
        self.m_norm = RMSNorm(self.head_dim, dtype=torch.float32)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim, dtype=torch.float32)
            self.k_norm = RMSNorm(self.head_dim, dtype=torch.float32)
        self.rotary = RotaryEmbedding(dim=self.head_dim)

    def forward(self, x, input_ids):
        q = rearrange(self.q_proj(x), "... (h d) -> ... h d", d=self.head_dim)
        k = rearrange(self.k_proj(x), "... (h d) -> ... h d", d=self.head_dim)
        m = rearrange(self.m_proj(input_ids), "... (h d) -> ... h d", d=self.head_dim)

        v = k + self.m_norm(m)                 # the value, before any position

        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = self.rotary(q, k)               # scores only

        o = flash_attn_func(q, k, v, causal=True)
        return self.o_proj(o.flatten(-2, -1))
```

Two details are easy to get wrong. `m_norm` is RMSNorm over `head_dim`, applied after reshaping to heads; normalizing the full `kv_dim` vector would couple the heads. And `v` is built from `k` before QK-norm and RoPE, so the content key and the scoring key are the same tensor at different points in the pipeline.

Inference folds the norm into the table: apply `m_norm` row-wise to `m_proj.weight` once, then the layer only needs `v = k + E_bar[ids]`. The benchmark's offload path does this before timing.

The repository layout is equally small:

- `fla/layers/memory_attn.py` — the layer above.
- `fla/models/memory/` — configuration, backbone, and causal LM.
- `profile/` — the offload prototype and inference comparisons.

## What This Is Not

**Not parameter-matched.** MA adds 190M–1.5B parameters depending on configuration. The comparisons match training tokens, not parameters. The paper is explicit that the results do not isolate structure from capacity; neither does this post.

**Not a latency win.** The value matmul disappears, but prefill and decode move by a few percent at one operating point. If throughput is the goal, measure the workload.

**Not external retrieval.** There is no corpus, index, or retriever. The table is a learned parameter, closer to a per-layer value embedding than to RAG. "Memory" here means capacity in the weights, addressed by token ID.

**Not a KV-cache replacement yet.** The measured system keeps a conventional KV cache. MA-Recall is an analysis with no benchmarks.

**Not routing or sparsity.** Every row is a dense parameter. A lookup touches few rows per token, but that does not by itself make the gradients or optimizer states sparse; in this implementation the table is a normal embedding.

**Not a scaling study.** 24 layers, 0.3B–1.4B dense backbones, FineWeb-10BT, 2K context. The paper lists SWA, linear attention, larger models, and MoE as ongoing.

## Closing Thoughts

The decomposition is the part worth keeping: attention's value path carries two kinds of information, contextual and token-specific, and a dense projection is only one way to split them. Once you write $\mathbf{V} = \mathbf{X}\mathbf{W}_K + \operatorname{Norm}(\mathbf{E}[\mathbf{s}])$, the design space is larger than "replace values with embeddings." Which layers get a table? What is the table indexed by — tokens, n-grams, positions? How are the two halves normalized? Does the second half need to be stored at all?

The negative results are as useful as the positive ones. Memory alone is worse than the baseline; position in the value path hurts; injecting memory into the output is the worst placement. Those constraints make the space smaller and more interesting.

The cost story is clean: more parameters, less arithmetic, and a lookup that can live off the GPU. Whether that is a good trade depends on the bottleneck. If you are GPU-memory bound and have CPU bandwidth to spare, MA-Offload is attractive. If you need parameter parity or are already bandwidth bound, it is a harder sell. The paper's position is that this is a starting point — and MA-Recall, the part that would make the table pay for itself in cache memory, is still unbuilt.

## Citation

```bibtex
@misc{kang2026memoryattention,
  title         = {Memory Attention},
  author        = {Jiale Kang},
  year          = {2026},
  eprint        = {2609.28399},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2609.28399}
}
```
