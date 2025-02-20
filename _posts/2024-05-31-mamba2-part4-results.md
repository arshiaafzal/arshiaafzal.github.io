---
layout: distill
title: LION 🦁 Part IV - Results
description: Comprehensive results of LION on Vision, MLM and LION variants
tags: math ai
giscus_comments: false
date: 2024-05-31
featured: false
thumbnail: assets/img/lion.jpg

authors:
  - name: Arshia Afzal 
    url:
    affiliations:
      name: Writer of blogpost
  - name: Elias Abad Rocamora
    url:
    affiliations:
  - name: Leyla Naz Candogan
    url:
    affiliations:
  - name: Pol Puigdemont Plana
    url:
    affiliations:
      name: Writer of blogpost
  - name: Francesco Tonin
    url:
    affiliations:
  - name: Yongtao Wu
    url:
    affiliations:
  - name : Volkan Cevher
    url:
    affiliations:
      name: All authors are with EPFL

bibliography: albert.bib

toc:
  - name: Vision Results
    subsections:
      - name: Base Model Performance
      - name: Memory Efficiency
      - name: Training Time Analysis
  - name: MLM Results
    subsections:
      - name: BERT Comparison
  - name: LION Architecture Variants
    subsections:
      - name: Architecture Trade-offs

---

1. [Part I - Full Linear Attention]({% post_url 2024-05-31-mamba2-part1-model %})
2. [Part II - Bi-directional RNN]({% post_url 2024-05-31-mamba2-part2-theory %})
3. [Part III -  Chunkwise Parallel from of LION]({% post_url 2024-05-31-mamba2-part3-algorithm %})
4. Part IV - Results



In this fourth part of our LION series, we will present and discuss a selection of experimental results across various domains, including vision tasks, masked language modeling (MLM), and different LION architectures. These results not only highlight LION's versatility and efficiency across diverse applications but also serve as a preview of the comprehensive findings detailed in the full paper.

## Image Classification Performance Overview
### Model Comparisons
We evaluated LION's performance, efficiency, and training times against state-of-the-art SSMs and Transformers for image classification. The results demonstrate that LION achieves competitive performance while offering significant advantages in training speed and efficiency.

| Model | #Param | Imagenet Top-1 Acc. | Train. time |
|-------|--------|---------------------|------------|
| ViT | 86M | $77.9$ | $\times 1$ |
| DeiT | 86M | $\underline{81.8}$ | $\times 1$ |
| Hydra | 104M | $81.0$ | $\times 2.51$ |
| Vim | 98M | $\mathbf{81.9}$ | $\times 10.86$ |
| LION-lit | 86M | $74.7$ | $\mathbf{\times 0.73}$ |
| LION-D | 86M | $77.8$ | $\times \underline{1.39}$ |
| $LION-D^{\natural}$ | 86M | $80.2$ | $\times 1.48$ |
| $LION-S$ | 86M | $76.3$ | $\times 1.46$ |
| $LION-S^{\natural}$ | 86M | $79.9$ | $\times 1.68$ |

As shown in the table above, LION models achieve competitive performance with vision-specific SSMs like Vim, while being significantly faster during training. LION-D performs comparably to Vim and surpasses Hydra, while training approximately 7x faster than Vim. Notably, LION-lit demonstrates the highest training speed across all models, showing that training with full linear attention is significantly faster than chunkwise parallel training (used in Hydra) and considerably faster than the scan algorithm, even with optimized GPU kernels (as used in Vim).

### Memory Efficiency

The LION family demonstrates excellent memory efficiency across both vision and language tasks. Figure below shows inference memory usage with a batch size of 64 across different image resolutions, LION models maintain reasonable memory consumption even at high resolutions up to 2496 pixels, while adding minimal training overhead in BERT-style language modeling scenarios. In contrast, baseline models like ViT and DeiT run out of memory (OOM) at much lower resolutions, highlighting LION's memory scaling capabilities regardless of the application domain.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/fig1_plot.svg" title="Memory Usage Comparison" caption="Memory usage during inference across different architectures with batch size 64. LION models maintain reasonable memory consumption at high resolutions while other models run out of memory." class="img-fluid rounded z-depth-1" %}
    </div>
</div>

### Training Time Analysis

The LION family demonstrates remarkable training efficiency across both vision and language tasks. As shown in the table below, LION variants add minimal training overhead compared to standard Transformers, with some variants even training faster.

| Task | LION-lit | LION-retnet | LION-S | Hydra | Vim |
|------|----------|-------------|---------|--------|-----|
| Vision | $\times 0.73$ | $\times 1.39$ | $\times 1.46$ | $\times 2.51$ | $\times 10.86$ |
| MLM | $\times 0.95$ | $\times 1.10$ | $\times 1.32$ | $\times 3.13$ | ✗ |
{: .table-caption}
*Training Times (relative to Transformer) ↓*

For vision tasks, LION-lit achieves remarkable speed, training 27% faster than standard Transformers. Even the more complex LION variants maintain competitive training times, with LION-retnet and LION-S training only ~1.4x slower than Transformers. This is significantly better than competing approaches like Hydra (2.51x slower) and Vim (10.86x slower).

In MLM tasks, the efficiency gains are even more pronounced. LION-lit nearly matches Transformer training speed at just 0.95x, while LION-retnet adds only 10% overhead. Even LION-S remains efficient at 1.32x. All LION variants significantly outperform Hydra's 3.13x slowdown, while Vim is not applicable to MLM tasks (marked as ✗).

## MLM Results

For masked language modeling (MLM) tasks, we evaluated LION models against BERT and Hydra on both MLM pretraining and GLUE benchmark finetuning. The results show that LION variants achieve competitive performance while maintaining excellent training efficiency.

| Model | MLM Acc. | GLUE | Train. time |
|-------|----------|------|-------------|
| BERT | $\underline{69.88}$ | $\mathbf{82.95}$ | $\times 1$ |
| Hydra | $\mathbf{71.18}$ | $\underline{81.77}$ | $\times 3.13$ |
| LION-lit | $67.11$ | $80.76$ | $\times \mathbf{0.95}$ |
| LION-retnet | $68.64$ | $81.34$ | $\times \underline{1.10}$ |
| LION-S | $69.16$ | $81.58$ | $\times 1.32$ |
{: .table-caption}
*C4 MLM and GLUE results for the LARGE scale ($334$M). For each dataset, the best and second best results are highlighted with bold and underline respectively.*

## LION Architecture Variants and Trade-offs

Let's explore how different LION variants handle the trade-off between memory usage and inference speed. We'll look at three key approaches:

1. Full Linear Attention - The standard approach using the full attention matrix
2. Bidirectional RNN - Our memory-efficient RNN formulation 
3. LION Chunk - A balanced approach using chunked computation

### Memory vs Speed Trade-offs

The first plot below shows how these approaches compare in terms of memory efficiency and inference speed. The RNN approach proves to be the most memory-efficient, while full attention uses the most memory. LION chunk provides a nice middle ground - it uses less memory than full attention while actually achieving faster inference speeds than both alternatives. This makes it particularly attractive when you need to balance performance with resource constraints.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/fig3_plot.svg" title="Impact of Chunk Size" caption="Analysis of how chunk size affects model performance across different LION variants." class="img-fluid rounded z-depth-1" %}
    </div>
</div>

### Detailed Performance Analysis

Looking more closely at the memory-time trade-off across different LION variants, we can see some interesting patterns. While RNN remains the most memory-efficient across all models, both chunking and full attention hit memory limits much sooner. The chunking approach matches or beats full attention's inference speed for simpler variants like LION-RetNet. However, with more complex variants like LION-S, chunking is only faster at lower resolutions - at higher resolutions, the overhead from mask calculations starts to slow it down.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/linear_chunking.svg" title="Linear Chunking Analysis" caption="Evaluation of linear chunking strategies and their impact on model efficiency." class="img-fluid rounded z-depth-1" %}
    </div>
</div>

### Selective Chunking Results

The final analysis examines how different chunking strategies perform across sequence lengths. This helps inform which approach is best for different scenarios - chunking tends to be optimal for LION-lit and LION-RetNet when memory allows, while RNN can be preferable for handling complex masks at high resolutions.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/selective_chunking.svg" title="Selective Chunking Analysis" caption="Performance comparison of selective chunking approaches across different sequence lengths." class="img-fluid rounded z-depth-1" %}
    </div>
</div>
