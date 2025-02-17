---
layout: distill
title: LION Part I - Full Linear Attention
description: LION part
tags:
giscus_comments: false
date: 2025-02-17
featured: false
thumbnail: assets/img/8.jpg
authors:
  - name: Arshia Afzal
    url:
    affiliations:
      name: EPFL


---


{% include figure.liquid loading="eager" path="assets/img/8.jpg" %}



[[Paper](https://arxiv.org/abs/2405.21060)]
[[Code](https://github.com/state-spaces/mamba)]


1.  Part I - Full Linear Attention
<!-- 2. [Part II - LION: Bidirectional RNN for FUll Linear Attention]({% post_url 2024-05-31-mamba2-part2-theory %})
3. [Part III - LION Chunk: Chunkwise Parallel of LION]({% post_url 2024-05-31-mamba2-part3-algorithm %})
4. [Part IV - Results]({% post_url 2024-05-31-mamba2-part4-systems %}) -->

Recently, Transformers with Linear Attention and State Space Models (SSMs) have gained significant popularity for causal sequence modeling due to their ability to efficiently support both parallel training and RNN-like inference. These models have demonstrated impressive accuracy in causal tasks, particularly in causal language modeling. However, their evaluation in bi-directional sequence modeling, such as image classification and masked language modeling, has been relatively limited. In contrast, SSMs, particularly Mamba, have been extensively evaluated in vision tasks, including models like Vision Mamba and Hydra, which represent official extensions of Mamba for bi-directional sequence modeling.



### Problem 1 (Understanding)
From a conceptual standpoint, one of the reasons we found SSMs so fascinating is how they just feel _fundamental_. One way this is exemplified is how they have rich ties to many major paradigms of sequence models.

