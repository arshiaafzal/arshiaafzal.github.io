---
layout: distill
title:  LION 🦁 Part III - Chunkwise Parallel from of LION
description: 
tags:
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
  - name: LION-Chunk
    subsections:
      - name: LION-D Chunk
      - name: LION-S Chunk


---

1. [Part I - Full Linear Attention]({% post_url 2024-05-31-mamba2-part1-model %})
2. [Part II - Bi-directional RNN]({% post_url 2024-05-31-mamba2-part2-theory %})
3. Part III - Chunkwise Parallel from of LION
4. [Part IV - Results]({% post_url 2024-05-31-mamba2-part4-results %})


Since we have now established the LION theorem, which maps full linear attention into a bidirectional RNN in [Part II]({% post_url 2024-05-31-mamba2-part2-theory %}) of this series, a key question arises:  

Given that RNNs are efficient and attention is fast, can we strike a balance between them?  

For causal Transformers like DeltaNet and GLA, as well as the SSD algorithm in Mamba2, a chunkwise parallel form of full linear attention could be an effective solution. Additionally, in models like Hydra, this balance is achieved by applying two SSD algorithms. However, can we derive a unified framework for chunking full linear attention, particularly for LION-S and LION-D, where the decay factor is fixed and the mask $\mathbf{M}$ follows a Toeplitz structure?  But it is important that this chunkwise form is particularly useful for **inference**, since during training, as we said, full linear attention will provide the highest throughput, especially for short sequences, which is the case for bidirectional tasks. The aim of chunking full attention in LION is to maintain a balance between efficiency and speed, particularly during inference. Since LION benefits from stable masks, it does not require chunking during training, unlike SSMs such as Hydra.

## LION-Chunk

As mentioned above, the main purpose of chunking is to balance the memory-speed tradeoff during inference. The key idea is that instead of processing the entire sequence of length $L$, we divide it into $N$ subsequences of length $C$, where $N \times C = L$.  
To achieve this, we start with the full linear attention formulation:  

 $$\mathbf{Y} = (\mathbf{Q} \mathbf{K}^\top \odot \mathbf{M}) \mathbf{V}$$
 
 we first chunk the queries, keys and values into submatrices and chunks as followes 

$$
\mathbf{Q}_{[i]} , \mathbf{K}_{[i]}, \mathbf{V}_{[i]}  \in \mathbb{R}^{C \times d}
$$

Now, given the masked full linear attention in the form  $ (\mathbf{A} \odot \mathbf{M})$  for each chunk of the mask and attention, we need to construct the chunkwise form, which consists of four parts: 

- Chunkwise form of the attention matrix $\mathbf{A}_{[ij]}$
- Chunkwise form for the scaling matrix $\mathbf{C}$ which the chunkwise form can be written as $\mathbf{C}_{[ij]}$
- The chunked hidden state to shape the unscaled output $\mathbf{S}_{[ij-1]}$
- Finally the output of the chunk $i$ which is $\mathbf{Y}_{[i]}$

using these chunked matrices we shape the full linear atteniton in chunk form as bellow:

> **LION Chunk**
> 
> $$\begin{aligned}
    \mathbf{A}_{[ij]} & = \mathbf{Q}_{[i]}\mathbf{K}_{[j]}^\top \odot \mathbf{M}_{[ij]}, \\
     \mathbf{C}_{[ij]} &= \mathbf{C}_{[ij-1]} + \text{Sum} (\mathbf{A}_{[ij]}), \\
     \mathbf{S}_{[ij]} & =\mathbf{S}_{[ij-1]} + \mathbf{A}_{[ij]} \mathbf{V}_{[j]} , \\
     \mathbf{Y}_{[i]} & = \frac{\mathbf{S}_{[iN]}}{\mathbf{C}_{[iN]}}
\end{aligned}$$
{: .block-tip}

where $\text{Sum}$ operations applies summation over the row of the input matrix. And $\mathbf{M}_{[ij]}$ corresponds to a submatrix  of the full maks $\mathbf{M}$ at chunk $ij$ like:

$$
\mathbf{M}_{[ij]} = \mathbf{M}_{iC+1:i(C+1),jC+1:j(C+1)} \in \mathbb{R}^{C \times C}.
$$


Don't be intimidated by the above formulation—it is actually quite intuitive when visualized. It clearly shows the steps taken to achieve the chunkwise form.

Let's start by chunking the attention matrix $\mathbf{A}$. To better understand this, let's examine the full attention for a sequence of $L=9$ with $C=3$ chunk size in detail below:


{% include figure.liquid loading="eager" path="assets/img/att_chunk.png"%}

As seen above actually the chunking is just calculating the query and keys respective for each boxed sub-matrix which we have shown three in above illustrarion for upper, lower and diagonal chunks. As seen for all the attention matrix for the chunk $ij$ is calculated the same way only by multiplying the respective queries $\mathbf{Q}_{[i]}$ and keys $\mathbf{Q}_{[j]}$ for that chunk


{% include figure.liquid loading="eager" path="assets/img/maskdec_chunk.png"%}

{% include figure.liquid loading="eager" path="assets/img/masksel_chunk.png"%}



### LION-D Chunk




``` python
def Casual_Mask_Decay_Partial(a_i , L,start,end):
    idx = torch.arange(L,device=a_i.device)
    I, J = torch.meshgrid(idx, idx[start:end], indexing='ij')
    E = (torch.abs((I-J)).float().view(1,1,L,len(idx[start:end])))
    M = torch.sigmoid(a_i).view(1,-1,1,1)**E
    return M
```


### LION-S Chunk


## Next Up

In the [final part of this series]({% post_url 2024-05-31-mamba2-part4-results %}), we finally show the advantages of using LION compared to other ways of training SSM or Linear Transformers 


