---
layout: distill
title:  LION 🦁 Part III - Chunkwise Parallel from of LION
description: 
tags:
giscus_comments: false
date: 2024-02-20
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
      - name: LION-S Chunk
      - name: LION-D Chunk


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

As seen above, chunking simply involves computing the queries and keys for each boxed sub-matrix, as illustrated for the upper, lower, and diagonal chunks. For every attention matrix chunk $[ij]$, the computation follows the same pattern—multiplying the corresponding queries and keys for that chunk.  

But does the same approach apply to selective and fixed masks?  

In reality, chunking the attention mask is slightly different and even more critical than chunking attention itself due to its unique structure. Below, we provide a detailed explanation of how to chunk the attention mask for LION-D and LION-S.

🚀 **Note:** Please pay close attention to this section, as the visualization and details of this part of chunking are not included in the paper.


### LION-D Chunk

Let's start with the decay mask, as it is simpler and easier to visualize. For LION-D with a fixed mask, the final mask is a Toeplitz mask constructed using the scalar decay factor $\lambda$.  Now, let's examine how the mask is structured.

{% include figure.liquid loading="eager" path="assets/img/maskdec_chunk.png"%}

As seen, the full mask of LION-D (or full RetNet mask) is constructed simply by the submatrix of $\Gamma$, which is a Toeplitz matrix itself. Regardless of where the chunk is located, whether in the upper or lower part of the mask matrix $\mathbf{M}$, it retains the same property of being a fraction of the Toeplitz matrix $\Gamma$ as bellow:


$$
\mathbf{M}_{[ij]} = \Gamma \lambda^{|i-j|}
$$

which can simply be implemented in Pytorch.

### The code for LION-D Chunk Mask

``` python
def Mask_Decay_Partial(a_i , L,start,end):
    idx = torch.arange(L,device=a_i.device)
    I, J = torch.meshgrid(idx, idx[start:end], indexing='ij')
    E = (torch.abs((I-J)).float().view(1,1,L,len(idx[start:end])))
    M = torch.sigmoid(a_i).view(1,-1,1,1)**E
    return M
```


### LION-S Chunk

Despite LION-D the full mask of LION-S is more tricky since the upper lower and the diagonal part of the mask are shaped differently:

- The <span style="background-color: rgb(255, 248, 203); padding: 3px; color:black">Upper</span> part is influenced only by the decay factors applied from the end to the beginning of the sequence.  
- The <span style="background-color: rgb(254, 200, 201); padding: 3px; color:black">Diagonal</span> part incorporates contributions from both directions, spanning from the start to the end and from the end to the start.  
- The <span style="background-color: rgb(208, 243, 248); padding: 3px; color:black">Lower</span> part is influenced only by the decay factors applied from the beginning to the end of the sequence.

Let's visualize LION-S mask as well:
  
{% include figure.liquid loading="eager" path="assets/img/masksel_chunk.png"%}


As seen above, the chunk 1,3, for example, has only the cumulative decay factors multiplied from the beginning up to the last three sequence elements, while the chunk 3,1 has only the decay factors multiplied from the end up to the first three sequence elements. And this is the reason for using the matrices $\mathbf{L}^F$ and $\mathbf{L}^B$ to compute the cumulative products of the decay factors, progressing from the beginning to the end of the sequence and in reverse which can be created simply by `L^F = cumprod(a)` and `L^B = cumprod(Flip(a))`. 



### The code for LION-S Chunk Mask

``` python
def mask_forward(tensor,chunk_index,chunk_len):
    cumprod = torch.clamp(tensor.cumprod(dim=-1),1e-6)
    A = cumprod.unsqueeze(-1) / cumprod.unsqueeze(-2)[...,chunk_index*chunk_len:(chunk_index+1)*chunk_len]
    return torch.tril(A,diagonal = -chunk_index*chunk_len)

def mask_backward(tensor,chunk_index,chunk_len):
    cumprod = torch.clamp(tensor.cumprod(dim=-1),1e-6)
    A = cumprod.unsqueeze(-1)[...,chunk_index*chunk_len:(chunk_index+1)*chunk_len,:] / cumprod.unsqueeze(-2)
    return torch.triu(A.transpose(-1,-2),diagonal = -chunk_index*chunk_len)

def Mask_Selective_Partial(vec,chunk_index,chunk_len):
    B,H,L = vec.shape
    A_for = create_matrix_from_tensor_forward(torch.cat((torch.ones_like(vec[..., :2]), vec[...,1:-1]), dim=-1),chunk_index,chunk_len)
    A_back = create_matrix_from_tensor_backward(torch.cat((torch.ones_like(vec[..., :1]), vec[...,1:]), dim=-1),chunk_index,chunk_len)
    I  = torch.diag_embed(torch.ones((B,H,L-chunk_index*chunk_len)),offset = -chunk_index*chunk_len)[...,:A_for.shape[-1]]
    return A_for + A_back - I.to(A_for.device)
```


Now that we have all elemnts in place let's see how these models are working in practice on real-world datasets for masked language modeling and image classification.

## Next Up

In the [final part of this series]({% post_url 2024-05-31-mamba2-part4-results %}), we present the advantages of using LION compared to other methods for training SSMs or Linear Transformers.

We will present the trade-offs for different LION models and compare them with other well-known SSMs and Softmax Transformers.

