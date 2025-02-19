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
4. [Part IV - Results]({% post_url 2024-05-31-mamba2-part4-systems %})


Since we have now established the LION theorem, which maps full linear attention into a bidirectional RNN in [Part II]({% post_url 2024-05-31-mamba2-part2-theory %}) of this series, a key question arises:  

Given that RNNs are efficient and attention is fast, can we strike a balance between them?  

For causal Transformers like DeltaNet and GLA, as well as the SSD algorithm in Mamba2, a chunkwise parallel form of full linear attention could be an effective solution. Additionally, in models like Hydra, this balance is achieved by applying two SSD algorithms. However, can we derive a unified framework for chunking full linear attention, particularly for LION-S and LION-D, where the decay factor is fixed and the mask $\mathbf{M}$ follows a Toeplitz structure?  But it is important that this chunkwise form is particularly useful for **inference**, since during training, as we said, full linear attention will provide the highest throughput, especially for short sequences, which is the case for bidirectional tasks. The aim of chunking full attention in LION is to maintain a balance between efficiency and speed, particularly during inference. Since LION benefits from stable masks, it does not require chunking during training, unlike SSMs such as Hydra.

## LION-Chunk

Chunkwise parallel form of full linear attention is kinda straight forward lets start by chunking the queries keys and values:


$$
\mathbf{Q}_{[i]} , \mathbf{K}_{[i]}, \mathbf{V}_{[i]}  \in \mathbb{R}^{C \times C}
$$

we should now shape the chunkwise form which consist of four parts:

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


Now that this has been stated and proven, we will describe how to construct the **chunkwise mask** from the attention mask $\mathbf{M}$, particularly for the **fixed** and **selective** masks in our framework. The chunkwise form of the mask for chunk of $i$ and $j$ is annotated as $\mathbf{M}_{[ij]}$.

### LION-D Chunk

For the fixed mask, we have:  

{% include figure.liquid loading="eager" path="assets/img/fixed.png"%}


with $\mathbf{L} \in \mathbb{R}^C$ and $\mathbf{\Gamma} \in \mathbb{R}^{C\times C}$ being the vector and matrix used for creating the chunked mask and they are only depending on the decay parameter $\lambda$ and the chunk size $C$. For the fixed decay mask we have  $\mathbf{L}_i = \lambda^i$. The chunkwise mask for chunk $i$ , $j$ can be written as:

$$
\mathbf{M}_{[ij]} = \mathbf{L}_{[i]} \frac{1}{\mathbf{L}_{[j]}} = \lambda^{|i-j|} \mathbf{L}_{[0]} \frac{1}{\mathbf{L}_{[0]}}.
$$

Similarly, for the upper triangular part:

$$
\mathbf{M}_{[ij]} = \lambda^{|i-j|} \frac{1}{\mathbf{L}^\top_{[0]}} \mathbf{L}_{[0]}.
$$

For diagonal chunks the mask is fixed an equal to 

$$\Gamma = \lambda^{|i-j|} $$ 

which is smaller version of the decay full mask $\mathbf{M}$.


### LION-S Chunk

We first partition the SSM (semiseparable) matrix into blocks of size $\mathtt{Q} \times \mathtt{Q}$.
Then, we use the properties of semiseparable matrices to factorize each off-diagonal block, which is low rank.

1. (*Orange*) Each diagonal block is a smaller semiseparable matrix; we can compute this multiplication however we like; in particular, using the quadratic (attention-like) form of SSD.
2. (*Green*) There are only $\mathtt{T} / \mathtt{Q}$ total different green blocks because many of them are shared. These can be computed with a batched matmul.
3. (*Yellow*) Notice that the yellow terms themselves form a 1-semiseparable matrix; in other words, this step is equivalently to an SSM scan (on some modified $A$ factors)!
4. (*Blue*) Similar to green, these can be computed with a batched matmul.

### SSD Algorithm: Chunking and State Passing

An alternative interpretation of the algorithm involves reasoning about how the SSM operates on the actual sequence.
We first split the sequence of input into blocks (or chunks) of size $\mathtt{Q}$.
The steps then have the interpretation
1. **Intra-chunk outputs**: compute the local output of each chunk (*what is the output per chunk supposing that the initial state (to the chunk) is 0?*)
2. **Chunk states**: compute the final state of each chunk (*what is the final state per chunk supposing that the initial state (to the chunk) is 0?*)
3. **Pass states**: compute a recurrence on all of the chunks' final states -- using any desired algorithm, e.g. parallel or sequential scan (*what is the actual final state per chunk taking into account all previous inputs?*)
4. **Output states**: for each chunk, given its true initial state (computed in Step 3), compute the contribution to the output just from the initial state

Either way, we see that most of the algorithm (Step 1, 2, and 4) leverages matmuls (and hence tensor cores), and also can be computed completely in parallel!
Only Step 3 requires a scan, but it operates on a much shorter sequence and usually only takes a small fraction of the time of the full algorithm.

### Special Cases

We note that special cases of this algorithm have been seen before. In particular RetNet<d-cite key="sun2023retentive"></d-cite>, which we showed in Part II to be a special case of SSD, mention a "chunkwise" algorithm which computes the quadratic form on a chunk of the input one-at-a-time and passes the final state to the next chunk.
This turns out to be essentially equivalent to the SSD algorithm specialized to a restricted case (i.e. a decay matrix mask $L$).
Our derivation comes from a different direction---the block matrix decomposition---which also makes it more obvious how to parallelize this algorithm and make it really fast in practice.

Other forms of "chunkwise" recurrences have recently become popular, such as in [Gated Linear Attention (GLA)](https://arxiv.org/abs/2312.06635)<d-cite key="yang2024gated"></d-cite>.

## The Code

In the "Minimal SSD" code that we provide in the paper and the [code release](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/ssd_minimal.py), we delineate each of these four steps.
As promised, this algorithm is not only faster but also much easier to implement than the original selective scan of Mamba,
coming in at just around 25 lines of code!

[//]: # <d-code block language="python">

```python
def segsum(x):
    """Naive segment sum calculation. exp(segsum(A)) produces a 1-SS matrix,
       which is equivalent to a scalar SSM."""
    T = x.size(-1)
    x_cumsum = torch.cumsum(x, dim=-1)
    x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum

def ssd(X, A, B, C, block_len=64, initial_states=None):
    """
    Arguments:
        X: (batch, length, n_heads, d_head)
        A: (batch, length, n_heads)
        B: (batch, length, n_heads, d_state)
        C: (batch, length, n_heads, d_state)
    Return:
        Y: (batch, length, n_heads, d_head)
    """
    assert X.dtype == A.dtype == B.dtype == C.dtype
    assert X.shape[1] % block_len == 0

    # Rearrange into blocks/chunks
    X, A, B, C = [rearrange(x, "b (c l) ... -> b c l ...", l=block_len) for x in (X, A, B, C)]

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A))
    Y_diag  = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp', C, states, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = rearrange(Y_diag+Y_off, "b c l h p -> b (c l) h p")
    return Y, final_state
```

[//]: # </d-code>


## The Details

Let's talk about a couple of additional details in the implementation (these don't even appear in the full paper, so pay attention!) that unpack some of the choices in this reference code.

### The SSM Scan

In the above code, we utilized the connection between scalar SSM recurrences

$$
h_{t+1} = A_t h_t + B_t x_t
$$

and matrix multiplication by 1-semiseparable matrices

$$
  L =
  \begin{bmatrix}
    1 & \\
    a_1 & 1 & \\
    a_2a_1 & a_2 & 1 \\
    \vdots & \vdots & \ddots & \ddots \\
    a_{\mathtt{T}-1}\dots a_1 & a_{\mathtt{T}-1}\dots a_2 & \dots & a_{\mathtt{T}-1} & 1 \\
  \end{bmatrix}
$$

which we covered in Part II (and Section 3.2.2 of the paper).
In this minimal implementation, we compute Step 3 of the algorithm, which is computing a scalar SSM by *any* algorithm of our choice,
by explicitly materializing a 1-SS matrix and doing dense matrix multiplication.

We use this version for several reasons:
1. Code-wise, it's simpler to materialize and multiply by this matrix than to actually implement a parallel associative scan
2. Because of the block decomposition of the SSM matrix, the sequence length $\mathtt{T}$ is reduced by a factor of $\approx 100$ -- so doing the scan in time $O(\mathtt{T}^2)$ instead of $O(\mathtt{T})$ isn't too bad
3. We have to materialize a 1-SS matrix anyways for Step 1 of the algorithm (the diagonal blocks), so might as well reuse the code ¯\\\_(ツ)\_/¯

While this example code is simpler and reasonably efficient on GPU (and probably TPU as well!), it's no longer truly linear at long sequences. Our more optimized Triton implementation does replace the 1-SS multiplication in Step 3 with an actual associative scan.

### Stability

#### Attempt 1: Ratios of cumprods
The first naive attempt may be to notice that the entries of this matrix are cumulative products 

$$
a_{i:j}^\times = a_i \times \cdots \times a_{j-1} = \frac{a_{i:\mathtt{T}}^\times}{a_{j:\mathtt{T}}^\times}
$$

However, this runs into severe numerical issues because these products can get really tiny (imagine $a_t \approx 0.9$ and powering it up for a sequence length $\mathtt{T}$ in the thousands!)


#### Fix 1: The Segment Sum (`segsum`) Operation

The second attempt would be to do all of this in log-space, because all the $a_t$ are positive; so the products become additions, and instead of `cumprod`s to deal with we have `cumsum`s instead. Then in order to compute the 1-SS matrix, we just have to compute the sums $\log a_i + \dots + \log a_{j-1}$ for every *segment* $[i:j]$. We call this the **segment sum (segsum)** primitive, analogous to cumulative sum (cumsum).

#### Attempt 2: Differences of cumsums


The obvious way to do this again is using the same idea as above, but in log space

$$
a_{i:j}^\times = \exp\left( \log a_i + \cdots + \log a_{j-1} \right) = \left( (\log a)_{i:\mathtt{T}}^+ - (\log a)_{j:\mathtt{T}}^+ \right)
$$

where we compute a single cumulative sum of $a$ along the time axis, and then compute all pairwise differences.
In code, we can do this with

```python
def segsum_unstable(x):
    """Naive segment sum calculation."""
    T = x.size(-1)
    x_cumsum = torch.cumsum(x, dim=-1)
    x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum
```

(and then the 1-semiseparable matrix is just the exponential of this output).

Sums/differences are a lot more stable than products/quotients, so this should work – right?

#### Fix 2: Remove All Subtractions

Unfortunately, it turns out this still doesn't work.
The values of this 1-SS matrix roughly represent the SSM dynamics, which are very sensitive to these values of $a_t$, so we have to be very precise.
And even in log space, these cumsums can be fairly large, which runs into [catastrophic cancellation](https://en.wikipedia.org/wiki/Catastrophic_cancellation) when subtracted. So we really have to find a way to compute this matrix with only additions, while still vectorizing everything…

#### Attempt 3: Stable Segsum

This leads to the helper function in the reference SSD code.
Instead of computing a single cumsum and then subtracting, we find a way to use a batch of independent cumsums that immediately produces the right answer without subtraction.

These details do matter! Without the right implementation of these primitives, the basic SSD algorithm produces NaNs immediately during training (even with FP32).

### Discretization
This lineage of structured state space models developed from [S4](https://arxiv.org/abs/2111.00396) and [its](https://arxiv.org/abs/2110.13985) [predecessors](https://arxiv.org/abs/2008.07669) which were viewed as continuous-time systems.<d-cite key="gu2023thesis"></d-cite><d-cite key="gu2022efficiently"></d-cite><d-cite key="gu2021combining"></d-cite><d-cite key="gu2020hippo"></d-cite>

In Mamba, however, we don't really view the SSM as continuous anymore.
In fact, as mentioned in the Discussion (Section 5) of the [original paper](https://arxiv.org/abs/2312.00752), Mamba trades off with S4 on modeling different types of data:
* S4 is a continuous-time model that excels at modeling continuous data, e.g. perceptual signals such as audio waveforms and pixel-level vision.
* Mamba is a discrete-time model that excels at modeling discrete data, e.g. tokenized data such as language.

However, the parameterization of Mamba still used the same discretization step as in prior structured SSMs, where there is another parameter $\Delta$ being modeled. We do this because the discretization step has other side effects such as properly normalizing the activations <d-cite key="gu2023train"></d-cite><d-cite key="orvieto2023resurrecting"></d-cite> which is important for performance.

The initializations and parameterizations from the previous [theory on structured SSMs](https://arxiv.org/abs/2206.12037) still work out-of-the-box, so why fix what's not broken?

Despite this, we're pretty sure that the discretization step isn't really necessary for Mamba.
In the Mamba-2 paper, we chose to work directly with the "discrete parameters" $A$ and $B$, which in all previous structured SSM papers (including Mamba-1) were denoted $(\bar{A}, \bar{B})$ and defined through an additional transformation

$$
\begin{align*}
\bar{A} &= \exp(e^{\Delta A}) \\
\bar{B} &= (\exp(e^{\Delta A}) - I) A^{-1} B
\end{align*}
$$

This doesn't pose any problems: to use the continuous SSM parameterization, simply transform the parameters through the above formulas before plugging into the SSD code above.

In the full Mamba-2 code, we also kept the same parameterization and discretization step as in Mamba---again, why fix what's not broken?---but hypothesize that "discrete-centric" variants
(such as the *gamma normalization* of [LRU](https://arxiv.org/abs/2303.06349)<d-cite key="orvieto2023resurrecting"></d-cite> and [Griffin](https://arxiv.org/abs/2402.19427)<d-cite key="de2024griffin"></d-cite>)
should work equally well.

> #### Is Discretization Necessary?
>
> It's useful for other structured SSMs, but perhaps not needed for Mamba. But it's just a simple invertible transformation, so use either discrete or continuous parameterizations as you like! 
{: .block-tip}

## What's Next
In the [final part of this series]({% post_url 2024-05-31-mamba2-part4-systems %}), we'll continue talking about the implementation of Mamba-2, but on a more macroscopic level; about the entire neural network, instead of just details of the core SSD layer.

We'll also talk about the actual speed of the algorithm covered in this post.

