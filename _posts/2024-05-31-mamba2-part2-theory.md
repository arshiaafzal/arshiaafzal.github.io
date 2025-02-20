---
layout: distill
title:  LION 🦁 Part II - Bi-directional RNN
description: 
tags:
giscus_comments: false
date: 2025-02-20
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
  - name: Finding Bidirectional RNN Equal to Full Linear Attention
  - name: Some Important details of our RNN
  - name: Different Masks of LION
    subsections:
      - name: LION-🔥 
      - name: LION-D
      - name: LION-S
  - name: LION Attention Block

---

1. [Part I - Full Linear Attention]({% post_url 2024-05-31-mamba2-part1-model %})
2. Part II - Bi-directional RNN
3. [Part III - Chunkwise Parallel from of LION]({% post_url 2024-05-31-mamba2-part3-algorithm %})
4. [Part IV - Results]({% post_url 2024-05-31-mamba2-part4-results %})


In [Part I]({% post_url 2024-05-31-mamba2-part1-model %}) of this series, we defined full linear attention with masking and scaling.  
Similar to all linear transformers designed for causal sequence modeling, we aim to derive an RNN form for efficiency during inference.  
In this section, we establish and theoretically demonstrate the equivalent bidirectional RNN for the Linear Transformer.



## Finding Bidirectional RNN Equal to Full Linear Attention

Let's start by separating the upper, lower, and diagonal elements of the attention matrix and the mask. Since the idea of a bidirectional RNN is to process the sequence in both the forward order (from first to last) and the reverse order (from last to first), these naturally correspond to the upper and lower parts of the attention matrix and mask.

Ideally, we aim to construct an RNN that is equivalent to the masked and scaled Linear Attention. Let's start by seperating upper and lower parts of the attention and mask:

{% include figure.liquid loading="eager" path="assets/img/att_mask_color.png"%}

**Note:** We made a strong effort to maintain a consistent color coding for this section of the blog post and throughout our paper :).  

- Wherever you see a <span style="background-color: rgb(255, 248, 203); padding: 3px; color:black">Yellow</span> color, it indicates the **upper part of the matrix (non-causal)**.  
- Whenever you see a <span style="background-color: rgb(254, 200, 201); padding: 3px; color:black">Red</span> color, it represents the **diagonal elements**.  
- Whenever you see a <span style="background-color: rgb(208, 243, 248); padding: 3px; color:black">Blue</span> color, it corresponds to the **lower triangular (causal) part**.  

Let's seperate the attention into upper and lower parts:

{% include figure.liquid loading="eager" path="assets/img/att_sep.png"%}

This formulation represents both the causal and non-causal forms of attention. Ideally, we aim to model each triangular part using an RNN.Similarly, we can also separate the mask in the same way:

{% include figure.liquid loading="eager" path="assets/img/mask_sep.png"%}

Let's also write the scaling part of the masked attention $\mathbf{Y} = \text{Scale}(\mathbf{Q} \mathbf{K}^\top \odot \mathbf{M} ) \mathbf{V}$ as:

$$
\begin{aligned}
   \mathbf{Y} = \big(\text{scale}(\mathbf{Q}\mathbf{K}^{\top} \odot \mathbf{M})\big) \mathbf{V}  
    = (\mathbf{C}^{-1}(\mathbf{Q}\mathbf{K}^{\top} \odot \mathbf{M}))\mathbf{V}, \hspace{1mm}
   \mathbf{C}_i = \mathbf{q}^{\top}_i\sum\limits_{j=1}^{L} \mathbf{M}_{ij}\mathbf{k}_j. 
\end{aligned} 
$$

Also, we can decompose the scaling matrix $$\mathbf{C}_i$$ as:

$$
\begin{aligned}
\mathbf{C}_{i}=
  \underbrace{\mathbf{q}^{\top}_i\sum\nolimits_{j=1}^{i} \mathbf{M}_{ij}\mathbf{k}_j - \frac{1}{2} \mathbf{q}^{\top}_i\mathbf{k}_i}_{\mathbf{C}^F_i} + \underbrace{\mathbf{q}^{\top}_i\sum\nolimits_{j=i}^{L} \mathbf{M}_{ij}\mathbf{k}_j - \frac{1}{2} \mathbf{q}^{\top}_i\mathbf{k}_i}_{\mathbf{C}^B_i}
\end{aligned} 
$$

Now we replace tha bove scaling matrix $\mathbf{C}$ in the output of the attention form of $\mathbf{Y} = \text{Scale}(\mathbf{Q} \mathbf{K}^\top \odot \mathbf{M} ) \mathbf{V}$ .Interestingly, many terms naturally cancel out with each other.

{% include figure.liquid loading="eager" path="assets/img/proofC.png"%}

This results in only the forward and backward directions of the RNN remaining. As observed, the forward path aligns with causal linear attention with masking. Now, we need to demonstrate that the backward path follows the same RNN structure in the reverse direction. We can simply flip the upper triangular matrices using the [exchange matrix](https://en.wikipedia.org/wiki/Exchange_matrix) $$\mathbf{J}_L$$ and the function $$F(X) = \mathbf{J}_L X \mathbf{J}_L$$:

{% include figure.liquid loading="eager" path="assets/img/flip.png"%}

Cool! Now, both the upper part (equivalent to the RNN in the forward direction) and the lower part (equivalent to the RNN in the backward direction) can be formulated as RNNs. This is exactly what we need to construct our bidirectional RNN equivalent to full linear attention.

> **LION: Reccurence form**
> 
> $$ \begin{aligned} \mathbf{S}_i^{F/B} &= \lambda_i \mathbf{S}^{F/B}_{i-1} + \mathbf{k}_i \mathbf{v}_i^{\top}, \\ 
\mathbf{z}^{F/B}_i &= \lambda_i \mathbf{z}^{F/B}_{i-1} + \mathbf{k}_i,  \\
c^{F/B}_i & = \mathbf{q}_i^{\top} \mathbf{z}^{F/B}_{i} - \frac{\mathbf{q}_i^{\top} \mathbf{k}_i}{2},  \\
\mathbf{y}^{F/B}_i &= \mathbf{q}_i^{\top} \mathbf{S}^{F/B}_i - \frac{\mathbf{q}_i^{\top} \mathbf{k}_i}{2} \mathbf{v}_i, \\ 
out&put: \mathbf{y}_i = \frac{\mathbf{y}^{F}_i + \mathbf{y}^{B}_i}{c^F_i + c^B_i}. \\ \end{aligned} $$
{: .block-tip}


The terms $\frac{\mathbf{q}_i^{\top} \mathbf{k}_i}{2}$ and $\frac{\mathbf{q}_i^{\top} \mathbf{k}_i}{2}$ are subtracted to avoid double counting. This bi-directional RNN is equivalent to scaled and masked linear attention described in previous section of this blogpost.


## Some Important details of our RNN

> Only the states $$c^{F/B}_i$$ and $$\mathbf{y}^{F/B}_i$$ are stored per token, resulting in $$\mathcal{O}(Ld)$$ memory usage. In contrast, naively storing full matrix-valued hidden states would require $$\mathcal{O}(Ld^2)$$, which becomes infeasible for large models.

> Forward and backward recurrences run independently, completing in $$L$$ time steps with $$L$$ memory units, compared to $$2L$$ in the naive approach. 

{% include figure.liquid loading="eager" path="assets/img/memory.png" title="Memory Allocation of LION in RNN form" caption="Memory allocation in LION during Forward and Backward recurrences." %}


All in one we can visulaize our framework nicely like:

{% include figure.liquid loading="eager" path="assets/img/frlion.png" title="LION" caption="LION 🦁: Our framework for training in parallel using Full Linear Attention which also supports the efficient bi-directional RNN format." %}

## Different Masks of LION

Now that we have created our framework let's see what are the choices of the decay factor $$\lambda_i$$ and how they resemble the famous linear Transformer models. Let's set:

> $\lambda_i=1$ this results in mighty simple Linear Transformer (cite) which we refrer to as <span style="background-color: rgb(230, 255, 230); padding: 3px; color:black">LION-🔥 </span> which is LION-Lit resembling Linear Transformer.

> $\lambda_i=\lambda$ this results in mighty RetNet (cite) which we refrer to as <span style="background-color: rgb(229, 204, 230); padding: 3px; color:black">LION-D </span>

> $\lambda_i=\sigma(\mathbf{W}\mathbf{x}_i)$ being input dependent, and bi-directional Linear Transformer inspired by selectivity of Mamba2 (cite) which we refrer to as <span style="background-color: rgb(255, 233, 211) ; padding: 3px; color:black">LION-S </span>


We evaluate all above models, extended to bidirectional sequence modeling using LION, on several bidirectional tasks. Also as all Linear Transformers use feature mapping $\phi(.)$ to queries and keys we also applied SILU shifted $\phi(x)=$ `(SILU(x)+0.5)/(norm(SILU(x)+0.5))` non-linear activation function. Let's delve deep in each of these models in LION framework.

### LION-🔥 

LION-🔥 is an extension of the very first Linear Transformer (cite). Without any masking, the bidirectional parallel form can be simply written as:

$$\mathbf{Y} = Scale(\mathbf{Q} \mathbf{K}^\top )\mathbf{V} $$

and the RNN form of the above parallel full linear attention is simply the RNN form mentioned above in this section in green box just by simply not using any mask.


### LION-D

By fixing $$\lambda_i = \lambda$$, the mask $$\mathbf{M}$$ has the form:

$$
\begin{align}
    \mathbf{M}_{ij} = \lambda^{|i-j|}, \quad \mathbf{D}_{ij} = |i-j|\log(\lambda), \quad \mathbf{M} = \exp(\mathbf{D}). \notag
\end{align}
$$

$$\mathbf{M}$$ above is a Toeplitz mask cite(tnn) and therefore, creating the decay mask can be made even faster using simple PyTorch commands. To ensure numerical stability, we bound the parameter $$\lambda$$ using the **sigmoid function**, setting $$\lambda = \sigma(a)$$. Without this constraint, the scalar $$\lambda^L$$ could become excessively large, leading to instability. Additionally, as we all know, summation is generally more numerically stable than multiplication. Therefore, in some cases, instead of multiplying a matrix repeatedly, we can leverage summation for improved stability. However, in practice, for **RetNet-style masks** with a fixed decay, multiplication remains stable. This allows for a more straightforward implementation when generating the mask in code:

```python
def Decay_Mask(a , L):
    idx = torch.arange(L,device=a_i.device)
    I, J = torch.meshgrid(idx, idx, indexing='ij')
    E = (torch.abs((I-J)).float().view(1,1,L,L))
    M = torch.sigmoid(a).view(1,-1,1,1)**E
    return M
```


### LION-S

Observing the structure of $\mathbf{M}$, its upper ($\mathbf{M}^B$) and lower ($\mathbf{M}^F$) triangular parts are rank-1 [semi-separable matrices](https://people.cs.kuleuven.be/~raf.vandebril/homepage/publications/papers_html/qrq_07/node16.html) (cite), allowing for efficient computation via matrix multiplications.  

During training, the decay factors $\lambda_i$ are stacked into ${\lambda}^F \in \mathbb{R}^L$, and the cumulative product  

$$
\mathbf{L}^F = cumprod(\lambda^F) = \prod_{k=0}^{i} \lambda^F_k
$$

is used to generate the lower triangular mask \(\mathbf{M}^F\). For the upper triangular mask $\mathbf{M}^B$, the input sequence is flipped, and the decay factors are computed as  

$$
\boldsymbol{\lambda}^B = \text{Flip}(\boldsymbol{\lambda}^F), \quad \mathbf{L}^B = cumprod(\boldsymbol{\lambda}^B).
$$

The masks are then constructed as,  $\mathbf{M}^F =$ `tril(LF@inv(LF)^T)` for the forward part and  $\mathbf{M}^B =$ `triu(LB@inv(LB)^T)` for the backward part.
where `tril(.)` and `trilu(.)` extract the lower and upper triangular parts of the input matrix respectively. The full mask is then obtained as  

$$
\mathbf{M} = \mathbf{M}^F + \mathbf{M}^B - \mathbf{I}.
$$  

To improve numerical stability, the selective scalar $\lambda_i$ is designed in exponential form  

$$
\lambda_i = e^{a_i}.
$$ 

This results in the cumulative sum:  

$$
\mathbf{D}^F_{ij} = 
\begin{cases} 
\sum_{k=i}^{j+1} a_k, & \text{if } i > j,  \\
\sum_{k=i+1}^{j} a_k, & \text{if } i < j,  \\
0, & \text{if } i = j,
\end{cases}
$$

$$
\mathbf{M^F} = \exp(\mathbf{D^F}),
$$

where $\exp(\cdot)$ is applied element-wise. The same process applies to $\mathbf{M}^B$ by flipping the input sequence order.  

Here, $\mathbf{D}^{F/B} = cumsum(\mathbf{a}^{F/B})$, where $\mathbf{a} \in \mathbb{R}^L$ contains the selective exponents $a_i$.  

Ensuring stability is crucial, as $\mathbf{L}^{F/B}$ can overflow or underflow when forming the full mask without chunking. To mitigate this, we define  

$$
a_i = \log(\sigma(\mathbf{W}_{a}^\top\mathbf{x}_i + b)),
$$

where $\sigma(.)$ is the sigmoid function. This approach ensures numerical stability by bounding $a_i$ within the interval $[0,1]$. 

**Note:** It is crucial that the activation function is a **sigmoid**, as other activations do not produce stable masks and can lead to NaN values in the loss function. To maintain stability, **chunking** is required during training. This issue has been specifically highlighted in the **Mamba2** blog post.  
We provide a detailed explanation in the **Results** section of this blog post, where we discuss why using **full attention** is beneficial for achieving **high throughput** during training.

The code for building the mask of LION-S is so simple and flexible even in Pytorch:

```python
def create_causal_mask_lions(tensor):
    cumsum = torch.cumsum(tensor, dim=-1 )
    cumprod = torch.exp(cumsum)
    A = torch.matmul(cumprod.unsqueeze(-1) , 1/ ( cumprod.unsqueeze(-1).transpose(-1,-2) + 1e-7 )  )
    return torch.tril(A)
```

```python
def Selective_Mask(vec):
    vec_shape = vec.shape
    A_for = create_matrix_from_tensor(vec.unsqueeze(-1).transpose(-1,-2)).squeeze()
    A_back = create_matrix_from_tensor(torch.cat((vec,torch.ones((vec_shape[0],vec_shape[1],1),device=vec.device)),dim=-1)[:,:,1:].unsqueeze(-1).transpose(-1,-2)).transpose(-1,-2).squeeze()
    return A_for + A_back - torch.eye(A_for.shape[-1]).to(A_for.device)
```

## LION Attention Block

We can formulate the parallel attention form of LION as shown below, supporting all three extensions of our main experiments:

```python
Class LION_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., silunorm: bool = False, Mask_type):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.Mask_type = Mask_type

        if Mask_type == 'Lit':
            None
        if Mask_type == 'Selective':
          self.a_i = nn.Linear(dim, num_heads)
        if Mask_type == 'Decay':
          self.a_i = nn.Parameter(torch.randn(num_heads))

        self.non_lin = silu_shifted
        self.silunorm = silunorm

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = self.non_lin(qkv[0], silunorm=self.silunorm), self.non_lin(qkv[1], silunorm=self.silunorm), qkv[2]

        a_i = self.a_i(x).transpose(-1,-2)
        a_i = torch.log(1 - torch.nn.functional.sigmoid(a_i) + 1e-7)

        if self.Mask_type == 'Selective':
          M = Selective_Mask(a_i)

        if self.Mask_type == 'Decay':
          M = Decay_Mask(a_i)

        if self.Mask_type == 'Lit':
          M = 1

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = M * attn

        # Scaling
        attn = torch.log(attn + 1e-7)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
```


> **Question:** As seen above, the **RNN** is more efficient than the **Transformer** since it only requires storing the output for each token, resulting in a memory complexity of **$\mathcal{O}(Ld)$**, as opposed to storing the full attention matrix, which requires **$\mathcal{O}(L^2 d)$**.  Can we achieve a balance between the speed of attention parallelism and the efficiency of an RNN?


We will answer this question in our next section by introducing LION-Chunk.


## Next Up

- In the next section of this series, we will describe how to apply a **chunkwise parallel form** for LION, allowing us to balance between the *RNN structure* and the *attention-based* formulation.

- We show the numercial results and experiments on Imagenet and C4 dataset :)
