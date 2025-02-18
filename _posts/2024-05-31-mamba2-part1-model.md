---
layout: distill
title: LION 🦁 Part I - Full Linear Attention
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
      name: EPFL
  - name: Elias Abad Rocamora
    url:
    affiliations:
      name: EPFL
  - name: Leyla Naz Candogan
    url:
    affiliations:
      name: EPFL
  - name: Pol Puigdemont Plana
    url:
    affiliations:
      name: EPFL
  - name: Francesco Tonin
    url:
    affiliations:
      name: EPFL
  - name: Yongtao Wu
    url:
    affiliations:
      name: EPFL
  - name : Volkan Cevher
    url:
    affiliations:
      name: EPFL

bibliography: albert.bib


toc:
  - name: From Causal to Full Linear Attention
  - name: Creating Scaled and Masked Full Attention
 
---

{% include figure.liquid loading="eager" path="assets/img/lion.jpg" %}


[[Paper](https://arxiv.org/abs/2405.21060)]
[[Code](https://github.com/state-spaces/mamba)]


**We sincerely appreciate Albert Gu and Tri Dao for their insightful blog posts, which have been invaluable in shaping our own!**

1. Part I - Full Linear Attention
2. [Part II - LION: Bi-directional RNN for Full Linear Attention]({% post_url 2024-05-31-mamba2-part2-theory %})
3. [Part III - LION Chunk: CHunkwise Parallel of LION]({% post_url 2024-05-31-mamba2-part3-algorithm %})
4. [Part IV - The Systems]({% post_url 2024-05-31-mamba2-part4-systems %})

Recently, Transformers with Linear Attention and State Space Models (SSMs) have gained significant popularity for causal sequence modeling due to their ability to efficiently support both parallel training and RNN-like inference. These models have demonstrated impressive accuracy in causal tasks, particularly in causal language modeling. However, their evaluation in bi-directional sequence modeling, such as image classification and masked language modeling, has been relatively limited. In contrast, SSMs, particularly Mamba, have been recently evaluated in vision tasks, including models like Vision Mamba and Hydra, which represent official extensions of Mamba for bi-directional sequence modeling.

We’re curious to explore whether Linear Attention Transformers, including the simple Linear Transformer and RetNet or simple selctive varient, can perform effectively on bi-directional sequence modeling. Or more precicley what modifications are needed to adapt them for tasks like image classification and masked language modeling? 😊

Let’s break this down with three key questions:

### Question 1 (Applicability)

Given that Linear Transformers can be formulated as RNNs and offer efficiency benefits during inference, alongside parallel training for causal sequence modeling, can they also exhibit similar benefits for bi-directional processing? If so, what would the parallel form look like, and what would be the equivalent bi-directional RNN form?

### Question 2 (Performance)

Assuming we’ve addressed the first question, can simple Linear Transformers—such as Linear Trans (cite), RetNet (cite), or even basic linear attention with a selective decay factor—perform well on bi-directional tasks, such as image classification or masked language modeling?

### Question 3 (Training Throughput)

While bi-directional SSMs like Hydra and Vision Mamba show impressive performance on bi-directional sequence modeling tasks, they tend to be difficult and slow to train compared to Transformers with full attention (e.g., ViT and BERT). If we’ve answered the first two questions affirmatively, can Linear Transformers match the accuracy of deep bi-directional SSMs while maintaining the training throughput of softmax Transformers and the inference efficiency of RNNs/SSMs? Also, maybe we can achive this without need for CUDA kernel programming and simply using torch ;)


## From Causal to Full Linear Attention

Let's start with Linear Attention Reccurence:

$$
\begin{aligned} 
& S_i = S_{i-1} + k_i v^\top_i, \quad z_i =  z_{i-1} + k_i, \\
& Scaled: y_i = \frac{q^\top_i S_i}{q^\top_i z_i}, \quad Non-Scaled: y_i  = q^\top_i S_i \\ 
\end{aligned}
$$

Above is the RNN form of the Linear Attention which have the parallel form of:

$$\mathbf{Y} = Scale \left(\mathbf{Q} \mathbf{K}^\top  \odot \mathbf{M}^C \right)$$


and the mask $\mathbf{M}^C$ is a lower triangular binary matrix. Causal Linear Transformers are a class of models introduced following the development of Linear Transformers as shown above (cite). These models typically define a recurrence of the form:  

$$
\begin{aligned} 
S_i = \boldsymbol{\Lambda_i} \star S_{i-1} + \gamma_i k_i v^\top_i, \quad z_i = \boldsymbol{\Lambda_i} \star z_{i-1} + \gamma_i k_i, \\
Scaled: y_i = \frac{q^\top_i S_i}{q^\top_i z_i}, \quad Non-Scaled: y_i  = q^\top_i S_i \\ 
\end{aligned}
$$

Here, $$\boldsymbol{\Lambda_i}$$ and $$\gamma_i$$ are decay factors introduced after Linear Transformers to enhance their performance. (Spoiler alert ⚠️: this family of Linear Transformers has strong connections to SSMs, as explored in works like (DeltaNet) and (Mamba-2) 😉). For simplicity, we consider $$\boldsymbol{\Lambda_i} = \lambda_i$$ as a scalar in this study. As shown, this choice is as effective as the full matrix form. We now present the general scaled linear attention in the following form:

$$
\begin{aligned} 
S_i &= \lambda_i  S_{i-1} + \gamma_i k_i v^\top_i,\\
z_i &= \lambda_i  z_{i-1} + \gamma_i k_i, \\
y_i &= \frac{q^\top_i S_i}{q^\top_i z_i} \\ 
\end{aligned}
$$

The first goal is to extend the causal linear attention parallel form  

$$
\mathbf{Y} = \text{Scale} \left(\mathbf{Q} \mathbf{K}^\top  \odot \mathbf{M}^C \right)
$$

to a fully *scaled* and *masked* attention mechanism for linear attention.

## Creating Scaled and Masked Full Attention

The first step is quite simple: the masked and scaled attention can naturally take the following form, as suggested by its name:

> **Full Linear Attention**
> 
> $$ \mathbf{Y} = \text{Scale} \left(\mathbf{Q} \mathbf{K}^\top  \odot \mathbf{M} \right)$$
{: .block-tip}

The important part is how to well define the matrix $$\mathbf{M}$$. A natural choice is to extend the causal mask $$\mathbf{M^C}$$, where the causal mask between tokens $$i,j$$ is given by $$\mathbf{M}^C_{ij} = \lambda_{j+1} \lambda_{j+2} \dots \lambda_i$$, representing the product of all selective scalers between $$i$$ and $$j$$. In the bidirectional case, the full mask should preserve this property. Since this is indeed a desirable property, one can interpret it as a form of relative positional encoding between two tokens. Saying so the mask cen be shaped as:

$$
\begin{aligned}
    \mathbf{M}_{ij} = 
    \begin{cases} 
    \Pi_{k=j+1}^{i}{\lambda_k}, & i > j  \\
    1 & i=j\\ 
    \Pi_{k=i+1}^{j}{\lambda_k}, & i < j.
\end{cases} 
\end{aligned}
$$


To recap, the full output of full LInear Attention can be presented as:

<span style="font-size: 0.8em;">
$$
\mathbf{Y} = Scale
    \left(
   \underbrace{\left( \renewcommand*{\arraystretch} \begin{array}{ccccc}
       \mathbf{q}_1^{\top}\mathbf{k}_1  &  \mathbf{q}_1^{\top}\mathbf{k}_2 & \cdots &  \mathbf{q}_1^{\top}\mathbf{k}_L \\
     \mathbf{q}_2^{\top}\mathbf{k}_1  &   \mathbf{q}_2^{\top}\mathbf{k}_2  &   \cdots &  \mathbf{q}_2^{\top}\mathbf{k}_L\\
     \vdots &  \vdots & \ddots  &  \vdots \\
      \mathbf{q}_L^{\top}\mathbf{k}_1 &   \mathbf{q}_L^{\top}\mathbf{k}_2  &   \cdots  & 
     \mathbf{q}_L^{\top}\mathbf{k}_L\\
  \end{array} \right)}_{\hspace{1mm} \mathbf{A} = \mathbf{Q} \mathbf{K}^{\top}}  \odot
   \underbrace{ \left(  \renewcommand*{\arraystretch} \begin{array}{ccccc}
    1  & \lambda_2 & \lambda_2 \lambda_3  & \cdots & \lambda_2 \cdots \lambda_L \\
    \lambda_1 &  1 & \lambda_3 & \cdots & \lambda_3 \cdots \lambda_L \\
    \lambda_1 \lambda_2 & \lambda_2 & 1 & \cdots & \lambda_4 \cdots \lambda_L \\
    \vdots & \vdots & \vdots & \ddots &  \vdots \\
    \lambda_{L-1} \cdots \lambda_1 & \lambda_{L-1} \cdots \lambda_2 & \lambda_{L-1} \cdots \lambda_3 & \cdots &   1 \\   
\end{array}  \right)  }_{\hspace{1mm} \mathbf{M}}  \right) \left( \renewcommand*{\arraystretch} \begin{array}{c}
    \mathbf{v}_1^\top \\  
    \mathbf{v}_2^\top \\
    \mathbf{v}_3^\top \\  
    \vdots \\
    \mathbf{v}_L^\top \\   
  \end{array} \right)
$$
</span>


The above represents the full **Li**near Attenti**on** in parallel form, which also inspired the name of our framework, **LION** 🦁. Now that we have established full linear attention for bidirectional sequence modeling, it's time to derive its equivalent bidirectional RNN.

## Next Up  

- We introduce our framework, **LION**, which derives an equivalent bidirectional RNN for full linear attention.  

- Within this framework, we demonstrate how various Linear Transformers can be extended to their bidirectional counterparts.  

- We explore the construction of stable masks \(\mathbf{M}\), enabling models using LION to:  
  - Train in parallel using full attention.  
  - Infer efficiently like an RNN.  

- Finally, we introduce a **chunkwise parallel** variant of LION to balance recurrence and parallelism 🙂.




