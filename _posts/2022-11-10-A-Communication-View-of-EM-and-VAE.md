---
title: 'Viewing EM-Style Generation as Communication'
date: 2022-11-10
permalink: /posts/2022/11/Viewing-EM-Style-Generation-as-Communication/
tags:
  - Math
  - Theory
  - Generation
  - Machine-Machine Communication
  - EM
  - VAE
---
## Table of Contents
- [Table of Contents](#table-of-contents)
- [Viewing EM-Style Generation as Communication](#viewing-em-style-generation-as-communication)
  - [Gentle Recap of the Communication Objective](#gentle-recap-of-the-communication-objective)
  - [Derivation of EM](#derivation-of-em)
    - [Bringing Up the Underlying Information (Latent)](#bringing-up-the-underlying-information-latent)
    - [Explicit `E-Step` and `M-Step`](#explicit-e-step-and-m-step)
  - [Interpretation](#interpretation)
  - [Cite this blog](#cite-this-blog)

## Viewing EM-Style Generation as Communication

### Gentle Recap of the Communication Objective
In my [previous post](https://ja1zhou.github.io/posts/2022/10/Emergent-Communication/), I have shared a framework of inspecting the subject of `Communication`. Here is a recap.

$$\mathcal{L}_{\theta, \phi} = \underbrace{\mathcal{H}(X|M_\theta)}_{\mathcal{L}_\text{info}} + \underbrace{\mathbb{E}_{m\sim\pi_\theta }D_{KL}(\rho^\theta(\cdot|m)||\rho_\phi(\cdot|m))}_{\mathcal{L}_{\text{commonsense}}}$$

In a  categorical reference game with typical `log likelihoods`, the loss function $\mathcal{L}$ can be decomposed into 
- maximizing information gain (as seen in the `entropy` term)
- minimizing `commonsense` between speaker and listener (this is the way I put it, as seen in the `KL Divergence` term)

It should be obvious that the `Lewis Game` is a special case of `EM-based` generations. In the derivation of `EM` and `VAE` algorithms, I have spotted striking similarities. I try to incorporate them into `generalized communication` in this blog. The derivations in this blogs should be consistent with other available sources, yet I would use `terms` that are not necessarily consistent with their existing usages. I use these terms only to exposit on my insights.

### Derivation of EM
#### Bringing Up the Underlying Information (Latent)
`EM` starts with `MLE`. Suppose we have observed data $\{x_i\}_{i=1}^N$, and that $x\sim p(x)$ where $p(x)$ is the underlying distribution. We aim to fit a distribution parameterized on $\theta$ to describe $p(x)$:

$$\theta_{MLE}=\argmax_{\theta}\sum_{i=1}^{N}\log P(x_i|\theta) =\argmax_{\theta}\log P(\bold x|\theta) \\
x_i\overset{\text{i.i.d}}{\sim}p(x)$$


Traditionally, we introduce the idea of `latent vector` to help approximate $p(x)$. Here, we say that $p(x)$ needs only to be characterized by `limited features`. That is, $x$ could be (ideally) faithfully reconstructed by `communicating` $z$ instead. For this reason, we refer to $z$ as the `information` of $x$, and $x$ as its `full form`. 

$$p(x_i|\theta)=\int_{z_i}p(x_i,z_i|\theta)dz_i=\int_{z_i}p(x_i|z_i,\theta)p(z_i|\theta)dz_i$$

Suppose that $z$ can be sufficiently and more easily described by $\theta$, it follows that iterating all possible `essential information` $z$ and `decoding` it back to $x$ would give us the distribution of $x$. Therefore, the crucial assumption here is that $p(x|\theta)$ is redundant, and could be instead sufficiently describe using $p(z|\theta)$.

For simplicity, we omit the `\bold` command in formulas. To learn $z$ well, we need to introduce interaction between $z$ and $x$. Generally, we have

$$p(x|\theta)=\frac{p(x,z|\theta)} {p(z|x,\theta)}$$

$p(z|x, \theta)$ `encodes` the inputs to `information`. 

`Expectation` over $z$ would not impact $x$. Moreover, we would like to derive an `iterative optimization`. To this end, it may be reasonable to perform an expectation over $p(z|x, \theta^{(t)})$. Note that for the left hand side, expectation over $z$ does not have an impact on $x$. It is still $\log p(x|\theta)$. 

$$\mathbb{E}_{p(z|x,\theta^{(t)})}\left[\log p(x|\theta)\right]=\mathbb{E}_{p(z|x,\theta^{(t)})}\left[\log p(x,z|\theta)\right]-\mathbb{E}_{p(z|x,\theta^{(t)})}\left[\log p(z|x,\theta)\right]\\=Q(\theta,\theta^{(t)})-K(\theta,\theta^{(t)})$$

Obviously, for $\forall \theta$, we have 

$$\begin{aligned}K(\theta,\theta^{(t)}) &= -D_{KL}(p(z|x,\theta^{(t)})||p(z|x,\theta)) - H(p(z|x,\theta^{(t)}))\\ &\le K(\theta^{(t)},\theta^{(t)})\end{aligned}$$

Where $H(p(z|x,\theta^{(t)})$ denotes the entropy. Suppose we can calculate $\theta^{(t+1)}$ as: 

$$\argmax_{\theta}Q(\theta,\theta^{(t)})$$

It follows that:

$$\log p(x|\theta^{(t+1)})\ge\log p(x|\theta^{(t)})$$

Therefore, we have to calculate 

$$\theta^{(t+1)}=\argmax_\theta\int_z\log p(x,z|\theta)\cdot p(z|x,\theta^{(t)})dz$$

It seems that no `estimation` is present. Yet, the above scenario is solvable based on strong inductive bias (i.e. GMM), which is essentially `estimation`. Here, $p(x,z|\theta)$ is tractable through $p(z|\theta)\cdot p(x|z, \theta)$. $p(z|x, \theta^{(t)})$ also has closed form. 

#### Explicit `E-Step` and `M-Step`
It is not always tractable to give a closed-form distribution of $p(z|x,\theta^{(t)})$ and $p(x|z, \theta)$, say that we implicitly `estimate` them using neural nets $p(x|z, \theta):\text{decoder}$ and $q(z|x, \phi):\text{encoder}$. Intuitively, they match the `speaker` and `listener` in reference games. We instead get:



$$\log p(x|\theta)=\mathbb{E}_{q(z|x,\phi)}\left[\log p(x,z|\theta)\right]-\mathbb{E}_{q(z|x,\phi)}\left[\log p(z|x,\theta)\right]\\=D_{KL}\left(q(z|x,\phi)||p(z|x,\theta)\right)+\int_z q(z|x,\phi)\log\frac{p(x,z|\theta)}{q(z|x,\phi)}dz$$

Here is what is tricky. In the explicit `E` step, we are actually minimizing the `KL Divergence` in the first term.

$$\phi^{(t+1)}=\argmin_\phi D_{KL}\left(q(z|x,\phi)||p(z|x,\theta^{(t)})\right)$$



It is obvious that the latter term is `ELBO`. The `M` step follows:

$$\theta^{(t+1)}=\argmax\int_z q(z|x,\phi^{(t+1)})\log\frac{p(x,z|\theta)}{q(z|x,\phi^{(t+1)})}dz$$

`ELBO` can be further expanded:

$$\begin{aligned}\mathbb{E}_{q(z|x,\phi)}\left[ \log\frac{p(x,z|\theta)}{q(z|x,\phi)} \right]&=\mathbb{E}_{q(z|x,\phi)}\left[ \log\frac{p(x|z,\theta)p(z|\theta)}{q(z|x,\phi)} \right] \\&=\mathbb{E}_{q(z|x,\phi)}\left[ \log p(x|z,\theta) \right]+\mathbb{E}_{q(z|x,\phi)}\left[\log\frac{p(z|\theta)}{q(z|x,\phi)}\right] \\&=\mathbb{E}_{q(z|x,\phi)}\left[ \log p(x|z,\theta) \right]-D_{KL}\left( q(z|x,\phi)||p(z|\theta) \right)\end{aligned}$$

Together, we have:

$$\begin{aligned}\log p(x|\theta)&=\mathbb{E}_{q(z|x,\phi)}\left[ \log p(x|z,\theta) \right] + D_{KL}\left(q(z|x,\phi)||p(z|x,\theta)\right)\\&-D_{KL}\left( q(z|x,\phi)||p(z|\theta) \right)\end{aligned}$$

What is worth noticing is that although $D_{KL}\left(q(z|x,\phi)||p(z|x,\theta)\right)$ is positive, in `EM` implementation, we are actually minimizing it.

Here, I deviate in details from other deductions, such as [Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/abs/2208.11970). But a strikingly similar formula in it is derived from the perspective of maximizing `ELBO` alone, and under the setting of `hierarchical VAEs`. The way I put it is:

$$\begin{aligned}EM&\Leftrightarrow\max [\mathbb{E}_{q(z|x,\phi)}\left[ \log p(x|z,\theta) \right] - D_{KL}\left(q(z|x,\phi)||p(z|x,\theta)\right)\\&-D_{KL}\left( q(z|x,\phi)||p(z|\theta) \right)]\end{aligned}$$

Where $\theta$ and $\phi$ are trained jointly instead of alternatively.
### Interpretation
What I like about this (probably) new view is that the terms are more intuive:

$$\begin{cases}\mathbb{E}_{q(z|x,\phi)}\left[ \log p(x|z,\theta) \right]\rightarrow\text{reconstruction (information)}\\D_{KL}\left(q(z|x,\phi)||p(z|x,\theta)\right)\rightarrow\text{consistency (commonsense)}\\D_{KL}\left( q(z|x,\phi)||p(z|\theta) \right)\rightarrow\text{priori (explicability; protocol?)}\end{cases}$$

First, $\mathbb{E}_{q(z|x,\phi)}\left[ \log p(x|z,\theta) \right]$. It essentially measures how well models can reconstruct the original distribution in its `full form (x)` using just the communicated `information (z)`.

Second, $D_{KL}\left(q(z|x,\phi)||p(z|x,\theta)\right)$. It essentially tries to align the `listener`'s and the `speaker`'s understanding of information `z`. Or, if we use a fancy term, `commomsense`. This is crucial to research. Basically, $z$ here is `natural languages` for us humans. natural languages are symbolic and compresses information. That we share the same `commomsense` makes sure that we can communicate effectively. 

Finally, $D_{KL}\left( q(z|x,\phi)||p(z|\theta) \right)$. This term explicitly matches the priori we set for $z$ (eg. Gaussian distribution in VAEs). Through this term, we enforce desired properties onto the latent variable (or `information`) $z$ . More importantly, this term could be interpreted as imposing constraints on `interpretability`, especially in the case of communication. The priori distribution that we definitely intend to match is that of `natural language`. One step further, under the scenario of `machine-machine communication` and `multi-agent collaboration`, $p(z|\theta)$ could be understood as a `protocol`. It follows that by encouraging `minimal` yet `sufficient commonsense`, a decent `protocol` should emerge. 

### Cite this blog
```latex
@online{ZhejianZhou_EMCOMM,
        title={Viewing EM-Style Generation as Communication},
        author={Zhejian Zhou},
        year={2022},
        month={Nov},
        url={\url{https://ja1zhou.github.io/posts/2022/11/Viewing-EM-Style-Generation-as-Communication/}},
}
```