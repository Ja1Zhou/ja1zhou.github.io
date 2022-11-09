---
title: 'Viewing EM and VAE as Communication'
date: 2022-11-06
permalink: /posts/2022/11/Viewing-EM-and-VAE-as-Communication/
tags:
  - Math
  - Generation
  - Machine-Machine Communication
  - EM
  - VAE
---
## Table of Contents
- [Table of Contents](#table-of-contents)
- [Viewing EM and VAE as Communication](#viewing-em-and-vae-as-communication)
  - [Derivation of EM](#derivation-of-em)
    - [Bringing Up the Underlying Information](#bringing-up-the-underlying-information)
  - [log likelihood reward](#log-likelihood-reward)
  - [General Form](#general-form)


## Viewing EM and VAE as Communication
In my [previous post](https://ja1zhou.github.io/posts/2022/10/Emergent-Communication/), I have shared a framework of inspecting the subject of `Communication`. Here is a recap.

$$\mathcal{L}_{\theta, \phi} = \underbrace{\mathcal{H}(X|M_\theta)}_{\mathcal{L}_\text{info}} + \underbrace{\mathbb{E}_{m\sim\pi_\theta }D_{KL}(\rho^\theta(\cdot|m)||\rho_\phi(\cdot|m))}_{\mathcal{L}_{\text{adapt}}}$$

In a  categorical reference game with typical `log likelihoods`, the loss function $\mathcal{L}$ can be decomposed into 
- maximizing information gain (as seen in the `entropy` term)
- minimizing `commonsense` between speaker and listener (this is the way I put it, as seen in the `KL Divergence` term)

In the derivation of `EM` and `VAE` algorithms, I have spotted striking similarities. I try to incorporate them into `generalized communication` in this blog. The derivations in this blogs should be consistent with other available sources, yet I would use `terms` that are not necessarily consistent with their existing usages. I use these terms only to exposit on my insights.

### Derivation of EM
#### Bringing Up the Underlying Information
`EM` starts with `MLE`. Suppose we have observed data $\{x_i\}_{i=1}^N$, and that $x\sim p(x)$ where $p(x)$ is the underlying distribution. We aim to fit a distribution parameterized on $\theta$ to describe $p(x)$:

$$\theta_{MLE}=\argmax_{\theta}\sum_{i=1}^{N}\log P(x_i|\theta) =\argmax_{\theta}\log P(\bold x|\theta) \\
x_i\overset{\text{i.i.d}}{\sim}p(x)$$


Traditionally, we introduce the idea of `latent vector` to help approximate $p(x)$. Here, we say that $p(x)$ needs only to be characterized by `limited features`. That is, $x$ could be (ideally) faithfully reconstructed by `communicating` $z$ instead. For this reason, we refer to $z$ as the `information` of $x$, and $x$ as the `full representation`. 

$$p(x_i|\theta)=\int_{z_i}p(x_i,z_i|\theta)dz_i=\int_{z_i}p(x_i|z_i,\theta)p(z_i|\theta)dz_i$$

For simplicity, we omit the `\bold` command in formulas. To learn $z$ well, we need to introduce interaction between $z$ and $x$. Generally, we have

$$p(x|\theta)=\frac{p(x,z|\theta)} {p(z|x,\theta)}$$

Specifically, we would like to learn $p(z|x, \theta)$, which `encodes` the inputs to `information`. One important insight is that `expectation` over $z$ would not impact x. Moreover, we would like to derive an `iterative optimization`. To this end, it may be reasonable to perform an expectation over $p(z|x, \theta^{(t)})$: 

$$\mathbb{E}_{p(z|x,\theta^{(t)})}\left[\log p(x|\theta)\right]=\mathbb{E}_{p(z|x,\theta^{(t)})}\left[\log p(x,z|\theta)\right]-\mathbb{E}_{p(z|x,\theta^{(t)})}\left[\log p(z|x,\theta)\right]\\=Q(\theta,\theta^{(t)})-K(\theta,\theta^{(t)})$$

Obviously, for $\forall \theta$, we have 
$$K(\theta,\theta^{(t)}) = -D_{KL}(p(z|x,\theta^{(t)})||p(z|x,\theta)) - H(p(z|x,\theta^{(t)}))\\ \le K(\theta^{(t)},\theta^{(t)})$$

Where $H(p(z|x,\theta^{(t)})$ denotes the entropy. Suppose we can calculate $\theta^{(t+1)}$ as: 

$$\argmax_{\theta}Q(\theta,\theta^{(t)})$$

It follows that:

$$\log p(x|\theta^{(t+1)})\ge\log p(x|\theta^{(t)})$$

Therefore, in `EM`, we have to calculate 

$$\theta^{(t+1)}=\argmax_\theta\int_z\log p(x,z|\theta)\cdot p(z|x,\theta^{(t)})dz$$

### log likelihood reward
With $r_\phi = \log \rho_\phi(x|m)$, we have:
$$\mathcal{L}_{\theta,\phi} = -\int dx\ p_X(x)\int dm\ \pi_\theta(m|x)\ \log \rho_\phi(x|m)$$

With speaker parameters $\theta$ fixed, we have:
$$\min_\phi \mathcal{L}_{\theta,\phi} \Leftrightarrow\min_\phi \int\int \rho_{\phi^*}(\cdot|m)\log \frac{\rho_{\phi^*}(\cdot|m)}{\log \rho_\phi(\cdot|m)}=\min_\phi D_{KL}({\rho_{\phi^*}(\cdot|m)}||{\log \rho_\phi(\cdot|m)})$$

$$\rho_{\phi^*}(\cdot|m)=\rho^\theta(\cdot|m) = \frac{p_X(x)\pi_\theta(m|x)}{\mathbb{E}_{x'\sim p}\ p_X(x')\pi_\theta(m|x')}$$

If we rewrite $r_\phi(x, m) = r^\theta(x, m) + r_\phi(x, m) - r^\theta(x, m)$, we'll have
$$\mathcal{L}_{\theta, \phi} = \underbrace{\mathcal{H}(X|M_\theta)}_{\mathcal{L}_\text{info}} + \underbrace{\mathbb{E}_{m\sim\pi_\theta }D_{KL}(\rho^\theta(\cdot|m)||\rho_\phi(\cdot|m))}_{\mathcal{L}_{\text{adapt}}}$$

### General Form
In the appendix, the paper goes on to discuss more generalized forms of reward functions and the formulas are a bit intimidating. However, I find the loss function with `log likelihood` inspiring enough. For effective and interpretable communication of AI agents, besides communicating `information`, it is crucial to align their `understanding` with `human commonsense`. 