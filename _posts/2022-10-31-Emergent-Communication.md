---
title: 'Emergent Communication'
date: 2022-10-31
permalink: /posts/2022/10/Emergent-Communication/
tags:
  - Math
  - NLP
  - Machine-to-Machine Communication
---
## Table of Contents
- [Table of Contents](#table-of-contents)
- [Emergent Communication](#emergent-communication)
  - [Objective Overview](#objective-overview)
  - [Lewis Reconstruction Game](#lewis-reconstruction-game)
  - [log likelihood reward](#log-likelihood-reward)
  - [General Form](#general-form)


## Emergent Communication
I would like to share an interesting paper accepted to `Neurips 2022` in this blog, [Emergent Communication: Generalization and Overfitting in Lewis Games](https://arxiv.org/abs/2209.15342). I have been following work on `machine-to-machine communication`, and in this paper, a framework is proposed to understand what is `optimized` in a `reference game`. 
### Objective Overview
$$\mathcal{L}_{\theta, \phi} = \underbrace{\mathcal{H}(X|M_\theta)}_{\mathcal{L}_\text{info}} + \underbrace{\mathbb{E}_{m\sim\pi_\theta }D_{KL}(\rho^\theta(\cdot|m)||\rho_\phi(\cdot|m))}_{\mathcal{L}_{\text{adapt}}}$$

We'll dive into the details later. As it turns out, in a  categorical reference game with typical `log likelihoods`, the loss function $\mathcal{L}$ can be decomposed into 
- maximizing information gain (as seen in the `entropy` term)
- minimizing `commonsense` between speaker and listener (this is the way I put it, as seen in the `KL Divergence` term)

### Lewis Reconstruction Game
we have input $x\sim p_X,\ x\in\mathcal{X}$. `Speaker` observes $x$ and sends message $m\in\mathcal{M}$ according to policy $\pi_\theta(\cdot|x)$. The decoded distribution of the listener is $\rho_\phi(\cdot|m)$. Defined some reward $r_\phi(x, m)$, the loss takes form of:
$$\mathcal{L}_{\theta,\phi} = -\mathbb{E}_{x\sim p, m\sim \pi_\theta}[r_\phi(x, m)]$$

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