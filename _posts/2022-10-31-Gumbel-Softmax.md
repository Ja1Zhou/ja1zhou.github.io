---
title: 'Gumbel Softmax'
date: 2022-10-31
permalink: /posts/2022/10/Gumbel-Softmax/
tags:
  - Math
  - NLP
  - Discrete Optimization
---
## Table of Contents
- [Table of Contents](#table-of-contents)
- [Gumbel Softmax](#gumbel-softmax)
  - [REINFORCE](#reinforce)
  - [Gumbel-Max](#gumbel-max)
    - [Gumbel Distribution](#gumbel-distribution)
    - [Deriving Gumbel-Max](#deriving-gumbel-max)
  - [Putting It Together](#putting-it-together)


## Gumbel Softmax 
[(original paper)](https://arxiv.org/abs/1611.01144)

`Gumbel Softmax` aims to tackle `discrete optimization`. This blog contains my understanding of it, as well as necessary deductions.
### REINFORCE
Suppose that there is an underlying distribution $p_\theta(z)$ of variable $z$. We would like to optimize $\theta$, on which the distribution is parameterized on. Yet the act of sampling $z$ from $p_\theta(z)$ precludes the back-propagation of gradients to $\theta$.

Originally, discrete optimization is tackled in RL style through REINFORCE algorithm, where `gradient of expectation` is converted to `expectation of gradient`.

$$\nabla_\theta\mathbb{E}_z[\mathcal{L}(z)] = \nabla_\theta \int dz\ p_\theta(z)\mathcal{L}(z) = \int dz\ p_\theta(z)\nabla_\theta \log p_\theta(z)\mathcal{L}(z)\\
=\mathbb{E}_z[\mathcal{L}(z)\nabla_\theta \log p_\theta(z)]$$

More problems arise, such as `high variance (convergence issues)` and `sample efficiency` at each step. 

`Gumbel Softmax` relieves them to a certain extent. 
### Gumbel-Max
Sampling from an arbitrary discrete distribution can be reparameterized through the `Gumbel-Max` trick. Suppose that we would like to draw $z$ from its distribution parameterized by $\theta$, i.e. $p_\theta(z) = [p_1, p_2, ..., p_C]$, where $C = cardinality(z)$. `Gumbel-Max` takes the form of:
$$\argmax_i(\log p_i - \log(-\log\varepsilon_i)),\ \varepsilon_i\ i.i.d.\sim\mathcal{U}(0, 1)$$

It can be prooved that: 

$$P(\argmax_i(\log p_i - \log(-\log\varepsilon_i))==j) = p_j$$

**Merit**: The parameters of the underlying distribution now appear explicitly in the sampling process. Moreover, the sampling process is transferred to $\varepsilon$.
#### Gumbel Distribution
Let us inspect the distribution of 

$$\xi = -\log(-\log\varepsilon),\ \varepsilon\sim\mathcal{U}(0,1)$$

Its density function can be derived as 

$$F(X) = P(\xi \leq X) = P(-\log(-\log\varepsilon)\leq X)\\
= P(-\log \varepsilon \geq \exp(-X)) = P(\varepsilon\leq\exp(-\exp(-X)))\\
=\exp(-\exp(-X))
$$

#### Deriving Gumbel-Max
Denote $\log p_i-\log(-\log\varepsilon_i)$ as $\mathbb{G}(i)$.

$$P(\argmax_i(\log p_i - \log(-\log\varepsilon_i))==j)\\
=\prod_iP(\mathbb{G}(j)\geq\mathbb{G}(i))
$$

We have

$$\mathbb{G}(j)\geq\mathbb{G}(i)\Leftrightarrow\ \log p_i-\log(-\log\varepsilon_i)\geq \log p_j-\log(-\log\varepsilon_j)\\
\\
\Leftrightarrow \frac{p_i}{-\log \varepsilon_i}\leq \frac{p_j}{-\log \varepsilon_j} \Leftrightarrow p_i\log\varepsilon_j\geq p_j\log\varepsilon_i\\
\Leftrightarrow\varepsilon_i\leq\varepsilon_j^{\frac{p_i}{p_j}}
$$

$\Rightarrow$

$$P(\mathbb{G}(j)\geq\mathbb{G}(i)|\varepsilon_j =\hat\varepsilon_j) = \hat\varepsilon_j^{\frac{p_i}{p_j}}
$$

$\Rightarrow$

$$\prod_iP(\mathbb{G}(j)\geq\mathbb{G}(i)|\varepsilon_j =\hat\varepsilon_j) = \hat\varepsilon_j^{\frac{\sum_ip_i}{p_j}} = \hat\varepsilon_j^{\frac{1-p_j}{p_j}}$$

$\Rightarrow$

$$P(\mathbb{G}(j)\geq\mathbb{G}(i)) = \int_0^1 d\hat\varepsilon_j\ \hat\varepsilon_j^{\frac{1-p_j}{p_j}} = p_j\varepsilon^{\frac{1}{p_j}}|_{\varepsilon=0}^1 = p_j$$

Therefore, sampling from arbitrary categorical distribution can be  reparameterized to sampling from Gumbel distribution. 
### Putting It Together
Recall that sampling from arbitrary categorical distribution can be reparameterized as sampling $\varepsilon_i$ from Gumbel distribution. 

$$\argmax_i(\log p_i - \log(-\log\varepsilon_i)),\ \varepsilon_i\ i.i.d.\sim\mathcal{U}(0, 1)$$

To tackle the problem underlying discrete optimization, we would like to approximate `discrete operations` with continuous operations, such that gradients flow back to the parameters that the sampled distribution is conditioned on (Here, $p_i$). 

It is intuitive that by replacing `argmax` with `softmax`, we would obtain a continuous approximation where gradients to distribution parameter $p_i$ are well defined. Moreover, by annealing the temperature to 0, the `softmax` operation would approach `argmax` in both sampled value and expectation value.

In the computation graph, we can replace the intended sampled values with:

$$\sum_i\frac{v_i\cdot \exp((\log p_i- \log(-\log\varepsilon_i))/\tau)}{\sum_j \exp((\log p_j- \log(-\log\varepsilon_j))/\tau)}$$

This approach basically alleviates the sampling process by resorting to `expectation values`.