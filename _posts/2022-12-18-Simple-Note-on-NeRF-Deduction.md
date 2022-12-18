---
title: 'Simple Note on NeRF Deduction'
date: 2022-11-11
permalink: /posts/2022/12/Simple-Note-on-NeRF-Deduction/
tags:
  - Math
  - Theory
  - CV
---
## Table of Contents
- [Table of Contents](#table-of-contents)
- [Simple Note on NeRF Deduction](#simple-note-on-nerf-deduction)
  - [Formulation](#formulation)
  - [Definitions and Relation](#definitions-and-relation)
  - [Deduction](#deduction)
  - [Approximation](#approximation)
  - [Volume Rendering](#volume-rendering)

## Simple Note on NeRF Deduction
This blog serves as a minimum note on the recent [ECCV tutorial](https://sites.google.com/berkeley.edu/nerf-tutorial/home) on NeRF fundamentals. 

### Formulation
![formulation](/images/NeRF_fundamentals/formulation.png)
We have a view direction, or a ray. We intend to return the RGB value at a specific location along that vector.

### Definitions and Relation
![definition](/images/NeRF_fundamentals/relation.png)

$$
P\left[\text{no hits before }t\right] = T(t) \\
P\left[\text{hit at }t\right] = \sigma(t)dt \\
$$

$\sigma(t)$ and $T(t)$ are related by the following equation, which is intuitive:

$$
\underbrace{T(t+dt)}_{\text{No hit at }t+dt} = \underbrace{T(t)}_{\text{No hit at }t} \times \underbrace{(1-\sigma(t)dt)}_{\text{No hit at }dt}
$$

### Deduction
$$
\begin{cases}
T(t+dt)  = T(t) \times (1-\sigma(t)dt) \\  
T(t+dt)  = T(t) + T'(t)dt \ (\text{Taylor Expansion}) \\
\end{cases}
$$

$\Rightarrow$

$$
T(t) + T'(t)dt = T(t) - T(t)\sigma(t)dt \\
\frac{T'(t)}{T(t)}dt = -\sigma(t)dt \\
log T(t) = \int_{t_0}^{t}-\sigma(s)ds \\
\therefore\ T(t) = \exp(\int_{t_0}^{t}-\sigma(s)ds)
$$

The expected color along the ray:

$$
\int_{t_0}^{t_{n+1}}T(t)\sigma(t)\bm c(t)dt
$$

### Approximation
Note the nested integration (another inside $T(t)$). Approximate the outside integration assuming local consistency ($\sigma$ and $\bm c$ remains constant).

$$
\int_{t_0}^{t_{n+1}} T(t)\sigma(t)\bm c(t)dt \approx \sum_{i=0}^n\int_{t_i}^{t_{i+1}}T(t)\sigma_i\bm c_i dt
$$

Let $\delta_i = t_{i+1}-t_i$

$T(t)$ can be rewritten as:

$$
\begin{aligned}
T(t) &= \underbrace{\prod_{i=0}^{m=\argmax_j t_{j+1}\leq t}\exp(-\int_{t_i}^{t_{i+1}}\sigma_i ds)}_{1}\cdot \underbrace{\exp(-\int_{t_{m+1}}^t\sigma_{m+1}ds)}_2 \\
&= \underbrace{\exp(-\sum_{i=0}^m\sigma_i\delta_i)}_1\cdot \underbrace{\exp(-\sigma_{m+1}(t-t_{m+1}))}_2 \\
&= \underbrace{T_{m+1}}_1\cdot \underbrace{\exp(-\sigma_{m+1}(t-t_{m+1}))}_2
\end{aligned}
$$

$\Rightarrow$

$$
\begin{aligned}
\int_{t_0}^{t_{n+1}} T(t)\sigma(t)\bm c(t)dt &\approx \sum_{i=0}^n\int_{t_i}^{t_{i+1}}T(t)\sigma_i\bm c_i dt \\
&=\sum_{i=0}^n T_i\sigma_i\bm c_i\int_{t_{i}}^{t_{i+1}}\exp(-\sigma_{i}(t-t_i))dt \\
&=\sum_{i=0}^n T_i\sigma_i\bm c_i \frac{\exp (-\sigma_i(t_{i+1}-t_i))-1}{-\sigma_i} \\
&=\sum_{i=0}^n T_i\bm c_i \underbrace{(1-\exp(-\sigma_i\delta_i))}_{\text{opacity } \alpha_i} \\
&=\sum_{i=0}^n T_i\bm c_i\alpha_i
\end{aligned}
$$

$$
(T_i=\prod_{j=0}^{i-1}(1-\alpha_j))
$$

### Volume Rendering
In the previous section, we are basically saying that 

$$
\text{color} = \sum_{i=0}^n T_i\bm c_i\alpha_i
$$

By replacing $\bm c(t)$, we can render other quantities of interest.