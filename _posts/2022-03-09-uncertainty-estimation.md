---
title: "Uncertainty Estimation"
date: 2022-03-09
categories:
  - Blog
tags:
  - NLP
  - uncertainty

---

## Basic

[Law of total variance](https://en.wikipedia.org/wiki/Law_of_total_variance): with random variable $X$ and $Y$, the variance of $Y$ can be decomposed as:

$$\mathrm{Var}(Y) = \underbrace{\mathrm{E}[\mathrm{Var}(Y|X)]}_{\text{Model Uncertainty}} + \underbrace{\mathrm{Var}(\mathrm{E}[Y|X])}_{\text{Data Uncertainty}}$$

Therefore, output uncertainty can be explained by (1) variance from model parameter uncertainty (2) variance from inherent data uncertainty.

### Variance $\sigma^2$

$$\begin{align}
\mathrm{Var} (X) &= \mathrm{E}[(X - \mu)^2]\\
&= \mathrm{E} [(X - \mathrm{E}[X])^2]\\
&= \mathrm{E}[X^2] - E[X]^2
\end{align}
$$

### MAP

See previous post.

$$
\begin{align}
\theta_{MAP} &= \text{argmax}_{\theta} \; \log P(\theta|X) \\
&= \text{argmax}_{\theta} \; \log P(X|\theta) P(\theta)\\
&= \text{argmax}_{\theta} \; \underbrace{\sum_i \log P(x_i|\theta)}_{MLE} + \log P(\theta)
\end{align}
$$

### BNN


