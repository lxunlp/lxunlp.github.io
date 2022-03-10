---
title: "Uncertainty Estimation"
date: 2022-03-09
categories:
  - Blog
tags:
  - NLP
  - uncertainty

---

Uncertainty: $\approx$ total variance on output
* Epistemic uncertainty (model uncertainty)
* Aleatoric uncertainty (data uncertainty)
  * Heteroscedastic uncertainty (data-dependent uncertainty)
  * Homoscedastic uncertainty (data-independent or task-dependent uncertainty)

## Basic

[Law of total variance](https://en.wikipedia.org/wiki/Law_of_total_variance): with random variable $X$ and $Y$, the variance of $Y$ can be decomposed as:

$$\mathrm{Var}(Y) = \underbrace{\mathrm{E}[\mathrm{Var}(Y|X)]}_{\mathrm{Model Uncertainty}} + \underbrace{\mathrm{Var}(\mathrm{E}[Y|X])}_{\mathrm{Data Uncertainty}}$$

### Variance $\theta^2$

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


