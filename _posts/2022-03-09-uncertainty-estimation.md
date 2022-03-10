---
title: "Uncertainty Estimation"
date: 2022-03-09
categories:
  - Blog
tags:
  - NLP
  - uncertainty

---

### MAP

See previous post.

$$
\begin{align}
\theta_{MAP} &= \text{argmax}_{\theta} \; \log P(\theta|X) \\
&= \text{argmax}_{\theta} \; \log P(X|\theta) P(\theta)\\
&= \text{argmax}_{\theta} \; \underbrace{\sum_i \log P(x_i|\theta)}_{MLE} + \underbrace{\log P(\theta)}_{\text{Priori}}
\end{align}
$$

### BNN

Instead of a single model, regard the model as a posterior distribution given the observed data $X$. Therefore, the predicted output is also marginalized over the model posteriori, which should gives more accurate prediction and uncertainty estimation.

$$p(y|x) = \int_{\theta} p(y|f^{\theta}(x)) p(\theta) d\theta$$

## Basic

[Law of total variance](https://en.wikipedia.org/wiki/Law_of_total_variance): with random variable $X$ and $Y$, the variance of $Y$ can be decomposed as:

$$\mathrm{Var}(Y) = \underbrace{\mathrm{E}[\mathrm{Var}(Y|X)]}_{\text{Model Uncertainty}} + \underbrace{\mathrm{Var}(\mathrm{E}[Y|X])}_{\text{Data Uncertainty}}$$

Therefore, output uncertainty can be explained by (1) variance from model parameter uncertainty (2) variance from inherent data uncertainty.

### Variance $\sigma^2$

$$\begin{align}
\mathrm{Var} (x) &= \mathrm{E}[(x - \mu)^2]\\
&= \mathrm{E} [(x - \mathrm{E}[x])^2]\\
&= \mathrm{E}[x^2] - E[x]^2
\end{align}
$$

### MAP

See previous post.

$$
\begin{align}
\theta_{MAP} &= \text{argmax}_{\theta} \; \log P(\theta|X) \\
&= \text{argmax}_{\theta} \; \log P(X|\theta) P(\theta)\\
&= \text{argmax}_{\theta} \; \underbrace{\sum_i \log P(x_i|\theta)}_{MLE} + \underbrace{\log P(\theta)}_{\text{Priori}}
\end{align}
$$


