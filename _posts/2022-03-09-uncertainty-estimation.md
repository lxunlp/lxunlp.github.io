---
title: "Uncertainty Estimation"
date: 2022-03-09
categories:
  - Blog
tags:
  - NLP
  - uncertainty

---

## Basics

### MAP

Let $D$ be the observed data. See previous post.

$$
\begin{align}
\theta_{MAP} &= \text{argmax}_{\theta} \; \log P(\theta|D) \\
&= \text{argmax}_{\theta} \; \log P(D|\theta) P(\theta)\\
&= \text{argmax}_{\theta} \; \underbrace{\sum_i \log P(d_i|\theta)}_{MLE} + \underbrace{\log P(\theta)}_{\text{Priori}}
\end{align}
$$

### BNN

Instead of a single model, regard the model parameter $\theta$ as a posterior distribution given the observed data $D$. Therefore, the predicted output is also marginalized over the model posteriori, which should gives more accurate prediction and uncertainty estimation.

$$p(y|x, D) = \int_{\theta} p \big(y|f^{\theta}(x) \big) p(\theta|D) d\theta$$

Analytically, we usually parameterize the model posteriori $p(\theta|D)$ by a simple distribution such as Gaussian to perform tractable variational inference.

For simple linear regression, we can get the analytical form:

$$
\begin{align}
W_{MLE} &= \mathrm{argmax}_{\theta} \log \hat{y}\\
&= \mathrm{argmax}_{\theta} \log \frac{1}{\sqrt{2\pi}\sigma} + \log \bigg( \exp \big( -\frac{(\hat{y} - W^T x)^2}{2 \sigma^2} \big) \bigg)\\
&= \mathrm{argmin}_{\theta} \frac{1}{2 \sigma^2}(\hat{y} - W^T x)^2 \;+\; \log \sigma
\end{align}
$$

For deep neural nets, dropout variational inference is proved to serve as the effective approximation

## Uncertainty

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
