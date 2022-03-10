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

### Basic

$$\begin{align}
\mathrm{Var} (X) &= \mathrm{E}[(X - \mu)^2]\
&= \mathrm{E} [(X - \mathrm{E}[X])^2]\
&= \mathrm{E}[X^2] - E[X]^2
\end{align}
$$
