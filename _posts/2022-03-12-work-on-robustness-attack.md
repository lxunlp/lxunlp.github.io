---
title: "Work on Robustness and Attacking"
date: 2022-03-12
categories:
  - Blog
tags:
  - NLP
  - robustness

---

## Robustness

**Unifying Model Explainability and Robustness for Joint Text Classification and Rationale Extraction**. Li et al. AAAI'22\
Discrete word-replacement attack & embedding perturbation attack.\
<http://arxiv.org/abs/2112.10424>

## Causal Robustness

**C2L: Causally Contrastive Learning for Robust Text Classification**. Choi et al. AAAI'22\
Improve robustness and generalization by identifying more causal features.\
(1) use gradient magnitude to identify candidate causal words\
(2) validate candidates by replacing with other plausible words (selected by BERT masking) and observing label change\
(3) push repr of non-causal replacement samples similar, so that encoding focuses more on the causal features\
<https://aaai-2022.virtualchair.net/poster_aaai11767>
