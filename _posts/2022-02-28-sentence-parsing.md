---
title: "Work on Sentence-Level Parsing"
date: 2022-02-28
categories:
  - Blog 
tags:
  - NLP
  - parsing

---

## Task: Semantic Role Labeling (SRL)

**Jointly Predicting Predicates and Arguments in Neural Semantic Role Labeling**. He et al. ACL'18\
Similar to Lee'17: enumerate and prune for both arguments and predicates; then perform (ARG x PRED) scoring on label/relation space.\
<https://aclanthology.org/P18-2058/>

**Syntax-aware Multilingual Semantic Role Labeling**. He et al. EMNLP'19\
<https://aclanthology.org/D19-1538/>

**Span-Based Semantic Role Labeling with Argument Pruning and Second-Order Inference**. Jia et al. AAAI'22\
Independently prune arguments first; then perform bilinear scoring between each words/predicates and arguments. Note that edges are predicted first, and labels are predicted separately (necessary?).\
Second-order helps little.\
<https://www.aaai.org/AAAI22Papers/AAAI-7985.JiaZ.pdf>


## Others

**A Cross-Task Analysis of Text Span Representations**. Toshniwal et al. 2020\
Max Pooling ~= Endpoint ~= Coherent. Although Endpoint takes 2x size.\
<https://aclanthology.org/2020.repl4nlp-1.20>
