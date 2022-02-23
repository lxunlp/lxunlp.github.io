---
title: "Work on Sequence Encoding"
date: 2022-02-18
categories:
  - Blog
tags:
  - NLP
  - encoding
  - Transformers

---

This article is being updated.

Most of the recent work focuses on the Transformers-based architecture. This is an incomplete summary. Multilingual models are included in another post; they are excluded here.

**Attention Is All You Need**. Vaswani et al. NIPS'17.\
The Transformers-architecture.\
<https://arxiv.org/abs/1706.03762>

**Improving Language Understanding by Generative Pre-Training**. Radford et al. 2018\
GPT, the first pre-trained model on Transformers with auto-regressive LM.\
<https://www.semanticscholar.org/paper/Improving-Language-Understanding-by-Generative-Radford/cd18800a0fe0b668a1cc19f2ec95b5003d0a5035>

**BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**. Devlin et al. NAACL'19\
BERT, the first pre-trained model on Transformers with Masked LM (global bidirectional context compared to vanilla auto-regressive LM).\
<https://www.aclweb.org/anthology/N19-1423/>

**RoBERTa: A Robustly Optimized BERT Pretraining Approach**. Liu et al. 2019\
Main change: use dynamic masking in pretraining; use large batch size and long sequence without NSP.\
<https://arxiv.org/abs/1907.11692>

**ALBERT: A Lite BERT for Self-supervised Learning of Language Representations**. Lan et al. ICLR'20\
Changes: (1) share parameters across layers but with widen hidden states (2) factorized embedding to reduce memory (3) add sentence-order prediction objective. Overall, ALBERT gives better performance under the same size of parameters (shared layers but larger hidden states).\
<https://arxiv.org/abs/1909.11942>

**SpanBERT: Improving Pre-training by Representing and Predicting Spans**. Joshi et al. TACL'20\
Better for span-selection tasks such as extractive QA or coreference resolution: use MLM + Span Boundary Objective (mask a span and predict span tokens using boundary tokens) in pretraining. \
<https://www.aclweb.org/anthology/2020.tacl-1.5/>

**ELECTRA: PRE-TRAINING TEXT ENCODERS AS DISCRIMINATORS RATHER THAN GENERATORS**. Clark et al. ICLR'20\
More efficient pre-training objective: a discriminator to predict whether a token is replaced with a plausible alternative by a small MLM model (generator). Although similar to GAN, generator is trained independently. More efficient since prediction is on all tokens, leading to better performance under the same condition compared with MLM.\
<https://arxiv.org/pdf/2003.10555.pdf>

**BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension**. Lewis et al. 2019\
Pretrain S2S, with MLM on encoder and auto-regressive on decoder. Better for generation tasks.\
<https://arxiv.org/abs/1910.13461>

**Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer**. Raffel et al. JMLR'20\
Pretrain S2S and cast different downstream tasks as generation tasks.\
Pretraining objective: see Section 3.3\
<https://arxiv.org/abs/1910.10683>

## For Long Sequence

**Transformer-XL: Attentive Language Models beyond a Fixed-Length Context**. Dai et al. ACL'19\
Encode variable length input (auto-regressive): use a fixed-length Transformers segment and apply recurrence on the input; utilize relative position in self-attention to address cross-segment distance.\
<https://www.aclweb.org/anthology/P19-1285>

**XLNet: Generalized Autoregressive Pretraining for Language Understanding**. Yang et al. NeurIPS'19\
Enhanced model upon Transformer-XL: use Permutation Language Modeling in auto-regressive LM but with permuted sequence to encode bidirectionally. Also some adaptations in the implementation to cope with permutation.\
<https://arxiv.org/pdf/1906.08237.pdf>

**Reformer: The Efficient Transformer**. Kitaev et al. ICLR'20\
Encode large sequence (auto-regressive): (1) use shared QK matrices (2) use Locality Sensitive Hashing (LSH) in the self-attention, to approximately find a subset of K with high prob on Q, without computing full QK (3) use reversible residual connections in the attention and feedforward layer to eliminate full copy of forward values for back-propagation.\
Ablations: (1) reversible layers do not really hurt performance (2) hash bucket size is a trade-off between performance and complexity.\
<https://arxiv.org/abs/2001.04451>

**Longformer: The Long-Document Transformer**. Beltagy et al. 2020\
Enable long sequence at thousands level (either MLM or auto-regressive): (1) use sliding-window local attention (similar to CNN), therefore $O(n)$ instead of $O(n^2)$; full attention is implicitly obtained by stacking layers; varying window size and dilation at each layer (2) task specific global attention (viewed as constant).\
<https://arxiv.org/abs/2004.05150>

**Linformer: Self-Attention with Linear Complexity**. Wang et al. 2020\
<https://arxiv.org/abs/2006.04768>

**Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting**. Zhou et al. AAAI'21\
<https://arxiv.org/abs/2012.07436>

<https://huggingface.co/blog/long-range-transformers>

## For Distillation

**Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned**. Voita et al. ACL'\
<https://aclanthology.org/P19-1580>

**TinyBERT: Distilling BERT for Natural Language Understanding**. Jiao et al. EMNLP-Findings'20\
Pre-training objective (on each layer): (1) MSE on raw attention score (2) MSE on FFNN (3) MSE on embedding\
Finetuning objective: temperature softmax on prediction\
Finetuning data augmentation: replace words with synonyms\
Finding: on downstream tasks, two finetuning techniques have more impact than pre-training objective.\
<https://www.aclweb.org/anthology/2020.findings-emnlp.372>

**MINILM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers**. Wang et al. NeurIPS'20\
Pre-training objective (only on last layer): (1) KL on attention prob distribution (2) KL on value relations\
No other objectives.\
<https://arxiv.org/abs/2002.10957>

**Differentiable Subset Pruning of Transformer Heads**. Li et al. TACL'21\
User-specified number of pruned heads.\
<https://arxiv.org/abs/2108.04657>

## For Incorporating Knowledge

**ERNIE: Enhanced Language Representation with Informative Entities**. Zhang et al. ACL'19\
<https://arxiv.org/abs/1905.07129>

**K-BERT: Enabling Language Representation with Knowledge Graph**. Liu et al. AAAI'20\
Inject triplets into the sequence with soft position embedding and masked attention.\
<https://arxiv.org/abs/1909.07606>

## Recent Survey

**Pre-Trained Models: Past, Present and Future**. Han et al. arXiv'21\
<https://arxiv.org/abs/2106.07139>
