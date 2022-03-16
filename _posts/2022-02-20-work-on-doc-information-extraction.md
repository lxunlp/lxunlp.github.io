---
title: "Work on Document Information Extraction"
date: 2022-02-20
categories:
  - Blog
tags:
  - NLP
  - extraction
  - IE

---

# Task: Doc-Level RE

**DocRED: A Large-Scale Document-Level Relation Extraction Dataset**. Yao et al. ACL'19\
Multi-label classification on entity-pair. Predicting: bilinear on entity pair\
<https://aclanthology.org/P19-1074/>

**DWIE: an entity-centric dataset for multi-task document-level information extraction**. Zaporojets et al. 2021\
Dataset similar to DocRED, news articles with entity linking to Wiki.\
<https://arxiv.org/pdf/2009.12626.pdf>

**HacRED: A Large-Scale Relation Extraction Dataset Toward Hard Cases in Practical Applications**. Cheng et al. ACL
Findings'21\
Hard cases selected by a trained classifier and verified by human.\
<https://aclanthology.org/2021.findings-acl.249>

**Multi-Task Identification of Entities, Relations, and Coreference for Scientific Knowledge Graph Construction**. Luan et
al. EMNLP'18\
SciERC dataset: similar to DocRED but on scientific abstract.\
<https://aclanthology.org/D18-1360/>

**SCIREX: A Challenge Dataset for Document-Level Information Extraction**. Jain et al. ACL'20\
SciREX dataset: similar to  SciERC but on very long documents. Approach: separate out COREF as clustering (probably due to too long doc).\
<https://aclanthology.org/2020.acl-main.670>

**CodRED: A Cross-Document Relation Extraction Dataset for Acquiring Knowledge in the Wild**. Yao et al. EMNLP'21\
<https://aclanthology.org/2021.emnlp-main.366/>

**Dialogue-Based Relation Extraction**. Yu et al. ACL'21\
On Friends data.\
<https://aclanthology.org/2020.acl-main.444>

**BioCreative V CDR task corpus: a resource for chemical disease relation extraction**. Li et al. 2016\
<https://academic.oup.com/database/article/doi/10.1093/database/baw068/2630414>

## Approach: Pretraining

**Matching the Blanks: Distributional Similarity for Relation Learning**. Livio et al. ACL'19\
Sent-level.\
<https://aclanthology.org/P19-1279/>

**Learning from Context or Names? An Empirical Study on Neural Relation Extraction**. Peng et al. EMNLP'20\
Sent-level.\
Analysis: context is the main source to support classification, entity type also provides critical information; mention
itself may provide shallow clues without need to understand context. Approach: contrastive learning utilizing KB; help
with low-resource\
<https://aclanthology.org/2020.emnlp-main.298>

**Coreferential Reasoning Learning for Language Representation**. Ye et al. EMNLP'20\
Doc-level. Pretrain with Mention
Reference Prediction (in addition to MLM): mask one mention and optimize the marginal likelihood of repeated mentions (
heuristic assumption: repeated noun mentions in a sequence refer to each other). Improvement is dataset-specific;
trivial for DocRED.\
<https://aclanthology.org/2020.emnlp-main.582.pdf>

**ERICA: Improving Entity and Relation Understanding for Pre-trained Language Models via Contrastive Learning**. Qin et al.
ACL'21\
Doc-level. Contrastive learning on two pretraining tasks on entity and relation representation; help with
low-resource\
<https://aclanthology.org/2021.acl-long.260>

## Approach: Denoising Distant Supervision

**Denoising Relation Extraction from Document-level Distant Supervision**. Xiao et al. EMNLP'20\
TODO\
<https://aclanthology.org/2020.emnlp-main.300>

## Approach: Joint Tasks

**A Hierarchical Multi-task Approach for Learning Embeddings from Semantic Tasks**. Sanh et al. AAAI'19\
Hierarchical repr
for each level among e2e multi-tasks (mention + coref + RE); each level is BiLSTM taking [orig; prev level]; hierarchy
order matters. RE is observed to benefit from NER and coref. Only share encoding; representation receives no task
interference.\
<https://arxiv.org/pdf/1811.06031.pdf>

**A General Framework for Information Extraction using Dynamic Span Graphs**. Luan et al. NAACL'19\
DYGIE: span-based
pipeline + refinement/propagation on both coref and RE; coref prop (higher-order) helps very trivially; RE propagation
helps a bit more. Share encoding; representation has task interference from each task propagation. However, still
pipeline and no explicit task interaction.\
<https://aclanthology.org/N19-1308/>

**Entity, Relation, and Event Extraction with Contextualized Span Representations**. Wadden et al. EMNLP'19\
DYGIE++: (1) use
BERT (2) add event extraction and its propagation.\
<https://aclanthology.org/D19-1585/>

**An End-to-end Model for Entity-level Relation Extraction using Multi-instance Learning**. Eberts and Ulges. EACL'21\
Span-based pipeline + multi-instance RE (substitute entity-pair repr with localized mention context).\
<https://aclanthology.org/2021.eacl-main.319>

**Injecting Knowledge Base Information into End-to-End Joint Entity and Relation Extraction and Coreference Resolution**.
Verlinden et al. ACL Findings'21\
Span-based pipeline + Concat mention with attended KB candidate repr.\
<https://aclanthology.org/2021.findings-acl.171>

**Document-level Entity-based Extraction as Template Generation**. Huang et al. EMNLP'21\
Benefits as generation: (1)
generation decoding captures entity dependencies (2) naturally n-ary without exponential combinations of entities.\
<https://aclanthology.org/2021.emnlp-main.426>

## Approach: Joint Tasks (Non-Doc-Level)

**A Frustratingly Easy Approach for Entity and Relation Extraction**. Zhong and Chen. NAACL'21\
<https://aclanthology.org/2021.naacl-main.5/>

**Cross-Task Instance Representation Interactions and Label Dependencies for Joint Information Extraction with Graph
Convolutional Networks**. Nguyen et al. NAACL'21\
<https://aclanthology.org/2021.naacl-main.3>

**A Partition Filter Network for Joint Entity and Relation Extraction**. Yan et al. EMNLP'21\
Task-interaction in LSTM encoder.\
<https://aclanthology.org/2021.emnlp-main.17/>

**REBEL: Relation Extraction By End-to-end Language generation**. Cabot and Navigli. EMNLP Findings'21\
<https://aclanthology.org/2021.findings-emnlp.204/>

**GenerativeRE: Incorporating a Novel Copy Mechanism and Pretrained Model for Joint Entity and Relation Extraction**. Cao and Ananiadou. EMNLP Findings'21\
<https://aclanthology.org/2021.findings-emnlp.182/>

**OneRel: Joint Entity and Relation Extraction with One Module in One Step**. Shang et al. AAAI'22\
<https://arxiv.org/pdf/2203.05412>

**Joint Entity and Relation Extraction with Set Prediction Networks**. Sui et al. AAAI'22\
<https://arxiv.org/pdf/2011.01675>

## Approach: Evidence-Guided

**Three Sentences Are All You Need: Local Path Enhanced Document Relation Extraction**. Huang et al. ACL'21\
Heuristic rules
to select evidence sentences (on average < 3 sentences) and discard others.\
<https://aclanthology.org/2021.acl-short.126>

**Entity and Evidence Guided Relation Extraction for DocRED**. Huang et al. 2020\
<https://arxiv.org/abs/2008.12283>

**EIDER: Evidence-enhanced Document-level Relation Extraction**. Xie et al. 2021\
Train a simple sentence-evidence binary
classier per entity pair, then use predicted evidence to predict relations. Final results use fusion between full doc
and evidence only.\
<https://arxiv.org/pdf/2106.08657>

## Approach: Sequence-based Encoding

**HIN: Hierarchical Inference Network for Document-Level Relation Extraction**. Tang et al. PAKDD'20\
Hierarchical encoding:
doc repr of entity pair, which is attended sent repr conditioned on entity pair repr.\
<https://arxiv.org/pdf/2003.12754>

**Document-Level Relation Extraction with Adaptive Thresholding and Localized Context Pooling**. Zhou et al. AAAI'21\
(1) Marginalized pos and neg relation prediction by using a learned threshold\
(2) gather entity-level self-attention to concat as a new context emb for entity pair.\
<https://arxiv.org/pdf/2010.11304>

**Entity Structure Within and Throughout: Modeling Mention Dependencies for Document-Level Relation Extraction**. Xu et al.
AAAI'21\
Structural encoding (token-level coref and co-occurance structure) into self-attention.\
<https://arxiv.org/pdf/2102.10249>

**Learning Logic Rules for Document-level Relation Extraction**. Ru et al. EMNLP'21\
TODO\
<https://aclanthology.org/2021.emnlp-main.95>

**Modular Self-Supervision for Document-Level Relation Extraction**. Zhang et al. EMNLP'21\
TODO\
<https://aclanthology.org/2021.emnlp-main.429>

**Document-Level Relation Extraction with Reconstruction**. Xu et al. AAAI'21\
<https://arxiv.org/abs/2012.11384>

## Approach: Graph-based Encoding

**Connecting the Dots: Document-level Neural Relation Extraction with Edge-oriented Graphs**. Christopoulou et al. EMNLP'19\
Heterogeneous graph.\
<https://aclanthology.org/D19-1498/>

**Global Context-enhanced Graph Convolutional Networks for Document-level Relation Extraction**. Zhou et al. COLING'20\
<https://aclanthology.org/2020.coling-main.461>

**Global-to-Local Neural Networks for Document-Level Relation Extraction**. Wang et al. EMNLP'20\
<https://aclanthology.org/2020.emnlp-main.303>

**Double Graph Based Reasoning for Document-level Relation Extraction**. Zeng et al. EMNLP'20\
<https://aclanthology.org/2020.emnlp-main.127>

**Reasoning with Latent Structure Refinement for Document-Level Relation Extraction**. Nan et al. ACL'21\
Similar to
higher-order propagation: compute scores to construct a latent structure for entities, then GCN to obtain entity repr.\
<https://aclanthology.org/2020.acl-main.141>

**Discriminative Reasoning for Document-level Relation Extraction**. Xu et al. ACL Findings'21\
TODO\
<https://aclanthology.org/2021.findings-acl.144>

**SIRE: Separate Intra- and Inter-sentential Reasoning for Document-level Relation Extraction**. Zeng et al. ACL Findings'21\
TODO\
<https://aclanthology.org/2021.findings-acl.47>

**MRN: A Locally and Globally Mention-Based Reasoning Network for Document-Level Relation Extraction**. Li et al. ACL
Findings'21\
TODO\
<https://aclanthology.org/2021.findings-acl.117>

# Task: for other Datasets

**Enhancing Dialogue-based Relation Extraction by Speaker and Trigger Words Prediction**. Zhao et al. ACL Findings'21\
On DialogRE.\
<https://aclanthology.org/2021.findings-acl.402>

**Selecting Optimal Context Sentences for Event-Event Relation Extraction**. Trong et al. AAAI'22\
Pairwise scoring on given event spans: use REINFORCE to select important sentences and compress documents.\
<https://www.aaai.org/AAAI22Papers/AAAI-3912.ManH.pdf>

# Task: n-ary RE

**Cross-Sentence N-ary Relation Extraction with Graph LSTMs**. Peng et al. TACL'17\
TODO\
<https://aclanthology.org/Q17-1008/>

**Document-Level N-ary Relation Extraction with Multiscale Representation Learning**. Jia et al. NAACL'19\
Entity-centric n-ary relation extraction.\
<https://aclanthology.org/N19-1370/>

# Task: Few-Shot IE

**FewRel 2.0: Towards More Challenging Few-Shot Relation Classification**. Gao et al. EMNLP'19\
<https://aclanthology.org/D19-1649/>

**Entity Concept-enhanced Few-shot Relation Extraction**. Yang et al. ACL'21\
<https://aclanthology.org/2021.acl-short.124>

**Exploring Task Difficulty for Few-Shot Relation Extraction**. Han et al. EMNLP'21\
<https://aclanthology.org/2021.emnlp-main.204>

**Label Verbalization and Entailment for Effective Zero and Few-Shot Relation Extraction**. Sainz et al. EMNLP'21\
<https://aclanthology.org/2021.emnlp-main.92>

**Towards Realistic Few-Shot Relation Extraction**. Brody et al. EMNLP'21\
<https://aclanthology.org/2021.emnlp-main.433/>

# Task: Open-IE

## Cluster-based

**Open Relation Extraction: Relational Knowledge Transfer from Supervised Data to Unsupervised Data**. Wu et al. EMNLP'19\
<https://aclanthology.org/D19-1021>

**SelfORE: Self-supervised Relational Feature Learning for Open Relation Extraction**. Hu et al. EMNLP'20\
<https://aclanthology.org/2020.emnlp-main.299>

**Element Intervention for Open Relation Extraction**. Liu et al. ACL'21\
<https://aclanthology.org/2021.acl-long.361>

**Unsupervised Relation Extraction: A Variational Autoencoder Approach**. Yuan and Eldardiry. EMNLP'21\
<https://aclanthology.org/2021.emnlp-main.147>

**A Relation-Oriented Clustering Method for Open Relation Extraction**. Zhao et al. EMNLP'21\
<https://aclanthology.org/2021.emnlp-main.765/>

## Predicate-based

**Supervised Open Information Extraction**. Stanovsky et al. NAACL'18\
<https://aclanthology.org/N18-1081>

**IMOJIE: Iterative Memory-Based Joint Open Information Extraction**. Kolluru et al. ACL'20\
<https://aclanthology.org/2020.acl-main.521>

**OpenIE6: Iterative Grid Labeling and Coordination Analysis for Open Information Extraction**. Kolluru et al. EMNLP'20\
<https://aclanthology.org/2020.emnlp-main.306/>

**Multi2OIE: Multilingual Open Information Extraction Based on Multi-Head Attention with BERT**. Ro et al. EMNLP Findings'20\
<https://arxiv.org/abs/2009.08128>

**LSOIE: A Large-Scale Dataset for Supervised Open Information Extraction**. Solawetz and Larson. EACL'21\
<https://aclanthology.org/2021.eacl-main.222>

**Zero-Shot Information Extraction as a Unified Text-to-Triple Translation**. Wang et al. EMNLP'21\
<https://aclanthology.org/2021.emnlp-main.94>

**DetIE: Multilingual Open Information Extraction Inspired by Object Detection**. Vasilkovsky et al. AAAI'22\
<https://www.aaai.org/AAAI22Papers/AAAI-8073.VasilkovskyM.pdf>

# Task: Continual RE

**Sentence Embedding Alignment for Lifelong Relation Extraction**. Wang et al. NAACL'19

**Continual Relation Learning via Episodic Memory Activation and Reconsolidation**. Xu et al. ACL'20

**Curriculum-Meta Learning for Order-Robust Continual Relation Extraction**. Wu et al. AAAI'21

# Task: KG Completion

**Inductive Relation Prediction by BERT**. Zha et al. AAAI'22\
Linearize each local subgraph individually and encode by BERT, paired with target triple to do MIL scoring.\
Thereafter to achieve both semantic representation and implicit rule inference.\
<https://arxiv.org/pdf/2103.07102>

# Analysis

**More Data, More Relations, More Context and More Openness: A Review and Outlook for Relation Extraction**. Han et al. AACL'20\
<https://aclanthology.org/2020.aacl-main.75>
