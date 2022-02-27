---
title: "Work on Dialogues"
date: 2022-02-24
categories:
  - Blog
tags:
  - NLP
  - dialogue

---

This article is being updated.

## Dialogue Representation

**Pretraining Methods for Dialog Context Representation Learning**. Mehri et al. ACL'19\
Evaluate four pretraining objectives on dialogue data to improve the dialogue representation.\
<https://www.aclweb.org/anthology/P19-1373/>

**ConveRT: Efficient and Accurate Conversational Representations from Transformers**. Henderson et al. 2019\
Conversational Representations from Transformers: use response-selection as unsupervised pretraining objective for dialogue tasks. Transformer layers are shared for input and response encoding.\
To encode more history context which is shown to be helpful in other tasks, optionally use another Transformer encoder to encode history and average two to get final representation (is averaging really useful?).\
<https://arxiv.org/pdf/1911.03688>

**TOD-BERT: Pre-trained Natural Language Understanding for Task-Oriented Dialogue**. Wu et al. EMNLP'20\
Similar to ConveRT: dual encoder for context and response with contrastive loss on response selection.\
<https://www.aclweb.org/anthology/2020.emnlp-main.66>

**Probing Task-Oriented Dialogue Representation from Language Models**. Wu and Xiong. EMNLP'20\
Probe LMs on dialogue datasets by two metrics: (1) linear classifier on example's CLS (2) mutual information on examples' CLS clustering.\
Finding: ConveRT and TOD-BERT-jnt perform the best.\
Question: freeze CLS or not??\
<https://www.aclweb.org/anthology/2020.emnlp-main.409>

**DIALOGLM: Pre-trained Model for Long Dialogue Understanding and Summarization**. Zhong et al. AAAI'22\
Similar to BART or UniLM on dialogue domain for generation tasks, and introduce dialogue-specific corruption.\
Sparse attention for long sequence helps little.\
<https://arxiv.org/pdf/2109.02492.pdf>

## Downstream Tasks (Non-Generation)

**Dialogue-Based Relation Extraction**. Yu et al. ACL'20\
Propose cross-sentence relation extraction task on dialogue with new dataset and baselines.\
Input: context + two arguments.\
<https://www.aclweb.org/anthology/2020.acl-main.444>

**MIE: A Medical Information Extractor towards Medical Dialogues**. Zhang et al. ACL'20\
Extract (symptom, status) on medical dialogues, with predefined symptoms and status. Treat it as multiclass classification (each symptom-status being a class).\
Instead of naive dual input like QA on BERT, it uses attention to obtain symptom-specific and status-specific representation first on each utterance respectively, then aggregate two representation by cross-utterance attention to get (symptom-specific)-specific utterance representation, then get the final score of each (symptom, status).\
<https://www.aclweb.org/anthology/2020.acl-main.576>

**Learning to Identify Follow-Up Questions in Conversational Question Answering**. Kundu et al. ACL'20\
New task, multiclass classification with triplet input: (candidate follow-up question, passage, conversation history). Model: propose three-way attentive pooling among input.\
<https://www.aclweb.org/anthology/2020.acl-main.90>

**MuTual: A Dataset for Multi-Turn Dialogue Reasoning**. Cui et al. ACL'20\
New dataset on dialogue reasoning with multi-choice question. Input: context + 4 candidates. Baseline: treat it as multiclass classification with dual input.\
<https://www.aclweb.org/anthology/2020.acl-main.130>

**Span-ConveRT: Few-shot Span Extraction for Dialog with Pretrained Conversational Representations**. Coope et al. ACL'20\
Tagging as span extraction. Similar to BERT-LSTM-CRF, this work uses ConveRT-CNN-CRF with some feature tweaking.\
<https://www.aclweb.org/anthology/2020.acl-main.11>

**DialogueGCN: A Graph Convolutional Neural Network for Emotion Recognition in Conversation**. Ghosal et al. EMNLP'19\
Classify emotion for each utterance by modeling speaker-aware dialogue context. Each utterance is the concatenation of two representation: (1) speaker-agnostic representation by sequential encoding on RNN (2) speaker-dependent representation by building utterance graph with speaker dependency and temporal dependency, aggregate utterance/node by GCN.\
<https://www.aclweb.org/anthology/D19-1015>

**Enhancing Dialogue Symptom Diagnosis with Global Attention and Symptom Graph**. Chen et al. EMNLP'19\
Task: identify symptom and status in medical conversation. Model: (1) extract symptom by tagging, modeling the utterance with attention on document-level and corpus-level (2) build symptom graph where edge indicates two symptoms co-occur at the same dialogue (3) status classification on each symptom/node.\
<https://www.aclweb.org/anthology/D19-1508>

**Response Selection for Multi-Party Conversations with Dynamic Topic Tracking**. Wang et al. EMNLP'20\
Response selection with auxiliary multi-tasking and fine-grained representation.\
<https://www.aclweb.org/anthology/2020.emnlp-main.533/>
