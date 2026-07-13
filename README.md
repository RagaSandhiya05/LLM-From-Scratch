# LLM-From-Scratch
A step-by-step implementation of a GPT-style Large Language Model from scratch using Python and PyTorch.


# 🚀 NovaGPT: Building a GPT from Scratch 

## 🧠 Introduction

Large Language Models (LLMs) are deep learning models built on the **Transformer architecture** that understand and generate human-like text by learning patterns from massive amounts of textual data.

Instead of memorizing sentences, they learn relationships between words, grammar, context, and semantics, enabling them to perform tasks such as **text generation, question answering, summarization, translation, and code generation**.

In this repository, I build a simplified GPT model completely from scratch using **PyTorch**, implementing every major component step by step—from **tokenization** to **text generation**—without relying on high-level libraries.


# 📖 How Large Language Models Work


<p align="center">
  <img src="https://testrigor.com/wp-content/uploads/2025/12/How-Do-Large-Language-Models-Work.jpeg" width="800">
</p>

<p align="center">
  <em><strong>Figure 1:</strong> High-level architecture of a GPT-style Large Language Model.</em>
</p>


# 🌍 Applications of Large Language Models


<p align="center">
  <img src="https://dailydoseofai.in/wp-content/uploads/2025/07/image-4.png" width="800">
</p>

<p align="center">
  <em><strong>Figure 2:</strong> Common applications of Large Language Models across NLP and multimodal AI.</em>
</p>


# 📘 Day 1 – Tokenization

## 📖 Overview

Large Language Models (LLMs) cannot directly understand raw text. Before any processing begins, the input text must be broken down into smaller meaningful units called **tokens**. This process is known as **Tokenization** and serves as the first step in every Natural Language Processing (NLP) pipeline.

Depending on the tokenizer, tokens may represent **characters, words, subwords, or byte pairs**. Modern LLMs primarily use **Subword Tokenization** techniques such as **Byte Pair Encoding (BPE)**, allowing them to efficiently represent language while handling unknown or rare words.


## 🎯 Objective

* Understand the importance of tokenization in NLP.
* Explore character-level and word-level tokenization.
* Learn the intuition behind Byte Pair Encoding (BPE).
* Prepare raw text for language model processing.


## 🧠 Concepts Covered

* Character Tokenization
* Word Tokenization
* Subword Tokenization
* Byte Pair Encoding (BPE)
* Tokens vs Words


## 📊 Visual Workflow

```text
                    RAW TEXT
                        │
                        ▼
          "Artificial Intelligence"
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
Character Tokenization          Word Tokenization
[A,r,t,i,...]              ["Artificial","Intelligence"]
        │                               │
        └───────────────┬───────────────┘
                        ▼
            Byte Pair Encoding (BPE)
                        │
                        ▼
             Final Token Sequence
```


<p align="center">
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRZXotAyaiT5gcL1cFBf-HmqVc24A6_EqDj0Gn40noFZg&s=10" width="800">
</p>

<p align="center">
  <em><strong>Figure 3:</strong> Tokenization Process</em>
</p>


## 🚀 Learning Outcome

By the end of this notebook, you will understand how raw text is transformed into tokens—the fundamental building blocks processed by modern language models.


# 📘 Day 2 – Vocabulary

## 📖 Overview

After tokenization, each unique token must be assigned a numerical identifier so that it can be processed by neural networks. This collection of unique tokens is called the **Vocabulary**.

Every token is mapped to a unique integer ID, enabling efficient numerical representation of language. Unknown or unseen words are typically mapped to a special **`<UNK>`** token.


## 🎯 Objective

* Build a vocabulary from tokenized text.
* Create Word → ID mappings.
* Create ID → Word mappings.
* Handle unseen or unknown tokens.


## 🧠 Concepts Covered

* Vocabulary
* Token IDs
* Word-to-ID Mapping
* ID-to-Word Mapping
* Unknown Token (`<UNK>`)


## 📊 Visual Workflow

```text
                Tokenized Text
                       │
                       ▼
     ["I", "love", "Python", "love"]
                       │
                       ▼
          Extract Unique Tokens
                       │
                       ▼
      ["I", "love", "Python"]
                       │
                       ▼
          Create Vocabulary
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
   Word → ID Mapping           ID → Word Mapping
         │                           │
         └─────────────┬─────────────┘
                       ▼
             Numerical Representation
```


<p align="center">
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSJsaI12zptERiW3IPA1jp1dDDUNqDsyISyXST519nCSw&s=10" width="800">
</p>

<p align="center">
  <em><strong>Figure 4:</strong> Vocabulary Construction</em>
</p>


## 🚀 Learning Outcome

By the end of this notebook, you will understand how language is converted into numerical IDs, enabling neural networks to process textual data efficiently.


# 📘 Day 3 – Embeddings

## 📖 Overview

Neural networks cannot learn meaningful relationships from integer token IDs alone. **Embeddings** transform token IDs into dense numerical vectors that capture semantic relationships between words.

Unlike one-hot encoding, embeddings allow similar words to have similar vector representations, making them an essential component of modern language models.


## 🎯 Objective

* Understand why embeddings are required.
* Learn how embedding layers work.
* Convert token IDs into dense vectors.
* Explore semantic representation in vector space.


## 🧠 Concepts Covered

* Embedding Layer
* Dense Vector Representation
* One-Hot Encoding vs Embeddings
* Semantic Similarity
* Vector Space Representation


## 📊 Visual Workflow

```text
             Token IDs
        [3, 7, 12, 5]
                  │
                  ▼
          Embedding Layer
                  │
                  ▼
      ┌────────────────────────┐
      │ [0.21, -0.43, 0.91...] │
      │ [0.54,  0.18, 0.33...] │
      │ [0.89, -0.11, 0.47...] │
      └────────────────────────┘
                  │
                  ▼
      Dense Vector Representations
```


<p align="center">
  <img src="https://miro.medium.com/1*tnDiRDrL0nwZA8VwWSXuFQ.png" width="800">
</p>

<p align="center">
  <em><strong>Figure 5:</strong> Word Embedding Representation</em>
</p>


## 🚀 Learning Outcome

By the end of this notebook, you will understand how embeddings transform discrete token IDs into meaningful dense vectors that capture semantic and contextual information.


# 📘 Day 4 – Next Word Prediction

## 📖 Overview

Next-word prediction is the fundamental learning objective of GPT-style language models. Given a sequence of previous tokens (context), the model learns to predict the most probable next token.

By repeatedly learning this objective over large datasets, language models gradually develop an understanding of grammar, context, syntax, and semantic relationships.


## 🎯 Objective

* Prepare context-target training pairs.
* Learn how autoregressive language modeling works.
* Predict the next word using previous context.
* Understand the foundation of GPT training.


## 🧠 Concepts Covered

* Context Window
* Target Word
* Sequence Modeling
* Next-Word Prediction
* Autoregressive Learning


## 📊 Visual Workflow

```text
          Training Sentence

"I love learning Artificial Intelligence"

                │
                ▼

Context: ["I"] ----------------------► Target: "love"

Context: ["I","love"] --------------► Target: "learning"

Context: ["I","love","learning"] ---> Target: "Artificial"

Context: ["love","learning","Artificial"] ---> Target: "Intelligence"

                │
                ▼
        Language Model Training
                │
                ▼
     Predict the Next Most Likely Word
```


<p align="center">
  <img src="https://media.licdn.com/dms/image/v2/D4D22AQFuFg980oGU_A/feedshare-shrink_800/B4DZahhAKdHwAg-/0/1746466506016?e=2147483647&v=beta&t=zz501it1S8QTMq032VvfWcy4Qbzy5YkIRHJnmPWTdNo" width="800">
</p>

<p align="center">
  <em><strong>Figure 6:</strong> Next Word Prediction</em>
</p>


## 🚀 Learning Outcome

By the end of this notebook, you will understand how GPT models learn by predicting the next token in a sequence, forming the foundation of autoregressive language generation and modern Large Language Models.


# 📘 Day 5 – Softmax & Cross-Entropy Loss

## 📖 Overview

After a language model predicts scores (known as **logits**) for every word in the vocabulary, these scores must be converted into probabilities. The **Softmax** function transforms raw logits into a probability distribution where the sum of all probabilities equals **1**.

To evaluate how well the model's prediction matches the actual target word, **Cross-Entropy Loss** is used. During training, the model minimizes this loss through backpropagation, enabling it to make increasingly accurate predictions over time.


## 🎯 Objective

* Understand logits and probability distributions.
* Learn how Softmax converts logits into probabilities.
* Calculate Cross-Entropy Loss.
* Understand how language models learn from prediction errors.


## 🧠 Concepts Covered

* Logits
* Softmax Function
* Probability Distribution
* Cross-Entropy Loss
* Model Optimization


## 📊 Visual Workflow

```text
             Context Tokens
                    │
                    ▼
             Neural Network
                    │
                    ▼
                Raw Logits
        [2.3, 0.8, 4.1, 1.5]
                    │
                    ▼
               Softmax Layer
                    │
                    ▼
       Probability Distribution
      [0.08, 0.02, 0.84, 0.06]
                    │
                    ▼
        Compare with True Target
                    │
                    ▼
          Cross-Entropy Loss
                    │
                    ▼
           Update Model Weights
```


<p align="center">
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTGQ0jt_OxWkg4d8GU3hEizkMfCtVBksTgs5wUP-_ZJ8g&s=10" width="800">
</p>

<p align="center">
  <em><strong>Figure 7:</strong> Softmax Probability Distribution</em>
</p>


## 🚀 Learning Outcome

By the end of this notebook, you'll understand how GPT models convert predictions into probabilities and measure prediction errors using **Cross-Entropy Loss** during training.


# 📘 Day 6 – Temperature Sampling

## 📖 Overview

After training, language models generate text by selecting the next token based on predicted probabilities. **Temperature Sampling** controls the randomness of this selection. Lower temperatures produce more deterministic and focused outputs, while higher temperatures encourage more diverse and creative text generation.


## 🎯 Objective

* Understand temperature scaling.
* Compare deterministic and random sampling.
* Observe how temperature affects generated text.
* Learn different decoding strategies.


## 🧠 Concepts Covered

* Temperature
* Greedy Decoding
* Random Sampling
* Probability Scaling
* Text Generation


## 📊 Visual Workflow

```text
              Model Logits
                   │
                   ▼
         Apply Temperature (T)

        T = 0.3      T = 1.0      T = 2.0
           │            │            │
           ▼            ▼            ▼
      Very Sharp     Balanced     More Uniform
      Distribution  Distribution  Distribution
           │            │            │
           ▼            ▼            ▼
    Predict Best     Natural      Creative &
      Token          Output      Random Output
```


<p align="center">
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSHXDMNW5znWqU-stAmwbVBf3HZinKenYdAiEhGmEe8WA&s=10" width="800">
</p>

<p align="center">
  <em><strong>Figure 8:</strong> Temperature Sampling</em>
</p>


## 🚀 Learning Outcome

By the end of this notebook, you'll understand how temperature influences the creativity, diversity, and randomness of generated text.


# 📘 Day 7 – Self-Attention & Causal Masking

## 📖 Overview

**Self-Attention** is the core mechanism behind Transformer models. Instead of processing words sequentially like RNNs, every token attends to other relevant tokens in the sequence. GPT models use **Causal Masking**, ensuring that each token can only attend to previous tokens and never future tokens, preserving autoregressive text generation.


## 🎯 Objective

* Understand the Query-Key-Value mechanism.
* Compute attention scores.
* Learn causal masking.
* Visualize how tokens attend to one another.


## 🧠 Concepts Covered

* Query (Q)
* Key (K)
* Value (V)
* Attention Scores
* Scaled Dot-Product Attention
* Causal Mask


## 📊 Visual Workflow

```text
             Input Embeddings
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
      Query        Key        Value
        │           │           │
        └─────── Dot Product ───┘
                    │
                    ▼
          Attention Score Matrix
                    │
                    ▼
             Apply Causal Mask
                    │
                    ▼
           Softmax Normalization
                    │
                    ▼
         Weighted Sum of Values
                    │
                    ▼
            Attention Output
```


<p align="center">
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRvPP1ufpP6KVQ1Bj21akgFlBz1rCT_b_tG2F_ZxCTTZw&s=10" width="800">
</p>

<p align="center">
  <em><strong>Figure 9:</strong> Self-Attention Mechanism</em>
</p>


## 🚀 Learning Outcome

By the end of this notebook, you'll understand how GPT models learn contextual relationships between words while preventing future information leakage through causal masking.


# 📘 Day 8 – Multi-Head Attention

## 📖 Overview

Rather than using a single attention mechanism, Transformers employ **Multi-Head Attention**, where multiple attention heads learn different contextual relationships simultaneously. Each head focuses on distinct linguistic patterns such as syntax, semantics, or long-range dependencies. Their outputs are then combined to create a richer representation of the input sequence.


## 🎯 Objective

* Understand parallel attention heads.
* Learn how multiple heads capture different relationships.
* Combine attention outputs.
* Build the Multi-Head Attention mechanism.


## 🧠 Concepts Covered

* Attention Heads
* Parallel Attention
* Concatenation
* Linear Projection
* Context Representation


## 📊 Visual Workflow

```text
                Input Embeddings
                       │
      ┌────────┬────────┬────────┬────────┐
      ▼        ▼        ▼        ▼
    Head 1   Head 2   Head 3   Head 4
      │        │        │        │
      ▼        ▼        ▼        ▼
 Attention  Attention Attention Attention
      │        │        │        │
      └────────┴────────┴────────┘
                Concatenate
                     │
                     ▼
            Linear Projection
                     │
                     ▼
         Multi-Head Attention Output
```


<p align="center">
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT0pBcCc1Su--Juaj8SMZlQC1pwyLFOY1YsKOWiU-9MDQ&s=10" width="800">
</p>

<p align="center">
  <em><strong>Figure 10:</strong> Multi-Head Attention</em>
</p>


## 🚀 Learning Outcome

By the end of this notebook, you'll understand how multiple attention heads enable GPT models to capture richer contextual information, significantly improving language understanding and generation.


# 📘 Day 9 – Positional Encoding

## 📖 Definition

Transformers process all words simultaneously instead of one-by-one. Because of this, they have no inherent understanding of word order.

**Positional Encoding** adds mathematical position information to every word embedding so the model knows where each token appears in the sentence.

Instead of learning positions, GPT uses sinusoidal functions (**sine & cosine**) to generate unique positional vectors.

### Final Representation

```text
Final Input = Word Embedding + Positional Encoding
```


## 🎯 Objective

* Understand why transformers need positional information.
* Implement sinusoidal positional encoding.
* Add positional vectors to token embeddings.
* Compare embeddings before and after encoding.


## 🧠 Concepts Learned

* Sequential Information
* Sinusoidal Encoding
* Position Matrix
* Embedding Enhancement
* Sequence Representation


## 🔄 Workflow

```text
Sentence
      │
      ▼
Tokenization
      │
      ▼
Embeddings
      │
      ▼
Generate Positional Encoding
      │
      ▼
Embedding + Position
      │
      ▼
Final Transformer Input
```


<p align="center">
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSLfvfhCNL11wiK_0uvjkUdRopNLamdMsQ7pfa_Kv9Yqw&s=10" width="800">
</p>

<p align="center">
  <em><strong>Figure 11:</strong> Positional Encoding</em>
</p>


## 📂 Implemented

✔ Built positional encoding from scratch

✔ Generated sine & cosine values

✔ Added positional vectors to embeddings

✔ Displayed encoding matrix

✔ Compared original vs encoded embeddings


## 📊 Output

```text
Original Embedding

↓

Positional Encoding Matrix

↓

Encoded Embedding

↓

Transformer Ready Input
```


## 📚 Mathematical Formula

### For even dimensions

```text
PE(pos,2i)=sin(pos/10000^(2i/d))
```

### For odd dimensions

```text
PE(pos,2i+1)=cos(pos/10000^(2i/d))
```


# 📘 Day 10 – Feed Forward Network, Layer Normalization & Residual Connections

## 📖 Definition

Every transformer block consists of two major parts:

* Multi-Head Self Attention
* Feed Forward Neural Network

To improve stability and preserve information, GPT also uses:

* Residual Connections
* Layer Normalization

These together allow transformers to train very deep networks efficiently.


## 🎯 Objective

* Build the Feed Forward Network.
* Understand skip (residual) connections.
* Learn Layer Normalization.
* Combine them into one Transformer Block.


## 🧠 Concepts Learned

* Linear Layers
* ReLU Activation
* Residual Connection
* LayerNorm
* Transformer Block Architecture


## 🔄 Workflow

```text
Input
   │
   ▼
Multi-Head Attention
   │
   ▼
Residual Connection
   │
   ▼
LayerNorm
   │
   ▼
Feed Forward Network
   │
   ▼
Residual Connection
   │
   ▼
LayerNorm
   │
   ▼
Transformer Block Output
```


<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:876/1*7sjcgd_nyODdLbZSxyxz_g.png" width="800">
</p>

<p align="center">
  <em><strong>Figure 12:</strong> Transformer Block</em>
</p>


## 📂 Implemented

✔ Feed Forward Neural Network

✔ Residual Connection

✔ Layer Normalization

✔ Complete Transformer Block

✔ Output Visualization


## 📊 Output

```text
Input Tensor

↓

Attention Output

↓

Feed Forward Output

↓

Normalized Output
```


# 📘 Day 11 – Building the Mini GPT Architecture

## 📖 Definition

After implementing all the individual components, they are assembled together into a complete GPT architecture.

This includes:

* Token Embeddings
* Positional Encoding
* Multiple Transformer Blocks
* Final Linear Layer
* Vocabulary Prediction

The model now behaves like a simplified GPT.


## 🎯 Objective

Build a complete transformer language model capable of predicting the next word.


## 🧠 Concepts Learned

* GPT Pipeline
* Model Architecture
* Stacking Layers
* Vocabulary Prediction
* Forward Propagation


## 🔄 Workflow

```text
Input Tokens
      │
      ▼
Token Embedding
      │
      ▼
Positional Encoding
      │
      ▼
Transformer Block 1
      │
      ▼
Transformer Block 2
      │
      ▼
Linear Layer
      │
      ▼
Vocabulary Scores (Logits)
```


<p align="center">
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQtde2mtSyn8a3AKERWRE6xkQpYwU-Am8LrDWI4CD_yvw&s=10" width="800">
</p>

<p align="center">
  <em><strong>Figure 13:</strong> Mini GPT Architecture</em>
</p>


## 📂 Implemented

✔ MiniGPT Class

✔ Transformer Block Stack

✔ Embedding Layer

✔ Positional Encoding

✔ Output Projection Layer


## 📊 Output

```text
Sentence

↓

Logits

↓

Vocabulary Prediction
```



# 📘 Day 12 – Training & Text Generation (NovaGPT)

## 📖 Definition

The final day combines everything built during the previous eleven days into a fully functional miniature GPT language model.

The model learns language patterns from a small dataset and generates new text using autoregressive prediction.

This project demonstrates the complete workflow behind GPT—from raw text to generated sentences.

### 🚀 Project Name

**NovaGPT (Mini GPT from Scratch)**


## 🎯 Objective

* Train a GPT model.
* Predict next words.
* Generate complete sentences.
* Build an interactive text generator.


## 🧠 Concepts Learned

* Autoregressive Language Modeling
* Training Loop
* Cross Entropy Loss
* Adam Optimizer
* Temperature Sampling
* Text Generation


## 🔄 Complete GPT Pipeline

```text
Raw Text
    │
    ▼
Tokenization
    │
    ▼
Vocabulary
    │
    ▼
Embeddings
    │
    ▼
Positional Encoding
    │
    ▼
Multi Head Attention
    │
    ▼
Feed Forward
    │
    ▼
Transformer Blocks
    │
    ▼
Vocabulary Prediction
    │
    ▼
Next Word
    │
    ▼
Generated Sentence
```


<p align="center">
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRAmrEeK-Nbl83xGeZgLyMj951q4ZpsVadzGbHEm2H_vw&s=10" width="800">
</p>

<p align="center">
  <em><strong>Figure 14:</strong> NovaGPT End-to-End Pipeline</em>
</p>


## 📂 Implemented

✔ Complete Mini GPT Model

✔ Training Pipeline

✔ Forward Pass

✔ Loss Calculation

✔ Backpropagation

✔ Adam Optimizer

✔ Temperature Sampling

✔ Interactive GPT

✔ Prompt-based Generation


## 📊 Sample Output

### Prompt

```text
Artificial Intelligence
```

### Generated Text

```text
Artificial intelligence is transforming the world through deep learning and modern language models.
```

---

### Prompt

```text
Machine Learning
```

### Generated Text

```text
Machine learning enables computers to discover hidden patterns from large datasets.
```

---

### Prompt

```text
Python
```

### Generated Text

```text
Python is widely used for artificial intelligence, deep learning and data science applications.
```


# 🏆 Final Learning Journey

```text
Day 1  → Tokenization

↓

Day 2  → Vocabulary

↓

Day 3  → Embeddings

↓

Day 4  → Next Word Prediction

↓

Day 5  → Softmax & Cross Entropy

↓

Day 6  → Temperature Sampling

↓

Day 7  → Self Attention

↓

Day 8  → Multi Head Attention

↓

Day 9  → Positional Encoding

↓

Day 10 → Feed Forward + LayerNorm + Residual

↓

Day 11 → Mini GPT Architecture

↓

Day 12 → NovaGPT (Training + Text Generation)
```
