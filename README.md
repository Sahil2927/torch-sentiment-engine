# Twitter Entity Sentiment Analysis via From-Scratch Transformer Encoder

---

## Project Overview
This project implements a complete, production-grade Natural Language Processing (NLP) pipeline that trains a custom **Transformer Encoder architecture built entirely from scratch** using PyTorch. The model is trained on the Kaggle Twitter Entity Sentiment Analysis dataset to classify social media posts into four distinct categories: **Positive**, **Negative**, **Neutral**, or **Irrelevant**. 

Developed and optimized inside Google Colab leveraging a **T4 GPU**, this model achieves an exceptional **94.20% validation accuracy** over 20 epochs without any pre-trained language model weights, showing robust generalization and zero overfitting.

---

## Key Features
* **Custom Transformer Engineering:** Every core mathematical block—including Scaled Dot-Product Attention, Multi-Head Attention routing, and Layer Normalization skip connections—is written manually in PyTorch.
* **Automated Data Pipelines:** Integrates directly with the Kaggle API to download, extract, clean, and map text fields seamlessly inside a temporary environment.
* **Production-Grade Analytics Suite:** Generates an end-to-end evaluation breakdown including a Scikit-Learn classification report and a localized Seaborn Confusion Matrix graphic.
* **Cloud Storage Integration:** Includes a specialized backup workflow that automatically mounts Google Drive to preserve trained model weights (`.pt`), tokenizer structural states, text metric summaries, and visual evaluation charts.

---

## Architecture Design

The model uses a pure Encoder-Only architecture designed for high-fidelity token contextualization before sequence reduction.

`Raw Text Tweet ➔ BERT Tokenizer ➔ Token & Positional Embeddings ➔ Stacked Encoder Layers (MHA + FFN) ➔ Masked Mean Pooling ➔ Linear Classification Head ➔ Sentiment Probability`

### Mathematical Formulations

#### 1. Multi-Head Self-Attention
Contextual dynamics are modeled using multi-lens scaling projection maps:

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

#### 2. Position-Wise Feed-Forward Network (FFN)
Non-linear spatial transformations are enforced independently at each sequence step via a GELU activation layer:

$$\text{FFN}(x) = \text{GELU}(xW_1+b_1)W_2+b_2$$

### Architectural Hyperparameters
* **Vocabulary Size:** 30,522 (BERT Uncased Mapping)
* **Max Sequence Length:** 64 Tokens
* **Hidden Dimension ($d_{model}$):** 256
* **Attention Heads ($H$):** 8
* **Feed-Forward Expansion Size ($d_{ff}$):** 512
* **Stacked Encoder Layers:** 3
* **Target Categories:** 4

---

## Model Performance & Results

The architecture demonstrated highly stable optimization metrics across a comprehensive 20-epoch training curriculum:

* **Final Training Accuracy:** 94.13%
* **Final Validation Accuracy:** 94.20%
* **Training Convergence Speed:** ~38 iterations/second on Nvidia T4 Tensor Core hardware.

### Final Classification Report

| Sentiment Class | Precision | Recall | F1-Score | Evaluation Support |
| :--- | :--- | :--- | :--- | :--- |
| **Negative** | 0.93 | 0.97 | 0.95 | 266 |
| **Positive** | 0.97 | 0.92 | 0.95 | 277 |
| **Neutral** | 0.93 | 0.94 | 0.93 | 285 |
| **Irrelevant** | 0.93 | 0.94 | 0.93 | 172 |
| **Overall Accuracy** | | | **0.94** | **1000** |

---

## Tech Stack & Libraries
* **Core Framework:** Python 3.x, PyTorch (CUDA Accelerator)
* **Dataset Management:** Kaggle API, Pandas, NumPy
* **Tokenization:** Hugging Face `transformers` (BERT Tokenizer Engine)
* **Metrics & Plotting:** Scikit-Learn, Matplotlib, Seaborn
* **Environment:** Google Colab (T4 GPU Hardware Runtime)

---

## Getting Started & Execution Flow

### 1. Repository Setup
Clone this repository and ensure all mandatory baseline frameworks are installed in your execution environment:
```bash
pip install torch transformers datasets scikit-learn matplotlib seaborn tqdm
