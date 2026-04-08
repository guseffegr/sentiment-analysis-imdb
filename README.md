# Logistic Regression vs Neural Network on IMDb Dataset

A comparative study of Logistic Regression and a Feedforward Neural Network for binary sentiment classification on the IMDb movie review dataset. Text is represented using TF-IDF features. The goal is to evaluate how well each model separates positive from negative reviews.

---

## Task

**Binary sentiment classification** — predict whether a movie review is positive (1) or negative (0).

Dataset: [IMDb Large Movie Review Dataset](https://huggingface.co/datasets/imdb) (25,000 train / 25,000 test, perfectly balanced).

---

## Approach

### Preprocessing
- Lowercase conversion
- Removal of HTML tags (`<br />`)
- Punctuation removal via regex `[^a-z\s]`

### Vectorization — TF-IDF
| Parameter | Value | Reason |
|---|---|---|
| `max_features` | 30,000 | Limit vocabulary to most frequent tokens |
| `ngram_range` | (1, 2) | Capture unigrams and bigrams (e.g. *"not good"*) |
| `sublinear_tf` | True | Log-normalize term frequencies |
| `min_df` | 2 | Remove very rare tokens |
| `max_df` | 0.95 | Remove near-universal tokens |

---

## Models

### Baseline — Logistic Regression
Standard `sklearn` `LogisticRegression` trained on TF-IDF sparse vectors.

### Neural Network (PyTorch)
Architecture: **30000 → 256 → 64 → 1**

| Layer | Detail |
|---|---|
| Linear(30000, 256) | Input layer |
| BatchNorm1d + ReLU + Dropout(0.6) | Regularisation |
| Linear(256, 64) | Hidden layer |
| BatchNorm1d + ReLU + Dropout(0.4) | Regularisation |
| Linear(64, 1) | Output (logit) |

**Training config:**
- Loss: `BCEWithLogitsLoss`
- Optimizer: `Adam(lr=1e-5, weight_decay=1e-2)`
- Epochs: 10
- Batch size: 64

---

## Results

| Model | Accuracy |
|---|---|
| Logistic Regression | 89.89% |
| Neural Network | 90.25% |

Both models achieve similar results because TF-IDF features are already largely linearly separable. The NN shows a slight edge thanks to its non-linearity.

---

## Project Structure

```
name.ipynb          # Main notebook
README.md           # This file
```

---

## Requirements

```
torch
scikit-learn
datasets
numpy
matplotlib
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Limitations & Future Work

1. **Lemmatization** — reduce words to base form to lower noise and potentially improve accuracy
2. **Optuna hyperparameter search** — systematic search for optimal learning rate, dropout, architecture
3. **Sequence-aware models** — LSTM or BERT to capture word order and contextual meaning, which TF-IDF-based models cannot
