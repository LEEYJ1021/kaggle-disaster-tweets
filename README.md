# Kaggle — NLP with Disaster Tweets 🚨

## Overview
This repository contains my mini-project submission for the Kaggle competition  
[**Natural Language Processing with Disaster Tweets**](https://www.kaggle.com/competitions/nlp-getting-started).

**Goal:** Build a compact, runnable Jupyter Notebook that demonstrates:  
- Quick exploratory data analysis (EDA)  
- A fast baseline model using **TF-IDF + Logistic Regression**  
- A lightweight deep learning model using **Embedding + Bi-LSTM**  
- Focus on **speed** (fast iteration, GPU if available)

- 
---

## Dataset
- Provided by Kaggle: `/kaggle/input/nlp-getting-started`
- Files:
  - `train.csv` (7613 rows, 5 columns)
  - `test.csv` (3263 rows, 4 columns)
  - `sample_submission.csv`

**Target:**  
- `0` → Not disaster  
- `1` → Disaster  

---

## Results

### Baseline — TF-IDF + Logistic Regression
- **Accuracy:** ~0.81  
- **F1 Score:** ~0.76  

### Deep Model — Bi-LSTM
- **Accuracy:** ~0.80  
- **F1 Score:** ~0.75  

**Key takeaway:** Logistic Regression baseline is strong and very fast; the Bi-LSTM improves recall but requires more training time.

---

## Quick Tips for Speed
- Use small vocab (10k–15k words) and short sequence length (60–100).  
- TF-IDF + Logistic Regression gives quick leaderboard results.  
- Enable **GPU** on Kaggle for faster LSTM training.  
- Try small pretrained embeddings (e.g., GloVe 50d) for accuracy boosts.  

---

## Deliverables
1. **Notebook**: `disaster_tweets_fast.ipynb` (with alternating Markdown & Code)  
2. **GitHub repo**: this repository  
3. **Kaggle leaderboard screenshot**: uploaded separately  

---

## References
- Kaggle competition page: [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)  
- [scikit-learn documentation](https://scikit-learn.org/stable/) — TF-IDF, LogisticRegression  
- [TensorFlow / Keras docs](https://www.tensorflow.org/api_docs) — Tokenizer, Embedding, LSTM  

---

## Notes
- Run the notebook **top-to-bottom** on Kaggle.  
- Enable GPU in Notebook settings for deep learning.  
- Submissions (`submission_baseline.csv`, `submission_lstm.csv`) are ready to upload to Kaggle.  
