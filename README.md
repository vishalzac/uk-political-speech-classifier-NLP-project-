# 🇬🇧 UK Political Party Classifier — NLP Project

Predicting whether a UK conference speech was delivered by a **Labour** or **Conservative** politician — using only the text of the speech.

**Lancaster University | SCC453 | 2025**

---

## 📌 Overview

This project builds a text classification pipeline on the [UK Political Speeches dataset](https://www.kaggle.com/datasets/andrewsale/uk-political-speeches) (Kaggle). Given a conference speech, the model predicts which party the speaker belongs to.

**Best result: 89.6% accuracy — Logistic Regression with TF-IDF bigrams**

---

## 📂 Dataset

- **Source:** UK Political Speeches dataset (Kaggle)
- **File used:** `speeches_bps.csv` — the only file with party labels
- **Corpus:** 240 speeches (123 Labour, 117 Conservative)
- **Class balance:** 95.1%
- **Average speech length:** ~2,916 words

---

## 🔧 Pipeline

### 1. Preprocessing
- Lowercasing and removal of non-alphabetic characters
- Tokenisation with NLTK
- Stop word removal — standard NLTK list + **52 custom political speech terms** (e.g. *applause*, *minister*, *conference*)
- Lemmatisation with WordNetLemmatizer
- Tokens of ≤2 characters discarded
- Retained ~51.3% of words per speech on average

### 2. Iterative Noise Removal
Discovered **cross-party reference bias** — Labour speakers frequently name Conservative opponents (*tories*, *tory*), causing these to act as false Labour signals. Party names and other noise terms were added to the stop list, growing it to **281 terms total**.

> This improved Logistic Regression accuracy from **87.5% → 89.6%**

### 3. Feature Extraction
- TF-IDF vectorisation (max 5,000 features)
- Unigrams + bigrams (`ngram_range=(1,2)`)
- `min_df=2`, `sublinear_tf=True`
- Vectoriser fitted on training data only (no data leakage)

### 4. Classifiers
| Model | CV F1 | Test Accuracy | Test F1 |
|---|---|---|---|
| **Logistic Regression** | 0.863 | **89.6%** | **0.895** |
| Naive Bayes | 0.847 | 87.5% | 0.875 |
| Random Forest | 0.880 | 83.3% | 0.833 |

- 80/20 stratified train/test split (`random_state=42`)
- Stratified 5-fold cross-validation on training set

---

## 🔍 Key Findings

**Labour-predictive terms:** poverty, investment, minimum wage, fairness, full employment

**Conservative-predictive terms:** empire, free enterprise, tariff, savings, nation, prosperity

**Interesting misclassifications (all at low model confidence 0.53–0.63):**
- **Theresa May (2016, 2017)** — predicted Labour; her "burning injustices" framing sounded Labour-like
- **Margaret Thatcher (1990)** — predicted Labour; speech was for a children's welfare charity
- **Ramsay MacDonald (1924)** — predicted Conservative; deliberately used moderate language as the first Labour PM
- **Michael Foot (1981)** — predicted Conservative; his classical oratorical style differed from typical Labour vocabulary

---

## 🛠️ Tech Stack

- Python, Jupyter Notebook
- `scikit-learn`, `nltk`, `pandas`
- Dataset loaded via `kagglehub`

---

## 🚀 Run it Yourself

```bash
# Clone the repo
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# Install dependencies
pip install kagglehub pandas scikit-learn nltk

# Open the notebook
jupyter notebook Project_Scc_453_Final_Code.ipynb
```

The notebook downloads the dataset automatically via `kagglehub` — no manual file upload needed.

---

## 📄 Report

Full write-up available in `Final_report_Scc453.pdf` — covers methodology, results, misclassification analysis, and future work directions (BERT fine-tuning, Hansard transcripts, temporal modelling).
