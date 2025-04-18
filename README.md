# Fake News Classifier & NLP Pipeline

This project performs **fake news classification** using a variety of Natural Language Processing (NLP) techniques. It explores how linguistic features differ between fake and factual news, and uses this insight to build a machine learning classifier. The workflow includes preprocessing, part-of-speech tagging, named entity recognition, sentiment analysis, topic modeling, and finally, text classification.

---

## Objective

To detect fake news by:

- Cleaning and preprocessing raw news text
- Exploring linguistic patterns in fake vs factual content
- Extracting sentiment scores using VADER
- Performing topic modeling with LDA and LSA
- Training classification models to predict if news is fake or factual

---

## Files Included

- `nlp_fake_new_classifier.ipynb` – Jupyter Notebook containing the full workflow
- `requirements.txt` – Python dependencies required to run the notebook
- `fake_news.csv` – Dataset used in this project (ensure this is placed in the same directory)

---

## Libraries Used

- `pandas`, `matplotlib`, `seaborn` – Data handling and visualization
- `nltk`, `spacy`, `gensim` – NLP preprocessing and modeling
- `vaderSentiment` – Sentiment analysis
- `scikit-learn` – Text vectorization and classification

---

## How to Run This Project

### 1. Clone the Repository

```bash
git clone https://github.com/pratyushkr9420/fake_news_classifier.git
cd fake_news_classifier
```

### 2. Install Required Packages

Create a virtual environment (recommended) and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download NLP Resources

After installing dependencies, run the following commands to download required models and corpora:

```bash
python -m nltk.downloader punkt stopwords wordnet
python -m spacy download en_core_web_sm
```

---

## What's Inside the Notebook

### 1. Data Exploration

- Visualizing class distribution of fake and factual news
- Checking for null values

### 2. POS Tagging & Named Entity Recognition

- Using `spaCy` to extract token-level linguistic features
- Visualizing most common parts of speech and named entities

### 3. Text Preprocessing

- Lowercasing, punctuation & stopword removal
- Tokenization and lemmatization

### 4. Sentiment Analysis

- Scoring articles using `VADER`
- Comparing sentiment trends in fake vs factual news

### 5. Topic Modeling

- Latent Dirichlet Allocation (LDA)
- Latent Semantic Analysis (LSA) with TF-IDF

### 6. Text Classification

- Vectorization using `CountVectorizer`
- Logistic Regression and Support Vector Machine (SGDClassifier)
- Evaluation using accuracy and classification report

---

## Sample Outputs

- Accuracy scores of classification models
- Charts for token frequency, POS tags, sentiment distribution
- Coherence scores for topic modeling
- Interpretable topics from LDA/LSA

---

## Author

Pratyush Kumar – [kumarpratyush6294@gmail.com](mailto:your.email@example.com)

---

## License

This project is licensed under the [MIT License](LICENSE).
