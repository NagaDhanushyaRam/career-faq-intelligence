# 🎯 Career FAQ Intelligence

AI-powered FAQ recommendation system for job and career questions.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

## 🚀 Features

- **Dual Search Modes:** TF-IDF (keyword) and Sentence-BERT (semantic)
- **24,724 Career FAQs** from 3 datasets
- **Fast Retrieval** with FAISS indexing
- **Professional UI** built with Streamlit

## 📊 Datasets

| Source | Records |
|--------|---------|
| CareerVillage Q&A | 23,064 |
| Entry Level Career QA | 1,620 |
| HR Interview Questions | 40 |

## 🛠️ Tech Stack

- Python 3.11+
- Streamlit
- Sentence-Transformers (SBERT)
- FAISS
- scikit-learn (TF-IDF)
- NLTK

## 📁 Project Structure

```
├── app.py                 # Streamlit app (deployment version)
├── streamlit_app.py       # Full app with API integration
├── api/
│   └── main.py           # FastAPI backend
├── src/
│   ├── data_pipeline.py  # Data processing
│   ├── preprocessing.py  # Text preprocessing
│   ├── tfidf_retriever.py
│   └── sbert_retriever.py
├── data/
│   └── processed/
│       └── faq_corpus.csv
├── requirements.txt
└── config.py
```

## 🚀 Quick Start

### Local Development

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/career-faq-intelligence.git
cd career-faq-intelligence

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### With API (Full Version)

```bash
# Terminal 1: Start API
python -m uvicorn api.main:app --reload --port 8000

# Terminal 2: Start UI
streamlit run streamlit_app.py
```

## 📈 Evaluation Results

| Metric | TF-IDF | SBERT |
|--------|--------|-------|
| P@1 | 0.77 | **0.79** |
| MRR | **0.812** | 0.810 |
| P@5 | **0.87** | 0.84 |

## 📝 License

MIT License

## 👥 Authors

- Naga Dhanushya Ram Munnanuru
- Jaya Peda Vignesh Reddy Duggempudi

*COSC 757 - Data Mining, Towson University, Fall 2025*
