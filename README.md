🛍️ Sentiment Analysis Dashboard for Product & App Reviews

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sentiment-analysis-dl.streamlit.app/)

A machine learning web dashboard built with **Python**, **Streamlit**, and **Scikit-Learn** implementing the **Multinomial Naive Bayes** algorithm to classify customer reviews of products and mobile applications into **Positive** and **Negative** sentiment.

> 🚀 **Live Demo**: Try the web application live without any installation at [**sentiment-analysis-dl.streamlit.app**](https://sentiment-analysis-dl.streamlit.app/)

> **Link Video Presentasi**: [**Klik di sini**](https://drive.google.com/file/d/1M9jGlk7VNSiSU1MaRCgjYbOEhIFpmc7T/view?usp=drivesdk)

---

🚀 Key Features

- **📊 Model Evaluation & Comparison**: Side-by-side comparative analysis of Naive Bayes powered by **Word Frequency (`CountVectorizer`)** vs. **TF-IDF (`TfidfVectorizer`)** evaluated on the identical stratified test set (Accuracy, Weighted Precision, Recall, F1-Score, and Confusion Matrix).
- **🔍 Misclassification Inspector**: Deep-dive error analysis section that isolates test instances where model predictions deviated from ground truth labels.
- **✍️ Single Review Sentiment Analysis**: Real-time sentiment classification with confidence score gauges, interactive probability donut charts, and keyword contribution chips (*Feature Highlights*).
- **📁 Batch Processing & CSV Export**: Upload custom CSV/Excel review datasets, run high-speed batch predictions, inspect statistical breakdowns, and export enriched classification results to CSV.
- **📖 Dataset Explorer & Theoretical Formulations**: Comprehensive theoretical breakdown with interactive $\KaTeX$ formulas for Bayes' Theorem, Laplace Smoothing, and TF-IDF weighting, along with a searchable dataset filter.

🧠 Methodology & Mathematical Formulation

1. Multinomial Naive Bayes (MNB)
Multinomial Naive Bayes models the distribution of word frequencies in text documents:

$$P(c|d) = \frac{P(c) \cdot P(d|c)}{P(d)}$$

Under the naive conditional independence assumption:

$$P(c|d) \propto P(c) \prod_{i=1}^{n} P(t_i|c)$$

Where:
- $P(c)$ represents the class prior probability ($c \in \{\text{positive}, \text{negative}\}$).
- $P(t_i|c)$ represents the likelihood of term $t_i$ given class $c$, estimated using **Laplace Smoothing** ($\alpha = 1.0$):

$$P(t_i|c) = \frac{N_{c, t_i} + \alpha}{\sum_{t'} N_{c, t'} + \alpha \cdot |V|}$$

2. Feature Extraction: CountVectorizer vs TF-IDF
- **Word Frequency (`CountVectorizer`)**: Quantifies raw term occurrences within the document.
- **TF-IDF (`TfidfVectorizer`)**: Discounts frequent global stopwords while prioritizing discriminative, category-specific terms:

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{|D|}{1 + |\{d \in D : t \in d\}|}\right)$$


📂 Project Structure

```text
sentiment-analysis-product-app/
├── .gitignore               # Git ignore rules for venv, caches, and temp files
├── README.md                # Project documentation and setup guide
├── requirements.txt         # Python project dependencies
├── app.py                   # Streamlit web application & interactive UI
├── streamlit_app.py         # Entry point for Streamlit Community Cloud
├── data/
│   └── dataset_ulasan.csv   # Labeled dataset of product & application reviews
└── src/
    ├── preprocessing.py     # Text cleaning, normalization & tokenization
    ├── model_trainer.py     # Scikit-learn model training & evaluation pipeline
    └── sentiment_engine.py  # Inference engine for single & batch predictions
```

---

## 🛠️ Installation & Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/DnielLim/sentiment-analysis-product-app.git
cd sentiment-analysis-product-app
```

### 2. Set Up Virtual Environment & Dependencies
```bash
# On Windows
python -m venv .venv
.venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

3. Launch the Application
```bash
streamlit run app.py
```

Open your web browser and navigate to `http://localhost:8501`.

---

📦 Requirements

- `streamlit>=1.32.0`
- `pandas>=2.0.0`
- `numpy>=1.24.0`
- `scikit-learn>=1.3.0`
- `plotly>=5.18.0`
- `openpyxl>=3.1.0`


📄 License

This project is licensed under the [MIT License](LICENSE).
