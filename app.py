import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import os

from src.model_trainer import SentimentModelTrainer
from src.sentiment_engine import SentimentEngine
from src.preprocessing import clean_text

# Page Config
st.set_page_config(
    page_title="Product & App Review Sentiment Analysis Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (CSS - Modern Dark Theme & Glassmorphism)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
    }
    
    /* Fix Streamlit Top Bar */
    header, [data-testid="stHeader"], .stAppHeader, [data-testid="stAppHeader"], div[id*="stHeader"], [data-testid="stToolbar"] {
        background-color: #0b0f19 !important;
        background: #0b0f19 !important;
        color: #f1f5f9 !important;
    }
    header *, [data-testid="stHeader"] *, .stAppHeader * {
        background-color: transparent !important;
        color: #f1f5f9 !important;
    }
    .main .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
    }
    
    /* Hero Header Banner */
    .hero-card {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 40%, #4338ca 100%);
        padding: 30px 36px;
        border-radius: 20px;
        color: white;
        box-shadow: 0 20px 35px -10px rgba(67, 56, 202, 0.4);
        margin-bottom: 28px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        position: relative;
        overflow: hidden;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 10px;
        background: linear-gradient(90deg, #ffffff 0%, #cbd5e1 50%, #a5b4fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #c7d2fe;
        max-width: 950px;
        line-height: 1.6;
        font-weight: 400;
    }
    
    /* Custom Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }
    
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px 16px;
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid #334155;
        border-radius: 14px;
        margin-bottom: 20px;
    }
    .sidebar-brand-icon {
        background: linear-gradient(135deg, #6366f1, #4338ca);
        width: 44px;
        height: 44px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
    }
    .sidebar-brand-text {
        font-size: 1.15rem;
        font-weight: 700;
        color: #ffffff;
    }
    .sidebar-brand-sub {
        font-size: 0.8rem;
        color: #94a3b8;
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 18px 20px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #6366f1;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #ffffff;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 4px;
    }
    
    /* Sentiment Result Badges */
    .badge-positive {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 14px 24px;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.3) 100%);
        border: 1px solid #10b981;
        border-radius: 14px;
        color: #34d399;
        font-weight: 700;
        font-size: 1.4rem;
        box-shadow: 0 10px 25px -5px rgba(16, 185, 129, 0.25);
    }
    .badge-negative {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 14px 24px;
        background: linear-gradient(135deg, rgba(244, 63, 94, 0.2) 0%, rgba(225, 29, 72, 0.3) 100%);
        border: 1px solid #f43f5e;
        border-radius: 14px;
        color: #fb7185;
        font-weight: 700;
        font-size: 1.4rem;
        box-shadow: 0 10px 25px -5px rgba(244, 63, 94, 0.25);
    }
    
    /* Highlight Chips */
    .chip {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 4px;
    }
    .chip-pos {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .chip-neg {
        background-color: rgba(244, 63, 94, 0.15);
        color: #fb7185;
        border: 1px solid rgba(244, 63, 94, 0.4);
    }

    /* Custom Styled Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #0f172a;
        padding: 6px;
        border-radius: 14px;
        border: 1px solid #1e293b;
    }
    .stTabs [data-baseweb="tab"] {
        height: 46px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 10px;
        color: #94a3b8;
        font-weight: 600;
        border: none;
        padding: 8px 18px;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-icon">🤖</div>
        <div>
            <div class="sidebar-brand-text">AI Sentiment Lab</div>
            <div class="sidebar-brand-sub">Product & App Reviews</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("📂 Sumber Data Training")
    dataset_options = {}
    if os.path.exists("data/clean_data.csv"):
        dataset_options["clean_data.csv (Hasil Preprocessing M1)"] = "data/clean_data.csv"
    if os.path.exists("data/dataset_ulasan.csv"):
        dataset_options["dataset_ulasan.csv (Bawaan Git)"] = "data/dataset_ulasan.csv"
    dataset_options["📤 Upload File CSV Sendiri"] = "upload"
        
    selected_dataset_label = st.selectbox(
        "Pilih Sumber Dataset:",
        options=list(dataset_options.keys()),
        index=0
    )
    selected_dataset_path = dataset_options[selected_dataset_label]

    uploaded_df = None
    if selected_dataset_path == "upload":
        uploaded_train_file = st.file_uploader(
            "Upload File CSV untuk Training:",
            type=["csv"],
            help="File harus memiliki kolom teks ulasan dan label sentimen ('positif'/'negatif')."
        )
        if uploaded_train_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_train_file)
                st.caption(f"📁 Ditemukan {len(uploaded_df)} baris data.")
                col_txt = st.selectbox("Kolom Ulasan:", options=uploaded_df.columns, index=0)
                available_sent_cols = [c for c in uploaded_df.columns if c != col_txt]
                if available_sent_cols:
                    col_snt = st.selectbox("Kolom Sentimen:", options=available_sent_cols, index=0)
                    uploaded_df = uploaded_df.rename(columns={col_txt: "ulasan", col_snt: "sentimen"})
                else:
                    st.warning("File memerlukan kolom label sentimen!")
            except Exception as e:
                st.error(f"Gagal membaca file: {e}")

    train_button = st.button("🚀 Latih Model Sekarang", type="primary", use_container_width=True)

    if train_button:
        if selected_dataset_path == "upload" and uploaded_df is None:
            st.error("⚠️ Silakan upload file CSV terlebih dahulu sebelum melatih model!")
        else:
            with st.spinner("Sedang melatih model Naive Bayes..."):
                new_trainer = SentimentModelTrainer()
                if selected_dataset_path == "upload":
                    new_trainer.train_models(df_input=uploaded_df)
                    st.session_state["trained_dataset"] = f"Uploaded CSV ({len(uploaded_df)} ulasan)"
                else:
                    new_trainer.data_path = selected_dataset_path
                    new_trainer.train_models()
                    st.session_state["trained_dataset"] = selected_dataset_label
                new_engine = SentimentEngine(new_trainer)
                st.session_state["trainer"] = new_trainer
                st.session_state["engine"] = new_engine
            st.success("✅ Model berhasil dilatih!")

    is_trained = "trainer" in st.session_state and st.session_state["trainer"] is not None

    if is_trained:
        st.divider()
        st.subheader("⚙️ Algorithm Settings")
        
        active_model_key = st.radio(
            "Select Text Feature Extractor:",
            options=["tfidf", "count"],
            format_func=lambda x: "Multinomial Naive Bayes (TF-IDF)" if x == "tfidf" else "Multinomial Naive Bayes (Word Frequency)",
            help="Compare Naive Bayes performance using TF-IDF weighting vs. Word Frequency (CountVectorizer)."
        )
        
        selected_model_name = "TF-IDF" if active_model_key == "tfidf" else "Word Frequency"
        trainer = st.session_state["trainer"]
        engine = st.session_state["engine"]
        results = trainer.results
        active_results = results[active_model_key]
        
        st.divider()
        st.subheader("📊 Training Dataset Statistics")
        st.markdown(f"""
        - 📂 **Dataset Aktif:** `{st.session_state.get('trained_dataset', selected_dataset_label)}`
        - 📦 **Total Dataset:** `{len(trainer.df)} reviews`
        - 🏋️ **Training Set (80%):** `{len(trainer.X_train)} reviews`
        - 🧪 **Test Set (20%):** `{len(trainer.X_test)} reviews`
        - 🎯 **Split Method:** `Stratified Split (random_state=42)`
        """)

# Hero Header Banner
st.markdown("""
<div class="hero-card">
    <div class="hero-title">🛍️ Product & App Review Sentiment Analysis Dashboard</div>
    <div class="hero-subtitle">
        Supervised text classification system utilizing the <b>Multinomial Naive Bayes</b> algorithm to categorize customer reviews into <b>Positive</b> or <b>Negative</b> sentiment. Features interactive side-by-side performance evaluation between <b>Word Frequency (CountVectorizer)</b> and <b>TF-IDF</b> feature extractors.
    </div>
</div>
""", unsafe_allow_html=True)

# Guard: Jika model belum dilatih, tampilkan panduan untuk klik tombol latih
if not is_trained:
    st.info("👈 **Model belum dilatih.** Silakan pilih sumber dataset di sidebar sebelah kiri, kemudian klik tombol **'🚀 Latih Model Sekarang'** untuk memulai training dan melihat evaluasi model.")
    st.stop()

# Main Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Model Evaluation & Comparison",
    "✍️ Single Review Prediction",
    "📁 Batch Prediction & Export",
    "📖 Dataset Explorer & Theory"
])

# ==============================================================================
# TAB 1: MODEL EVALUATION & COMPARISON
# ==============================================================================
with tab1:
    st.header("📊 Model Performance Evaluation & Comparison")
    st.markdown("Experimental comparison of **Multinomial Naive Bayes** using **Word Frequency** vs. **TF-IDF** on the **exact same** stratified test split (20% test split).")
    
    count_res = results["count"]
    tfidf_res = results["tfidf"]
    
    # Side-by-side Metric Cards
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Accuracy</div>
            <div class="metric-value">{tfidf_res['accuracy']*100:.1f}%</div>
            <div class="metric-sub">TF-IDF vs {count_res['accuracy']*100:.1f}% Count</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Weighted Precision</div>
            <div class="metric-value">{tfidf_res['precision']*100:.1f}%</div>
            <div class="metric-sub">TF-IDF vs {count_res['precision']*100:.1f}% Count</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Weighted Recall</div>
            <div class="metric-value">{tfidf_res['recall']*100:.1f}%</div>
            <div class="metric-sub">TF-IDF vs {count_res['recall']*100:.1f}% Count</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Weighted F1-Score</div>
            <div class="metric-value">{tfidf_res['f1_score']*100:.1f}%</div>
            <div class="metric-sub">TF-IDF vs {count_res['f1_score']*100:.1f}% Count</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.write("")
    
    # Comparison Chart & Confusion Matrix
    col_chart, col_cm = st.columns([1.1, 0.9])
    
    with col_chart:
        st.subheader("📈 Evaluation Metrics Comparison")
        metrics_df = pd.DataFrame([
            {"Model": "Naive Bayes (Word Frequency)", "Metric": "Accuracy", "Score (%)": count_res["accuracy"] * 100},
            {"Model": "Naive Bayes (Word Frequency)", "Metric": "Precision", "Score (%)": count_res["precision"] * 100},
            {"Model": "Naive Bayes (Word Frequency)", "Metric": "Recall", "Score (%)": count_res["recall"] * 100},
            {"Model": "Naive Bayes (Word Frequency)", "Metric": "F1-Score", "Score (%)": count_res["f1_score"] * 100},
            
            {"Model": "Naive Bayes (TF-IDF)", "Metric": "Accuracy", "Score (%)": tfidf_res["accuracy"] * 100},
            {"Model": "Naive Bayes (TF-IDF)", "Metric": "Precision", "Score (%)": tfidf_res["precision"] * 100},
            {"Model": "Naive Bayes (TF-IDF)", "Metric": "Recall", "Score (%)": tfidf_res["recall"] * 100},
            {"Model": "Naive Bayes (TF-IDF)", "Metric": "F1-Score", "Score (%)": tfidf_res["f1_score"] * 100},
        ])
        
        fig_metrics = px.bar(
            metrics_df,
            x="Metric",
            y="Score (%)",
            color="Model",
            barmode="group",
            text_auto=".1f",
            color_discrete_sequence=["#6366f1", "#38bdf8"]
        )
        fig_metrics.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f1f5f9"),
            yaxis=dict(range=[0, 105], gridcolor="#1e293b"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_metrics, use_container_width=True)
        
    with col_cm:
        st.subheader(f"🎯 Confusion Matrix: {active_results['name']}")
        cm = active_results["confusion_matrix"]
        labels = active_results["classes"]
        label_title_map = {"positif": "Positive", "negatif": "Negative"}
        
        fig_cm = px.imshow(
            cm,
            x=[f"Pred: {label_title_map.get(l, l.title())}" for l in labels],
            y=[f"True: {label_title_map.get(l, l.title())}" for l in labels],
            text_auto=True,
            color_continuous_scale="Purples",
            aspect="auto"
        )
        fig_cm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f1f5f9"),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        
    st.divider()

    # Feature Importance Top Words
    st.subheader(f"🔥 Most Influential Keywords ({active_results['name']})")
    top_feats = trainer.get_top_features(feature_type=active_model_key, top_n=10)
    
    if top_feats is not None:
        col_kw1, col_kw2 = st.columns(2)
        with col_kw1:
            fig_pos_words = px.bar(
                top_feats["positive"],
                x="Weight Score",
                y="Word/Phrase",
                orientation="h",
                title="Top 10 POSITIVE Indicator Words",
                color_discrete_sequence=["#10b981"]
            )
            fig_pos_words.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f1f5f9"), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_pos_words, use_container_width=True)
            
        with col_kw2:
            fig_neg_words = px.bar(
                top_feats["negative"],
                x="Weight Score",
                y="Word/Phrase",
                orientation="h",
                title="Top 10 NEGATIVE Indicator Words",
                color_discrete_sequence=["#f43f5e"]
            )
            fig_neg_words.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f1f5f9"), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_neg_words, use_container_width=True)
            
    st.divider()
    
    # Classification Report Per Class Table
    st.subheader(f"📋 Per-Class Classification Report ({active_results['name']})")
    rep_dict = active_results["classification_report"]
    
    rep_rows = []
    for cls_name in ["negatif", "positif"]:
        if cls_name in rep_dict:
            display_name = "Negative" if cls_name == "negatif" else "Positive"
            rep_rows.append({
                "Sentiment Class": display_name,
                "Precision": f"{rep_dict[cls_name]['precision']*100:.1f}%",
                "Recall": f"{rep_dict[cls_name]['recall']*100:.1f}%",
                "F1-Score": f"{rep_dict[cls_name]['f1-score']*100:.1f}%",
                "Support (Samples)": int(rep_dict[cls_name]['support'])
            })
            
    df_rep = pd.DataFrame(rep_rows)
    st.dataframe(df_rep, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Misclassification Inspector
    st.subheader("🔍 Misclassification Analysis (Error Inspection)")
    st.markdown("This inspection table isolates test instances where **the model prediction differed from the ground truth label**.")
    
    df_mis = active_results["misclassified_df"]
    if len(df_mis) == 0:
        st.success(f"🎉 Outstanding! Model {active_results['name']} correctly classified all test samples with 100% accuracy.")
    else:
        st.warning(f"Identified **{len(df_mis)}** misclassified samples out of **{active_results['total_test']}** total test samples.")
        st.dataframe(
            df_mis[["Index", "Category", "Original Review", "True Label", "Predicted Label", "Confidence"]],
            use_container_width=True,
            hide_index=True
        )

# ==============================================================================
# TAB 2: SINGLE REVIEW PREDICTION
# ==============================================================================
with tab2:
    st.header("✍️ Single Review Sentiment Prediction")
    st.markdown("Enter a product or mobile application review in the text box below to analyze sentiment in real-time.")
    
    # Sample Review Quick Selection Buttons
    st.markdown("**💡 Or select a preloaded sample review for quick testing:**")
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)

    sample_reviews = {
        "hp": "Kamera HP sangat jernih, hasil foto malam hari memuaskan. Seller ramah dan pengiriman sangat cepat!",
        "baju": "Kualitas barang sangat buruk, bahan tipis gampang sobek dan ukurannya tidak sesuai deskripsi. Kecewa!",
        "ewallet": "Aplikasi dompet digital sangat mudah digunakan, transfer antar bank gratis tanpa biaya admin dan topup cepat.",
        "lag": "Aplikasi sering crash dan keluar sendiri saat checkout pembayaran. Sangat mengganggu dan bikin jengkel."
    }

    if "review_input" not in st.session_state:
        st.session_state["review_input"] = ""
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None
    if "last_result_text" not in st.session_state:
        st.session_state["last_result_text"] = ""
    if "last_result_model" not in st.session_state:
        st.session_state["last_result_model"] = active_model_key

    def set_sample_and_analyze(text):
        st.session_state["review_input"] = text
        res_sample = engine.predict_single(text, model_type=active_model_key)
        st.session_state["last_result"] = res_sample
        st.session_state["last_result_text"] = text
        st.session_state["last_result_model"] = active_model_key

    col_s1.button("📱 Positive Product (Phone)", use_container_width=True, on_click=set_sample_and_analyze, args=(sample_reviews["hp"],))
    col_s2.button("👕 Negative Product (Shirt)", use_container_width=True, on_click=set_sample_and_analyze, args=(sample_reviews["baju"],))
    col_s3.button("💳 Positive App (E-Wallet)", use_container_width=True, on_click=set_sample_and_analyze, args=(sample_reviews["ewallet"],))
    col_s4.button("⚡ Negative App (Crash/Lag)", use_container_width=True, on_click=set_sample_and_analyze, args=(sample_reviews["lag"],))

    input_text = st.text_area(
        "Enter Review Text:",
        key="review_input",
        height=130,
        placeholder="Example: Fast shipping, high-quality product, and excellent seller communication. Highly recommended!"
    )

    col_btn1, col_btn2 = st.columns([3, 1])

    with col_btn1:
        if st.button("🚀 Analyze Sentiment Now", type="primary", use_container_width=True):
            input_val = st.session_state.get("review_input", "").strip()
            if not input_val:
                st.error("Please enter a review text first or select one of the samples above.")
                st.session_state["last_result"] = None
            else:
                res_calc = engine.predict_single(input_val, model_type=active_model_key)
                st.session_state["last_result"] = res_calc
                st.session_state["last_result_text"] = input_val
                st.session_state["last_result_model"] = active_model_key

    with col_btn2:
        def clear_single_review():
            st.session_state["review_input"] = ""
            st.session_state["last_result"] = None
            st.session_state["last_result_text"] = ""

        st.button("🗑️ Reset Text", use_container_width=True, on_click=clear_single_review)

    # Sync if model selection changed in sidebar
    if (
        st.session_state.get("last_result") is not None
        and st.session_state.get("last_result_text")
        and st.session_state.get("last_result_model") != active_model_key
    ):
        st.session_state["last_result"] = engine.predict_single(
            st.session_state["last_result_text"],
            model_type=active_model_key
        )
        st.session_state["last_result_model"] = active_model_key

    res = st.session_state.get("last_result")
    if res is not None:
        st.divider()
        col_res1, col_res2 = st.columns([1.1, 0.9])

        with col_res1:
            st.subheader("📌 Sentiment Classification Result")
            lbl = str(res["label"])
            conf = float(res["confidence"]) * 100

            if lbl == "positif":
                st.markdown(f'<div class="badge-positive">😊 SENTIMENT: POSITIVE ({conf:.1f}% Confidence)</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="badge-negative">😡 SENTIMENT: NEGATIVE ({conf:.1f}% Confidence)</div>', unsafe_allow_html=True)

            st.write("")
            st.markdown(f"**Model Used:** `{active_results['name']}`")
            st.markdown(f"**Preprocessed Text:** `{res['cleaned_text']}`")

            # Probability Table
            st.markdown("**Probability Distribution:**")
            prob_df = pd.DataFrame([
                {"Sentiment": "Positive 😊", "Probability": f"{res['probabilities']['positif']*100:.2f}%"},
                {"Sentiment": "Negative 😡", "Probability": f"{res['probabilities']['negatif']*100:.2f}%"}
            ])
            st.dataframe(prob_df, hide_index=True, use_container_width=True)

        with col_res2:
            st.subheader("📊 Probability Donut Chart")
            prob_donut = pd.DataFrame({
                "Sentiment": ["Positive", "Negative"],
                "Value": [res['probabilities']['positif'], res['probabilities']['negatif']]
            })
            fig_donut = px.pie(
                prob_donut,
                values="Value",
                names="Sentiment",
                hole=0.6,
                color="Sentiment",
                color_discrete_map={"Positive": "#10b981", "Negative": "#f43f5e"}
            )
            fig_donut.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f1f5f9"),
                legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        st.write("")
        st.subheader("🔤 Influential Sentiment Keywords (Feature Highlights)")
        words_contrib = res["highlighted_words"]
        if len(words_contrib) > 0:
            chips_html = ""
            for item in words_contrib:
                cls_name = "chip-pos" if item["polarity"] == "positif" else "chip-neg"
                icon = "👍" if item["polarity"] == "positif" else "👎"
                chips_html += f'<span class="chip {cls_name}">{icon} {item["word"]}</span>'
            st.markdown(chips_html, unsafe_allow_html=True)

            st.write("")
            words_df = pd.DataFrame([
                {
                    "Word/Phrase": item["word"],
                    "Positive Score": f"{item['positif_score']:.4f}",
                    "Negative Score": f"{item['negatif_score']:.4f}",
                    "Polarity Tendency": "Positive 👍" if item["polarity"] == "positif" else "Negative 👎"
                }
                for item in words_contrib
            ])
            st.dataframe(words_df, hide_index=True, use_container_width=True)
        else:
            st.info("No specific vocabulary keywords detected.")

# ==============================================================================
# TAB 3: BATCH PREDICTION & EXPORT
# ==============================================================================
with tab3:
    st.header("📁 Batch Prediction & Results Export")
    st.markdown("Upload your review dataset in CSV or Excel format, or run automated sentiment classification on preloaded reviews.")
    
    col_up1, col_up2 = st.columns([1.5, 1])
    
    with col_up1:
        uploaded_file = st.file_uploader(
            "Upload Review Dataset (CSV or Excel):",
            type=["csv", "xlsx"],
            help="Ensure your file contains at least one text column containing reviews."
        )
        
    with col_up2:
        st.markdown("**Or Use Built-in Dataset:**")
        use_default_batch = st.checkbox(f"Use Built-in Dataset ({len(trainer.df)} reviews)", value=True)
        
    df_to_predict = None
    text_col = "ulasan"
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_to_predict = pd.read_csv(uploaded_file)
            else:
                df_to_predict = pd.read_excel(uploaded_file)
                
            st.success(f"File uploaded successfully! Found {len(df_to_predict)} rows of data.")
            text_col = st.selectbox("Select Review Text Column:", options=df_to_predict.columns)
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")
    elif use_default_batch:
        df_to_predict = trainer.df.copy()
        text_col = "ulasan_bersih" if "ulasan_bersih" in df_to_predict.columns else "ulasan"
        
    if df_to_predict is not None:
        if st.button("⚡ Run Batch Classification Now", type="primary", use_container_width=True):
            with st.spinner("Processing batch sentiment predictions..."):
                batch_res = engine.predict_batch(df_to_predict, text_column=text_col, model_type=active_model_key)
                
                pos_count = (batch_res["prediksi_sentimen"] == "positif").sum()
                neg_count = (batch_res["prediksi_sentimen"] == "negatif").sum()
                total_cnt = len(batch_res)
                
                st.divider()
                st.subheader("📊 Batch Sentiment Analysis Summary")
                
                col_b1, col_b2, col_b3, col_b4 = st.columns(4)
                col_b1.metric("Total Reviews", f"{total_cnt}")
                col_b2.metric("Positive Sentiment 😊", f"{pos_count} ({pos_count/total_cnt*100:.1f}%)")
                col_b3.metric("Negative Sentiment 😡", f"{neg_count} ({neg_count/total_cnt*100:.1f}%)")
                col_b4.metric("Average Confidence", f"{batch_res['confidence_score'].mean()*100:.1f}%")
                
                st.write("")
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    fig_b_pie = px.pie(
                        names=["Positive", "Negative"],
                        values=[pos_count, neg_count],
                        title="Overall Sentiment Distribution",
                        color=["Positive", "Negative"],
                        color_discrete_map={"Positive": "#10b981", "Negative": "#f43f5e"},
                        hole=0.5
                    )
                    fig_b_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#f1f5f9"))
                    st.plotly_chart(fig_b_pie, use_container_width=True)
                    
                with col_chart2:
                    if "kategori" in batch_res.columns:
                        cat_df = batch_res.groupby(["kategori", "predicted_sentiment"]).size().reset_index(name="Count")
                        fig_cat = px.bar(
                            cat_df,
                            x="kategori",
                            y="Count",
                            color="predicted_sentiment",
                            title="Sentiment Distribution by Category",
                            barmode="group",
                            color_discrete_map={"Positive": "#10b981", "Negative": "#f43f5e"}
                        )
                        fig_cat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f1f5f9"), xaxis_title="Category", yaxis_title="Count")
                        st.plotly_chart(fig_cat, use_container_width=True)
                        
                st.divider()
                st.subheader("📋 Full Prediction Table")
                
                display_cols = [c for c in [text_col, "cleaned_text", "predicted_sentiment", "confidence_score", "probabilitas_positif", "probabilitas_negatif"] if c in batch_res.columns]
                st.dataframe(
                    batch_res[display_cols],
                    use_container_width=True,
                    hide_index=True
                )
                
                # CSV Export Button
                csv_buffer = batch_res.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Sentiment Analysis Results (CSV)",
                    data=csv_buffer,
                    file_name="sentiment_analysis_results.csv",
                    mime="text/csv",
                    type="primary",
                    use_container_width=True
                )

# ==============================================================================
# TAB 4: DATASET EXPLORER & THEORY
# ==============================================================================
with tab4:
    st.header("📖 Dataset Explorer & Methodology Theory")
    
    st.subheader("📚 Theoretical Foundations & Mathematical Formulations")
    
    st.markdown(r"""
    ### 1. Sentiment Analysis & Supervised Classification
    Sentiment analysis is the process of computationally extracting opinions, evaluations, and emotional polarity from text (such as product and application reviews). Supervised learning utilizes labeled datasets (*positive* and *negative*) to train statistical predictive models.

    ### 2. Multinomial Naive Bayes (MNB)
    Multinomial Naive Bayes is a specialized formulation of Naive Bayes optimized for discrete features such as term frequencies (*word counts*) or weighted *TF-IDF* scores across document vectors.

    Bayes' Theorem is expressed as:
    $$P(c|d) = \frac{P(c) \cdot P(d|c)}{P(d)}$$

    Applying the naive feature conditional independence assumption:
    $$P(c|d) \propto P(c) \prod_{i=1}^{n} P(t_i|c)$$

    Where:
    - $P(c)$ represents the class prior probability for $c \in \{\text{positive}, \text{negative}\}$.
    - $P(t_i|c)$ denotes the conditional probability of term $t_i$ occurring given class $c$, estimated using **Laplace Smoothing** ($\alpha = 1.0$):
    
    $$P(t_i|c) = \frac{N_{c, t_i} + \alpha}{\sum_{t'} N_{c, t'} + \alpha \cdot |V|}$$
    
    - $N_{c, t_i}$ is the frequency of word $t_i$ across documents of class $c$.
    - $|V|$ denotes the total vocabulary dimension (number of unique tokens).

    ### 3. Text Feature Extraction: CountVectorizer vs. TF-IDF
    - **Word Frequency (`CountVectorizer`)**: Constructs a document-term matrix of absolute term occurrences or n-grams.
    - **TF-IDF (`TfidfVectorizer`)**: Discounts ubiquitously frequent words and amplifies terms with high discriminative power for specific classes:
    
    $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{|D|}{1 + |\{d \in D : t \in d\}|}\right)$$
    """)
    
    st.divider()
    st.subheader("🔍 Raw Dataset Explorer")
    
    df_raw = trainer.df
    
    filter_cat = st.selectbox("Filter Category:", options=["All Categories"] + list(df_raw["kategori"].unique() if "kategori" in df_raw.columns else []))
    filter_sent = st.selectbox("Filter Sentiment:", options=["All Sentiments", "positif", "negatif"], format_func=lambda x: "All Sentiments" if x == "All Sentiments" else ("Positive" if x == "positif" else "Negative"))
    
    filtered_df = df_raw.copy()
    if filter_cat != "All Categories":
        filtered_df = filtered_df[filtered_df["kategori"] == filter_cat]
    if filter_sent != "All Sentiments":
        filtered_df = filtered_df[filtered_df["sentimen"] == filter_sent]
        
    st.write(f"Showing **{len(filtered_df)}** of **{len(df_raw)}** total reviews.")
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)
