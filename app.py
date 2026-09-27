import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io

from src.model_trainer import SentimentModelTrainer
from src.sentiment_engine import SentimentEngine
from src.preprocessing import clean_text

# Page Config
st.set_page_config(
    page_title="Dashboard Analisis Sentimen Ulasan Produk & Aplikasi",
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
    
    /* Fix Streamlit Top White Bar */
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
        font-size: 1.4rem;
    }
    .sidebar-brand-text {
        font-weight: 700;
        font-size: 1.1rem;
        color: #ffffff;
    }
    .sidebar-brand-sub {
        font-size: 0.78rem;
        color: #94a3b8;
    }

    /* Metric Cards */
    .metric-card {
        background: #161e2e;
        border: 1px solid #232f48;
        padding: 20px 22px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
        position: relative;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #4f46e5;
    }
    .metric-value {
        font-size: 2.1rem;
        font-weight: 800;
        color: #38bdf8;
        letter-spacing: -0.02em;
        margin: 4px 0;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 4px;
    }

    /* Sentiment Badges */
    .badge-container {
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 16px 20px;
        border-radius: 16px;
        margin-bottom: 20px;
    }
    .badge-positive {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.25) 100%);
        border: 1.5px solid #10b981;
        color: #34d399;
        font-weight: 800;
        font-size: 1.3rem;
        padding: 12px 24px;
        border-radius: 14px;
        display: inline-flex;
        align-items: center;
        gap: 10px;
        box-shadow: 0 10px 25px -5px rgba(16, 185, 129, 0.3);
    }
    .badge-negative {
        background: linear-gradient(135deg, rgba(244, 63, 94, 0.15) 0%, rgba(225, 29, 72, 0.25) 100%);
        border: 1.5px solid #f43f5e;
        color: #fb7185;
        font-weight: 800;
        font-size: 1.3rem;
        padding: 12px 24px;
        border-radius: 14px;
        display: inline-flex;
        align-items: center;
        gap: 10px;
        box-shadow: 0 10px 25px -5px rgba(244, 63, 94, 0.3);
    }

    /* Word Chips */
    .chip {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
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

# Initialize Session State
@st.cache_resource
def load_and_train_models():
    trainer = SentimentModelTrainer(data_path="data/dataset_ulasan.csv")
    trainer.train_models()
    engine = SentimentEngine(trainer)
    return trainer, engine

trainer, engine = load_and_train_models()
results = trainer.results

# Sidebar Configuration
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-icon">🤖</div>
        <div>
            <div class="sidebar-brand-text">AI Sentiment Lab</div>
            <div class="sidebar-brand-sub">Produk & Aplikasi Review</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("⚙️ Pengaturan Algoritma")
    
    active_model_key = st.radio(
        "Pilih Ekstraktor Fitur Teks:",
        options=["tfidf", "count"],
        format_func=lambda x: "Multinomial Naive Bayes (TF-IDF)" if x == "tfidf" else "Multinomial Naive Bayes (Frekuensi Kata)",
        help="Bandingkan performa Naive Bayes menggunakan pembobotan TF-IDF vs Frekuensi Kata (CountVectorizer)."
    )
    
    selected_model_name = "TF-IDF" if active_model_key == "tfidf" else "Frekuensi Kata"
    active_results = results[active_model_key]
    
    st.divider()
    st.subheader("📊 Statistik Dataset Latih")
    st.markdown(f"""
    - 📦 **Total Dataset:** `{len(trainer.df)} ulasan`
    - 🏋️ **Data Latih (80%):** `{len(trainer.X_train)} ulasan`
    - 🧪 **Data Uji (20%):** `{len(trainer.X_test)} ulasan`
    - 🎯 **Split Method:** `Stratified Split (random_state=42)`
    """)
    
    st.write("")
    if st.button("🔄 Latih Ulang Model", use_container_width=True, type="secondary"):
        st.cache_resource.clear()
        st.rerun()

# Hero Header Banner
st.markdown("""
<div class="hero-card">
    <div class="hero-title">🛍️ Dashboard Analisis Sentimen Ulasan Produk & Aplikasi</div>
    <div class="hero-subtitle">
        Sistem klasifikasi teks terawasi menggunakan algoritma <b>Multinomial Naive Bayes</b> untuk mengelompokkan ulasan pengguna ke dalam kategori <b>Positif</b> atau <b>Negatif</b>. Mendukung evaluasi perbandingan fitur <b>Frekuensi Kata (CountVectorizer)</b> dan <b>TF-IDF</b> secara interaktif.
    </div>
</div>
""", unsafe_allow_html=True)

# Main Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Evaluasi & Perbandingan Model",
    "✍️ Prediksi Ulasan Tunggal",
    "📁 Prediksi Massal & Ekspor",
    "📖 Eksplorasi Dataset & Teori"
])

# ==============================================================================
# TAB 1: MODEL EVALUATION & COMPARISON
# ==============================================================================
with tab1:
    st.header("📊 Evaluasi & Perbandingan Performa Model")
    st.markdown("Perbandingan eksperimen **Multinomial Naive Bayes** menggunakan fitur **Frekuensi Kata** vs **TF-IDF** pada pembagian data uji yang **sama persis** (20% test split).")
    
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
            <div class="metric-label">Precision Weighted</div>
            <div class="metric-value">{tfidf_res['precision']*100:.1f}%</div>
            <div class="metric-sub">TF-IDF vs {count_res['precision']*100:.1f}% Count</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Recall Weighted</div>
            <div class="metric-value">{tfidf_res['recall']*100:.1f}%</div>
            <div class="metric-sub">TF-IDF vs {count_res['recall']*100:.1f}% Count</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">F1-Score Weighted</div>
            <div class="metric-value">{tfidf_res['f1_score']*100:.1f}%</div>
            <div class="metric-sub">TF-IDF vs {count_res['f1_score']*100:.1f}% Count</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.write("")
    
    # Comparison Chart & Confusion Matrix
    col_chart, col_cm = st.columns([1.1, 0.9])
    
    with col_chart:
        st.subheader("📈 Grafik Perbandingan Metrik Evaluasi")
        metrics_df = pd.DataFrame([
            {"Model": "Naive Bayes (Frekuensi Kata)", "Metrik": "Accuracy", "Nilai (%)": count_res["accuracy"] * 100},
            {"Model": "Naive Bayes (Frekuensi Kata)", "Metrik": "Precision", "Nilai (%)": count_res["precision"] * 100},
            {"Model": "Naive Bayes (Frekuensi Kata)", "Metrik": "Recall", "Nilai (%)": count_res["recall"] * 100},
            {"Model": "Naive Bayes (Frekuensi Kata)", "Metrik": "F1-Score", "Nilai (%)": count_res["f1_score"] * 100},
            
            {"Model": "Naive Bayes (TF-IDF)", "Metrik": "Accuracy", "Nilai (%)": tfidf_res["accuracy"] * 100},
            {"Model": "Naive Bayes (TF-IDF)", "Metrik": "Precision", "Nilai (%)": tfidf_res["precision"] * 100},
            {"Model": "Naive Bayes (TF-IDF)", "Metrik": "Recall", "Nilai (%)": tfidf_res["recall"] * 100},
            {"Model": "Naive Bayes (TF-IDF)", "Metrik": "F1-Score", "Nilai (%)": tfidf_res["f1_score"] * 100},
        ])
        
        fig_metrics = px.bar(
            metrics_df,
            x="Metrik",
            y="Nilai (%)",
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
        
        fig_cm = px.imshow(
            cm,
            x=[f"Prediksi {l.title()}" for l in labels],
            y=[f"Aktual {l.title()}" for l in labels],
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
    st.subheader(f"🔥 Kata Kunci Paling Berpengaruh ({active_results['name']})")
    top_feats = trainer.get_top_features(feature_type=active_model_key, top_n=10)
    
    if top_feats is not None:
        col_kw1, col_kw2 = st.columns(2)
        with col_kw1:
            fig_pos_words = px.bar(
                top_feats["positive"],
                x="Skor Bobot",
                y="Kata/Frasa",
                orientation="h",
                title="Top 10 Kata Indikator POSITIF",
                color_discrete_sequence=["#10b981"]
            )
            fig_pos_words.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f1f5f9"), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_pos_words, use_container_width=True)
            
        with col_kw2:
            fig_neg_words = px.bar(
                top_feats["negative"],
                x="Skor Bobot",
                y="Kata/Frasa",
                orientation="h",
                title="Top 10 Kata Indikator NEGATIF",
                color_discrete_sequence=["#f43f5e"]
            )
            fig_neg_words.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f1f5f9"), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_neg_words, use_container_width=True)
            
    st.divider()
    
    # Classification Report Per Class Table
    st.subheader(f"📋 Laporan Klasifikasi Per Kelas ({active_results['name']})")
    rep_dict = active_results["classification_report"]
    
    rep_rows = []
    for cls_name in ["negatif", "positif"]:
        if cls_name in rep_dict:
            rep_rows.append({
                "Kelas Sentimen": cls_name.title(),
                "Precision": f"{rep_dict[cls_name]['precision']*100:.1f}%",
                "Recall": f"{rep_dict[cls_name]['recall']*100:.1f}%",
                "F1-Score": f"{rep_dict[cls_name]['f1-score']*100:.1f}%",
                "Jumlah Sample (Support)": int(rep_dict[cls_name]['support'])
            })
            
    df_rep = pd.DataFrame(rep_rows)
    st.dataframe(df_rep, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Misclassification Inspector
    st.subheader("🔍 Pemeriksaan Contoh Salah Prediksi (Misclassification Analysis)")
    st.markdown("Fitur ini menampilkan contoh ulasan dari data uji di mana **prediksi model berbeda dengan label sebenarnya**.")
    
    df_mis = active_results["misclassified_df"]
    if len(df_mis) == 0:
        st.success(f"🎉 Luar biasa! Model {active_results['name']} mengklasifikasikan seluruh data uji dengan 100% tepat.")
    else:
        st.warning(f"Ditemukan **{len(df_mis)}** contoh salah prediksi dari total **{active_results['total_test']}** data uji.")
        st.dataframe(
            df_mis[["Index", "Kategori", "Ulasan Asli", "Label Sebenarnya", "Prediksi Model", "Kepercayaan"]],
            use_container_width=True,
            hide_index=True
        )

# ==============================================================================
# TAB 2: SINGLE REVIEW PREDICTION
# ==============================================================================
with tab2:
    st.header("✍️ Prediksi Sentimen Ulasan Tunggal")
    st.markdown("Masukkan ulasan produk atau aplikasi pada kolom di bawah ini untuk menganalisis sentimen secara real-time.")
    
    # Sample Review Quick Selection Buttons
    st.markdown("**💡 Atau pilih contoh ulasan sampel untuk uji cepat:**")
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    
    sample_text = ""
    if col_s1.button("📱 Produk Positif (HP)", use_container_width=True):
        sample_text = "Kamera HP sangat jernih, hasil foto malam hari memuaskan. Seller ramah dan pengiriman sangat cepat!"
    if col_s2.button("👕 Produk Negatif (Baju)", use_container_width=True):
        sample_text = "Kualitas barang sangat buruk, bahan tipis gampang sobek dan ukurannya tidak sesuai deskripsi. Kecewa!"
    if col_s3.button("💳 Aplikasi Positif (E-Wallet)", use_container_width=True):
        sample_text = "Aplikasi dompet digital sangat mudah digunakan, transfer antar bank gratis tanpa biaya admin dan topup cepat."
    if col_s4.button("⚡ Aplikasi Negatif (Lag)", use_container_width=True):
        sample_text = "Aplikasi sering crash dan keluar sendiri saat checkout pembayaran. Sangat mengganggu dan bikin jengkel."
        
    input_text = st.text_area(
        "Masukkan Teks Ulasan:",
        value=sample_text,
        height=130,
        placeholder="Contoh: Pengiriman barang sangat cepat, produk original dan berfungsi dengan baik. Recommended seller!"
    )
    
    if st.button("🚀 Analisis Sentimen Sekarang", type="primary", use_container_width=True):
        if not input_text.strip():
            st.error("Silakan masukkan teks ulasan terlebih dahulu.")
        else:
            res = engine.predict_single(input_text, model_type=active_model_key)
            
            st.divider()
            col_res1, col_res2 = st.columns([1.1, 0.9])
            
            with col_res1:
                st.subheader("📌 Hasil Klasifikasi Sentimen")
                lbl = str(res["label"])
                conf = float(res["confidence"]) * 100
                
                if lbl == "positif":
                    st.markdown(f'<div class="badge-positive">😊 SENTIMEN: POSITIF ({conf:.1f}% Confident)</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="badge-negative">😡 SENTIMEN: NEGATIF ({conf:.1f}% Confident)</div>', unsafe_allow_html=True)
                    
                st.write("")
                st.markdown(f"**Model Digunakan:** `{active_results['name']}`")
                st.markdown(f"**Teks Setelah Preprocessing:** `{res['cleaned_text']}`")
                
                # Probability Table
                st.markdown("**Distribusi Probabilitas:**")
                prob_df = pd.DataFrame([
                    {"Sentimen": "Positif 😊", "Probabilitas": f"{res['probabilities']['positif']*100:.2f}%"},
                    {"Sentimen": "Negatif 😡", "Probabilitas": f"{res['probabilities']['negatif']*100:.2f}%"}
                ])
                st.dataframe(prob_df, hide_index=True, use_container_width=True)
                
            with col_res2:
                st.subheader("📊 Donut Chart Probabilitas")
                prob_donut = pd.DataFrame({
                    "Sentimen": ["Positif", "Negatif"],
                    "Nilai": [res['probabilities']['positif'], res['probabilities']['negatif']]
                })
                fig_donut = px.pie(
                    prob_donut,
                    values="Nilai",
                    names="Sentimen",
                    hole=0.6,
                    color="Sentimen",
                    color_discrete_map={"Positif": "#10b981", "Negatif": "#f43f5e"}
                )
                fig_donut.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#f1f5f9"),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig_donut, use_container_width=True)
                
            st.write("")
            st.subheader("🔤 Kata Kunci Penentu Sentimen (Feature Highlights)")
            words_contrib = res["highlighted_words"]
            if len(words_contrib) > 0:
                chips_html = ""
                for item in words_contrib:
                    cls_name = "chip-pos" if item["polarity"] == "positif" else "chip-neg"
                    icon = "👍" if item["polarity"] == "positif" else "👎"
                    chips_html += f'<span class="chip {cls_name}">{icon} {item["word"]}</span>'
                st.markdown(chips_html, unsafe_allow_html=True)
                
                st.write("")
                words_df = pd.DataFrame(words_contrib)
                words_df.columns = ["Kata/Frasa", "Skor Positif", "Skor Negatif", "Kecenderungan"]
                st.dataframe(words_df, hide_index=True, use_container_width=True)
            else:
                st.info("Tidak ada kata kunci spesifik dalam kamus yang terdeteksi.")

# ==============================================================================
# TAB 3: BATCH PREDICTION & EXPORT
# ==============================================================================
with tab3:
    st.header("📁 Prediksi Massal (Batch) & Ekspor Hasil")
    st.markdown("Unggah berkas CSV/Excel ulasan Anda atau jalankan klasifikasi otomatis pada kumpulan ulasan.")
    
    col_up1, col_up2 = st.columns([1.5, 1])
    
    with col_up1:
        uploaded_file = st.file_uploader(
            "Unggah Berkas Ulasan (CSV atau Excel):",
            type=["csv", "xlsx"],
            help="Pastikan berkas memiliki setidaknya satu kolom yang berisi teks ulasan."
        )
        
    with col_up2:
        st.markdown("**Atau Gunakan Dataset Bawaan:**")
        use_default_batch = st.checkbox("Gunakan Dataset Bawaan (220 ulasan)", value=True)
        
    df_to_predict = None
    text_col = "ulasan"
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_to_predict = pd.read_csv(uploaded_file)
            else:
                df_to_predict = pd.read_excel(uploaded_file)
                
            st.success(f"Berkas berhasil diunggah! Ditemukan {len(df_to_predict)} baris data.")
            text_col = st.selectbox("Pilih Kolom Teks Ulasan:", options=df_to_predict.columns)
        except Exception as e:
            st.error(f"Gagal membaca berkas: {e}")
    elif use_default_batch:
        df_to_predict = trainer.df.copy()
        text_col = "ulasan"
        
    if df_to_predict is not None:
        if st.button("⚡ Jalankan Klasifikasi Massal Sekarang", type="primary", use_container_width=True):
            with st.spinner("Memproses prediksi sentimen massal..."):
                batch_res = engine.predict_batch(df_to_predict, text_column=text_col, model_type=active_model_key)
                
                pos_count = (batch_res["prediksi_sentimen"] == "positif").sum()
                neg_count = (batch_res["prediksi_sentimen"] == "negatif").sum()
                total_cnt = len(batch_res)
                
                st.divider()
                st.subheader("📊 Ringkasan Hasil Analisis Sentimen Massal")
                
                col_b1, col_b2, col_b3, col_b4 = st.columns(4)
                col_b1.metric("Total Ulasan", f"{total_cnt}")
                col_b2.metric("Sentimen Positif 😊", f"{pos_count} ({pos_count/total_cnt*100:.1f}%)")
                col_b3.metric("Sentimen Negatif 😡", f"{neg_count} ({neg_count/total_cnt*100:.1f}%)")
                col_b4.metric("Rata-rata Kepercayaan", f"{batch_res['skor_kepercayaan'].mean()*100:.1f}%")
                
                st.write("")
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    fig_b_pie = px.pie(
                        names=["Positif", "Negatif"],
                        values=[pos_count, neg_count],
                        title="Distribusi Sentimen Keseluruhan",
                        color=["Positif", "Negatif"],
                        color_discrete_map={"Positif": "#10b981", "Negatif": "#f43f5e"},
                        hole=0.5
                    )
                    fig_b_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#f1f5f9"))
                    st.plotly_chart(fig_b_pie, use_container_width=True)
                    
                with col_chart2:
                    if "kategori" in batch_res.columns:
                        cat_df = batch_res.groupby(["kategori", "prediksi_sentimen"]).size().reset_index(name="Jumlah")
                        fig_cat = px.bar(
                            cat_df,
                            x="kategori",
                            y="Jumlah",
                            color="prediksi_sentimen",
                            title="Distribusi Sentimen Berdasarkan Kategori",
                            barmode="group",
                            color_discrete_map={"positif": "#10b981", "negatif": "#f43f5e"}
                        )
                        fig_cat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f1f5f9"))
                        st.plotly_chart(fig_cat, use_container_width=True)
                        
                st.divider()
                st.subheader("📋 Tabel Prediksi Lengkap")
                
                st.dataframe(
                    batch_res[[text_col, "cleaned_text", "prediksi_sentimen", "skor_kepercayaan", "probabilitas_positif", "probabilitas_negatif"]],
                    use_container_width=True,
                    hide_index=True
                )
                
                # CSV Export Button
                csv_buffer = batch_res.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Unduh Hasil Analisis Sentimen (CSV)",
                    data=csv_buffer,
                    file_name="hasil_analisis_sentimen.csv",
                    mime="text/csv",
                    type="primary",
                    use_container_width=True
                )

# ==============================================================================
# TAB 4: DATASET EXPLORER & THEORY
# ==============================================================================
with tab4:
    st.header("📖 Eksplorasi Dataset & Teori Metodologi")
    
    st.subheader("📚 Dasar Teori & Formulasi Matematika")
    
    st.markdown(r"""
    ### 1. Analisis Sentimen & Klasifikasi Terawasi (Supervised Classification)
    Analisis sentimen adalah proses ekstraksi opini dan evaluasi terhadap teks (ulasan produk atau aplikasi) untuk menentukan polaritas emosi. Klasifikasi terawasi memanfaatkan dataset yang telah diberi label (*positif* dan *negatif*) untuk melatih model statistik.

    ### 2. Multinomial Naive Bayes (MNB)
    Multinomial Naive Bayes adalah varian Naive Bayes yang sangat efektif untuk representasi frekuensi kata (*word counts*) atau bobot *TF-IDF* dalam dokumen teks.

    Teorema Bayes menyatakan:
    $$P(c|d) = \frac{P(c) \cdot P(d|c)}{P(d)}$$

    Dengan asumsi independensi fitur (*naive assumption*):
    $$P(c|d) \propto P(c) \prod_{i=1}^{n} P(t_i|c)$$

    Di mana:
    - $P(c)$ adalah probabilitas prior dari kelas $c \in \{\text{positif}, \text{negatif}\}$.
    - $P(t_i|c)$ adalah probabilitas kemunculan term/kata $t_i$ dalam kelas $c$, dihitung menggunakan *Laplace Smoothing* ($\alpha = 1.0$):
    
    $$P(t_i|c) = \frac{N_{c, t_i} + \alpha}{\sum_{t'} N_{c, t'} + \alpha \cdot |V|}$$

    ### 3. Ekstraksi Fitur Teks: CountVectorizer vs TF-IDF
    - **Frekuensi Kata (CountVectorizer)**: Mengukur jumlah kemunculan absolut setiap kata/n-gram dalam dokumen.
    - **TF-IDF (Term Frequency-Inverse Document Frequency)**: Memberikan bobot tinggi pada kata yang sering muncul dalam dokumen tertentu tetapi jarang muncul di dokumen lain:
    
    $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{|D|}{1 + |\{d \in D : t \in d\}|}\right)$$
    """)
    
    st.divider()
    st.subheader("🔍 Eksplorasi Data Mentah (Dataset Explorer)")
    
    df_raw = trainer.df
    
    filter_cat = st.selectbox("Filter Kategori:", options=["Semua Kategori"] + list(df_raw["kategori"].unique() if "kategori" in df_raw.columns else []))
    filter_sent = st.selectbox("Filter Sentimen:", options=["Semua Sentimen", "positif", "negatif"])
    
    filtered_df = df_raw.copy()
    if filter_cat != "Semua Kategori":
        filtered_df = filtered_df[filtered_df["kategori"] == filter_cat]
    if filter_sent != "Semua Sentimen":
        filtered_df = filtered_df[filtered_df["sentimen"] == filter_sent]
        
    st.write(f"Menampilkan **{len(filtered_df)}** dari **{len(df_raw)}** total ulasan.")
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)
