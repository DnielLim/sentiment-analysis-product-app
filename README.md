# 🛍️ Dashboard Analisis Sentimen Ulasan Produk & Aplikasi

Sistem analisis sentimen berbasis **Python** dan **Streamlit** menggunakan algoritma **Multinomial Naive Bayes** untuk mengklasifikasikan ulasan produk dan aplikasi ke dalam sentimen **Positif** dan **Negatif**.

---

## 🚀 Fitur Utama

- **📊 Evaluasi & Perbandingan Model**: Perbandingan performa Naive Bayes berbasis **Frekuensi Kata (CountVectorizer)** vs **TF-IDF (TfidfVectorizer)** secara *side-by-side* (Accuracy, Precision, Recall, F1-Score, Confusion Matrix).
- **🔍 Inspektur Contoh Salah Prediksi**: Mengidentifikasi ulasan data uji yang gagal diprediksi dengan benar oleh model untuk analisis kesalahan (*error analysis*).
- **✍️ Analisis Ulasan Tunggal**: Prediksi sentimen real-time dengan kartu indikator, *donut chart* probabilitas, dan kata kunci penentu sentimen (*feature highlight chips*).
- **📁 Analisis Massal & Ekspor (Batch Processing)**: Pengunggahan berkas CSV/Excel ulasan pengguna, analisis statistik distribusi sentimen, dan ekspor hasil ke CSV.
- **📖 Teori & Eksplorasi Dataset**: Dokumentasi rumus matematika Bayes, *Laplace Smoothing*, *TF-IDF* ($\KaTeX$), serta *Dataset Explorer* interaktif.

---

## 🛠️ Panduan Instalasi & Jalankan

1. **Clone Repositori**:
   ```bash
   git clone https://github.com/Clov768/sentiment-analysis-product-app.git
   cd sentiment-analysis-product-app
   ```

2. **Buat Virtual Environment & Install Dependensi**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Jalankan Aplikasi Streamlit**:
   ```bash
   streamlit run app.py
   ```

Buka peramban di `http://localhost:8501`.
