import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Struktur fungsi predict() 
def predict(admin, info, prod, bounce, exit_rate, page_val):
    # 1. Menyusun input menjadi format matriks array numerik 2D 
    data_mentah = [[admin, info, prod, bounce, exit_rate, page_val]]
    
    # 2. Menjalankan preprocessing yang konsisten (Scaling) sebelum masuk model
    data_siap = scaler.transform(data_mentah)
    
    # 3. Memanggil model untuk melakukan prediksi
    hasil = model.predict(data_siap)
    
    # 4. Mengembalikan hasil prediksi indeks tunggal (return hasil[0])
    return hasil[0]


# TAMPILAN / UI FRONTEND (Streamlit Dashboard)
st.set_page_config(page_title="Shopper Analytics Dashboard", layout="wide")

# Informasi Klaster untuk memperjelas Output Hasil ke Pengguna
kelompok_info = {
    0: {"nama": "Casual Visitor", "ket": "Sesi singkat, klik halaman minim, tingkat keluar tinggi."},
    1: {"nama": "Content Explorer", "ket": "Banyak membaca informasi umum toko atau halaman bantuan."},
    2: {"nama": "Potential Buyer", "ket": "Aktivitas katalog produk intens, berpeluang besar membeli."},
    3: {"nama": "High-Value Shopper", "ket": "Nilai halaman sangat tinggi, siap melakukan transaksi."}
}

st.sidebar.markdown("### 🛠️ Menu Program")
pilihan_menu = st.sidebar.radio("Pilih Halaman:", ["📌 Beranda Program", "🛍️ Online Shopper Segment Analyze"])

if pilihan_menu == "📌 Beranda Program":
    st.markdown("## 🛍️ Online Shopper Segment Analyze ##")
    st.write("Program ini adalah sebuah Sistem Pengambil Keputusan berbasis Aplikasi Web Lokalnya (Dashboard) yang berfungsi untuk mengelompokkan karakteristik perilaku pengunjung toko online (e-commerce).")
    with st.container(border=True):
        st.markdown("#### 📖 Ringkasan Singkat Sistem")
        st.write("program ini sukses menerapkan Software Deployment Workflow Universitas Bunda Mulia dengan memisahkan secara efisien antara proses hitung data secara offline (train_model.py) dan aplikasi dasbor interaktif secara online (app.py), di mana bagian antarmuka (Frontend/UI) Streamlit mengumpulkan parameter masukan dalam visualisasi grafik berwarna biru, sementara latar belakang sistem (Backend) memuat file biner (model.pkl dan scaler.pkl) untuk mengeksekusi rumus matematika Z-score serta fungsi predict() secara real-time guna menghasilkan rekomendasi taktis klasifikasi perilaku konsumen toko online yang cepat dan akurat.")

elif pilihan_menu == "🛍️ Online Shopper Segment Analyze":
    st.markdown("## 📊 Klasifikasi Online Shopper")
    st.write("Masukkan aktivitas pengunjung pada indikator di bawah:")
    st.write("---")
    
    kol_kiri, kol_kanan = st.columns([1, 2])
    
    # Tampilan Form Input dari Pengguna (Frontend) 
    with kol_kiri:
        with st.container(border=True):
            st.markdown("#### Parameter Perilaku Pengguna")
            admin = st.number_input("Aktivitas Administratif:", min_value=0, value=2)
            info = st.number_input("Aktivitas Informasi:", min_value=0, value=1)
            prod = st.number_input("Aktivitas Produk:", min_value=0, value=20)
            st.write("---")
            bounce = st.slider("Bounce Rate:", 0.0, 0.2, 0.01, step=0.001, format="%.3f")
            exit_rate = st.slider("Exit Rate:", 0.0, 0.2, 0.02, step=0.001, format="%.3f")
            st.write("---")
            page_val = st.number_input("Page Values:", min_value=0.0, value=15.0, step=0.5)
            
            # Tombol submit/kirim input dari pengguna
            tombol_hitung = st.button("Analisis Segmentasi", type="primary", use_container_width=True)
            
    # Tampilan Hasil Pengolahan (Backend Mengirim ke Frontend) 
    with kol_kanan:
        st.markdown("#### 📈 Hasil Analisis AI")
        
        if tombol_hitung:
            # Memanggil fungsi predict dengan argumen sesuai urutan fitur
            indeks_klaster = predict(admin, info, prod, bounce, exit_rate, page_val)
            hasil = kelompok_info[indeks_klaster]
            
            st.write("🔍 **Hasil Klasifikasi Kelompok:**")
            st.info(f"**Kelompok {indeks_klaster} : {hasil['nama']}**")
            st.markdown(f"**Pola Perilaku:** {hasil['ket']}")
            
            st.write("---")
            # Visualisasi Dashboard Tambahan Tema Warna Biru (#1f77b4) 
            st.write("📊 **Grafik Parameter Pengunjung:**")
            data_grafik = pd.DataFrame({
                'Kategori Aktivitas': ['Total Halaman Dibuka', 'Page Values (x10)', 'Bounce Rate (x100)'],
                'Nilai Skor': [admin + info + prod, page_val * 10, bounce * 100]
            })
            st.bar_chart(data=data_grafik, x='Kategori Aktivitas', y='Nilai Skor', color='#1f77b4')
        else:
            st.info("Sistem standby. Silakan isi parameter lalu tekan tombol Analisis Segmentasi.")