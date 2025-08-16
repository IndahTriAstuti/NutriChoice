import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from itertools import product

# Konfigurasi halaman
st.set_page_config(
    page_title="NutriChoice - Rekomendasi Makanan",
    page_icon="🍽",
    layout="wide"
)

# Fungsi memuat dan membersihkan data
@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)  # ambil folder tempat app.py berada
    file_path = os.path.join(base_path, "nutrition.csv")
    df = pd.read_csv(file_path)
    return df
    required_cols = ['name', 'image', 'type', 'calories', 'proteins', 'fat', 'carbohydrate']
    df_clean = df.dropna(subset=required_cols).reset_index(drop=True)
    return df_clean, ['calories', 'proteins', 'fat', 'carbohydrate']

# Fungsi menampilkan makanan
def tampilkan_makanan(df_result, jumlah_kolom=2):
    container = st.container()
    baris = [df_result[i:i + jumlah_kolom] for i in range(0, len(df_result), jumlah_kolom)]
    for batch in baris:
        cols = container.columns(jumlah_kolom)
        for i, (_, row) in enumerate(batch.iterrows()):
            with cols[i]:
                st.markdown("---")
                st.markdown(f"#### 🍽 {row['name']}")
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(row['image'], width=140)
                with col2:
                    st.markdown(f"""
                        <div style='font-size:14px; line-height:1.6;'>
                            <strong>Kalori:</strong> {row['calories']:.1f} kkal<br>
                            <strong>Protein:</strong> {row['proteins']:.1f} g<br>
                            <strong>Lemak:</strong> {row['fat']:.1f} g<br>
                            <strong>Karbo:</strong> {row['carbohydrate']:.1f} g
                        </div>
                    """, unsafe_allow_html=True)

# Load data
df_clean, features = load_data()

# Header halaman
st.markdown("""
    <h1 style='text-align: center; padding: 10px 20px; margin: 30px auto 10px auto; color: #2E8B57;'>🍽 NutriChoice: Rekomendasi Makanan Untukmu</h1>
    <hr style='margin-top: 0; margin-bottom: 100px; border: none; border-top: 3px solid #2E8B57; width: 100%;'>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("""
    <div style="text-align: center;">
        <img src="https://cdn-icons-png.flaticon.com/512/2922/2922510.png" width="80">
        <p><strong>Selamat datang!</strong><br>Gunakan menu di bawah untuk mulai eksplorasi 🍴</p>
    </div>
""", unsafe_allow_html=True)

# Navigasi menu
menu = st.sidebar.radio("Navigasi:", [
    "🔍 Cari Berdasarkan Nama",
    "🥦 Cari Berdasarkan Nutrisi",
    "🔥 Hitung Kebutuhan Kalori"
])

# Pipeline KNN global
knn_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('knn', NearestNeighbors(n_neighbors=6))
])
knn_pipeline.fit(df_clean[features])

# Menu pencarian berdasarkan nama
if menu == "🔍 Cari Berdasarkan Nama":
    st.title("🔍 Cari Rekomendasi Berdasarkan Nama Makanan")
    input_nama = st.text_input("Masukkan kata kunci nama makanan (contoh: 'ikan')")
    if st.button("🔍 Cari Rekomendasi") and input_nama.strip():
        import re
        pattern = rf"\b{re.escape(input_nama.lower())}\b"
        hasil_cocok = df_clean[df_clean['name'].str.lower().str.contains(pattern, regex=True, na=False)].reset_index(drop=True)
        if hasil_cocok.empty:
            st.warning(f"❌ Tidak ditemukan makanan dengan kata '{input_nama}'.")
        else:
            st.markdown(f"### 🍽 Ditemukan {len(hasil_cocok)} makanan dengan nama *'{input_nama}'*:")
            tampilkan_makanan(hasil_cocok)

# Menu pencarian berdasarkan nutrisi
elif menu == "🥦 Cari Berdasarkan Nutrisi":
    st.title("🥦 Cari Rekomendasi Berdasarkan Kandungan Nutrisi")
    nutrisi_dict = {
        "Kalori": "calories",
        "Protein": "proteins",
        "Lemak": "fat",
        "Karbo": "carbohydrate",
        "Serat": "fiber",
        "Gula": "sugar",
        "Natrium": "sodium"
    }
    nutrisi_label = st.selectbox("Pilih Jenis Nutrisi:", list(nutrisi_dict.keys()))
    nutrisi = nutrisi_dict[nutrisi_label]
    input_value = st.number_input(f"Masukkan jumlah {nutrisi_label} (gram atau kkal):", min_value=0.0, format="%.1f")
    if st.button("Cari Makanan Serupa"):
        if nutrisi not in df_clean.columns:
            st.warning("Data nutrisi ini tidak tersedia dalam dataset.")
        else:
            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('knn', NearestNeighbors(n_neighbors=10))
            ])
            pipeline.fit(df_clean[[nutrisi]])
            distances, indices = pipeline.named_steps['knn'].kneighbors(
                pipeline.named_steps['scaler'].transform(
                    pipeline.named_steps['imputer'].transform([[input_value]])
                )
            )
            rekomendasi = df_clean.iloc[indices[0]]
            st.markdown("### 📟 Informasi Nutrisi:")
            st.info(f"Menampilkan makanan dengan nilai *{nutrisi_label}* mendekati *{input_value}*")
            tampilkan_makanan(rekomendasi)

# Menu kalkulasi kebutuhan kalori + kombinasi rekomendasi makanan
elif menu == "🔥 Hitung Kebutuhan Kalori":
    st.title("🔥 Hitung Kebutuhan Kalori Harian")
    col1, col2 = st.columns(2)
    with col1:
        berat = st.number_input("Masukkan berat badan Anda (kg):", min_value=90, step=1, format="%d")
        tinggi = st.number_input("Masukkan tinggi badan Anda (cm):", min_value=175, step=1, format="%d")
        usia = st.number_input("Masukkan usia Anda:", min_value=25, step=1, format="%d")
    with col2:
        jenis_kelamin = st.radio("Jenis Kelamin:", ["Laki-laki", "Perempuan"])
        aktivitas = st.selectbox("Tingkat Aktivitas Fisik:", ["Rendah", "Sedang", "Tinggi"])
        defisit_opsi = st.selectbox("Pilih Defisit Kalori:", ["Tanpa Defisit", "Defisit 500 kkal", "Defisit 750 kkal"])

    if st.button("Hitung Kalori Harian dan Tampilkan Rekomendasi"):
        if jenis_kelamin == "Laki-laki":
            bmr = 88.362 + (13.397 * berat) + (4.799 * tinggi) - (5.677 * usia)
        else:
            bmr = 447.593 + (9.247 * berat) + (3.098 * tinggi) - (4.330 * usia)
        
        # Faktor aktivitas berdasarkan jenis kelamin
        if jenis_kelamin == "Laki-laki":
            faktor_aktivitas = {"Rendah": 1.65, "Sedang": 1.76, "Tinggi": 2.10}[aktivitas]
        else:
            faktor_aktivitas = {"Rendah": 1.55, "Sedang": 1.70, "Tinggi": 2.00}[aktivitas]
        kebutuhan_kalori = max((bmr * faktor_aktivitas) - (500 if defisit_opsi == "Defisit 500 kkal" else 750 if defisit_opsi == "Defisit 750 kkal" else 0), 0)
        st.success(f"🌟 Kebutuhan kalori harian Anda: {round(kebutuhan_kalori)} kkal")

        target_sarapan = 0.35 * kebutuhan_kalori
        target_siang = 0.35 * kebutuhan_kalori
        target_malam = 0.30 * kebutuhan_kalori

        # Tampilkan target kalori untuk debugging
        # st.info(f"🎯 Target kalori: Sarapan {round(target_sarapan)} kkal, Siang {round(target_siang)} kkal, Malam {round(target_malam)} kkal")

        # Ambil lebih banyak sampel dan gunakan semua data jika tersedia
        karbo = df_clean[df_clean['type'].str.lower() == 'karbo']
        lauk = df_clean[df_clean['type'].str.lower() == 'lauk']
        sayur_masak = df_clean[df_clean['type'].str.lower() == 'sayuran masak']
        buah = df_clean[df_clean['type'].str.lower() == 'buah']
        camilan = df_clean[df_clean['type'].str.lower() == 'camilan']
        minuman = df_clean[df_clean['type'].str.lower() == 'minuman']

        # Jika data terlalu sedikit, gunakan semua data yang tersedia
        if len(karbo) > 10:
            karbo = karbo.sample(n=10, random_state=1)
        if len(lauk) > 10:
            lauk = lauk.sample(n=10, random_state=1)
        if len(sayur_masak) > 10:
            sayur_masak = sayur_masak.sample(n=10, random_state=1)
        if len(buah) > 10:
            buah = buah.sample(n=10, random_state=1)
        if len(camilan) > 10:
            camilan = camilan.sample(n=10, random_state=1)
        if len(minuman) > 10:
            minuman = minuman.sample(n=10, random_state=1)

        tambahan_pagi_list = pd.concat([buah, minuman])
        semua_kombinasi = []
        
        # Toleransi yang lebih fleksibel berdasarkan kebutuhan kalori
        toleransi_total = max(200, kebutuhan_kalori * 0.1)  # 10% dari kebutuhan kalori atau minimal 200
        toleransi_per_waktu = max(150, target_sarapan * 0.15)  # 15% dari target per waktu makan

        # st.info(f"📊 Toleransi: Total ±{round(toleransi_total)} kkal, Per waktu makan ±{round(toleransi_per_waktu)} kkal")

        with st.spinner("🔄 Mencari kombinasi terbaik..."):
            # Batasi jumlah iterasi untuk performa
            max_iterasi = 1000
            iterasi = 0
            
            for k, l1, tambahan_pagi in product(karbo.itertuples(), lauk.itertuples(), tambahan_pagi_list.itertuples()):
                if iterasi >= max_iterasi:
                    break
                    
                kal_sarapan = k.calories + l1.calories + tambahan_pagi.calories
                if abs(kal_sarapan - target_sarapan) > toleransi_per_waktu:
                    continue

                for l2, s, k2 in product(lauk.itertuples(), sayur_masak.itertuples(), karbo.itertuples()):
                    if iterasi >= max_iterasi:
                        break
                        
                    kal_siang = l2.calories + s.calories + k2.calories
                    if abs(kal_siang - target_siang) > toleransi_per_waktu:
                        continue

                    for b, c, m in product(buah.itertuples(), camilan.itertuples(), minuman.itertuples()):
                        iterasi += 1
                        if iterasi >= max_iterasi:
                            break
                            
                        kal_malam = b.calories + c.calories + m.calories
                        total = kal_sarapan + kal_siang + kal_malam
                        selisih = abs(total - kebutuhan_kalori)

                        if selisih <= toleransi_total:
                            semua_kombinasi.append((selisih, (k, l1, tambahan_pagi, l2, s, k2, b, c, m)))

        semua_kombinasi.sort(key=lambda x: x[0])
        rekomendasi = [combo for _, combo in semua_kombinasi[:3]]

        if not rekomendasi:
            st.warning("❌ Tidak ditemukan kombinasi makanan yang mendekati target kalori Anda.")
            st.info(f"Total kombinasi valid ditemukan: {len(semua_kombinasi)} kombinasi.")
            
            # Tampilkan informasi debugging
            st.markdown("### 🔍 Informasi Debugging:")
            st.write(f"- Kebutuhan kalori: {round(kebutuhan_kalori)} kkal")
            st.write(f"- Toleransi total: ±{round(toleransi_total)} kkal")
            st.write(f"- Toleransi per waktu makan: ±{round(toleransi_per_waktu)} kkal")
            st.write(f"- Jumlah makanan karbo: {len(karbo)}")
            st.write(f"- Jumlah makanan lauk: {len(lauk)}")
            st.write(f"- Jumlah sayuran: {len(sayur_masak)}")
            st.write(f"- Jumlah buah: {len(buah)}")
            st.write(f"- Jumlah camilan: {len(camilan)}")
            st.write(f"- Jumlah minuman: {len(minuman)}")
            
            # Tampilkan beberapa contoh makanan dari setiap kategori
            st.markdown("### 📋 Contoh Makanan yang Tersedia:")
            col1, col2 = st.columns(2)
            with col1:
                if not karbo.empty:
                    st.write("**Karbohidrat:**")
                    for _, row in karbo.head(3).iterrows():
                        st.write(f"- {row['name']}: {row['calories']} kkal")
                if not lauk.empty:
                    st.write("**Lauk:**")
                    for _, row in lauk.head(3).iterrows():
                        st.write(f"- {row['name']}: {row['calories']} kkal")
            with col2:
                if not sayur_masak.empty:
                    st.write("**Sayuran:**")
                    for _, row in sayur_masak.head(3).iterrows():
                        st.write(f"- {row['name']}: {row['calories']} kkal")
                if not buah.empty:
                    st.write("**Buah:**")
                    for _, row in buah.head(3).iterrows():
                        st.write(f"- {row['name']}: {row['calories']} kkal")
        else:
            for i, (k, l1, tambahan_pagi, l2, s, k2, b, c, m) in enumerate(rekomendasi, 1):
                st.markdown(f"### 🥗 Kombinasi #{i}")

                # ======== SARAPAN ========
                st.markdown("---")
                st.markdown("#### 🍽 Sarapan")
                tampilkan_makanan(pd.DataFrame([k._asdict(), l1._asdict(), tambahan_pagi._asdict()]))
                total_sarapan = k.calories + l1.calories + tambahan_pagi.calories
                st.info(f"🍳 Total Kalori Sarapan: **{round(total_sarapan)} kkal**")

                # ======== MAKAN SIANG ========
                st.markdown("---")
                st.markdown("#### 🍼 Makan Siang")
                tampilkan_makanan(pd.DataFrame([l2._asdict(), s._asdict(), k2._asdict()]))
                total_siang = l2.calories + s.calories + k2.calories
                st.info(f"🍱 Total Kalori Makan Siang: **{round(total_siang)} kkal**")

                # ======== MAKAN MALAM ========
                st.markdown("---")
                st.markdown("#### 🌚 Makan Malam")
                tampilkan_makanan(pd.DataFrame([b._asdict(), c._asdict(), m._asdict()]))
                total_malam = b.calories + c.calories + m.calories
                st.info(f"🌙 Total Kalori Makan Malam: **{round(total_malam)} kkal**")

                # ======== TOTAL HARIAN ========
                total = total_sarapan + total_siang + total_malam
                st.success(f"🔥 Total Kalori Harian: **{round(total)} kkal**")
                st.markdown("---")


# CSS tambahan
st.markdown("""
<style>
.nutri-card {
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 15px;
    padding: 15px;
    margin: 10px 0;
    border: 1px solid #ddd;
    border-radius: 10px;
    background-color: #f9f9f9;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
}
.nutri-image {
    border-radius: 10px;
}
.nutri-inline {
    font-size: 14px;
    color: #444;
    margin-top: 5px;
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 1rem;
}
img {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)
