import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean

# =======================
# 1. Load dataset
# =======================
df = pd.read_csv("../data/nutrition.csv")

# =======================
# 2. Seleksi fitur utama
# =======================
features = ['calories', 'proteins', 'fat', 'carbohydrate']

# Tampilkan hasil tahap 4.2.1 Pemilihan Fitur Nutrisi ke terminal
print("=== Tahap 4.2.1: Pemilihan Fitur Nutrisi ===")
print(f"Fitur yang digunakan untuk sistem rekomendasi: {features}")
print("Fitur ini dipilih karena merupakan kandungan gizi makro yang relevan dalam pencarian makanan serupa berdasarkan nutrisi.\n")

# =======================
# 3. Pembersihan data
# =======================
df_clean = df.dropna(subset=features + ['name']).copy()
df_clean = df_clean.drop_duplicates(subset=features + ['name']).reset_index(drop=True)

# =======================
# 4. Pipeline KNN utama
# =======================
knn_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('knn', NearestNeighbors(n_neighbors=5))
])
knn_pipeline.fit(df_clean[features])

# =======================
# 5. Fungsi perhitungan BMR & TDEE
# =======================
def calculate_calories(gender, weight, height, age, activity_level):
    if gender.lower() == 'pria':
        bmr = 66.5 + (13.75 * weight) + (5.003 * height) - (6.75 * age)
    elif gender.lower() == 'wanita':
        bmr = 655.1 + (9.563 * weight) + (1.850 * height) - (4.676 * age)
    else:
        raise ValueError("Gender harus 'pria' atau 'wanita'.")

    activity_factors = {
        'sedikit': 1.2,
        'ringan': 1.375,
        'sedang': 1.55,
        'tinggi': 1.725,
        'sangat tinggi': 1.9
    }

    if activity_level not in activity_factors:
        raise ValueError("Level aktivitas tidak valid. Gunakan: sedikit, ringan, sedang, tinggi, sangat tinggi.")

    daily_calories = bmr * activity_factors[activity_level]
    deficit_min = daily_calories - 500
    deficit_max = daily_calories - 750

    return round(bmr, 2), round(daily_calories, 2), round(deficit_min, 2), round(deficit_max, 2)

# =======================
# 6. Menu interaktif CLI
# =======================
while True:
    print("\n=== Sistem Rekomendasi Makanan Berdasarkan Nutrisi ===")
    print("1. Cari berdasarkan nama makanan")
    print("2. Cari berdasarkan 1 jenis nutrisi")
    print("3. Rekomendasi makanan rendah kalori (≤ 150 kkal)")
    print("4. Hitung kebutuhan kalori harian")
    print("5. Evaluasi rekomendasi makanan (Top-N dan Jarak Euclidean)")
    print("Ketik 'exit' untuk keluar.")

    menu = input("Pilih menu (1/2/3/4/5/exit): ").strip().lower()

    if menu == 'exit':
        print("Terima kasih telah menggunakan sistem rekomendasi.")
        break

    elif menu == '1':
        user_input = input("Masukkan nama makanan: ").strip()
        if user_input not in df_clean['name'].values:
            print("\u26a0  Makanan tidak ditemukan dalam dataset.")
            continue

        selected_food = df_clean[df_clean['name'] == user_input]
        food_data = selected_food[features]

        print(f"\nNutrisi untuk '{user_input}':")
        print(selected_food[['name'] + features].to_string(index=False))

        transformed_input = knn_pipeline.named_steps['scaler'].transform(
            knn_pipeline.named_steps['imputer'].transform(food_data)
        )

        distances, indices = knn_pipeline.named_steps['knn'].kneighbors(transformed_input)

        recommendations = df_clean.iloc[indices[0][1:]]
        recommendations["distance"] = distances[0][1:]

        print(f"\n🍽  Rekomendasi makanan mirip dengan '{user_input}':")
        for _, row in recommendations.iterrows():
            print(f"- {row['name']}: {row['calories']} kkal, Protein: {row['proteins']}g, Lemak: {row['fat']}g, Karbo: {row['carbohydrate']}g")
            print(f"  Gambar: {row['image']}")

    elif menu == '2':
        print("Pilih jenis nutrisi:")
        for i, f in enumerate(features, 1):
            print(f"{i}. {f.title()}")

        try:
            nutr_idx = int(input("Masukkan nomor nutrisi (1-4): "))
            nutr_name = features[nutr_idx - 1]
        except:
            print("\u26a0  Input tidak valid.")
            continue

        try:
            value = float(input(f"Masukkan jumlah {nutr_name}: "))
        except:
            print("\u26a0  Harus berupa angka.")
            continue

        knn_nutrisi = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('knn', NearestNeighbors(n_neighbors=5))
        ])
        knn_nutrisi.fit(df_clean[[nutr_name]])

        input_transformed = knn_nutrisi.named_steps['scaler'].transform(
            knn_nutrisi.named_steps['imputer'].transform([[value]])
        )

        distances, indices = knn_nutrisi.named_steps['knn'].kneighbors(input_transformed)
        recommendations = df_clean.iloc[indices[0]]

        print(f"\n🍽  Rekomendasi makanan dengan {nutr_name} mendekati {value}:")
        for _, row in recommendations.iterrows():
            print(f"- {row['name']}: {row['calories']} kkal, Protein: {row['proteins']}g, Lemak: {row['fat']}g, Karbo: {row['carbohydrate']}g")
            print(f"  Gambar: {row['image']}")

    elif menu == '3':
        print("\n🔎 Mencari rekomendasi makanan rendah kalori (≤ 150 kkal)...")
        df_low_cal = df_clean[df_clean['calories'] <= 150].reset_index(drop=True)

        if df_low_cal.empty:
            print("\u26a0  Tidak ada makanan yang memenuhi kriteria rendah kalori.")
            continue

        low_cal_ref = df_low_cal[features].mean().values.reshape(1, -1)

        knn_lowcal_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('knn', NearestNeighbors(n_neighbors=5))
        ])
        knn_lowcal_pipeline.fit(df_low_cal[features])

        transformed = knn_lowcal_pipeline.named_steps['scaler'].transform(
            knn_lowcal_pipeline.named_steps['imputer'].transform(low_cal_ref)
        )

        distances, indices = knn_lowcal_pipeline.named_steps['knn'].kneighbors(transformed)
        recommendations = df_low_cal.iloc[indices[0]]

        print(f"\n🥗 Rekomendasi makanan rendah kalori mirip satu sama lain:")
        for _, row in recommendations.iterrows():
            print(f"- {row['name']}: {row['calories']} kkal, Protein: {row['proteins']}g, Lemak: {row['fat']}g, Karbo: {row['carbohydrate']}g")
            print(f"  Gambar: {row['image']}")

    elif menu == '4':
        try:
            gender = input("Jenis kelamin (pria/wanita): ").strip().lower()
            weight = float(input("Berat badan (kg): "))
            height = float(input("Tinggi badan (cm): "))
            age = int(input("Usia (tahun): "))
            print("Pilih level aktivitas: sedikit, ringan, sedang, tinggi, sangat tinggi")
            activity_level = input("Level aktivitas: ").strip().lower()

            bmr, total_cal, def_min, def_max = calculate_calories(
                gender, weight, height, age, activity_level
            )

            print("\n📊 Hasil Perhitungan Kalori:")
            print(f"BMR (Basal Metabolic Rate): {bmr} kalori")
            print(f"Kebutuhan kalori harian: {total_cal} kalori")
            print(f"Target defisit kalori:")
            print(f" - Defisit 500 kalori: {def_min} kalori")
            print(f" - Defisit 750 kalori: {def_max} kalori")

        except Exception as e:
            print(f"\u26a0  Terjadi kesalahan input: {e}")

    elif menu == '5':
        def evaluate_topn_similarity(input_name, topn=5):
            if input_name not in df_clean['name'].values:
                print(f"\n\u26a0\ufe0f Makanan '{input_name}' tidak ditemukan dalam dataset.")
                return

            input_row = df_clean[df_clean['name'] == input_name]
            input_features = input_row[features]

            transformed = knn_pipeline.named_steps['scaler'].transform(
                knn_pipeline.named_steps['imputer'].transform(input_features)
            )

            distances, indices = knn_pipeline.named_steps['knn'].kneighbors(transformed, n_neighbors=topn+1)
            neighbors = df_clean.iloc[indices[0][1:]]

            print(f"\n Evaluasi Top-{topn} Rekomendasi untuk '{input_name}':\n")
            print("Nutrisi makanan input:")
            print(input_row[['name'] + features].to_string(index=False))

            print(f"\nRekomendasi teratas:")
            for i, (_, row) in enumerate(neighbors.iterrows(), 1):
                print(f"\n{i}. {row['name']}")
                for f in features:
                    delta = abs(row[f] - input_row.iloc[0][f])
                    print(f"   {f.title()}: {row[f]} (selisih {delta:.2f})")

        def evaluate_euclidean_manual(input_name, topn=5):
            if input_name not in df_clean['name'].values:
                print(f"\n\u26a0\ufe0f Makanan '{input_name}' tidak ditemukan dalam dataset.")
                return

            input_row = df_clean[df_clean['name'] == input_name]
            input_vec = input_row[features].values[0]

            print(f"\n Evaluasi Jarak Euclidean Manual untuk '{input_name}':\n")
            distances = []

            for _, row in df_clean.iterrows():
                if row['name'] == input_name:
                    continue
                vec = [row[f] for f in features]
                dist = euclidean(input_vec, vec)
                distances.append((row['name'], dist, row['calories'], row['proteins'], row['fat'], row['carbohydrate']))

            distances.sort(key=lambda x: x[1])
            print(f"Top-{topn} makanan dengan jarak Euclidean terkecil:")

            for i in range(topn):
                name, dist, cal, prot, fat, carb = distances[i]
                print(f"{i+1}. {name} - Jarak: {dist:.4f}")
                print(f"   Kalori: {cal}, Protein: {prot}, Lemak: {fat}, Karbo: {carb}")

        input_name = input("Masukkan nama makanan untuk evaluasi: ").strip()
        evaluate_topn_similarity(input_name, topn=5)
        evaluate_euclidean_manual(input_name, topn=5)

    else:
        print("\u26a0  Menu tidak valid. Coba lagi.")