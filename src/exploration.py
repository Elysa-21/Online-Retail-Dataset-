import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def explore_data(df):
    """
    Perform data exploration on the cleaned DataFrame.
    """
    print("3. MELAKUKAN DATA EXPLORATION...\n")

    # ------------------------------------------------------
    # 3A. Statistik Deskriptif
    # ------------------------------------------------------

    print("\n🔹 Statistik Deskriptif (Lengkap):")
    print(df.describe(include="all").transpose())

    print("\n🔹 Median:")
    print(df.median(numeric_only=True))

    print("\n🔹 Mode:")
    print(df.mode().head(1))

    # ------------------------------------------------------
    # 3B. Korelasi
    # ------------------------------------------------------

    print("\n🔹 Korelasi Numerik:")
    corr_matrix = df.corr(numeric_only=True)
    print(corr_matrix)

    # Simpan heatmap
    plt.figure(figsize=(8,6))
    sns.heatmap(corr_matrix, annot=True, cmap="Blues")
    plt.title("Heatmap Korelasi")
    plt.savefig("output/exploration/heatmap_correlation.png")
    plt.close()

    print("✔ Heatmap korelasi disimpan: data/heatmap_correlation.png")

    # ------------------------------------------------------
    # 3C. Visualisasi Eksploratif
    # ------------------------------------------------------

    # Histogram Quantity
    plt.figure(figsize=(7,5))
    plt.hist(df["Quantity"], bins=50)
    plt.title("Histogram Quantity")
    plt.xlabel("Quantity")
    plt.ylabel("Frequency")
    plt.savefig("output/exploration/hist_quantity.png")
    plt.close()

    # Histogram TotalPrice
    plt.figure(figsize=(7,5))
    plt.hist(df["TotalPrice"], bins=50)
    plt.title("Histogram TotalPrice")
    plt.xlabel("Total Price")
    plt.ylabel("Frequency")
    plt.savefig("output/exploration/hist_totalprice.png")
    plt.close()

    # Scatter Quantity vs Price
    plt.figure(figsize=(7,5))
    plt.scatter(df["Quantity"], df["Price"], alpha=0.3)
    plt.title("Scatter: Quantity vs Price")
    plt.xlabel("Quantity")
    plt.ylabel("Price")
    plt.savefig("output/exploration/scatter_quantity_price.png")
    plt.close()

    # Boxplot Quantity
    plt.figure(figsize=(7,5))
    sns.boxplot(x=df["Quantity"])
    plt.title("Boxplot Quantity")
    plt.savefig("output/exploration/boxplot_quantity.png")
    plt.close()

    print("✔ Semua grafik EDA tersimpan di folder data/")

    # ============================================
    # 4. INSIGHT AWAL
    #============================================

    print("\n4. INSIGHT AWAL DARI DATA:")

    print("- Total Transaksi:", len(df))
    print("- Customer Unik:", df["Customer ID"].nunique())
    print("- Produk paling populer:", df["Description"].mode()[0])
    print("- Rata-rata nilai transaksi:", df["TotalPrice"].mean())
    print("- Outlier Quantity (di atas 99th percentile):",
          (df["Quantity"] > df["Quantity"].quantile(0.99)).sum(), "transaksi")

    print("\nKorelasi terbesar terhadap TotalPrice:")
    print(df.corr(numeric_only=True)["TotalPrice"].sort_values(ascending=False).head())

    print("\n🟢 Data Exploration selesai.")
    print("==========================================\n")