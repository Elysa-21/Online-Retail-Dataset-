import pandas as pd
import matplotlib
matplotlib.use('Agg')  # untuk menampilkan grafik di layar
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_clean_with_stats(filepath):
    """
    Load raw data, perform initial exploration, clean data, and show post-cleaning stats.
    Returns cleaned DataFrame.
    """
    # ============================================
    # 1. LOAD RAW DATA (SEBELUM CLEANING)
    #============================================

    print("\n1. LOADING RAW DATA...")
    raw_df = pd.read_csv(filepath, encoding='ISO-8859-1')

    print("\n=== DATA SEBELUM CLEANING ===")
    print("Jumlah baris:", len(raw_df))
    print("Jumlah kolom:", len(raw_df.columns))
    print("\nMissing values:")
    print(raw_df.isnull().sum())

    print("\nJumlah duplikasi:", raw_df.duplicated().sum())

    print("\nStatistik deskriptif awal:")
    print(raw_df.describe())

    # Visualize before cleaning
    visualize_before_cleaning(raw_df)

    # ============================================
    # 2. CLEANING DATA
    #============================================

    print("\n2. CLEANING DATA MENGGUNAKAN FUNCTION load_and_clean_data()...")
    df = load_and_clean_data(filepath)
    df = df.reset_index(drop=True)

    print("\n=== DATA SETELAH CLEANING ===")
    print("Jumlah baris:", len(df))
    print("Jumlah kolom:", len(df.columns))
    print("\nMissing values setelah cleaning:")
    print(df.isnull().sum())

    print("\nJumlah duplikasi setelah cleaning:", df.duplicated().sum())

    print("\nStatistik deskriptif setelah cleaning:")
    print(df.describe())

    # Visualize after cleaning
    visualize_after_cleaning(df)

    print("\n🟢 Data Cleaning selesai.")
    print("==========================================\n")

    return df

def load_and_clean_data(filepath):
    """
    Load dataset and perform data cleaning.
    Returns cleaned DataFrame.
    """
    df = pd.read_csv(filepath)
    print("Data awal:", df.shape)

    # Hapus duplikasi
    df = df.drop_duplicates()

    # Hapus missing values
    df = df.dropna(subset=['Description', 'Customer ID'])

    # Ubah format tanggal
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Hapus nilai tidak wajar (Quantity negatif atau Price <= 0)
    df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]

    # Tambah kolom TotalPrice
    df["TotalPrice"] = df["Quantity"] * df["Price"]

    print("Data setelah cleaning:", df.shape)

    return df

def visualize_before_cleaning(df):
    """
    Visualize data before cleaning.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Data Distribution Before Cleaning', fontsize=16)

    # Histograms
    sns.histplot(df['Quantity'], ax=axes[0, 0], kde=True, color='blue')
    axes[0, 0].set_title('Quantity Histogram')

    sns.histplot(df['Price'], ax=axes[0, 1], kde=True, color='green')
    axes[0, 1].set_title('Price Histogram')

    if 'TotalPrice' in df.columns:
        sns.histplot(df['TotalPrice'], ax=axes[0, 2], kde=True, color='red')
    else:
        axes[0, 2].text(0.5, 0.5, 'TotalPrice not yet calculated', ha='center', va='center')
        axes[0, 2].set_title('TotalPrice Histogram')
    axes[0, 2].set_title('TotalPrice Histogram')

    # Box plots
    sns.boxplot(y=df['Quantity'], ax=axes[1, 0], color='blue')
    axes[1, 0].set_title('Quantity Box Plot')

    sns.boxplot(y=df['Price'], ax=axes[1, 1], color='green')
    axes[1, 1].set_title('Price Box Plot')

    if 'TotalPrice' in df.columns:
        sns.boxplot(y=df['TotalPrice'], ax=axes[1, 2], color='red')
    else:
        axes[1, 2].text(0.5, 0.5, 'TotalPrice not yet calculated', ha='center', va='center')
    axes[1, 2].set_title('TotalPrice Box Plot')

    plt.tight_layout()
    plt.savefig('output/cleaning/before_cleaning.png')
    plt.close()

def visualize_after_cleaning(df):
    """
    Visualize data after cleaning.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Data Distribution After Cleaning', fontsize=16)

    # Histograms
    sns.histplot(df['Quantity'], ax=axes[0, 0], kde=True, color='blue')
    axes[0, 0].set_title('Quantity Histogram')

    sns.histplot(df['Price'], ax=axes[0, 1], kde=True, color='green')
    axes[0, 1].set_title('Price Histogram')

    sns.histplot(df['TotalPrice'], ax=axes[0, 2], kde=True, color='red')
    axes[0, 2].set_title('TotalPrice Histogram')

    # Box plots
    sns.boxplot(y=df['Quantity'], ax=axes[1, 0], color='blue')
    axes[1, 0].set_title('Quantity Box Plot')

    sns.boxplot(y=df['Price'], ax=axes[1, 1], color='green')
    axes[1, 1].set_title('Price Box Plot')

    sns.boxplot(y=df['TotalPrice'], ax=axes[1, 2], color='red')
    axes[1, 2].set_title('TotalPrice Box Plot')

    plt.tight_layout()
    plt.savefig('output/cleaning/after_cleaning.png')
    plt.close()
