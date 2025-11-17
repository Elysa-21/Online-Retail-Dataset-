import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # untuk menampilkan grafik di layar
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_clean_data(filepath):
    """
    Load dataset and perform data cleaning.
    Returns cleaned DataFrame.
    """
    df = pd.read_csv(filepath)
    print("Data awal:", df.shape)

    # Visualize before cleaning
    visualize_before_cleaning(df)

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

    # Visualize after cleaning
    visualize_after_cleaning(df)

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
    plt.savefig('data/before_cleaning.png')
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
    plt.savefig('data/after_cleaning.png')
    plt.close()
