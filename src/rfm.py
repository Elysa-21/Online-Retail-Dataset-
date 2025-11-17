import pandas as pd

def calculate_rfm(df):
    """
    Hitung fitur RFM per Customer.
    """

    # Pastikan tanggal tipe datetime
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Tanggal referensi = tanggal terakhir pada dataset
    reference_date = df["InvoiceDate"].max()

    # Hitung RFM
    rfm = df.groupby("Customer ID").agg({
        "InvoiceDate": lambda x: (reference_date - x.max()).days,   # Recency
        "Invoice": "nunique",                                       # Frequency
        "TotalPrice": "sum"                                         # Monetary
    })

    # Rename kolom
    rfm.columns = ["Recency", "Frequency", "Monetary"]

    # Reset index
    rfm = rfm.reset_index()

    return rfm
