import pandas as pd
from sklearn.preprocessing import StandardScaler

def select_and_normalize_features(df):
    """
    Select numeric features and normalize them.
    Returns scaled features and scaler object.
    """
    # Pilih fitur numerik
    features = df[['Quantity', 'Price', 'TotalPrice']]

    # Normalisasi Data
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    return scaled_features, scaler
