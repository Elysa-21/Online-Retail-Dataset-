# ============================================
# main.py — Data Cleaning + Data Exploration
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Tambahkan path project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import fungsi cleaning dan exploration
from src.cleaning import load_and_clean_with_stats
from src.exploration import explore_data
from src.feature import run_feature_engineering

# 1. LOAD RAW DATA, CLEANING, DAN STATISTIK
df = load_and_clean_with_stats("data/dataset.csv")

# 2. DATA EXPLORATION (EDA)
explore_data(df)

#3. Feature Engineering
df_fe, X, y = run_feature_engineering(df)
