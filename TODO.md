# TODO: Refactor Anomaly Detection Script

## Step 1: Create cleaning.py ✅
- Load dataset from data/dataset.csv
- Implement data cleaning: remove duplicates, drop missing values in Description and Customer ID, convert InvoiceDate to datetime, remove invalid Quantity and Price, add TotalPrice column
- Add visualizations: histograms and box plots for Quantity, Price, TotalPrice before and after cleaning using seaborn

## Step 2: Create features.py
- Select numeric features: Quantity, Price, TotalPrice
- Normalize the features using StandardScaler

## Step 3: Create modeling.py
- Implement Elbow Method for optimal k
- Perform K-Means clustering with k=3 (or based on elbow)
- Evaluate clusters with Silhouette Score
- Visualize clusters with scatter plot
- Identify anomalies as the smallest cluster

## Step 4: Update main.py
- Import functions from cleaning, features, modeling
- Orchestrate the workflow: call cleaning, then features, then modeling
- Print results and show plots

## Step 5: Test the refactored script
- Run python data/main.py to ensure it works and generates plots
