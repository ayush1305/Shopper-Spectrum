import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

def train_and_export_pipeline(data_path, models_dir="models"):
    print(f"Starting data preprocessing and model training pipeline using: {data_path}...")
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Load Dataset
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Source dataset not found at {data_path}.")
        
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} transactions.")
    
    # 2. Case-Insensitive Column Normalization
    rename_dict = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower == 'invoiceno':
            rename_dict[col] = 'InvoiceNo'
        elif col_lower == 'stockcode':
            rename_dict[col] = 'StockCode'
        elif col_lower == 'description':
            rename_dict[col] = 'Description'
        elif col_lower == 'quantity':
            rename_dict[col] = 'Quantity'
        elif col_lower == 'invoicedate':
            rename_dict[col] = 'InvoiceDate'
        elif col_lower == 'unitprice':
            rename_dict[col] = 'UnitPrice'
        elif col_lower == 'customerid':
            rename_dict[col] = 'CustomerID'
        elif col_lower == 'country':
            rename_dict[col] = 'Country'
        elif col_lower in ['totalamount', 'totalspend']:
            rename_dict[col] = 'TotalSpend'
            
    df = df.rename(columns=rename_dict)
    print(f"Normalized columns: {list(df.columns)}")
    
    # 3. Data Preprocessing
    # Remove rows with missing CustomerID
    df = df.dropna(subset=['CustomerID'])
    df['CustomerID'] = df['CustomerID'].astype(int)
    
    # Exclude cancelled invoices (InvoiceNo starting with 'C')
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C', na=False)]
    
    # Remove negative or zero quantities and prices
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    
    print(f"Filtered dataset to {len(df)} clean transactions.")
    
    # Save the cleaned dataset to parquet format (highly compressed, perfect for GitHub)
    cleaned_parquet_path = os.path.join(models_dir, "cleaned_data.parquet")
    df.to_parquet(cleaned_parquet_path, index=False)
    print(f"Saved cleaned data to {cleaned_parquet_path} (compressed Parquet format).")
    
    # Convert InvoiceDate to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # 4. Feature Engineering (RFM Analysis)
    latest_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    # Calculate Total Spend if it wasn't pre-computed
    if 'TotalSpend' not in df.columns:
        df['TotalSpend'] = df['Quantity'] * df['UnitPrice']
    
    # Group by CustomerID
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (latest_date - x.max()).days, # Recency
        'InvoiceNo': 'nunique',                                # Frequency (number of unique transactions)
        'TotalSpend': 'sum'                                    # Monetary
    }).rename(columns={
        'InvoiceDate': 'Recency',
        'InvoiceNo': 'Frequency',
        'TotalSpend': 'Monetary'
    }).reset_index()
    
    print(f"Engineered RFM features for {len(rfm)} unique customers.")
    
    # 5. Standardize RFM values
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    
    # 6. Run Clustering (KMeans with 4 clusters)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Compute silhouette score and inertia for validation
    score = silhouette_score(rfm_scaled, rfm['Cluster'])
    inertia = kmeans.inertia_
    print(f"K-Means complete. Inertia: {inertia:.2f}, Silhouette Score: {score:.4f}")
    
    # 7. Map cluster labels dynamically based on RFM averages
    # We rank clusters using a composite value score: F * M / (R + 1)
    cluster_stats = rfm.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    }).reset_index()
    
    # Composite score: higher is better
    cluster_stats['ValueScore'] = (cluster_stats['Frequency'] * cluster_stats['Monetary']) / (cluster_stats['Recency'] + 1)
    # Sort from highest value to lowest value
    cluster_stats = cluster_stats.sort_values(by='ValueScore', ascending=False).reset_index(drop=True)
    
    # Define Segment Labels
    labels = ['High-Value', 'Regular', 'Occasional', 'At-Risk']
    cluster_mapping = {row['Cluster']: labels[i] for i, row in cluster_stats.iterrows()}
    
    # Map back to RFM dataframe
    rfm['Segment'] = rfm['Cluster'].map(cluster_mapping)
    print("Cluster Profiling Averages:")
    print(rfm.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean())
    
    # Save the RFM dataframe with segments to parquet
    rfm_parquet_path = os.path.join(models_dir, "rfm_data.parquet")
    rfm.to_parquet(rfm_parquet_path, index=False)
    
    # Save models
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    joblib.dump(kmeans, os.path.join(models_dir, "kmeans.pkl"))
    print("Saved Scaler and KMeans models.")
    
    # 8. Recommendation System Approach (Item-based Collaborative Filtering)
    print("Building Item-based Collaborative Filtering matrices...")
    
    # To save memory and time with larger datasets, let's limit product similarity computations 
    # to top selling items, or compute on complete catalog if catalog size is reasonable.
    # Group by description and get unique customer counts
    product_counts = df.groupby('Description')['CustomerID'].nunique()
    # If the product catalog is very large, keep the top 1000 items to avoid running out of memory.
    valid_products = product_counts.nlargest(1000).index
    df_filtered_products = df[df['Description'].isin(valid_products)]
    
    # Create Customer-Item pivot matrix (purchase count)
    customer_item_matrix = df_filtered_products.pivot_table(
        index='CustomerID', 
        columns='Description', 
        values='Quantity', 
        aggfunc='count', 
        fill_value=0
    )
    
    # Apply cosine similarity between products (columns)
    product_similarity = cosine_similarity(customer_item_matrix.T)
    product_similarity_df = pd.DataFrame(
        product_similarity, 
        index=customer_item_matrix.columns, 
        columns=customer_item_matrix.columns
    )
    
    # Save product similarity matrix
    similarity_path = os.path.join(models_dir, "similarity.pkl")
    joblib.dump(product_similarity_df, similarity_path)
    print(f"Saved product similarity matrix of shape {product_similarity_df.shape} to {similarity_path}")
    
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    if os.path.exists("data/online_retail_cleaned.csv"):
        train_and_export_pipeline("data/online_retail_cleaned.csv")
    elif os.path.exists("data/online_retail.csv"):
        train_and_export_pipeline("data/online_retail.csv")
    else:
        print("No raw dataset found in data/ directory. Please run generate_data.py first or copy your retail CSV.")
