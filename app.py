import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Shopper Spectrum - E-Commerce Analytics Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for glassmorphism and modern UI elements
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0e1117 0%, #161b22 100%);
    }
    
    /* Header card styling */
    .title-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 25px;
        margin-bottom: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }
    
    .title-card h1 {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ff8a00, #e52e71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    
    /* Custom cards for recommendations */
    .rec-card {
        background: rgba(255, 255, 255, 0.02);
        border-left: 5px solid #ff8a00;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 15px 20px;
        margin-bottom: 12px;
        transition: transform 0.2s, background-color 0.2s;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .rec-card:hover {
        transform: translateX(5px);
        background: rgba(255, 255, 255, 0.05);
    }
    
    .rec-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #ffffff;
    }
    
    .rec-meta {
        font-size: 0.85rem;
        color: #8b949e;
        margin-top: 5px;
        display: flex;
        justify-content: space-between;
    }
    
    .rec-score {
        font-weight: bold;
        color: #00f2fe;
    }
    
    /* Segment result cards */
    .segment-card {
        border-radius: 12px;
        padding: 25px;
        margin-top: 20px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 10px 20px rgba(0,0,0,0.25);
    }
    
    .segment-high {
        background: linear-gradient(135deg, rgba(255, 138, 0, 0.15) 0%, rgba(229, 46, 113, 0.15) 100%);
        border-left: 6px solid #ff8a00;
    }
    
    .segment-regular {
        background: linear-gradient(135deg, rgba(0, 242, 254, 0.15) 0%, rgba(79, 172, 254, 0.15) 100%);
        border-left: 6px solid #00f2fe;
    }
    
    .segment-occasional {
        background: linear-gradient(135deg, rgba(67, 233, 123, 0.15) 0%, rgba(56, 249, 215, 0.15) 100%);
        border-left: 6px solid #43e97b;
    }
    
    .segment-at-risk {
        background: linear-gradient(135deg, rgba(248, 80, 50, 0.15) 0%, rgba(231, 24, 24, 0.15) 100%);
        border-left: 6px solid #f85032;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to check if models and preprocessed data exist
def check_assets():
    required_files = [
        "models/scaler.pkl",
        "models/kmeans.pkl",
        "models/similarity.pkl",
        "models/cleaned_data.parquet",
        "models/rfm_data.parquet"
    ]
    return all(os.path.exists(f) for f in required_files)

# Sidebar UI
st.sidebar.markdown("<h2 style='text-align: center;'>🛒 Shopper Spectrum</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align: center; color:#8b949e; font-size:0.9rem;'>Customer Segmentation & Product Recommendations</p>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# Main Navigation
menu = st.sidebar.radio(
    "Navigation Menu",
    ["📊 Dashboard & Insights", "🛒 Product Recommendations", "👥 Customer Segmentation", "⚙️ Data & Model Control"]
)

# Retraining Status Sidebar Helper
assets_exist = check_assets()

if not assets_exist:
    st.sidebar.warning("⚠️ Data and models are not initialized.")
    if st.sidebar.button("⚙️ Bootstrap App with Mock Data"):
        with st.spinner("Generating mock dataset..."):
            from generate_data import generate_synthetic_data
            generate_synthetic_data("data/online_retail.csv")
        with st.spinner("Training models & processing data..."):
            from train_models import train_and_export_pipeline
            train_and_export_pipeline()
        st.success("App bootstrapped successfully!")
        st.rerun()
else:
    st.sidebar.success("✅ Models & processed data loaded.")

# Load models and data if assets exist
@st.cache_resource
def load_model_assets():
    if not check_assets():
        return None
    scaler = joblib.load("models/scaler.pkl")
    kmeans = joblib.load("models/kmeans.pkl")
    similarity_df = joblib.load("models/similarity.pkl")
    cleaned_df = pd.read_parquet("models/cleaned_data.parquet")
    rfm_df = pd.read_parquet("models/rfm_data.parquet")
    
    # Pre-parse dates to datetime
    cleaned_df['InvoiceDate'] = pd.to_datetime(cleaned_df['InvoiceDate'])
    
    return scaler, kmeans, similarity_df, cleaned_df, rfm_df

# Load the cached assets
assets = load_model_assets()

# ----------------- PAGE 1: DASHBOARD & INSIGHTS -----------------
if menu == "📊 Dashboard & Insights":
    st.markdown("""
    <div class="title-card">
        <h1>📊 Dashboard & E-Commerce Insights</h1>
        <p style="color:#8b949e; margin:0;">Exploratory Data Analysis, transaction patterns, and RFM-based customer clusters</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not assets:
        st.info("💡 Please click **Bootstrap App with Mock Data** in the sidebar or upload your data in the **Data & Model Control** tab to activate the dashboard!")
    else:
        scaler, kmeans, similarity_df, cleaned_df, rfm_df = assets
        
        # High Level KPIs
        total_revenue = (cleaned_df['Quantity'] * cleaned_df['UnitPrice']).sum()
        total_orders = cleaned_df['InvoiceNo'].nunique()
        total_customers = cleaned_df['CustomerID'].nunique()
        avg_order_value = total_revenue / total_orders
        
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        
        with kpi_col1:
            st.metric("Total Revenue", f"${total_revenue:,.2f}")
        with kpi_col2:
            st.metric("Unique Orders", f"{total_orders:,}")
        with kpi_col3:
            st.metric("Active Customers", f"{total_customers:,}")
        with kpi_col4:
            st.metric("Average Invoice Value", f"${avg_order_value:.2f}")
            
        st.markdown("---")
        
        # Visualizations Tabs
        tab1, tab2, tab3 = st.tabs(["🛒 Sales & Product Trends", "📈 RFM Distributions", "👥 Cluster Interpretation"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Top Selling Products
                st.subheader("Top 10 Best-Selling Products")
                top_products = cleaned_df.groupby('Description')['Quantity'].sum().reset_index()
                top_products = top_products.sort_values(by='Quantity', ascending=False).head(10)
                fig_products = px.bar(
                    top_products,
                    y='Description',
                    x='Quantity',
                    orientation='h',
                    color='Quantity',
                    color_continuous_scale='Oranges',
                    labels={'Quantity': 'Units Sold', 'Description': 'Product'},
                    template='plotly_dark'
                )
                fig_products.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_products, use_container_width=True)
                
            with col2:
                # Country-wise Sales Volume
                st.subheader("Transaction Volume by Country")
                country_sales = cleaned_df.groupby('Country')['InvoiceNo'].nunique().reset_index()
                country_sales = country_sales.sort_values(by='InvoiceNo', ascending=False)
                fig_country = px.bar(
                    country_sales,
                    x='Country',
                    y='InvoiceNo',
                    color='InvoiceNo',
                    color_continuous_scale='Tealgrn',
                    labels={'InvoiceNo': 'Number of Orders'},
                    template='plotly_dark'
                )
                fig_country.update_layout(margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_country, use_container_width=True)
                
            # Sales over time
            st.subheader("Monthly Sales Trend")
            cleaned_df['YearMonth'] = cleaned_df['InvoiceDate'].dt.to_period('M').astype(str)
            sales_trend = cleaned_df.groupby('YearMonth').agg({'Quantity': 'sum', 'InvoiceNo': 'nunique'}).reset_index()
            fig_trend = px.line(
                sales_trend,
                x='YearMonth',
                y='Quantity',
                markers=True,
                labels={'Quantity': 'Total Items Sold', 'YearMonth': 'Period'},
                template='plotly_dark'
            )
            fig_trend.update_traces(line_color='#e52e71', line_width=3)
            fig_trend.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_trend, use_container_width=True)
            
        with tab2:
            st.subheader("Recency, Frequency, & Monetary Distributions")
            col_d1, col_d2, col_d3 = st.columns(3)
            
            with col_d1:
                # Recency
                fig_rec = px.histogram(rfm_df, x='Recency', nbins=30, color_discrete_sequence=['#ff8a00'], template='plotly_dark', title="Recency (Days since last purchase)")
                st.plotly_chart(fig_rec, use_container_width=True)
            with col_d2:
                # Frequency
                fig_freq = px.histogram(rfm_df, x='Frequency', nbins=30, color_discrete_sequence=['#00f2fe'], template='plotly_dark', title="Frequency (Number of orders)")
                st.plotly_chart(fig_freq, use_container_width=True)
            with col_d3:
                # Monetary
                fig_mon = px.histogram(rfm_df, x='Monetary', nbins=30, color_discrete_sequence=['#e52e71'], template='plotly_dark', title="Monetary (Total spend)")
                st.plotly_chart(fig_mon, use_container_width=True)
                
        with tab3:
            # 3D Plot of Clusters
            st.subheader("3D Interactive Visualization of Customer Clusters")
            fig_3d = px.scatter_3d(
                rfm_df,
                x='Recency',
                y='Frequency',
                z='Monetary',
                color='Segment',
                color_discrete_map={
                    'High-Value': '#ff8a00',
                    'Regular': '#00f2fe',
                    'Occasional': '#43e97b',
                    'At-Risk': '#f85032'
                },
                hover_name='CustomerID',
                log_y=True,
                log_z=True,
                opacity=0.8,
                template='plotly_dark'
            )
            fig_3d.update_layout(
                scene=dict(
                    xaxis_title='Recency (Days)',
                    yaxis_title='Log Frequency (Orders)',
                    zaxis_title='Log Monetary ($)'
                ),
                margin=dict(l=0, r=0, b=0, t=0),
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            st.plotly_chart(fig_3d, use_container_width=True)
            
            # Cluster stats table
            st.subheader("Average Cluster Characteristics")
            cluster_summary = rfm_df.groupby('Segment').agg({
                'CustomerID': 'count',
                'Recency': 'mean',
                'Frequency': 'mean',
                'Monetary': 'mean'
            }).rename(columns={'CustomerID': 'Customer Count'}).reset_index()
            
            # Formatting floats
            cluster_summary['Recency'] = cluster_summary['Recency'].map(lambda x: f"{x:.1f} days")
            cluster_summary['Frequency'] = cluster_summary['Frequency'].map(lambda x: f"{x:.1f} orders")
            cluster_summary['Monetary'] = cluster_summary['Monetary'].map(lambda x: f"${x:,.2f}")
            cluster_summary['Customer Count'] = cluster_summary['Customer Count'].map(lambda x: f"{x:,}")
            
            st.dataframe(cluster_summary, use_container_width=True, hide_index=True)

# ----------------- PAGE 2: PRODUCT RECOMMENDATIONS -----------------
elif menu == "🛒 Product Recommendations":
    st.markdown("""
    <div class="title-card">
        <h1>🛒 Collaborative Filtering Recommendation System</h1>
        <p style="color:#8b949e; margin:0;">Find products that are commonly purchased together using item-to-item similarity</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not assets:
        st.info("💡 Please click **Bootstrap App with Mock Data** in the sidebar or upload your data in the **Data & Model Control** tab to activate recommendations!")
    else:
        scaler, kmeans, similarity_df, cleaned_df, rfm_df = assets
        
        # Get list of products
        product_list = sorted(list(similarity_df.index))
        
        # Input elements
        col_rec1, col_rec2 = st.columns([2, 1])
        
        with col_rec1:
            selected_product = st.selectbox(
                "Search or Select a Product Name:",
                options=product_list,
                placeholder="Type to search...",
                index=0
            )
            
        with col_rec2:
            num_recs = st.slider("Number of recommendations:", min_value=3, max_value=10, value=5)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("✨ Get Recommendations", type="primary"):
            if selected_product in similarity_df.index:
                # Find similarity scores
                sim_scores = similarity_df[selected_product].sort_values(ascending=False)
                
                # Exclude the selected product itself
                sim_scores = sim_scores.drop(selected_product, errors='ignore')
                
                top_recs = sim_scores.head(num_recs)
                
                # Display recommendations
                st.subheader(f"Recommendations for: **{selected_product}**")
                
                # Match stock codes
                stock_mapping = cleaned_df[['Description', 'StockCode']].drop_duplicates().set_index('Description')['StockCode'].to_dict()
                
                for idx, (prod_name, score) in enumerate(top_recs.items()):
                    stock_code = stock_mapping.get(prod_name, "N/A")
                    match_percentage = score * 100
                    
                    st.markdown(f"""
                    <div class="rec-card">
                        <div class="rec-title">{idx+1}. {prod_name}</div>
                        <div class="rec-meta">
                            <span>Stock Code: <code>{stock_code}</code></span>
                            <span class="rec-score">Match: {match_percentage:.1f}%</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("Product name not found in similarity index.")

# ----------------- PAGE 3: CUSTOMER SEGMENTATION -----------------
elif menu == "👥 Customer Segmentation":
    st.markdown("""
    <div class="title-card">
        <h1>👥 Customer Segment Predictor</h1>
        <p style="color:#8b949e; margin:0;">Input user metrics (Recency, Frequency, Monetary) to identify customer segment</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not assets:
        st.info("💡 Please click **Bootstrap App with Mock Data** in the sidebar or upload your data in the **Data & Model Control** tab to activate predictions!")
    else:
        scaler, kmeans, similarity_df, cleaned_df, rfm_df = assets
        
        col_in1, col_in2 = st.columns([1, 1])
        
        with col_in1:
            st.markdown("### 📋 Enter Customer Attributes")
            # Set defaults based on 25th, 50th, 75th percentiles of dataset
            r_min, r_max = int(rfm_df['Recency'].min()), int(rfm_df['Recency'].max())
            f_min, f_max = int(rfm_df['Frequency'].min()), int(rfm_df['Frequency'].max())
            m_min, m_max = float(rfm_df['Monetary'].min()), float(rfm_df['Monetary'].max())
            
            recency_in = st.number_input(
                "Recency (Days since last purchase)",
                min_value=0,
                max_value=1000,
                value=int(rfm_df['Recency'].median()),
                help="How many days ago did this customer make their last purchase?"
            )
            
            frequency_in = st.number_input(
                "Frequency (Total number of invoices)",
                min_value=1,
                max_value=1000,
                value=int(rfm_df['Frequency'].median()),
                help="How many unique transactions did this customer make?"
            )
            
            monetary_in = st.number_input(
                "Monetary Value ($ Total spent)",
                min_value=0.01,
                max_value=500000.0,
                value=float(rfm_df['Monetary'].median()),
                format="%.2f",
                help="What is the aggregate revenue generated by this customer?"
            )
            
            submit_prediction = st.button("🔮 Predict Customer Segment", type="primary")
            
        with col_in2:
            st.markdown("### 📖 Segment Key Definitions")
            st.markdown("""
            * **👑 High-Value:** Frequent, recent shoppers who generate substantial revenue. 
            * **📈 Regular:** Steady customers who shop moderately and have average monetary value.
            * **🌿 Occasional:** Low frequency, inactive customers who buy rarely and spend small amounts.
            * **🚨 At-Risk:** Previously active or high-value shoppers who have not purchased in a long time. Needs activation.
            """)
            
        if submit_prediction:
            # Scale RFM values
            input_df = pd.DataFrame([[recency_in, frequency_in, monetary_in]], columns=['Recency', 'Frequency', 'Monetary'])
            input_scaled = scaler.transform(input_df)
            
            # Predict Cluster ID
            cluster_id = kmeans.predict(input_scaled)[0]
            
            # Look up dynamic mapping from RFM stats saved during training
            cluster_stats = rfm_df.groupby('Cluster').agg({
                'Recency': 'mean',
                'Frequency': 'mean',
                'Monetary': 'mean'
            }).reset_index()
            cluster_stats['ValueScore'] = (cluster_stats['Frequency'] * cluster_stats['Monetary']) / (cluster_stats['Recency'] + 1)
            cluster_stats = cluster_stats.sort_values(by='ValueScore', ascending=False).reset_index(drop=True)
            
            labels = ['High-Value', 'Regular', 'Occasional', 'At-Risk']
            cluster_mapping = {row['Cluster']: labels[i] for i, row in cluster_stats.iterrows()}
            predicted_segment = cluster_mapping.get(cluster_id, "Unknown")
            
            # Display based on class
            st.markdown("### 🎯 Prediction Result:")
            
            if predicted_segment == 'High-Value':
                st.markdown("""
                <div class="segment-card segment-high">
                    <h2 style='margin:0;'>👑 Segment: High-Value Customer</h2>
                    <p style='margin:10px 0;'>This customer displays high transaction frequency, large spend values, and bought from us recently.</p>
                    <strong>💡 Marketing Strategy:</strong>
                    <ul>
                        <li>Offer exclusive loyalty rewards & VIP benefits.</li>
                        <li>Give early-bird access to new products.</li>
                        <li>Personal shopper/high-touch communication services.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
            elif predicted_segment == 'Regular':
                st.markdown("""
                <div class="segment-card segment-regular">
                    <h2 style='margin:0;'>📈 Segment: Regular Customer</h2>
                    <p style='margin:10px 0;'>This customer behaves steadily, making occasional purchases with moderate spends.</p>
                    <strong>💡 Marketing Strategy:</strong>
                    <ul>
                        <li>Cross-sell related products or offer product bundles.</li>
                        <li>Engage with standard newsletters and updates.</li>
                        <li>Provide loyalty discounts to boost order frequencies.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
            elif predicted_segment == 'Occasional':
                st.markdown("""
                <div class="segment-card segment-occasional">
                    <h2 style='margin:0;'>🌿 Segment: Occasional Customer</h2>
                    <p style='margin:10px 0;'>This customer buys rarely, spends minor amounts, and hasn't visited recently.</p>
                    <strong>💡 Marketing Strategy:</strong>
                    <ul>
                        <li>Send flash-sale notifications and time-bound coupons.</li>
                        <li>Provide low-barrier discount codes (e.g. Free Shipping).</li>
                        <li>Promote best-selling and trending products.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
            else:  # At-Risk
                st.markdown("""
                <div class="segment-card segment-at-risk">
                    <h2 style='margin:0;'>🚨 Segment: At-Risk Customer</h2>
                    <p style='margin:10px 0;'>This customer has not purchased in a long time. They may be churning.</p>
                    <strong>💡 Marketing Strategy:</strong>
                    <ul>
                        <li>Send direct "We Miss You" Win-Back email campaigns with heavy discounts.</li>
                        <li>Conduct surveys to understand their dissatisfaction.</li>
                        <li>Pitch personalized product updates based on their past history.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

# ----------------- PAGE 4: DATA & MODEL CONTROL -----------------
elif menu == "⚙️ Data & Model Control":
    st.markdown("""
    <div class="title-card">
        <h1>⚙️ Data & Model Management</h1>
        <p style="color:#8b949e; margin:0;">Upload your custom dataset, train models, and manage output files</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📥 Upload Your Custom Transaction Dataset")
    st.info("The CSV file must match the format containing: `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`.")
    
    uploaded_file = st.file_uploader("Upload online_retail.csv", type=['csv'])
    
    if uploaded_file is not None:
        if st.button("🚀 Process Uploaded Data & Retrain Models", type="primary"):
            # Save file to data folder
            os.makedirs("data", exist_ok=True)
            with open("data/online_retail.csv", "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success("File uploaded successfully to `data/online_retail.csv`!")
            
            # Retrain model
            with st.spinner("Preprocessing transaction data and retraining machine learning models..."):
                try:
                    from train_models import train_and_export_pipeline
                    train_and_export_pipeline()
                    st.success("🎉 Models successfully trained and outputs saved in compressed format!")
                    st.cache_resource.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Error during training pipeline: {str(e)}")
                    
    st.markdown("---")
    st.markdown("### 📂 Generated Model Assets & Compression Details")
    
    col_stat1, col_stat2 = st.columns(2)
    
    with col_stat1:
        st.markdown("#### Saved Files Size Check")
        # List size of models files
        files_to_check = [
            ("Cleaned Data (Parquet)", "models/cleaned_data.parquet"),
            ("RFM Data with Clusters (Parquet)", "models/rfm_data.parquet"),
            ("Product Similarity Matrix (PKL)", "models/similarity.pkl"),
            ("Standard Scaler Model (PKL)", "models/scaler.pkl"),
            ("KMeans Clustering Model (PKL)", "models/kmeans.pkl")
        ]
        
        file_stats = []
        for name, path in files_to_check:
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                file_stats.append({"File Name": name, "Path": path, "Size (MB)": f"{size_mb:.3f} MB", "Status": "Active"})
            else:
                file_stats.append({"File Name": name, "Path": path, "Size (MB)": "N/A", "Status": "Missing"})
                
        st.table(pd.DataFrame(file_stats))
        
    with col_stat2:
        st.markdown("#### 💡 Parquet & Compression Solution")
        st.success("""
        **No GitHub Limits Exceeded!**
        
        If your raw database is large (e.g. 24MB+):
        1. Keep the large raw CSV locally in your `data/` folder (add it to `.gitignore` so it doesn't push to GitHub).
        2. The pipeline converts the data into a **Parquet format** (`cleaned_data.parquet` and `rfm_data.parquet`).
        3. This slashes file sizes by **up to 90%** (24MB shrinks to ~2-3MB).
        4. You only commit the `models/` directory (containing the lightweight parquet files and PKL models) to GitHub!
        5. The Streamlit app loads these Parquet files instantly, ensuring extremely fast page loading and clean deployments without breaking GitHub upload rules.
        """)
