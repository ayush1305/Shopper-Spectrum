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

# Clean, simple custom CSS ONLY for results and elements we want blue (no radio overrides!)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Title Card Styling */
    .title-card {
        background: #f8fafc;
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 12px;
        padding: 25px;
        margin-bottom: 25px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
    }
    
    .title-card h1 {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 5px;
    }
    
    .title-card p {
        color: #475569 !important;
        font-weight: 400;
    }
    
    /* 1. Style tabs selection indicator and text color to Blue */
    button[data-baseweb="tab"] {
        transition: all 0.2s ease !important;
    }
    
    button[data-baseweb="tab"]:hover {
        color: #3b82f6 !important;
    }
    
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #3b82f6 !important;
        border-bottom-color: #3b82f6 !important;
    }
    
    div[data-baseweb="tab-highlight"] {
        background-color: #3b82f6 !important;
    }
    
    /* 2. Style all primary buttons to Blue instead of default red/orange */
    button[data-testid="baseButton-primary"] {
        background-color: #3b82f6 !important;
        color: white !important;
        border: 1px solid #3b82f6 !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2) !important;
        transition: all 0.2s ease !important;
    }
    
    button[data-testid="baseButton-primary"]:hover {
        background-color: #1d4ed8 !important;
        border-color: #1d4ed8 !important;
        color: white !important;
        box-shadow: 0 4px 16px rgba(29, 78, 216, 0.4) !important;
    }
    
    button[data-testid="baseButton-primary"]:active {
        background-color: #1e3a8a !important;
        border-color: #1e3a8a !important;
        color: white !important;
    }
    
    /* 3. Style Streamlit Sliders to Blue (Thumb and Track highlight) */
    div[data-testid="stSlider"] div[role="slider"] {
        background-color: #3b82f6 !important;
        border-color: #3b82f6 !important;
    }
    
    div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div {
        background: #3b82f6 !important;
    }
    
    /* Style slider value text to Blue */
    div[data-testid="stSlider"] div[data-baseweb="slider"] + div {
        color: #3b82f6 !important;
    }
    
    /* 4. Style input focus borders to blue */
    div[data-testid="stNumberInput"] input:focus,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 1px #3b82f6 !important;
    }
    
    /* Recommendation Item Card */
    .rec-card {
        background: #ffffff;
        border-left: 5px solid #3b82f6;
        border-right: 1px solid rgba(15, 23, 42, 0.08);
        border-top: 1px solid rgba(15, 23, 42, 0.08);
        border-bottom: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 8px;
        padding: 15px 20px;
        margin-bottom: 12px;
        transition: transform 0.2s, background-color 0.2s;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
    }
    
    .rec-card:hover {
        transform: translateX(4px);
        background: #f8fafc;
    }
    
    .rec-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #0f172a;
    }
    
    .rec-meta {
        font-size: 0.85rem;
        color: #475569;
        margin-top: 5px;
        display: flex;
        justify-content: space-between;
    }
    
    .rec-score {
        font-weight: bold;
        color: #2563eb;
    }
    
    /* Customer Segment Result Card */
    .segment-card {
        border-radius: 12px;
        padding: 25px;
        margin-top: 20px;
        border: 1px solid rgba(15, 23, 42, 0.08);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    }
    
    .segment-high {
        background: rgba(59, 130, 246, 0.08);
        border-left: 6px solid #1e40af;
    }
    
    .segment-regular {
        background: rgba(96, 165, 250, 0.08);
        border-left: 6px solid #3b82f6;
    }
    
    .segment-occasional {
        background: rgba(147, 197, 253, 0.05);
        border-left: 6px solid #60a5fa;
    }
    
    .segment-at-risk {
        background: rgba(100, 116, 139, 0.08);
        border-left: 6px solid #64748b;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to check if models and preprocessed data exist (directly in the root directory)
def check_assets():
    required_files = [
        "scaler.pkl",
        "kmeans.pkl",
        "similarity.pkl",
        "cleaned_data.parquet",
        "rfm_data.parquet"
    ]
    return all(os.path.exists(f) for f in required_files)

# Sidebar UI
st.sidebar.markdown("<h2 style='text-align: center; color: #0f172a;'>🛒 Shopper Spectrum</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align: center; color:#475569; font-size:0.9rem;'>Customer Segmentation & Product Recommendations</p>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# Main Navigation (Uses native Streamlit styling configured via config.toml)
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
            # Train and output directly to current directory
            train_and_export_pipeline("data/online_retail.csv", output_dir=".")
        st.success("App bootstrapped successfully!")
        st.rerun()
else:
    st.sidebar.success("✅ Models & processed data loaded.")

# Load models and data if assets exist
@st.cache_resource
def load_model_assets():
    if not check_assets():
        return None
    scaler = joblib.load("scaler.pkl")
    kmeans = joblib.load("kmeans.pkl")
    similarity_df = joblib.load("similarity.pkl")
    cleaned_df = pd.read_parquet("cleaned_data.parquet")
    rfm_df = pd.read_parquet("rfm_data.parquet")
    
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
        <p style="margin:0;">Exploratory Data Analysis, transaction patterns, and RFM-based customer clusters</p>
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
                # Top Selling Products (Cohesive Blues)
                st.subheader("Top 10 Best-Selling Products")
                top_products = cleaned_df.groupby('Description')['Quantity'].sum().reset_index()
                top_products = top_products.sort_values(by='Quantity', ascending=False).head(10)
                fig_products = px.bar(
                    top_products,
                    y='Description',
                    x='Quantity',
                    orientation='h',
                    color='Quantity',
                    color_continuous_scale='Blues',
                    labels={'Quantity': 'Units Sold', 'Description': 'Product'},
                    template='plotly_white'
                )
                fig_products.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_products, use_container_width=True)
                
            with col2:
                # Country-wise Sales Volume (Cohesive Blues)
                st.subheader("Transaction Volume by Country")
                country_sales = cleaned_df.groupby('Country')['InvoiceNo'].nunique().reset_index()
                country_sales = country_sales.sort_values(by='InvoiceNo', ascending=False)
                fig_country = px.bar(
                    country_sales,
                    x='Country',
                    y='InvoiceNo',
                    color='InvoiceNo',
                    color_continuous_scale='Blues',
                    labels={'InvoiceNo': 'Number of Orders'},
                    template='plotly_white'
                )
                fig_country.update_layout(margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_country, use_container_width=True)
                
            # Sales over time (Blue line)
            st.subheader("Monthly Sales Trend")
            cleaned_df['YearMonth'] = cleaned_df['InvoiceDate'].dt.to_period('M').astype(str)
            sales_trend = cleaned_df.groupby('YearMonth').agg({'Quantity': 'sum', 'InvoiceNo': 'nunique'}).reset_index()
            fig_trend = px.line(
                sales_trend,
                x='YearMonth',
                y='Quantity',
                markers=True,
                labels={'Quantity': 'Total Items Sold', 'YearMonth': 'Period'},
                template='plotly_white'
            )
            fig_trend.update_traces(line_color='#3b82f6', line_width=3)
            fig_trend.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_trend, use_container_width=True)
            
        with tab2:
            st.subheader("Recency, Frequency, & Monetary Distributions")
            col_d1, col_d2, col_d3 = st.columns(3)
            
            with col_d1:
                # Recency (Primary Blue)
                fig_rec = px.histogram(rfm_df, x='Recency', nbins=30, color_discrete_sequence=['#3b82f6'], template='plotly_white', title="Recency (Days since last purchase)")
                st.plotly_chart(fig_rec, use_container_width=True)
            with col_d2:
                # Frequency (Secondary Blue)
                fig_freq = px.histogram(rfm_df, x='Frequency', nbins=30, color_discrete_sequence=['#60a5fa'], template='plotly_white', title="Frequency (Number of orders)")
                st.plotly_chart(fig_freq, use_container_width=True)
            with col_d3:
                # Monetary (Light Blue)
                fig_mon = px.histogram(rfm_df, x='Monetary', nbins=30, color_discrete_sequence=['#93c5fd'], template='plotly_white', title="Monetary (Total spend)")
                st.plotly_chart(fig_mon, use_container_width=True)
                
        with tab3:
            # 3D Plot of Clusters (Cohesive Blue-to-Gray scale)
            st.subheader("3D Interactive Visualization of Customer Clusters")
            fig_3d = px.scatter_3d(
                rfm_df,
                x='Recency',
                y='Frequency',
                z='Monetary',
                color='Segment',
                color_discrete_map={
                    'High-Value': '#1e40af',    # Deep Tech Blue
                    'Regular': '#3b82f6',       # Bright Blue
                    'Occasional': '#60a5fa',    # Sky Blue
                    'At-Risk': '#64748b'        # Slate Grey-Blue
                },
                hover_name='CustomerID',
                log_y=True,
                log_z=True,
                opacity=0.85,
                template='plotly_white'
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
        <p style="margin:0;">Find products that are commonly purchased together using item-to-item similarity</p>
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
        <p style="margin:0;">Input user metrics (Recency, Frequency, Monetary) to identify customer segment</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not assets:
        st.info("💡 Please click **Bootstrap App with Mock Data** in the sidebar or upload your data in the **Data & Model Control** tab to activate predictions!")
    else:
        scaler, kmeans, similarity_df, cleaned_df, rfm_df = assets
        
        col_in1, col_in2 = st.columns([1, 1])
        
        with col_in1:
            st.markdown("### 📋 Enter Customer Attributes")
            
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
            * **👑 High-Value:** Frequent, recent shoppers who generate substantial revenue. (Deep Blue)
            * **📈 Regular:** Steady customers who shop moderately and have average monetary value. (Active Blue)
            * **🌿 Occasional:** Low frequency, inactive customers who buy rarely and spend small amounts. (Sky Blue)
            * **🚨 At-Risk:** Previously active or high-value shoppers who have not purchased in a long time. (Muted Slate)
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
                    <h2 style='margin:0; color:#1e3a8a;'>👑 Segment: High-Value Customer</h2>
                    <p style='margin:10px 0; color:#1e293b;'>This customer displays high transaction frequency, large spend values, and bought from us recently.</p>
                    <strong style="color:#1e3a8a;">💡 Marketing Strategy:</strong>
                    <ul style="color:#334155;">
                        <li>Offer exclusive loyalty rewards & VIP benefits.</li>
                        <li>Give early-bird access to new products.</li>
                        <li>Personal shopper/high-touch communication services.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
            elif predicted_segment == 'Regular':
                st.markdown("""
                <div class="segment-card segment-regular">
                    <h2 style='margin:0; color:#1d4ed8;'>📈 Segment: Regular Customer</h2>
                    <p style='margin:10px 0; color:#1e293b;'>This customer behaves steadily, making occasional purchases with moderate spends.</p>
                    <strong style="color:#1d4ed8;">💡 Marketing Strategy:</strong>
                    <ul style="color:#334155;">
                        <li>Cross-sell related products or offer product bundles.</li>
                        <li>Engage with standard newsletters and updates.</li>
                        <li>Provide loyalty discounts to boost order frequencies.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
            elif predicted_segment == 'Occasional':
                st.markdown("""
                <div class="segment-card segment-occasional">
                    <h2 style='margin:0; color:#0284c7;'>🌿 Segment: Occasional Customer</h2>
                    <p style='margin:10px 0; color:#1e293b;'>This customer buys rarely, spends minor amounts, and hasn't visited recently.</p>
                    <strong style="color:#0284c7;">💡 Marketing Strategy:</strong>
                    <ul style="color:#334155;">
                        <li>Send flash-sale notifications and time-bound coupons.</li>
                        <li>Provide low-barrier discount codes (e.g. Free Shipping).</li>
                        <li>Promote best-selling and trending products.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
            else:  # At-Risk
                st.markdown("""
                <div class="segment-card segment-at-risk">
                    <h2 style='margin:0; color:#475569;'>🚨 Segment: At-Risk Customer</h2>
                    <p style='margin:10px 0; color:#1e293b;'>This customer has not purchased in a long time. They may be churning.</p>
                    <strong style="color:#475569;">💡 Marketing Strategy:</strong>
                    <ul style="color:#334155;">
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
        <p style="margin:0;">Upload your custom dataset, train models, and manage output files</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📥 Upload Your Custom Transaction Dataset")
    st.info("The CSV file must match the format containing: `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`.")
    
    uploaded_file = st.file_uploader("Upload online_retail.csv", type=['csv'])
    
    if uploaded_file is not None:
        if st.button("🚀 Process Uploaded Data & Retrain Models", type="primary"):
            # Save file to data folder
            os.makedirs("data", exist_ok=True)
            with open("data/online_retail_cleaned.csv", "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success("File uploaded successfully to `data/online_retail_cleaned.csv`!")
            
            # Retrain model and output to root
            with st.spinner("Preprocessing transaction data and retraining machine learning models..."):
                try:
                    from train_models import train_and_export_pipeline
                    train_and_export_pipeline("data/online_retail_cleaned.csv", output_dir=".")
                    st.success("🎉 Models successfully trained and outputs saved in compressed format at the project root!")
                    st.cache_resource.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Error during training pipeline: {str(e)}")
                    
    st.markdown("---")
    st.markdown("### 📂 Generated Model Assets & Compression Details")
    
    col_stat1, col_stat2 = st.columns(2)
    
    with col_stat1:
        st.markdown("#### Saved Files Size Check (Direct Root Location)")
        # List size of models files
        files_to_check = [
            ("Cleaned Data (Parquet)", "cleaned_data.parquet"),
            ("RFM Data with Clusters (Parquet)", "rfm_data.parquet"),
            ("Product Similarity Matrix (PKL)", "similarity.pkl"),
            ("Standard Scaler Model (PKL)", "scaler.pkl"),
            ("KMeans Clustering Model (PKL)", "kmeans.pkl")
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
        
        1. Keep your large raw CSV locally in your `data/` folder (add it to `.gitignore` so it doesn't push to GitHub).
        2. The pipeline converts the data into **Parquet format** (`cleaned_data.parquet` and `rfm_data.parquet`) and saves them directly in the root directory.
        3. This slashes file sizes by **up to 92%** (e.g. 42MB shrinks to ~3MB).
        4. You only commit the lightweight `.parquet` and `.pkl` files located in the root of your repository to GitHub!
        5. Drag and drop all files directly into your GitHub repository root. No nested folders required!
        """)
