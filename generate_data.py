import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_synthetic_data(output_path, num_records=12000):
    print("Generating synthetic E-commerce dataset...")
    np.random.seed(42)
    
    # Products catalog (Description, StockCode, BasePrice)
    products = [
        ("WHITE HANGING HEART T-LIGHT HOLDER", "85123A", 2.55),
        ("REGENCY CAKESTAND 3 TIER", "22423", 12.75),
        ("JUMBO BAG RED RETROSPOT", "85099B", 1.95),
        ("ASSORTED COLOUR BIRD ORNAMENT", "84879", 1.69),
        ("PARTY BUNTING", "47566", 4.95),
        ("LUNCH BAG RED RETROSPOT", "20725", 1.65),
        ("SET OF 3 CAKE TINS PANTRY DESIGN", "22960", 7.95),
        ("PAPER CHAIN KIT 50'S CHRISTMAS", "22086", 2.95),
        ("NATURAL SLATE HEART CHALKBOARD", "22907", 2.95),
        ("HEART OF WICKER LARGE", "22411", 2.95),
        ("JAM MAKING SET WITH JARS", "22961", 4.25),
        ("ALARM CLOCK BAKELITE RED", "21931", 3.75),
        ("ALARM CLOCK BAKELITE GREEN", "21929", 3.75),
        ("SET/6 RED SPOTTY PAPER CUPS", "21790", 0.85),
        ("SET/6 RED SPOTTY PAPER PLATES", "21791", 0.85),
        ("WOODEN PICTURE FRAME WHITE WASH", "22457", 2.95),
        ("BAKING SET 9 PIECE RETROSPOT", "22910", 4.95),
        ("RETROSPOT TEA SET CERAMIC 11 PC", "22969", 8.50),
        ("VICTORIAN GLASS HANGING T-LIGHT", "21754", 1.25),
        ("GARDENERS KNEELING PAD KEEP CALM", "22355", 1.65)
    ]
    
    # Countries
    countries = ["United Kingdom", "Germany", "France", "EIRE", "Spain", "Netherlands", "Belgium", "Switzerland"]
    country_weights = [0.85, 0.04, 0.03, 0.02, 0.02, 0.01, 0.01, 0.02]
    
    # Define Customer Archetypes to ensure realistic clustering
    # Archetype: (Recency Range Days, Frequency Range Invoices, Monetary Spend per Item Multiplier, CustomerID List)
    customer_ids = np.arange(12000, 12600)
    
    archetype_allocations = {}
    # 1. High-Value: IDs 12000 - 12080 (Recent, Frequent, Big Spenders)
    # 2. Regular: IDs 12081 - 12280 (Steady, Medium activity)
    # 3. Occasional: IDs 12281 - 12480 (Rare, Occasional)
    # 4. At-Risk: IDs 12481 - 12599 (Inactive but previously active)
    
    for cid in customer_ids:
        if cid < 12080:
            archetype_allocations[cid] = 'High-Value'
        elif cid < 12280:
            archetype_allocations[cid] = 'Regular'
        elif cid < 12480:
            archetype_allocations[cid] = 'Occasional'
        else:
            archetype_allocations[cid] = 'At-Risk'
            
    # Time settings
    end_date = datetime(2023, 12, 31)
    
    data = []
    
    # Store invoice counters per customer to generate consistent dates
    invoice_dates_memo = {}
    
    for i in range(num_records):
        # Determine Customer ID (allow 10% missing CustomerID to simulate dirty data)
        if np.random.rand() < 0.10:
            cust_id = np.nan
            country = "United Kingdom"
            archetype = 'Regular'
        else:
            cust_id = int(np.random.choice(customer_ids))
            country = np.random.choice(countries, p=country_weights)
            archetype = archetype_allocations[cust_id]
            
        # Invoice details based on archetype
        if cust_id not in invoice_dates_memo or np.random.rand() < 0.2:
            # Generate new Invoice Date based on archetype
            if archetype == 'High-Value':
                # Purchases in the last 30 days
                days_ago = np.random.randint(0, 30)
            elif archetype == 'Regular':
                # Purchases in the last 120 days
                days_ago = np.random.randint(0, 120)
            elif archetype == 'Occasional':
                # Purchases anytime in the 2 years
                days_ago = np.random.randint(0, 730)
            else: # At-Risk
                # Haven't purchased in the last 200 - 700 days
                days_ago = np.random.randint(200, 700)
                
            inv_date = end_date - timedelta(days=days_ago, hours=np.random.randint(8, 18), minutes=np.random.randint(0, 59))
            invoice_num = str(np.random.randint(536365, 581587))
            invoice_dates_memo[cust_id] = (invoice_num, inv_date)
        else:
            invoice_num, inv_date = invoice_dates_memo[cust_id]
            
        # Let's handle cancellations (5% of invoices are cancellations, start with 'C')
        is_cancelled = False
        if not np.isnan(cust_id) and np.random.rand() < 0.05:
            is_cancelled = True
            invoice_num = 'C' + invoice_num
            
        # Pick product. Create correlations (Item-based collaborative filtering triggers)
        # We select products based on some probability profiles
        product_prob = np.ones(len(products)) / len(products)
        
        # If we have a customer ID, make their purchases more correlated
        if not np.isnan(cust_id):
            if cust_id % 3 == 0:
                # Group A: T-Light & Wicker
                product_prob[0] = 0.4  # White Hanging Heart T-Light Holder
                product_prob[9] = 0.3  # Heart of Wicker Large
                product_prob[18] = 0.2 # Victorian Glass T-Light
            elif cust_id % 3 == 1:
                # Group B: Bag Red & Lunch Bag Red
                product_prob[2] = 0.4  # Jumbo Bag Red Retrospot
                product_prob[5] = 0.3  # Lunch Bag Red Retrospot
                product_prob[11] = 0.15 # Alarm Clock Red
                product_prob[12] = 0.15 # Alarm Clock Green
            else:
                # Group C: Regency Cakestand & Cake Tins
                product_prob[1] = 0.4  # Regency Cakestand 3 Tier
                product_prob[6] = 0.3  # Set of 3 Cake Tins
                product_prob[16] = 0.2 # Baking Set
                product_prob[17] = 0.1 # Retrospot Tea Set
                
        # Normalize probabilities
        product_prob = product_prob / np.sum(product_prob)
        prod_idx = np.random.choice(len(products), p=product_prob)
        prod_desc, stock_code, base_price = products[prod_idx]
        
        # Determine Quantity
        if is_cancelled:
            qty = -int(np.random.randint(1, 10))
        else:
            if archetype == 'High-Value':
                qty = int(np.random.randint(5, 50))
            elif archetype == 'Regular':
                qty = int(np.random.randint(2, 15))
            else:
                qty = int(np.random.randint(1, 5))
                
        # Handle dirty data: negative quantities (without C), zero price, negative price
        # 0.5% records with negative quantity without C
        # 0.5% records with 0 price
        rand_val = np.random.rand()
        if rand_val < 0.005:
            qty = -int(np.random.randint(1, 5))
        elif rand_val < 0.01:
            base_price = 0.0
        elif rand_val < 0.012:
            base_price = -1.50
            
        data.append({
            "InvoiceNo": invoice_num,
            "StockCode": stock_code,
            "Description": prod_desc,
            "Quantity": qty,
            "InvoiceDate": inv_date.strftime("%Y-%m-%d %H:%M"),
            "UnitPrice": base_price,
            "CustomerID": cust_id,
            "Country": country
        })
        
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully created at {output_path} with {len(df)} rows.")

if __name__ == "__main__":
    generate_synthetic_data("data/online_retail.csv")
