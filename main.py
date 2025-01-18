import os
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from itertools import combinations
from collections import defaultdict
import asyncpg
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

DB_URL = os.environ.get("DB_URL")

if not DB_URL:
    raise ValueError("⚠️ Database URL is missing. Set DB_URL in environment variables.")

# Define category mapping
category_mapping = {
    "Batter": ["Idli/Dosa Batter", "Ragi Batter"],
    "Idli": ["Steamed Idli", "Ragi Idli"],
    "Curry": ["Sambar"],
    "Chutney": ["Peanut Chutney", "Onion Chutney"],
    "Masala": ["Sambar Powder"]
}

# Assign categories
def assign_category(name):
    for category, items in category_mapping.items():
        if name in items:
            return category
    return "Unknown"

async def fetch_data():
    """Fetch order data from Supabase PostgreSQL"""
    conn = await asyncpg.connect(DB_URL)
    
    query = """
    SELECT customer_orderitem.order_id, customer_product.name AS product
    FROM customer_orderitem
    INNER JOIN customer_product
    ON customer_orderitem.product_id = customer_product.id
    ORDER BY customer_orderitem.id;
    """
    
    rows = await conn.fetch(query)
    await conn.close()

    return pd.DataFrame(rows, columns=["order_id", "product"])  # Convert to DataFrame

async def prepare_data():
    """Precompute co-occurrence and recommendation scores"""
    global recommendation_scores

    # Fetch data
    data = await fetch_data()

    # Assign categories
    data["category"] = data["product"].apply(assign_category)

    # Group by order_id and aggregate products in the same order
    order_items = data.groupby('order_id')['product'].apply(list).reset_index()

    # Compute co-occurrence matrix
    co_occurrence = defaultdict(lambda: defaultdict(int))

    for _, order in order_items.iterrows():
        items = order['product']
        for item1, item2 in combinations(items, 2):
            co_occurrence[item1][item2] += 1
            co_occurrence[item2][item1] += 1

    co_occurrence_df = pd.DataFrame(co_occurrence).fillna(0)
    co_occurrence_log = np.log1p(co_occurrence_df)
    co_occurrence_weighted = co_occurrence_log / co_occurrence_log.max().max()

    # Create category similarity matrix
    category_similarity = pd.DataFrame(0, index=co_occurrence_weighted.index, columns=co_occurrence_weighted.columns)

    for item1 in category_similarity.index:
        for item2 in category_similarity.columns:
            if item1 != item2 and assign_category(item1) == assign_category(item2):
                category_similarity.loc[item1, item2] = 1

    # Define weight factors
    alpha = 0.7  # Co-occurrence weightage
    beta = 0.3   # Category similarity weightage

    # Compute final recommendation scores
    recommendation_scores = alpha * co_occurrence_weighted + beta * category_similarity
    recommendation_scores.fillna(0, inplace=True)

# Define API Request Model
class OrderRequest(BaseModel):
    order_items: list[str]
    top_n: int = 3  # Default top 3 recommendations

def recommend_items(order_items, top_n=3):
    """Generate recommendations based on co-occurrence and category similarity"""
    valid_items = [item for item in order_items if item in recommendation_scores.index]

    if not valid_items:
        return {}

    scores = recommendation_scores.loc[valid_items].sum().sort_values(ascending=False)

    # Exclude already bought items
    scores = scores.drop(valid_items, errors="ignore")

    return scores.head(top_n).to_dict()

# API Endpoint
@app.post("/recommend")
async def get_recommendations(request: OrderRequest):
    recommendations = recommend_items(request.order_items, request.top_n)
    return {"recommendations": recommendations}

@app.on_event("startup")
async def on_startup():
    """Precompute recommendation scores when FastAPI starts"""
    await prepare_data()
