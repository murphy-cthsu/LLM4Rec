"""
Convert user session data to format required by the GPT4Rec model.

This script processes a CSV file containing user session history and user class information,
transforming it into the format needed by the GPT4Rec model.
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from collections import defaultdict

def parse_session_entry(session_str):
    """Parse a session entry from the input CSV format."""
    if not session_str or pd.isna(session_str):
        return None
    
    # Extract session ID, event type, and product ID using regex
    pattern = r'\[(.*?), datetime\.datetime\((.*?)\), \'(.*?)\', \'(.*?)\'\]'
    match = re.match(pattern, session_str)
    
    if match:
        return {
            'session_id': match.group(1),
            'datetime': match.group(2),
            'event_type': match.group(3),
            'product_id': match.group(4)
        }
    return None

def convert_data_for_cllm(input_csv, output_dir, dataset_name="user_session_data"):
    """
    Convert user session data to the format required by the CLLM pipeline.
    
    Args:
        input_csv: Path to the input CSV file
        output_dir: Directory to save the processed data
        dataset_name: Name of the dataset folder
    """
    # Create output directories
    dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "user_item_texts"), exist_ok=True)
    
    # Read the CSV file
    df = pd.read_csv(input_csv)
    
    # Collect session data
    user_sessions = defaultdict(list)
    user_class = {}
    all_products = set()
    
    # Process each row
    for _, row in df.iterrows():
        class_name = row.get('Class')
        session_id = None
        
        # Go through numbered columns with session data
        for i in range(120):  # Assuming columns 0-119 might contain session data
            col_name = str(i)
            if col_name not in row or pd.isna(row[col_name]):
                continue
                
            session_data = parse_session_entry(row[col_name])
            if not session_data:
                continue
                
            # Get the session ID (will be the same for all entries in a row)
            if not session_id:
                session_id = session_data['session_id']
            
            # Store the product ID and event type
            user_sessions[session_id].append({
                'product_id': session_data['product_id'],
                'event_type': session_data['event_type']
            })
            
            # Keep track of all unique products
            all_products.add(session_data['product_id'])
            
            # Store the user class
            if class_name and session_id:
                user_class[session_id] = class_name
    print(user_class)
    # Create mappings for users and items
    unique_users = sorted(list(user_sessions.keys()))
    unique_items = sorted(list(all_products))
    
    user_map = {user_id: idx for idx, user_id in enumerate(unique_users)}
    item_map = {item_id: idx for idx, item_id in enumerate(unique_items)}
    
    num_users = len(user_map)
    num_items = len(item_map)
    
    print(f"Found {num_users} unique users (sessions)")
    print(f"Found {num_items} unique products")
    
    # Create the interaction matrix (focusing on purchase events)
    rows = []
    cols = []
    data = []
    # print(user_map)
    for user_id, sessions in user_sessions.items():
        user_idx = user_map[user_id]
        
        # Get products the user purchased
        purchased_products = set()
        for session in sessions:
            if session['event_type'] == 'purchase':
                purchased_products.add(session['product_id'])
        
        # Add to the interaction matrix
        for product_id in purchased_products:
            if product_id in item_map:  # Just to be safe
                rows.append(user_idx)
                cols.append(item_map[product_id])
                data.append(1.0)  # Binary interaction
    
    # Create the sparse matrix
    train_matrix = csr_matrix((data, (rows, cols)), shape=(num_users, num_items))
    
    # Save the interaction matrix
    save_npz(os.path.join(dataset_dir, "train_matrix.npz"), train_matrix)
    
    # Create metadata file
    meta_data = {
        "num_users": num_users,
        "num_items": num_items
    }
    
    with open(os.path.join(dataset_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta_data, f)
    
    # Create review-like content (using user class as descriptive text)
    reviews = []
    for user_id, sessions in user_sessions.items():
        user_idx = user_map[user_id]
        user_type = user_class.get(user_id, "unknown")
        
        # Create a review-like entry for each purchased product
        for session in sessions:
            if session['event_type'] == 'purchase':
                product_id = session['product_id']
                item_idx = item_map[product_id]
                
                # Format: ("user_X reviewed item_Y", "descriptive text")
                prompt = f"user_{user_idx} reviewed item_{item_idx}"
                content = f"This user is a {user_type} who purchased {product_id}."
                
                reviews.append((prompt, content))
    
    # Save review data
    with open(os.path.join(dataset_dir, "user_item_texts", "review.pkl"), "wb") as f:
        pickle.dump(reviews, f)
    
    # Create a mapping file to convert back to original IDs later
    mappings = {
        "user_map": user_map,
        "item_map": item_map,
        "user_class": user_class
    }
    
    with open(os.path.join(dataset_dir, "mappings.pkl"), "wb") as f:
        pickle.dump(mappings, f)
    
    print(f"Dataset prepared and saved to {dataset_dir}")
    print(f"Number of users: {num_users}")
    print(f"Number of items: {num_items}")
    print(f"Number of interactions: {len(data)}")
    print(f"Number of reviews: {len(reviews)}")

if __name__ == "__main__":
    convert_data_for_cllm(
        input_csv="../sampled_data_with_predicted_class.csv",
        output_dir="data",
        dataset_name="user_session_data"
    )