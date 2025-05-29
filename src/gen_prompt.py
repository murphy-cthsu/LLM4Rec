
"""
Convert user session data to format required by the CLLM4Rec model with class-based user IDs.

This enhanced script processes a CSV file containing user session history and user class information
that has been tagged by an LLM, transforming it into the format needed by the CLLM4Rec model.
Users with the same class label will share a common user ID prefix.
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
    
    # Try an alternative format (in case the regex doesn't match)
    try:
        # Remove brackets and split by commas
        parts = session_str.strip('[]').split(',')
        if len(parts) >= 4:
            session_id = parts[0].strip()
            datetime_part = parts[1].strip()
            event_type = parts[2].strip().strip("'")
            product_id = parts[3].strip().strip("'")
            
            return {
                'session_id': session_id,
                'datetime': datetime_part,
                'event_type': event_type,
                'product_id': product_id
            }
    except:
        pass
    
    print(f"WARNING: Could not parse session entry: {session_str}")
    return None

def load_and_analyze_sample_data(input_csv, num_samples=5):
    """
    Load and analyze a sample of the input CSV to understand the data structure.
    This function helps visualize what user sessions and classes look like in the dataset.
    
    Args:
        input_csv: Path to the input CSV file
        num_samples: Number of sample rows to display
    """
    print(f"\nANALYZING SAMPLE DATA FROM {input_csv}")
    print("="*80)
    
    try:
        # Load the CSV file
        df = pd.read_csv(input_csv)
        print(f"CSV loaded with {len(df)} rows and {len(df.columns)} columns")
        
        # Display column names to understand the dataset structure
        print("\nFirst few column names:")
        for col in list(df.columns)[:10]:
            print(f"  - {col}")
        print("  ... plus additional columns")
        
        # Show class distribution
        if "Class" in df.columns:
            print("\nClass distribution in sample data:")
            class_counts = df["Class"].value_counts()
            for class_name, count in class_counts.items():
                print(f"  {class_name}: {count} ({count/len(df)*100:.1f}%)")
        
        # Print some actual column values to see the format
        print("\nSample column values to understand the session format:")
        for col in ['0', '1', '2', '3', '4']:
            if col in df.columns:
                non_null_values = df[col].dropna()
                if len(non_null_values) > 0:
                    sample_value = non_null_values.iloc[0]
                    print(f"Column {col} sample: {sample_value}")
                    
                    # Try parsing it to confirm format
                    parsed = parse_session_entry(sample_value)
                    if parsed:
                        print(f"  Parsed: session_id={parsed['session_id']}, event_type={parsed['event_type']}, product_id={parsed['product_id']}")
                    else:
                        print(f"  WARNING: Could not parse this value with current regex!")
        
        # Examine a few rows to understand the session entries
        print(f"\nAnalyzing {num_samples} sample rows:")
        for i, (_, row) in enumerate(df.head(num_samples).iterrows()):
            print(f"\nSample {i+1}:")
            print(f"  Class: {row.get('Class', 'N/A')}")
            
            # Find an example of session entry in the numbered columns
            found_example = False
            for col in [str(j) for j in range(50)]:  # Check first 50 columns
                if col in row and not pd.isna(row[col]):
                    print(f"  Example session entry (Column {col}): {row[col]}")
                    found_example = True
                    break
            
            if not found_example:
                print("  No valid session entry found in first 50 columns")
    
    except Exception as e:
        print(f"Error analyzing sample data: {str(e)}")
        import traceback
        traceback.print_exc()

def convert_data_for_cllm(input_csv, output_dir, dataset_name="user_session_data", class_column="Class", validation_ratio=0.1, test_ratio=0.1, min_user_interactions=2):
    """
    Convert user session data with LLM-generated user classes to the format required by the CLLM4Rec pipeline.
    Users with the same class label will be assigned the SAME user ID (a numerical ID).
    
    Args:
        input_csv: Path to the input CSV file with LLM-tagged user classifications
        output_dir: Directory to save the processed data
        dataset_name: Name of the dataset folder
        class_column: The column name in CSV containing LLM-generated user classifications
        validation_ratio: Percentage of user interactions to use for validation
        test_ratio: Percentage of user interactions to use for testing
        min_user_interactions: Minimum number of interactions required for a user to be included
    """
    # Create output directories
    dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "user_item_texts"), exist_ok=True)
    
    # Read the CSV file
    print(f"Reading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    print(f"Data loaded with {len(df)} rows.")
    
    # Collect session data
    user_sessions = defaultdict(list)
    user_class = {}
    all_products = set()
    
    # Process each row
    for _, row in df.iterrows():
        # Get the LLM-generated class (user type/behavior classification)
        class_name = row.get(class_column, "unknown")
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
                'event_type': session_data['event_type'],
                'datetime': session_data['datetime']  # Store datetime for potential chronological sorting
            })
            
            # Keep track of all unique products
            all_products.add(session_data['product_id'])
            
            # Store the LLM-generated user class
            if class_name and session_id:
                user_class[session_id] = class_name
    
    print(f"Extracted user sessions for {len(user_sessions)} users")
    print(f"Found {len(user_class)} user classifications from LLM")
    
    # Filter users with too few interactions (optional)
    if min_user_interactions > 1:
        filtered_users = {
            user_id: sessions for user_id, sessions in user_sessions.items() 
            # if len([s for s in sessions if s['event_type'] == 'purchase']) >= min_user_interactions
        }
        print(f"Filtered users from {len(user_sessions)} to {len(filtered_users)} based on minimum interactions")
        user_sessions = filtered_users
    
    # Group users by class label
    class_to_users = defaultdict(list)
    for user_id, class_name in user_class.items():
        if user_id in user_sessions:  # Only include users that passed the filtering
            class_to_users[class_name].append(user_id)
    
    # Create user ID based solely on class number (each class gets ONE numerical ID)
    # Map original user IDs to their class-based numerical ID
    class_name_set = set()
    class_id_map = {}
    class_based_user_map = {}
    
    # Assign a numerical ID (starting from 0) to each class
    for class_idx, class_name in enumerate(sorted(class_to_users.keys())):
        # Map class name to a numerical ID
        class_id_map[class_name] = class_idx
        
        # Map all users of this class to this numerical ID
        for user_id in class_to_users[class_name]:
            class_based_user_map[user_id] = class_idx
    
    print("\nAssigned numerical class IDs:")
    for class_name, class_id in class_id_map.items():
        user_count = len(class_to_users[class_name])
        print(f"  Class: {class_name} -> User ID: {class_id} (assigned to {user_count} users)")
    
    # Create mappings for items
    unique_items = sorted(list(all_products))
    
    # Map items to numeric indices
    item_map = {item_id: idx for idx, item_id in enumerate(unique_items)}
    
    num_users = len(class_id_map)  # Number of classes = number of unique users
    num_items = len(item_map)
    
    print(f"Found {num_users} unique classes (now treated as users)")
    print(f"Found {num_items} unique products")
    
    # Merge interactions by class to create class-level data
    class_train_data = defaultdict(list)
    class_val_data = defaultdict(list)
    class_test_data = defaultdict(list)
    class_purchase_data = defaultdict(list)
    class_load_data = defaultdict(list)
    
    # Group all interactions by class
    for orig_user_id, sessions in user_sessions.items():
        if orig_user_id not in class_based_user_map:
            continue
            
        class_id = class_based_user_map[orig_user_id]
        
        # Separate purchases and loads
        for session in sessions:
            product_id = session['product_id']
            event_type = session['event_type']
            
            if event_type == 'purchase':
                if product_id not in class_purchase_data[class_id]:
                    class_purchase_data[class_id].append(product_id)
            elif event_type == 'load':
                if product_id not in class_load_data[class_id]:
                    class_load_data[class_id].append(product_id)
    
    # Split products into train/val/test for each class
    for class_id, products in class_purchase_data.items():
        if len(products) < 2:
            continue
            
        # Shuffle products to randomize the split
        np.random.shuffle(products)
        
        # Determine split sizes
        n_products = len(products)
        n_test = max(1, int(n_products * test_ratio))
        n_val = max(1, int(n_products * validation_ratio))
        n_train = n_products - n_test - n_val
        
        if n_train < 1:
            n_train = 1
            n_val = min(1, n_val)
            n_test = n_products - n_train - n_val
        
        # Split the products
        train_items = products[:n_train]
        val_items = products[n_train:n_train+n_val]
        test_items = products[n_train+n_val:]
        
        # Assign to class data
        class_train_data[class_id] = train_items
        class_val_data[class_id] = val_items
        class_test_data[class_id] = test_items
    
    # Create train, validation, and test matrices
    def create_interaction_matrix(user_item_data, item_map, num_users, num_items):
        rows = []
        cols = []
        data = []
        for user_id, items in user_item_data.items():
            for product_id in items:
                if product_id in item_map:
                    rows.append(user_id)  # User ID is already the numerical class ID
                    cols.append(item_map[product_id])
                    data.append(1.0)  # Binary interaction
        return csr_matrix((data, (rows, cols)), shape=(num_users, num_items))
    
    # Create matrices
    train_matrix = create_interaction_matrix(class_train_data, item_map, num_users, num_items)
    val_matrix = create_interaction_matrix(class_val_data, item_map, num_users, num_items)
    test_matrix = create_interaction_matrix(class_test_data, item_map, num_users, num_items)
    
    # Save matrices
    save_npz(os.path.join(dataset_dir, "train_matrix.npz"), train_matrix)
    save_npz(os.path.join(dataset_dir, "val_matrix.npz"), val_matrix)
    save_npz(os.path.join(dataset_dir, "test_matrix.npz"), test_matrix)
    
    # Create metadata file
    meta_data = {
        "num_users": num_users,  # This is now the number of unique classes
        "num_items": num_items,
        "class_names": list(class_id_map.keys())
    }
    
    with open(os.path.join(dataset_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta_data, f)
    
    # Create user class distribution stats
    class_distribution = {}
    for class_name, users in class_to_users.items():
        class_distribution[class_name] = len(users)
    
    print("\nLLM-generated user class distribution:")
    for class_name, count in sorted(class_distribution.items(), key=lambda x: x[1], reverse=True):
        print(f"  {class_name}: {count} users ({count/sum(class_distribution.values())*100:.1f}%)")
    
    # Create review-like content using LLM-generated user classes
    reviews = []
    class_id_to_name = {v: k for k, v in class_id_map.items()}
    
    for class_id, class_items in class_train_data.items():
        # Get the corresponding class name
        class_name = class_id_to_name.get(class_id, "unknown")
        
        # Create review-like entries for each purchased product
        for product_id in class_items:
            if product_id in item_map:
                item_idx = item_map[product_id]
                
                # Format: ("class_id reviewed item_Y", "descriptive text with LLM-generated class")
                prompt = f"<user_{class_id}> reviewed <item_{item_idx}>"
                content = f"This user is a {class_name} who purchased {product_id}."
                
                reviews.append((prompt, content))
    
    # Save review data for content modeling
    with open(os.path.join(dataset_dir, "user_item_texts", "review.pkl"), "wb") as f:
        pickle.dump(reviews, f)
    
    # Create mappings file for reference and later use
    mappings = {
        "user_map": {i: i for i in range(num_users)},  # Identity mapping for user indices
        "item_map": item_map,
        "class_id_map": class_id_map,  # Class name to numerical ID
        "class_id_to_name": class_id_to_name,  # Numerical ID to class name
        "original_to_class_id": class_based_user_map,  # Original user ID to numerical class ID
        "reverse_item_map": {idx: item_id for item_id, idx in item_map.items()},
        "user_class": class_id_to_name  # For compatibility with original code
    }
    
    # Also save the user interaction data by type for prompt generation
    mappings["class_purchase_data"] = class_purchase_data
    mappings["class_load_data"] = class_load_data
    
    with open(os.path.join(dataset_dir, "mappings.pkl"), "wb") as f:
        pickle.dump(mappings, f)
    
    print(f"\nDataset prepared and saved to {dataset_dir}")
    print(f"Number of class-based users: {num_users}")
    print(f"Number of items: {num_items}")
    print(f"Number of training interactions: {len(train_matrix.data)}")
    print(f"Number of validation interactions: {len(val_matrix.data)}")
    print(f"Number of test interactions: {len(test_matrix.data)}")
    print(f"Number of review texts: {len(reviews)}")
    
    return dataset_dir

def generate_cllm_prompt_files(dataset_dir, prompt_format="soft+hard"):
    """
    Generate prompt files for CLLM4Rec training and fine-tuning based on the CLLM4Rec paper.
    Modified to account for numerical class IDs and both load and purchase operations.
    
    Args:
        dataset_dir: Directory containing the processed dataset
        prompt_format: Type of prompting to use ('soft+hard' is recommended in the paper)
    """
    # Load mappings and data
    print("\nLoading mappings and data for prompt generation...")
    
    try:
        with open(os.path.join(dataset_dir, "mappings.pkl"), "rb") as f:
            mappings = pickle.load(f)
        
        print("Successfully loaded mappings.pkl")
        print(f"Keys in mappings: {list(mappings.keys())}")
        
        # Print some key information for debugging
        if "class_id_to_name" in mappings:
            print(f"Number of classes: {len(mappings['class_id_to_name'])}")
            print(f"Sample class_id_to_name: {list(mappings['class_id_to_name'].items())[:3]}")
        else:
            print("WARNING: 'class_id_to_name' not found in mappings")
            
        if "class_purchase_data" in mappings:
            print(f"Number of classes with purchase data: {len(mappings['class_purchase_data'])}")
            for class_id, purchases in list(mappings['class_purchase_data'].items())[:2]:
                print(f"  Class {class_id}: {len(purchases)} purchased items")
        else:
            print("WARNING: 'class_purchase_data' not found in mappings")
            
        if "class_load_data" in mappings:
            print(f"Number of classes with load data: {len(mappings['class_load_data'])}")
            for class_id, loads in list(mappings['class_load_data'].items())[:2]:
                print(f"  Class {class_id}: {len(loads)} loaded items")
        else:
            print("WARNING: 'class_load_data' not found in mappings")
        
        with open(os.path.join(dataset_dir, "user_item_texts", "review.pkl"), "rb") as f:
            reviews = pickle.load(f)
        
        print(f"Successfully loaded review.pkl with {len(reviews)} reviews")
        if reviews:
            print(f"Sample review: {reviews[0]}")
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None
    
    # Extract necessary mappings - using more reliable key access
    user_map = mappings.get("user_map", {})
    item_map = mappings.get("item_map", {})
    class_id_to_name = mappings.get("class_id_to_name", {})
    reverse_item_map = mappings.get("reverse_item_map", {})
    class_purchase_data = mappings.get("class_purchase_data", {})
    class_load_data = mappings.get("class_load_data", {})
    
    # Create directory for prompts
    prompts_dir = os.path.join(dataset_dir, "prompts")
    os.makedirs(prompts_dir, exist_ok=True)
    
    # Generate collaborative prompts for training - handling both load and purchase
    print("\nGenerating collaborative prompts...")
    collaborative_prompts = []
    
    for class_id, class_name in class_id_to_name.items():
        # Get purchased and loaded items for this class
        purchased_items = class_purchase_data.get(class_id, [])
        loaded_items = class_load_data.get(class_id, [])
        
        print(f"Class {class_id} ({class_name}): {len(purchased_items)} purchases, {len(loaded_items)} loads")
        
        if purchased_items:
            # Format for purchase interactions
            prompt = f"<user_{class_id}> has purchased "
            collaborative_prompts.append((class_id, prompt, class_name, "purchase"))
            print(f"  Added purchase prompt: {prompt}")
        
        if loaded_items:
            # Format for load interactions
            prompt = f"<user_{class_id}> has viewed/loaded "
            collaborative_prompts.append((class_id, prompt, class_name, "load"))
            print(f"  Added load prompt: {prompt}")
    
    # Generate content prompts for training
    print("\nGenerating content prompts...")
    content_prompts = []
    content_errors = 0
    
    for i, (prompt_text, content_text) in enumerate(reviews):
        # Format: "class_id reviewed item_Y" -> "This user is a {class_name} who purchased {product_id}."
        if "reviewed" in prompt_text:
            parts = prompt_text.split()
            if len(parts) >= 3:
                user_part = parts[0]  # class_id as number
                item_part = parts[2]  # "item_Y"
                
                try:
                    class_id = int(user_part)
                    item_idx = int(item_part.split('_')[1])
                    class_name = class_id_to_name.get(class_id, "unknown")
                    
                    # Format according to updated approach
                    prompt = f"<user_{class_id}> writes a review for <{item_part}> :"
                    content_prompts.append((class_id, item_idx, prompt, content_text))
                    
                    if i < 2:  # Print first few for debugging
                        print(f"  Added content prompt: {prompt}")
                except Exception as e:
                    content_errors += 1
                    if content_errors <= 3:  # Limit error messages
                        print(f"  Error parsing review {i}: {str(e)}, prompt_text: {prompt_text}")
    
    print(f"Processed {len(reviews)} reviews, encountered {content_errors} errors")
    
    # Save the prompts
    with open(os.path.join(prompts_dir, "collaborative_prompts.pkl"), "wb") as f:
        pickle.dump(collaborative_prompts, f)
    
    with open(os.path.join(prompts_dir, "content_prompts.pkl"), "wb") as f:
        pickle.dump(content_prompts, f)
    
    # Generate recommendation-oriented prompts for fine-tuning, now with both load and purchase information
    print("\nGenerating recommendation prompts...")
    from scipy.sparse import load_npz
    
    try:
        train_matrix = load_npz(os.path.join(dataset_dir, "train_matrix.npz"))
        print(f"Loaded train_matrix with shape {train_matrix.shape}, {train_matrix.nnz} nonzero elements")
    except Exception as e:
        print(f"Error loading train matrix: {str(e)}")
        train_matrix = None
    
    mask_ratio = 0.5  # Mask 50% of interactions for fine-tuning
    recommendation_prompts = []
    
    if train_matrix is not None:
        for class_id, class_name in class_id_to_name.items():
            try:
                # Get the class's training items
                if class_id >= train_matrix.shape[0]:
                    print(f"  Warning: class_id {class_id} is out of bounds for train_matrix with shape {train_matrix.shape}")
                    continue
                    
                user_items = train_matrix[class_id].nonzero()[1]
                print(f"  Class {class_id} ({class_name}): {len(user_items)} training items")
                
                if len(user_items) < 2:
                    print(f"  Skipping class {class_id} - too few training items")
                    continue
                
                # Get purchased and loaded items (using indices)
                purchased_item_indices = [item_map[item_id] for item_id in class_purchase_data.get(class_id, []) 
                                        if item_id in item_map]
                loaded_item_indices = [item_map[item_id] for item_id in class_load_data.get(class_id, []) 
                                      if item_id in item_map]
                
                print(f"  Class {class_id}: {len(purchased_item_indices)} purchased, {len(loaded_item_indices)} loaded item indices")
                
                # Only create recommendation prompts if we have enough purchase interactions
                if len(purchased_item_indices) < 2:
                    print(f"  Skipping class {class_id} - too few purchase indices")
                    continue
                    
                # Randomly mask some purchased items
                n_purchases = len(purchased_item_indices)
                n_masked = max(1, int(n_purchases * mask_ratio))
                
                # Create a boolean mask
                mask = np.zeros(n_purchases, dtype=bool)
                mask_indices = np.random.choice(n_purchases, n_masked, replace=False)
                mask[mask_indices] = True
                
                # Separate masked and unmasked items
                unmasked_purchases = np.array(purchased_item_indices)[~mask]
                masked_purchases = np.array(purchased_item_indices)[mask]
                
                if len(unmasked_purchases) == 0:
                    print(f"  Skipping class {class_id} - no unmasked purchases")
                    continue
                
                # Convert to item strings for the prompt
                unmasked_purchase_strs = [f"<item_{item_idx}>" for item_idx in unmasked_purchases]
                
                # Create the standard purchase-based prompt
                rec_prompt = f"<user_{class_id}> has purchased {' '.join(unmasked_purchase_strs)}, the user will also purchase:"
                recommendation_prompts.append((class_id, rec_prompt, masked_purchases))
                print(f"  Added standard recommendation prompt for class {class_id}")
                
                # If we have loaded items, create a context-aware prompt
                if loaded_item_indices:
                    loaded_item_strs = [f"<item_{item_idx}>" for item_idx in loaded_item_indices[:5]]  # Limit to 5 items for clarity
                    
                    # Viewed-then-purchased prompt
                    rec_prompt_contextual = f"<user_{class_id}> has viewed {' '.join(loaded_item_strs)} and purchased {' '.join(unmasked_purchase_strs)}, the user will also purchase:"
                    recommendation_prompts.append((class_id, rec_prompt_contextual, masked_purchases))
                    print(f"  Added contextual recommendation prompt for class {class_id}")
            except Exception as e:
                print(f"  Error processing class {class_id}: {str(e)}")
    
    with open(os.path.join(prompts_dir, "recommendation_prompts.pkl"), "wb") as f:
        pickle.dump(recommendation_prompts, f)
    
    print(f"\nGenerated prompts for CLLM4Rec training and saved to {prompts_dir}:")
    print(f"  Collaborative prompts: {len(collaborative_prompts)}")
    print(f"  Content prompts: {len(content_prompts)}")
    print(f"  Recommendation prompts: {len(recommendation_prompts)}")
    
    return dataset_dir

def inspect_dataset(dataset_dir):
    """
    Inspect the generated dataset, showing class-to-user mappings and sample prompts.
    
    Args:
        dataset_dir: Directory containing the processed dataset
    """
    print("\n" + "="*80)
    print("INSPECTING DATASET: ", dataset_dir)
    print("="*80)
    
    # Load mappings and prompt data
    with open(os.path.join(dataset_dir, "mappings.pkl"), "rb") as f:
        mappings = pickle.load(f)
    
    # Class ID mappings
    class_id_to_name = mappings.get("class_id_to_name", {})
    class_id_map = mappings.get("class_id_map", {})
    
    # 1. Print class label to numerical user ID mappings
    print("\nCLASS LABEL TO NUMERICAL USER ID MAPPINGS:")
    print("-"*50)
    
    for class_name, class_id in sorted(class_id_map.items(), key=lambda x: x[1]):
        print(f"Class: {class_name} -> Numerical ID: {class_id}")
    
    # 2. Load and display sample prompts
    prompts_dir = os.path.join(dataset_dir, "prompts")
    
    # Load collaborative prompts
    collab_prompts_path = os.path.join(prompts_dir, "collaborative_prompts.pkl")
    if os.path.exists(collab_prompts_path):
        with open(collab_prompts_path, "rb") as f:
            collaborative_prompts = pickle.load(f)
        
        print("\n\nSAMPLE COLLABORATIVE PROMPTS:")
        print("-"*50)
        # Get a sample of purchase and load prompts
        purchase_prompts = [p for p in collaborative_prompts if p[3] == "purchase"][:3]
        load_prompts = [p for p in collaborative_prompts if p[3] == "load"][:3]
        
        print("\nPurchase prompts:")
        for class_id, prompt, class_name, op_type in purchase_prompts:
            print(f"Class ID {class_id} (Type: {class_name}):")
            print(f"  {prompt}<item_list>")
        
        print("\nLoad/View prompts:")
        for class_id, prompt, class_name, op_type in load_prompts:
            print(f"Class ID {class_id} (Type: {class_name}):")
            print(f"  {prompt}<item_list>")
    
    # Load content prompts
    content_prompts_path = os.path.join(prompts_dir, "content_prompts.pkl")
    if os.path.exists(content_prompts_path):
        with open(content_prompts_path, "rb") as f:
            content_prompts = pickle.load(f)
        
        print("\n\nSAMPLE CONTENT PROMPTS:")
        print("-"*50)
        for class_id, item_idx, prompt, content in content_prompts[:3]:
            class_name = class_id_to_name.get(class_id, "unknown")
            print(f"Class ID {class_id} (Type: {class_name}):")
            print(f"  Prompt: {prompt}")
            print(f"  Content: {content}")
            print()
    
    # Load recommendation prompts
    rec_prompts_path = os.path.join(prompts_dir, "recommendation_prompts.pkl")
    if os.path.exists(rec_prompts_path):
        with open(rec_prompts_path, "rb") as f:
            recommendation_prompts = pickle.load(f)
        
        print("\nSAMPLE RECOMMENDATION PROMPTS:")
        print("-"*50)
        
        # Try to find examples of different prompt types
        standard_prompt = None
        contextual_prompt = None
        
        for class_id, prompt, masked_items in recommendation_prompts:
            class_name = class_id_to_name.get(class_id, "unknown")
            
            if "viewed" in prompt and contextual_prompt is None:
                contextual_prompt = (class_id, prompt, masked_items, class_name)
            elif "viewed" not in prompt and standard_prompt is None:
                standard_prompt = (class_id, prompt, masked_items, class_name)
                
            if standard_prompt and contextual_prompt:
                break
                
        # Display standard prompts
        if standard_prompt:
            class_id, prompt, masked_items, class_name = standard_prompt
            print(f"\nStandard purchase-based prompt:")
            print(f"Class ID {class_id} (Type: {class_name}):")
            print(f"  Prompt: {prompt}")
            print(f"  Targets to predict: {masked_items}")
            
        # Display contextual prompts
        if contextual_prompt:
            class_id, prompt, masked_items, class_name = contextual_prompt
            print(f"\nContextual prompt (with view/load information):")
            print(f"Class ID {class_id} (Type: {class_name}):")
            print(f"  Prompt: {prompt}")
            print(f"  Targets to predict: {masked_items}")
            
        # If we didn't find specific examples, show the first few
        if not (standard_prompt or contextual_prompt):
            for class_id, prompt, masked_items in recommendation_prompts[:3]:
                class_name = class_id_to_name.get(class_id, "unknown")
                print(f"Class ID {class_id} (Type: {class_name}):")
                print(f"  Prompt: {prompt}")
                print(f"  Targets to predict: {masked_items}")
                print()

if __name__ == "__main__":
    # Analyze a sample of the input CSV
    for i in range (10):
        file_path=f"data/cleaned_data_class_{i}.csv"
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist.")
            continue
        load_and_analyze_sample_data(f"data/cleaned_data_class_{i}.csv")

        # Process the LLM-tagged data and convert to CLLM4Rec format
        dataset_dir = convert_data_for_cllm(
            # input_csv="data/sampled_data_with_predicted_class.csv",
            input_csv=f"data/cleaned_data_class_{i}.csv",
            output_dir="data",
            dataset_name=f"user_session_data_{i}",
            class_column="Class"  # Column containing LLM-generated user classifications
        )
        
        # Generate prompt files for CLLM4Rec training
        generate_cllm_prompt_files(dataset_dir)
        
        # Inspect the dataset to see class-to-user mappings and sample prompts
        inspect_dataset(dataset_dir)