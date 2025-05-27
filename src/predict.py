'''
MIT License
Copyright (c) 2024 Yaochen Zhu
'''

import re
import os
import sys
import pickle
import argparse
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.nn import functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from scipy.sparse import load_npz
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

sys.path.append("src/libs")
from tokenizer import TokenizerWithUserItemIDTokensBatch

from data import UserItemContentGPTDatasetBatch
from data import RecommendationGPTTrainGeneratorBatch
from data import RecommendationGPTTestGeneratorBatch

from model import GPT4RecommendationBaseModel
from model import ContentGPTForUserItemWithLMHeadBatch
from model import CollaborativeGPTwithItemRecommendHead
from util import Recall_at_k, NDCG_at_k

# Configuration for local paths
local_root = "checkpoints"
if not os.path.exists(local_root):
    os.makedirs(local_root, exist_ok=True)

def main():
    # Use regular device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
        help="specify the dataset for experiment")
    parser.add_argument("--lambda_V", type=float, required=True,
        help="specify the regularization parameter")
    parser.add_argument("--data_path", type=str, required=True,
        help="path to your dataset directory")
    parser.add_argument("--pretrained_path", type=str, required=True,
        help="path to pretrained models directory")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B",
        help="Qwen model name or path")
    parser.add_argument("--use_half_precision", action="store_true",
        help="Use half precision (fp16) for memory efficiency")
    parser.add_argument("--model_type", type=str, choices=["content", "collaborative"], 
                       default="collaborative",
                       help="Which model to use for prediction")
    args = parser.parse_args()
    
    dataset = args.dataset
    lambda_V = args.lambda_V
    data_path = args.data_path
    pretrained_path = args.pretrained_path
    model_name = args.model_name
    use_half_precision = args.use_half_precision
    model_type = args.model_type
    
    print("-----Current Setting-----")
    print(f"dataset: {dataset}")
    print(f"lambda_V: {lambda_V}")
    print(f"data_path: {data_path}")
    print(f"pretrained_path: {pretrained_path}")
    print(f"model_name: {model_name}")
    print(f"use_half_precision: {use_half_precision}")
    print(f"model_type: {model_type}")
    
    # Check if GPU is available
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU available, running on CPU")
    
    '''
        Get the basic information of the dataset
    '''
    print("-----Begin Obtaining Dataset Info-----")
    data_root = os.path.join(data_path, dataset)
    meta_path = os.path.join(data_root, "meta.pkl")

    with open(meta_path, "rb") as f:
        meta_data = pickle.load(f)
        
    num_users = meta_data["num_users"]
    num_items = meta_data["num_items"]
    print(f"num_users: {num_users}")
    print(f"num_items: {num_items}")
    print("-----End Obtaining Dataset Info-----\n")

    '''
        Load Qwen3 model and tokenizer
    '''
    print("-----Begin Loading Qwen3 Model and Tokenizer-----")
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with appropriate precision
        if use_half_precision and torch.cuda.is_available():
            qwen_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16
            )
        else:
            qwen_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float32
            )
        
        print(f"Successfully loaded {model_name}")
        print(f"Model dtype: {next(qwen_model.parameters()).dtype}")
        print(f"Vocab size: {qwen_model.config.vocab_size}")
        print(f"Hidden size: {qwen_model.config.hidden_size}")
        
    except Exception as e:
        print(f"Error loading Qwen model: {e}")
        print("Please check if the model name/path is correct")
        return
    print("-----End Loading Qwen3 Model and Tokenizer-----\n")

    '''
        Create extended tokenizer with user/item tokens
    '''
    print("-----Begin Creating Extended Tokenizer-----")
    try:
        # Use the updated TokenizerWithUserItemIDTokensBatch that works with Qwen
        extended_tokenizer = TokenizerWithUserItemIDTokensBatch(
            model_name,  # Use the Qwen model name directly
            num_users,
            num_items
        )
        
        print(f"Successfully created extended tokenizer")
        print(f"Original vocab size: {qwen_model.config.vocab_size}")
        print(f"Extended vocab size: {len(extended_tokenizer.tokenizer)}")
        print(f"Added {num_users} user tokens and {num_items} item tokens")
        
    except Exception as e:
        print(f"Error creating extended tokenizer: {e}")
        return
    print("-----End Creating Extended Tokenizer-----\n")

    '''
        Obtain the testing data generator
    '''
    print("-----Begin Obtaining the Test Data Generator-----")
    train_mat_path = os.path.join(data_root, "train_matrix.npz")
    test_mat_path = os.path.join(data_root, "test_matrix.npz")
    
    print(f"Loading train data from {train_mat_path}...")
    if not os.path.exists(train_mat_path):
        raise FileNotFoundError(f"Train matrix not found at {train_mat_path}")
    
    print(f"Loading test data from {test_mat_path}...")
    if not os.path.exists(test_mat_path):
        raise FileNotFoundError(f"Test matrix not found at {test_mat_path}")
    
    # Get the testing data generator
    train_mat = load_npz(train_mat_path)
    test_mat = load_npz(test_mat_path)
    test_data_gen = RecommendationGPTTestGeneratorBatch(extended_tokenizer, train_mat, test_mat)

    print("Success!")
    print("-----End Obtaining the Test Data Generator-----\n")

    '''
        Create config compatible with Qwen3
    '''
    print("-----Begin Setting Up the Config-----")
    # Use Qwen model's config as base and extend it
    config = qwen_model.config
    config.num_users = num_users
    config.num_items = num_items
    # Update vocab_size to include the extended vocabulary
    config.vocab_size = len(extended_tokenizer.tokenizer)  # This includes user/item tokens
    
    print(f"[Config] Users: {num_users}")
    print(f"[Config] Items: {num_items}")
    print(f"[Config] Original vocab size: {qwen_model.config.vocab_size}")
    print(f"[Config] Extended vocab size: {config.vocab_size}")
    print(f"[Config] Hidden size: {config.hidden_size}")
    print("Success!")
    print("-----End Setting Up the Config-----\n")

    '''
        Instantiate the GPT for recommendation model
    '''
    print("-----Begin Instantiating the Recommendation GPT Model-----")
    
    # Create base model
    base_model = GPT4RecommendationBaseModel(config, qwen_model)

    # Load pretrained embeddings
    model_root = os.path.join(local_root, "models", dataset, model_type)
    
    user_emb_path = os.path.join(model_root, f"user_embeddings_{lambda_V}.pt") 
    item_emb_path = os.path.join(model_root, f"item_embeddings_{lambda_V}.pt") 
    
    print(f"Loading pretrained embeddings from {model_root}...")
    
    if not os.path.exists(user_emb_path):
        raise FileNotFoundError(f"User embeddings not found at {user_emb_path}")
    if not os.path.exists(item_emb_path):
        raise FileNotFoundError(f"Item embeddings not found at {item_emb_path}")

    # Load the embeddings
    base_model.user_embeddings.load_state_dict(
        torch.load(user_emb_path, map_location=device))
    print("Load pretrained user embeddings: Success!")
    
    base_model.item_embeddings.load_state_dict(
        torch.load(item_emb_path, map_location=device))
    print("Load pretrained item embeddings: Success!")

    # Create the recommendation model
    rec_model = CollaborativeGPTwithItemRecommendHead(config, base_model)
    print("Success!")
    print("-----End Instantiating the Recommendation GPT Model-----\n")

    '''
        Create data loader for testing
    '''
    print("-----Begin Creating the DataLoader-----")

    # Create the testing data loader
    batch_size = 64 if use_half_precision else 32  # Adjust batch size based on precision
    test_data_loader = DataLoader(test_data_gen, 
                                  batch_size=batch_size, 
                                  collate_fn=test_data_gen.collate_fn,
                                  num_workers=0)  # Set to 0 to avoid multiprocessing issues
    
    print(f"Test batches: {len(test_data_loader)}")
    print(f"Total test samples: {len(test_data_gen)}")
    print("-----End Creating the DataLoader-----\n")

    # Move model to device
    rec_model.to(device)
    
    # Set the model to evaluation mode
    rec_model.eval()
    
    # Initialize metrics
    total_recall_20 = 0
    total_recall_40 = 0
    total_NDCG_100 = 0
    total_samples = 0

    print("-----Begin Evaluation-----")
    with torch.no_grad():
        progress_bar = tqdm(test_data_loader, desc="Evaluating", ncols=100)
        
        for batch_idx, (input_ids, train_mat, target_mat, attention_mask) in enumerate(progress_bar):
            try:
                # Move tensors to the correct device
                input_ids = input_ids.to(device)
                train_mat = train_mat.to(device)
                target_mat = target_mat.to(device)
                attention_mask = attention_mask.to(device)

                # Get item scores and rank them
                rec_loss, item_scores = rec_model(input_ids, 
                                                    target_mat, 
                                                    attention_mask)
                
                # Set score of interacted items to the lowest
                item_scores[train_mat > 0] = -float("inf")  

                # Calculate Recall@K and NDCG@K for each user
                target_mat_cpu = target_mat.cpu().numpy()
                item_scores_cpu = item_scores.cpu().numpy()
                
                batch_recall_20 = Recall_at_k(target_mat_cpu, item_scores_cpu, k=20, agg="sum")
                batch_recall_40 = Recall_at_k(target_mat_cpu, item_scores_cpu, k=40, agg="sum")
                batch_NDCG_100 = NDCG_at_k(target_mat_cpu, item_scores_cpu, k=100, agg="sum")
                
                total_recall_20 += batch_recall_20
                total_recall_40 += batch_recall_40
                total_NDCG_100 += batch_NDCG_100
                total_samples += target_mat.shape[0]
                
                # Update progress bar
                current_recall_20 = total_recall_20 / total_samples
                current_recall_40 = total_recall_40 / total_samples
                current_NDCG_100 = total_NDCG_100 / total_samples
                
                progress_bar.set_postfix({
                    "R@20": f"{current_recall_20:.4f}",
                    "R@40": f"{current_recall_40:.4f}",
                    "N@100": f"{current_NDCG_100:.4f}"
                })
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM error at batch {batch_idx}, skipping...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

    # Calculate final average metrics
    final_recall_20 = total_recall_20 / total_samples
    final_recall_40 = total_recall_40 / total_samples
    final_NDCG_100 = total_NDCG_100 / total_samples
    
    print("-----Evaluation Complete-----")
    print(f"Final Testing Results:")
    print(f"Recall@20: {final_recall_20:.4f}")
    print(f"Recall@40: {final_recall_40:.4f}")
    print(f"NDCG@100: {final_NDCG_100:.4f}")
    print(f"Total samples evaluated: {total_samples}")
    
    # Save results
    results_dir = os.path.join(local_root, "results", dataset)
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"results_{model_type}_{lambda_V}.txt")
    
    with open(results_path, "w") as f:
        f.write("Model_Type,Lambda_V,Recall@20,Recall@40,NDCG@100,Total_Samples\n")
        f.write(f"{model_type},{lambda_V},{final_recall_20:.4f},{final_recall_40:.4f},{final_NDCG_100:.4f},{total_samples}\n")
    
    print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    main()