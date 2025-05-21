'''
MIT License
Copyright (c) 2024 Yaochen Zhu
Modified for single GPU execution without accelerator framework
'''

import re
import os
import sys
import pickle
import random
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
from transformers import GPT2Model, GPT2Config
from transformers import GPT2Tokenizer

sys.path.append("src/libs")
from tokenizer import TokenizerWithUserItemIDTokensBatch

from data import UserItemContentGPTDatasetBatch
from data import RecommendationGPTTrainGeneratorBatch
from data import RecommendationGPTTestGeneratorBatch

from model import GPT4RecommendationBaseModel
from model import ContentGPTForUserItemWithLMHeadBatch
from model import CollaborativeGPTwithItemRecommendHead

from util import Recall_at_k, NDCG_at_k

# Define local directories
data_root = "data"  # Local data directory
local_root = "checkpoints"  # Temporary directory

# Create necessary directories
os.makedirs(data_root, exist_ok=True)
os.makedirs(local_root, exist_ok=True)

_config = {
    "activation_function": "gelu_new",
    "architectures": [
    "GPT2LMHeadModel"
    ],
    "attn_pdrop": 0.1,
    "bos_token_id": 50256,
    "embd_pdrop": 0.1,
    "eos_token_id": 50256,
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-05,
    "model_type": "gpt2",
    "n_ctx": 1024,
    "n_embd": 768,
    "n_head": 12,
    "n_layer": 12,
    "n_positions": 1024,
    "resid_pdrop": 0.1,
    "summary_activation": None,
    "summary_first_dropout": 0.1,
    "summary_proj_to_labels": True,
    "summary_type": "cls_index",
    "summary_use_proj": True,
    "task_specific_params": {
    "text-generation": {
        "do_sample": True,
        "max_length": 50
    }
    },
    "vocab_size": 50257
}

def main():
    # Set up CUDA device if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="amazon",
        help="specify the dataset for experiment")
    parser.add_argument("--lambda_V", type=str, default="0.01",
        help="specify the regularization parameter")
    parser.add_argument("--model_path", type=str, required=True, help="path to models directory")
    args = parser.parse_args()
    model_root=args.model_path
    dataset = args.dataset
    lambda_V = float(args.lambda_V)
    os.makedirs(model_root, exist_ok=True)

    print("-----Current Setting-----")
    print(f"dataset: {dataset}")
    print(f"lambda_V: {args.lambda_V}")
    print(f"device: {device}")
    
    # Check for GPU
    if torch.cuda.is_available():
        print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU")
    
    '''
        Get the basic information of the dataset
    '''
    print("-----Begin Obtaining Dataset Info-----")
    dataset_dir = os.path.join(data_root, dataset)
    meta_path = os.path.join(dataset_dir, "meta.pkl")

    # Load the meta data
    with open(meta_path, "rb") as f:
        meta_data = pickle.load(f)
        
    num_users = meta_data["num_users"]
    num_items = meta_data["num_items"]
    print(f"num_users: {num_users}")
    print(f"num_items: {num_items}")
    print("-----End Obtaining Dataset Info-----\n")

    '''
        Obtain the tokenizer with user/item tokens
    '''
    print("-----Begin Obtaining the Tokenizer-----")
    tokenizer_root = model_root
    print(f"Loading pretrained tokenizer from {tokenizer_root}...")
    
    vocab_file = os.path.join(tokenizer_root, "vocab.json")
    merges_file = os.path.join(tokenizer_root, "merges.txt")
    
    # Check if tokenizer files exist
    if not os.path.exists(vocab_file) or not os.path.exists(merges_file):
        print("Please download the tokenizer files first!")
        print(f"Expected locations: {vocab_file} and {merges_file}")
        return
    
    tokenizer = TokenizerWithUserItemIDTokensBatch(vocab_file, 
                                                   merges_file,
                                                   num_users,
                                                   num_items)
    print("Success!")
    print("-----End Obtaining the Tokenizer-----\n")
    
    '''
        Define the review data generator
    '''
    print("-----Begin Obtaining the Review Data Generator-----")
    review_path = os.path.join(dataset_dir, "user_item_texts", "review.pkl")
    print(f"Loading data from {review_path}...")
    
    # Check if the review file exists
    if not os.path.exists(review_path):
        print(f"Review file not found at {review_path}")
        return
    
    review_data_gen = UserItemContentGPTDatasetBatch(tokenizer, review_path)
    print("Success!")
    print("-----End Obtaining the Review Data Generator-----\n")

    '''
        Obtain the training/validation data generator
    '''
    print("-----Begin Obtaining the Collaborative Data Generator-----")
    train_mat_path = os.path.join(dataset_dir, "train_matrix.npz")
    val_mat_path = os.path.join(dataset_dir, "val_matrix.npz")
    
    # Check if the matrix files exist
    if not os.path.exists(train_mat_path) or not os.path.exists(val_mat_path):
        print(f"Matrix files not found: {train_mat_path} or {val_mat_path}")
        return
    
    # Get the training data generator
    train_mat = load_npz(train_mat_path)
    train_data_gen = RecommendationGPTTrainGeneratorBatch(tokenizer, train_mat)

    # Get the validation data generator
    val_mat = load_npz(val_mat_path)
    val_data_gen = RecommendationGPTTestGeneratorBatch(tokenizer, train_mat, val_mat)

    print("Success!")
    print("-----End Obtaining the Collaborative Data Generator-----\n")

    '''
        Extend the config of the original GPT model
    '''
    print("-----Begin Setting Up the Config-----")
    config = GPT2Config(**_config)
    config.num_users = num_users
    config.num_items = num_items
    print("[Config] User: ",num_users)
    print("[Config] User: ",num_items)

    print("Success!")
    print("-----End Setting Up the Config-----\n")

    '''
        Instantiate the pretrained GPT2 model
    '''
    print("-----Begin Instantiating the Pretrained GPT Model-----")
    gpt2model = GPT2Model(config)
    pretrained_weights_path = os.path.join(model_root, "pytorch_model.bin")
    
    # Check if the pretrained weights file exists
    if not os.path.exists(pretrained_weights_path):
        print(f"Pretrained weights not found at {pretrained_weights_path}")
        print("You need to download the pretrained GPT2 model!")
        return
    
    gpt2model.load_state_dict(torch.load(pretrained_weights_path, map_location=device), strict=False)
    print("Success!")
    print("-----End Instantiating the Pretrained GPT Model-----\n")
    
    '''
        Create directories for model outputs
    '''
    content_model_dir = os.path.join(local_root, dataset, "content")
    rec_model_dir = os.path.join(local_root, dataset, "rec")
    collaborative_model_dir = os.path.join(local_root, dataset, "collaborative")
    
    os.makedirs(content_model_dir, exist_ok=True)
    os.makedirs(rec_model_dir, exist_ok=True)
    os.makedirs(collaborative_model_dir, exist_ok=True)
    
    '''
        Instantiate the GPT for recommendation content model
    '''
    print("-----Begin Instantiating the Content GPT Model-----")
    content_base_model = GPT4RecommendationBaseModel(config, gpt2model)
    
    # Paths for pretrained embeddings
    pretrained_user_emb_path = os.path.join(content_model_dir, f"user_embeddings_{args.lambda_V}.pt") 
    pretrained_item_emb_path = os.path.join(content_model_dir, f"item_embeddings_{args.lambda_V}.pt") 
    print("User Embedding Path: ", pretrained_user_emb_path)
    # Check if we have pretrained content embeddings and load them if available
    if os.path.exists(pretrained_user_emb_path) and os.path.exists(pretrained_item_emb_path):
        print("Loaded User checkpoint shape:", torch.load(pretrained_user_emb_path, map_location=device)['weight'].shape)
        content_base_model.user_embeddings.load_state_dict(
            torch.load(pretrained_user_emb_path, map_location=device))
        print("Load pretrained user embeddings: Success!")
        
        content_base_model.item_embeddings.load_state_dict(
            torch.load(pretrained_item_emb_path, map_location=device))
        print("Load pretrained item embeddings: Success!")
    else:
        print("No pretrained content embeddings found. Will train from scratch.")

    content_model = ContentGPTForUserItemWithLMHeadBatch(config, content_base_model)
    print("Success!")
    print("-----End Instantiating the Content GPT Model-----\n")
    
    '''
        Instantiate the GPT for recommendation model
    '''
    print("-----Begin Instantiating the Collaborative GPT Model-----")
    base_model = GPT4RecommendationBaseModel(config, gpt2model)

    # Paths for pretrained embeddings
    collab_user_emb_path = os.path.join(collaborative_model_dir, f"user_embeddings_{args.lambda_V}.pt") 
    collab_item_emb_path = os.path.join(collaborative_model_dir, f"item_embeddings_{args.lambda_V}.pt") 
    
    # Check if we have pretrained collaborative embeddings and load them if available
    if os.path.exists(collab_user_emb_path) and os.path.exists(collab_item_emb_path):
        base_model.user_embeddings.load_state_dict(
            torch.load(collab_user_emb_path, map_location=device))
        print("Load pretrained user embeddings: Success!")
        
        base_model.item_embeddings.load_state_dict(
            torch.load(collab_item_emb_path, map_location=device))
        print("Load pretrained item embeddings: Success!")
    else:
        print("No pretrained collaborative embeddings found. Will train from scratch.")

    rec_model = CollaborativeGPTwithItemRecommendHead(config, base_model)
    print("Success!")
    print("-----End Instantiating the Collaborative GPT Model-----\n")

    '''
        Freeze the parameters of the pretrained GPT2 for content model
    '''
    for name, param in rec_model.named_parameters():
        # we allow only user/item token embeddings to be trained
        if ('user_embeddings' not in name) and \
           ('item_embeddings' not in name):
            param.requires_grad = False

    print("-----Trainable Parameters-----")
    for name, param in rec_model.named_parameters():
        if param.requires_grad:
            print("{} : {}".format(name, param.shape))
    
    print("\n-----Non-trainable Parameters-----")
    for name, param in rec_model.named_parameters():
        if not param.requires_grad:
            print("{} : {}".format(name, param.shape))

    '''
        Set up the training details
    '''
    print("-----Begin Setting Up the Training Details-----")
    learning_rate = 1e-4
    batch_size = 20
    val_batch_size = 256
    num_epochs = 5 #150

    '''
        Create the DataLoaders
    '''
    print("-----Begin Creating the DataLoader-----")

    # Create the training data loader
    train_data_loader = DataLoader(train_data_gen, 
                                   batch_size=batch_size, 
                                   collate_fn=train_data_gen.collate_fn)

    # Create the validation data loader
    val_data_loader = DataLoader(val_data_gen, 
                                 batch_size=val_batch_size, 
                                 collate_fn=val_data_gen.collate_fn)
    
    # Create the review data loader with the custom collate_fn
    review_data_loader = DataLoader(review_data_gen, 
                                    batch_size=batch_size, 
                                    collate_fn=review_data_gen.collate_fn)
    print("-----End Creating the DataLoader-----\n")

    # Set the model to the training mode
    rec_model.to(device)
    content_model.to(device)
    content_model.train()

    # Obtain the optimizer
    optimizer = optim.Adam(rec_model.parameters(), 
                           lr=learning_rate)
    
    review_optimizer = optim.Adam(content_model.parameters(), 
                                  lr=learning_rate)

    # Initialize best metrics
    review_best_loss = float('inf')
    best_val_rec_loss = float('inf')
    best_recall_20 = -float('inf')
    best_recall_40 = -float('inf')
    best_NDCG_100 = -float('inf')
    best_sum = -float('inf')

    print(f"Rec model weights will be saved to {rec_model_dir}!")
    print(f"Content model weights will be saved to {content_model_dir}!")
    print("-----End Setting Up the Training Details-----\n")

    '''
        Define the training loop
    '''
    print("-----Begin Rec GPT Training Loop-----")
    for epoch in range(num_epochs):
        # Set the model to the training mode
        rec_model.train()
        train_rec_loss = 0
        regularize_total_loss = 0 
        
        # Progress bar for recommendation training
        progress_bar = tqdm(train_data_loader, desc=f"Epoch {epoch + 1} - Rec Training", ncols=100)
        for input_ids, target_mat, attention_mask, input_ids_main in progress_bar:
            optimizer.zero_grad()

            # Move tensors to the correct device
            input_ids = input_ids.to(device)
            target_mat = target_mat.to(device)
            attention_mask = attention_mask.to(device)
            input_ids_main = input_ids_main.to(device)

            # Get content embeddings from content model
            with torch.no_grad():
                content_embeds = torch.cat(
                    (content_model.base_model.embed(input_ids),
                     content_model.base_model.embed(input_ids_main)),
                    axis=1
                ).to(device)

            # Forward pass
            outputs = rec_model(input_ids, 
                                target_mat, 
                                attention_mask=attention_mask,
                                regularize=True,
                                lambda_V=lambda_V,
                                main_ids=input_ids_main,
                                content_embeds=content_embeds)
            rec_loss = outputs[0]
            regularize_loss = outputs[1]

            # Backward pass and optimization
            rec_loss.backward()
            optimizer.step()

            train_rec_loss += rec_loss.item()
            regularize_total_loss += regularize_loss.item()
            progress_bar.set_postfix({"Rec Loss": rec_loss.item()})

        # Calculate average losses
        train_rec_loss = train_rec_loss / len(train_data_loader)
        regularize_average_loss = regularize_total_loss / len(train_data_loader)
        
        print(f"Epoch {epoch + 1} - Rec Loss: {train_rec_loss:.4f}")
        print(f"Epoch {epoch + 1} - Average Regularize Loss: {regularize_average_loss:.4f}")

        # Set the model to evaluation mode
        rec_model.eval()  
        val_rec_loss = 0
        cur_recall_20 = 0
        cur_recall_40 = 0
        cur_NDCG_100 = 0

        # Progress bar for validation
        progress_bar = tqdm(val_data_loader, desc=f"Epoch {epoch + 1} - Validation", ncols=100)
        with torch.no_grad():
            for input_ids, train_mat, target_mat, attention_mask in progress_bar:
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
                target_mat = target_mat.cpu().numpy()
                item_scores = item_scores.cpu().numpy()
                val_rec_loss += rec_loss.item()
                cur_recall_20 += Recall_at_k(target_mat, item_scores, k=20, agg="sum")
                cur_recall_40 += Recall_at_k(target_mat, item_scores, k=40, agg="sum")
                cur_NDCG_100 += NDCG_at_k(target_mat, item_scores, k=100, agg="sum")

        # Calculate average metrics for the validation set
        val_rec_loss /= len(val_data_loader)
        cur_recall_20 /= len(val_data_gen)
        cur_recall_40 /= len(val_data_gen)
        cur_NDCG_100 /= len(val_data_gen)
        cur_sum = cur_recall_20 + cur_recall_40 + cur_NDCG_100
    
        # Update the best metrics
        if val_rec_loss < best_val_rec_loss:
            best_val_rec_loss = val_rec_loss
        if cur_recall_20 > best_recall_20:
            best_recall_20 = cur_recall_20
        if cur_recall_40 > best_recall_40:
            best_recall_40 = cur_recall_40
        if cur_NDCG_100 > best_NDCG_100:
            best_NDCG_100 = cur_NDCG_100
            
        # Save best model based on sum of metrics
        if cur_sum > best_sum:
            best_sum = cur_sum
            # Save user embeddings
            user_emb_path = os.path.join(rec_model_dir, f"user_embeddings_{args.lambda_V}.pt")
            torch.save(rec_model.base_model.user_embeddings.state_dict(), user_emb_path)

            # Save item embeddings
            item_emb_path = os.path.join(rec_model_dir, f"item_embeddings_{args.lambda_V}.pt")
            torch.save(rec_model.base_model.item_embeddings.state_dict(), item_emb_path)
            print(f"Saved best rec model to {rec_model_dir}")

        print(f"Train Rec Loss: {train_rec_loss:.4f}")
        print(f"Val Rec Loss: {val_rec_loss:.4f} / Best Val Rec Loss: {best_val_rec_loss:.4f}")
        print(f"Cur Recall@20: {cur_recall_20:.4f} / Best Recall@20: {best_recall_20:.4f}")
        print(f"Cur Recall@40: {cur_recall_40:.4f} / Best Recall@40: {best_recall_40:.4f}")
        print(f"Cur NDCG@100: {cur_NDCG_100:.4f} / Best NDCG@100: {best_NDCG_100:.4f}")    
    
        # Training content model
        review_total_loss = 0
        regularize_total_loss = 0
        
        # Progress bar for content model training
        progress_bar = tqdm(review_data_loader, desc=f"Epoch {epoch + 1} - Content Training", ncols=100)
        for input_ids_prompt, input_ids_main, attention_mask in progress_bar:
            review_optimizer.zero_grad()

            # Move tensors to the correct device
            input_ids_prompt = input_ids_prompt.to(device)
            input_ids_main = input_ids_main.to(device)
            attention_mask = attention_mask.to(device)

            # Get embeddings from recommendation model
            with torch.no_grad():
                rec_embeds = rec_model.base_model.embed(input_ids_prompt).to(device)
                
            # Forward pass of the content GPT
            outputs = content_model(input_ids_prompt, 
                                    input_ids_main, 
                                    labels_main=input_ids_main,
                                    attention_mask=attention_mask,
                                    regularize=True,
                                    lambda_V=lambda_V,
                                    collaborative_embeds=rec_embeds)
            review_loss = outputs[0]
            regularize_loss = outputs[1]

            # Backward pass and optimization
            review_loss.backward()
            review_optimizer.step()

            review_total_loss += review_loss.item()
            regularize_total_loss += regularize_loss.item()
            progress_bar.set_postfix({"Review Loss": review_loss.item(),
                                     "Regularize Loss": regularize_loss.item()})

        # Calculate average losses
        review_average_loss = review_total_loss / len(review_data_loader)
        regularize_average_loss = regularize_total_loss / len(review_data_loader)
        
        print(f"Epoch {epoch + 1} - Review Average Loss: {review_average_loss:.4f}")
        print(f"Epoch {epoch + 1} - Average Regularize Loss: {regularize_average_loss:.4f}")

        # Save best content model
        if review_average_loss < review_best_loss:
            review_best_loss = review_average_loss

            # Save user embeddings
            user_emb_path = os.path.join(content_model_dir, f"user_embeddings_{args.lambda_V}.pt") 
            torch.save(content_model.base_model.user_embeddings.state_dict(), user_emb_path)
            
            # Save item embeddings
            item_emb_path = os.path.join(content_model_dir, f"item_embeddings_{args.lambda_V}.pt")
            torch.save(content_model.base_model.item_embeddings.state_dict(), item_emb_path)
            print(f"Saved best content model to {content_model_dir}")
        
    print("-----End Rec GPT Training Loop-----")


if __name__ == "__main__":
    main()