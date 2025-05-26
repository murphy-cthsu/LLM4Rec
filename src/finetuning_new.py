'''
MIT License
Copyright (c) 2024 Yaochen Zhu
Modified for Qwen3 single GPU execution
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

# Define local directories
data_root = "data"  # Local data directory
local_root = "checkpoints"  # Temporary directory

# Create necessary directories
os.makedirs(data_root, exist_ok=True)
os.makedirs(local_root, exist_ok=True)

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
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B",
        help="Qwen model name or path")
    args = parser.parse_args()
    
    model_root = args.model_path
    dataset = args.dataset
    lambda_V = float(args.lambda_V)
    model_name = args.model_name
    os.makedirs(model_root, exist_ok=True)

    print("-----Current Setting-----")
    print(f"dataset: {dataset}")
    print(f"lambda_V: {args.lambda_V}")
    print(f"model_name: {model_name}")
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
        Load Qwen model and create extended tokenizer
    '''
    print("-----Begin Loading Qwen Model and Tokenizer-----")
    try:
        # Load tokenizer
        base_tokenizer = AutoTokenizer.from_pretrained(model_name)
        if base_tokenizer.pad_token is None:
            base_tokenizer.pad_token = base_tokenizer.eos_token
        
        # Load model
        qwen_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=None  # We'll move to device manually
        )
        
        print(f"Successfully loaded {model_name}")
        print(f"Original vocab size: {qwen_model.config.vocab_size}")
        
    except Exception as e:
        print(f"Error loading Qwen model: {e}")
        return
    print("-----End Loading Qwen Model and Tokenizer-----\n")

    '''
        Create extended tokenizer with user/item tokens
    '''
    print("-----Begin Creating Extended Tokenizer-----")
    try:
        extended_tokenizer = TokenizerWithUserItemIDTokensBatch(
            model_name,
            num_users,
            num_items
        )
        
        print(f"Successfully created extended tokenizer")
        print(f"Extended vocab size: {len(extended_tokenizer.tokenizer)}")
        
        # Debug tokenizer setup
        print("=== Tokenizer Debug Info ===")
        print(f"PAD token ID: {extended_tokenizer.tokenizer.pad_token_id}")
        print(f"EOS token ID: {extended_tokenizer.tokenizer.eos_token_id}")
        
        # Check item token range
        item_0_token = extended_tokenizer._tokenize("<item_0>")[0]
        item_0_id = extended_tokenizer.tokenizer.convert_tokens_to_ids([item_0_token])[0]
        item_last_token = extended_tokenizer._tokenize(f"<item_{num_items-1}>")[0]
        item_last_id = extended_tokenizer.tokenizer.convert_tokens_to_ids([item_last_token])[0]
        
        print(f"<item_0> token: '{item_0_token}' -> ID: {item_0_id}")
        print(f"<item_{num_items-1}> token: '{item_last_token}' -> ID: {item_last_id}")
        print(f"Item token ID range: [{item_0_id}, {item_last_id}]")
        
        # Store these for later use
        extended_tokenizer.item_token_start_id = item_0_id
        extended_tokenizer.user_token_start_id = item_0_id - num_users  # Assuming users come before items
        
    except Exception as e:
        print(f"Error creating extended tokenizer: {e}")
        return
    print("-----End Creating Extended Tokenizer-----\n")
    
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
    
    review_data_gen = UserItemContentGPTDatasetBatch(extended_tokenizer, review_path)
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
    train_data_gen = RecommendationGPTTrainGeneratorBatch(extended_tokenizer, train_mat)

    # Get the validation data generator
    val_mat = load_npz(val_mat_path)
    val_data_gen = RecommendationGPTTestGeneratorBatch(extended_tokenizer, train_mat, val_mat)

    print("Success!")
    print("-----End Obtaining the Collaborative Data Generator-----\n")

    '''
        Extend the config of the original Qwen model
    '''
    print("-----Begin Setting Up the Config-----")
    config = qwen_model.config
    config.num_users = num_users
    config.num_items = num_items
    # Update vocab_size to include the extended vocabulary
    config.vocab_size = len(extended_tokenizer.tokenizer)
    
    print(f"[Config] Users: {num_users}")
    print(f"[Config] Items: {num_items}")
    print(f"[Config] Extended vocab size: {config.vocab_size}")
    print("Success!")
    print("-----End Setting Up the Config-----\n")
    
    '''
        Create directories for model outputs
    '''
    model_root= os.path.join(local_root, args.model_path)
    content_model_dir = os.path.join(model_root, dataset, "content")
    rec_model_dir = os.path.join(model_root, dataset, "rec")
    collaborative_model_dir = os.path.join(model_root, dataset, "collaborative")
    
    os.makedirs(content_model_dir, exist_ok=True)
    os.makedirs(rec_model_dir, exist_ok=True)
    os.makedirs(collaborative_model_dir, exist_ok=True)
    
    '''
        Instantiate the GPT for recommendation content model
    '''
    print("-----Begin Instantiating the Content GPT Model-----")
    content_base_model = GPT4RecommendationBaseModel(config, qwen_model)
    
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
    base_model = GPT4RecommendationBaseModel(config, qwen_model)

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
        Freeze the parameters of the pretrained Qwen for both models
    '''
    
    def setup_content_model_gradients(content_model):
        """
        Proper gradient setup for content model
        """
        print("🔧 Setting up content model gradients...")
        
        # First, freeze everything
        for param in content_model.parameters():
            param.requires_grad = False
        
        # Enable gradients for specific components
        # 1. User and item embeddings
        content_model.base_model.user_embeddings.weight.requires_grad = True
        content_model.base_model.item_embeddings.weight.requires_grad = True
        print("  ✅ User/Item embeddings enabled")
        
        # 2. LM head (create separate trainable head)
        if hasattr(content_model, 'lm_head'):
            # Recreate LM head as trainable
            embedding_dim = content_model.base_model.embedding_dim
            vocab_size = content_model.base_model.vocab_size
            
            # Create new trainable LM head
            content_model.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False, 
                                            dtype=content_model.base_model.model_dtype)
            
            # Initialize with existing embeddings
            input_embeddings = content_model.base_model.qwen_model.get_input_embeddings()
            with torch.no_grad():
                content_model.lm_head.weight.copy_(input_embeddings.weight)
            
            content_model.lm_head.weight.requires_grad = True
            print("  ✅ Separate trainable LM head created")
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in content_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in content_model.parameters())
        
        print(f"  📊 Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
        
        return trainable_params > 0
    
    # Setup gradients for recommendation model
    for name, param in rec_model.named_parameters():
        # we allow only user/item token embeddings to be trained
        if ('user_embeddings' not in name) and \
           ('item_embeddings' not in name):
            param.requires_grad = False

    print("-----Rec Model Trainable Parameters-----")
    rec_trainable_count = 0
    for name, param in rec_model.named_parameters():
        if param.requires_grad:
            print("{} : {}".format(name, param.shape))
            rec_trainable_count += param.numel()
    print(f"Total rec trainable parameters: {rec_trainable_count:,}")
    
    # Setup gradients for content model using the special function
    content_success = setup_content_model_gradients(content_model)
    
    print("\n-----Content Model Trainable Parameters-----")
    content_trainable_count = 0
    for name, param in content_model.named_parameters():
        if param.requires_grad:
            print("{} : {}".format(name, param.shape))
            content_trainable_count += param.numel()
    print(f"Total content trainable parameters: {content_trainable_count:,}")
    
    # Verify we have trainable parameters
    if rec_trainable_count == 0:
        print("❌ ERROR: No trainable parameters in rec model!")
        return
    if content_trainable_count == 0:
        print("❌ ERROR: No trainable parameters in content model!")
        return

    '''
        Set up the training details
    '''
    print("-----Begin Setting Up the Training Details-----")
    learning_rate = 1e-4
    batch_size = 8  # Smaller batch size for Qwen
    val_batch_size = 64  # Smaller validation batch size
    num_epochs = 5

    '''
        Create the DataLoaders
    '''
    print("-----Begin Creating the DataLoader-----")

    # Create the training data loader
    train_data_loader = DataLoader(train_data_gen, 
                                   batch_size=batch_size, 
                                   collate_fn=train_data_gen.collate_fn,
                                   num_workers=0)

    # Create the validation data loader
    val_data_loader = DataLoader(val_data_gen, 
                                 batch_size=val_batch_size, 
                                 collate_fn=val_data_gen.collate_fn,
                                 num_workers=0)
    
    # Create the review data loader with the custom collate_fn
    review_data_loader = DataLoader(review_data_gen, 
                                    batch_size=batch_size, 
                                    collate_fn=review_data_gen.collate_fn,
                                    num_workers=0)
    
    print(f"Train batches: {len(train_data_loader)}")
    print(f"Val batches: {len(val_data_loader)}")
    print(f"Review batches: {len(review_data_loader)}")
    print("-----End Creating the DataLoader-----\n")

    # Set the model to the training mode
    rec_model.to(device)
    content_model.to(device)
    content_model.train()

    # Obtain the optimizer with filtered parameters
    rec_trainable_params = [p for p in rec_model.parameters() if p.requires_grad]
    content_trainable_params = [p for p in content_model.parameters() if p.requires_grad]
    
    print(f"Rec optimizer will train {len(rec_trainable_params)} parameter groups")
    print(f"Content optimizer will train {len(content_trainable_params)} parameter groups")
    
    optimizer = optim.Adam(rec_trainable_params, lr=learning_rate)
    review_optimizer = optim.Adam(content_trainable_params, lr=learning_rate)

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

            try:
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

                # Check for NaN/Inf
                if torch.isnan(rec_loss) or torch.isinf(rec_loss):
                    print(f"⚠️  Invalid rec loss: {rec_loss}, skipping batch")
                    continue

                # Backward pass and optimization
                rec_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(rec_model.parameters(), max_norm=1.0)
                
                optimizer.step()

                train_rec_loss += rec_loss.item()
                regularize_total_loss += regularize_loss.item()
                progress_bar.set_postfix({"Rec Loss": rec_loss.item()})

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM error in rec training, skipping batch...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

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

                try:
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

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"OOM error in validation, skipping batch...")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e

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

            try:
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

                # Check if loss requires gradients
                if not review_loss.requires_grad:
                    print(f"⚠️  Review loss has no gradients, skipping batch")
                    print(f"   Loss value: {review_loss.item()}")
                    print(f"   Loss requires_grad: {review_loss.requires_grad}")
                    # Debug: Check if any content model parameters require grad
                    content_grad_params = [name for name, param in content_model.named_parameters() if param.requires_grad]
                    print(f"   Content model grad params: {content_grad_params}")
                    continue

                # Check for NaN/Inf
                if torch.isnan(review_loss) or torch.isinf(review_loss):
                    print(f"⚠️  Invalid review loss: {review_loss}, skipping batch")
                    continue

                # Backward pass and optimization
                review_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(content_model.parameters(), max_norm=1.0)
                
                review_optimizer.step()

                review_total_loss += review_loss.item()
                regularize_total_loss += regularize_loss.item()
                progress_bar.set_postfix({"Review Loss": review_loss.item(),
                                         "Regularize Loss": regularize_loss.item()})

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM error in content training, skipping batch...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

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
    print(f"Training completed!")
    print(f"Best content loss: {review_best_loss:.4f}")
    print(f"Best recall@20: {best_recall_20:.4f}")
    print(f"Best recall@40: {best_recall_40:.4f}")
    print(f"Best NDCG@100: {best_NDCG_100:.4f}")


if __name__ == "__main__":
    main()