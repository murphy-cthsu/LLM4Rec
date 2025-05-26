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
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from scipy.sparse import load_npz
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

sys.path.append("src/libs")
from tokenizer import TokenizerWithUserItemIDTokensBatch

from data import CollaborativeGPTGeneratorBatch
from data import UserItemContentGPTDatasetBatch

from model import GPT4RecommendationBaseModel
from model import CollaborativeGPTwithItemLMHeadBatch
from model import ContentGPTForUserItemWithLMHeadBatch

# Configuration for local paths
local_root = "checkpoints"
if not os.path.exists(local_root):
    os.makedirs(local_root, exist_ok=True)

def main():
    # Use regular device setup rather than Accelerator
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
    parser.add_argument("--share_base_model", action="store_true",
        help="Share base model between content and collaborative models to save memory")
    args = parser.parse_args()
    
    dataset = args.dataset
    lambda_V = args.lambda_V
    data_path = args.data_path
    pretrained_path = args.pretrained_path
    model_name = args.model_name
    use_half_precision = args.use_half_precision
    share_base_model = args.share_base_model
    
    print("-----Current Setting-----")
    print(f"dataset: {dataset}")
    print(f"lambda_V: {lambda_V}")
    print(f"data_path: {data_path}")
    print(f"pretrained_path: {pretrained_path}")
    print(f"model_name: {model_name}")
    print(f"use_half_precision: {use_half_precision}")
    print(f"share_base_model: {share_base_model}")
    
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
        
        # Test the tokenizer with a sample
        test_text = "<user_0> rated <item_1> with 5 stars. Great product!"
        tokens = extended_tokenizer._tokenize(test_text)
        print(f"Sample tokenization: {test_text}")
        print(f"Tokens: {tokens[:10]}...")  # Show first 10 tokens
        
    except Exception as e:
        print(f"Error creating extended tokenizer: {e}")
        return
    print("=== Debugging Tokenizer ===")
    print(f"PAD token ID: {extended_tokenizer.tokenizer.pad_token_id}")
    print(f"EOS token ID: {extended_tokenizer.tokenizer.eos_token_id}")

    # Check item token range
    item_0_token = extended_tokenizer._tokenize("<item_0>")[0]
    item_last_token = extended_tokenizer._tokenize(f"<item_{num_items-1}>")[0]
    print(f"Item token range: [{item_0_token}, {item_last_token}]")

    # Test what 151643 is
    try:
        decoded = extended_tokenizer.tokenizer.decode([151643])
        print(f"Token 151643 = '{decoded}'")
    except:
        print("Token 151643 is invalid")
        print("-----End Creating Extended Tokenizer-----\n")

    '''
        Define the review data generator
    '''
    print("-----Begin Obtaining the Review Data Generator-----")
    review_path = os.path.join(data_root, "user_item_texts", "review.pkl")
    print(f"Loading data from {review_path}...")
    if not os.path.exists(review_path):
        raise FileNotFoundError(f"Review data not found at {review_path}")
    
    review_data_gen = UserItemContentGPTDatasetBatch(extended_tokenizer, review_path)
    print("Success!")
    print("-----End Obtaining the Review Data Generator-----\n")

    '''
        Now we deal with the user/item interaction data
    '''
    print("-----Begin Obtaining the Collaborative Data Generator-----")
    train_mat_path = os.path.join(data_root, "train_matrix.npz")
    print(f"Loading data from {train_mat_path}...")
    if not os.path.exists(train_mat_path):
        raise FileNotFoundError(f"Train matrix not found at {train_mat_path}")
    
    train_mat = load_npz(train_mat_path)
    collaborative_data_gen = CollaborativeGPTGeneratorBatch(extended_tokenizer, train_mat)
    print("Success!")
    print("-----End Obtaining the Collaborative Data Generator-----\n")

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
        Instantiate the GPT for recommendation content model
    '''
    print("-----Begin Instantiating the Content GPT Model-----")
    
    # The model resizing is now handled inside GPT4RecommendationBaseModel
    content_base_model = GPT4RecommendationBaseModel(config, qwen_model)
    content_model = ContentGPTForUserItemWithLMHeadBatch(config, content_base_model)
    print("Success!")
    print("-----End Instantiating the Content GPT Model-----\n")

    '''
        Instantiate the GPT for recommendation collaborative model
    '''
    print("-----Begin Instantiating the Collaborative GPT Model-----")
    # Option to share base model to save memory
    if share_base_model:
        print("🔄 Sharing base model between content and collaborative models")
        collaborative_base_model = content_base_model
    else:
        print("🆕 Creating separate base model for collaborative model")
        collaborative_base_model = GPT4RecommendationBaseModel(config, qwen_model)
    
    collaborative_model = CollaborativeGPTwithItemLMHeadBatch(config, collaborative_base_model)
    print("Success!")
    print("-----End Instantiating the Collaborative GPT Model-----\n")
    
    '''
        Set up gradients properly for both models
    '''
    print("-----Begin Setting Up Gradients-----")
    
    def setup_model_gradients(model, model_name):
        """Setup gradients for a recommendation model"""
        print(f"🔧 Setting up gradients for {model_name}...")
        
        # First, freeze everything
        for param in model.parameters():
            param.requires_grad = False
        
        # Enable gradients for specific components
        try:
            # User and item embeddings (always trainable)
            model.base_model.user_embeddings.weight.requires_grad = True
            model.base_model.item_embeddings.weight.requires_grad = True
            print(f"  ✅ User/Item embeddings enabled")
            
            # Input embeddings (extended vocabulary)
            model.base_model.qwen_model.get_input_embeddings().weight.requires_grad = True
            print(f"  ✅ Input embeddings enabled")
            
            # Model-specific heads
            if hasattr(model, 'lm_head'):
                model.lm_head.weight.requires_grad = True
                print(f"  ✅ LM head enabled")
            
            if hasattr(model, 'item_head'):
                model.item_head.weight.requires_grad = True
                print(f"  ✅ Item head enabled")
            
        except Exception as e:
            print(f"  ❌ Error setting up gradients: {e}")
            return False
        
        # Count trainable parameters
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_count = sum(p.numel() for p in model.parameters())
        
        print(f"  📊 Trainable: {trainable_count:,} / {total_count:,} ({100*trainable_count/total_count:.2f}%)")
        
        return trainable_count > 0
    
    # Setup gradients for both models
    content_success = setup_model_gradients(content_model, "Content Model")
    collaborative_success = setup_model_gradients(collaborative_model, "Collaborative Model")
    
    if not (content_success and collaborative_success):
        print("❌ Failed to setup gradients properly")
        return
    
    print("✅ Gradient setup completed successfully!")
    print("-----End Setting Up Gradients-----\n")
    
    '''
        Set up the training details
    '''
    print("-----Begin Setting Up the Training Details-----")
    # Adjust learning rate and batch size for Qwen3
    learning_rate = 1e-3  # Slightly lower learning rate for larger model
    batch_size = 8 if use_half_precision else 8 # Smaller batch size for larger model
    num_pretrained_epochs = 2
    num_epochs = 2
    
    # Add gradient accumulation for effective larger batch size
    gradient_accumulation_steps = 4

    '''
        Create data loaders
    '''
    print("-----Begin Creating the DataLoader-----")
    # Create the review data loader with the custom collate_fn
    # Set num_workers=0 to avoid potential multiprocessing issues with extended tokenizer
    review_data_loader = DataLoader(review_data_gen, 
                                   batch_size=batch_size, 
                                   collate_fn=review_data_gen.collate_fn,
                                   shuffle=True,
                                   num_workers=0)

    # Create the collaborative data loader with the custom collate_fn
    collaborative_data_loader = DataLoader(collaborative_data_gen, 
                                          batch_size=batch_size, 
                                          collate_fn=collaborative_data_gen.collate_fn,
                                          shuffle=True,
                                          num_workers=0)
    
    print(f"Review batches per epoch: {len(review_data_loader)}")
    print(f"Collaborative batches per epoch: {len(collaborative_data_loader)}")
    print("-----End Creating the DataLoader-----\n")
    
    # Set the model to the training mode and move to device
    content_model.train()
    content_model.to(device)

    collaborative_model.train()
    collaborative_model.to(device)

    # Obtain the optimizer with weight decay for regularization
    review_optimizer = optim.AdamW(
        [p for p in content_model.parameters() if p.requires_grad], 
        lr=learning_rate,
        weight_decay=0.01
    )

    collaborative_optimizer = optim.AdamW(
        [p for p in collaborative_model.parameters() if p.requires_grad], 
        lr=learning_rate,
        weight_decay=0.01
    )
    
    # Add learning rate scheduler
    review_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        review_optimizer, T_max=num_pretrained_epochs + num_epochs
    )
    collaborative_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        collaborative_optimizer, T_max=num_epochs
    )
    
    # Initialize best_loss with infinity
    review_best_loss = float('inf')
    collaborative_best_loss = float('inf')

    # The place to save the model weights
    model_root = os.path.join(local_root, "models", dataset)
    content_model_root = os.path.join(model_root, "content")
    collaborative_model_root = os.path.join(model_root, "collaborative")
    
    # Create directories if they don't exist
    os.makedirs(content_model_root, exist_ok=True)
    os.makedirs(collaborative_model_root, exist_ok=True)
    
    print(f"Content model weights will be saved to {content_model_root}!")
    print(f"Collaborative model weights will be saved to {collaborative_model_root}!")
    print("-----End Setting Up the Training Details-----\n")

    '''
        Define the pretraining loop for the content GPT
    '''
    print("-----Begin Content GPT Pretraining Loop-----")
    for epoch in range(num_pretrained_epochs):
        review_total_loss = 0
        
        # Initialize tqdm progress bar
        progress_bar = tqdm(review_data_loader, 
                           desc=f"Content Pretrain Epoch {epoch + 1}", 
                           ncols=100)
        
        for batch_idx, (input_ids_prompt, input_ids_main, attention_mask) in enumerate(progress_bar):
            # Obtain the data
            input_ids_prompt = input_ids_prompt.to(device)
            input_ids_main = input_ids_main.to(device)
            attention_mask = attention_mask.to(device)

            try:
                # Forward pass
                outputs = content_model(input_ids_prompt, 
                                       input_ids_main, 
                                       labels_main=input_ids_main,
                                       attention_mask=attention_mask)
                review_loss = outputs[0]
                print("[Debug] Review Loss", review_loss)
                # Verify loss has gradients
                if not review_loss.requires_grad:
                    print(f"⚠️  Batch {batch_idx}: Loss has no gradients, skipping...")
                    continue
                
                # Scale loss for gradient accumulation
                review_loss = review_loss / gradient_accumulation_steps
                
                # Backward pass
                review_loss.backward()
                
                # Update weights every gradient_accumulation_steps
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Clip gradients to prevent explosion
                    torch.nn.utils.clip_grad_norm_(content_model.parameters(), max_norm=1.0)
                    review_optimizer.step()
                    review_optimizer.zero_grad()

                review_total_loss += review_loss.item() * gradient_accumulation_steps
                progress_bar.set_postfix({
                    "Loss": f"{review_loss.item() * gradient_accumulation_steps:.4f}",
                    "LR": f"{review_optimizer.param_groups[0]['lr']:.2e}"
                })
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM error at batch {batch_idx}, skipping...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

        review_average_loss = review_total_loss / len(review_data_loader)
        print(f"Epoch {epoch + 1} - Review Average Loss: {review_average_loss:.4f}")
        
        # Step scheduler
        review_scheduler.step()

        # Save best model
        if review_average_loss < review_best_loss:
            review_best_loss = review_average_loss

            # Save user embeddings
            user_emb_path = os.path.join(content_model_root, f"user_embeddings_{lambda_V}.pt")
            torch.save(content_model.base_model.user_embeddings.state_dict(), user_emb_path)
            print("Content Model Info: ", content_model.base_model.user_embeddings.weight.shape)
            
            # Save item embeddings
            item_emb_path = os.path.join(content_model_root, f"item_embeddings_{lambda_V}.pt")
            torch.save(content_model.base_model.item_embeddings.state_dict(), item_emb_path)
            
            print(f"Saved best content model with loss: {review_best_loss:.4f}")
            
    print("-----End Content GPT Pretraining Loop-----\n")

    '''
        Iteratively training the collaborative and content GPT model for recommendations
    '''
    print("-----Begin the Iterative Training Loop-----")
    for epoch in range(num_epochs):
        '''
            Optimize the collaborative GPT model
        '''
        collaborative_total_loss = 0
        regularize_total_loss = 0
        
        progress_bar = tqdm(collaborative_data_loader, 
                           desc=f"Epoch {epoch + 1} - Collaborative", 
                           ncols=120)
        
        for batch_idx, (input_ids_prompt, input_ids_main, attention_mask) in enumerate(progress_bar):
            input_ids_prompt = input_ids_prompt.to(device)
            input_ids_main = input_ids_main.to(device)
            attention_mask = attention_mask.to(device)

            try:
                # Get content embeddings without gradients
                with torch.no_grad():
                    content_embeds = torch.cat(
                        (content_model.base_model.embed(input_ids_prompt),
                         content_model.base_model.embed(input_ids_main)),
                        axis=1
                    ).to(device)
                    
                # Forward pass of the collaborative GPT
                labels_cleaned = input_ids_main.clone()
                labels_cleaned[labels_cleaned == 151643] = -100  # Replace padding token
                outputs = collaborative_model(input_ids_prompt, 
                                             input_ids_main, 
                                             labels_main=labels_cleaned,
                                             attention_mask=attention_mask,
                                             regularize=True,
                                             lambda_V=lambda_V,
                                             content_embeds=content_embeds)
                collaborative_loss = outputs[0]
                regularize_loss = outputs[1]
                
                # Verify loss has gradients
                if not collaborative_loss.requires_grad:
                    print(f"⚠️  Batch {batch_idx}: Collaborative loss has no gradients, skipping...")
                    continue
                
                # Scale loss for gradient accumulation
                collaborative_loss = collaborative_loss / gradient_accumulation_steps

                # Backward pass
                collaborative_loss.backward()
                
                # Update weights every gradient_accumulation_steps
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(collaborative_model.parameters(), max_norm=1.0)
                    collaborative_optimizer.step()
                    collaborative_optimizer.zero_grad()
                
                collaborative_total_loss += collaborative_loss.item() * gradient_accumulation_steps
                regularize_total_loss += regularize_loss.item()
                
                progress_bar.set_postfix({
                    "Collab": f"{collaborative_loss.item() * gradient_accumulation_steps:.4f}",
                    "Reg": f"{regularize_loss.item():.4f}"
                })
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM error in collaborative training at batch {batch_idx}, skipping...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        collaborative_average_loss = collaborative_total_loss / len(collaborative_data_loader)
        regularize_average_loss = regularize_total_loss / len(collaborative_data_loader)
        
        print(f"Epoch {epoch + 1} - Average Collaborative Loss: {collaborative_average_loss:.4f}")
        print(f"Epoch {epoch + 1} - Average Regularize Loss: {regularize_average_loss:.4f}")
        
        # Step scheduler
        collaborative_scheduler.step()
        
        # Save best collaborative model
        if collaborative_average_loss < collaborative_best_loss:
            collaborative_best_loss = collaborative_average_loss

            user_emb_path = os.path.join(collaborative_model_root, f"user_embeddings_{lambda_V}.pt")
            torch.save(collaborative_model.base_model.user_embeddings.state_dict(), user_emb_path)

            item_emb_path = os.path.join(collaborative_model_root, f"item_embeddings_{lambda_V}.pt")
            torch.save(collaborative_model.base_model.item_embeddings.state_dict(), item_emb_path)
            
            print(f"Saved best collaborative model with loss: {collaborative_best_loss:.4f}")

        '''
            Optimize the content GPT model
        '''
        review_total_loss = 0
        regularize_total_loss = 0
        
        progress_bar = tqdm(review_data_loader, 
                           desc=f"Epoch {epoch + 1} - Content", 
                           ncols=120)
        
        for batch_idx, (input_ids_prompt, input_ids_main, attention_mask) in enumerate(progress_bar):
            input_ids_prompt = input_ids_prompt.to(device)
            input_ids_main = input_ids_main.to(device)
            attention_mask = attention_mask.to(device)

            try:
                # Get collaborative embeddings without gradients
                with torch.no_grad():
                    collaborative_embeds = collaborative_model.base_model.embed(input_ids_prompt).to(device)
                    
                # Forward pass of the content GPT
                outputs = content_model(input_ids_prompt, 
                                       input_ids_main, 
                                       labels_main=input_ids_main,
                                       attention_mask=attention_mask,
                                       regularize=True,
                                       lambda_V=lambda_V,
                                       collaborative_embeds=collaborative_embeds)
                review_loss = outputs[0]
                regularize_loss = outputs[1]
                
                # Verify loss has gradients
                if not review_loss.requires_grad:
                    print(f"⚠️  Batch {batch_idx}: Content loss has no gradients, skipping...")
                    continue
                
                # Scale loss for gradient accumulation
                review_loss = review_loss / gradient_accumulation_steps

                # Backward pass
                review_loss.backward()
                
                # Update weights every gradient_accumulation_steps
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(content_model.parameters(), max_norm=1.0)
                    review_optimizer.step()
                    review_optimizer.zero_grad()

                review_total_loss += review_loss.item() * gradient_accumulation_steps
                regularize_total_loss += regularize_loss.item()
                
                progress_bar.set_postfix({
                    "Review": f"{review_loss.item() * gradient_accumulation_steps:.4f}",
                    "Reg": f"{regularize_loss.item():.4f}"
                })
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM error in content training at batch {batch_idx}, skipping...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

        review_average_loss = review_total_loss / len(review_data_loader)
        regularize_average_loss = regularize_total_loss / len(review_data_loader)
        
        print(f"Epoch {epoch + 1} - Review Average Loss: {review_average_loss:.4f}")
        print(f"Epoch {epoch + 1} - Content Regularize Loss: {regularize_average_loss:.4f}")

        # Step scheduler
        review_scheduler.step()

        # Save best content model
        if review_average_loss < review_best_loss:
            review_best_loss = review_average_loss

            user_emb_path = os.path.join(content_model_root, f"user_embeddings_{lambda_V}.pt") 
            torch.save(content_model.base_model.user_embeddings.state_dict(), user_emb_path)
            print("Content Model Info: ", content_model.base_model.user_embeddings.weight.shape)
            
            item_emb_path = os.path.join(content_model_root, f"item_embeddings_{lambda_V}.pt")
            torch.save(content_model.base_model.item_embeddings.state_dict(), item_emb_path)
            
            print(f"Saved best content model with loss: {review_best_loss:.4f}")

    print("-----End Iterative Training Loop-----")
    print(f"Training completed!")
    print(f"Best content loss: {review_best_loss:.4f}")
    print(f"Best collaborative loss: {collaborative_best_loss:.4f}")

if __name__ == "__main__":
    main()