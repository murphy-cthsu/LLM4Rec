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
from transformers import GPT2Model, GPT2Config
from transformers import GPT2Tokenizer

sys.path.append("src/libs")
from tokenizer import TokenizerWithUserItemIDTokensBatch

from data import CollaborativeGPTGeneratorBatch
from data import UserItemContentGPTDatasetBatch

from model import GPT4RecommendationBaseModel
from model import CollaborativeGPTwithItemLMHeadBatch
from model import ContentGPTForUserItemWithLMHeadBatch

# Configuration for local paths
local_root = "tmp"
if not os.path.exists(local_root):
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
    args = parser.parse_args()
    
    dataset = args.dataset
    lambda_V = args.lambda_V
    data_path = args.data_path
    pretrained_path = args.pretrained_path
    
    print("-----Current Setting-----")
    print(f"dataset: {dataset}")
    print(f"lambda_V: {lambda_V}")
    print(f"data_path: {data_path}")
    print(f"pretrained_path: {pretrained_path}")
    
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
        Obtain the tokenizer with user/item tokens
    '''
    print("-----Begin Obtaining the Tokenizer-----")
    # tokenizer_root = os.path.join(pretrained_path, "tokenizer")
    tokenizer_root=pretrained_path
    print(f"Loading pretrained tokenizer from {tokenizer_root}...")
    vocab_file = os.path.join(tokenizer_root, "vocab.json")
    merges_file = os.path.join(tokenizer_root, "merges.txt")
    
    # Check if files exist
    if not os.path.exists(vocab_file):
        raise FileNotFoundError(f"Vocabulary file not found at {vocab_file}")
    if not os.path.exists(merges_file):
        raise FileNotFoundError(f"Merges file not found at {merges_file}")
        
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
    review_path = os.path.join(data_root, "user_item_texts", "review.pkl")
    print(f"Loading data from {review_path}...")
    if not os.path.exists(review_path):
        raise FileNotFoundError(f"Review data not found at {review_path}")
    
    review_data_gen = UserItemContentGPTDatasetBatch(tokenizer, review_path)
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
    collaborative_data_gen = CollaborativeGPTGeneratorBatch(tokenizer, train_mat)
    print("Success!")
    print("-----End Obtaining the Collaborative Data Generator-----\n")

    '''
        Extend the config of the original GPT model
    '''
    print("-----Begin Setting Up the Config-----")
    config = GPT2Config(**_config)
    config.num_users = num_users
    config.num_items = num_items
    print("Success!")
    print("-----End Setting Up the Config-----\n")

    '''
        Instantiate the pretrained GPT2 model
    '''
    print("-----Begin Instantiating the Pretrained GPT Model-----")
    gpt2model = GPT2Model(config)
    gpt2_path = os.path.join(pretrained_path, "pytorch_model.bin")
    print(f"Loading pretrained weights from {gpt2_path}...")
    if not os.path.exists(gpt2_path):
        raise FileNotFoundError(f"Pretrained GPT2 model not found at {gpt2_path}")
    
    gpt2model.load_state_dict(torch.load(gpt2_path, map_location=device), strict=False)
    print("Success!")
    print("-----End Instantiating the Pretrained GPT Model-----\n")

    '''
        Instantiate the GPT for recommendation content model
    '''
    print("-----Begin Instantiating the Content GPT Model-----")
    content_base_model = GPT4RecommendationBaseModel(config, gpt2model)
    content_model = ContentGPTForUserItemWithLMHeadBatch(config, content_base_model)
    print("Success!")
    print("-----End Instantiating the Content GPT Model-----\n")

    '''
        Freeze the parameters of the pretrained GPT2 for content model
    '''
    for name, param in content_model.named_parameters():
        # we allow only user/item token embeddings to be trained
        if ('user_embeddings' not in name) and \
           ('item_embeddings' not in name):
            param.requires_grad = False

    print("-----Trainable Parameters-----")
    for name, param in content_model.named_parameters():
        if param.requires_grad:
            print("{} : {}".format(name, param.shape))
    
    print("\n-----Non-trainable Parameters-----")
    for name, param in content_model.named_parameters():
        if not param.requires_grad:
            print("{} : {}".format(name, param.shape))

    '''
        Instantiate the GPT for recommendation collaborative model
    '''
    print("-----Begin Instantiating the Collaborative GPT Model-----")
    collaborative_base_model = GPT4RecommendationBaseModel(config, gpt2model)
    collaborative_model = CollaborativeGPTwithItemLMHeadBatch(config, collaborative_base_model)
    print("Success!")
    print("-----End Instantiating the Collaborative GPT Model-----\n")

    '''
        Freeze the parameters of the pretrained GPT2 for collaborative model
    '''
    for name, param in collaborative_model.named_parameters():
        # we allow only user/item token embeddings to be trained
        if ('user_embeddings' not in name) and \
           ('item_embeddings' not in name):
            param.requires_grad = False

    print("-----Trainable Parameters-----")
    for name, param in collaborative_model.named_parameters():
        if param.requires_grad:
            print("{} : {}".format(name, param.shape))
        
    print("\n-----Non-Trainable Parameters-----")
    for name, param in collaborative_model.named_parameters():
        if not param.requires_grad:
            print("{} : {}".format(name, param.shape))

    '''
        Set up the training details
    '''
    print("-----Begin Setting Up the Training Details-----")
    learning_rate = 1e-3
    batch_size = 20
    num_pretrained_epochs = 3
    num_epochs = 5

    '''
        Create data loaders
    '''
    print("-----Begin Creating the DataLoader-----")
    # Create the review data loader with the custom collate_fn
    review_data_loader = DataLoader(review_data_gen, 
                                   batch_size=batch_size, 
                                   collate_fn=review_data_gen.collate_fn,
                                   shuffle=True)

    # Create the collaborative data loader with the custom collate_fn
    collaborative_data_loader = DataLoader(collaborative_data_gen, 
                                          batch_size=batch_size, 
                                          collate_fn=collaborative_data_gen.collate_fn,
                                          shuffle=True)
    print("-----End Creating the DataLoader-----\n")

    # Set the model to the training mode
    content_model.train()
    content_model.to(device)

    collaborative_model.train()
    collaborative_model.to(device)

    # Obtain the optimizer
    review_optimizer = optim.Adam(content_model.parameters(), 
                                 lr=learning_rate)

    collaborative_optimizer = optim.Adam(collaborative_model.parameters(), 
                                        lr=learning_rate)
    
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
        progress_bar = tqdm(review_data_loader, desc=f"Epoch {epoch + 1}", ncols=80)
        for input_ids_prompt, input_ids_main, attention_mask in progress_bar:
            review_optimizer.zero_grad()

            # Obtain the data
            input_ids_prompt = input_ids_prompt.to(device)
            input_ids_main = input_ids_main.to(device)
            attention_mask = attention_mask.to(device)

            # Forward pass
            outputs = content_model(input_ids_prompt, 
                                   input_ids_main, 
                                   labels_main=input_ids_main,
                                   attention_mask=attention_mask)
            review_loss = outputs[0]

            # Backward pass and optimization
            review_loss.backward()
            review_optimizer.step()

            review_total_loss += review_loss.item()
            progress_bar.set_postfix({"Review Loss": review_loss.item()})

        review_average_loss = review_total_loss / len(review_data_loader)
        print(f"Epoch {epoch + 1} - Review Average Loss: {review_average_loss:.4f}")

        # Check if the current loss is better than the best_loss
        if review_average_loss < review_best_loss:
            review_best_loss = review_average_loss

            # Save user embeddings
            user_emb_path = os.path.join(content_model_root, f"user_embeddings_{lambda_V}.pt")
            torch.save(content_model.base_model.user_embeddings.state_dict(), user_emb_path)
            
            # Save item embeddings
            item_emb_path = os.path.join(content_model_root, f"item_embeddings_{lambda_V}.pt")
            torch.save(content_model.base_model.item_embeddings.state_dict(), item_emb_path)
    print("-----End Content GPT Pretraining Loop-----")

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
        
        progress_bar = tqdm(collaborative_data_loader, desc=f"Epoch {epoch + 1} - Collaborative", ncols=100)
        for input_ids_prompt, input_ids_main, attention_mask in progress_bar:
            collaborative_optimizer.zero_grad()

            input_ids_prompt = input_ids_prompt.to(device)
            input_ids_main = input_ids_main.to(device)
            attention_mask = attention_mask.to(device)

            with torch.no_grad():
                content_embeds = torch.cat(
                    (content_model.base_model.embed(input_ids_prompt),
                     content_model.base_model.embed(input_ids_main)),
                    axis=1
                ).to(device)
                
            # Forward pass of the collaborative GPT
            outputs = collaborative_model(input_ids_prompt, 
                                         input_ids_main, 
                                         labels_main=input_ids_main,
                                         attention_mask=attention_mask,
                                         regularize=True,
                                         lambda_V=lambda_V,
                                         content_embeds=content_embeds)
            collaborative_loss = outputs[0]
            regularize_loss = outputs[1]

            # Backward pass and optimization
            collaborative_loss.backward()
            collaborative_optimizer.step()
            
            collaborative_total_loss += collaborative_loss.item()
            regularize_total_loss += regularize_loss.item()
            
            progress_bar.set_postfix({"Collab Loss": collaborative_loss.item(),
                                      "Reg Loss": regularize_loss.item()})
        
        collaborative_average_loss = collaborative_total_loss / len(collaborative_data_loader)
        print(f"Epoch {epoch + 1} - Average Collaborative Loss: {collaborative_average_loss:.4f}")
        
        regularize_average_loss = regularize_total_loss / len(collaborative_data_loader)
        print(f"Epoch {epoch + 1} - Average Regularize Loss: {regularize_average_loss:.4f}")
        
        # Check if the current loss is better than the best_loss
        if collaborative_average_loss < collaborative_best_loss:
            collaborative_best_loss = collaborative_average_loss

            # Save user embeddings
            user_emb_path = os.path.join(collaborative_model_root, f"user_embeddings_{lambda_V}.pt")
            torch.save(collaborative_model.base_model.user_embeddings.state_dict(), user_emb_path)

            # Save item embeddings
            item_emb_path = os.path.join(collaborative_model_root, f"item_embeddings_{lambda_V}.pt")
            torch.save(collaborative_model.base_model.item_embeddings.state_dict(), item_emb_path)

        '''
            Optimize the content GPT model
        '''
        review_total_loss = 0
        regularize_total_loss = 0
        
        progress_bar = tqdm(review_data_loader, desc=f"Epoch {epoch + 1} - Content", ncols=100)
        for input_ids_prompt, input_ids_main, attention_mask in progress_bar:
            review_optimizer.zero_grad()

            input_ids_prompt = input_ids_prompt.to(device)
            input_ids_main = input_ids_main.to(device)
            attention_mask = attention_mask.to(device)

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

            # Backward pass and optimization
            review_loss.backward()
            review_optimizer.step()

            review_total_loss += review_loss.item()
            regularize_total_loss += regularize_loss.item()
            progress_bar.set_postfix({"Review Loss": review_loss.item(),
                                      "Reg Loss": regularize_loss.item()})

        review_average_loss = review_total_loss / len(review_data_loader)
        print(f"Epoch {epoch + 1} - Review Average Loss: {review_average_loss:.4f}")
        
        regularize_average_loss = regularize_total_loss / len(review_data_loader)
        print(f"Epoch {epoch + 1} - Average Regularize Loss: {regularize_average_loss:.4f}")

        # Check if the current loss is better than the best_loss
        if review_average_loss < review_best_loss:
            review_best_loss = review_average_loss

            # Save user embeddings
            user_emb_path = os.path.join(content_model_root, f"user_embeddings_{lambda_V}.pt") 
            torch.save(content_model.base_model.user_embeddings.state_dict(), user_emb_path)
            
            # Save item embeddings
            item_emb_path = os.path.join(content_model_root, f"item_embeddings_{lambda_V}.pt")
            torch.save(content_model.base_model.item_embeddings.state_dict(), item_emb_path)

if __name__ == "__main__":
    main()