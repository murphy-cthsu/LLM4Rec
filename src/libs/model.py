'''
MIT License
Copyright (c) 2024 Yaochen Zhu
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer


class GPT4RecommendationBaseModel(nn.Module):
    '''
        The base class for collaborative GPT model, i.e.,
        the GPT model with extra user/item embeddings
    '''
    def __init__(self, config, qwen_model):
        super(GPT4RecommendationBaseModel, self).__init__()
        # Obtain the number of users, items
        # and the size of the original vocabulary
        self.num_users = config.num_users
        self.num_items = config.num_items
        self.vocab_size = config.vocab_size  # This is now the extended vocab size
        self.original_vocab_size = qwen_model.config.vocab_size  # Original Qwen vocab size
        self.config = config
        
        # Handle different config attribute names for embedding dimension
        if hasattr(config, 'n_embd') and config.n_embd is not None:
            self.embedding_dim = config.n_embd
        elif hasattr(config, 'hidden_size') and config.hidden_size is not None:
            self.embedding_dim = config.hidden_size
        else:
            raise ValueError("Either n_embd or hidden_size must be specified in the config.")
        
        # The pretrained Qwen model
        self.qwen_model = qwen_model
        
        # IMPORTANT: Ensure the model's embedding layer matches the extended vocabulary
        current_embed_size = qwen_model.get_input_embeddings().num_embeddings
        if self.vocab_size > current_embed_size:
            print(f"🔧 Resizing model embeddings from {current_embed_size} to {self.vocab_size}")
            qwen_model.resize_token_embeddings(self.vocab_size)
            print(f"✅ Model embeddings resized successfully")
        
        # Verify the resize worked
        final_embed_size = qwen_model.get_input_embeddings().num_embeddings
        if final_embed_size != self.vocab_size:
            raise RuntimeError(f"Failed to resize embeddings. Expected {self.vocab_size}, got {final_embed_size}")
        
        # Get the dtype of the base model
        self.model_dtype = next(qwen_model.parameters()).dtype
        
        # Calculate user/item token ID ranges in the extended vocabulary
        self.user_token_start = self.original_vocab_size
        self.user_token_end = self.original_vocab_size + self.num_users
        self.item_token_start = self.original_vocab_size + self.num_users
        self.item_token_end = self.original_vocab_size + self.num_users + self.num_items
        
        # Create separate embeddings for user/item tokens for collaborative filtering
        self.user_embeddings = nn.Embedding(self.num_users, self.embedding_dim, dtype=self.model_dtype)
        self.item_embeddings = nn.Embedding(self.num_items, self.embedding_dim, dtype=self.model_dtype)

        # Randomly initialize the new token embeddings
        initializer_range = getattr(config, 'initializer_range', 0.02)
        self.user_embeddings.weight.data.normal_(mean=0.0, std=initializer_range)
        self.item_embeddings.weight.data.normal_(mean=0.0, std=initializer_range)
        
        print(f"📊 Vocabulary Info:")
        print(f"   Original vocab size: {self.original_vocab_size}")
        print(f"   Extended vocab size: {self.vocab_size}")
        print(f"   User tokens range: [{self.user_token_start}, {self.user_token_end})")
        print(f"   Item tokens range: [{self.item_token_start}, {self.item_token_end})")
        print(f"   Model embedding size: {final_embed_size}")
        
    def embed(self, input_ids):
        # input_ids is a tensor of shape (batch_size, seq_length)
        device = input_ids.device
        
        # Validate input_ids range
        max_id = input_ids.max().item() if input_ids.numel() > 0 else 0
        min_id = input_ids.min().item() if input_ids.numel() > 0 else 0
        
        if max_id >= self.vocab_size or min_id < 0:
            print(f"❌ Invalid token IDs detected!")
            print(f"   Token ID range: [{min_id}, {max_id}]")
            print(f"   Valid range: [0, {self.vocab_size - 1}]")
            print(f"   Input shape: {input_ids.shape}")
            print(f"   Problematic IDs: {input_ids[input_ids >= self.vocab_size]}")
            raise ValueError(f"Token IDs out of range. Max ID: {max_id}, Vocab size: {self.vocab_size}")
        
        # Get base embeddings from the extended Qwen model
        embedding_layer = self.qwen_model.get_input_embeddings()
        base_embeddings = embedding_layer(input_ids)
        
        # Create masks for user and item tokens
        user_mask = ((input_ids >= self.user_token_start) & (input_ids < self.user_token_end)).long()
        item_mask = ((input_ids >= self.item_token_start) & (input_ids < self.item_token_end)).long()
        
        # Move masks to correct device
        user_mask = user_mask.to(device)
        item_mask = item_mask.to(device)
        
        # For user tokens, replace with our specialized embeddings
        if user_mask.any():
            user_indices = (input_ids - self.user_token_start) * user_mask
            user_indices = user_indices.clamp(0, self.num_users - 1)
            
            # Ensure embeddings are on correct device and dtype
            if self.user_embeddings.weight.device != device:
                self.user_embeddings = self.user_embeddings.to(device)
            if self.user_embeddings.weight.dtype != self.model_dtype:
                self.user_embeddings = self.user_embeddings.to(dtype=self.model_dtype)
                
            user_emb = self.user_embeddings(user_indices)
            # Replace base embeddings with specialized user embeddings
            base_embeddings = base_embeddings * (1 - user_mask.unsqueeze(-1).float()) + user_emb * user_mask.unsqueeze(-1).float()
        
        # For item tokens, replace with our specialized embeddings
        if item_mask.any():
            item_indices = (input_ids - self.item_token_start) * item_mask
            item_indices = item_indices.clamp(0, self.num_items - 1)
            
            # Ensure embeddings are on correct device and dtype
            if self.item_embeddings.weight.device != device:
                self.item_embeddings = self.item_embeddings.to(device)
            if self.item_embeddings.weight.dtype != self.model_dtype:
                self.item_embeddings = self.item_embeddings.to(dtype=self.model_dtype)
                
            item_emb = self.item_embeddings(item_indices)
            # Replace base embeddings with specialized item embeddings
            base_embeddings = base_embeddings * (1 - item_mask.unsqueeze(-1).float()) + item_emb * item_mask.unsqueeze(-1).float()
        
        return base_embeddings
        
    def forward(self, input_ids=None, **kwargs):
        # Obtain the embeddings of the input id sequence
        input_embeddings = self.embed(input_ids)
        # Use the underlying transformer model, not the CausalLM wrapper
        # This ensures we get BaseModelOutputWithPast instead of CausalLMOutputWithPast
        if hasattr(self.qwen_model, 'model'):
            # For CausalLM models, access the underlying transformer
            return self.qwen_model.model(inputs_embeds=input_embeddings, **kwargs)
        else:
            # Fallback for other model types
            return self.qwen_model(inputs_embeds=input_embeddings, **kwargs)


class CollaborativeGPTwithItemLMHeadBatch(nn.Module):
    '''
        Collaborative filtering model to learn user/item embeddings.
    '''
    def __init__(self, config, base_model):
        super(CollaborativeGPTwithItemLMHeadBatch, self).__init__()

        # Obtain the number of users, items, and vocabulary size
        self.num_users = config.num_users
        self.num_items = config.num_items
        self.vocab_size = config.vocab_size

        # Base GPT model with extended user/item ID token embeddings
        self.base_model = base_model
        
        # Get embedding dimension consistently
        embedding_dim = base_model.embedding_dim
        
        # Item recommendation head with correct dtype
        self.item_head = nn.Linear(embedding_dim, self.num_items, bias=False, dtype=base_model.model_dtype)
        
        # Tie the weights between the item embeddings and the item recommendation head
        self.item_head.weight = self.base_model.item_embeddings.weight 

    def forward(self,
                input_ids_prompt,
                input_ids_main,
                labels_main=None,
                attention_mask=None,
                regularize=False,
                lambda_V=None,
                content_embeds=None,
                **kwargs):
        # Base model forward pass for the prompt text
        outputs_prompt = self.base_model(input_ids=input_ids_prompt, 
                                         return_dict=True, 
                                         **kwargs)
        past_key_values = outputs_prompt.past_key_values

        # Base model forward pass for the main text with attention mask
        outputs_main = self.base_model(input_ids=input_ids_main,
                                       past_key_values=past_key_values,
                                       attention_mask=attention_mask,
                                       return_dict=True)

        item_logits = self.item_head(outputs_main.last_hidden_state)
        outputs = (item_logits,) + outputs_main[1:]

        if labels_main is not None:
            # Shift so that tokens < n predict n
            shift_logits = item_logits[..., :-1, :].contiguous()
            shift_labels = labels_main[..., 1:].contiguous()
            # print("[CollaborativeGPTwithItemLMHeadBatch] shift_labels", shift_labels)
            # Convert item tokens to indices (subtract original_vocab_size and num_users)
            shift_labels = shift_labels - 151669 - self.num_users
            # print("[CollaborativeGPTwithItemLMHeadBatch] (After) shift_labels", shift_labels)
            # Validate labels are in valid range
            valid_mask = (shift_labels >= 0) & (shift_labels < self.num_items)
            if not valid_mask.all():
                print(f"Warning: Found invalid labels. Range should be [0, {self.num_items-1}]")
                print(f"Label range found: [{shift_labels.min().item()}, {shift_labels.max().item()}]")
                # Clamp invalid labels to valid range
                shift_labels = shift_labels.clamp(0, self.num_items - 1)

            # Define the loss function
            loss_fct = CrossEntropyLoss()

            # Calculate the loss only where attention mask is one
            prompt_length = input_ids_prompt.shape[1]
            main_length = input_ids_main.shape[1]
        
            active_loss = attention_mask[:, prompt_length+1:].reshape(-1) == 1
            active_logits = shift_logits.view(-1, shift_logits.size(-1))[active_loss]
            active_labels = shift_labels.view(-1)[active_loss]

            # Language modeling loss for the item sequences
            loss = loss_fct(active_logits, active_labels)
            
            # Mutual regularization loss
            if regularize:
                collaborative_embeds = torch.cat(
                    (self.base_model.embed(input_ids_prompt),
                     self.base_model.embed(input_ids_main)),
                    axis=1
                )
                regularize_loss = lambda_V * torch.mean(
                    nn.MSELoss(reduction='sum')(
                        collaborative_embeds,
                        content_embeds)
                )
                loss += regularize_loss
                outputs = (loss, regularize_loss) + outputs
            else:
                outputs = (loss,) + outputs
        return outputs


class ContentGPTForUserItemWithLMHeadBatch(nn.Module):
    '''
        This class conducts language modeling to learn both
        user/item token embeddings via textual data, where
        we view the texts that include user/item ID as prompt.
        E.g.,
            inputs_ids_prompt:
              "user_1 writes the following review for item_1:"
            inputs_ids_main:
              "This item is too expensive."
        where we only calculate LM loss on the main texts.
    '''
    def __init__(self, config, base_model):
        super(ContentGPTForUserItemWithLMHeadBatch, self).__init__()
        self.base_model = base_model
        
        # Get embedding dimension consistently
        embedding_dim = base_model.embedding_dim
        
        self.lm_head = nn.Linear(embedding_dim, config.vocab_size, bias=False, dtype=base_model.model_dtype)

        # Tie weights between the output layer and the token embeddings
        # For Qwen models, we need to get the input embeddings from the model
        input_embeddings = self.base_model.qwen_model.get_input_embeddings()
        self.lm_head.weight = input_embeddings.weight

    def forward(self, 
                input_ids_prompt, 
                input_ids_main, 
                labels_main=None,
                attention_mask=None,
                regularize=False,
                lambda_V=None,
                collaborative_embeds=None,
                **kwargs):
        outputs_prompt = self.base_model(input_ids=input_ids_prompt, 
                                         return_dict=True, **kwargs)
        past_key_values = outputs_prompt.past_key_values

        # Calculate the language modeling loss for the main texts
        outputs_main = self.base_model(input_ids=input_ids_main, 
                                       past_key_values=past_key_values, 
                                       attention_mask=attention_mask,
                                       return_dict=True)

        lm_logits = self.lm_head(outputs_main.last_hidden_state)
        outputs = (lm_logits,) + outputs_main[1:]

        if labels_main is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels_main[..., 1:].contiguous()
            
            # Define the loss function
            loss_fct = CrossEntropyLoss()

            # Calculate the loss only where attention mask is one
            prompt_length = input_ids_prompt.shape[1]
            main_length = input_ids_main.shape[1]
            
            active_loss = attention_mask[:, prompt_length+1:].reshape(-1) == 1
            active_logits = shift_logits.view(-1, shift_logits.size(-1))[active_loss]
            active_labels = shift_labels.view(-1)[active_loss]

            # Language modeling loss for the token sequences
            loss = loss_fct(active_logits, active_labels)
            
            # Mutual regularization loss
            if regularize:
                # User/Item token embeddings only appear in the prompt
                content_embeds = self.base_model.embed(input_ids_prompt)
                regularize_loss = lambda_V * torch.mean(
                    nn.MSELoss(reduction='sum')(
                        content_embeds,
                        collaborative_embeds)
                )
                loss += regularize_loss
                outputs = (loss, regularize_loss) + outputs            
            else:
                outputs = (loss,) + outputs
        return outputs


class CollaborativeGPTwithItemRecommendHead(nn.Module):
    '''
        Recommend items to a user according to input queries.
        multinomial likelihood is put on all the items for a user.
    '''
    def __init__(self, config, base_model):
        super(CollaborativeGPTwithItemRecommendHead, self).__init__()
        # Obtain the number of users and items
        self.num_users = config.num_users
        self.num_items = config.num_items

        # Base GPT model with extended user/item ID token embeddings
        self.base_model = base_model
        
        # Get embedding dimension consistently
        embedding_dim = base_model.embedding_dim
        
        # Item recommendation head with correct dtype
        self.item_head = nn.Linear(embedding_dim, self.num_items, bias=False, dtype=base_model.model_dtype)
        
        # Tie the weights between the item embeddings and the item recommendation head
        self.item_head.weight = self.base_model.item_embeddings.weight 

    def forward(self, 
                input_ids=None, 
                target_ids=None,
                attention_mask=None,
                regularize=False,
                lambda_V=None,
                main_ids=None,
                content_embeds=None,
                **kwargs):    
        transformer_outputs = self.base_model(input_ids, 
                                              attention_mask=attention_mask, 
                                              **kwargs)
        hidden_states = transformer_outputs[0]

        # Find the indices of the last non-padding tokens
        last_non_pad_token_indices = attention_mask.sum(dim=1) - 1

        # Gather the last non-padding token embeddings
        last_token_hidden_states = torch.stack([
            hidden_states[i, idx, :] for i, idx in \
                enumerate(last_non_pad_token_indices)
        ])

        # Calculate the item scores
        item_scores = self.item_head(last_token_hidden_states)

        # Convert scores to multinomial probabilities
        item_log_probs = F.log_softmax(item_scores, dim=-1)
        
        # Calculating the multinomial loss
        neg_ll = -torch.mean(torch.sum(item_log_probs * target_ids, dim=-1))
        
        if regularize:
            # User/Item token embeddings only appear in the prompt
            rec_embeds_prompt = self.base_model.embed(input_ids)
            rec_embeds_target = self.base_model.embed(main_ids)
            rec_embeds = torch.cat(
                (rec_embeds_prompt, rec_embeds_target),
                axis=1
            )
            regularize_loss = lambda_V * torch.mean(
                nn.MSELoss(reduction='sum')(
                    rec_embeds,
                    content_embeds)
            )
            neg_ll += regularize_loss
            outputs = (neg_ll, regularize_loss, item_log_probs)
        else: 
            outputs = (neg_ll, item_log_probs)
        return outputs