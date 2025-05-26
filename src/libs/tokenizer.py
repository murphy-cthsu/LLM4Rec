'''
MIT License
Copyright (c) 2024 Yaochen Zhu
Modified for e-commerce data with angle bracket token format
'''

import re
import numpy as np
from transformers import AutoTokenizer

class TokenizerWithUserItemIDTokens:
    def __init__(self, pretrained_model_name_or_path, num_users, num_items, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        self.num_users = num_users
        self.num_items = num_items

        self.user_tokens = [f"<user_{i}>" for i in range(num_users)]
        self.item_tokens = [f"<item_{j}>" for j in range(num_items)]

        self.tokenizer.add_tokens(self.user_tokens + self.item_tokens)

        # 保存id範圍方便後續使用
        self.vocab_size = len(self.tokenizer)
        self.user_token_ids = [self.tokenizer.convert_tokens_to_ids(t) for t in self.user_tokens]
        self.item_token_ids = [self.tokenizer.convert_tokens_to_ids(t) for t in self.item_tokens]

    def _pre_tokenize(self, text):
        import re
        pattern = r'(<user_\d+>|<item_\d+>)'
        pieces = re.split(pattern, text)
        pieces = [p for p in pieces if p.strip()]
        return pieces

    def _tokenize(self, text):
        split_tokens = []
        pieces = self._pre_tokenize(text)
        for piece in pieces:
            if piece in self.user_tokens or piece in self.item_tokens:
                split_tokens.append(piece)
            else:
                split_tokens.extend(self.tokenizer.tokenize(piece))
        return split_tokens

    def encode(self, text, **kwargs):
        tokens = self._tokenize(text)
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def encode_batch(self, texts, max_length=None):
        encoded_inputs = []
        max_len = max(len(self._tokenize(t)) for t in texts)
        if max_length is None or max_length < max_len:
            max_length = max_len
        for text in texts:
            tokens = self._tokenize(text)
            ids = self.tokenizer.convert_tokens_to_ids(tokens)
            att_mask = [1]*len(ids)
            pad_len = max_length - len(ids)
            ids += [self.tokenizer.pad_token_id or 0]*pad_len
            att_mask += [0]*pad_len
            encoded_inputs.append((ids, att_mask))
        import numpy as np
        input_ids, attention_mask = zip(*encoded_inputs)
        return np.array(input_ids), np.array(attention_mask)

class TokenizerWithUserItemIDTokensBatch:
    def __init__(self, pretrained_model_name_or_path, num_users, num_items, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        self.num_users = num_users
        self.num_items = num_items
        
        # 定義特殊token字串
        self.user_tokens = [f"<user_{i}>" for i in range(num_users)]
        self.item_tokens = [f"<item_{j}>" for j in range(num_items)]

        # 新增特殊token到詞彙表
        self.tokenizer.add_tokens(self.user_tokens + self.item_tokens)

        # padding token id
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

    def _pre_tokenize(self, text):
        pattern = r'(<user_\d+>|<item_\d+>)'
        pieces = re.split(pattern, text)
        pieces = [p for p in pieces if p.strip()]
        return pieces

    def _tokenize(self, text):
        split_tokens = []
        pieces = self._pre_tokenize(text)
        for piece in pieces:
            if piece in self.user_tokens or piece in self.item_tokens:
                split_tokens.append(piece)
            else:
                split_tokens.extend(self.tokenizer.tokenize(piece))
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def encode_batch(self, texts, max_length=None):
        encoded_inputs = []
        max_len_in_batch = max(len(self._tokenize(t)) for t in texts)
        if max_length is None or max_length < max_len_in_batch:
            max_length = max_len_in_batch

        for text in texts:
            tokens = self._tokenize(text)
            input_ids = self.convert_tokens_to_ids(tokens)
            attention_mask = [1] * len(input_ids)

            padding_len = max_length - len(input_ids)
            input_ids += [self.pad_token_id] * padding_len
            attention_mask += [0] * padding_len

            encoded_inputs.append((input_ids, attention_mask))

        input_ids_batch, attention_mask_batch = zip(*encoded_inputs)
        return np.array(input_ids_batch), np.array(attention_mask_batch)
