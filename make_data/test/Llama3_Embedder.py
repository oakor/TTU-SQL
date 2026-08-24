#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 llama3.2 1B  embedding 
"""

import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Union


class Llama3Embedding:
    def __init__(self, model_path: str = "../output/llama321B/merged_models"):
        """
         llama3.2 1B  embedding
        
        Args:
            model_path: ，
        """
        print(f"Loading model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False
        )
        
        #  pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # ， embedding，
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.model.eval()
        print(f"✅ Model loaded: {model_path}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
         embedding
        
        Args:
            text: 
            
        Returns:
            embedding  (numpy array)
        """
        # Tokenize 
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512  # ，
        )
        
        # 
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # 
            outputs = self.model(**inputs)
            
            #  embedding
            #  padding token  embedding
            last_hidden_states = outputs.last_hidden_state
            
            #  attention mask  token
            attention_mask = inputs['attention_mask']
            
            #  token embedding 
            #  padding 0
            token_embeddings = last_hidden_states * attention_mask.unsqueeze(-1)
            
            # （）
            sequence_lengths = torch.sum(attention_mask, dim=-1, keepdim=True)
            sentence_embeddings = torch.sum(token_embeddings, dim=1) / sequence_lengths
            
            #  CPU  numpy ( float32  BFloat16 )
            embedding = sentence_embeddings.cpu().to(torch.float32).numpy()[0]
        
        return embedding
    
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """
         embedding
        
        Args:
            texts: 
            
        Returns:
            embedding 
        """
        embeddings = []
        for text in texts:
            emb = self.embed_text(text)
            embeddings.append(emb)
        return embeddings
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        
        
        Args:
            text1, text2: 
            
        Returns:
             (0-1)
        """
        emb1 = self.embed_text(text1)
        emb2 = self.embed_text(text2)
        
        # 
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        similarity = dot_product / (norm1 * norm2)
        
        return float(similarity)

def main():
    # 
    print("Initializing Llama3.2 1B embedding model...")
    embedder = Llama3Embedding()
    
    #  embedding
    test_text = "SELECT name FROM users WHERE age > 18"
    print(f"\nEmbedding text: {test_text}")
    embedding = embedder.embed_text(test_text)
    print(f"Embedding shape: {embedding.shape}")
    print(f"First 10 dimensions: {embedding[:10]}")
    
    #  embedding
    test_texts = [
        "SELECT name FROM users WHERE age > 18",
        "SELECT * FROM products WHERE price < 100",
        "SELECT COUNT(*) FROM orders WHERE status = 'completed'"
    ]
    print(f"\nEmbedding {len(test_texts)} texts...")
    embeddings = embedder.embed_texts(test_texts)
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"All shapes: {[emb.shape for emb in embeddings]}")
    
    # 
    print(f"\nTesting similarity calculation...")
    sim1 = embedder.similarity(
        "18",
        "21"
    )
    sim2 = embedder.similarity(
        "it",
        "i"
    )
    sim3 = embedder.similarity(
        "mp",
        "m"
    )
    sim4 = embedder.similarity(
        "INNER",
        "LEFT"
    )
    print(f"Similarity between similar queries: {sim1:.4f}")
    print(f"Similarity between different queries: {sim2:.4f}")
    print(f"Similarity between different queries: {sim3:.4f}")
    print(f"Similarity between different queries: {sim4:.4f}")


if __name__ == "__main__":
    main()