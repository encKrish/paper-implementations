# Transformer Utilities
# Different modules for implementing Transformer in Pytorch

import torch 
import torch.nn as nn
import torch.nn.functional as F

import math

# Self-Attention

class SelfAttention(nn.Module):
    '''
    Self Attention class as used in the paper "Attention is all you Need".
    Args:
        input_size (int): size of the input vector
        repr_dim (int): size of the Query, Key and Value vectors
    '''
    def __init__(self, input_size:int, repr_dim:int):
        super().__init__()
        
        if input_size <= 0 or repr_dim <= 0:
            raise ValueError('input_size and repr_dim can only be a natural number.')
        
        self.repr_dim = repr_dim
            
        self.genQ = nn.Linear(input_size, repr_dim)
        self.genK = nn.Linear(input_size, repr_dim)
        self.genV = nn.Linear(input_size, repr_dim)
        
    def forward(self, x):  # x -> Shape: (num_words, input_size)
        Query, Keys, Values = self.genQ(x), self.genK(x), self.genV(x) # Shape: (num_words, repr_dim)
        
        similarity = torch.mm(Query, Keys.T) # Shape: (num_words, num_words)
        scores = F.softmax(similarity, dim=1) / math.sqrt(self.repr_dim) 
        scores = torch.mm(scores, Values) # Shape: (num_words, repr_dim)
        
        return scores


# Multi-Head Attention

class MultiHeadAttention(nn.Module):
    '''
    Multi-Head Attention class as used in the paper "Attention is all you Need".
    Args:
        input_size (int): size of the input vector
        repr_dim (int): size of the output vector
        num_heads (int, optional): number of attention heads to use. defaults to 6.
        force_linear (bool, optional): if to use reshaping layer (linear) for single head. defaults to False.
    '''
    def __init__(self, input_size, repr_dim, num_heads=6, force_linear=False):
        super().__init__()
        
        self.num_heads = num_heads
        if self.num_heads <= 0:
            raise ValueError('num_heads can only be a natural number.')
                
        self.heads = nn.ModuleList([SelfAttention(input_size, repr_dim) for i in range(num_heads)])
        
        self.reshapeLinear = nn.Linear(num_heads*repr_dim, repr_dim)
        
    def forward(self, x):
        similarities = torch.Tensor([])
        for mod in self.heads:
            similarities = torch.cat((similarities, mod(x)), dim=1) # Shape: (num_words, n * repr_dim)
        
        if self.num_heads is not 1 or force_linear:
            similarities = self.reshapeLinear(similarities) # Shape: (num_words, repr_dim) 
            
        return similarities


# Positional Embeddings

class PositionalEmbedding(nn.Module):
    '''
    Transformer styled Positional Embedding. Returns both word embeddings and positional embeddings. Size of word embeddings == size of positional embeddings
    Args:
        num_embeddings (int): vocab size for embeddings
        embedding_dim (int): size of embedding vector
        padding_idx (int): id of the padding unit in vocabulary
    '''
    
    def __init__(self, num_embeddings, embedding_dim, padding_idx):
        super().__init__()
        
        if embedding_dim % 2 is not 0:
            raise ValueError('Transformer styled Positional Embedding requires embedding dim to be even.')
            
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx)
        self.embedding_dim = embedding_dim
        
    def forward(self, x):
        embeddings = self.embedding(x)
        num_words = embeddings.shape[0]
        
        # Concatenate rows of the positional embedding multiplied by position
        pos_embeddings = torch.Tensor([])
        for pos in range(1, num_words+1): 
            pos_embeddings = torch.cat((pos_embeddings, 
                                       self.get_pos_embedding(pos)),
                                     dim=0)
        return embeddings + pos_embeddings
    
    def get_pos_embedding(self, pos):
        p = torch.zeros(self.embedding_dim)
        for i in range(1, self.embedding_dim+1):
            
            if i % 2 is 0:
                w_k = (1/10000)**(i/self.embedding_dim)
                p[i-1] = torch.sin(pos * torch.Tensor([w_k]))
            
            else:
                w_k = (1/10000)**((i-1)/self.embedding_dim)
                p[i-1] = torch.cos(pos * torch.Tensor([w_k]))
        p = p.unsqueeze(0)
        return p
