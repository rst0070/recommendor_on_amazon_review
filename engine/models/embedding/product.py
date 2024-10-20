import torch.nn as nn
import torch

class ProductEmbedding(nn.Embedding):
    
    def __init__(
        self,
        num_embeddings: int, 
        embedding_dim: int
    ):
        super(ProductEmbedding, self).__init__(
                num_embeddings=num_embeddings, 
                embedding_dim=embedding_dim,
                padding_idx=0,
                dtype=torch.float32
            )