import torch.nn as nn
import torch
class RatingEmbedding(nn.Embedding):
    
    def __init__(
        self,
        num_embeddings: int, 
        embedding_dim: int
        ):
        """
        get embedding of rating! for combining with other information
        Args:
            num_embeddings (int): number of different rating == 5 (in the dataset: 1, 2, 3, 4 or 5)
            embedding_dim (int): needs to be same as other embeddings!
        """
        super(RatingEmbedding, self).__init__(
                num_embeddings, 
                embedding_dim,
                padding_idx=0,
                dtype=torch.float32
            )