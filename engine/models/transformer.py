import torch
import torch.nn as nn
from typing import Optional
from models.encoder.transformer import TransformerEncoderBlock
from collections import OrderedDict

class RatingEmbedding(nn.Embedding):
    def __init__(
        self,
        embedding_dim: int
        ):
        """
        get embedding of rating! for combining with other information
        Args:
            num_embeddings (int): number of different rating == 5 (in the dataset: 1, 2, 3, 4 or 5) but need for the value 0
            embedding_dim (int): needs to be same as other embeddings!
        """
        super(RatingEmbedding, self).__init__(5 + 1, embedding_dim, dtype=torch.float32)
        
class ProductEmbedding(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int, 
        embedding_dim: int
    ):
        super(ProductEmbedding, self).__init__(num_embeddings=num_embeddings, embedding_dim=embedding_dim, dtype=torch.float32)
        
class TransformerReg(nn.Module):
    
    def __init__(
        self,
        num_product:int,
        embedding_dim:int,
        num_transformer_block:int,
        ffn_hidden:int
        ):
        """
        
        """
        super(TransformerReg, self).__init__()
        
        self.product_embedding = ProductEmbedding(
                num_embeddings=num_product+1,
                embedding_dim=embedding_dim
            )
        
        self.rating_embedding = RatingEmbedding(
                embedding_dim=embedding_dim
            )
        
        self.regression_token = torch.nn.Parameter(
                torch.randn(1, 1, embedding_dim)
            )
        """
        regression_token is like CLS token : just letting model knows it needs to do regression
        """
        
        _encoders = OrderedDict()
        for i in range(0, num_transformer_block):
            _encoders[f"encoder{i}"] = TransformerEncoderBlock(
                    d_model=embedding_dim, 
                    ffn_hidden=ffn_hidden
                )
        
        self.encoders = nn.Sequential(_encoders)

        
    def forward(
        self, 
        target_product: Optional[torch.IntTensor] = None,
        products:Optional[torch.IntTensor] = None,
        ratings:Optional[torch.IntTensor] = None,
        is_training = True
        ) -> torch.Tensor:
        
        regression_token = self.regression_token # [1, 1, embedding_dim]
        reference = self.product_embedding(products) + self.rating_embedding(ratings) # [batch, num_reference, embedding_dim]
        B, R, _ = reference.size()
        
        x = torch.cat(
            (
                regression_token.repeat(B, R, 1), 
                reference
            ), dim = 1) # [batch, 1+num_reference, embedding_dim]
        x = self.encoders(x) # [batch, 1+num_reference, embedding_dim]
        
        x = x[:, 0,:] # [batch, embedding_dim]
        if not is_training: # if its for prediction: return expected user vector
            return x
        
        # for training, predict the rating by using similarity b/w user vector and target product
        target = self.product_embedding(target_product)[:, 0, :]

        x = target * x # [batch, embedding_dim]
        x = (torch.sum(x, dim=1) + 1) * 2.5 # scale to 0~5
        
        return x
    
        
if __name__ == "__main__":
    from torchinfo import summary
    
    num_product = 10000
    
    num_ref = 100
    emb_dim = 8
    
    device = 0
    
    model = TransformerReg(
            num_product=num_product,
            embedding_dim=emb_dim,
            num_transformer_block=3,
            ffn_hidden=80,
        ).to(device)
    summary(
        model = model, 
        input_data = [
                torch.randint(low=0, high=num_product-1, size=(2, 1)).to(device), # target product id
                torch.randint(low=0, high=num_product-1, size=(2, num_ref)).to(device), # reference ids of product
                torch.randint(low=1, high=5, size=(2, num_ref)).to(device), # reference ratings of product
            ]
    )
    summary(
        model = model, 
        input_data = [
                torch.randint(low=0, high=num_product-1, size=(2, 1)).to(device), # target product id
                torch.randint(low=0, high=num_product-1, size=(2, num_ref)).to(device), # reference ids of product
                torch.randint(low=1, high=5, size=(2, num_ref)).to(device), # reference ratings of product
                False
            ]
    )
        
        