import torch
import torch.nn as nn
from models.encoder.transformer import TransformerEncoderBlock
from models.embedding.product import ProductEmbedding
from models.embedding.rating import RatingEmbedding
from collections import OrderedDict

class TransformerReg(nn.Module):
    
    def __init__(
        self,
        num_product:int,
        num_rating:int,
        embedding_dim:int,
        num_transformer_block:int,
        ffn_hidden:int,
        device
        ):
        """
        
        """
        super(TransformerReg, self).__init__()
        
        self.product_embedding = ProductEmbedding(
                num_embeddings=num_product+1,
                embedding_dim=embedding_dim
            )
        
        self.rating_embedding = RatingEmbedding(
                num_embeddings=num_rating+1,
                embedding_dim=embedding_dim
            )
        
        self.target_token = torch.nn.Parameter(
                torch.randn(1, 1, embedding_dim)
            ).to(device)
        """
        Target token is like CLS token : just letting model knows it needs to be infered
        """
        
        _encoders = OrderedDict()
        for i in range(0, num_transformer_block):
            _encoders[f"encoder{i}"] = TransformerEncoderBlock(
                    d_model=embedding_dim, 
                    ffn_hidden=ffn_hidden
                )
        
        self.encoders = nn.Sequential(_encoders)
        
        self.ffn = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=embedding_dim),
            nn.Linear(in_features=embedding_dim, out_features=1),
            nn.BatchNorm1d(num_features=1),
            nn.Sigmoid()
        )

        
    def forward(
        self, 
        target_product:torch.IntTensor,
        products:torch.IntTensor,
        ratings:torch.IntTensor
        ) -> torch.Tensor:
        
        target = self.product_embedding(target_product) \
                + self.target_token
        # [batch, 1, emb_dim]
                
        ref = self.product_embedding(products) \
                + self.rating_embedding(ratings)
        # [batch, num_ref, emb_dim]
                
        x = torch.cat((ref, target), dim = 1) # [batch, num_ref + 1, emb_dim]
        x = self.encoders(x) # [batch, num_ref + 1, emb_dim]
        
        x = self.ffn(x[:, -1, :]) # [batch, 1]
        
        return x * 5.0
        
if __name__ == "__main__":
    from torchinfo import summary
    
    num_product = 10000
    num_rating = 5
    
    num_ref = 100
    emb_dim = 16
    
    device = 0
    
    model = TransformerReg(
            num_product=num_product,
            num_rating=num_rating,
            embedding_dim=emb_dim,
            num_transformer_block=5,
            ffn_hidden=80,
            device=device
        ).to(device)
    
    
    summary(
        model = model, 
        input_data = [
                torch.randint(low=0, high=num_product-1, size=(2, 1)).to(device), # target product id
                torch.randint(low=0, high=num_product-1, size=(2, num_ref)).to(device), # reference ids of product
                torch.randint(low=0, high=num_rating-1, size=(2, num_ref)).to(device), # reference ratings of product
            ]
    )
        
        