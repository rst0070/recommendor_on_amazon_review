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
        
        self.target_token = torch.zeros(
            (1, 1, embedding_dim),
            device=device
        )
        """
        Target token is like CLS token : just letting model knows it needs to be infered.
        It was better to set zeros then parameterizing it.
        """
        
        _encoders = OrderedDict()
        for i in range(0, num_transformer_block):
            _encoders[f"encoder{i}"] = TransformerEncoderBlock(
                    d_model=embedding_dim, 
                    ffn_hidden=ffn_hidden
                )
        
        self.encoders = nn.Sequential(_encoders)
        
        self.ffn = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=1),
            nn.Sigmoid()
        )
        """
        last layer for regression.
        The simple was better then using complex structure
        """

        
    def forward(
        self, 
        target_product:torch.IntTensor,
        products:torch.IntTensor,
        ratings:torch.IntTensor
        ) -> torch.Tensor:
        
        
        target = self.product_embedding(target_product) # [batch, 1, embedding_dim]
        target_rating = self.target_token # [batch, 1, embedding_dim]
        target = target + target_rating # [batch, 1, embedding_dim]
        
        reference = self.product_embedding(products) # [batch, num_reference, embedding_dim] num reference can be various
        reference_rating = self.rating_embedding(ratings) # [batch, num_reference, embedding_dim]
        reference = reference + reference_rating # [batch, num_reference, embedding_dim]
        
        
        x = torch.cat((target, reference), dim = 1) # [batch, 1+num_reference, embedding_dim]
        x = self.encoders(x) # [batch, 1+num_reference, embedding_dim]
        
        x = self.ffn(x[:, 0,:]) # [batch, 1]
        x = 5.0 * x
        
        return x
        
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
        
        