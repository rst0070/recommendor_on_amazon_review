import torch
import torch.nn as nn
from models.encoder.transformer import TransformerEncoderBlock
from models.embedding.product import ProductEmbedding
from models.embedding.rating import RatingEmbedding
from collections import OrderedDict

class TransformerCls(nn.Module):
    
    def __init__(
        self,
        device,
        num_product:int,
        num_rating:int = 5,
        embedding_dim:int = 64,
        num_transformer_block:int = 3,
        ffn_hidden:int = 80
        ):
        """
        TransformerCls is classification model.
        input: target product_id and reference data (pairs of  product_id and rating)
        classes: rating 1, 2, 3, 4 or 5
        """
        super(TransformerCls, self).__init__()
        
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
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim),
            nn.SiLU(),
            nn.BatchNorm1d(num_features=embedding_dim),
            nn.Linear(in_features=embedding_dim, out_features=num_rating),
            nn.BatchNorm1d(num_features=num_rating)
        )

        
    def forward(
        self, 
        target_product:torch.IntTensor,
        products:torch.IntTensor,
        ratings:torch.IntTensor
        ) -> torch.Tensor:
        
        p = self.product_embedding(products) # [batch, num_reference, embedding_dim] num reference can be various
        r = self.rating_embedding(ratings) # [batch, num_reference, embedding_dim]
        
        ref = torch.stack((p, r), dim = 2) # [batch, num_reference, 2, embedding_dim]
        _b, _r, _, _e = ref.size()
        ref = ref.view(_b, _r * 2, _e) # [batch, num_reference * 2, embedding_dim]
        
        target = self.product_embedding(target_product) # [batch, 1, embedding_dim]
        target_token = self.target_token.repeat(_b, 1, 1) # [batch, 1, embedding_dim]
        
        x = torch.cat((ref, target, target_token), dim = 1) # [batch, num_reference * 2 + 2, embedding_dim]
        x = self.encoders(x) # [batch, 1+num_reference, embedding_dim]
        
        cls_token = x[:, -1, :] # [batch, embedding_dim]
        x = self.classifier(cls_token) # [batch, 5]
        
        return x
        
if __name__ == "__main__":
    from torchinfo import summary
    
    num_product = 10000
    num_rating = 5
    
    num_ref = 10
    emb_dim = 64
    
    device = 0
    
    model = TransformerCls(
            device=device,
            num_product=num_product,
            num_rating=num_rating,
            embedding_dim=emb_dim,
            num_transformer_block=3,
            ffn_hidden=80,
        ).to(device)
    
    
    summary(
        model = model, 
        input_data = [
                torch.randint(low=0, high=num_product-1, size=(2, 1)).to(device), # target product id
                torch.randint(low=0, high=num_product-1, size=(2, num_ref)).to(device), # reference ids of product
                torch.randint(low=0, high=num_rating-1, size=(2, num_ref)).to(device), # reference ratings of product
            ]
    )
        
        