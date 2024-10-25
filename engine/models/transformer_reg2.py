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
        
        self.attention = nn.Sequential(
            nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim//2, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(in_channels=embedding_dim//2, out_channels=embedding_dim, kernel_size=1),
            nn.Softmax(dim=-1)
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(in_features=embedding_dim*2, out_features=embedding_dim),
            nn.SiLU(),
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
        
        p = self.product_embedding(products) # [batch, num_reference, embedding_dim] num reference can be various
        r = self.rating_embedding(ratings) # [batch, num_reference, embedding_dim]
        
        ref = torch.stack((p, r), dim = 2) # [batch, num_reference, 2, embedding_dim]
        _b, _r, _, _e = ref.size()
        ref = ref.view(_b, _r * 2, _e) # [batch, num_reference * 2, embedding_dim]
        
        target = self.product_embedding(target_product) # [batch, 1, embedding_dim]
        target_token = self.target_token.repeat(_b, 1, 1) # [batch, 1, embedding_dim]
        
        x = torch.cat((ref, target, target_token), dim = 1) # [batch, num_reference * 2 + 2, embedding_dim]
        x = self.encoders(x) # [batch, 1+num_reference, embedding_dim]
        
        b, c, e = x.size()
        x = x.view(b, e, c)
        
        w = self.attention(x.view(b, e, c))
        
        mu = torch.sum(x * w, dim=-1)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=-1) - mu**2 ).clamp(min=1e-4) )

        x = torch.cat((mu,sg),dim = -1)
        #x = torch.sum(x, dim = -1) # [batch, embedding_dim]
        x = self.ffn(x) # [batch, 1]
        
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
        
        