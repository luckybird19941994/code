import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from torch.nn import Dropout,Linear

class Mlp(nn.Module):

    def __init__(self,in_features=64,hidden_features=64,dropout_rate=0.1):
        super(Mlp, self).__init__()
        self.fc1 = Linear(in_features,hidden_features)
        self.fc2 = Linear(hidden_features,in_features)
        self.act_fn = nn.ReLU()
        self.dropout = Dropout(dropout_rate)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):

        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, embed_dim = 256, skip_lam = 2.0, mlp_hidden_dim = 128, drop_path = 0.0 ):

        super().__init__()

        self.mlp = Mlp(in_features=embed_dim,hidden_features=mlp_hidden_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.skip_lam = skip_lam
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self,x,policy=None):

        if policy != None:
            x = x * policy

        x = x + self.drop_path( self.mlp( self.norm2(x) ) ) / self.skip_lam

        return x

class PredictorLG(nn.Module):

    def __init__(self, embed_dim=384):
        super().__init__()

        self.mlp_local = nn.Sequential(
            Mlp(embed_dim,embed_dim),
            Mlp(embed_dim,embed_dim),
            Mlp(embed_dim,embed_dim)
        )

        self.mlp_global = nn.Sequential(
            Mlp(embed_dim, embed_dim),
            Mlp(embed_dim, embed_dim),
            Mlp(embed_dim, embed_dim)
        )

        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, 2),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self,x,policy):

        B, N, C = x.size()

        x = self.in_conv(x)

        local_x = self.mlp_local(x)
        global_x = ( self.mlp_global(x) * policy ).sum(dim=1,keepdim=True) / torch.sum(policy,dim=1,keepdim=True)
        x = torch.cat([local_x,global_x.expand(B,N,C)],dim=-1)
        x = self.out_conv(x)
        return x

class Selection(nn.Module):

    def __init__(self,embed_dim = 256,pruning_loc = [3,6,9],depth = 10,vis_mode = True ):

        super(Selection,self).__init__()

        predictor_list = [PredictorLG(embed_dim) for _ in range(len(pruning_loc))]
        self.score_predictor = nn.ModuleList(predictor_list)

        self.blocks = nn.ModuleList( [ Block(embed_dim=embed_dim) for i in range(depth) ] )

        self.pruning_loc = pruning_loc
        self.vis_mode = vis_mode

    def forward(self, x):

        b,c,h,w = x.shape

        prev_decision = torch.ones(b,c,1,dtype=x.dtype,device=x.device)

        p_count = 0
        out_pred_prob = []

        x = x.reshape(b, c, h * w)

        for i, blk in enumerate(self.blocks):

            if i in self.pruning_loc:

                pred_score = self.score_predictor[p_count](x,prev_decision).reshape(b,-1,2)

                hard_keep_decision = F.gumbel_softmax(pred_score,hard=True)[:,:,0:1] * prev_decision
                out_pred_prob.append( hard_keep_decision.reshape(b,c) )

                policy = hard_keep_decision
                x = blk(x,policy=policy)

                prev_decision = hard_keep_decision
                p_count += 1

            else:
                x = blk(x)

        return prev_decision.unsqueeze(-1),out_pred_prob

def _Selection():

    net = Selection()
    return net









