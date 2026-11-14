import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv
from typing import Literal, Tuple
from torch_geometric.nn.inits import reset, uniform
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_dim, h_dim, lyr_num: int, backbone: Literal['gcn', 'gat'], attn_heads: int, act_fn: str):
        super(Encoder, self).__init__()
        self.act_fn = {'relu': nn.ReLU(), 'prelu': nn.PReLU(), 'rrelu': nn.RReLU()}[act_fn]
        self.lyr_num = lyr_num
        self.fc = nn.Linear(in_dim, h_dim)
        self.bn = nn.BatchNorm1d(h_dim)
        self.bn2 = nn.BatchNorm1d(h_dim)
        if backbone == 'gcn':
            if lyr_num == 1:
                self.conv = [GCNConv(h_dim, h_dim)]
            if lyr_num >= 2:
                self.conv = [GCNConv(h_dim, h_dim * 2)]
                for _ in range(1, lyr_num-1):
                    self.conv.append(GCNConv(h_dim * 2, h_dim * 2))
                self.conv.append(GCNConv(h_dim * 2, h_dim))
        elif backbone == 'gat':
            if lyr_num == 1:
                self.conv = [GATConv(h_dim, h_dim, heads=attn_heads, concat=False)]
            if lyr_num >= 2:
                self.conv = [GATConv(h_dim, h_dim * 2, heads=attn_heads, concat=False)]
                for _ in range(1, lyr_num-1):
                    self.conv.append(GATConv(h_dim * 2, h_dim * 2, heads=attn_heads, concat=False))
                self.conv.append(GATConv(h_dim * 2, h_dim, heads=attn_heads, concat=False))
        self.conv = nn.ModuleList(self.conv)
        self.auger = GATConv(h_dim, h_dim, heads=1)
        
    def forward(self, x, edge_index, auger: bool = False) -> torch.Tensor:
        x = self.act_fn(self.fc(x))
        x = self.bn(x)
        res = x
        if auger is False:
            for lyr in range(self.lyr_num):
                x = self.act_fn(self.conv[lyr](x, edge_index))
        else:
            x = self.auger(x, edge_index)
        x = x + res
        x = self.bn2(x)
        return x


class AttnFusion(nn.Module):
    def __init__(self, hidden_dim):
        super(AttnFusion, self).__init__()
        self.attn1 = nn.Linear(hidden_dim, hidden_dim)
        self.attn2 = nn.Linear(hidden_dim, hidden_dim)
        self.attn_score1 = nn.Linear(hidden_dim, 1, bias=False)
        self.attn_score2 = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, emb1, emb2):
        attn_score1 = self.attn_score1(F.tanh(self.attn1(emb1)))
        attn_score2 = self.attn_score2(F.tanh(self.attn2(emb2)))
        attn_score = torch.cat([attn_score1, attn_score2], dim=-1)
        attn_score = F.softmax(attn_score, dim=-1)
        attn_score1, attn_score2 = torch.split(attn_score, attn_score1.size(-1), dim=-1)
        x = attn_score1 * emb1 + attn_score2 * emb2
        return x
    
    
class Discriminator(nn.Module):
    def __init__(self, h_dim):
        super(Discriminator, self).__init__()
        self.score_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))
        self.act_fn = nn.Sigmoid()
        self.h_dim = h_dim
        self.reset_parameters()
        
    def reset_parameters(self):
        uniform(self.h_dim, self.score_weight)
        
    def forward(self, emb1, emb2):
        score = self.act_fn(torch.matmul(emb1, torch.matmul(self.score_weight, emb2.t())))
        score = score.mean(dim=0).unsqueeze(0)
        return score
    

class RGSCLNet(nn.Module):
    def __init__(self, in_dim_poi, in_dim_building, h_dim, bsz: int, lyr_num: int,
                 backbone: Literal['gcn', 'gat', 'sage'], attn_heads: int, act_fn: str, alpha: float, miu: float):
        super(RGSCLNet, self).__init__()
        self.bsz = bsz
        self.alpha = alpha
        self.miu = miu
        self.ecd_poi = Encoder(in_dim_poi, h_dim, lyr_num, backbone, attn_heads, act_fn)
        self.ecd_building = Encoder(in_dim_building, h_dim, lyr_num, backbone, attn_heads, act_fn)
        self.disc_inter_region = Discriminator(h_dim)
        self.disc_intra_region = Discriminator(h_dim)
        self.attn_fusion = AttnFusion(h_dim)
        self.proj_poi = nn.Sequential(nn.Linear(h_dim, h_dim),
                                      nn.ELU(),
                                      nn.Linear(h_dim, h_dim))
        self.proj_building = nn.Sequential(nn.Linear(h_dim, h_dim),
                                           nn.ELU(),
                                           nn.Linear(h_dim, h_dim))
        self.proj_region = nn.Sequential(nn.Linear(h_dim, h_dim),
                                         nn.ELU(),
                                         nn.Linear(h_dim, h_dim))
        self.bn1 = nn.BatchNorm1d(h_dim)
        self.bn2 = nn.BatchNorm1d(h_dim)
        self.eps = 1e-7
        self.reset_parameters()
        
    def reset_parameters(self):
        reset(self.ecd_poi)
        reset(self.ecd_building)
    
    def _intra_region_disc(self, poi_emb, building_emb, neg_poi_emb, neg_building_emb, poi_view_region_emb,
                           building_view_region_emb):
        pos_score = (self.disc_intra_region(poi_emb, building_view_region_emb) + self.disc_intra_region(
            building_emb, poi_view_region_emb)) / 2
        neg_score = self.disc_intra_region(neg_poi_emb, building_view_region_emb) + self.disc_intra_region(
            neg_building_emb, poi_view_region_emb) / 2
        return pos_score, neg_score
    
    def _inter_region_disc(self, region_emb, neg_region_emb, aug_region_emb):
        pos_score = self.disc_inter_region(aug_region_emb, region_emb)
        neg_score = self.disc_inter_region(neg_region_emb, region_emb)
        return pos_score, neg_score
    
    def _l2_loss(self):
        loss = .0
        for pms in self.parameters():
            loss = loss + torch.norm(pms)
        return loss
    
    def get_region_emb(self, poi_graphs, building_graphs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        poi_emb = self.ecd_poi(poi_graphs.x, poi_graphs.edge_index)
        building_emb = self.ecd_building(building_graphs.x, building_graphs.edge_index)
        
        region_emb_ls = []
        poi_view_emb_ls = []
        building_view_emb_ls = []
        
        for region_id in poi_graphs.region_id.unique():
            poi_view_emb_idx = ((poi_graphs.is_dummy_node == 1) & (poi_graphs.region_id == region_id)
                                ).nonzero(as_tuple=True)[0]
            building_view_emb_idx = ((building_graphs.is_dummy_node == 1) & (building_graphs.region_id == region_id)
                                     ).nonzero(as_tuple=True)[0]
            region_emb_ls.append(self.attn_fusion(poi_emb[poi_view_emb_idx], building_emb[building_view_emb_idx]))
            
            poi_view_emb_ls.append(poi_emb[poi_view_emb_idx])
            building_view_emb_ls.append(building_emb[building_view_emb_idx])
            
        region_emb = torch.cat(region_emb_ls, dim=0)
        poi_view_region_emb = torch.cat(poi_view_emb_ls, dim=0)
        building_view_region_emb = torch.cat(building_view_emb_ls, dim=0)
        region_emb = self.bn1(region_emb)
        return region_emb, poi_view_region_emb, building_view_region_emb
        
    def criterion(self, poi_emb, building_emb, region_emb, aug_region_emb, poi_graphs, building_graphs):
        poi_emb = self.proj_poi(poi_emb)
        building_emb = self.proj_building(building_emb)
        region_emb = self.proj_region(region_emb)
        aug_region_emb = self.proj_region(aug_region_emb)
                
        inter_pos_score_ls, inter_neg_score_ls = [], []
        for region_emb_idx in range(region_emb.size(0)):
            rand_idx = torch.randperm(region_emb.size(0))
            neg_region_emb_idx = rand_idx[rand_idx != region_emb_idx]
            inter_pos_score, inter_neg_score = self._inter_region_disc(region_emb[region_emb_idx],
                                                                       region_emb[neg_region_emb_idx],
                                                                       aug_region_emb[region_emb_idx])
            inter_pos_score_ls.append(inter_pos_score)
            inter_neg_score_ls.append(inter_neg_score)
        inter_pos_score = torch.cat(inter_pos_score_ls, dim=0)
        inter_neg_score = torch.cat(inter_neg_score_ls, dim=0)
            
        intra_pos_score_ls, intra_neg_score_ls = [], []
        for region_id in poi_graphs.region_id.unique():
            poi_idx = ((poi_graphs.region_id == region_id) & (poi_graphs.is_dummy_node == 0)
                       ).nonzero(as_tuple=True)[0]
            poi_view_region_idx = ((poi_graphs.region_id == region_id) & (poi_graphs.is_dummy_node == 1)
                                   ).nonzero(as_tuple=True)[0]
            building_idx = ((building_graphs.region_id == region_id) & (building_graphs.is_dummy_node == 0)
                            ).nonzero(as_tuple=True)[0]
            building_view_region_idx = ((building_graphs.region_id == region_id) & (building_graphs.is_dummy_node == 1)
                                        ).nonzero(as_tuple=True)[0]
            
            neg_poi_idx = ((poi_graphs.region_id != region_id) & (poi_graphs.is_dummy_node == 0)
                           ).nonzero(as_tuple=True)[0]
            neg_building_idx = ((building_graphs.region_id != region_id) & (building_graphs.is_dummy_node == 0)
                                ).nonzero(as_tuple=True)[0]
        
            intra_pos_score, intra_neg_score = self._intra_region_disc(
                poi_emb[poi_idx], building_emb[building_idx], poi_emb[neg_poi_idx], building_emb[neg_building_idx],
                poi_emb[poi_view_region_idx], building_emb[building_view_region_idx]
            )
            
            intra_neg_score_ls.append(intra_neg_score)
            intra_pos_score_ls.append(intra_pos_score)
        
        intra_pos_score = torch.cat(intra_pos_score_ls, dim=0)
        intra_neg_score = torch.cat(intra_neg_score_ls, dim=0)
        
        loss_intra_region = -(
            torch.log(intra_pos_score + self.eps).mean() + torch.log(1 - intra_neg_score + self.eps).mean())
        loss_inter_region = -(
            torch.log(inter_pos_score + self.eps).mean() + torch.log(1 - inter_neg_score + self.eps).mean())
        
        loss_l2 = self._l2_loss()
        
        loss = self.alpha*loss_inter_region + (1-self.alpha)*loss_intra_region + self.miu*loss_l2
        return loss
        
    def forward(self, poi_graphs, building_graphs):
        poi_emb = self.ecd_poi(poi_graphs.x, poi_graphs.edge_index, auger=False)
        building_emb = self.ecd_building(building_graphs.x, building_graphs.edge_index, auger=False)
        
        aug_poi_emb = self.ecd_poi(poi_graphs.x, poi_graphs.edge_index, auger=True)
        aug_building_emb = self.ecd_building(building_graphs.x, building_graphs.edge_index, auger=True)
        
        region_emb_ls = []
        aug_region_emb_ls = []
        for region_id in poi_graphs.region_id.unique():
            poi_view_region_idx = ((poi_graphs.region_id == region_id) & (poi_graphs.is_dummy_node == 1)
                                   ).nonzero(as_tuple=True)[0]
            building_view_region_idx = ((building_graphs.region_id == region_id) & (building_graphs.is_dummy_node == 1)
                                        ).nonzero(as_tuple=True)[0]
            region_emb_ls.append(self.attn_fusion(poi_emb[poi_view_region_idx],
                                                  building_emb[building_view_region_idx]))
            aug_region_emb_ls.append(self.attn_fusion(aug_poi_emb[poi_view_region_idx],
                                                      aug_building_emb[building_view_region_idx]))
        region_emb = torch.cat(region_emb_ls, dim=0)
        aug_region_emb = torch.cat(aug_region_emb_ls, dim=0)
        
        region_emb = self.bn1(region_emb)
        aug_region_emb = self.bn2(aug_region_emb)
        
        return poi_emb, building_emb, region_emb, aug_region_emb, poi_graphs, building_graphs
        