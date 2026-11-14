import torch
import math
from model import RGSCLNet
import torch.nn as nn
import pytorch_warmup as warmup
from config import CONFIG
import time
from torch.utils.data import DataLoader
from torch_geometric.data import Batch


class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, poi_graphs, building_graphs):
        self.poi_graphs = poi_graphs
        self.building_graphs = building_graphs
        self.region_ids = list(poi_graphs.keys())

    def __len__(self):
        return len(self.region_ids)
    
    def __getitem__(self, idx):
        region_id = self.region_ids[idx]
        return self.poi_graphs[region_id], self.building_graphs[region_id]


def collate_fn(batch):
    poi_graphs, building_graphs = zip(*batch)
    return Batch.from_data_list(poi_graphs), Batch.from_data_list(building_graphs)
    

def trainer(cfg: CONFIG):
    t0 = time.time()
    cfg.set_args()
    
    building_graphs = torch.load(f'{cfg.filepath}/building_graphs.pt', map_location=torch.device(cfg.device))
    poi_graphs = torch.load(f'{cfg.filepath}/poi_graphs.pt', map_location=torch.device(cfg.device))
    
    building_feat = torch.cat([building_graph.x for building_graph in building_graphs.values()], dim=0)
    for building_graph in building_graphs.values():
        building_graph.x = (building_graph.x - building_feat.mean(dim=0)) / building_feat.std(dim=0)
    
    poi_dim = poi_graphs[list(poi_graphs.keys())[0]].x.size(1)
    building_dim = building_graphs[list(building_graphs.keys())[0]].x.size(1)
    
    data_ldr = DataLoader(GraphDataset(poi_graphs, building_graphs),
                          batch_size=cfg.args['bsz'],
                          shuffle=True,
                          collate_fn=collate_fn,
                          )
    
    model = RGSCLNet(
        in_dim_poi=poi_dim,
        in_dim_building=building_dim,
        h_dim=cfg.args['dim'],
        bsz=cfg.args['bsz'],
        lyr_num=cfg.args['lyr'],
        backbone='gat',
        attn_heads=cfg.args['attn_head'],
        act_fn=cfg.args['act_fn'],
        alpha=cfg.args['alpha'],
        miu=cfg.args['miu'],
    ).to(cfg.device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.args['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=cfg.args['gamma'])
    warmup_scheduler = warmup.LinearWarmup(optimizer, cfg.args['warmup_period'])
    
    model.train()
    model_save = ...
    loss_min = math.inf
    for epoch in range(cfg.args['epoch']):
        for batched_poi_graph, batched_building_graph in data_ldr:
            optimizer.zero_grad()
            outputs = model(batched_poi_graph, batched_building_graph)
            loss = model.criterion(*outputs)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            with warmup_scheduler.dampening():
                scheduler.step()
            if loss.item() < loss_min:
                model_save = model
                loss_min = loss.item()

    print(f'>> RGSCL training finished.')
    print(f'>> Training elapse: {(time.time()-t0)/60: .2f} (min)')
    
    region_emb_ls = []
    region_id_ls = []
    model_save.eval()
    for region_id in poi_graphs.keys():
        outputs = model_save.get_region_emb(poi_graphs[region_id], building_graphs[region_id])
        region_emb_ls.append(outputs[0])
        region_id_ls.append(region_id)
    region_emb = torch.cat(region_emb_ls, dim=0)
    region_id = torch.tensor(region_id_ls, dtype=torch.long)
    torch.save(region_emb, f'{cfg.filepath}/region_emb.pt')
    torch.save(region_id, f'{cfg.filepath}/region_id.pt')
    

if __name__ == '__main__':
    trainer(CONFIG())
    