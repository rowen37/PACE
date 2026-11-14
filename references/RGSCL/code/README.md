# **RGSCL**

This project is linked to a submission to *International Journal of Geographical Information Science*, entitled “Urban region representation learning via dual spatial contrasts”, which proposes a region graph spatial contrastive learning (RGSCL) framework for self-supervised region representation learning.

The project includes both the [data](#1-data), [source code](#2-codes), and [reproduction steps](#223-reproduction-steps-for-findings-in-this-work) that support the RGSCL.

## 1 Data 

Two .shp files for RGSCL pretraining and three .csv files for geographic mapping finetuning.

| Filename | Description |
| ---- | ---- |
| mocked_poi.shp | This file is the mocked POI data. |
| building.shp | This file is the building footprint data used in this study. |
| mocked_func.csv | This file is the mocked urban function ground truth. |
| mocked_price.csv | This file is the mocked housing price ground truth. |
| mocked_checkin.csv | This file is the mocked region popularity ground truth. |
| euluc_subsets.shp | This file is the subset of euluc data. |

> Note: As we are not permitted to publish the raw POI data, EULUC data, housing price data, and check-in data due to their terms and conditions, we provide mocked data instead. 

## 2 Codes

### 2.1 Environment

Please use Miniconda or Anaconda.
- python == 3.11.10
- torch == 2.4.1.post3
- torch-geometric == 2.6.1
- geopandas == 0.14.4
- pandas == 2.2.2
- numpy == 1.26.4
- scikit-learn == 1.5.1
- scipy == 1.13.1
- shapely == 2.0.6

### 2.2 Usage

#### 2.2.1 Training

Run the following command in the terminal to quick start.

```sh
python train.py
```

The arguments that can be specified in `train.py` are as follows:
- `--epoch`: the number of training epochs.
- `--lr`: the learning rate for optimization.
- `--dim`: The hidden dimension size of the model.
- `--bsz`: The batch size for training.
- `--lyr`: The number of GNN layers in the model.
- `--attn_head`: the number of attention heads in GAT.
- `--act_fn`: The activation function.
- `--alpha`: the hyperpms to balance intra- and inter-region loss.
- `--miu`: The strength of the l2 regularization.

The graph data class should contain the following attributes:

poi_graphs:
- `x`: POI initial features, which can be POI category embeddings (can even be one-hot vectors) or POI embeddings learned by other methods.
- `edge_index`: the POI-view graph structure, which is a tensor of shape (2, E), where E is the number of edges.
- `region_id`: the region id of each POI.
- `is_dummy_node`: A boolean value indicating whether each node is a dummy node.

building_graphs:
- `x`: building footprint initial features, which is calculated by morphology indices.
- `edge_index`: the building-view graph structure, which is a tensor of shape (2, $E$), where E is the number of edges.
- `region_id`: the region id of each building.
- `is_dummy_node`: A boolean value indicating whether each node is a dummy node.

#### 2.2.2 Evaluation

Run the following command in the terminal to quick start.

```sh
python evaluation.py
```

The arguments that can be specified in `evaluation.py` are as follows:
- `--task`: 'func', 'price', or 'checkin' represent the three evaluation tasks of urban function mapping, housing price mapping, and region popularity mapping respectively.
- `--repeats`: the number of repeated experiments.

#### 2.2.3 Reproduction steps for findings in this work

##### Steps to Tab. 1

Run the following command in the terminal to obtain the performance results of RGSCL on geographic mapping tasks.

```sh
python evaluation.py
```

The running results are stored in the eval_per_region_{`task`}.csv and eval_{`task`}.csv files under the path specified in the config file. Among them, eval_{`task`}.csv corresponds to the results in Tab. 1, and eval_per_region_{`task`}.csv records the average mapping accuracy of each region.

##### Steps to Fig. 6

Based on the obtained eval_per_region_{`task`}.csv, pass it to the `plt_map` function in `geo_mapping.py` to obtain the spatial visualization results of the geographic mapping performance evaluation and save it as eval_per_region_RGSC_`task`.png.
Note: You need to register the base map service API for [mapbox](www.mapbox.com) to define the variable `mapbox_api` in plt_map.

##### Steps to Fig. 7

Run the following command in the terminal to implement Fig. 7.

```sh
python align_uniform_viz.py
```

The argument that can be specified in `align_uniform_viz.py` are as follows:

- `f`: select 'alig' or 'unif' to specify the results of alignment or uniformity.

##### Steps to Fig. 8

Conduct a sensitivity study on the following three important hyperparameters of RGSCL:

- `h`: attention heads
- `d`: embedding dimension
- `λ`: loss function weight factor

There are the following ways to set hyperparameters:

- Adjust in `train.py`
```py
from config import CONFIG


cfg = CONFIG()
cfg.set_args()

for d in [16, 32, 64, 128, 256]:
    for h in [1, 2, 4, 8]:
        for λ in [.1, .3, .5, .7, .9]:
            cfg.args['dim'] = d
            cfg.args['attn_head'] = h
            cfg.args['alpha'] = λ
            ...  # train and evaluate RGSCL
```

- Adjust through command-line arguments.

- Modify the parameters in the `config.py`.

Instantiate RGSCL with different hyperparameter combinations, then train and evaluate it (i.e., repeat [Steps to Tab. 1](#steps-to-tab-1)) to obtain the parameter sensitivity results.

Run the following command to visualize the parameter sensitivity results and form Fig. 8.

```sh
python plt_pms_sensitivity.py
```

##### Steps to Tab. 3

Obtain the variant models of RGSCL simply through the following code:

- Variant (a): RGSCL w/o intra
```py
from config import CONFIG


cfg = CONFIG()
cfg.set_args()
cfg.args['alpha'] = 1
```

- Variant (b): RGSCL w/o inter
```py
from config import CONFIG


cfg = CONFIG()
cfg.set_args()
cfg.args['alpha'] = 0
```

- Variant (c): RGSCL w/o reg
```py
from config import CONFIG


cfg = CONFIG()
cfg.set_args()
cfg.args['miu'] = 0
```

- Variant (d): RGSCL-SL
```py
import torch
import torch.nn as nn
from torch_geometric.nn.inits import reset
from typing import Literal


class RGSC_SL(nn.Module):
    def __init__(self, h_dim, building_ecd, poi_ecd, out_dim):
        """
        hidden_dim (int): hidden dimension
        building_ecd (Encoder): building encoder
        poi_ecd (Encoder): poi encoder
        bsz (int): batch size
        """
        super(RGSC_SL, self).__init__()
        self.building_ecd = building_ecd
        self.poi_ecd = poi_ecd
        self.attn_fusion = AttnFusion(h_dim)
        self.mlp = nn.Sequential(nn.Linear(h_dim, h_dim),
                                 nn.BatchNorm1d(h_dim),
                                 nn.ReLU(),
                                 nn.Linear(h_dim, out_dim),
                                 )
        self.reset_parameters()
        
    def reset_parameters(self):
        reset(self.building_ecd)
        reset(self.poi_ecd)
    
    def criterion(self, region_emb, region_label, task: Literal['func', 'price', 'checkin']):
        loss_fn = {'func': nn.CrossEntropyLoss(), 'price': nn.MSELoss(), 'checkin': nn.MSELoss()}[task]
        loss = loss_fn(region_emb, region_label)
        return loss
        
    def forward(self, poi_graphs, building_graphs, label_dict: Dict[int, torch.Tensor]):
        
        poi_emb = self.poi_ecd(poi_graphs.x, poi_graphs.edge_index)
        building_emb = self.building_ecd(building_graphs.x, building_graphs.edge_index)
        
        region_emb_ls = []
        label_ls = []
        for region_id in poi_graphs.region_id.unique():
            poi_view_emb_idx = ((poi_graphs.is_dummy_node == 1) & (poi_graphs.region_id == region_id)
                                ).nonzero(as_tuple=True)[0]
            building_view_emb_idx = ((building_graphs.is_dummy_node == 1) & (building_graphs.region_id == region_id)
                                     ).nonzero(as_tuple=True)[0]
            if region_id.item() in label_dict:
                region_emb_ls.append(self.attn_fusion(poi_emb[poi_view_emb_idx], building_emb[building_view_emb_idx]))
                label_ls.append(label_dict[region_id.item()].unsqueeze(0))
            
        region_emb = torch.cat(region_emb_ls, dim=0)
        label = torch.cat(label_ls, dim=0)
        
        output = self.mlp(region_emb)
        return output, label
```

- Variant (e): RGSCL-GCN
```diff
# Adjust the code related to model instantiation in train.py

model = RGSCLNet(
        in_dim_poi=poi_dim,
        in_dim_building=building_dim,
        h_dim=cfg.args['dim'],
        bsz=cfg.args['bsz'],
        lyr_num=cfg.args['lyr'],
-       backbone='gat',
+       backbone='gcn',
        attn_heads=cfg.args['attn_head'],
        act_fn=cfg.args['act_fn'],
        alpha=cfg.args['alpha'],
        miu=cfg.args['miu'],
    ).to(cfg.device)
```

- Variant (f): RGSCL-mean
```diff
# Adjust the definition code of the forward method in model.py

def forward(self, poi_graphs, building_graphs):
        poi_emb = self.ecd_poi(poi_graphs.x, poi_graphs.edge_index, auger=False)
        building_emb = self.ecd_building(building_graphs.x, building_graphs.edge_index, auger=False)
        
        aug_poi_emb = self.ecd_poi(poi_graphs.x, poi_graphs.edge_index, auger=True)
        aug_building_emb = self.ecd_building(building_graphs.x, building_graphs.edge_index, auger=True)
        
        region_emb_ls = []
        aug_region_emb_ls = []
        for region_id in poi_graphs.region_id.unique():
-           poi_view_region_idx = ((poi_graphs.region_id == region_id) & (poi_graphs.is_dummy_node == 1)
-                                  ).nonzero(as_tuple=True)[0]
-           building_view_region_idx = ((building_graphs.region_id == region_id) & (building_graphs.is_dummy_node == 1)
-                                       ).nonzero(as_tuple=True)[0]
-           region_emb_ls.append(self.attn_fusion(poi_emb[poi_view_region_idx],
-                                                 building_emb[building_view_region_idx]))
-           aug_region_emb_ls.append(self.attn_fusion(aug_poi_emb[poi_view_region_idx],
-                                                     aug_building_emb[building_view_region_idx]))
+           poi_view_region_idx = ((poi_graphs.region_id == region_id) & (poi_graphs.is_dummy_node == 0)
+                                  ).nonzero(as_tuple=True)[0]
+           building_view_region_idx = ((building_graphs.region_id == region_id) & (building_graphs.is_dummy_node == 0)
+                                       ).nonzero(as_tuple=True)[0]
+           region_emb_ls.append(self.attn_fusion(poi_emb[poi_view_region_idx].mean(dim=0).unsqueeze(0),
+                                                 building_emb[building_view_region_idx].mean(dim=0).unsqueeze(0)))
+           aug_region_emb_ls.append(self.attn_fusion(aug_poi_emb[poi_view_region_idx].mean(dim=0).unsqueeze(0),
+                                                     aug_building_emb[building_view_region_idx].mean(dim=0).unsqueeze(0)))
        region_emb = torch.cat(region_emb_ls, dim=0)
        aug_region_emb = torch.cat(aug_region_emb_ls, dim=0)
        
        region_emb = self.bn1(region_emb)
        aug_region_emb = self.bn2(aug_region_emb)
        
        return poi_emb, building_emb, region_emb, aug_region_emb, poi_graphs, building_graphs
```

Execute [Steps to Tab. 1](#steps-to-tab-1) for each model variant to obtain the ablation results.

##### Steps to Tab. 6

Run the following command in the terminal.

```sh
python calc_dt_time.py
```

## 3 Structure
```
RGSCL
├── Data
|   ├── mocked_poi.shp === This file is the mocked POI data.
|   ├── building.shp === This file is the building footprint data used in this study.
|   ├── mocked_func.csv === This file is the mocked urban function ground truth.
|   ├── mocked_price.csv === This file is the mocked housing price ground truth.
|   └── mocked_checkin.csv === This file is the mocked region popularity ground truth.
|   └── euluc_subsets.shp === This file is the subset of euluc data.
└── Code
|	├── config.py  === Configuration setting for RGSCL implementation.
|	├── graph_module.py === Construct multi-view region graph.
|	├── utils.py === Defining of utility functions.
|	├── model.py === Defining of the RGSCL model.
|	├── geomapping.py === Visualize the performance results of geographic mapping.
|	├── plt_pms_sensitivity.py === Visualize the results of parameter sensitivity analysis.
|	├── align_uniform_viz.py === Visualize the results of representation space study.
|	├── train.py === Run and get region embeddings.
|	└── evaluation.py === Evaluate the region embeddings on three geographic mapping tasks.
└── README.md === This is the project description document.
```