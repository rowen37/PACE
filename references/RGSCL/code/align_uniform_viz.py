import matplotlib.pyplot as plt
import torch
from config import CONFIG
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
import seaborn as sns
import numpy as np
from typing import Literal

plt.rcParams['font.family'] = 'Arial'
plt.tick_params(left=False, bottom=False)


def uniformity_kde_viz(emb: np.ndarray, cfg: CONFIG):
    emb_reduc = TSNE(n_components=2, random_state=cfg.seed).fit_transform(emb)
    emb_normalized = normalize(emb_reduc, axis=1)
    plt.figure(figsize=(6, 6))
    sns.kdeplot(x=emb_normalized[:, 0],
                y=emb_normalized[:, 1],
                fill=True,
                cmap='viridis',
                alpha=.8,
                linewidths=0,
                color=(0, 0, 0, 0),
                levels=100,
                bw_adjust=.3)
    
    plt.title('', fontsize=18)
    plt.ylim(-1.2, 1.2)
    plt.xlim(-1.2, 1.2)
    plt.xticks([-1, -0.5, 0, 0.5, 1], ['-1', '-0.5', '0', '0.5', '1'])
    plt.yticks([-1, -0.5, 0, 0.5, 1], ['-1', '-0.5', '0', '0.5', '1'])
    plt.xlabel('')
    plt.ylabel('')
    plt.tick_params(labelsize=18)
    plt.savefig(f'{cfg.filepath}/figure/uniformity_gaussian_kde_{cfg.args["model_name"]}.png',
                dpi=300, bbox_inches='tight', pad_inches=.1)

    radian = np.arctan2(emb_normalized[:, 1], emb_normalized[:, 0])
    plt.figure(figsize=(6, 2))
    plt.xlim(-np.pi, np.pi)
    plt.ylim(bottom=0, top=.4)
    plt.yticks([0, .1, .2, .3, .4], ['0', '0.1', '0.2', '0.3', '0.4'])
    plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], ['−π', '−π/2', '0', 'π/2', 'π'])
    sns.kdeplot(radian, color='skyblue', linewidth=2, fill=True, alpha=.7)
    plt.tick_params(labelsize=18)
    plt.xlabel('Radian', fontsize=18)
    plt.ylabel('Density', fontsize=18)
    plt.title('', fontsize=18)
    plt.savefig(f'{cfg.filepath}/figure/uniformity_vmf_kde_{cfg.args["model_name"]}.png',
                dpi=300, bbox_inches='tight', pad_inches=.1)
    

def alignment_hist_viz(emb_pos_pair: np.ndarray, cfg: CONFIG):
    # emb_pos_pair: [emb_num, emb_dim, 2]
    l2_dist = np.linalg.norm(emb_pos_pair[:, 0] - emb_pos_pair[:, 1], axis=1)
    plt.figure(figsize=(6, 4))
    sns.histplot(l2_dist, bins=30, color='skyblue', alpha=0.7)
    mean = np.mean(l2_dist)
    plt.axvline(mean, color='red', linestyle='--', label=f'µ: {mean:.2f}', alpha=.8)
    plt.legend(fontsize=18)
    plt.title('', fontsize=18)
    plt.xlabel(r'$\ell_2$ distance', fontsize=18)
    plt.ylabel('Counts', fontsize=18)
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.tick_params(labelsize=18)
    plt.savefig(f'{cfg.filepath}/figure/alignment_hist.png',
                dpi=300, bbox_inches='tight', pad_inches=.1)


if __name__ == '__main__':
    cfg = CONFIG()
    
    def main(f: Literal['unif', 'alig']):
        cfg.args['model_name'] = 'RGSC'
        if f == 'unif':
            rgsc_emb = torch.load(f'{cfg.filepath}/region_emb.pt', map_location=torch.device(cfg.device)
                                  ).requires_grad_(False).numpy()
            uniformity_kde_viz(rgsc_emb, cfg)

        elif 'alig':
            rgsc_poi_view_emb = torch.load(f'{cfg.filepath}/poi_view_region_emb.pt',
                                           map_location=torch.device(cfg.device)).requires_grad_(False)
            rgsc_building_view_emb = torch.load(f'{cfg.filepath}/building_view_region_emb.pt',
                                                map_location=torch.device(cfg.device)).requires_grad_(False)
            rgsc_pair = torch.cat(
                (rgsc_poi_view_emb.unsqueeze(-1), rgsc_building_view_emb.unsqueeze(-1)), dim=-1).numpy()
            alignment_hist_viz(rgsc_pair, cfg)

    main('alig')
    main('unif')
