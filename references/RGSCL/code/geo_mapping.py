import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
import pandas as pd
import numpy as np
from typing import Literal
from config import CONFIG


def plt_map(eval_result_name: str, eval_indice: Literal['Accuracy', 'Absolute error']):
    region_boundary = gpd.read_file(f'{CONFIG().filepath}/study_area_boundry.shp').to_crs(epsg=4548)
    region = gpd.read_file(f'{CONFIG().filepath}/bj_euluc.shp').to_crs(epsg=4548)
    eval_result = pd.read_csv(f'{CONFIG().filepath}/{eval_result_name}.csv')
    region = region.merge(eval_result, left_on='UUID', right_on='region_id')
    
    region['eval_result'] += np.random.uniform(0, 1e-5, size=region.shape[0])
    if eval_indice == 'Accuracy':
        value_max, value_min = region['eval_result'].max(), region['eval_result'].min()
        bins = np.linspace(value_min, value_max, 6)
        labels = ['≤0.2', '≤0.4', '≤0.6', '≤0.8', '≤1']
        region['eval_level'] = pd.cut(region['eval_result'], bins=bins, labels=labels)
    elif eval_indice == 'Absolute error':
        bins = pd.qcut(region['eval_result'], q=5)
        labels = [f'≤{int(x.right)}' for x in bins.cat.categories]
        region['eval_level'] = pd.qcut(region['eval_result'], q=5, labels=labels)
    region = region.dropna(subset=['eval_level'])
    color_map = {0: '#44045A', 1: '#30688D', 2: '#35B777', 3: '#91D542', 4: '#F8E620'}
    colors = {_label: color_map[idx] for idx, _label in enumerate(labels)}
    region['color'] = region['eval_level'].map(colors)

    fig, ax = plt.subplots(figsize=(6, 6))
    region.plot(ax=ax, color=region['color'])
    region = region.to_crs(epsg=4548)
    region_boundary.plot(ax=ax, edgecolor='#FF7A7F', facecolor='none')
    mapbox_api = ...  # TODO: mapbox api
    ctx.add_basemap(ax, source=f'{mapbox_api}', crs=region.crs.to_string())
    ax.set_axis_off()

    for eval_level, color in colors.items():
        ax.plot([], [], color=color, label=eval_level, linewidth=8)
    
    ax.legend(title=f'{eval_indice}', loc='lower right', fontsize=10, frameon=True, title_fontsize='medium',
              handlelength=.7)
    plt.savefig(f'{CONFIG().filepath}/figure/{eval_result_name}.png', pad_inches=0, bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    plt_map('eval_per_region_RGSC_func', 'Accuracy')
    plt_map('eval_per_region_RGSC_price', 'Absolute error')
    plt_map('eval_per_region_RGSC_checkin', 'Absolute error')
