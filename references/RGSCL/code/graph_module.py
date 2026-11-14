from torch_geometric.data import Data
import torch
import numpy as np
import geopandas as gpd
from scipy.spatial import Delaunay
from config import CONFIG
from shapely.geometry import Point, LineString
import shapely
import pandas as pd
from torch_geometric.utils import to_undirected
from typing import Dict


class Graph():
    def __init__(self, region_file: str, poi_file: str, building_file: str, crs_epsg: int):
        """
        :region_file: urban region shp file
        :poi_file: poi shp file
        :building_file: building shp file
        :crs_epsg: coordinate reference system, ref. https://epsg.io/, required projected crs
        """
        self.region = gpd.read_file(region_file).to_crs(f'EPSG:{crs_epsg}')
        self.poi = gpd.read_file(poi_file).to_crs(f'EPSG:{crs_epsg}')
        self.building = gpd.read_file(building_file).to_crs(f'EPSG:{crs_epsg}')
        self._preprocess()

        self.building_graphs: Dict[int, Data] = ...
        self.poi_graphs: Dict[int, Data] = ...
        
        self._get_poi_graph()
        self._get_building_graph()
        
        for poi_graph in self.poi_graphs.values():
            self._add_region_node(poi_graph)
            poi_graph.edge_index = to_undirected(poi_graph.edge_index)
        for building_graph in self.building_graphs.values():
            self._add_region_node(building_graph)
            building_graph.edge_index = to_undirected(building_graph.edge_index)
  
    def _preprocess(self):
        self.poi['poi_id'] = self.poi['OBJECTID']
        self.poi = self.poi[['poi_id', 'second_level', 'geometry']]
        self.poi['poi_lon'] = self.poi['geometry'].centroid.x
        self.poi['poi_lat'] = self.poi['geometry'].centroid.y
        self.poi.drop_duplicates(subset=['poi_lon', 'poi_lat'], keep='first', inplace=True)
        self._init_poi_feat()
        self.building = self.building[self.building['geometry'].apply(
            lambda x: isinstance(x, shapely.geometry.Polygon))]
        self.building['building_id'] = self.building['OBJECTID']
        self.building = self.building[['building_id', 'geometry']]
        self.building['building_lon'] = self.building['geometry'].centroid.x
        self.building['building_lat'] = self.building['geometry'].centroid.y
        self._init_building_feat()
        self.region = self.region[['UUID', 'geometry']]
        self._sjoin()
        self._clean()
        
    def _clean(self, threshold=10):
        # drop sparse region (sample num < threshold)
        self.poi = self.poi.groupby('UUID').filter(lambda x: len(x) >= threshold)
        self.building = self.building.groupby('UUID').filter(lambda x: len(x) >= threshold)
        poi_region_id = set(self.poi['UUID'])
        building_region_id = set(self.building['UUID'])
        shared_region_id = poi_region_id & building_region_id
        self.poi = self.poi[self.poi['UUID'].isin(shared_region_id)]
        self.building = self.building[self.building['UUID'].isin(shared_region_id)]
        self.region = self.region[self.region['UUID'].isin(shared_region_id)]
        
    def _init_poi_feat(self):
        # init poi feature with one-hot encoding
        type_map = {_type: _idx for _idx, _type in enumerate(list(self.poi['second_lvl'].unique()))}
        coded_types = self.poi['second_lvl'].map(type_map)
        self.poi['type_emb'] = coded_types.apply(lambda x: np.eye(len(type_map))[x])
        
    def _init_building_feat(self):
        # building feature engineering, ref. DOI: 10.1016/j.isprsjprs.2019.02.010
        self.building['area'] = self.building['geometry'].area
        self.building['perimeter'] = self.building['geometry'].length
        self.building['radius'] = self.building['geometry'].apply(
            lambda x: np.mean([x.centroid.distance(Point(y)) for y in x.exterior.coords]))
        self.building['orientation'] = self.building['geometry'].apply(
            lambda x: np.arctan2(
                x.minimum_rotated_rectangle.exterior.coords[1][1] - x.minimum_rotated_rectangle.exterior.coords[0][1],
                x.minimum_rotated_rectangle.exterior.coords[1][0] - x.minimum_rotated_rectangle.exterior.coords[0][0]))
        self.building['compactness'] = self.building['geometry'].apply(lambda x: (x.area*4*3.14)/np.square(x.length))
        self.building['fractality'] = self.building['geometry'].apply(lambda x: 1-(np.log(x.area)/np.log(x.length))/2)
        self.building['concavity'] = self.building['area'] / self.building['geometry'].convex_hull.area
        self.building['elongation'] = self.building['geometry'].apply(
            lambda x: max(LineString(x.minimum_rotated_rectangle.exterior.coords[:2]).length,
                          LineString(x.minimum_rotated_rectangle.exterior.coords[1:3]).length
                          ) / min(LineString(x.minimum_rotated_rectangle.exterior.coords[:2]).length,
                                  LineString(x.minimum_rotated_rectangle.exterior.coords[1:3]).length)
            if min(LineString(x.minimum_rotated_rectangle.exterior.coords[:2]).length,
                   LineString(x.minimum_rotated_rectangle.exterior.coords[1:3]).length) != 0 else 0)
        self.building = self.building[['building_id', 'building_lon', 'building_lat', 'area', 'perimeter', 'radius',
                                       'orientation', 'compactness', 'fractality', 'concavity', 'elongation',
                                       'geometry']]
        
    def _sjoin(self):
        self.poi = gpd.sjoin(self.poi, self.region, how="inner", op="intersects")
        self.poi.drop(columns=['index_right', 'geometry'], inplace=True)
        self.poi = pd.DataFrame(self.poi)
        self.building = gpd.sjoin(self.building, self.region, how="inner", op="intersects")
        self.building.drop(columns=['index_right', 'geometry'], inplace=True)
        self.building = pd.DataFrame(self.building)
    
    @staticmethod
    def _dt_edge(coords) -> torch.LongTensor:
        tri = Delaunay(coords)
        edges = set()
        for edge_a, edge_b, edge_c in tri.simplices:
            edges.update([(edge_a, edge_b), (edge_b, edge_c), (edge_c, edge_a)])
        edge_index = np.array(list(edges))  # [num_edges, 2]
        edge_index = torch.tensor(edge_index, dtype=torch.long).T  # [2, num_edges]
        return edge_index
    
    def _add_region_node(self, poi_building_graph):
        poi_building_idx = torch.nonzero(poi_building_graph.is_dummy_node == 0, as_tuple=True)[0]
        poi_building_view_region_emb = torch.mean(poi_building_graph.x[poi_building_idx], dim=0).unsqueeze(0)
        poi_building_graph.x = torch.cat([poi_building_graph.x, poi_building_view_region_emb], dim=0)
        new_region_edge = torch.tensor(
            [[poi_building_graph.num_nodes-1] * len(poi_building_idx), poi_building_idx], dtype=torch.long)
        poi_building_graph.edge_index = torch.cat([poi_building_graph.edge_index, new_region_edge], dim=1)
        poi_building_graph.region_id = torch.cat([poi_building_graph.region_id,
                                                  poi_building_graph.region_id[0].unsqueeze(0)], dim=0)
        poi_building_graph.is_dummy_node = torch.cat([poi_building_graph.is_dummy_node, torch.tensor([1])], dim=0)
    
    def _get_poi_graph(self):
        self.poi_graphs = {}
        for region_id, group in self.poi.groupby('UUID'):
            group_df = pd.DataFrame(group.reset_index(drop=True))
            coords = group_df[['poi_lon', 'poi_lat']].values
            edge_index = self._dt_edge(coords)
            self.poi_graphs[region_id] = Data(
                x=torch.stack([torch.tensor(_, dtype=torch.float) for _ in group_df['type_emb'].tolist()], dim=0),
                edge_index=edge_index,
                region_id=torch.tensor(group['UUID'].values, dtype=torch.long),
                is_dummy_node=torch.tensor([0] * len(coords), dtype=torch.long)
            )
        
    def _get_building_graph(self):
        self.building_graphs = {}
        for region_id, group in self.building.groupby('UUID'):
            group_df = pd.DataFrame(group.reset_index(drop=True))
            coords = group_df[['building_lon', 'building_lat']].values
            edge_index = self._dt_edge(coords)
            self.building_graphs[region_id] = Data(
                x=torch.tensor(group_df[['area', 'perimeter', 'radius', 'orientation', 'compactness', 'fractality',
                                         'concavity', 'elongation']].values, dtype=torch.float),
                edge_index=edge_index,
                region_id=torch.tensor(group['UUID'].values, dtype=torch.long),
                is_dummy_node=torch.tensor([0] * len(coords), dtype=torch.long)
            )

    
if __name__ == "__main__":
    import time
    t0 = time.time()
    
    cfg = CONFIG()
    poi_file = f'{cfg.filepath}/bj_poi.shp'
    building_file = f'{cfg.filepath}/bj_building.shp'
    region_file = f'{cfg.filepath}/bj_euluc.shp'
    
    graph = Graph(region_file, poi_file, building_file, 4548)
    torch.save(graph.poi_graphs, f'{cfg.filepath}/poi_graphs.pt')
    torch.save(graph.building_graphs, f'{cfg.filepath}/building_graphs.pt')
    
    print(f'>> elapsed time: {time.time()-t0:.2f}s')
    