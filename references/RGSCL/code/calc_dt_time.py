import numpy as np
import geopandas as gpd
from scipy.spatial import Delaunay
from config import CONFIG
import time


def calc_empirical_time(shp_file, num):
    """
    :param shp_file: path to the shapefile
    :param num: number of points
    :return: empirical time
    """
    pnt = gpd.read_file(shp_file).to_crs(f'EPSG:4548')
    pnt['x'] = pnt['geometry'].centroid.x
    pnt['y'] = pnt['geometry'].centroid.y
    coords = pnt[['x', 'y']].values
    total_time = 0
    for i in range(10):
        coords_sampled = coords[np.random.choice(coords.shape[0], num, replace=False)]
        t0 = time.time()
        Delaunay(coords_sampled)
        elapsed_time = time.time() - t0
        total_time += elapsed_time
        print(f'Exp. {i+1}: {elapsed_time:.6f}s')
    print(f'Mean time: {total_time / 10:.6f}s')
    return total_time / 10


def calc_theoretical_time(num, num_base, t_base):
    """
    :param num: number of points
    :param num_base: base number of points
    :param t_base: base time
    :return: theoretical time
    """
    theo_time = t_base * ((num * np.log2(num)) / (num_base * np.log2(num_base)))
    print(f'Theoretical time: {theo_time:.6f}s')
    return theo_time


def calc_empirical_theory_ratio(t_empirical, t_theory):
    """
    :param t_empirical: empirical time
    :param t_theory: theoretical time
    :return: ratio of empirical time to theoretical time
    """
    ratio = t_empirical / t_theory
    print(f'Ratio of empirical time to theoretical time: {ratio:.6f}')
    return ratio

    
if __name__ == "__main__":
    cfg = CONFIG()
    cfg.set_seed()
    poi_file = f'{cfg.filepath}/bj_poi.shp'

    num_ls = [1_000, 5_000, 10_000, 50_000, 100_000, 200_000]
    for num in num_ls:
        calc_empirical_time(poi_file, num)

    