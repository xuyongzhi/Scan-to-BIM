import numpy as np
from plyfile import PlyData
import open3d as o3d
from utils_data3d.lib.pc_utils import COLOR_MAP_RGB
from utils_data3d.datasets.preprocess_stanford3d import Stanford3DDatasetConverter, make_pcd
CLASSES = Stanford3DDatasetConverter.CLASSES

def label2color(labels):
  colors = np.array([COLOR_MAP_RGB[l] for l in labels])
  return colors


def test():
    path = '/DS/Stanford3D/processed_instance/Area_1'
    filepath = f'{path}/conferenceRoom_1.ply'
    #filepath = f'{path}/hallway_8.ply'
    plydata = PlyData.read(filepath)
    data = plydata.elements[0].data
    coords = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T
    feats = np.array([data['red'], data['green'], data['blue']], dtype=np.float32).T
    categories = np.array(data['label'], dtype=np.int32)
    instances = np.array(data['instance'], dtype=np.int32)

    pcd0 = make_pcd(coords, feats, categories, instances, 'category')
    #o3d.visualization.draw_geometries([pcd0])

    # by category
    cat_min = categories.min()
    cat_max = categories.max()

    for cat in range(cat_min, cat_max+1):
      print(CLASSES[cat])
      mask = categories == cat
      if mask.sum()== 0:
        continue
      pcd_c = make_pcd(coords[mask], feats[mask], categories[mask], instances[mask], 'instance')
      o3d.visualization.draw_geometries([pcd_c])
    pass

if __name__ == '__main__':
  test()
