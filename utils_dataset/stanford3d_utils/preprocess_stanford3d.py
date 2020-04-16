import glob
import numpy as np
import os

from tqdm import tqdm

from utils_dataset.lib.pc_utils import save_point_cloud,  points_to_bbox
from tools.debug_utils import _show_3d_points_bboxes_ls, _show_3d_points_lines_ls

import MinkowskiEngine as ME

from collections import defaultdict
import open3d as o3d

STANFORD_3D_IN_PATH = '/cvgl/group/Stanford3dDataset_v1.2/'
STANFORD_3D_IN_PATH = '/DS/Stanford3D/Stanford3dDataset_v1.2_Aligned_Version/'
STANFORD_3D_OUT_PATH = '/DS/Stanford3D/aligned_processed_instance'

STANFORD_3D_TO_SEGCLOUD_LABEL = {
    4: 0,
    8: 1,
    12: 2,
    1: 3,
    6: 4,
    13: 5,
    7: 6,
    5: 7,
    11: 8,
    3: 9,
    9: 10,
    2: 11,
    0: 12,
}


class Stanford3DDatasetConverter:

  CLASSES = [
      'clutter', 'beam', 'board', 'bookcase', 'ceiling', 'chair', 'column', 'door', 'floor', 'sofa',
      'stairs', 'table', 'wall', 'window'
  ]
  TRAIN_TEXT = 'train'
  VAL_TEXT = 'val'
  TEST_TEXT = 'test'

  @classmethod
  def read_txt(cls, txtfile):
    # Read txt file and parse its content.
    with open(txtfile) as f:
      pointcloud = [l.split() for l in f]
    # Load point cloud to named numpy array.
    invalid_ids = [i for i in range(len(pointcloud)) if len(pointcloud[i]) != 6]
    if len(invalid_ids)>0:
      print('\n\n\n Found invalid points with len!=6')
      print(invalid_ids)
      print('\n\n\n')
    pointcloud = [p for p in pointcloud if len(p) == 6]
    pointcloud = np.array(pointcloud).astype(np.float32)
    assert pointcloud.shape[1] == 6
    xyz = pointcloud[:, :3].astype(np.float32)
    rgb = pointcloud[:, 3:].astype(np.uint8)
    return xyz, rgb

  @classmethod
  def convert_to_ply(cls, root_path, out_path):
    """Convert Stanford3DDataset to PLY format that is compatible with
    Synthia dataset. Assumes file structure as given by the dataset.
    Outputs the processed PLY files to `STANFORD_3D_OUT_PATH`.
    """

    txtfiles = glob.glob(os.path.join(root_path, '*/*/*.txt'))
    for txtfile in tqdm(txtfiles):
      file_sp = os.path.normpath(txtfile).split(os.path.sep)
      target_path = os.path.join(out_path, file_sp[-3])
      out_file = os.path.join(target_path, file_sp[-2] + '.ply')

      if os.path.exists(out_file):
        print(out_file, ' exists')
        continue

      annotation, _ = os.path.split(txtfile)
      subclouds = glob.glob(os.path.join(annotation, 'Annotations/*.txt'))
      coords, feats, labels = [], [], []
      all_instance_ids = defaultdict(list)
      for inst, subcloud in enumerate(subclouds):
        # Read ply file and parse its rgb values.
        xyz, rgb = cls.read_txt(subcloud)
        _, annotation_subfile = os.path.split(subcloud)
        clsidx = cls.CLASSES.index(annotation_subfile.split('_')[0])
        instance_id = int(os.path.splitext( os.path.basename(subcloud) )[0].split('_')[1])

        coords.append(xyz)
        feats.append(rgb)
        category = np.ones((len(xyz), 1), dtype=np.int32) * clsidx
        instance_ids = np.ones((len(xyz), 1), dtype=np.int32) * instance_id
        label = np.concatenate([category, instance_ids], 1)
        labels.append(label)

        all_instance_ids[clsidx].append(instance_id)
        pass

      for clsidx in all_instance_ids:
        if not len(all_instance_ids[clsidx]) == max(all_instance_ids[clsidx]):
          pass

      if len(coords) == 0:
        print(txtfile, ' has 0 files.')
      else:
        # Concat
        coords = np.concatenate(coords, 0)
        feats = np.concatenate(feats, 0)
        labels = np.concatenate(labels, 0)
        inds, collabels = ME.utils.sparse_quantize(
            coords,
            feats,
            labels[:,0],
            return_index=True,
            ignore_label=255,
            quantization_size=0.01  # 1cm
        )
        pointcloud = np.concatenate((coords[inds], feats[inds], labels[inds]), axis=1)

        # Write ply file.
        if not os.path.exists(target_path):
          os.makedirs(target_path)
        save_point_cloud(pointcloud, out_file, with_label=True, verbose=False)


def generate_splits(stanford_out_path):
  """Takes preprocessed out path and generate txt files"""
  split_path = './splits/stanford'
  if not os.path.exists(split_path):
    os.makedirs(split_path)
  for i in range(1, 7):
    curr_path = os.path.join(stanford_out_path, f'Area_{i}')
    files = glob.glob(os.path.join(curr_path, '*.ply'))
    files = [os.path.relpath(full_path, stanford_out_path) for full_path in files]
    out_txt = os.path.join(split_path, f'area{i}.txt')
    with open(out_txt, 'w') as f:
      f.write('\n'.join(files))


def gen_bboxes():
  from plyfile import PlyData
  ply_files = glob.glob(STANFORD_3D_OUT_PATH + '/*/*.ply')
  for plyf in ply_files:
      bbox_file = plyf.replace('ply', 'npy')
      plydata = PlyData.read(plyf)
      data = plydata.elements[0].data
      coords = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T
      feats = np.array([data['red'], data['green'], data['blue']], dtype=np.float32).T
      categories = np.array(data['label'], dtype=np.int32)
      instances = np.array(data['instance'], dtype=np.int32)
      #show_pcd(coords)

      cat_min = categories.min()
      cat_max = categories.max()

      bboxes = defaultdict(list)
      for cat in range(cat_min, cat_max+1):
        mask_cat = categories == cat
        cat_name = Stanford3DDatasetConverter.CLASSES[cat]
        num_cat = mask_cat.sum()
        print(f'\n{cat_name} has {num_cat} points')
        if num_cat == 0:
          continue
        coords_cat, instances_cat = coords[mask_cat], instances[mask_cat]

        ins_min = instances_cat.min()
        ins_max = instances_cat.max()
        for ins in range(ins_min, ins_max+1):
          mask_ins = instances_cat == ins
          num_ins = mask_ins.sum()
          print(f'\t{ins}th instance has {num_ins} points')
          if num_ins == 0:
            continue
          coords_cat_ins = coords_cat[mask_ins]
          if cat_name != 'clutter':
            #show_pcd(coords_cat_ins)
            bbox = points_to_bbox(coords_cat_ins, out_np=True)[None,:]
            bboxes[cat_name].append(bbox)
        pass
      for cat in bboxes:
        bboxes[cat] = np.concatenate(bboxes[cat], 0)


      # cal pcl scope
      min_pcl = coords.min(axis=0)
      min_pcl = np.floor(min_pcl*10)/10
      max_pcl = coords.max(axis=0)
      max_pcl = np.ceil(max_pcl*10)/10
      pcl_scope = np.concatenate([min_pcl, max_pcl], axis=0)
      bboxes['room'] = pcl_scope[None,:]

      np.save(bbox_file, bboxes)

      colors = instances.astype(np.int32)
      colors = feats
      #_show_pcd([coords], [colors], [bboxes['wall']])
      pass
  pass


def load_1_ply(filepath):
    from plyfile import PlyData
    plydata = PlyData.read(filepath)
    data = plydata.elements[0].data
    coords = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T
    feats = np.array([data['red'], data['green'], data['blue']], dtype=np.float32).T
    labels = np.array(data['label'], dtype=np.int32)
    instance = np.array(data['instance'], dtype=np.int32)
    return coords, feats, labels, None



def get_surface_normal():
  from tools.debug_utils import _make_pcd
  path = '/home/z/Research/mmdetection/data/stanford'
  files = glob.glob(path+'/*/*.ply')
  for fn in files:
    points, _, _, _ = load_1_ply(fn)
    pcd = _make_pcd(points)
    pcd.estimate_normals(fast_normal_computation = True)
    norm = np.asarray(pcd.normals)
    #o3d.visualization.draw_geometries( [pcd] )
    norm_fn = fn.replace('.ply', '-norm.npy')
    np.save( norm_fn, norm )
    print('save:  ', norm_fn)

def get_scene_pcl_scopes():
  path = '/home/z/Research/mmdetection/data/stanford'
  files = glob.glob(path+'/*/*.ply')
  for fn in files:
    points, _, _, _ = load_1_ply(fn)
    min_points = points.min(0)
    max_points = points.max(0)
    scope_i = np.vstack([min_points, max_points])
    scope_fn = fn.replace('.ply', '-scope.txt')
    np.savetxt( scope_fn, scope_i)
    print('save:  ', scope_fn)
  pass



if __name__ == '__main__':
  #Stanford3DDatasetConverter.convert_to_ply(STANFORD_3D_IN_PATH, STANFORD_3D_OUT_PATH)
  #generate_splits(STANFORD_3D_OUT_PATH)
  #gen_bboxes()
  #get_scene_pcl_scopes()
  get_surface_normal()

