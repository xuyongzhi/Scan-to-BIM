import collections

import numpy as np
import MinkowskiEngine as ME
from scipy.linalg import expm, norm


# Rotation matrix along axis with angle theta
def M(axis, theta):
  return expm(np.cross(np.eye(3), axis / norm(axis) * theta))


class Voxelizer:

  def __init__(self,
               voxel_size=1,
               clip_bound=None,
               use_augmentation=False,
               scale_augmentation_bound=None,
               rotation_augmentation_bound=None,
               translation_augmentation_ratio_bound=None,
               ignore_label=255,
               max_voxel_footprint=None ):
    """
    Args:
      voxel_size: side length of a voxel
      clip_bound: boundary of the voxelizer. Points outside the bound will be deleted
        expects either None or an array like ((-100, 100), (-100, 100), (-100, 100)).
      scale_augmentation_bound: None or (0.9, 1.1)
      rotation_augmentation_bound: None or ((np.pi / 6, np.pi / 6), None, None) for 3 axis.
        Use random order of x, y, z to prevent bias.
      translation_augmentation_bound: ((-5, 5), (0, 0), (-10, 10))
      ignore_label: label assigned for ignore (not a training label).
    """
    self.voxel_size = voxel_size
    self.clip_bound = clip_bound
    self.ignore_label = ignore_label
    self.max_voxel_footprint = max_voxel_footprint

    # Augmentation
    self.use_augmentation = use_augmentation
    self.scale_augmentation_bound = scale_augmentation_bound
    self.rotation_augmentation_bound = rotation_augmentation_bound
    self.translation_augmentation_ratio_bound = translation_augmentation_ratio_bound

  def get_transformation_matrix(self):
    voxelization_matrix, rotation_matrix = np.eye(4), np.eye(4)
    # Get clip boundary from config or pointcloud.
    # Get inner clip bound to crop from.

    # Transform pointcloud coordinate to voxel coordinate.
    # 1. Random rotation
    rot_mat = np.eye(3)
    rot_angles = []
    if self.use_augmentation and self.rotation_augmentation_bound is not None:
      if isinstance(self.rotation_augmentation_bound, collections.Iterable):
        rot_mats = []
        for axis_ind, rot_bound in enumerate(self.rotation_augmentation_bound):
          theta = 0
          axis = np.zeros(3)
          axis[axis_ind] = 1
          if rot_bound is not None:
            theta = np.random.uniform(*rot_bound)
            #if axis_ind == 2:
            #  theta = np.pi * 0.25
          rot_mats.append(M(axis, theta))
          rot_angles.append(theta)
        # Use random order
        np.random.shuffle(rot_mats)
        rot_mat = rot_mats[0] @ rot_mats[1] @ rot_mats[2]
      else:
        raise ValueError()
    rotation_matrix[:3, :3] = rot_mat
    # 2. Scale and translate to the voxel space.
    scale = 1 / self.voxel_size
    random_scale_rate = 1
    if self.use_augmentation and self.scale_augmentation_bound is not None:
      random_scale_rate = np.random.uniform(*self.scale_augmentation_bound)
      scale *= random_scale_rate
    np.fill_diagonal(voxelization_matrix[:3, :3], scale)
    # Get final transformation matrix.
    return voxelization_matrix, rotation_matrix, rot_angles, random_scale_rate

  def clip(self, coords, center=None, trans_aug_ratio=None):
    bound_min = np.min(coords, 0).astype(float)
    bound_max = np.max(coords, 0).astype(float)
    bound_size = bound_max - bound_min
    if center is None:
      center = bound_min + bound_size * 0.5
    if trans_aug_ratio is not None:
      trans = np.multiply(trans_aug_ratio, bound_size)
      center += trans
    lim = self.clip_bound

    if isinstance(self.clip_bound, (int, float)):
      if bound_size.max() < self.clip_bound:
        return None
      else:
        clip_inds = ((coords[:, 0] >= (-lim + center[0])) & \
            (coords[:, 0] < (lim + center[0])) & \
            (coords[:, 1] >= (-lim + center[1])) & \
            (coords[:, 1] < (lim + center[1])) & \
            (coords[:, 2] >= (-lim + center[2])) & \
            (coords[:, 2] < (lim + center[2])))
        return clip_inds

    # Clip points outside the limit
    clip_inds = ((coords[:, 0] >= (lim[0][0] + center[0])) & \
        (coords[:, 0] < (lim[0][1] + center[0])) & \
        (coords[:, 1] >= (lim[1][0] + center[1])) & \
        (coords[:, 1] < (lim[1][1] + center[1])) & \
        (coords[:, 2] >= (lim[2][0] + center[2])) & \
        (coords[:, 2] < (lim[2][1] + center[2])))
    return clip_inds

  def voxelize(self, coords, feats, labels, center=None):
    assert coords.shape[1] == 3 and coords.shape[0] == feats.shape[0] and coords.shape[0]
    if self.clip_bound is not None:
      trans_aug_ratio = np.zeros(3)
      if self.use_augmentation and self.translation_augmentation_ratio_bound is not None:
        for axis_ind, trans_ratio_bound in enumerate(self.translation_augmentation_ratio_bound):
          trans_aug_ratio[axis_ind] = np.random.uniform(*trans_ratio_bound)

      clip_inds = self.clip(coords, center, trans_aug_ratio)
      if clip_inds is not None:
        coords, feats = coords[clip_inds], feats[clip_inds]
        if labels is not None:
          labels = labels[clip_inds]

    # Get rotation and scale
    M_v, M_r, angles_rot, random_scale_rate = self.get_transformation_matrix()
    # Apply transformations
    rigid_transformation = M_v
    if self.use_augmentation:
      rigid_transformation = M_r @ rigid_transformation

    homo_coords = np.hstack((coords, np.ones((coords.shape[0], 1), dtype=coords.dtype)))
    if self.max_voxel_footprint is not None:
      rigid_transformation, scale_rate = self.auto_scale_inside_voxel_resolution(homo_coords, rigid_transformation)
    else:
      scale_rate = 1
    scale_rate *= random_scale_rate
    coords_aug = np.floor(homo_coords @ rigid_transformation.T[:, :3])

    # Align all coordinates to the origin.
    min_coords = coords_aug.min(0)
    M_t = np.eye(4)
    M_t[:3, -1] = -min_coords
    rigid_transformation = M_t @ rigid_transformation
    coords_aug = np.floor(coords_aug - min_coords)

    # key = self.hash(coords_aug)  # floor happens by astype(np.uint64)
    if labels is not None:
      coords_aug, feats, labels = ME.utils.sparse_quantize(
          coords_aug, feats, labels=labels, ignore_label=self.ignore_label)
    else:
      coords_aug, feats = ME.utils.sparse_quantize(
          coords_aug, feats)


    # rotate normal
    feats[:,3:6] = feats[:, 3:6] @ M_r.T[:3,:3]
    return coords_aug, feats, labels, rigid_transformation, angles_rot, scale_rate

  def auto_scale_inside_voxel_resolution(self, homo_coords, rigid_transformation):
    '''
    When the voxel resolution needs to be contained in a scope.
    The pcl scope could be out of the resolution because of rotation.
    assign a scale after the rotation.
    '''
    coords_aug = homo_coords @ rigid_transformation.T[:, :3]
    min_coords = coords_aug.min(0)
    max_coords = coords_aug.max(0)
    scope_coords = max_coords - min_coords
    voxel_footprint = np.product( scope_coords[:2] )
    scale_rate = np.sqrt( self.max_voxel_footprint / voxel_footprint)
    if scale_rate < 1:
      rigid_transformation[:3,:3] *= scale_rate
    else:
      scale_rate = 1
    return rigid_transformation, scale_rate

    #scale_rates = (np.array(self.voxel_resolution[:2])-1) / scope_coords[:2] # xy
    #scale_rates = (np.array(self.voxel_resolution)-1) / scope_coords # xyz
    #scale_rate = scale_rates.min()
    #print(f'scale_rates:{scale_rates}, min={scale_rate}')
    rigid_transformation[:3,:3] *= scale_rate
    return rigid_transformation, scale_rate

  def voxelize_temporal(self,
                        coords_t,
                        feats_t,
                        labels_t,
                        centers=None,
                        return_transformation=False):
    # Legacy code, remove
    if centers is None:
      centers = [
          None,
      ] * len(coords_t)
    coords_tc, feats_tc, labels_tc, transformation_tc = [], [], [], []

    # ######################### Data Augmentation #############################
    # Get rotation and scale
    M_v, M_r, angles_rot = self.get_transformation_matrix()
    # Apply transformations
    rigid_transformation = M_v
    if self.use_augmentation:
      rigid_transformation = M_r @ rigid_transformation
    # ######################### Voxelization #############################
    # Voxelize coords
    for coords, feats, labels, center in zip(coords_t, feats_t, labels_t, centers):

      ###################################
      # Clip the data if bound exists
      if self.clip_bound is not None:
        trans_aug_ratio = np.zeros(3)
        if self.use_augmentation and self.translation_augmentation_ratio_bound is not None:
          for axis_ind, trans_ratio_bound in enumerate(self.translation_augmentation_ratio_bound):
            trans_aug_ratio[axis_ind] = np.random.uniform(*trans_ratio_bound)

        clip_inds = self.clip(coords, center, trans_aug_ratio)
        if clip_inds is not None:
          coords, feats = coords[clip_inds], feats[clip_inds]
          if labels is not None:
            labels = labels[clip_inds]
      ###################################

      homo_coords = np.hstack((coords, np.ones((coords.shape[0], 1), dtype=coords.dtype)))
      coords_aug = np.floor(homo_coords @ rigid_transformation.T)[:, :3]

      coords_aug, feats, labels = ME.utils.sparse_quantize(
          coords_aug, feats, labels=labels, ignore_label=self.ignore_label)

      coords_tc.append(coords_aug)
      feats_tc.append(feats)
      labels_tc.append(labels)
      transformation_tc.append(rigid_transformation.flatten())

    return_args = [coords_tc, feats_tc, labels_tc]
    if return_transformation:
      return_args.append(transformation_tc)

    return tuple(return_args)


def test():
  N = 16575
  coords = np.random.rand(N, 3) * 10
  feats = np.random.rand(N, 4)
  labels = np.floor(np.random.rand(N) * 3)
  coords[:3] = 0
  labels[:3] = 2
  voxelizer = Voxelizer()
  print(voxelizer.voxelize(coords, feats, labels))


if __name__ == '__main__':
  test()
