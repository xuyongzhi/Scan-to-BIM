import logging
import sys
from scipy import spatial

from .registry import DATASETS

from utils_dataset.lib.utils import read_txt, fast_hist, per_class_iu
from utils_dataset.lib.dataset import VoxelizationDataset, DatasetPhase, str2datasetphase_type, cache
import utils_dataset.lib.transforms as t


@DATASETS.register_module
class VoxelDatasetBase(VoxelizationDataset):

  def __init__(self, phase, data_paths,  config, input_transform=None, target_transform=None
               ):
    if isinstance(phase, str):
      phase = str2datasetphase_type(phase)

    if config.return_transformation:
      collate_fn = t.cflt_collate_fn_factory(config.limit_numpoints)
    else:
      collate_fn = t.cfl_collate_fn_factory(config.limit_numpoints)

    prevoxel_transform_train = []
    if config.augment_data:
      prevoxel_transform_train.append(t.ElasticDistortion(self.ELASTIC_DISTORT_PARAMS))

    if len(prevoxel_transform_train) > 0:
      prevoxel_transforms = t.Compose(prevoxel_transform_train)
    else:
      prevoxel_transforms = None

    input_transforms = []
    if input_transform is not None:
      input_transforms += input_transform

    if config.augment_data:
      input_transforms += [
          t.RandomDropout(0.2),
          t.RandomHorizontalFlip(self.ROTATION_AXIS, self.IS_TEMPORAL),
          t.ChromaticAutoContrast(),
          t.ChromaticTranslation(config.data_aug_color_trans_ratio),
          t.ChromaticJitter(config.data_aug_color_jitter_std),
          # t.HueSaturationTranslation(config.data_aug_hue_max, config.data_aug_saturation_max),
      ]

    if len(input_transforms) > 0:
      input_transforms = t.Compose(input_transforms)
    else:
      input_transforms = None

    VoxelizationDataset.__init__(
        self,
        data_paths,
        data_root=self.data_root,
        prevoxel_transform=prevoxel_transforms,
        input_transform=input_transform,
        target_transform=target_transform,
        ignore_label=config.ignore_label,
        return_transformation=config.return_transformation,
        augment_data=config.augment_data,
        elastic_distortion=config.elastic_distortion,
        config=config)
    pass


