import os
import numpy as np
from utils_dataset.stanford3d_utils.stanford_pcl_dataset import Stanford_BEV

from .custom import CustomDataset
from .registry import DATASETS


@DATASETS.register_module
class Stanford_2D_Dataset(CustomDataset):
    def load_annotations(self, ann_folder):
        self.sfd = Stanford_BEV(ann_folder,
                           img_prefix=self.img_prefix,
                           test_mode=self.test_mode,
                           filter_edges=self.filter_edges,
                           classes = self.classes,
                           )
        self.cat_ids = self.sfd.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self._catid_2_cat = self.sfd._catid_2_cat
        self.img_ids = self.sfd.getImgIds()
        img_infos = self.sfd.img_infos
        return img_infos


    def _filter_imgs(self):
        """Filter images too small or without ground truths."""
        valid_inds = list(range(len(self)))
        return valid_inds

    def _set_group_flag(self):
      self.flag = np.zeros(len(self), dtype=np.uint8)


