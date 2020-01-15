import os
import numpy as np
from beike_data_utils import BEIKE

from .custom import CustomDataset
from .registry import DATASETS


@DATASETS.register_module
class BeikeDataset(CustomDataset):

    CLASSES = ( 'wall', 'window', 'door' )

    def load_annotations(self, ann_folder):
        self.beike = BEIKE(ann_folder)
        self.cat_ids = self.beike.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.beike.getImgIds()
        self.beike.rm_anno_withno_data(self.img_prefix)
        img_infos = self.beike.img_infos
        return img_infos


    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = list(range(len(self)))
        return valid_inds

    def _set_group_flag(self):
      self.flag = np.zeros(len(self), dtype=np.uint8)



