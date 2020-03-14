import os
import numpy as np
from utils_data3d.datasets.stanford_pcl_utils import STANFORD_PCL

from .custom import CustomDataset
from .registry import DATASETS


@DATASETS.register_module
class StanfordPclDataset(CustomDataset):

    #CLASSES = ( 'wall', 'window', 'door' )
    CLASSES = ( 'wall', )

    def load_annotations(self, ann_folder):
        self.sfdpcl = STANFORD_PCL(ann_folder, img_prefix=self.img_prefix)
        self.cat_ids = self.sfdpcl.getCatIds()
        self.cat2label = {
            cat_id: i + 0
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.sfdpcl.getImgIds()
        img_infos = self.sfdpcl.img_infos
        return img_infos


    def _filter_imgs(self):
        """Filter images too small or without ground truths."""
        valid_inds = list(range(len(self)))
        return valid_inds

    def _set_group_flag(self):
      self.flag = np.zeros(len(self), dtype=np.uint8)


