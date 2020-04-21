from utils_dataset.beike_utils.beike_pcl_dataset import BeikePcl
from .registry import DATASETS

@DATASETS.register_module
class BeikePclDataset(BeikePcl):
  def __init__(self, **kwargs):
    BeikePcl.__init__(self, **kwargs)
