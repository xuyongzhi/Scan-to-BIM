from utils_dataset.stanford3d_utils.stanford_pcl_dataset import StanfordPcl
from .registry import DATASETS

@DATASETS.register_module
class StanfordPclDataset(StanfordPcl):
  def __init__(self, **kwargs):
    StanfordPcl.__init__(self, **kwargs)
