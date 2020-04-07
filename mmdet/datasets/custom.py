import os.path as osp
import os
import mmcv
import numpy as np
from torch.utils.data import Dataset

from .pipelines import Compose
from .registry import DATASETS
np.set_printoptions(precision=3, suppress=True)
from configs.common import VISUAL_TOPVIEW_INPUT

@DATASETS.register_module
class CustomDataset(Dataset):
    """Custom dataset for detection.

    Annotation format:
    [
        {
            'filename': 'a.jpg',
            'width': 1280,
            'height': 720,
            'ann': {
                'bboxes': <np.ndarray> (n, 4),
                'labels': <np.ndarray> (n, ),
                'bboxes_ignore': <np.ndarray> (k, 4), (optional field)
                'labels_ignore': <np.ndarray> (k, 4) (optional field)
            }
        },
        ...
    ]

    The `ann` field is optional for testing.
    """

    CLASSES = None

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root=None,
                 img_prefix='',
                 input_style='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True):
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.input_style = input_style

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.proposal_file is None
                    or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root,
                                              self.proposal_file)
        # load annotations (and proposals)
        self.img_infos = self.load_annotations(self.ann_file)
        #self.unused_rm_anno_withno_data()
        if self.proposal_file is not None:
            self.proposals = self.load_proposals(self.proposal_file)
        else:
            self.proposals = None
        # filter images too small
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.img_infos = [self.img_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]
        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
        # processing pipeline
        self.pipeline = Compose(pipeline)

    def __len__(self):
        return len(self.img_infos)

    def load_annotations(self, ann_file):
        return mmcv.load(ann_file)

    def load_proposals(self, proposal_file):
        return mmcv.load(proposal_file)

    def get_ann_info(self, idx):
        return self.img_infos[idx]['ann']

    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []

    def unused_rm_anno_withno_data(self):
      n0 = len(self.img_infos)
      valid_inds = []
      valid_files = os.listdir(self.img_prefix)
      for i, img_info in enumerate(self.img_infos):
        filename = img_info['filename']
        if img_info['filename'] in valid_files:
          valid_inds.append(i)
      valid_img_infos = [self.img_infos[i] for i in valid_inds]
      self.img_infos = valid_img_infos
      n = len(self.img_infos)
      print(f'\n{n} valid scenes with annotation in total {n0} in {self.img_prefix}\n')

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        valid_inds = []
        for i, img_info in enumerate(self.img_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.img_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        results = self.pipeline(results)
        results['img_meta'].data['input_style'] = self.input_style

        if VISUAL_TOPVIEW_INPUT:
          show_results_train(results)
        return results

    def prepare_test_img(self, idx):
        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        from configs.common import LOAD_GT_TEST
        if LOAD_GT_TEST:
          ann_info = self.get_ann_info(idx)
          results['ann_info'] = ann_info
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        results = self.pipeline(results)
        results['img_meta'][0].data['input_style'] = self.input_style

        #show_results_test(results)
        return  results

def show_results_test(results):
  imgs = results['img']
  for img in imgs:
    img = img.data.cpu().numpy()
    img = np.moveaxis(img, 0, -1)
    mmcv.imshow(img[:,:,0:1])
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    pass

def show_results_train(results):
  from tools.debug_utils import _show_lines_ls_points_ls, _show_img_with_norm
  print('\ncustom, after data augmentation',results['img_meta'].data['filename'])
  img = results['img'].data.cpu().numpy()
  img = np.moveaxis(img, 0, -1)
  img_meta = results['img_meta'].data
  img_norm_cfg = img_meta['img_norm_cfg']
  mean = img_norm_cfg['mean']
  std = img_norm_cfg['std']

  img = img*std + mean

  gt_bboxes = results['gt_bboxes'].data.cpu().numpy()
  gt_labels = results['gt_labels'].data.cpu().numpy()

  flip = img_meta['flip']
  if 'rotate_angle' in img_meta:
    rotate_angle = img_meta['rotate_angle']
  else:
    rotate_angle = 0
  print(f'flip: {flip}\nrotate_angle: {rotate_angle}')

  _show_lines_ls_points_ls(img[:,:,0], [gt_bboxes])
  #_show_img_with_norm(img)
  #_show_lines_ls_points_ls(img[:,:,1:])
  import pdb; pdb.set_trace()  # XXX BREAKPOINT
  pass
