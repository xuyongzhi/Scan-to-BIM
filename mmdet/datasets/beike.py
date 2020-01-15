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
        img_infos = self.beike.img_infos
        self.img_num = self.beike.img_num
        return img_infos

    def _get_ann_info(self, idx):
        ann = {
          'lines': self.img_infos[idx]['lines'],
          'line_labels': self.img_infos[idx]['line_cat_ids'], }

        img_info = self.img_infos[idx]
        sn = img_info['filename']
        line_labels = img_info['line_cat_ids']
        #print(f'scene name: {sn}')
        #print(f'line_labels: {line_labels}')
        return ann

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = list(range(self.beike.img_num))
        return valid_inds
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.beike.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.filter_empty_gt and self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
      self.flag = np.zeros(self.img_num, dtype=np.uint8)

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann['segmentation'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def UN_prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        ann_info = img_info['ann']
        del img_info['ann']
        img = self.beike.load_data(idx, self.img_prefix)
        results =   {'img_prefix': self.img_prefix,
                    'seg_prefix': self.seg_prefix,
                    'proposal_file': self.proposal_file,
                     'bbox_fields': [],
                    'mask_fields': [],
                     'seg_fields': [],

                    'img_meta': self.img_infos, 'img':img, 'gt_bboxes': img_info['lines'],
                    'gt_labels':img_info['line_cat_ids'], 'corners':img_info['corners'],
                    'corner_labels': img_info['corner_cat_ids'] }
        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        results = self.pipeline(results)
        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        return results


