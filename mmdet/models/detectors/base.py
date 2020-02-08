from abc import ABCMeta, abstractmethod

import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch.nn as nn
import os

from mmdet.core import auto_fp16, get_classes, tensor2imgs
from configs.common import OBJ_DIM, OUT_PTS_DIM

class BaseDetector(nn.Module, metaclass=ABCMeta):
    """Base class for detectors"""

    def __init__(self):
        super(BaseDetector, self).__init__()
        self.fp16_enabled = False

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_shared_head(self):
        return hasattr(self, 'shared_head') and self.shared_head is not None

    @property
    def with_bbox(self):
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_mask(self):
        return hasattr(self, 'mask_head') and self.mask_head is not None

    @abstractmethod
    def extract_feat(self, imgs):
        pass

    def extract_feats(self, imgs):
        assert isinstance(imgs, list)
        for img in imgs:
            yield self.extract_feat(img)

    @abstractmethod
    def forward_train(self, imgs, img_metas, **kwargs):
        """
        Args:
            img (list[Tensor]): list of tensors of shape (1, C, H, W).
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has:
                'img_shape', 'scale_factor', 'flip', and my also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

             **kwargs: specific to concrete implementation
        """
        pass

    async def async_simple_test(self, img, img_meta, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def simple_test(self, img, img_meta, **kwargs):
        pass

    @abstractmethod
    def aug_test(self, imgs, img_metas, **kwargs):
        pass

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            from mmdet.apis import get_root_logger
            logger = get_root_logger()
            logger.info('load model from: {}'.format(pretrained))

    async def aforward_test(self, *, img, img_meta, **kwargs):
        for var, name in [(img, 'img'), (img_meta, 'img_meta')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(img)
        if num_augs != len(img_meta):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(img), len(img_meta)))
        # TODO: remove the restriction of imgs_per_gpu == 1 when prepared
        imgs_per_gpu = img[0].size(0)
        assert imgs_per_gpu == 1

        if num_augs == 1:
            return await self.async_simple_test(img[0], img_meta[0], **kwargs)
        else:
            raise NotImplementedError

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_meta (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(imgs), len(img_metas)))
        # TODO: remove the restriction of imgs_per_gpu == 1 when prepared
        imgs_per_gpu = imgs[0].size(0)
        assert imgs_per_gpu == 1

        if num_augs == 1:
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            return self.aug_test(imgs, img_metas, **kwargs)

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_meta, return_loss=True, **kwargs):
        """
        Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=True`, img and img_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=False`, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(img, img_meta, **kwargs)
        else:
            return self.forward_test(img, img_meta, **kwargs)

    def show_result(self, data, result, dataset=None, score_thr=0.3):
        if isinstance(result, tuple):
            bbox_result, segm_result = result
        else:
            bbox_result, segm_result = result, None
        assert bbox_result[0].shape[1] == OBJ_DIM + OUT_PTS_DIM + 1

        img_tensor = data['img'][0]
        img_metas = data['img_meta'][0].data[0]
        imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
        assert len(imgs) == len(img_metas)

        if dataset is None:
            class_names = self.CLASSES
        elif isinstance(dataset, str):
            class_names = get_classes(dataset)
        elif isinstance(dataset, (list, tuple)):
            class_names = dataset
        else:
            raise TypeError(
                'dataset must be a valid dataset name or a sequence'
                ' of class names, not {}'.format(type(dataset)))

        for img, img_meta in zip(imgs, img_metas):
            h, w, _ = img_meta['img_shape']
            img_show = img[:h, :w, :3]

            bboxes = np.vstack(bbox_result)
            # draw segmentation masks
            if segm_result is not None:
                segms = mmcv.concat_list(segm_result)
                inds = np.where(bboxes[:, -1] > score_thr)[0]
                for i in inds:
                    color_mask = np.random.randint(
                        0, 256, (1, 3), dtype=np.uint8)
                    mask = maskUtils.decode(segms[i]).astype(np.bool)
                    img_show[mask] = img_show[mask] * 0.5 + color_mask * 0.5
            # draw bounding boxes
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)
            ]
            labels = np.concatenate(labels)

            img_show = np.clip(img_show+1, a_min=None, a_max=255)
            class_names = tuple([c[0] for c in class_names])


            if OUT_PTS_DIM > 0:
              key_points = bboxes[:,-1-OUT_PTS_DIM:-1].reshape(bboxes.shape[0], -1,2)
              bboxes = np.concatenate([ bboxes[:,:OBJ_DIM], bboxes[:, -1:]], axis=1 )
            else:
              key_points = None

            filename = img_meta['filename']
            scene_name = os.path.basename(filename).replace('density.', '')
            if bboxes.shape[1] == 6:
              out_dir = './line_det_res/'
            else:
              out_dir = './box_det_res/'
            out_file = out_dir + scene_name
            if not os.path.exists(out_dir):
              os.makedirs(out_dir)

            if bboxes.shape[1] == 6:
                from mmdet.debug_tools import show_det_lines, show_det_lines_1by1
                #show_det_lines_1by1(
                show_det_lines(
                  img_show,
                  bboxes,
                  labels,
                  class_names=class_names,
                  score_thr=score_thr,
                  line_color='green',
                  thickness=2,
                  show=0,
                  out_file=out_file,
                  key_points=key_points)
                continue

            bboxes_s = bboxes.copy()

            if bboxes_s.shape[1] == 6:
              bboxes_s = bboxes_s[:,[0,1,2,3,-1]]
            assert bboxes_s.shape[1] == 5
            if key_points is not None:
              draw_key_points(img_show, key_points, bboxes_s, score_thr)
            mmcv.imshow_det_bboxes(
                img_show,
                bboxes_s,
                labels,
                class_names=class_names,
                score_thr=score_thr,
                bbox_color='green',
                thickness=1,
                out_file=out_file)

            pass
def draw_key_points(img, key_points, bboxes_s, score_thr,
                    point_color=(0,0,255), thickness=2):
    import cv2
    for i in range(bboxes_s.shape[0]):
      if bboxes_s[i,-1] < score_thr:
        continue
      for j in range(key_points.shape[1]):
        p = key_points[i][j].astype(np.int32)
        cv2.circle(img, (p[0], p[1]), 2, point_color, thickness=thickness)


