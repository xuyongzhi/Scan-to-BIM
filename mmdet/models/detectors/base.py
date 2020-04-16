from abc import ABCMeta, abstractmethod

import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch.nn as nn
import os
import torch

from mmdet.core import auto_fp16, get_classes, tensor2imgs
from configs.common import DIM_PARSE, DEBUG_CFG
from beike_data_utils.beike_utils import load_gt_lines_bk
from beike_data_utils.line_utils import  optimize_graph
from utils_dataset.gen_sparse_input import prepare_sparse_input, prepare_bev_sparse
from tools.debug_utils import _show_lines_ls_points_ls

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
        if isinstance(imgs[0], torch.Tensor):
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
        if 'input_style' in img_meta[0] and img_meta[0]['input_style'] == 'bev_sparse':
          img = prepare_bev_sparse(img, img_meta, **kwargs)
        if 'input_style' in img_meta[0] and img_meta[0]['input_style'] == 'pcl':
          img = prepare_sparse_input(img, img_meta, **kwargs)
          if not return_loss:
            img = [img]
            img_meta = [img_meta]

        if return_loss:
            return self.forward_train(img, img_meta, **kwargs)
        else:
            return self.forward_test(img, img_meta, **kwargs)


    def show_corner_hm(self, data, result, dataset=None, score_thr=0.3):
        from mmdet.debug_tools import show_heatmap, show_img_lines
        if isinstance(result, tuple):
            bbox_result, segm_result = result
        else:
            bbox_result, segm_result = result, None
        assert bbox_result[0].shape[1] == 4 # [4: cls, cen, ofs]
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
            if img.shape[-1] == 4:
              img_show = np.repeat(img[:h, :w, 0:1], 3, axis=2)
            else:
              img_show = img[:h, :w, :3]

            filename = img_meta['filename']
            scene_name = os.path.basename(filename).replace('.npy', '')
            out_dir = './cor_hm_res/'
            out_file = out_dir + scene_name
            if not os.path.exists(out_dir):
              os.makedirs(out_dir)

            gt_lines = load_gt_lines_bk(img_meta, img_show)
            #gt_lines = None

            bboxes = np.vstack(bbox_result)
            featmap_size = np.sqrt(bboxes.shape[0]).astype(np.int32)
            bboxes = bboxes.reshape( featmap_size, featmap_size, 4 )
            ffs = (featmap_size, featmap_size)
            cor_scores = (bboxes[:,:,0] + bboxes[:,:,1])/2

            #show_heatmap(bboxes[:,:,0], (h,w), gt_lines=gt_lines, score_thr=0.5)

            show_img_lines(img_show, gt_lines, name=out_file+'_gt.png', only_draw=1)
            show_heatmap(bboxes[:,:,0], (h,w), out_file+'_cls.png', gt_lines=gt_lines)
            show_heatmap(bboxes[:,:,0], (h,w), out_file+'_cls_.png')
            show_heatmap(bboxes[:,:,1], (h,w), out_file+'_centerness.png', gt_lines=gt_lines)
            show_heatmap(cor_scores, (h,w), out_file+'_cls_cen.png', gt_lines=gt_lines)
            pass


    def show_result_graph(self, data, result, dataset=None, score_thr=0.5):
        num_classes = len(data['img_meta'][0]['classes'])+1
        self.dim_parse = DIM_PARSE(num_classes)

        bbox_result = result['det_bboxes']
        if bbox_result[0].shape[0] > 0:
            assert bbox_result[0].shape[1] == self.dim_parse.OUT_DIM_FINAL
        else:
          print('no box detected')
          return

        is_pcl_input = isinstance(data['img_meta'][0], dict)
        if is_pcl_input:
          boader_aug = 20
          draw_scale = 2

          img_metas = data['img_meta']
          pcls = [data['img'][0][:,1:]]
          w, h = img_metas[0]['dynamic_vox_size_aug'][:2] * draw_scale + boader_aug * 2
          img_shape = (h,w,3)
          imgs = [np.zeros(img_shape, dtype=np.int8)]
        else:
          img_metas = data['img_meta'][0].data[0]
          img_tensor = data['img'][0]
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
            img_show = img[:,:, :3]

            bboxes = np.vstack(bbox_result)
            # draw bounding boxes
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)
            ]
            labels = np.concatenate(labels)

            img_show = np.clip(img_show, a_min=None, a_max=255)
            class_names = tuple([c[0] for c in class_names])

            scores = bboxes[:,-1]
            mask = scores >= score_thr
            bboxes = bboxes[mask]
            labels = labels[mask]

            if bboxes.shape[0] == 0:
              continue


            if self.dim_parse.OUT_EXTAR_DIM > 0:
                bboxes_refine, bboxes_init, points_refine, points_init,\
                  score_refine, score_final, score_ave, corner0_score,\
                  corner1_score, corner0_center, corner1_center, score_composite = \
                      self.dim_parse.parse_bboxes_out(bboxes, 'final')

                if is_pcl_input:
                  bboxes_refine[:,:4] *= draw_scale
                  bboxes_init[:,:4]  *= draw_scale
                  points_init  *= draw_scale
                  points_refine  *= draw_scale

                  bboxes_refine[:,:4] += boader_aug
                  bboxes_init[:,:4] += boader_aug
                  points_init += boader_aug
                  points_refine += boader_aug

                #_show_lines_ls_points_ls((512,512), [bboxes_refine])

                lines_graph, score_graph, labels_graph = optimize_graph(bboxes_refine, score_composite, labels, self.dim_parse.OBJ_REP, opt_graph_cor_dis_thr=10)
                lines_graph = np.concatenate([lines_graph, score_graph], axis=1)

                #scores_filter = np.squeeze(score_ave)
                scores_filter = np.squeeze(score_composite)

                bboxes = lines_composite = np.concatenate([bboxes_refine, score_composite], axis=1)
                lines_ave = np.concatenate([bboxes_refine, score_ave], axis=1)

                lines_init = np.concatenate([bboxes_init, score_refine], axis=1)
                lines_refine = np.concatenate([bboxes_refine, score_final], axis=1)
                is_with_corner =  corner0_score is not None
                if is_with_corner:
                  lines_corner0_score = np.concatenate([bboxes_refine, corner0_score], axis=1)
                  lines_corner1_score = np.concatenate([bboxes_refine, corner1_score], axis=1)
                  lines_corner0_center = np.concatenate([bboxes_refine, corner0_center], axis=1)
                  lines_corner1_center = np.concatenate([bboxes_refine, corner1_center], axis=1)
                num_box = bboxes.shape[0]
                key_points_refine = points_refine.reshape(num_box, -1, 2)
                key_points_init = points_init.reshape(num_box, -1, 2)
            else:
                key_points_init = None
                key_points_refine = None

            filename = img_meta['filename'].split('.')[0]
            scene_name = os.path.basename(filename).replace('.npy', '')
            nstr = str(self.num_imgs) if self.num_imgs is not None else ''
            out_dir_out = f'./line_det_{DEBUG_CFG.OBJ_LEGEND}_res_{nstr}/final/'
            out_dir_middle = f'./line_det_{DEBUG_CFG.OBJ_LEGEND}_res_{nstr}/middle/'
            if not os.path.exists(out_dir_out):
              os.makedirs(out_dir_out)
            if not os.path.exists(out_dir_middle):
              os.makedirs(out_dir_middle)

            from tools.debug_utils import _show_det_lines, show_det_lines_1by1
            #show_fun = show_det_lines_1by1
            show_fun = _show_det_lines
            show_points = 0

            lines_list = [lines_graph, lines_composite, lines_ave, lines_init, lines_refine, ]
            names_list = ['graph.png', 'composite.png', 'ave.png', 'init.png', 'refine.png', ]
            if is_with_corner:
              lines_list  += [lines_corner0_score, lines_corner1_score, lines_corner0_center, lines_corner1_center]
              names_list  += ['corner0_cls.png', 'corner1_cls.png', 'corner0_cen.png', 'corner1_cen.png']
            for i in range(len(lines_list)):
                bboxes_ = lines_list[i]
                name_  = names_list[i]
                if name_ in ['graph.png', 'composite.png']:
                  out_dir = out_dir_out
                else:
                  out_dir = out_dir_middle
                out_file_i = out_dir + scene_name + '_' + name_
                key_points = None
                show_fun(img_show, bboxes_, labels, class_names=class_names, score_thr=score_thr, line_color='random',thickness=1, show=0,
                          out_file=out_file_i, key_points=key_points, point_color='random', scores=scores_filter)
                if show_points:
                  if  name_ == 'init.png':
                    key_points = key_points_init
                  else:
                    key_points = key_points_refine
                  out_file_i = out_dir + scene_name + '_' + name_ + '_P.png'
                  show_fun(img_show, bboxes_, labels, class_names=class_names, score_thr=score_thr, line_color='green',thickness=2, show=0,
                          out_file=out_file_i, key_points=key_point, scores=scores)
                if self.dim_parse.OUT_EXTAR_DIM == 0 or DEBUG_CFG.OBJ_LEGEND == 'rotation':
                  break

    def show_result(self, data, result, dataset=None, score_thr=0.3, num_imgs=None):
        num_classes = len(data['img_meta'][0]['classes'])+1
        self.dim_parse = DIM_PARSE(num_classes)
        self.num_imgs = num_imgs

        if DEBUG_CFG.OUT_CORNER_HM_ONLY:
          self.show_corner_hm(data, result, dataset, score_thr)
          return

        if self.dim_parse.OBJ_REP == 'lscope_istopleft':
          self.show_result_graph(data, result, dataset, score_thr)
          return

        if isinstance(result, tuple):
            bbox_result, segm_result = result
        else:
            bbox_result, segm_result = result, None
        if bbox_result[0].shape[0] > 0:
            assert bbox_result[0].shape[1] == self.dim_parse.OUT_DIM_FINAL
        else:
          print('no box detected')
          return

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
            if img.shape[-1] == 4:
              img_show = np.repeat(img[:h, :w, 0:1], 3, axis=2)
            else:
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

            bboxes_s = bboxes.copy()
            assert bboxes.shape[1] == 5

            if key_points_init is not None:
                draw_key_points(img_show, key_points_init, bboxes_init, score_thr)
                mmcv.imshow_det_bboxes(
                    img_show.copy(),
                    bboxes_init,
                    labels,
                    class_names=class_names,
                    score_thr=score_thr,
                    bbox_color='green',
                    thickness=1,
                    out_file=out_file.replace('.jpg','_init.jpg'))


                draw_key_points(img_show, key_points_refine, bboxes_refine, score_thr)
                mmcv.imshow_det_bboxes(
                    img_show.copy(),
                    bboxes_refine,
                    labels,
                    class_names=class_names,
                    score_thr=score_thr,
                    bbox_color='green',
                    thickness=1,
                    out_file=out_file.replace('.jpg','_refine.jpg'))
            else:
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

