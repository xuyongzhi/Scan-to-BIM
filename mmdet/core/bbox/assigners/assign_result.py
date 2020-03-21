import torch


class AssignResult(object):
    """
    Stores assignments between predicted and truth boxes.

    Attributes:
        num_gts (int): the number of truth boxes considered when computing this
            assignment

        gt_inds (LongTensor): for each predicted box indicates the 1-based
            index of the assigned truth box. 0 means unassigned and -1 means
            ignore.

        max_overlaps (FloatTensor): the iou between the predicted box and its
            assigned truth box.

        labels (None | LongTensor): If specified, for each predicted box
            indicates the category label of the assigned truth box.

    Example:
        >>> # An assign result between 4 predicted boxes and 9 true boxes
        >>> # where only two boxes were assigned.
        >>> num_gts = 9
        >>> max_overlaps = torch.LongTensor([0, .5, .9, 0])
        >>> gt_inds = torch.LongTensor([-1, 1, 2, 0])
        >>> labels = torch.LongTensor([0, 3, 4, 0])
        >>> self = AssignResult(num_gts, gt_inds, max_overlaps, labels)
        >>> print(str(self))  # xdoctest: +IGNORE_WANT
        <AssignResult(num_gts=9, gt_inds.shape=(4,), max_overlaps.shape=(4,),
                      labels.shape=(4,))>
        >>> # Force addition of gt labels (when adding gt as proposals)
        >>> new_labels = torch.LongTensor([3, 4, 5])
        >>> self.add_gt_(new_labels)
        >>> print(str(self))  # xdoctest: +IGNORE_WANT
        <AssignResult(num_gts=9, gt_inds.shape=(7,), max_overlaps.shape=(7,),
                      labels.shape=(7,))>
    """

    def __init__(self, num_gts, gt_inds, max_overlaps, labels=None, env='',
                 img_meta=None):
        self.num_gts = num_gts
        self.gt_inds = gt_inds
        self.max_overlaps = max_overlaps
        self.labels = labels
        self.pos_dist = None

        # for debug:
        if self.gt_inds is not None:
            self.pos_inds = torch.nonzero(self.gt_inds > 0).squeeze()
            self.gt_inds_valid = self.gt_inds[self.pos_inds] - 1
            self.neg_inds = torch.nonzero(self.gt_inds == 0).squeeze()
            #self.ign_inds = torch.nonzero(self.gt_inds < 0).squeeze()
            self.num_pos_inds = self.pos_inds.numel()
            #self.num_neg_inds = self.neg_inds.numel()
            #self.num_ign_inds = self.ign_inds.numel()

            self.lost_gt = self.num_pos_inds < self.num_gts
            if self.lost_gt and False:
              print('\tgt num = {}, pos inds num = {}, lost gt!'\
                .format(self.num_gts, self.num_pos_inds) +\
                '\tcore/bbox/assigners/assign_result.py from \t{}'.format(env))
              if img_meta is not None:
                print(img_meta['filename'])
              #print('\n\n')

    def add_gt_(self, gt_labels):
        self_inds = torch.arange(
            1, len(gt_labels) + 1, dtype=torch.long, device=gt_labels.device)
        self.gt_inds = torch.cat([self_inds, self.gt_inds])

        # Was this a bug?
        # self.max_overlaps = torch.cat(
        #     [self.max_overlaps.new_ones(self.num_gts), self.max_overlaps])
        # IIUC, It seems like the correct code should be:
        self.max_overlaps = torch.cat(
            [self.max_overlaps.new_ones(len(gt_labels)), self.max_overlaps])

        if self.labels is not None:
            self.labels = torch.cat([gt_labels, self.labels])

    def __nice__(self):
        """
        Create a "nice" summary string describing this assign result
        """
        parts = []
        parts.append('num_gts={!r}'.format(self.num_gts))
        if self.gt_inds is None:
            pass
        else:
            try:
             parts.append('pos num={!r}'.format(
                self.pos_inds.shape[0]))
            except:
              parts.append('err pos_inds')
              print('\n\n')
              print(self.pos_inds)
              print('\n\n')
              pass
        if self.max_overlaps is None:
            pass
        else:
            mean_max_ol = self.max_overlaps[self.pos_inds].mean().\
                            cpu().data.numpy()
            parts.append('mean max_overlap={:.3}'.format(\
                mean_max_ol))
        if self.labels is None:
            parts.append('labels={!r}'.format(self.labels))
        else:
            parts.append('labels.shape={!r}'.format(tuple(self.labels.shape)))
        if self.pos_dist is not  None:
            parts.append('pos_dist min_mean_max = {} '.format(self.pos_dist_str))
        return ', '.join(parts)

    def __repr__(self):
        nice = self.__nice__()
        classname = self.__class__.__name__
        return '<{}({}) at {}>'.format(classname, nice, hex(id(self)))

    def __str__(self):
        classname = self.__class__.__name__
        nice = self.__nice__()
        return '<{}({})>'.format(classname, nice)

    # for debug:
    def dg_add_pos_dist(self, pos_dist):
      self.pos_dist = pos_dist
      self.summary_pos_dist = torch.stack([pos_dist.min(), pos_dist.mean(), pos_dist.max()])
      self.pos_dist_str = ','.join([f'{d:.2}' for d in self.summary_pos_dist.cpu().data.numpy()])
