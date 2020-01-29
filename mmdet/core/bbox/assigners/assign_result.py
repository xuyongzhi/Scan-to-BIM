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

    def __init__(self, num_gts, gt_inds, max_overlaps, labels=None, env=''):
        self.num_gts = num_gts
        self.gt_inds = gt_inds
        self.max_overlaps = max_overlaps
        self.labels = labels
        self.pos_dist = None

        # for debug:
        if self.gt_inds is not None:
            self.pos_inds = torch.nonzero(self.gt_inds > 0).squeeze()
            self.neg_inds = torch.nonzero(self.gt_inds == 0).squeeze()
            self.ign_inds = torch.nonzero(self.gt_inds < 0).squeeze()
            self.num_pos_inds = self.pos_inds.numel()
            self.num_neg_inds = self.pos_inds.numel()
            self.num_ign_inds = self.pos_inds.numel()

            self.lost_gt = self.num_pos_inds < self.num_gts
            if self.lost_gt:
              print('gt num = {}, pos inds num = {}, lost gt!' +\
                    '\n\t core/bbox/assigners/assign_result.py\n\t ' + env\
              .format(self.num_gts, self.num_pos_inds) )
              pass

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
            parts.append('gt_inds={!r}'.format(self.gt_inds))
        else:
            parts.append('gt_inds.shape={!r}, pos_inds.shape={!r}'.format(
                self.gt_inds.shape, self.pos_inds.shape))
        if self.max_overlaps is None:
            parts.append('max_overlaps={!r}'.format(self.max_overlaps))
        else:
            parts.append('max_overlaps.shape={!r}'.format(
                tuple(self.max_overlaps.shape)))
        if self.labels is None:
            parts.append('labels={!r}'.format(self.labels))
        else:
            parts.append('labels.shape={!r}'.format(tuple(self.labels.shape)))
        if self.pos_dist is not  None:
            parts.append('\npos_dist summary = {} '.format(
              self.summary_pos_dist))
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
      self.summary_pos_dist = torch.stack([pos_dist.min(), pos_dist.mean(), pos_dist.std()])
