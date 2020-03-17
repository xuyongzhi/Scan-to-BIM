import platform
from functools import partial

from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from torch.utils.data import DataLoader

from .sampler import DistributedGroupSampler, DistributedSampler, GroupSampler

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def build_dataloader(dataset,
                     imgs_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     **kwargs):
    """Build PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (Dataset): A PyTorch dataset.
        imgs_per_gpu (int): Number of images on each GPU, i.e., batch size of
            each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        kwargs: any keyword argument to be used to initialize DataLoader

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    if dist:
        rank, world_size = get_dist_info()
        # DistributedGroupSampler will definitely shuffle the data to satisfy
        # that images on each GPU are in the same group
        if shuffle:
            sampler = DistributedGroupSampler(dataset, imgs_per_gpu,
                                              world_size, rank)
        else:
            sampler = DistributedSampler(
                dataset, world_size, rank, shuffle=False)
        batch_size = imgs_per_gpu
        num_workers = workers_per_gpu
    else:
        sampler = GroupSampler(dataset, imgs_per_gpu) if shuffle else None
        batch_size = num_gpus * imgs_per_gpu
        num_workers = num_gpus * workers_per_gpu

    dataset_name = dataset.__class__.__name__
    if dataset_name in ['StanfordPclDataset']:
      data_loader = build_pcl_data_loader(dataset, batch_size, sampler, num_workers, imgs_per_gpu, **kwargs)
    else:
      data_loader = DataLoader(
          dataset,
          batch_size=batch_size,
          sampler=sampler,
          num_workers=num_workers,
          collate_fn=partial(collate, samples_per_gpu=imgs_per_gpu),
          pin_memory=False,
          **kwargs)

    return data_loader

def build_pcl_data_loader(dataset, batch_size, sampler, num_workers, imgs_per_gpu, **kwargs):
  import utils_data3d.lib.transforms as t
  return_transformation = False
  limit_numpoints = 0
  if return_transformation:
    collate_fn = t.cflt_collate_fn_factory(limit_numpoints)
  else:
    collate_fn = t.cfl_collate_fn_factory(limit_numpoints)

  data_loader = DataLoader(
          dataset,
          batch_size=batch_size,
          sampler=sampler,
          num_workers=num_workers,
          collate_fn=collate_fn,
          pin_memory=False,
          **kwargs)
  return data_loader

