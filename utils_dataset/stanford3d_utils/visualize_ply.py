from stanford_pcl_dataset import StanfordPclDataset

def main():
  sfd_dataset = StanfordPclDataset(ann_file = '/home/z/Research/mmdetection/data/stanford',
                                   img_prefix = '/train.txt')

  if __name__ == '__main__':
    main()
