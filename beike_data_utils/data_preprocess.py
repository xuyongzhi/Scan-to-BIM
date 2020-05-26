import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)

#from gen_merged_pcl import gen_merged_pcl
#gen_merged_pcl()


from data_writer import gen_topview
gen_topview()

from split_dataset import gen_split_scene_list
gen_split_scene_list()

from beike_utils import last_steps
last_steps()
