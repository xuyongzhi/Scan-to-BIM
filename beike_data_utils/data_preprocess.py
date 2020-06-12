import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)

#from gen_merged_pcl import gen_merged_pcl
#gen_merged_pcl()

'''
(1) pool_num: The number of processes.  pool_num=0 for no multi-processing.
(2) All the scenes in data/ply will be firstly sorted.
Then use scene_start and max_scene_num to control the scenes to be processed in this procedure.
Every operation will skip exist file. So  repeat process does not cost long time.
(3) all.txt is the sorted scene list from scene_start to (scene_start+max_scene_num)
train.txt, test.txt will update every time because they are randomly sampled from all.txt.
'''

# To make sure all data generated successfully. Please use pool_num=0 run all
# the pre-processing afture using pool_num>0
pool_num = 8

from gen_scene_list_scope import gen_scene_list_pcl_scope
# max_scene_num = None for all scenes
gen_scene_list_pcl_scope(scene_start=0, max_scene_num = 200, pool_num=pool_num)

from gen_topview_from_pcl import gen_top_view
gen_top_view( pool_num=pool_num)

from beike_utils import gen_connection_gt
gen_connection_gt(pool_num)
