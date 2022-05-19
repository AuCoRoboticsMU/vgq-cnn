import numpy as np
from autolab_core.tensor_dataset import TensorDataset
import json
from path_settings import DATA_PATH
import os
from tqdm import tqdm

"""
Script to create a undersampled dataset from a start dataset. Grasps are skipped according to an
undersampling ratio to ensure a stable amount of grasps over all beta angles.
"""

def get_undersample_ratio(beta_grasps, goal=6500000):
    """
    Get undersampling ratio.
    Parameters
    ----------
    beta_grasps (np.array): Number of grasps in each beta step
    goal (int): Number of grasps in the whole dataset

    Returns
    -------
    undersample_ratio (np.array): Undersampling ratio.
    """
    goal_per_grid = min(beta_grasps)
    undersample_ratio = goal_per_grid / beta_grasps

    return undersample_ratio


# Paths to datasets
data_dir = DATA_PATH + 'vg_dset_sample/tensors/'
output_dir = DATA_PATH + 'vg_dset_beta_num/'


# Initiate tensor datasets
config_filename = data_dir + '../config.json'
with open(config_filename, 'r') as myfile:
    data = myfile.read()
tensor_config = json.loads(data)
undersampled_dset = TensorDataset(output_dir, tensor_config)
tensor_datapoint = undersampled_dset.datapoint_template

# Load sampling distribution
bg = np.load(data_dir + '../beta_grasps.npy')
undersample_distribution = get_undersample_ratio(bg, goal=62500)
undersample_distribution = np.append(undersample_distribution, 1.0)

# Load tensor identifiers from dataset path
files_in_dir = os.listdir(data_dir)
filetypes = list(set(['_'.join(file.split('_')[:-1]) for file in files_in_dir]))
tensors_in_dir = list(set([split[-9:-4] for split in files_in_dir]))
tensors_in_dir.sort()

if tensors_in_dir[0] == '.DS_S':
    tensors_in_dir.remove('.DS_S')
if 'nfig.' in tensors_in_dir:
    tensors_in_dir.remove('nfig.')

unwanted_filetypes = ['', '.DS', 'histogram_02536', 'grasp', 'force_closure', 'binary_ims_raw', 'binary_ims_tf',
                      'depth_ims_raw_table', 'depth_ims_raw', 'table_mask', 'depth_ims_tf']
for type in unwanted_filetypes:
    if type in filetypes:
        filetypes.remove(type)

data = {}
data_2 = {}
image_label = None
for tensor in tqdm(tensors_in_dir):
    # Load tensor
    for filetype in filetypes:
        data[filetype] = np.load("%s%s_%s.npz" % (data_dir, filetype, tensor))['arr_0']
    for cnt in range(len(data['image_labels'])):
        beta_angle = int(np.rad2deg(data['hand_poses'][cnt, -1]) // 5)
        # Skip grasps according to sampling probability
        p_take = undersample_distribution[beta_angle]
        if np.random.choice([False, True], p=[p_take, 1 - p_take]):
            continue
        # Save grasps if not skipped
        for filetype in filetypes:
            tensor_datapoint[filetype] = data[filetype][cnt]
        undersampled_dset.add(tensor_datapoint)

undersampled_dset.flush()
