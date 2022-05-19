import numpy as np
from autolab_core.tensor_dataset import TensorDataset
import json
from path_settings import DATA_PATH
import os
from tqdm import tqdm

"""
Script to create a undersampled dataset from a start dataset. Grasps are skipped to end up in the
same number of grasps in each delta phi grid of 5 degree
"""

def get_undersample_ratio(phi_grasps):
    """
    Get undersampling ratio based on number of grasps per 5degrees in phi.

    Parameters
    ----------
    phi_grasps (np.array): Number of grasps in each phi step


    Returns
    -------
    undersample_ratio (np.array): Undersampling ratio for each delta phi of 5 degrees to end up with a stable number
                                    of grasps over phi.
    """
    goal_per_grid = min(phi_grasps)
    undersample_ratio = goal_per_grid / phi_grasps

    return undersample_ratio


# Paths to datasets
data_dir = DATA_PATH + 'vg_dset_sample/tensors/'
output_dir = DATA_PATH + 'vg_dset_phi_num/'

# Initiate tensor datasets
config_filename = data_dir + '../config.json'
with open(config_filename, 'r') as myfile:
    data = myfile.read()
tensor_config = json.loads(data)
undersampled_dset = TensorDataset(output_dir, tensor_config)
tensor_datapoint = undersampled_dset.datapoint_template

# Load sampling distribution
bg = np.load(data_dir + '../phi_grasps.npy')
undersample_distribution = get_undersample_ratio(bg)
undersample_distribution = np.append(undersample_distribution, 1.0)

# Load tensor identifiers from dataset path
files_in_dir = os.listdir(data_dir)
filetypes = list(set(['_'.join(file.split('_')[:-1]) for file in files_in_dir]))
tensors_in_dir = list(set([string.split('_')[-1][:-4] for string in files_in_dir]))
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
image_label = None
for tensor in tqdm(tensors_in_dir):
    # Load tensor
    for filetype in filetypes:
        data[filetype] = np.load("%s%s_%s.npz" % (data_dir, filetype, tensor))['arr_0']
    for cnt in range(len(data['image_labels'])):
        phi_angle = int(np.rad2deg(data['camera_poses'][cnt, 1]) // 5)
        # Skip grasps according to sampling probability
        p_take = undersample_distribution[phi_angle]
        if np.random.choice([False, True], p=[p_take, 1 - p_take]):
            continue
        # Save grasps if not skipped
        for filetype in filetypes:
            tensor_datapoint[filetype] = data[filetype][cnt]
        undersampled_dset.add(tensor_datapoint)

undersampled_dset.flush()
