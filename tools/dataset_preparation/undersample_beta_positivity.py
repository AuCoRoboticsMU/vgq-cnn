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

def get_undersample_ratio(positivity_rate, goal_positivity_ratio=20):
    """
    Get undersampling probability.
    Parameters
    ----------
    positivity_rate (np.array): Positivity rate in the base dataset
    goal_positivity_ratio (int): Goal positivity ratio

    Returns
    -------
    undersample_ratio (np.array): Undersampling ratio to gain a positivity ratio of gt_positivity_ratio.
    """
    scaling_factor = 100 / goal_positivity_ratio - 1
    negativity_rate = 100 - positivity_rate
    undersample_ratio = (positivity_rate * scaling_factor) / negativity_rate

    return undersample_ratio


# Paths to datasets
data_dir = DATA_PATH + 'vg_dset_sample/tensors/'
output_dir = DATA_PATH + 'vg_dset_pos_r/'

# Initiate tensor datasets
config_filename = data_dir + '../config.json'
with open(config_filename, 'r') as myfile:
    data = myfile.read()
tensor_config = json.loads(data)
undersampled_dset = TensorDataset(output_dir, tensor_config)
tensor_datapoint = undersampled_dset.datapoint_template

# Load sampling distribution
negative_distribution = get_undersample_ratio(np.load(data_dir + '../beta_positivity_rate.npy'),
                                              goal_positivity_ratio=19)
negative_distribution = np.append(negative_distribution, 1.0)
positive_distribution = np.ones(negative_distribution.shape)
positive_distribution[negative_distribution > 1] = 1 / negative_distribution[negative_distribution > 1]
negative_distribution[negative_distribution > 1] = 1.0

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
    positive = data['robust_ferrari_canny'] >= 0.002
    for cnt in range(len(data['image_labels'])):
        beta_angle = int(np.rad2deg(data['hand_poses'][cnt, -1]) // 5)
        # Skip grasps according to sampling probability
        if not positive[cnt]:
            p_take = negative_distribution[beta_angle]
            if np.random.choice([False, True], p=[p_take, 1 - p_take]):
                continue
        else:
            p_take = positive_distribution[beta_angle]
            if np.random.choice([False, True], p=[p_take, 1 - p_take]):
                continue
        # Save grasps if not skipped
        for filetype in filetypes:
            tensor_datapoint[filetype] = data[filetype][cnt]
        undersampled_dset.add(tensor_datapoint)

undersampled_dset.flush()
