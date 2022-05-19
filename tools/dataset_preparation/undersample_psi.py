import numpy as np
from autolab_core.tensor_dataset import TensorDataset
from tqdm import tqdm
import json
from path_settings import DATA_PATH
import os

"""
Script to remove grasps with high psi angles.
"""

# Highest angle between grasp approach axis and camera principal ray, psi [deg], to still be included in the
# undersampled dataset
highest_psi_angle = 90

# Paths to datasets
dataset = 'vg_dset_sample'
testing_dir = DATA_PATH + dataset + '/tensors/'
stripped_dir = DATA_PATH + dataset + '_psi_%d/' % highest_psi_angle


# Initiate tensor datasets
config_filename = testing_dir + '../config.json'
with open(config_filename, 'r') as myfile:
    data = myfile.read()
tensor_config = json.loads(data)
stripped_dset = TensorDataset(stripped_dir, tensor_config)
tensor_datapoint = stripped_dset.datapoint_template

# Load tensor identifiers from original dataset
files_in_dir = os.listdir(testing_dir)
filetypes = list(set(['_'.join(file.split('_')[:-1]) for file in files_in_dir]))
if 'nfig.' in filetypes:
    filetypes.remove('nfig.')
if '' in filetypes:
    filetypes.remove('')
tensors_in_dir = list(set([split[-10:-4] for split in files_in_dir]))
tensors_in_dir.sort()
if 'nfig.' in tensors_in_dir:
    tensors_in_dir.remove('nfig.')

print("Remove grasps with high psi")
data = {}

for tensor in tqdm(tensors_in_dir):
    for filetype in filetypes:
        data[filetype] = np.load(testing_dir + filetype + '_' + tensor + '.npz')['arr_0']
    psi = np.rad2deg(data['hand_poses'][:, -2])
    for cnt in range(0, len(data['hand_poses'])):
        if psi[cnt] > highest_psi_angle:
            continue
        for filetype in filetypes:
            tensor_datapoint[filetype] = data[filetype][cnt]
        stripped_dset.add(tensor_datapoint)
stripped_dset.flush()


