import numpy as np
from autolab_core.tensor_dataset import TensorDataset
import itertools
import json
from path_settings import DATA_PATH
import os
from tqdm import tqdm
from shutil import copyfile

"""
Script to split an training and testing set object wise from a dataset.
Option to replicate other split
"""

# Size of the test set
test_size = 0.1
REPLICATE = True
SINGLE_IMAGES = False

# Paths to dataset
dset = '2022_oblique-C-xyzq_2_psi_30_beta'
data_dir = '{}{}/tensors/'.format(DATA_PATH, dset)
training_dir = '{}{}_Training/'.format(DATA_PATH, dset)
testing_dir = '{}{}_Testing/'.format(DATA_PATH, dset)
replication_test_dir = DATA_PATH + 'overhead-A-xyzr_Testing/tensors/'
if 'images' in os.listdir(data_dir + '../'):
    SINGLE_IMAGES = True

# Initiate tensor datasets
config_filename = data_dir + '../config.json'
with open(config_filename, 'r') as myfile:
    data = myfile.read()
tensor_config = json.loads(data)
training_dset = TensorDataset(training_dir, tensor_config)
testing_dset = TensorDataset(testing_dir, tensor_config)
tensor_datapoint = testing_dset.datapoint_template

try:
    copyfile(data_dir + 'config.json', training_dir + '/tensors/config.json')
    copyfile(data_dir + 'config.json', testing_dir + '/tensors/config.json')
except FileNotFoundError:
    print("Could not find config file. Skip copying.")

if SINGLE_IMAGES:
    os.mkdir(training_dir + '/images/')
    os.mkdir(testing_dir + '/images/')

# Get tensor identifiers of original dataset
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

test_obj_labels = []

if REPLICATE:
    print("Load split")
    label_dir = testing_dir + '../test_object_labels.txt'
    if os.path.exists(label_dir):
        test_obj_labels = np.loadtxt(label_dir)
    else:
        files_in_dir = os.listdir(replication_test_dir)
        tensors_in_replication_dir = list(set([split[-9:-4] for split in files_in_dir]))
        tensors_in_replication_dir.sort()
        obj_labels = list(np.load(replication_test_dir + 'object_labels_' + tensors_in_replication_dir[0] + '.npz')['arr_0'])
        for tensor in tensors_in_replication_dir[1:]:
            obj_labels.extend(list(np.load(replication_test_dir + 'object_labels_' + tensor + '.npz')['arr_0']))
        test_obj_labels = list(set(obj_labels))
        test_obj_labels.sort()
        np.savetxt(label_dir, test_obj_labels, fmt='%d')
else:
    print("Plan split")
    # Load object labels
    obj_labels = np.load(data_dir + 'obj_labels_' + tensors_in_dir[0] + '.npz')['arr_0']
    for tensor in tensors_in_dir[1:]:
        obj_labels = np.concatenate((obj_labels, np.load(data_dir + 'obj_labels_' + tensor + '.npz')['arr_0']))

    # Get number of grasps per object
    grasps_per_object = np.zeros(1495)
    for x, y in itertools.groupby(obj_labels):
        grasps_per_object[x] += len(list(y))

    # Plan split
    num_test_grasps = 0
    objects = np.arange(1495)
    np.random.shuffle(objects)

    for i in range(0, 1495):
        test_obj_labels.append(objects[i])
        num_test_grasps += grasps_per_object[i]
        if num_test_grasps >= test_size * len(tensors_in_dir) * tensor_config['datapoints_per_file']:
            break

# Create split
print("Create split")
data = {}
image_label = None
cur_dir = testing_dir
for tensor in tqdm(tensors_in_dir):
    for filetype in filetypes:
        data[filetype] = np.load(data_dir + filetype + '_' + tensor + '.npz')['arr_0']
    for cnt in range(0, len(data['object_labels'])):
        # if data['image_labels'][cnt] < 100000:
        #     continue
        for filetype in filetypes:
            tensor_datapoint[filetype] = data[filetype][cnt]
        if data['object_labels'][cnt] in test_obj_labels:
            testing_dset.add(tensor_datapoint)
            cur_dir = testing_dir
        else:
            training_dset.add(tensor_datapoint)
            cur_dir = training_dir
        # if SINGLE_IMAGES and data['image_labels'][cnt] != image_label:
        #     filename = '/depth_im_{:07d}.npz'.format(data['image_labels'][cnt])
        #     copyfile(data_dir + '../images' + filename, cur_dir + 'images' + filename)
        #     image_label = data['image_labels'][cnt]
training_dset.flush()
testing_dset.flush()

