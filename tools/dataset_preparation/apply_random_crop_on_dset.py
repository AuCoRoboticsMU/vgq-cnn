import numpy as np
from autolab_core.tensor_dataset import TensorDataset
import json
from path_settings import DATA_PATH
import os
from tqdm import tqdm
from shutil import copyfile
from gqcnn.utils import random_im_crop
from PIL import Image

"""
Applies random cropping to a dataset. Has to be used on the rendered image data before training the network.
"""

# Adjust parameters if needed
im_height = 32
im_width = 32
kappa = 36

# Adjust dataset name
dset = 'vgq_dset_sample'

# Paths to dataset
data_dir = DATA_PATH + dset + '/tensors/'
validation_dir = DATA_PATH + dset + '_{}px_{}k/'.format(im_height, kappa)


# Initiate tensor datasets
config_filename = data_dir + '../config.json'
with open(config_filename, 'r') as myfile:
    data = myfile.read()
tensor_config = json.loads(data)
tensor_config['fields']['depth_ims_tf_table'] = {'channels': 1,
                                                 'dtype': "float64",
                                                 'height': im_height,
                                                 'width': im_width}

validation_dset = TensorDataset(validation_dir, tensor_config)
tensor_datapoint = validation_dset.datapoint_template

# Copy config file if available
try:
    copyfile(data_dir + 'config.json', validation_dir + '/tensors/config.json')
except FileNotFoundError:
    print("Cant find file {}".format(data_dir + 'config.json'))

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
for filetype in unwanted_filetypes:
    if filetype in filetypes:
        filetypes.remove(filetype)


# Apply random crop to image data. Save in original tensor file format
print("Crop data")
im = None
data = {}
for tensor in tqdm(tensors_in_dir):
    for filetype in filetypes:
        data[filetype] = np.load(data_dir + filetype + '_' + tensor + '.npz')['arr_0']
    for cnt in range(0, len(data['image_labels'])):
        for filetype in filetypes:
            tensor_datapoint[filetype] = data[filetype][cnt]

        grasp = data['hand_poses'][cnt]
        if cnt == 0 or data['image_labels'][cnt-1] != data['image_labels'][cnt]:
            filename = '../images/depth_im_{:07d}.npz'.format(data['image_labels'][cnt])
            im = np.load(data_dir + filename)['arr_0'].squeeze()
        cropped_im, new_grasp = random_im_crop(im,
                                               (96, 96),
                                               grasp[0],
                                               grasp[1],
                                               kappa=kappa,
                                               debug=False)
        if cropped_im is None:
            print("Image could not be cropped. Skip.")
            continue
        if im_height != 96:
            cropped_im = np.asarray(Image.fromarray(cropped_im).resize((im_height,
                                                                        im_width),
                                                                       resample=Image.BILINEAR))

            tensor_datapoint['hand_poses'][0] = new_grasp[0] // 3
            tensor_datapoint['hand_poses'][1] = new_grasp[1] // 3
        else:
            tensor_datapoint['hand_poses'][0] = new_grasp[0]
            tensor_datapoint['hand_poses'][1] = new_grasp[1]
        tensor_datapoint['depth_ims_tf_table'] = cropped_im.reshape((im_height,
                                                                     im_width,
                                                                     1))
        validation_dset.add(tensor_datapoint)
validation_dset.flush()

