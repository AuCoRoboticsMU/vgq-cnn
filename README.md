## VGQ-CNN: Moving Beyond Fixed Cameras and Top-Grasps for Grasp Quality Prediction

We present the Versatile Grasp Quality Convolutional Neural Network (VGQ-CNN), a grasp quality prediction 
network for 6-DOF grasps. VGQ-CNN can be used when evaluating grasps for objects seen from a wide range 
of camera poses or mobile robots without the need to retrain the network. By defining the grasp orientation 
explicitly as an input to the network, VGQ-CNN can evaluate 6-DOF grasp poses, moving beyond the 4-DOF 
grasps used in most image-based grasp evaluation methods like GQ-CNN. We train VGQ-CNN on our new Versatile 
Grasp dataset (VG-dset), containing 6-DOF grasps observed from a wide range of camera poses. VGQ-CNN 
achieves a balanced accuracy of 82.1% on our test-split while generalising to a variety of camera poses. 
Meanwhile, it achieves competitive performance for overhead cameras and top-grasps with a balanced 
accuracy of 74.2% compared to GQ-CNN's 76.6%. We also propose a modified network architecture, 
Fast-VGQ-CNN, that speeds up inference using a shared encoder architecture and can make 128 grasp quality 
predictions in 12ms on a CPU.

This codebase and our data ([data.zip](DATALINK)) is made available for other researchers to use and build upon. We describe
the use of the scripts and how to replicate our datasets and training processes below.


## Usage of the code base

To use the codebase, you will need three main directories, `$DATA_DIR`, `$RESULT_DIR` and `$SCRIPT_DIR`. While 
`$SCRIPT_DIR` is the directory of this code (e.g. `~/grasping/vgq-cnn/`), you can choose `$DATA_DIR` and `$RESULT_DIR`
freely (e.g. `~/grasping/data/` and `~/grasping/results/`)
After downloading the data, it is stored in `$DATA_DIR`, the results will be stored in `$RESULT_DIR`. 

All code is run from `$SCRIPT_DIR`.
The code available here can easiest be executed using a docker image with nvidia gpu support. 
To build the docker image, use the bash script
`build_docker.sh`. To run the docker file, adjust `$DATA_DIR` and `$RESULT_DIR` in `run_docker.sh` and run it.
`$DATA_DIR` will be mounted to `/data/` and `$RESULT_DIR` will be mounted to `/results/` in the docker container.

### Train and analyse a network

In order to train vgq-cnn, run
```
python3 tools/train.py /data/vg_dset_kappa_0_Trainingset --name $YOUR_MODEL_NAME
```

To analyse vgq-cnn on vg_dset, run
```
python3 tools/analysis.py $YOUR_MODEL_NAME /data/vg_dset_kappa_0_Testset/tensors/
```

The analysis will be stored in `$RESULTS/$YOUR_MODEL_NAME_on_vg_dset_kappa_0_Testset/`.

### Preparing a dataset

If you want to change the dataset composition and the sampling strategies, described in chapter VI.B. in
[our paper](https://arxiv.org/abs/2203.04874), you can modify and use our scripts in `tools/dataset_preparation/`.
The complete, undersampled version of [VG-dset](ZIP FILE) is available as a zip file for download. Unpack it into `$DATA_DIR`
and run/modify the following scripts for your usecase.

- Use `undersample_psi.py` to remove grasps with a high psi angle from the dataset
- Use `undersample_beta_positivity.py` to undersample a dataset for stable positivity rate among beta (grasp approach axis vs. table normal).
- Use `undersample_phi.py` to undersample a dataset for a stable number of grasps over phi (camera elevation angle).
- Use `undersample_beta.py` to undersample a dataset for a stable number of grasps over beta.
- Use `apply_random_crop_on_dset.py` to crop 300x300 images before training/testing. Kappa can be modified in the script for fast-vgq-cnn.
- Use `apply_dexnet_crop_on_dset.py` to crop 300x300 images before training/testing. Centres, rotates and crops the images.
- Use `create_objectwise_split.py` to create an object-wise split between the train/validation and test data

------------------------

### Acknowledgements

This publication has emanated from research supported in part by Grants from Science Foundation Ireland under 
Grant numbers 18/CRT/6049 and 16/RI/3399.
The opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do 
not necessarily reflect the views of the Science Foundation Ireland.

This code is based on Berkeley AUTOLAB's GQCNN package, 
available at [github](https://github.com/BerkeleyAutomation/gqcnn).

If you use the code, datasets, or models in a publication, please cite:

A. Konrad, J. McDonald and R. Villing, "VGQ-CNN: Moving beyond fixed cameras and top-grasps for grasp quality
prediction," to appear in IEEE World Congress on Computational Intelligence (WCCI), 2022.

along with

J. Mahler, J. Liang, S. Niyaz, M. Laskey, R. Doan, X. Liu, J. A. Ojea,
and K. Goldberg, “Dex-net 2.0: Deep learning to plan robust grasps with
synthetic point clouds and analytic grasp metrics,” in Robotics: Science
and Systems (RSS), 2017.

### Contact

If you're having questions about our project, please contact [Anna Konrad](mailto:anna.konrad.2020@mumail.ie),
[Prof. John McDonald](mailto:john.mcdonald@mu.ie) or [Dr. Rudi Villing](mailto:rudi.villing@mu.ie).
