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

This work has been accepted for the International Joint Conference on Neural Networks (IJCNN) 2022.

Our [code](https://github.com/AuCoRoboticsMU/vgq-cnn) and our 
[data](https://doi.org/10.5281/zenodo.6606333) can be accessed here.

If you use our code, please cite

A. Konrad, J. McDonald and R. Villing, "VGQ-CNN: Moving beyond fixed cameras and top-grasps for grasp quality
prediction," to appear in International Joint Conference on Neural Networks (IJCNN), 2022.

along with

J. Mahler, J. Liang, S. Niyaz, M. Laskey, R. Doan, X. Liu, J. A. Ojea,
and K. Goldberg, “Dex-net 2.0: Deep learning to plan robust grasps with
synthetic point clouds and analytic grasp metrics,” in Robotics: Science
and Systems (RSS), 2017.

### Acknowledgements

This publication has emanated from research supported in part by Grants from Science Foundation Ireland under 
Grant numbers 18/CRT/6049 and 16/RI/3399.
The opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do 
not necessarily reflect the views of the Science Foundation Ireland.

### Contact

If you're having questions about any of our projects, please contact [Anna Konrad](mailto:anna.konrad.2020@mumail.ie),
[Prof. John McDonald](mailto:john.mcdonald@mu.ie) or [Dr. Rudi Villing](mailto:rudi.villing@mu.ie).