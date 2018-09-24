# PWLS-ULTRA: An Efﬁcient Clustering and Learning-Based Approach for Low-Dose 3D CT Image Reconstruction

IEEE Xplore: https://ieeexplore.ieee.org/document/8352798/

Arxiv: https://arxiv.org/abs/1703.09165

![image](https://github.com/xuehangzheng/PWLS-ULTRA-for-Low-Dose-3D-CT-Image-Reconstruction/blob/master/Diagram_of_PWLS-ULTRA.pdf)

# Data:

We provide XCAT phantom[1](both 2D slices and 3D volume), some pre-learned transforms (and dictionary), and reconstructions for initialization used in this paper. So you could quickly reproduce results in our paper right after running the main code of the specific mehthod that you are interested.

"axial_proj_data_maker_HU.m/fan_proj_data_maker_HU.m" is used for simulating 3D cone-beam (2D fan-beam) geometry, generating corresponding (noisy) sinograms, and performing FDK on these sinograms, etc.

# Implementation:

The codes should be run with Michigan Image Reconstruction Toolbox (MIRT). http://web.eecs.umich.edu/~fessler/code/index.html

Note that this toolbox only supports Linux/Mac. Before running the main code for reconstruction, you just need to run "setup.m" in MIRT.



# Comparison Methods:

We also provide our own implmentations of the comparison methods in this paper: PWLS-EP[2], PWLS-DL[3]. 


# Reference: 

[1] W. P. Segars, M. Mahesh, T. J. Beck, E. C. Frey, and B. M. W. Tsui, “Realistic CT simulation using the 4D XCAT phantom,” Med. Phys., vol. 35, no. 8, pp. 3800–3808, Aug. 2008.

[2] H. Nien and J. A. Fessler, “Relaxed linearized algorithms for faster X-ray CT image reconstruction,” IEEE Trans. Med. Imag., vol. 35, no. 4, pp. 1090–1098, Apr. 2016.

[3] Q. Xu, H. Yu, X. Mou, L. Zhang, J. Hsieh, and G. Wang, “Low-dose X-ray CT reconstruction via dictionary learning,” IEEE Trans. Med. Imag., vol. 31, no. 9, pp. 1682–1697, Sep. 2012.

# Contact:
If you have any questions or suggestions, please feel free to contact me via zhxhang@sjtu.edu.cn.
