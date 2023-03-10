# Rotation-Invariant-Learning
The official implemetation of our CVIU 2023 publication, entitled as 

Learning rotation equivalent scene representation from instance-level semantics: A novel top-down perspective

# Background & Project Overview

Due to the variation of imaging conditions and object distributions, many visual recognition tasks need to distinguish scenes in which key objects and region of interests (RoIs) are varied greatly in terms of orientations.

In this work, we present a novel rotation equivalent scene representation learning scheme from a top-down perspective. In this scheme, no efforts are required to extract convolutional features from multiple rotated samples or by using rotated convolution filters, which perfectly eliminates the flaw in the existing bottom-up pipelines that the convolution operation is sensitive to rotation due to its rotation in-equivalent nature.
An overview of the proposed method can be found in the attached figure. 

![avatar](/framework.png)

# Developing Environment & Getting Started

The code is implemented on top of the Python3, and there are only a few dependencies that a development need to config.
Before starting the code, please ensure the below packages and the corresponding versions are available.
```
Python > 3.5

Tensorflow > 1.6

OpenCV > 3

Numpy > 1.16
```
The datasets this paper use are all publicly available, and can be found in 
<a href="https://captain-whu.github.io/AID/"> AID</a>,
<a href="https://github.com/smilell/AG-CNN"> LAG</a>, and 
<a href="https://www.researchgate.net/publication/249656240_Kylberg_Texture_Dataset_v_10"> KTD</a>, respectively
.

The ResNet-50 pre-trained model can be downloaded from <a href="https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models"> here</a> and is supposed to put into the ```checkpoint``` file folder.

# Training
Step 1, run the ```tfdata.py``` file to transfer the data into the tf.record file format.
```
python tfdata.py
```

Step 2, run the ```training.py``` file to start training.
```
python training.py
```

# Tesing

Run the ```test1.py file``` to test the performance of a single model.
```
python test1.py
```

Run the ```testall.py``` file to test the performance of all the models in the checkpoints file folder. 
```
python testall.py
```

Please note, before using the ```testall.py``` script, please remember to delete a file named ```checkpoint``` in the ```checkpoints``` file folder.

# Citation

If you find this project useful, or use the source code from this project, please consider citing our work as
```
@article{bi2023learning,
  title={Learning rotation equivalent scene representation from instance-level semantics: A novel top-down perspective},
  author={Bi, Qi and You, Shaodi and Ji, Wei and Gevers, Theo},
  journal={Computer Vision and Image Understanding},
  pages={103635},
  year={2023},
  publisher={Elsevier}
}
```

# Contact Information

Qi Bi

q.bi@uva.nl   q_bi@whu.edu.cn   2009biqi@163.com
