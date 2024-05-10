# SWformer-VO: A Monocular Visual Odometry Model Based on Swin Transformer （https://ieeexplore.ieee.org/document/10490096）

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]

This paper benefited from TSformer - VO, thank them for their contribution to (https://github.com/aofrancani/TSformer-VO)
## 0. Abstract
Traditional deep learning-based visual odometry estimation methods typically involve depth point cloud data, optical flow data, images, and manually designed geometric constraints.  These methods explore both temporal and spatial dimensions to facilitate the regression of pose data models, heavily relying on data-driven techniques and manual engineering designs. The aim of this paper is to pursue a simpler end-to-end approach, transforming such problems into image-based ones, achieving 6DoF visual odometry regression solely through continuous grayscale image data, thus reaching state-of-the-art performance levels. This paper introduces a novel monocular visual odometry network structure, leveraging the Swin Transformer as the backbone network, named SWformer-VO. It enables direct estimation of the six degrees of freedom camera pose under monocular camera conditions, utilizing a modest volume of image sequence data through an end-to-end methodology. SWformer-VO introduces an Embed module called "Mixture Embed," which fuses consecutive pairs of images into a single frame and converts them into tokens passed into the backbone network.  This approach replaces traditional temporal sequence schemes by addressing the problem at the image level.Building upon this foundation, the paper continually improves and optimizes various parameters of the backbone network. Additionally, experiments are conducted to explore the impact of different layers and depths of the backbone network on accuracy. Excitingly, on the KITTI dataset, SWformer-VO demonstrates superior accuracy compared to common deep learning-based methods such as SFMlearner, Deep-VO, TSformer-VO, Depth-VO-Feat, GeoNet, Masked Gans, and others introduced in recent years. Moreover, the effectiveness of SWformer-VO is also validated on our self-collected dataset consisting of nine indoor corridor routes for visual odometry.

## 1. Dataset
Download the [KITTI odometry dataset (grayscale).](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)


Create a simbolic link (Windows) or a softlink (Linux) to the dataset in the `dataset` folder:

- On Windows:
```mklink /D <path_to_your_project>\code_root_dir\data <path_to_your_downloaded_data>```
- On Linux: 
```ln -s <path_to_your_downloaded_data> <path_to_your_project>/code_root_dir/data```

The data structure should be as follows:
```
|---code_root_dir
    |---data
        |---sequences_jpg
            |---00
                |---image_0
                    |---000000.png
                    |---000001.png
                    |---...
                |---image_1
                    |...
                |---image_2
                    |---...
                |---image_3
                    |---...
            |---01
            |---...
		|---poses
			|---00.txt
			|---01.txt
			|---...
```


## 2. Setup
- Create a virtual environment using Anaconda and activate it:
```
conda create -n vo python==3.8.0
conda activate vo
```
- Install dependencies (with environment activated):
```
pip install -r requirements.txt
```

## 3. Usage

**PS**: So far we are changing the settings and hyperparameters directly in the variables and dictionaries. As further work, we will use pre-set configurations with the `argparse` module to make a user-friendly interface.

### 3.1. Training

In `train.py`:
- Manually set configuration in `args` (python dict);
- Manually set the model hyperparameters in `model_params` (python dict);
- Save and run the code `train.py`.

### 3.2. Inference

In `predict_poses.py`:
- Manually set the variables to read the checkpoint and sequences.

| **Variables**   | **Info**                                                                                                             |
|-----------------|----------------------------------------------------------------------------------------------------------------------|
| checkpoint_path | String with the path to the trained model you want to use for inference.  Ex: checkpoint_path = "checkpoints/Model1" |
| checkpoint_name | String with the name of the desired checkpoint (name of the .pth file).  Ex: checkpoint_name = "checkpoint_model2_exp19" |
| sequences       | List with strings representing the KITTI sequences.  Ex: sequences = ["03", "04", "10"]                              |

### 3.3. Visualize Trajectories
In `plot_results.py`:
- Manually set the variables to the checkpoint and desired sequences, similarly to [Inference](#42-inference)


## 4. Evaluation
The evaluation is done with the [KITTI odometry evaluation toolbox](https://github.com/Huangying-Zhan/kitti-odom-eval). Please go to the [evaluation repository](https://github.com/Huangying-Zhan/kitti-odom-eval) to see more details about the evaluation metrics and how to run the toolbox.


## Citation
Please cite our paper you find this research useful in your work:

```@ARTICLE{10490096,
  author={Wu, Zhigang and Zhu, Yaohui},
  journal={IEEE Robotics and Automation Letters}, 
  title={SWformer-VO: A Monocular Visual Odometry Model Based on Swin Transformer}, 
  year={2024},
  volume={9},
  number={5},
  pages={4766-4773},
  keywords={Transformers;Visual odometry;Cameras;Training;Odometry;Image segmentation;Deep learning;Deep learning;monocular visual odometry;transformer},
  doi={10.1109/LRA.2024.3384911}}

```

## References

（https://github.com/aofrancani/TSformer-VO）

