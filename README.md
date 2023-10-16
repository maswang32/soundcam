# SoundCam: A Dataset for Finding Humans Using Room Acoustics

### [Website](https://sites.google.com/view/soundcam) | [Dataset](https://purl.stanford.edu/xq364hd5023) | [Video](https://www.youtube.com/watch?v=HAhJLgj8maI) | [Paper]()

Code providing examples for loading and running experiments on the SoundCam Dataset.

[Mason Wang<sup>1</sup>](https://www.linkedin.com/in/mason-wang-3b5288104/) | [Samuel Clarke<sup>1</sup>](https://samuelpclarke.com/) | [Jui-Hsien Wang<sup>2</sup>](http://juiwang.com/) | [Ruohan Gao<sup>3</sup>](https://ruohangao.github.io/) | [Jiajun Wu<sup>1</sup>](jiajunwu.com)

<sup>1</sup>Stanford, <sup>2</sup>Adobe, <sup>3</sup>Meta Reality Labs

NeurIPS Datasets and Benchmarks Poster 2023.

## Downloading our Dataset

Simply go to the directory you would like to download the dataset in, and run `./download.sh`. The dataset is also available for download [here](https://purl.stanford.edu/xq364hd5023).

## Loading our Dataset

An example for loading the Treated Room dataset's Room Impulse Responses is shown in loading_data_examples.ipynb. The file also provides an example for loading training data.

## Organization


### indices
The `indices` folder contains `.npy` files with train/valid/test splits for all of the tasks. `train_indices.npy`, `valid_indices.npy`, and `test_indices.npy`  contain numpy arrays of length 800, 100, and 100, respectively, and provide the indices of the intended training, validation, and test sets, respectively for the localization task. The `..._class.npy` and `..._binary.npy` files contain similar splits for the 5-way classification task and the binary detection task.

### VGGish
This folder contains source code for running the VGGish-based baselines described in our paper. `train_vggish_class.py` should be used for the detection and identification tasks, while `train_vggish_localization` should be used for the localization task.

### posekernel
This folder contains source code for running the time-of-arrival baseline on the localization task. `posekernellifter.py` precomputes a set of spatial encodings for this task, and `pk_inference` is used to use these spatial encodings to predict the person's location.

### datasets
This folder contains information about each of the datasets collected, including speaker and microphone locations.


## Running Baselines on the SoundCam Dataset


Here is an example command to run the VGGish baseline on the living room dataset, with all 10 microphones.

```
python train_vggish_localization.py <path to audio> <path to centroid.npy> --error_path <directory to save errors> --save_path <path to save model weights> --num_channels 10 --multi_chan --living
```

Here is an example command to run person identification:

```
python train_vggish_class.py <path to audio> /../indices/labels_class.npy --error_path <directory to save errors> --save_path <path to save model weights> --num_channels 10 --multi_chan --darkroom --num_categories 5 --train_indices /../indices/train_indices_class.npy --valid_indices /../indices/valid_indices_class.npy --test_indices /../indices/test_indices_class.npy
```


Here is an example to run binary detection using the pretrained VGGish with resampling:

```
python train_vggish_class.py <path to audio> /../indices/labels_binary.npy --error_path <directory to save errors> --save_path <path to save model weights> --num_channels 10 --darkroom --num_categories 1 --pretrained --resample --train_indices  /../indices/train_indices_empty.npy --valid_indices /../indices/valid_indices_empty.npy --test_indices /../indices/test_indices_empty.npy --lr 0.0001 --empty_dir <path to audio from empty room>
```

## Citation
```
@article{soundcam2023,
  title={SoundCam: A Dataset for Finding Humans Using Room Acoustics},
  author={Mason Wang and Samuel Clarke and Jui-Hsien Wang and Ruohan Gao and Jiajun Wu},
  year={2023},
  booktitle={Arkiv},
}
```
