# SoundCam: A Dataset for Finding Humans Using Room Acoustics
This repository provides code for running the baselines described in the SoundCam paper. Please check out our website, where the dataset is hosted: https://sites.google.com/view/soundcam

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
