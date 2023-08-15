# SoundCam: A Dataset for Tasks in Tracking and Identifying Humans from Real Room Acoustics
This repository provides code for running the baselines described in the SoundCam paper. Please check out our website, where the dataset is hosted: https://sites.google.com/view/soundcam

Here is an example command to run the VGGish baseline on the living room dataset, with all 10 microphones.

```python train_vggish_localization.py <path to deconvolved.npy> <path to centroid.npy> --error_path <directory to save errors> --save_path <path to save model weights> --num_channels 10 --multi_chan --living'''

