import sys
import os
sys.path.insert(0, '../datasets/')
import numpy as np
import matplotlib.pyplot as plt

import posekernellifter
import dataset
import roomsetup
import darkrooms
import conference
import chris
import argparse


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=str, help='Path to the Dataset folder, e.g. May18')
    parser.add_argument('save_dir', type=str, help='Path to save to')
    parser.add_argument('--darkroom', action='store_true', default=False)
    parser.add_argument('--conference', action='store_true', default=False)
    parser.add_argument('--chris', action='store_true', default=False)
    parser.add_argument('--n_mics', type=int, help='Number of Microphones')
    parser.add_argument('--no_test_indices', action='store_true', default=False)

    args = parser.parse_args()

    if args.n_mics == 10:
        mic_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    if args.darkroom:
        room = darkrooms.dr.room_setup
        if args.n_mics == 4:
            mic_indices = [0, 5, 6, 9]
        if args.n_mics == 2:
            mic_indices = [0, 6]
        if args.n_mics == 1:
            mic_indices = [0]


    elif args.conference:
        room = conference.conference_room.room_setup
        if args.n_mics == 4:
            mic_indices = [0, 1, 5, 6]
        if args.n_mics == 2:
            mic_indices = [1, 6]
        if args.n_mics == 1:
            mic_indices = [1]

    elif args.chris:
        room = chris.cr.room_setup
        if args.n_mics == 4:
            mic_indices = [8, 0, 5, 3]
        if args.n_mics == 2:
            mic_indices = [0, 5]
        if args.n_mics == 1:
            mic_indices = [0]

            
    test_indices = np.load("/../indices/test_indices.npy")

    c = np.load(os.path.join(args.dataset_dir,"/preprocessed/centroid.npy"))

    if args.no_test_indices:
        test_indices = np.arange(c.shape[0])
        
    c = c[test_indices, :]
    pk_save_path = os.path.join(args.save_dir, "ds_pose_kernels.npy")
    
    print("Loading Pose Kernels")
    pks = np.load(pk_save_path, mmap_mode='r')[test_indices, :, :14400]
    pks = pks[:, mic_indices, :]

    print("Pose Kernels Loaded")
    pkl = posekernellifter.PoseKernelLifter(room, int((room.x_max - room.x_min)/50), int((room.y_max - room.y_min)/50))
    errors = pkl.inference(pks, c, False, alpha=3.5, mic_indices = mic_indices)
    
    print("\n\n\n\nNUM MICS")
    print(args.n_mics)
    print("ERRORS:")
    print(np.mean(errors))
    print(np.std(errors))
    print(np.median(errors))
    print("\n\n\n\n")


