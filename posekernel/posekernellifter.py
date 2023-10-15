import numpy as np
import torch
import scipy.signal as signal
import matplotlib.pyplot as plt
from pylab import rcParams
import argparse
import os.path

import sys
sys.path.insert(0, '../datasets/')
import dataset
import roomsetup
import darkrooms



#Change this if we're using a different room
room = darkrooms.dr.room_setup


fs = 48000


def compute_pk_new(e_rirs, f_rirs, envelope_size = 10, smooth=True, direct=False):
    
    n_data = e_rirs.shape[0]
    n_mics = e_rirs.shape[1]
    total_length = e_rirs.shape[2]


    #Compute Average Empty RIR
    e_rir = np.mean(e_rirs, axis=0)

    if not direct:
        print("convolving")
        if smooth:
            e_rir_env = np.sqrt(signal.convolve(e_rir**2, np.ones((1, envelope_size)))[:, int(envelope_size/2):])
            f_rir_env = np.sqrt(signal.convolve(f_rirs**2, np.ones((1, 1, envelope_size)))[:, :, int(envelope_size/2):])

        else:
            e_rir_env = np.abs(e_rir)
            f_rir_env =  np.abs(f_rir)

        e_rir_env = np.nan_to_num(e_rir_env, nan=0)
        f_rir_env = np.nan_to_num(f_rir_env, nan=0)

        print("done convolving")
        # return e_rir_env, f_rir_env
        # pose_kernels = np.maximum(f_rir_env-e_rir_env, 0)
        pose_kernels = 2*(f_rir_env-e_rir_env)/np.maximum(0.01, np.abs(f_rir_env+e_rir_env))
    else:
        print("subtracting")
        differences = f_rirs - e_rir
        print("convolving")
        pose_kernels = np.sqrt(signal.convolve(differences**2, np.ones((1, 1, envelope_size)))[:, :, int(envelope_size/2):])
        pose_kernels = np.nan_to_num(pose_kernels, nan=0)

    return pose_kernels

def compute_tdoa(speaker_pos, mic_pos, xy):
    speaker_to_xy = np.linalg.norm(speaker_pos[:2] - xy)
    xy_to_mic = np.linalg.norm(mic_pos[:2] - xy)
    total_2d_dist = speaker_to_xy + xy_to_mic
    time = total_2d_dist/343000
    return time


class PoseKernelLifter(object):
    def __init__(self,
            room_setup,
            hm_shape_0,
            hm_shape_1):

        self.r = room_setup
        self.hm_shape_0 = hm_shape_0
        self.hm_shape_1 = hm_shape_1
        x = np.linspace(room_setup.x_min, room_setup.x_max, hm_shape_0)
        y = np.linspace(room_setup.y_min, room_setup.y_max, hm_shape_1)
        X, Y = np.meshgrid(x, y)
        self.XY = np.stack((X,Y), axis=-1)
    
    def compute_spatial(self, mic_idx, kernel, smooth=True):
        
        result = torch.zeros((self.hm_shape_1, self.hm_shape_0))
        
        #print(mic_idx)
        for i in range(self.XY.shape[0]):
            for j in range(self.XY.shape[1]):
                xy = self.XY[i,j]
                tdoa = compute_tdoa(self.r.speaker_xyz, self.r.mic_xyzs[mic_idx], xy)
                sdoa = int(round(tdoa * fs))
                result[i,j] = kernel[sdoa]
                
                if smooth:
                    result[i,j] = np.mean(kernel[sdoa-3:sdoa+3]) 
                       
        return result
    

    def image_loc(self, locs):    
        xs = locs[...,0]
        ys = locs[...,1]
        
        new_ys = -(ys - self.r.y_max)/(self.r.y_max-self.r.y_min)
        new_xs = (xs - self.r.x_min)/(self.r.x_max-self.r.x_min)
        normalized =  np.stack((new_xs, new_ys), axis=-1)
        return np.array([self.hm_shape_0,self.hm_shape_1])*normalized


    def inference(self, env_poses, centroid, plot=False, save=False, save_dir=None, alpha=3.5, mic_indices=None):

        if plot:
            #Making Graphs Bigger
            rcParams['figure.figsize'] = 7, 7

            #Converting to image coordinates
            mic_image = self.image_loc(self.r.mic_xyzs)
            speaker_image = self.image_loc(self.r.speaker_xyz)
            centroid_image = self.image_loc(centroid)

        #Keeping track of errors, mult maps
        errors = []
        spatial_encodings = []

        for i in range(env_poses.shape[0]):
            results = []

            for j in range(env_poses.shape[1]):
                if mic_indices is None:
                    results.append(self.compute_spatial(j, env_poses[i,j]).flip(0))
                else:
                    results.append(self.compute_spatial(mic_indices[j], env_poses[i,j]).flip(0))

            if len(results)==1:
                results = results[0]
            else:
                results = np.array(results)
            
            #Generate Multiply Map
            if len(mic_indices)>1:
                mult_map = np.prod((results + alpha), axis=0)
            else:
                mult_map = results+alpha
            
            #Use the maximum to get the predicted location
            max_val = (mult_map==torch.max(mult_map)).nonzero()

            #Check Shape
            predicted_loc = np.array([((max_val[0,1]/self.hm_shape_0)*(self.r.x_max-self.r.x_min)+self.r.x_min), (( (self.hm_shape_1-max_val[0,0])/self.hm_shape_1)*(self.r.y_max-self.r.y_min)+self.r.y_min)])  
            error = np.linalg.norm(predicted_loc-centroid[i])
            print("Error:")
            print(error)
            print(i)
            errors.append(error)


            if plot:
                #Display the Multiply Map
                plt.figure(1000+i)
                plt.imshow(mult_map)
                plt.scatter(centroid_image[i,0], centroid_image[i,1], c="pink", label="Ground Truth Location", s=81)
                plt.scatter(mic_image[[mic_indices],0],mic_image[[mic_indices],1], color='blue',label="Microphones", marker='X', s=144)
                plt.scatter(speaker_image[0], speaker_image[1], color='red', marker='s', label="Speaker", s=100)
                plt.xticks([])
                plt.yticks([])

                #Show the predicted location
                image_loc = self.image_loc(predicted_loc)
                plt.scatter(image_loc[0], image_loc[1], color='orange', label="Predicted location", s=81)
               
                if not success:
                    plt.legend(bbox_to_anchor=(1.03, 1.0), loc='upper left',fontsize=16)


                if save:
                    plt.savefig(os.path.join(save_dir, "{:04d}".format(i) + ".jpg"))


            print(i)
        
        return errors#, spatial_returns


#python posekernellifter.py /viscam/projects/soundcam/baseline_files/May18Real --e_rir /viscam/projects/soundcam/datasets/human_rgbd/May18Empty/preprocessed/deconvolved.npy --f_rir /viscam/projects/soundcam/datasets/human_rgbd/May18Real/preprocessed/deconvolved.npy --centroid /viscam/projects/soundcam/datasets/human_rgbd/May18Real/preprocessed/centroid.npy --stage 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('save_dir', type=str, help='Save Path for Pose Kernels')
    parser.add_argument('--e_rir', type=str, help='Directory of Empty Impulse Responses')
    parser.add_argument('--f_rir', type=str, help='Directory of Full Impulse Responses')
    parser.add_argument('--centroid', type=str, help='Directory to centroid', default='')
    parser.add_argument('--stage', type=int, help='If there has already been preprocessing', default=0)
    parser.add_argument('--subtract_direct', action='store_true', default=False)
    parser.add_argument('--prefix', type=str, help='prefix', default='')

    args = parser.parse_args()

    #Load the Room
    pkl = PoseKernelLifter(room, 101, 101)

    if args.stage<1:
        #Load Empty and Full Room Impulse Responses
        e_rirs = np.load(args.e_rir)
        print("Empty RIRs loaded")
        f_rirs = np.load(args.f_rir)
        print("Full RIRs loaded")

        #Compute Pose Kernels
        pk = compute_pk_new(e_rirs, f_rirs)
        print("Pose Kernels Computed")

        np.save(os.path.join(args.save_dir, args.prefix+"pose_kernels.npy"), pk)
        print("Pose Kernels Saved")

    else:
        pk = np.load(os.path.join(args.save_dir, args.prefix+"pose_kernels.npy"))

    print("exiting")
    exit()
    
    #Get Centroids
    centroid = np.load(args.centroid)
    c = np.array([centroid[:, 1], -centroid[:, 0]]).T + np.array([[pkl.r.camera_origin_xyz[0], pkl.r.camera_origin_xyz[1]]])

    #Make Path for Maps
    if not os.path.exists(os.path.join(args.save_dir, args.prefix+"maps")):
        os.mkdir(os.path.join(args.save_dir, args.prefix+"maps"))

    map_dir = os.path.join(args.save_dir, args.prefix+"maps")

    print("Computing Spatial Encodings")
    errors, spatial_encodings = pkl.inference(pk, c, True, True, map_dir)

    print("ALL ERRORS")
    print(errors)
    print("AVG ERROR")
    print(np.mean(errors))

    np.save(os.path.join(args.save_dir, args.prefix+"errors.npy"), errors)
    np.save(os.path.join(args.save_dir, args.prefix+"spatial_encodings.npy"), spatial_encodings)