import numpy as np

#Locations of Cone, Tweeter, Microphones, Centroid in mm
speaker_bottom_left = np.array([-15.25, 4+15/16, 38.5])*25.4
speaker_bottom_right = np.array([-10.75, 8+7/8, 38.5])*25.4
speaker_top_left = np.array([-15.25, 4+15/16, 54.5])*25.4
speaker_top_right = np.array([-10.75, 8+7/8, 54.5])*25.4
speaker_xyz = (speaker_bottom_left+speaker_bottom_right+speaker_top_left+speaker_top_right)/4

#Microphone Locations
mic_height = 50.3125
feet = 12
mic_array_vector = np.array([1, 0, 0])
mic_1 = np.array([10*feet, -12*feet,mic_height])*25.4
mic_4 = np.array([4*feet - 1, -12*feet+21.5, 50.3125])*25.4
mic_3 = mic_4 + np.array([3.625, 0, 0])*25.4
mic_2 = mic_3 + np.array([25.375, -1.6875, 0])*25.4
mic_5 = mic_4 + np.array([-25.5625, -1.625, 0])*25.4
mic_6 = np.array([-2*feet, -12*feet, mic_height])*25.4
mic_7 = np.array([-2*feet-3.25, -2*feet+1.25, mic_height])*25.4

#This microphone was in a slightly different spot
mic_7_occlusions = np.array([-2*feet-3.25, -2*feet+1.25, mic_height])*25.4
mic_10 = np.array([12*10, 18.3125, mic_height])*25.4
mic_9 = mic_10 - 23.625*mic_array_vector*25.4
mic_8 = mic_9 - 20*mic_array_vector*25.4

mic_xyzs_base = np.stack((mic_1, mic_2, mic_3, mic_4, mic_5, mic_6, mic_7, mic_8, mic_9, mic_10), axis=0)
mic_xyzs_occlusions = np.stack((mic_1, mic_2, mic_3, mic_4, mic_5, mic_6, mic_7_occlusions, mic_8, mic_9, mic_10), axis=0)

#Location of walls
top_left = np.array([-47.5, 46.75])*25.4
top_right = np.array([10*feet+23.5, 46.75])*25.4
bottom_left = np.array([-47.5, -12*feet - 19.625])*25.4
bottom_right = np.array([10*feet+23.5, -12*feet - 19.625])*25.4
walls = np.array([top_left, top_right, bottom_right, bottom_left, top_left])

#Origin Location - NEED TO ADD
camera_origin_xyz = np.array([4*feet-10,-5*feet+10,58+3/16])*25.4

#Wall Bounds
x_min = top_left[0]
x_max = top_right[0]
y_min = bottom_right[1]
y_max = top_right[1]