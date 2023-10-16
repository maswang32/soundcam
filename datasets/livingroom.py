import numpy as np

#Microphone Locations
mic_height = 50.3125
feet = 12
y_tile = 23.5
x_tile = 11 +7/8

camera_origin_location = np.array([-6*x_tile-5.75, -y_tile, 45+13/16])*25.4

mic_1 = np.array([-11*x_tile - 1, -5*y_tile - 6-3/8, mic_height])*25.4 #6
mic_2= np.array([-13*x_tile - 0.5 , -2*y_tile - 1, mic_height])*25.4
mic_3= np.array([-11*x_tile - 3.75 - 1/16, 15.375, mic_height])*25.4
mic_4= np.array([-11*x_tile + 1.25 + 1/16, 2.5*y_tile+3.75, mic_height])*25.4
mic_5 = np.array([-6*x_tile - 5 -15/16, 2.5*y_tile+4.75, mic_height])*25.4
mic_6 = np.array([5+3/8, 2+1/8, mic_height])*25.4
mic_7 = np.array([6, -38, mic_height])*25.4
mic_8 = np.array([12.75, -4*y_tile - 3.125, mic_height])*25.4
mic_9 = np.array([-3, -6*y_tile - 0.5, mic_height])*25.4
mic_10 = np.array([-4*x_tile - 5.75, -6*y_tile - 0.75, mic_height])*25.4

mic_xyzs = np.stack((mic_1, mic_2, mic_3, mic_4, mic_5, mic_6, mic_7, mic_8, mic_9, mic_10),axis=0)

SPEAKER_BOTTOM_RIGHT_Y = (1200.15+1196.975+1206.5)/3
SPEAKER_BOTTOM_RIGHT_X = (88.9 +107.95+101.6)/3
SPEAKER_BOTTOM_LEFT_Y = (1327.15+1311.55712764+1317.625)/3
SPEAKER_BOTTOM_LEFT_X = -76.98583188

speaker_xyz_bottom_right = np.array([SPEAKER_BOTTOM_RIGHT_X, SPEAKER_BOTTOM_RIGHT_Y, 44.5*25.4])
speaker_xyz_bottom_left = np.array([SPEAKER_BOTTOM_LEFT_X, SPEAKER_BOTTOM_LEFT_Y, 44.5*25.4])
speaker_xyz_top_right = np.array([SPEAKER_BOTTOM_RIGHT_X, SPEAKER_BOTTOM_RIGHT_Y, (44.5+17)*25.4])
speaker_xyz_top_left = np.array([SPEAKER_BOTTOM_LEFT_X, SPEAKER_BOTTOM_LEFT_Y, (44.5+17)*25.4])
speaker_xyz = (speaker_xyz_bottom_right+speaker_xyz_bottom_left+speaker_xyz_top_right+speaker_xyz_top_left)/4


walls=None
x_min= -4000
x_max= 500
y_min= -4000
y_max= 2000

# cr = dataset.Dataset(roomsetup.RoomSetup(speaker_xyz,
#                 mic_xyzs,
#                 x_min,
#                 x_max,
#                 y_min,
#                 y_max,
#                 walls), DATASET_PATH)
