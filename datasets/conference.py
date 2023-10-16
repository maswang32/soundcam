import numpy as np


#Microphone Locations
mic_height = 50.3125
feet = 12
length = 267
width = 106+23.75

#Previous
camera_origin_xyz = np.array([67+3/16, -140.5, 65+3/16])*25.4

mic_1 = np.array([98+1/8, -253.75,  mic_height])*25.4
mic_2 = np.array([53+1/16, -255.5,  mic_height])*25.4
mic_3 = np.array([7+7/8, -(length-53),  mic_height])*25.4
mic_4 = np.array([5+13/16, -(length-(101+7/8)),  mic_height])*25.4
mic_5 = np.array([6+7/8, -52.5,  mic_height])*25.4
mic_6 = np.array([14+5/8, -12.75,  mic_height])*25.4
mic_7 = np.array([width-17.25, -10,  mic_height])*25.4
mic_8 = np.array([width-(7+3/8), -47.5,  mic_height])*25.4
mic_9 = np.array([width-(7+13/16), -161.25,  mic_height])*25.4
mic_10 = np.array([width-8, -(224+7/8), mic_height])*25.4
mic_xyzs = np.stack((mic_1, mic_2, mic_3, mic_4, mic_5, mic_6, mic_7, mic_8, mic_9, mic_10),axis=0)

#Speaker
SPEAKER_BOTTOM_LEFT = np.array([3.5+57.5, -17, 8+1/8+28+13/16])*25.4
SPEAKER_BOTTOM_RIGHT = np.array([3.5+57.5+8.75, -17, 8+1/8+28+13/16])*25.4
SPEAKER_TOP_LEFT = np.array([3.5+57.5, -17, 8+1/8+28+13/16+17])*25.4
SPEAKER_TOP_RIGHT = np.array([3.5+57.5+8.75, -17, 8+1/8+28+13/16+17])*25.4
speaker_xyz = (SPEAKER_BOTTOM_LEFT+SPEAKER_BOTTOM_RIGHT+SPEAKER_TOP_LEFT+SPEAKER_TOP_RIGHT)/4


#Location of walls
top_left = np.array([0, 0])*25.4
top_right = np.array([width, 0])*25.4
bottom_left = np.array([0, -length])*25.4
bottom_right = np.array([width, -length])*25.4
walls = np.array([top_left, top_right, bottom_right, bottom_left, top_left])

#Wall Bounds
x_min = top_left[0]
x_max = top_right[0]
y_min = bottom_right[1]
y_max = top_right[1]

# conference_room = dataset.Dataset(roomsetup.RoomSetup(speaker_xyz,
#                 mic_xyzs,
#                 x_min,
#                 x_max,
#                 y_min,
#                 y_max,
#                 walls), DATASET_PATH)
