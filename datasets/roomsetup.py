import matplotlib.pyplot as plt
from pylab import rcParams

class RoomSetup(object):
    def __init__(self,
                speaker_xyz,
                mic_xyzs,
                x_min: float,
                x_max: float,
                y_min: float,
                y_max: float,
                walls=None):

        self.speaker_xyz = speaker_xyz
        self.mic_xyzs = mic_xyzs
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.walls = walls
        self.n_mics = mic_xyzs.shape[0]

    def plot_room(self, centroid, levels=None, legend=True, camera_coords=True, vmax=0.3, vmin=0.1945841):

        
        c = centroid

        #Scatter Plot
        rcParams['figure.figsize'] = 8,8
        plt.scatter(self.speaker_xyz[0], self.speaker_xyz[1], label='Speaker', color='green')
        plt.scatter(self.mic_xyzs[:,0], self.mic_xyzs[:,1], label = 'Mics', color='orange')

        if levels is None:
            plt.scatter(c[:,0], c[:,1], label = 'Centroids', c ='green', s=4)
        else:
            plt.scatter(c[:,0], c[:,1], label = 'Human Locations', c=levels, cmap="bwr", s=9)

        if self.walls is not None:
            plt.plot(self.walls[:,0], self.walls[:,1] , marker = 'o', color='black', label = 'Walls')

        #plt.scatter(0, 0, label = 'Origin', c ='black')
        plt.xlim([self.x_min, self.x_max])
        plt.ylim([self.y_min, self.y_max])
        plt.axis('equal')

        if legend:
            plt.legend()