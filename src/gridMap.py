import matplotlib.pyplot as plt
import numpy as np

class gridMap:
    def __init__(self, origin, width, height, resolution, data):
        assert len(origin) == 2
        assert data.shape[0] == width*resolution
        assert data.shape[1] == height*resolution
        self.origin = origin
        self.width = width
        self.height = height
        self.resolution = resolution
        self.data = data

    def plot(self):
        """
        Plot the grid map
        """
        I = 1 - np.transpose(self.data)
        plt.imshow(I, cmap="gray", vmin=0, vmax=1, origin ="lower",
                   extent=(self.origin[0], self.origin[0] + self.width,
                           self.origin[1], self.origin[1] + self.height))
        plt.show()

def main():
    origin = [0, 0]
    width = 10
    height = 5
    resolution = 2

    data = np.zeros((width*resolution, height*resolution))
    data[0][0] = 1
    data[19][0] = 0.5
    
    gridMap(origin, width, height, resolution, data).plot()

if __name__ == '__main__':
    main()