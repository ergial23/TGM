import matplotlib.pyplot as plt
import numpy as np

class lidarScan:
    def __init__(self, angles, ranges):
        self.ranges = ranges
        self.angles = angles
        self.numReadings = len(ranges)

    def computeCartesian(self):
        return np.column_stack([self.ranges * np.cos(self.angles), self.ranges * np.sin(self.angles)])

    def plot(self, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.plot(self.Cartesian[:, 0], self.Cartesian[:, 1], 'r.')