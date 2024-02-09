import matplotlib.pyplot as plt
import numpy as np

class lidarScan:
    def __init__(self, angles, ranges):
        self.ranges = ranges
        self.angles = angles
        self.numReadings = len(ranges)

    def computeCartesian(self):
        return np.column_stack([self.ranges * np.cos(self.angles), self.ranges * np.sin(self.angles)])
    
    def removeNoReturn(self, maxRange):
        self.angles = self.angles[self.ranges < maxRange]
        self.ranges = self.ranges[self.ranges < maxRange]
        self.numReadings = len(self.ranges)

    def computeRelativeCartesian(self, relPose):
        angles = self.angles + relPose[2]
        x = self.ranges * np.cos(angles) + relPose[0]
        y = self.ranges * np.sin(angles) + relPose[1]
        return np.column_stack([x, y])

    def plot(self, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.plot(self.Cartesian[:, 0], self.Cartesian[:, 1], 'r.')