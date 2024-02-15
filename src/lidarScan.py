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
        angles = self.angles[self.ranges < maxRange]
        ranges = self.ranges[self.ranges < maxRange]
        return lidarScan(angles, ranges)

    def computeRelativeCartesian(self, relPose):
        angles = self.angles + relPose[2]
        x = self.ranges * np.cos(angles) + relPose[0]
        y = self.ranges * np.sin(angles) + relPose[1]
        return np.column_stack([x, y])

    def plot(self, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.plot(self.computeCartesian()[:, 0], self.computeCartesian()[:, 1], 'r.')
        ax.axis('equal')
        plt.show()

class lidarScan3D:
    def __init__(self, points3D):
        self.points3D = points3D
        self.numReadings = len(points3D)

    def removeGround(self, groundThreshold):
        return lidarScan3D(self.points3D[self.points3D[:, 2] > groundThreshold])
    
    def convertTo2D(self):
        return lidarScan(np.arctan2(self.points3D[:, 1], self.points3D[:, 0]), np.sqrt(self.points3D[:, 0]**2 + self.points3D[:, 1]**2))
    
    def plot(self, ax=None):
        if ax is None:
            ax = plt.gca()
        ax = plt.axes(projection='3d')  # Add this line to create a 3D projection
        ax.scatter(self.points3D[:, 0], self.points3D[:, 1], self.points3D[:,2], 'r')
        ax.axis('equal')
        plt.show()