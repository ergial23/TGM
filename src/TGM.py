import numpy as np
import matplotlib.pyplot as plt
from gridMap import gridMap
from skimage.morphology import disk
from scipy.signal import convolve2d

class TGM:
    def __init__(self,origin, width, height, resolution, staticPrior, dynamicPrior, weatherPrior, maxVelocity, saturationLimits):
        self.origin = origin
        self.width = width
        self.height = height
        self.resolution = resolution

        self.staticPrior = staticPrior
        self.dynamicPrior = dynamicPrior
        self.weatherPrior = weatherPrior
        self.freePrior = 1 - staticPrior - dynamicPrior - weatherPrior

        r = int(maxVelocity * resolution)
        shape = disk(r).astype(float)
        self.D0 = 1 / np.sum(shape)
        shape /= np.sum(shape)
        shape[len(shape)//2, len(shape)//2] = 0
        self.convShape = shape

        self.staticMap = np.ones((width*resolution, height*resolution)) * staticPrior
        self.dynamicMap = np.ones((width*resolution, height*resolution)) * dynamicPrior
        self.weatherMap = np.ones((width*resolution, height*resolution)) * weatherPrior

        self.satLowS = saturationLimits[0]
        self.satHighS = saturationLimits[1]
        self.satLowD = saturationLimits[2]
        self.satHighD = saturationLimits[3]

        self.x_t = []

    def update(self, instGridMap, x_t):
        assert isinstance(instGridMap, gridMap)
        instMap = instGridMap.data
        # Update ego position (used for visualization purposes only)
        self.x_t = x_t
        # Split the instantaneous map into static, dynamic, weather and free maps
        instStaticMap = instMap * self.staticPrior / (self.staticPrior + self.dynamicPrior + self.weatherPrior)
        instDynamicMap = instMap * self.dynamicPrior / (self.staticPrior + self.dynamicPrior + self.weatherPrior)
        instWeatherMap = instMap * self.weatherPrior / (self.staticPrior + self.dynamicPrior + self.weatherPrior)
        instFreeMap = 1 - instStaticMap - instDynamicMap - instWeatherMap
        # Predict based on previous measurements
        predStaticMap, predDynamicMap, predWeatherMap = self.predict()
        predFreeMap = 1 - predStaticMap - predDynamicMap - predWeatherMap
        #
        nStatic = instStaticMap * predStaticMap / self.staticPrior
        nDynamic = instDynamicMap * predDynamicMap / self.dynamicPrior
        if self.weatherPrior != 0:
            nWeather = instWeatherMap * predWeatherMap / self.weatherPrior
        else:
            nWeather = np.zeros_like(instWeatherMap)
        nFree = instFreeMap * predFreeMap / self.freePrior

        total = nStatic + nDynamic + nWeather + nFree
        staticMatrix = nStatic / total
        dynamicMatrix = nDynamic / total
        weatherMatrix = nWeather / total

        staticMatrix[staticMatrix > self.satHighS] = self.satHighS
        staticMatrix[staticMatrix < self.satLowS] = self.satLowS

        dynamicMatrix[dynamicMatrix > self.satHighD] = self.satHighD
        dynamicMatrix[dynamicMatrix < self.satLowD] = self.satLowD

        self.staticMap = staticMatrix
        self.dynamicMap = dynamicMatrix
        self.weatherMap = weatherMatrix

    def predict(self):
        predStaticMap = self.staticMap

        dynamicStay = self.dynamicMap * self.D0
        bounceBack = conv2prior(self.staticMap, self.convShape, self.staticPrior) * self.dynamicMap
        dynamicMove = conv2prior(self.dynamicMap, self.convShape, self.dynamicPrior) * (1 - self.staticMap)

        predDynamicMap = dynamicStay + bounceBack + dynamicMove

        predWeatherMap = (1 - predStaticMap - predDynamicMap) * self.weatherPrior / (self.weatherPrior + self.freePrior)

        return predStaticMap, predDynamicMap, predWeatherMap
    
    def plotStaticMap(self, fig=None):
        if fig is None:
            fig = plt.figure()
        I = 1 - np.transpose(self.staticMap)
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(I, cmap="gray", vmin=0, vmax=1, origin ="lower",
                   extent=(self.origin[0], self.origin[0] + self.width,
                           self.origin[1], self.origin[1] + self.height))
        if self.x_t is not None and len(self.x_t) != 0:
            plt.plot(self.x_t[0], self.x_t[1], 'ro')
        plt.show()

    def plotDynamicMap(self, fig=None):
        if fig is None:
            fig = plt.figure()
        I = 1 - np.transpose(self.dynamicMap)
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(I, cmap="gray", vmin=0, vmax=1, origin ="lower",
                   extent=(self.origin[0], self.origin[0] + self.width,
                           self.origin[1], self.origin[1] + self.height))
        if self.x_t is not None and len(self.x_t) != 0:
            plt.plot(self.x_t[0], self.x_t[1], 'ro')
        plt.show()

    def plotCombinedMap(self, fig=None):
        if fig is None:
            fig = plt.figure()
        I = np.zeros((self.height*self.resolution, self.width*self.resolution, 3))
        I[:,:,0] = 1 - np.transpose(1.0*self.staticMap + 0.0*self.dynamicMap + 1.0*self.weatherMap)
        I[:,:,1] = 1 - np.transpose(0.5*self.staticMap + 0.5*self.dynamicMap + 0.0*self.weatherMap)
        I[:,:,2] = 1 - np.transpose(0.0*self.staticMap + 1.0*self.dynamicMap + 1.0*self.weatherMap)
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(I, vmin=0, vmax=1, origin ="lower",
                   extent=(self.origin[0], self.origin[0] + self.width,
                           self.origin[1], self.origin[1] + self.height))
        if self.x_t is not None and len(self.x_t) != 0:
            plt.plot(self.x_t[0], self.x_t[1], 'ro')
        plt.pause(0.1)
    
def conv2prior(map, convShape, prior):
    # Pad the map with the prior before making the convolution
    sx, sy = convShape.shape
    px = (sx - 1) // 2
    py = (sy - 1) // 2
    paddedMap = np.pad(map, ((px, px), (py, py)), constant_values=prior)
    intConv = convolve2d(paddedMap, convShape, mode='valid')
    conv = intConv
    return conv

if __name__ == '__main__':
    origin = [0, 0]
    width = 10
    height = 5
    resolution = 2
    staticPrior = 0.3
    dynamicPrior = 0.3
    weatherPrior = 0.01
    maxVelocity = 1
    saturationLimits = [0.1, 0.9, 0.1, 0.9]
    tgm = TGM(origin, width, height, resolution, staticPrior, dynamicPrior, weatherPrior, maxVelocity, saturationLimits)
    tgm.plotCombinedMap()