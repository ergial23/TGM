from sensorModel import sensorModel
from lidarScan import lidarScan
from gridMap import gridMap
from TGM import TGM
from SLAM import lsqnl_matching
import numpy as np
import matplotlib.pyplot as plt

def readLidarData(path, i):
    with open(path + "z_" + str(i) + ".csv") as data:
        z_t = lidarScan(*np.array([line.split(",") for line in data]).astype(float).T)
    return z_t

def readPoseData(path, i):
    with open(path + "x_" + str(i) + ".csv") as data:
        x_t = np.array([line.split(",") for line in data]).astype(float)[0]
    return x_t

def run():
    # PARAMETERS

    isSLAM = True

    path = "../logs/sim_corridor/"

    origin = [0,0]
    width = 150
    height = 50
    resolution = 2

    staticPrior = 0.3
    dynamicPrior = 0.3
    weatherPrior = 0.01
    maxVelocity = 1/resolution
    saturationLimits = [0.001, 0.999, 0.001, 0.999]

    sensorRange = 50
    invModel = [0.1, 0.9]
    occPrior = staticPrior + dynamicPrior + weatherPrior

    simHorizon = 300

    # Create Sensor Model and TGM
    sM = sensorModel(origin, width, height, resolution, sensorRange, invModel ,occPrior)
    tgm = TGM(origin, width, height, resolution, staticPrior, dynamicPrior, weatherPrior, maxVelocity, saturationLimits)

    # Main loop
    fig= plt.figure()
    for i in range(1, simHorizon):
        # Get sensor data
        z_t = readLidarData(path, i)

        # Compute robot pose with SLAM or get it from log
        if (not isSLAM) or (i == 1):
            x_t = readPoseData(path, i)
        else:
            x_t = readPoseData(path, i)
            x_t = lsqnl_matching(z_t, tgm.computeStaticGridMap(), x_t, sensorRange)
            x_t = x_t.x

        print(x_t)

        # Generate instantaneous grid map
        gm = sM.generateGridMap(z_t, x_t)

        # Update TGM
        tgm.update(gm, x_t)

        # Plot maps
        fig.clear()
        tgm.plotCombinedMap(fig)


if __name__ == '__main__':
    run()