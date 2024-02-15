from sensorModel import sensorModel
from lidarScan import lidarScan, lidarScan3D
from gridMap import gridMap
from TGM import TGM
from SLAM import lsqnl_matching
import numpy as np
import matplotlib.pyplot as plt
import time

def readLidarData(path, i):
    with open(path + "z_" + str(i) + ".csv") as data:
        z_t = lidarScan(*np.array([line.split(",") for line in data]).astype(float).T)
    return z_t

def readLidarData3D(path, i):
    with open(path + "z_" + str(i) + ".csv") as data:
        z_t_3D = lidarScan3D(np.array([line.split(",") for line in data]).astype(float))
        z_t = z_t_3D.removeGround(-0.5).convertTo2D()
    return z_t

def readPoseData(path, i):
    with open(path + "x_" + str(i) + ".csv") as data:
        x_t = np.array([line.split(",") for line in data]).astype(float)[0]
    return x_t

def run():
    # PARAMETERS

    isSLAM = True
    numTimeStepsSLAM = 3

    is3D = True

    #path = "../logs/sim_corridor/"
    path = "../logs/2024-02-13-10-35-56/"

    origin = [0,0]
    width = 150
    height = 50
    resolution = 2

    staticPrior = 0.3
    dynamicPrior = 0.3
    weatherPrior = 0.01
    maxVelocity = 1/resolution
    saturationLimits = [0, 1, 0, 1] # Use this for SLAM
    #saturationLimits = [0.001, 0.999, 0.001, 0.999]

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
        start = time.time()

        # Get sensor data
        if is3D:
            z_t = readLidarData3D(path, i)
        else:
            z_t = readLidarData(path, i)

        # Compute robot pose with SLAM or get it from log
        if (not isSLAM) or (i <= numTimeStepsSLAM):
            x_t = readPoseData(path, i)
        else:
            x_t = lsqnl_matching(z_t, tgm.computeStaticGridMap(), x_t, sensorRange).x

        print(x_t)

        # Generate instantaneous grid map
        gm = sM.generateGridMap(z_t, x_t)

        # Update TGM
        tgm.update(gm, x_t)

        # Plot maps
        fig.clear()
        tgm.plotCombinedMap(fig)
        print('Time: ' + str(time.time() - start))


if __name__ == '__main__':
    run()