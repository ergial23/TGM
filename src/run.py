from sensorModel import sensorModel
from lidarScan import lidarScan, lidarScan3D
from gridMap import gridMap
from TGM import TGM
from SLAM import lsqnl_matching
import numpy as np
import matplotlib.pyplot as plt
import time
import subprocess
import os

def readLidarData(path, i):
    with open(path + "z_" + str(i) + ".csv") as data:
        z_t = lidarScan(*np.array([line.split(",") for line in data]).astype(float).T)
    return z_t

def readLidarData3D(path, i):
    with open(path + "z_" + str(i) + ".csv") as data:
        z_t_3D = lidarScan3D(np.array([line.split(",") for line in data]).astype(float))
        z_t = z_t_3D.removeGround(0).removeSky(1).convertTo2D()
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

    saveVideo = False

    logID = "2024-02-13-10-36-09"
    path = "../logs/" + logID + "/"

    origin = [0,0]
    width = 150
    height = 150
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

    initialTimeStep = 350
    simHorizon = 350

    # Create Sensor Model and TGM
    sM = sensorModel(origin, width, height, resolution, sensorRange, invModel ,occPrior)
    tgm = TGM(origin, width, height, resolution, staticPrior, dynamicPrior, weatherPrior, maxVelocity, saturationLimits)

    # Main loop
    fig= plt.figure()
    for i in range(initialTimeStep, initialTimeStep + simHorizon):
        start = time.time()

        # Get sensor data
        if is3D:
            z_t = readLidarData3D(path, i).voxelGridFilter(1/resolution)
        else:
            z_t = readLidarData(path, i)

        # Compute robot pose with SLAM or get it from log
        if not isSLAM:
            x_t = readPoseData(path, i)
        elif i <= initialTimeStep + numTimeStepsSLAM:
            try:
                x_t = readPoseData(path, i)
            except:
                x_t = np.array([width/2, height/2, 0])
        else:
            x_t = lsqnl_matching(z_t, tgm.computeStaticGridMap(), x_t, sensorRange).x

        print(x_t)

        # Generate instantaneous grid map
        gm = sM.generateGridMap(z_t, x_t)

        # Update TGM
        tgm.update(gm, x_t)

        # Plot maps
        fig.clear()
        tgm.plotCombinedMap(fig, saveImg=saveVideo, imgName='../videos/frame_' + str(i))
        print('Time: ' + str(time.time() - start))

    # Save video
    if saveVideo:
        subprocess.call(['ffmpeg', '-framerate', '8', '-i', '../videos/frame_%d.png', '-r', '10', '-pix_fmt', 'yuv420p','../videos/' + logID + '.mp4'])
        for file in os.listdir('../videos/'):
            if file.endswith('.png'):
                os.remove('../videos/' + file)

if __name__ == '__main__':
    run()