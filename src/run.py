import numpy as np
import matplotlib.pyplot as plt
import time

from utilities import readLidarData, readLidarData3D, readPoseData, createVideo
from sensorModel import sensorModel
from TGM import TGM
from SLAM import lsqnl_matching

def run():
    # PARAMETERS
    logID = "2024-02-13-10-36-09"
    is3D = True
    initialTimeStep = 1
    simHorizon = 350
    
    isSLAM = True
    numTimeStepsSLAM = 1
    
    saveVideo = True
    
    origin = [0,0]
    width = 150
    height = 150
    resolution = 2
    
    staticPrior = 0.3
    dynamicPrior = 0.3
    weatherPrior = 0.01
    maxVelocity = 1/resolution
    saturationLimits = [0, 1, 0, 1]
    
    sensorRange = 50
    invModel = [0.1, 0.9]
    occPrior = staticPrior + dynamicPrior + weatherPrior

    # Paths
    logPath = './logs/' + logID + '/'
    videoPath = './videos/'

    # Create Sensor Model and TGM
    sM = sensorModel(origin, width, height, resolution, sensorRange, invModel ,occPrior)
    tgm = TGM(origin, width, height, resolution, staticPrior, dynamicPrior, weatherPrior, maxVelocity, saturationLimits)

    # Main loop
    fig= plt.figure()
    for i in range(initialTimeStep, initialTimeStep + simHorizon):
        timeStart = time.time()

        # Import sensor data
        if is3D:
            z_t_3D = readLidarData3D(logPath, i)
            z_t = z_t_3D.removeGround(-0.5).removeSky(0).convertTo2D().removeClosePoints(3).voxelGridFilter(1/resolution).orderByAngle()
        else:
            z_t = readLidarData(logPath, i)
        timeData = time.time()

        # Compute robot pose with SLAM or get it from log
        if not isSLAM:
            x_t = readPoseData(logPath, i)
        elif i <= initialTimeStep + numTimeStepsSLAM:
            try:
                x_t = readPoseData(logPath, i)
            except:
                x_t = np.array([width/2, height/2, 0])
        else:
            x_t = lsqnl_matching(z_t, tgm.computeStaticGridMap(), x_t, sensorRange).x
        timeSLAM = time.time()

        # Compute instantaneous grid map with inverse sensor model
        gm = sM.generateGridMap(z_t, x_t)
        timeSensorModel = time.time()

        # Update TGM
        tgm.update(gm, x_t)
        timeTGM = time.time()

        # Plot maps
        fig.clear()
        tgm.plotCombinedMap(fig, saveImg=saveVideo, imgName= videoPath + 'frame_' + str(i-initialTimeStep+1))
        timePlot = time.time()

        # Print times
        print('Data:    ' + str(timeData - timeStart))
        print('SLAM:    ' + str(timeSLAM - timeData))
        print('InvSenM: ' + str(timeSensorModel - timeSLAM))
        print('TGM:     ' + str(timeTGM - timeSensorModel))
        print('Plots:   ' + str(timePlot - timeTGM))
        print('Total:   ' + str(time.time() - timeStart))
        print('')

    # Save video
    if saveVideo:
        createVideo(logID, videoPath)

if __name__ == '__main__':
    run()