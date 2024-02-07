from sensorModel import sensorModel
from lidarScan import lidarScan
from TGM import TGM
import numpy as np
import matplotlib.pyplot as plt

def run():
    # PARAMETERS

    logPath = "../logs/sim_corridor/"
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
        with open(logPath + "z_" + str(i) + ".csv") as data:
            z_t = lidarScan(*np.array([line.split(",") for line in data]).astype(float).T)

        # Get robot pose
        with open(logPath + "x_" + str(i) + ".csv") as data:
            x_t = np.array([line.split(",") for line in data]).astype(float)[0]

        # Generate instantaneous grid map
        gm = sM.generateGridMap(z_t, x_t)

        # Update TGM
        tgm.update(gm, x_t)

        # Plot maps
        fig.clear()
        tgm.plotCombinedMap(fig)


if __name__ == '__main__':
    run()