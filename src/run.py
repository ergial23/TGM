from sensorModel import sensorModel
from TGM import TGM
import numpy as np

def run():
    # PARAMETERS
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

    simHorizon = 100

    # Create Sensor Model and TGM
    sM = sensorModel(origin, width, height, resolution, sensorRange, invModel ,occPrior)
    tgm = TGM(origin, width, height, resolution, staticPrior, dynamicPrior, weatherPrior, maxVelocity, saturationLimits)

    # Main loop
    for i in range(1, simHorizon):
        # Get sensor data and robot pose
        with open("../logs/sim_corridor/z_" + str(i) + ".csv") as data:
            z_t = np.array([line.split(",") for line in data]).astype(float)
        
        with open("../logs/sim_corridor/x_" + str(i) + ".csv") as data:
            x_t = np.array([line.split(",") for line in data]).astype(float)[0]

        # Generate instantaneous grid map
        gm = sM.generateGridMap(z_t, x_t)
        gm.plot()

        # Update TGM
        tgm.update(gm, x_t)

        # Plot maps
        tgm.plotCombinedMap()


if __name__ == '__main__':
    run()