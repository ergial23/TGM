import numpy as np
from gridMap import gridMap
import time

class sensorModel:
    def __init__ (self, origin, width, height, resolution, range, invModel ,occPrior):
        self.origin = origin
        self.width = width
        self.height = height
        self.resolution = resolution
        self.range = range
        self.invModel = invModel
        self.occPrior = occPrior

    def generateGridMap(self, z_t, x_t):
        """
        Generate raycasted grid map from 2D sensor data
        """

        # I am missing to populate the data
        """
        data = np.zeros((self.width*self.resolution, self.height*self.resolution))
        data[0][0] = 1
        data[19][0] = 0.5
        return gridMap(self.origin, self.width, self.height, self.resolution, data)
        """

        ang, dist = z_t[:,0], z_t[:,1]
        ang = ang + x_t[2]
        dist[dist>self.range] = self.range
        ox = x_t[0] + np.cos(ang) * dist
        oy = x_t[1] + np.sin(ang) * dist
        ix_t = ((x_t[0:2]-self.origin) * self.resolution).astype(int)
        data = np.ones((self.width*self.resolution, self.height*self.resolution)) * (1-self.occPrior)
        for (x, y, d) in zip(ox, oy, dist):
            # x coordinate of the detection in grid frame
            ix = int(round((x - self.origin[0]) * self.resolution))
            # y coordinate of the detection in grid frame
            iy = int(round((y - self.origin[1]) * self.resolution))
            laser_beams = bresenham((ix_t[0], ix_t[1]), (ix, iy))  # line form the lidar to the occupied point
            for laser_beam in laser_beams:
                data[laser_beam[0]][laser_beam[1]] = self.invModel[0]  # free area
            if d<self.range:
                data[ix][iy] = self.invModel[1]  # occupied area 1.0
                #data[ix + 1][iy] = 1.0  # extend the occupied area
                #data[ix][iy + 1] = 1.0  # extend the occupied area
                #data[ix + 1][iy + 1] = 1.0  # extend the occupied area
        return gridMap(self.origin, self.width, self.height, self.resolution, data)

def bresenham(start, end):
    # setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    is_steep = abs(dy) > abs(dx)  # determine how steep the line is
    if is_steep:  # rotate line
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    # swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
    dx = x2 - x1  # recalculate differentials
    dy = y2 - y1  # recalculate differentials
    error = int(dx / 2.0)  # calculate error
    y_step = 1 if y1 < y2 else -1
    # iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = [y, x] if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += y_step
            error += dx
    if swapped:  # reverse the list if the coordinates were swapped
        points.reverse()
    points = np.array(points)
    return points
    
def file_lidar(f):
    """
    Reading LIDAR laser beams (angles and corresponding distance data)
    """
    with open(f) as data:
        measures = [line.split(",") for line in data]
    #angles = []
    #distances = []
    #for measure in measures:
    #    angles.append(float(measure[0]))
    #    distances.append(float(measure[1]))
    #angles = np.array(angles)
    #distances = np.array(distances)
    #return angles, distances
    z_t = np.array(measures)
    return z_t

def main():
    origin = [0,0]
    width = 150
    height = 50
    resolution = 2
    range = 50
    invModel = [0.1, 0.9]
    occPrior = 0.5
    sM = sensorModel(origin, width, height, resolution, range, invModel ,occPrior)

    #z_t = file_lidar("../logs/sim_corridor/z_1.csv")
    with open("../logs/sim_corridor/z_100.csv") as data:
        z_t = np.array([line.split(",") for line in data]).astype(float)
    
    with open("../logs/sim_corridor/x_100.csv") as data:
        x_t = np.array([line.split(",") for line in data]).astype(float)[0]

    #x_t = np.array((10,10,0))
    #z_t = np.array([[0,20]])
    print(x_t)
    import time
    start = time.time()
    gm = sM.generateGridMap(z_t, x_t)
    print(time.time() - start)
    gm.plot()
    

if __name__ == '__main__':
    main()