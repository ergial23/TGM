import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from lidarScan import lidarScan

def lsqnl_matching(scan, lsq_map, x0, max_range):
    # Remove the no-return scans from scan
    lsq_scan = scan.removeNoReturn(max_range)
    x = least_squares(lsq_fun, x0, max_nfev=500, args=(lsq_scan, lsq_map))
    return x

def lsq_fun(relPose, lsq_scan, lsq_map):
    limit_x = lsq_map.width
    limit_y = lsq_map.height
    cell_length = 1 / lsq_map.resolution
    #print(limit_x, limit_y, cell_length)

    # Remove the no-return scans from lsq_scan
    #scan = np.column_stack([lsq_scan.rangles, lsq_scan.angles])
    #scan = scan[scan[:, 0] < lsq_max_range]

    #lsq_scan = lidarScan(scan[:, 0], scan[:, 1])

    # Transform the scan according to relPose
    transCart = lsq_scan.computeRelativeCartesian(relPose)
    #transScan = transformScan(lsq_scan, relPose)
    #transScan = [scan[0] + relPose[0], scan[1] + relPose[1], scan[2] + relPose[2]]

    # Smooth cost function
    #cost = 1 - interp2d(np.array((np.arange(cell_length / 2, limit_x - cell_length / 2, cell_length),
    #                    np.arange(cell_length / 2, limit_y - cell_length / 2, cell_length))),
    #                    np.flip(lsq_map), transCart[:, 0], transCart[:, 1],
    #                    kind='cubic', fill_value=0)

    # Smooth cost function
    #x = np.arange(cell_length / 2, limit_x - cell_length / 2, cell_length)
    #print(x)
    #y = np.arange(cell_length / 2, limit_y - cell_length / 2, cell_length)
    x = np.linspace(cell_length / 2, limit_x - cell_length / 2, lsq_map.data.shape[0])
    y = np.linspace(cell_length / 2, limit_y - cell_length / 2, lsq_map.data.shape[1])
    #print(x.shape, y.shape, lsq_map.shape)
    interp = RegularGridInterpolator((x, y), lsq_map.data, bounds_error=False, method='cubic', fill_value=0)

    cost = 1 - interp(transCart, method='cubic')

    # Plot the cost:
    '''
    X, Y = np.meshgrid(x, y)
    Xq, Yq = np.meshgrid(np.arange(cell_length / 2, limit_x - cell_length / 2, cell_length / 2),
                          np.arange(cell_length / 2, limit_y - cell_length / 2, cell_length / 2))
    Vq = 1 - interp((Xq, Yq), method='cubic')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Xq, Yq, Vq, edgecolor='none')
    ax.plot(transCart[:, 0], transCart[:, 1], cost, 'r.')
    plt.show()
    '''

    return cost