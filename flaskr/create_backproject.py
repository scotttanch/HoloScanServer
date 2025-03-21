import bisect
import sys
from itertools import islice
from more_itertools import divide
import matplotlib.pyplot as plt
from readgssi.dzt import readdzt
from time import monotonic
import numpy as np
from numpy import sqrt
from mpi4py import MPI
from path_tools import reduce_resolution, interpolate_domain

C = 2.99792458 * 10 ** 8


def batched(iterable, n, *, strict=False):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError('batched(): incomplete batch')
        yield batch


def lerp(x, x0, fx0, x1, fx1):
    """
    Linearly interpolates a value between two others

    Args:
        x (float | int): point of interpolation/extrapolation
        x0 (float | int): lower bound of interpolation
        fx0 (float | int): function value at the lower bound
        x1 (float | int): upper bound of interpolation
        fx1 (float | int): function value at the upper bound

    Notes:
        f(x) = f(x0) + (f(x1)-f(x0))/(x1-x0)*(x-x0)

    Returns:

    """
    if x0 == x1:
        return fx0
    else:
        fx = fx0 + (fx1 - fx0)/(x1 - x0)*(x - x0)
        return fx


def read_raw_path(file):
    xs = []
    ys = []
    zs = []
    with open(file, "r+") as f:
        lines = f.readlines()
    for line in lines:
        x, y, z = line.split(',')
        xs.append(float(x))
        ys.append(float(y))
        zs.append(float(z))
    return xs, ys, zs


def get_positions(xs, ys, zs, num_traces, spm):
    """
    Given some path defined by (xs, ys, zs), find a number of antenna positions along that line spaced per the provided scans/meter (spm)

    Args:
        xs (list[float]): x-cordinates of the scan path
        ys (list[float]): y-cordinates of the scan path
        zs (list[float]): z-cordinates of the scan path
        num_traces (int): number of traces collected along the scan
        spm (flaot): spacing between traces in meters

    Returns:

    """
    spm = 1/spm
    # Assuming the first scan is at the first position
    _xks = [xs[0]]
    _yks = [ys[0]]
    _zks = [zs[0]]

    # generate the length along the curve of each antenna position
    distances = [num*spm for num in range(1, num_traces)]

    # generate the legnth along the curve of each point in oringal set
    org_distances = [0.0]
    for i in range(1, len(xs)):
        d = sqrt((xs[i] - xs[i-1])**2 + (ys[i] - ys[i-1])**2 + (zs[i] - zs[i-1])**2)
        org_distances.append(org_distances[-1] + d)

    for distance in distances:
        # for each distance we want to find the two points that it is between
        lower_bound = bisect.bisect_left(org_distances, distance) - 1
        upper_bound = lower_bound + 1

        # Make sure that we dont try to index out of bounds
        if upper_bound >= len(org_distances):
            upper_bound = len(org_distances) - 1
            lower_bound = upper_bound - 1

        _xks.append(lerp(distance, org_distances[lower_bound], xs[lower_bound], org_distances[upper_bound], xs[upper_bound]))
        _yks.append(lerp(distance, org_distances[lower_bound], ys[lower_bound], org_distances[upper_bound], ys[upper_bound]))
        _zks.append(lerp(distance, org_distances[lower_bound], zs[lower_bound], org_distances[upper_bound], zs[upper_bound]))

    return _xks, _yks, _zks


def travel_time(pk, p0, eps_r):
    """
    Computes the two-way travel time between points pk and p0 through some medium with permitivity eps_r

    Args:
        pk (tuple[float, float, float]): Postion of Antenna at an instance t
        p0 (tuple[float, float, float]): Position of pixel p0 in the projection surface
        eps_r (float): relative permitivity of the soil

    Returns:

    """
    global C

    tau = (2 * sqrt((p0[0] - pk[0])**2 + (p0[1] - pk[1])**2 + (p0[2] - pk[2])**2)) / (C / sqrt(eps_r))
    return tau


def traces_lookup(trace, t, t_max, correction):
    """
    Finds the value of a trace at time t

    Args:
        trace (list[float]):
        t (float):
        t_max (float):
        correction (float):

    Returns:

    """
    if t >= t_max:
        return 0.0

    trace_length = len(trace)
    sample_num = int(((t+correction)/t_max) * trace_length)

    if sample_num >= trace_length:
        return 0.0
    else:
        return trace[sample_num]


def background_removal(scan):
    """
    Performs an SVD background removal of a B-Scan

    Args:
        scan (np.ndarray):

    Returns:

    """

    u, s, vh = np.linalg.svd(scan, full_matrices=False)
    s[0] = 0
    scan = np.dot(u * s, vh)

    return scan

# TODO: Update this to check for a property override file in the scan folder
def override_exists():
    return True


def main():

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:

        start = monotonic()
        dzt_file = (sys.argv[1])
        csv_file = (sys.argv[2])
        img_file = (sys.argv[3])
        resolution = float(sys.argv[4])

        header, data, _ = readdzt(dzt_file)

        scan = data[0]

        _ps = read_raw_path(csv_file)
        ps = reduce_resolution(*_ps, resolution=0.05, mean=False, endpoint=True)

        # 0. Get Information about the scan from the header and scan itself

        num_samples, num_traces = np.shape(scan)

        if override_exists():
            epsr = 10
        else:
            epsr = header['rhf_epsr']

        t_max = header['rhf_range'] * 10 ** -9
        depth = header['rhf_depth']

        # 1. Obtain a set of antenna positions along the scan line
        xks, yks, zks = get_positions(*ps, num_traces=num_traces, spm=header['rhf_spm'])

        # 2. Prepare the Image domain
        # 2.1 Interpolate the path to the desired image resolution
        domain_x, domain_y, domain_z = interpolate_domain(*ps, resolution=resolution)

        # 2.2 Find the number of points in the depth direction
        z_size = int(depth / resolution)

        # 2.3 for each point on the surface, construct a column vector extending from the surface to the maximum depth
        cols = []
        for index in range(len(domain_x)):
            surface = domain_z[index]
            subsurface = surface - depth
            z_range = np.linspace(surface, subsurface, z_size, endpoint=True)
            x_range = [domain_x[index] for _ in z_range]
            y_range = [domain_y[index] for _ in z_range]
            col = np.array(list(zip(x_range, y_range, z_range)), dtype="f,f,f")
            cols.append(col)

        # 2.4 Stack the column vectors together to create the image domain, Transposed so that the xy cordinate for each column is the same
        domain = np.vstack(cols)
        domain = domain.T

        # 2.5 Create the image in the same shape as the cordinate domain
        image = np.zeros_like(domain, dtype=float)

        # 3 Process Traces
        # 3.1 find the time zero correction as the time of the first negative peak
        avg_trace = np.mean(scan, axis=1)
        threshold = np.min(avg_trace)
        index = [i for i, v in enumerate(avg_trace) if v <= threshold][0]
        correction = t_max / num_samples * index

        # 3.2 Perform an svd background removal
        bg_scan = background_removal(scan)

        # 3.3 Extract individual traces from the total scan and zip together with their cordinates
        traces = [bg_scan[:, num] for num in range(num_traces)]
        trace_loc = list(zip(traces, zip(xks, yks, zks)))
        batches = [list(c) for c in divide(size, trace_loc)]

    else:
        domain = None
        image = None
        batches = None
        epsr = None
        t_max = None
        correction = None

    domain = comm.bcast(domain, root=0)
    image = comm.bcast(image, root=0)
    epsr = comm.bcast(epsr, root=0)
    batches = comm.bcast(batches, root=0)
    t_max = comm.bcast(t_max, root=0)
    correction = comm.bcast(correction, root=0)

    domain_shape = np.shape(domain)
    batch = batches[rank]
    for i in range(domain_shape[0]):
        for j in range(domain_shape[1]):
            x0, y0, z0 = domain[i][j]
            for trace, pk in batch:
                time = travel_time(pk, (x0, y0, z0), epsr)
                value = traces_lookup(trace, time, t_max, correction)
                image[i][j] = image[i][j] + value

    image = comm.gather(image, root=0)

    if rank == 0:
        final = np.zeros_like(image[0])
        # sum the cells of the image element wise
        for each in image:
            final = np.add(final, each)

        # normalize image
        max_val = np.max(final, axis=None)
        min_val = np.min(final, axis=None)
        final = final + abs(min_val)
        final = np.divide(final, (max_val - min_val))
        
        # TODO: Make this work like the reduced texture
        # Instead of normalizing, make final = abs(final) so the top and bottom of targets gets blurred into one cell?
        
        final = np.hstack((np.fliplr(final), final))
        plt.imsave(img_file, final)
        
        elapsed = (monotonic()-start)/60
        #print(f"{size},{resolution},{elapsed}")

    return


if __name__ == "__main__":
    main()
