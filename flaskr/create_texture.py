import numpy as np
import numpy.ma as ma

# Global Values (Maybe Can be played with via some web interface)
smoothing_window = 5        # Window to smooth trace over (samples)
dewow_window = 10           # Window to dewow trace over (samples)
bg_window = 2.5             # Window to average scan over (meters)
gain_vals = [-2, 9, 12]     # Linear Gain Profile (dB)
threshold = 0.60            # Minimum value to retain opacity (unitless)
red_val = 1                 # Red channel value [0:1] (unitless)
blu_val = 0                 # Blue channel value [0:1] (unitless)
grn_val = 0                 # Green channel value [0:1] (unitless)


# Functions Taken from GPRPy (gprpyTools.py) Need to figure out how to cite these a little better
# I've made stylistic changes but the functions are the same
def dewow(data: np.matrix, window: int) -> np.matrix:
    """
    Subtracts from each sample along each trace an
    along-time moving average.

    Can be used as a low-cut filter.

    Args:
        data: data matrix whose columns contain the traces
        window: length of moving average window in samples

    Returns:
        newdata: data matrix after dewow

    """
    totsamps = data.shape[0]
    # If the window is larger or equal to the number of samples,
    # then we can do a much faster dewow
    if window >= totsamps:
        newdata = data - np.matrix.mean(data, 0)
    else:
        newdata = np.asmatrix(np.zeros(data.shape))
        halfwid = int(np.ceil(window / 2.0))

        # For the first few samples, it will always be the same
        avgsmp = np.matrix.mean(data[0:halfwid + 1, :], 0)
        newdata[0:halfwid + 1, :] = data[0:halfwid + 1, :] - avgsmp

        # for each sample in the middle
        for smp in range(halfwid, totsamps - halfwid + 1):
            winstart = int(smp - halfwid)
            winend = int(smp + halfwid)
            avgsmp = np.matrix.mean(data[winstart:winend + 1, :], 0)
            newdata[smp, :] = data[smp, :] - avgsmp

        # For the last few samples, it will always be the same
        avgsmp = np.matrix.mean(data[totsamps - halfwid:totsamps + 1, :], 0)
        newdata[totsamps - halfwid:totsamps + 1, :] = data[totsamps - halfwid:totsamps + 1, :] - avgsmp

    return newdata


def smooth(data: np.matrix, window: int) -> np.matrix:
    """
    Replaces each sample along each trace with an
    along-time moving average.

    Can be used as high-cut filter.

    Args:
        data: data matrix whose columns contain the traces
        window: length of moving average window in samples

    Returns:
        newdata: data matrix after applying smoothing
    """

    totsamps = data.shape[0]
    # If the window is larger or equal to the number of samples,
    # then we can do a much faster dewow
    if window >= totsamps:
        newdata = np.matrix.mean(data, 0)
    elif window == 1:
        newdata = data
    elif window == 0:
        newdata = data
    else:
        newdata = np.asmatrix(np.zeros(data.shape))
        halfwid = int(np.ceil(window / 2.0))

        # For the first few samples, it will always be the same
        newdata[0:halfwid + 1, :] = np.matrix.mean(data[0:halfwid + 1, :], 0)

        # for each sample in the middle
        for smp in range(halfwid, totsamps - halfwid + 1):
            winstart = int(smp - halfwid)
            winend = int(smp + halfwid)
            newdata[smp, :] = np.matrix.mean(data[winstart:winend + 1, :], 0)

        # For the last few samples, it will always be the same
        newdata[totsamps - halfwid:totsamps + 1, :] = np.matrix.mean(data[totsamps - halfwid:totsamps + 1, :], 0)

    return newdata


def rem_mean_trace(data: np.matrix, ntraces: int) -> np.matrix:
    """
    Subtracts from each trace the average trace over
    a moving average window.

    Can be used to remove horizontal arrivals,
    such as the airwave.

    Args:
        data: data matrix whose columns contain the traces
        ntraces: window width; over how many traces to take the moving average.

    Returns:
        newdata: data matrix after subtracting average traces
    """

    data = np.asmatrix(data)
    tottraces = data.shape[1]
    # For ridiculous ntraces values, just remove the entire average
    if ntraces >= tottraces:
        newdata = data - np.matrix.mean(data, 1)
    else:
        newdata = np.asmatrix(np.zeros(data.shape))
        halfwid = int(np.ceil(ntraces / 2.0))

        # First few traces, that all have the same average
        avgtr = np.matrix.mean(data[:, 0:halfwid + 1], 1)
        newdata[:, 0:halfwid + 1] = data[:, 0:halfwid + 1] - avgtr

        # For each trace in the middle
        for tr in range(halfwid, tottraces - halfwid + 1):
            winstart = int(tr - halfwid)
            winend = int(tr + halfwid)
            avgtr = np.matrix.mean(data[:, winstart:winend + 1], 1)
            newdata[:, tr] = data[:, tr] - avgtr

        # Last few traces again have the same average
        avgtr = np.matrix.mean(data[:, tottraces - halfwid:tottraces + 1], 1)
        newdata[:, tottraces - halfwid:tottraces + 1] = data[:, tottraces - halfwid:tottraces + 1] - avgtr

    return newdata


# Functions Written By Me
def get_gain(gain_points: list[float | int], samples: int, scans: int) -> np.ndarray:
    """
    Assemble a 2-Dimensional Gain Array to be applied to a B-Scan. Represents a linear range gain being applied to each
    trace in the B-Scan.
    Args:
        gain_points: Gain values in dB. Points are distributed evenly along the sample window
        samples: Number of samples in each trace
        scans: Number of traces in the Scan

    Returns:
        gain_array:

    """
    x = np.linspace(0, samples, samples, endpoint=True)
    xp = np.linspace(0, samples, len(gain_points), endpoint=True)
    fp = gain_points
    y = np.interp(x, xp, fp)

    gain_profile = list(map(lambda p: 10 ** (p/10), y))
    gain_array = np.stack([gain_profile for _ in range(scans)], axis=1)
    return gain_array


def process_dzt(radar_header: dict, radar_data: list[np.ndarray[float]]) -> np.ndarray:
    """
    Perform basic processing of a B-scan saved as a dzt file. Performs dewow and smoothing, followed by application of gain,
    and finally background subtraction.
    Args:
        radar_header: Header dictionary returned by the readgssi.dzt.readdzt function
        radar_data: Radar data returned by readgssi.dzt.readdzt function

    Returns:
        processed:

    """
    global smoothing_window, dewow_window, gain_vals, bg_window
    # 1.1 Read In
    channel_0 = np.asmatrix(radar_data[0])

    # 1.2 DeWow and Smoth
    filtered = dewow(smooth(channel_0, smoothing_window), dewow_window)

    # 1.3 Apply Gain
    gain_array = np.asmatrix(get_gain(gain_vals, channel_0.shape[0], channel_0.shape[1]))
    gained = np.multiply(filtered, gain_array)

    # 1.4 Mean Background Removal with a window of 2.5 meters
    background_removed = rem_mean_trace(gained, radar_header['rhf_spm'] * bg_window)

    proccessed = np.asarray(background_removed)
    return proccessed


def create_st(data: np.ndarray) -> np.ndarray:
    """
    Creates the two-sided “Standard Texture” for a HoloScan from a processed or raw B-Scan.
    Args:
        data: 2-Dimensional array representing a B-Scan

    Returns:
        img:

    """
    flipped = np.fliplr(data)
    img = np.hstack((flipped, data))

    return img


def create_rtt(data: np.ndarray) -> np.ndarray:
    """
    Creates the two-sided “Reduced Transparent Texture” for a Holoscan from a processed or raw B-Scan
    Args:
        data: 2-Dimensional array representing a B-Scan

    Returns:
        final:

    """
    global threshold, red_val, bg_window, grn_val

    # Start off by normalizing the scan
    norm_data = (data - np.min(data)) / (np.max(data) - np.min(data))

    # Create a masked array where values less than the threshold are invalid
    masked: ma.masked_array = ma.masked_less(norm_data, threshold)

    # Use the masked array to create a ones like, basically setting all valid values to one
    ones = np.ones_like(masked)

    # Create a filled array from the ones, where invalid values are replaced with 0
    binary_image = ma.filled(ones, 0)

    # Assemble and apply color masks to the binary image
    r_channel = binary_image * np.full_like(binary_image, red_val)
    b_channel = binary_image * np.full_like(binary_image, blu_val)
    g_channel = binary_image * np.full_like(binary_image, grn_val)

    # Stack together the color channels with the binary channel representing opacity to create an RGBA image
    rgba_im = np.dstack([r_channel, b_channel, g_channel, binary_image])

    # flip and stack the RGBA image with its original
    final = np.hstack((np.fliplr(rgba_im), rgba_im))

    return final


def create_textures(radar_header: dict, radar_data: list[np.ndarray[float]]) -> tuple[np.ndarray, np.ndarray]:
    """
    Create the standard and reduced texture provided from .dzt data
    Args:
        radar_header: Header dictionary returned by the readgssi.dzt.readdzt function
        radar_data: Radar data returned by readgssi.dzt.readdzt function

    Returns:
        (st, rtt): a tuple containing the standard texture in the first position, and the reduced in the second

    """
    # Process DZT
    data = process_dzt(radar_header, radar_data)

    # Create Standard Texture
    st = create_st(data)

    # Create Reduced Transparent Texture
    rtt = create_rtt(data)

    return st, rtt


def test():
    # TODO: Given some set of DZT files make sure none of them cause a crash for some reason
    pass

# Potentially useful later
# def bandpass(lowedge: float, highedge: float):
#     # fs is the sample frequenc
#     # assuming that the lowedge and high edge are comming in Hz, need to convert to rads/s
#
#     # start of the cutoff region freq, should be between the stopband edges
#     passbands = [lowedge+0.1, highedge-0.1]
#     # cut off frequencies
#     stopbands = [lowedge, highedge]
#     gain_stop = 60
#     gain_pass = 0.25
#     sos = signal.iirdesign(passbands, stopbands, gain_pass, gain_stop, ftype='butter', analog=False, output='sos', )
#     return sos

# Texture Creation Steps
# Step 1: Reading and Basic Processing
    # 1.1 Read in DZT file
    # 1.2 Perform some kind of bandpass filtering
    # 1.3 Apply some gain function, either linear or an adaptive gain from GPRPy
    # 1.4 Background Removal, either mean, svd, or windowed mean from GPRPy
    # 1.5 Return the processed array
# Step 2: Create the standard Texture
    # 2.1 Read in array
    # 2.2 Flip LR
    # 2.3 hstack flipped, orignal
    # 2.4 return standard texture
# Step 3: Create Reduced Transparent Texture
    # 3.1 Read in array
    # 3.2 Normalize array
    # 3.3 Create a masked array, where values below a threshold are invalid
        # Instead of thresholding maybe I could do the areas where the value changes quickly? More like edges?
        # Worth looking into but im not sure that gets me anything
    # 3.4 Create a masked ones like from the masked array
    # 3.5 Create a binary array by filling the masked ones with zeros
    # 3.6 Multiply the binary array by the appropriate color values (purely aestetic)
    # 3.7 Depth stack the 3 color channels with the Alpha channel
    # 3.8 Flip LR
    # 3.9 hstack flipped, orig
    # 3.10 return the reduced transparent texture (rtt)
# Step 4: Return textures, to be saved by the create resource wrapper
