import numpy

from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
from matplotlib import colormaps


def threshold(grid, *,
              interval:tuple[float, float]=(-16.0, 1.0),
              count:int=251,
              kernel:str='gaussian',
              bandwidth:float=0.2,
              plot:bool=False,
              output:str=None):
    """ KDE based threshold computation """
    indicator = numpy.sort(grid.flatten())
    indicator = indicator[~numpy.isnan(indicator)].reshape(-1, 1)
    points = numpy.linspace(*interval, count)
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(indicator)
    score = kde.score_samples(points.reshape(-1, 1))
    peaks, info = find_peaks(score, height=(None, None))
    *_, chaotic, regular = peaks[numpy.exp(info['peak_heights']).argsort()]
    position = numpy.exp(score[regular:chaotic]).argmin()
    threshold = points[regular + position]
    if plot:
        _, (ax, ay) = plt.subplots(nrows=2, ncols=1, figsize=(12, 2*3))
        ax.plot(indicator, color='blue', alpha=0.75)
        ax.axhline(threshold, linestyle='dashed', color='black', alpha=0.75)
        ax.axhline(points[regular], linestyle='dashed', color='blue', alpha=0.75)
        ax.axhline(points[chaotic], linestyle='dashed', color='red', alpha=0.75)
        ax.set_ylim(*interval)
        ay.hist(indicator, bins=100, range=interval, density=True, color='blue', alpha=0.75)
        ay.plot(points, numpy.exp(score), color='black', linestyle='dashed', alpha=0.75)
        ay.errorbar(points[regular], numpy.exp(score[regular]), ms=5, marker='o', color='blue', alpha=0.75)
        ay.errorbar(points[chaotic], numpy.exp(score[chaotic]), ms=5, marker='o', color='red', alpha=0.75)
        ay.axvline(threshold, linestyle='dashed', color='black', alpha=0.75)
        ay.set_xlim(*interval)
        ay.set_ylim(0.0, 1.0)
        plt.tight_layout()
        if output:
            plt.savefig(output, dpi=200)
        plt.show()
    return threshold


def classify(grid, data, threshold:float, *,
             regular=0.0,
             chaotic=1.0,
             mask:bool=True,
             plot:bool=False,
             xmin:float=-1.0,
             xmax:float=+1.0,
             ymin:float=-1.0,
             ymax:float=+1.0,
             output:str=None):
    """ Thrshold classification """
    table = numpy.copy(grid)
    table = numpy.where(table >= threshold, regular, chaotic)
    table[numpy.isnan(grid)] = float('nan')
    bounded = ~numpy.isnan(table.flatten())
    X = data[bounded]
    y = table.flatten()[bounded]
    if plot:
        cmap = colormaps.get_cmap('gray')
        cmap.set_bad(color='gray')
        plt.figure(figsize=(8, 8))
        plt.gca().set_facecolor('gray')
        plt.scatter(*X.T, c=y, cmap=cmap, s=0.01)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.tight_layout()
        if output:
            plt.savefig(output, dpi=200)
        plt.show()
    return ((X, y), bounded.reshape(grid.shape)) if mask else (X, y)
