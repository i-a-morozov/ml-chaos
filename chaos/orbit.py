import numpy
import numba


def orbit_factory(mapping, *, fastmath=True, parallel=False):
    """ Generates a callable for one trajectory computation """
    @numba.jit('float64[:, :](int64, float64[:], float64[:])', nopython=True, fastmath=fastmath, parallel=parallel)
    def orbit(n, k, x):
        xs = numpy.zeros((n + 1, len(x)))
        xs[0] = x
        for i in range(1, n + 1):
                xs[i] = x = mapping(k, x)
        return xs
    return orbit


def table_factory(mapping, *, fastmath=True, parallel=True):
    """ Generates a callable for several trajectories computation (parallel over initial values) """
    orbit = orbit_factory(mapping, fastmath=True, parallel=False)
    @numba.jit('float64[:, :, :](int64, float64[:], float64[:, :])', nopython=True, fastmath=fastmath, parallel=parallel)
    def table(n, k, x):
        l, d = x.shape
        xs = numpy.zeros((l, n + 1, d))
        for i in numba.prange(len(x)):
            xs[i] = orbit(n, k, x[i])
        return xs
    return table