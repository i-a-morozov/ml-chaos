import numpy
import numba


def orbit_factory(mapping):
    """ Generates a callable for one trajectory computation """
    @numba.jit('float64[:, :](int64, float64[:], float64[:])', nopython=True, fastmath=True, parallel=False)
    def orbit(n, k, x):
        xs = numpy.zeros((n + 1, len(x)))
        xs[0] = x
        for i in range(1, n + 1):
                xs[i] = x = mapping(k, x)
        return xs
    return orbit


def table_factory(mapping):
    """ Generates a callable for several trajectories computation (parallel over initial values) """
    orbit = orbit_factory(mapping)
    @numba.jit('float64[:, :, :](int64, float64[:], float64[:, :])', nopython=True, fastmath=True, parallel=True)
    def table(n, k, x):
        xs = numpy.zeros((len(x), n + 1, len(x)))
        for i in numba.prange(len(x)):
            xs[i] = orbit(n, k, x[i])
        return xs
    return table