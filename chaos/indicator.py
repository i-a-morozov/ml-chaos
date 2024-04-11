import numpy
import numba


def survival_factory(mapping, *, fastmath=False, parallel=True):
    """ Generates survival indicator """
    @numba.jit('float64[:](int64, float64[:], float64[:, :])', nopython=True, fastmath=fastmath, parallel=parallel)
    def survival(n, k, xs):
        out = numpy.zeros(len(xs))
        for i in numba.prange(len(xs)):
            x = xs[i]
            for _ in range(n):
                x = mapping(k, x)
            out[i] = numpy.linalg.norm(x)
        return out
    return survival


def rem_factory(forward, inverse, *, level=1.0E-15, perturbation=0.0, fastmath=False, parallel=True):
    """ Generates REM indicator """
    @numba.jit('float64[:](int64, float64[:], float64[:, :])', nopython=True, fastmath=fastmath, parallel=parallel)
    def rem(n, k, xs):
        out = numpy.zeros(len(xs))
        for i in numba.prange(len(xs)):
            x = xs[i]
            X = x
            for _ in range(n):
                X = forward(k, X)
            X = X + perturbation
            for _ in range(n):
                X = inverse(k, X)
            out[i] = numpy.log10(level + numpy.linalg.norm(x - X))
        return out
    return rem


@numba.jit('float64[:](int64, float64)', nopython=True, fastmath=False, parallel=False)
def window(n, s):
    """ Analytical filter """
    t = numpy.linspace(0.0, (n - 1.0)/n, n)
    f = numpy.exp(-1.0/((1.0 - t)**s*t**s))
    return f/numpy.sum(f)


@numba.jit('float64(float64[:], float64[:, :])', nopython=True, fastmath=False)
def frequency(f, xs):
    """ Frequency estimation """
    qs, ps = xs
    return numpy.ascontiguousarray(f) @ (numpy.diff(numpy.arctan2(qs, ps)) % (2.0*numpy.pi))/(2.0*numpy.pi)


def fma_factory(orbit, *, level=1.0E-15, fastmath=False, parallel=True):
    """ Generates FMA indicator """
    @numba.jit('float64[:](float64[:], float64[:], float64[:, :])', nopython=True, fastmath=fastmath, parallel=parallel)
    def fma(f, k, xs):
        n = len(f)
        out = numpy.zeros(len(xs))
        for i in numba.prange(len(xs)):
            x = orbit(n, k, xs[i])
            X = x[-1]
            x = numpy.ascontiguousarray(x.T).reshape(-1, 2, n + 1)
            a = numpy.zeros(len(x))
            for j in range(len(x)):
                a[j] = frequency(f, x[j])
            x = orbit(n, k, X)
            x = numpy.ascontiguousarray(x.T).reshape(-1, 2, n + 1)
            b = numpy.zeros(len(x))
            for j in range(len(x)):
                b[j] = frequency(f, x[j])
            out[i] = numpy.log10(level + numpy.linalg.norm(a - b))
        return out
    return fma


def tangent_factory(mapping, jacobian, fastmath=False, parallel=False):
    """ Generates a tangent mapping callable with (normalized) dynamics of deviation vectors """
    @numba.jit('Tuple((float64[:], float64[:, :]))(float64[:], float64[:], float64[:, :])', nopython=True, fastmath=fastmath, parallel=parallel)
    def tangent(k, x, v):
        x = mapping(k, x)
        v = numpy.ascontiguousarray(v)
        m = numpy.ascontiguousarray(jacobian(k, x))
        for i in range(len(v)):
            v[i] = m @ v[i]
            v[i] = v[i]/numpy.linalg.norm(v[i])
        return x, v
    return tangent


@numba.jit('float64(float64[:, :])', nopython=True, fastmath=False)
def product(v):
    """ Returns product of SVD values """
    _, s, _ = numpy.linalg.svd(v, full_matrices=False)
    return numpy.prod(s)


def gali_factory(tangent, *, level=1.0E-15, fastmath=False, parallel=True):
    """ Generates FMA indicator """
    @numba.jit('float64[:](int64, float64[:], float64[:, :], float64[:, :, :])', nopython=True, fastmath=fastmath, parallel=parallel)
    def gali(n, k, xs, vs):
        out = numpy.zeros(len(xs))
        Xs = numpy.copy(xs)
        Vs = numpy.copy(vs)
        for i in numba.prange(len(xs)):
            x = Xs[i]
            v = Vs[i]
            for _ in range(n):
                x, v = tangent(k, x, v)
            out[i] = numpy.nan if numpy.isnan(x.prod() * v.prod()) else numpy.log10(level + product(v))
        return out
    return gali
