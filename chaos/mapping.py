"""
Collection of symplectic mappings

Naming
------

[name]_[dimension]_[kind]

[name]      -- mapping name, e.g. polynomial
[dimension] -- phase space dimension, e.g. 2d
[kind]      -- can be forward, inverse or tangent, the latter if fact returns jacobian at a point

Signature
---------

k -- numpy array with mapping parameters (knobs)
x -- numpy array with initial condition

@numba.jit('float64[:](float64[:], float64[:])', nopython=True, fastmath=False, parallel=False)
def [name]_[dimension]_[kind](k, x):
    ...
    return x

"""
import numpy
import numba


@numba.jit('float64[:](float64[:], float64[:])', nopython=True, fastmath=False, parallel=False)
def polynomial_2d_forward(k, x):
    """ Forward polynomial mapping """
    w, s = k
    q, p = x
    q, p = p, -q + w*p + (1.0 - s)*p**2 + s*p**3
    return numpy.array([q, p])


@numba.jit('float64[:](float64[:], float64[:])', nopython=True, fastmath=False, parallel=False)
def polynomial_2d_inverse(k, x):
    """ Inverse polynomial mapping """
    w, s = k
    q, p = x
    q, p = -p + w*q + (1.0 - s)*q**2 + s*q**3, q
    return numpy.array([q, p])


@numba.jit('float64[:, :](float64[:], float64[:])', nopython=True, fastmath=False, parallel=False)
def polynomial_2d_jacobian(k, x):
    """ Jacobian for forward polynomial mapping """
    w, s = k
    q, p = x
    return numpy.array([[0.0, 1.0], [-1.0, w + 2.0*(1.0 - s)*p + 3.0*s*p**2]])