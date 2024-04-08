"""
Aliases

"""

from chaos.mapping import polynomial_2d_forward as forward
from chaos.mapping import polynomial_2d_inverse as inverse
from chaos.mapping import polynomial_2d_jacobian as jacobian

from chaos.orbit import orbit_factory
from chaos.orbit import table_factory

from chaos.indicator import survival_factory
from chaos.indicator import rem_factory
from chaos.indicator import window
from chaos.indicator import fma_factory
from chaos.indicator import tangent_factory
from chaos.indicator import gali_factory

from chaos.utils import threshold
from chaos.utils import classify

orbit = orbit_factory(forward)
table = table_factory(forward)
tangent = tangent_factory(forward, jacobian)

survival = survival_factory(forward)
rem = rem_factory(forward, inverse)
fma = fma_factory(orbit)
gali = gali_factory(tangent)
