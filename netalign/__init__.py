__version__ = "0.1.1"
__citation__ = """
Devkota, K., Cowen, L. J., Blumer, A., & Hu, X. (2023). Fast Approximate IsoRank for Scalable Global Alignment of Biological Networks. bioRxiv, 2023-03.
"""

from . import (
    approx_isorank,
    duomundo
)

__all__ = [
    "approx_isorank",
    "duomundo"
]