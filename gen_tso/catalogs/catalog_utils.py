# Copyright (c) 2024 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

"""
A collection of low-level routines to handle catalogs.
"""

__all__ = [
    'is_letter',
    'is_candidate',
    'to_float',
]

import pickle
import re

from astropy.io import ascii
import numpy as np


def is_letter(name):
    """
    Check if name ends with a blank + lower-case letter (it's a planet)
    """
    return name[-1].islower() and name[-2] == ' '


def is_candidate(name):
    """
    Check if name ends with a blank + lower-case letter (it's a planet)
    """
    return name[-3] == '.' and name[-2:].isnumeric()


def to_float(value):
    """
    Cast string to None or float type.
    """
    if value == 'None':
        return None
    return float(value)


