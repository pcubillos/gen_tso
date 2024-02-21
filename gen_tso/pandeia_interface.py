from dataclasses import dataclass

import numpy as np

from pandeia.engine.calc_utils import (
#    build_default_calc,
    get_instrument_config,
)
#from pandeia.engine.perform_calculation import perform_calculation
#from pandeia.engine.etc3D import setup
#import pandeia.engine.sed as sed
#from pandeia.engine.instrument_factory import InstrumentFactory


#@dataclass(frozen=True, order=True)
@dataclass(order=True)
class Detector:
    name: str
    label: str
    instrument: str
    obs_type: str
    grism_title: str
    grisms: list
    filter_title: str
    filters: list
    subarrays: list
    readout: list


def generate_all_instruments():
    telescope = 'jwst'
    instrument = 'miri'
    ins_config = get_instrument_config(telescope, instrument)
    config = ins_config['mode_config']['lrsslitless']

    dispersers = [disperser.upper() for disperser in config['dispersers']]
    subarrays = [subarray.upper() for subarray in config['subarrays']]
    readout = [read.upper() for read in config['readout_patterns']]

    lrs = Detector(
        'lrsslitless',
        'Low Resolution Spectroscopy (LRS)',
        'MIRI',
        'spectroscopy',
        'Disperser',
        dispersers,
        '',
        [''],
        subarrays,
        readout,
    )


    instrument = 'nircam'
    ins_config = get_instrument_config(telescope, instrument)
    config = ins_config['mode_config']['ssgrism']

    filters = [filter.upper() for filter in config['filters']]
    subarrays = [subarray.upper() for subarray in config['subarrays']]
    readout = [read.upper() for read in config['readout_patterns']]

    nircam_grism = Detector(
        'ssgrism',
        'LW Grism Time Series',
        'NIRCam',
        'spectroscopy',
        'Grism',
        ['GRISMR'],
        'Filter',
        filters,
        subarrays,
        readout,
    )


    instrument = 'niriss'
    ins_config = get_instrument_config(telescope, instrument)
    config = ins_config['mode_config']['soss']

    filters = [filter.upper() for filter in config['filters']]
    subarrays = [subarray.upper() for subarray in config['subarrays']]
    readout = ['NIS', 'NISRAPID']

    soss = Detector(
        'soss',
        'Single Object Slitless Spectroscopy (SOSS)',
        'NIRISS',
        'spectroscopy',
        'Disperser',
        ['GR700XD (cross-dispersed)'],
        'Filter',
        filters,
        subarrays,
        readout,
    )


    instrument = 'nirspec'
    ins_config = get_instrument_config(telescope, instrument)
    config = ins_config['mode_config']['bots']
    subarrays = [subarray.upper() for subarray in config['subarrays']]
    readout = [read.upper() for read in config['readout_patterns']]

    grating_filter = []
    gratings = ins_config['config_constraints']['dispersers']
    for grating, filters in gratings.items():
        for filter in filters['filters']:
            grating_filter.append(f'{grating}/{filter}'.upper())

    bots = Detector(
        'bots',
        'Bright Object Time Series (BOTS)',
        'NIRSpec',
        'spectroscopy',
        'Grating/Filter',
        grating_filter,
        'Slit',
        ['S1600A1 (1.6" x 1.6")'],
        subarrays,
        readout,
    )

    bots.wl_ranges = {
        'G140M/F070LP': (0.70, 1.27),
        'G140M/F100LP': (0.97, 1.84),
        'G235M/F170LP': (1.66, 3.07),
        'G395M/F290LP': (2.87, 5.10),
        'G140H/F070LP': (0.81, 1.27),
        'G140H/F100LP': (0.97, 1.82),
        'G235H/F170LP': (1.66, 3.05),
        'G395H/F290LP': (2.87, 5.14),
        'PRISM/CLEAR': (0.60, 5.30),
    }


    detectors = [
        lrs,
        nircam_grism,
        soss,
        bots,
    ]

    return detectors

