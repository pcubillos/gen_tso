# Copyright (c) 2024 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

import json
from dataclasses import dataclass
import os

import numpy as np
import requests

from pandeia.engine.calc_utils import (
#    build_default_calc,
    get_instrument_config,
)
#from pandeia.engine.perform_calculation import perform_calculation
#from pandeia.engine.etc3D import setup
import pandeia.engine.sed as sed
#from pandeia.engine.instrument_factory import InstrumentFactory
import pandeia.engine

from synphot.config import conf, Conf


def check_pandeia_version():
    # TBD: see what's going on here
    pandeia.engine.pandeia_version()


def fetch_vega():
    # TBD: check synphot path exists

    vega = Conf.vega_file
    url = vega.defaultvalue
    query_parameters = {}
    response = requests.get(url, params=query_parameters)
    if not response.ok:
        print('Could not download Vega reference spectrum')
        # show url, download manually?, put it in path

    path = os.path.dirname(conf.vega_file)
    if not os.path.exists(path):
        os.mkdir(path)
    with open(conf.vega_file, mode="wb") as file:
        file.write(response.content)


def exposure_time(
        instrument, calculation=None,
        nexp=None, nint=None, ngroup=None, readout=None, subarray=None,
    ):
    """
    Based on pandeia.engine.exposure.
    nircam full is not giving the right numbers, all else OK.
    """
    if isinstance(instrument, str):
        telescope = 'jwst'
        ins_config = get_instrument_config(telescope, instrument)
    else:
        ins_config = instrument.ins_config

    tfffr  = ins_config['subarray_config']['default'][subarray]['tfffr']
    tframe = ins_config['subarray_config']['default'][subarray]['tframe']
    nframe = ins_config['readout_pattern_config'][readout]['nframe']
    ndrop2 = ins_config['readout_pattern_config'][readout]['ndrop2']
    ndrop1 = ndrop3 = 0
    nreset1 = nreset2 = 1
    if 'nreset1' in ins_config['readout_pattern_config'][readout]:
        nreset1 = ins_config['readout_pattern_config'][readout]['nreset1']
    if 'nreset2' in ins_config['readout_pattern_config'][readout]:
        nreset2 = ins_config['readout_pattern_config'][readout]['nreset2']

    exposure_time = nexp * (
        tfffr * nint +
        tframe * (
            nreset1 + (nint-1) * nreset2 +
            nint * (ndrop1 + (ngroup-1) * (nframe + ndrop2) + nframe + ndrop3)
        )
    )
    return exposure_time


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
    lrs.ins_config = get_instrument_config(telescope, instrument)

    #mrs_ts = Detector(
    #    'mrs_ts',
    #    'Medium Resolution Spectroscopy (MRS) time series',
    #    'MIRI',
    #    'spectroscopy',
    #)

    #imaging_ts = Detector(
    #    'imaging_ts',
    #    'Imaging time series',
    #    'MIRI',
    #    'photometry',
    #)

    #miri_ta = Detector(
    #    'target_acq',
    #    'Target Acquisition',
    #    'MIRI',
    #    'acquisition',
    #)


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
    nircam_grism.ins_config = get_instrument_config(telescope, instrument)

    #nircam_grism = Detector(
    #    'lw_ts',
    #    'long wavelength time-series imaging',
    #    'NIRCam',
    #    'photometry',
    #)

    #nircam_grism = Detector(
    #    'sw_ts',
    #    'short wavelength time-series imaging',
    #    'NIRCam',
    #    'photometry',
    #)

    #nircam_ta = Detector(
    #    'target_acq',
    #    'Target Acquisition',
    #    'NIRCam',
    #    'acquisition',
    #)


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
    soss.ins_config = get_instrument_config(telescope, instrument)

    #niriss_ta = Detector(
    #    'target_acq',
    #    'Target Acquisition',
    #    'NIRISS',
    #    'acquisition',
    #)


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
    bots.ins_config = get_instrument_config(telescope, instrument)

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

    #nirspec_ta = Detector(
    #    'target_acq',
    #    'Target Acquisition',
    #    'NIRSpec',
    #    'acquisition',
    #)


    detectors = [
        lrs,
        #mrs,
        #miri_imaging,
        #miri_ta,
        nircam_grism,
        #nircam_sw,
        #nircam_lw,
        #nircam_ta,
        soss,
        #niriss_ta,
        bots,
        #nirspec_ta,
    ]

    return detectors



def load_sed_list(source):
    """
    Load list of available PHOENIX or Kurucz stellar SED models

    Parameters
    ----------
    source: String
        SED source: 'phoenix' or 'k93models'
    """
    sed_path = sed.default_refdata_directory
    with open(f'{sed_path}/sed/{source}/spectra.json', 'r') as f:
        info = json.load(f)
    teff = np.array([model['teff'] for model in info.values()])
    log_g = np.array([model['log_g'] for model in info.values()])
    names = np.array([model['display_string'] for model in info.values()])
    tsort = np.argsort(teff)[::-1]
    return names[tsort], teff[tsort], log_g[tsort]


def find_closest_sed(m_teff, m_logg, teff, logg):
    """
    A vert simple cost-function to find the closest stellar model within
    a non-regular Teff-log_g grid.
    """
    cost = (
        np.abs(np.log10(teff/m_teff)) +
        np.abs(logg-m_logg) / 15.0
    )
    idx = np.argmin(cost)
    return idx



