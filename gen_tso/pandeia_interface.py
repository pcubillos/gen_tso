# Copyright (c) 2024 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

import json
from dataclasses import dataclass
import os

import numpy as np
import requests

import pandeia.engine
from pandeia.engine.calc_utils import (
    build_default_calc,
    get_instrument_config,
)
from pandeia.engine.perform_calculation import perform_calculation
import pandeia.engine.sed as sed
from pandeia.engine.normalization import NormalizationFactory


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

    # When switching instrument, subarray and readout updates are not atomic
    if subarray not in ins_config['subarray_config']['default']:
        return 0.0
    if readout not in ins_config['readout_pattern_config']:
        return 0.0

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


def saturation_time(instrument, ngroup=None, readout=None, subarray=None):
    """
    Compute JWST's saturation time for the given instrument configuration.
    Based on pandeia.engine.exposure.
    """
    if isinstance(instrument, str):
        telescope = 'jwst'
        ins_config = get_instrument_config(telescope, instrument)
    else:
        ins_config = instrument.ins_config

    # When switching instrument, subarray and readout updates are not atomic
    if subarray not in ins_config['subarray_config']['default']:
        return 0.0
    if readout not in ins_config['readout_pattern_config']:
        return 0.0

    tframe = ins_config['subarray_config']['default'][subarray]['tframe']
    nframe = ins_config['readout_pattern_config'][readout]['nframe']
    ndrop2 = ins_config['readout_pattern_config'][readout]['ndrop2']
    ndrop1 = 0

    saturation_time = tframe * (
        ndrop1 + (ngroup - 1) * (nframe + ndrop2) + nframe
    )
    return saturation_time



def make_scene(sed_type, sed_model, norm_band, norm_magnitude):
    sed = {'sed_type': sed_type}

    if sed_type == 'flat':
        sed['unit'] = sed_model
    elif sed_type in ['phoenix', 'k93models']:
        sed['key'] = sed_model
    elif sed_type == 'blackbody':
        sed['temp'] = sed_model

    normalization = {
        'type': 'photsys',
        'bandpass': norm_band,
        'norm_flux': norm_magnitude,
        'norm_fluxunit': 'vegamag',
    }

    spectrum = {
        'sed': sed,
        'normalization': normalization,
        'extinction': {
            'bandpass': 'j',
            'law': 'mw_rv_31',
            'unit': 'mag',
            'value': 0,
        },
        'lines': [],
        'redshift': 0,
    }

    scene = {
        'spectrum': spectrum,
        'position': {'orientation': 0.0, 'x_offset': 0.0, 'y_offset': 0.0},
        'shape': {'geometry': 'point'},
    }
    return scene


def extract_sed(scene):
    """
    Get the flux (mJ) and wavelength array from a given scene dict.

    Returns
    -------
    wave: 1D float array
    flux: 1D float array

    Examples
    --------
    >>> import pandeia_interface as jwst

    >>> sed_type = 'phoenix'
    >>> sed_model = 'k5v'
    >>> norm_band = '2mass,ks'
    >>> norm_magnitude = 8.637
    >>> scene1 = jwst.make_scene(sed_type, sed_model, norm_band, norm_magnitude)
    >>> wl1, phoenix = jwst.extract_sed(scene1)

    >>> sed_type = 'blackbody'
    >>> sed_model = 4250.0
    >>> scene2 = jwst.make_scene(sed_type, sed_model, norm_band, norm_magnitude)
    >>> wl2, bb = jwst.extract_sed(scene2)

    >>> plt.figure(0)
    >>> plt.clf()
    >>> plt.plot(wl1, phoenix, c='b')
    >>> plt.plot(wl2, bb, c='xkcd:green')
    >>> plt.xlim(0.5, 12)

    """
    normalization = NormalizationFactory(
        config=scene['spectrum']['normalization'], webapp=True,
    )
    sed_model = sed.SEDFactory(config=scene['spectrum']['sed'], webapp=True, z=0)
    wave, flux = normalization.normalize(sed_model.wave, sed_model.flux)
    return wave.value, flux.value


class Calculation():
    def __init__(self, instrument, mode):
        self.telescope = 'jwst'
        self.instrument = instrument
        self.mode = mode
        self.calc = build_default_calc(self.telescope, self.instrument, self.mode)

    def get_configs(self, output=None):
        ins_config = get_instrument_config(self.telescope, self.instrument)
        config = ins_config['mode_config'][self.mode]

        subarrays = config['subarrays']
        screen_output = f'subarrays: {subarrays}\n'

        if self.instrument == 'niriss':
            readouts = ins_config['readout_patterns']
        else:
            readouts = config['readout_patterns']
        screen_output += f'readout patterns: {readouts}\n'

        if self.instrument == 'nirspec':
            gratings_dict = ins_config['config_constraints']['dispersers']
            gratings = filters = dispersers = []
            for grating, filter_list in gratings_dict.items():
                for filter in filter_list['filters']:
                    gratings.append(f'{grating}/{filter}')
            screen_output += f'grating/filter pairs: {gratings}'
        else:
            filters = config['filters']
            dispersers = [disperser for disperser in config['dispersers']]
            screen_output += f'dispersers: {dispersers}\n'
            screen_output += f'filters: {filters}'

        if output is None:
            print(screen_output)
        elif output == 'readouts':
            return readouts
        elif output == 'subarrays':
            return subarrays
        elif output == 'filters':
            return filters
        elif output == 'dispersers':
            return dispersers
        elif self.instrument=='nirspec' and output=='gratings':
            return gratings
        else:
            raise ValueError(f"Invalid config output: '{output}'")


    def set_scene(self, sed_type, sed_model, norm_band, norm_magnitude):
        scene = make_scene(sed_type, sed_model, norm_band, norm_magnitude)
        self.calc['scene'] = [scene]

    def perform_calculation(
        self, nint, ngroup,
        filter=None, readout=None, subarray=None, disperser=None,
    ):
        if readout is not None:
            self.calc['configuration']['detector']['readout_pattern'] = readout
        if subarray is not None:
            self.calc['configuration']['detector']['subarray'] = subarray

        if self.instrument == 'nircam':
            self.calc['configuration']['instrument']['disperser'] = 'grismr'
            self.calc['configuration']['instrument']['filter'] = filter
        elif self.instrument == 'nirspec':
            self.calc['configuration']['instrument']['disperser'] = disperser
            self.calc['configuration']['instrument']['filter'] = filter
        elif self.instrument == 'niriss':
            self.calc['configuration']['instrument']['filter'] = filter
            self.calc['strategy']['order'] = 1
            # DataError: No mask configured for SOSS order 2.
        elif self.instrument == 'miri':
            pass

        self.calc['configuration']['detector']['nexp'] = 1 # dither
        self.calc['configuration']['detector']['nint'] = nint
        self.calc['configuration']['detector']['ngroup'] = ngroup

        self.report = perform_calculation(self.calc)
        return self.report


    def get_saturation_values(self, filter, readout, subarray, disperser):
        """
        Calculate the brightest pixel rate (e-/s) and full_well (e-)
        for the current instrument and scene configuration, which once known,
        are sufficient to calculate the saturation level once the
        saturation  time is known.

        Examples
        --------
        >>> import pandeia_interface as jwst

        >>> pando = jwst.Calculation(inst_name, mode)
        >>> pando.set_scene(
        >>>     sed_type='phoenix', sed_model='k2v',
        >>>     norm_band='2mass,ks', norm_magnitude=7.459,
        >>> )
        >>> brightest_pixel_rate, full_well = pando.get_saturation_values(
        >>>     disperser='grismr', filter='f444w',
        >>>     readout='rapid', subarray='subgrism64',
        >>> )
        """
        ngroup = 2
        saturation_run = self.perform_calculation(
            nint=1, ngroup=ngroup,
            readout=readout, subarray=subarray,
            disperser=disperser,
            filter=filter,
        )
        pando_results = saturation_run['scalar']
        brightest_pixel_rate = pando_results['brightest_pixel']
        #sat_fraction = pando_results['fraction_saturation'] / ngroup

        full_well = (
            brightest_pixel_rate
            * pando_results['saturation_time']
            / pando_results['fraction_saturation']
        )
        return brightest_pixel_rate, full_well


# This is the front-end
@dataclass(order=True)
class Detector:
    name: str
    label: str
    instrument: str
    obs_type: str
    disperser_title: str
    dispersers: list
    filter_title: str
    filters: list
    subarrays: list
    readouts: list
    disperser_default: str
    filter_default: str
    subarray_default: str
    readout_default: str


def generate_all_instruments():
    telescope = 'jwst'
    # Spectroscopy
    # 'mrs_ts': 'MRS Time Series',
    # Imaging
    # 'imaging_ts': 'Imaging Time Series',
    # 'sw_ts': 'SW Time Series',
    # 'lw_ts': 'LW Time Series',

    instrument = 'miri'
    mode = 'lrsslitless'
    cal = Calculation(instrument, mode)
    dispersers = [d.upper() for d in cal.get_configs('dispersers')]
    filters = ['']
    subarrays = [s.upper() for s in cal.get_configs('subarrays')]
    readouts = [r.upper() for r in cal.get_configs('readouts')]

    lrs = Detector(
        mode,
        'Low Resolution Spectroscopy (LRS)',
        'MIRI',
        'spectroscopy',
        'Disperser',
        dispersers,
        '',
        [''],
        subarrays,
        readouts,
        disperser_default=dispersers[0],
        filter_default=filters[0],
        subarray_default=subarrays[0],
        readout_default=readouts[0],
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
    mode = 'ssgrism'
    cal = Calculation(instrument, mode)
    dispersers = [d.upper() for d in cal.get_configs('dispersers')]
    filters = [f.upper() for f in cal.get_configs('filters')]
    subarrays = [s.upper() for s in cal.get_configs('subarrays')]
    readouts = [r.upper() for r in cal.get_configs('readouts')]

    nircam_grism = Detector(
        mode,
        'LW Grism Time Series',
        'NIRCam',
        'spectroscopy',
        'Grism',
        dispersers,
        'Filter',
        filters,
        subarrays,
        readouts,
        disperser_default=dispersers[0],
        filter_default=filters[3],
        subarray_default=subarrays[3],
        readout_default=readouts[0],
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
    mode = 'soss'
    cal = Calculation(instrument, mode)
    dispersers = [d.upper() for d in cal.get_configs('dispersers')]
    #dispersers = ['GR700XD (cross-dispersed)']
    filters = [f.upper() for f in cal.get_configs('filters')]
    subarrays = [s.upper() for s in cal.get_configs('subarrays')]
    readouts = [r.upper() for r in cal.get_configs('readouts')]
    readouts.reverse()

    soss = Detector(
        mode,
        'Single Object Slitless Spectroscopy (SOSS)',
        'NIRISS',
        'spectroscopy',
        'Disperser',
        dispersers,
        'Filter',
        filters,
        subarrays,
        readouts,
        disperser_default=dispersers[0],
        filter_default=filters[0],
        subarray_default=subarrays[0],
        readout_default=readouts[0],
    )
    soss.ins_config = get_instrument_config(telescope, instrument)

    #niriss_ta = Detector(
    #    'target_acq',
    #    'Target Acquisition',
    #    'NIRISS',
    #    'acquisition',
    #)


    instrument = 'nirspec'
    mode = 'bots'
    cal = Calculation(instrument, mode)
    dispersers = ['S1600A1 (1.6" x 1.6")']
    filters = [f.upper() for f in cal.get_configs('filters')]
    subarrays = [s.upper() for s in cal.get_configs('subarrays')]
    subarrays.reverse()
    readouts = [r.upper() for r in cal.get_configs('readouts')]
    readouts.reverse()

    bots = Detector(
        mode,
        'Bright Object Time Series (BOTS)',
        'NIRSpec',
        'spectroscopy',
        'Slit',
        dispersers,
        'Grating/Filter',
        filters,
        subarrays,
        readouts,
        disperser_default=dispersers[0],
        filter_default=filters[6],
        subarray_default=subarrays[0],
        readout_default=readouts[0],
    )
    bots.ins_config = get_instrument_config(telescope, instrument)

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
    keys = np.array(list(info.keys()))
    tsort = np.argsort(teff)[::-1]
    return keys[tsort], names[tsort], teff[tsort], log_g[tsort]


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



