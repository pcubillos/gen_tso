# Copyright (c) 2024 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    'read_noise_variance',
    'exposure_time',
    'bin_search_exposure_time',
    'saturation_time',
    'saturation_level',
    'load_sed_list',
    'find_closest_sed',
    'extract_sed',
    'make_scene',
    'set_depth_scene',
    'simulate_tso',
    '_print_pandeia_report',
    '_print_pandeia_exposure',
    '_print_pandeia_saturation',
    'print_pandeia_report',
    'PandeiaCalculation',
]

from collections.abc import Iterable
import copy
from itertools import product
import json
import os
import random

import numpy as np
import requests
import scipy.interpolate as si
import pandeia.engine.sed as sed
from pandeia.engine.calc_utils import (
    build_default_calc,
    get_instrument_config,
)
from pandeia.engine.perform_calculation import perform_calculation
from pandeia.engine.normalization import NormalizationFactory
from synphot.config import conf, Conf

from ..utils import constant_resolution_spectrum, format_text


def check_pandeia_version():
    has_refdata = "pandeia_refdata" in os.environ
    has_synphot = "PYSYN_CDBS" in os.environ
    if not has_refdata:
        print('Unset reference data environment variable ("pandeia_refdata")')
    if not has_synphot:
        print('Unset synphot environment variable ("PYSYN_CDBS")')
    return


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


def read_noise_variance(report, ins_config):
    """
    Extract the read noise from instrument configuration data.

    Parameters
    ----------
    report: dict
        A pandeia perform_calculation() output.
    ins_config: dict
        A pandeia get_instrument_config() output for a given instrument.

    Returns
    -------
    read_noise: Float
        The instrumental read noise (electrons per pixel?)
    """
    report_config = report['input']['configuration']['instrument']
    if report_config['mode'] == 'mrs_ts':
        aperture = report_config['aperture']
        aperture = ins_config['aperture_config'][aperture]['detector']
    else:
        #aperture = self.calc['configuration']['instrument']['aperture']
        aperture = report_config['aperture']

    if aperture not in ins_config['detector_config']:
        aperture = 'default'

    read_noise = ins_config['detector_config'][aperture]['rn']
    if isinstance(read_noise, dict):
        read_noise = read_noise['default']

    return read_noise


def _exposure_time_function(instrument, subarray, readout, ngroup, nexp=1):
    """
    Return a callable that evaluates the exposure time as a function
    of nint.

    Parameters
    ----------
    instrument: String
        Which instruments (miri, nircam, niriss, or nirspec).
    subarray: String
        Subarray mode for the given instrument.
    readout: String
        Readout pattern mode for the given instrument.
    ngroup: Integeer
        Number of groups per integration.  Must be >= 2.
    nexp: Integer
        Number of exposures.

    Returns
    -------
    exp_time: Float
        Exposure time in seconds.

    """
    if isinstance(instrument, str):
        telescope = 'jwst'
        ins_config = get_instrument_config(telescope, instrument)
    else:
        ins_config = instrument.ins_config

    # When switching instrument, subarray and readout updates are not atomic
    if subarray not in ins_config['subarray_config']['default']:
        return None
    if readout not in ins_config['readout_pattern_config']:
        return None

    subarray_config = ins_config['subarray_config']['default'][subarray]
    readout_config = ins_config['readout_pattern_config'][readout]
    tfffr = subarray_config['tfffr']
    tframe = subarray_config['tframe']
    nframe = readout_config['nframe']
    ndrop2 = readout_config['ndrop2']
    ndrop1 = ndrop3 = 0
    nreset1 = nreset2 = 1
    if 'nreset1' in readout_config:
        nreset1 = readout_config['nreset1']
    elif 'nreset1' in subarray_config:
        nreset1 = subarray_config['nreset1']
    if 'nreset2' in readout_config:
        nreset2 = readout_config['nreset2']

    def exp_time(nint):
        time = nexp * (
            tfffr * nint +
            tframe * (
                nreset1 + (nint-1) * nreset2 +
                nint * (ndrop1 + (ngroup-1) * (nframe + ndrop2) + nframe + ndrop3)
            )
        )
        return time
    return exp_time


def exposure_time(instrument, subarray, readout, ngroup, nint, nexp=1):
    """
    Calculate the exposure time for the given instrumental setup.
    Based on pandeia.engine.exposure.

    Parameters
    ----------
    instrument: String
        Which instruments (miri, nircam, niriss, or nirspec).
    subarray: String
        Subarray mode for the given instrument.
    readout: String
        Readout pattern mode for the given instrument.
    ngroup: Integeer
        Number of groups per integration.  Must be >= 2.
    nint: Integer
        Number of integrations.
    nexp: Integer
        Number of exposures.

    Returns
    -------
    exp_time: Float
        Exposure time in seconds.

    Examples
    --------
    >>> import gen_tso.pandeia_io as jwst

    >>> inst = 'nircam'
    >>> subarray = 'subgrism64'
    >>> readout = 'rapid'
    >>> nint = 1
    >>> ngroup = 90
    >>> exp_time = jwst.exposure_time(inst, subarray, readout, ngroup, nint)
    >>> print(exp_time)
    """
    exp_time = _exposure_time_function(
        instrument, subarray, readout, ngroup, nexp,
    )
    if exp_time is None:
        return 0.0
    return exp_time(nint)


def bin_search_exposure_time(
        instrument, subarray, readout, ngroup, obs_time, nexp=1,
    ):
    """
    Binary search for nint such that exp_time(nint) > obs_time

    Parameters
    ----------
    instrument: String
        Which instruments (miri, nircam, niriss, or nirspec).
    subarray: String
        Subarray mode for the given instrument.
    readout: String
        Readout pattern mode for the given instrument.
    ngroup: Integeer
        Number of groups per integration.  Must be >= 2.
    obs_time: Integer
        Total observation time to aim for (in hours).
    nexp: Integer
        Number of exposures.

    Returns
    -------
    nint: Integer
        Number of integrations to reach obs_time.
    exp_time: Float
        Exposure time in seconds.

    Examples
    --------
    >>> import gen_tso.pandeia_io as jwst

    >>> instrument = 'miri'
    >>> subarray = 'slitlessprism'
    >>> readout = 'fastr1'
    >>> ngroup = 30
    >>> obs_time = 6.0
    >>> nint, exp_time = jwst.bin_search_exposure_time(
    >>>     instrument, subarray, readout, ngroup, obs_time,
    >>> )
    >>> print(nint, exp_time)
    """
    exp_time = _exposure_time_function(
        instrument, subarray, readout, ngroup, nexp,
    )
    if exp_time is None:
        return 0, 0.0

    obs_time_sec = 3600 * obs_time
    n1 = 1
    t1 = exp_time(n1)
    if obs_time_sec < t1:
        return n1, t1

    n2 = int(obs_time_sec / t1)
    t2 = exp_time(n2)
    while t2 < obs_time_sec:
        n1 = n2
        n2 = n1*2
        t2 = exp_time(n2)
    while n2-n1 > 1:
        n = int(0.5*(n1+n2))
        t = exp_time(n)
        if obs_time_sec > t:
            n1 = n
        else:
            n2 = n

    return n2, exp_time(n2)


def saturation_time(instrument, ngroup, readout, subarray):
    """
    Compute JWST's saturation time for the given instrument configuration.
    Based on pandeia.engine.exposure.

    Examples
    --------
    >>> import gen_tso.pandeia_io as jwst

    >>> # Calculate integration time for a given instrument setup:
    >>> inst = 'nircam'
    >>> ngroup = 90
    >>> readout = 'rapid'
    >>> subarray = 'subgrism64'
    >>> integ_time = jwst.saturation_time(inst, ngroup, readout, subarray)
    >>> print(integ_time)
    30.6549
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


def saturation_level(reports, get_max=False):
    """
    Compute saturation values for a given perform_calculation output.

    Parameters
    ----------
    reports: Dictionary or list of dictionaries
        One or more pandeia's perform_calculation() output dictionary
        or a tso_calculation output dictionary.
        If there is more than one input report, return arrays of
        saturation values for each input.
    get_max: Bool
        If True and there is  more than one input report, return the
        saturation values for the report that's quickest to saturate.

    Returns
    -------
    brightest_pixel_rate: Float
        e- per second rate at the brightest pixel.
    full_well: Float
        Number of e- counts to saturate the detector.

    Examples
    --------
    >>> import gen_tso.pandeia_io as jwst

    >>> inst = 'nircam'
    >>> readout = 'rapid'
    >>> pando = jwst.PandeiaCalculation(inst, 'ssgrism')
    >>> pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)
    >>> result = pando.perform_calculation(
    >>>     ngroup=2, nint=683, readout='rapid', filter='f444w',
    >>> )
    >>> pixel_rate, full_well = jwst.saturation_level(result)

    >>> # Now I can calculate the saturation level for any integration time:
    >>> # (for the given filter and scene)
    >>> subarray = 'subgrism64'
    >>> for ngroup in [2, 97, 122]:
    >>>     integ_time = jwst.saturation_time(inst, ngroup, readout, subarray)
    >>>     sat_level = pixel_rate * integ_time / full_well * 100
    >>>     print(f'Sat. fraction for {ngroup:3d} groups: {sat_level:5.1f}%')
    Sat. fraction for   2 groups:   1.6%
    Sat. fraction for  97 groups:  79.4%
    Sat. fraction for 122 groups:  99.9%
    """
    if not isinstance(reports, list):
        reports = [reports]
    # Unpack TSO dictionary
    if 'report_in' in reports[0]:
        reports = (
            [report['report_in'] for report in reports] +
            [report['report_out'] for report in reports]
        )

    ncalc = len(reports)
    brightest_pixel_rate = np.zeros(ncalc)
    full_well = np.zeros(ncalc)
    for i,report in enumerate(reports):
        brightest_pixel_rate[i] = report['scalar']['brightest_pixel']
        full_well[i] = (
            brightest_pixel_rate[i]
            * report['scalar']['saturation_time']
            / report['scalar']['fraction_saturation']
        )

    if len(reports) == 1:
        return brightest_pixel_rate[0], full_well[0]
    if get_max:
        idx = np.argmax(brightest_pixel_rate*full_well)
        return brightest_pixel_rate[idx], full_well[idx]
    return brightest_pixel_rate, full_well


def load_sed_list(source):
    """
    Load list of available PHOENIX or Kurucz stellar SED models

    Parameters
    ----------
    source: String
        SED source: 'phoenix' or 'k93models'

    Returns
    -------
    keys: list of strings
        SED model keys (as used in a Pandeia scene dict).
    names: list of strings
        SED model names (as seen on the ETC).
    teff: list of floats
        SED model effective-temperature grid.
    log_g: list of floats
        SED model log(g) grid.

    Examples
    --------
    >>> import gen_tso.pandeia as jwst
    >>> # PHOENIX models
    >>> keys, names, teff, log_g = jwst.load_sed_list('phoenix')
    >>> # Kurucz models
    >>> keys, names, teff, log_g = jwst.load_sed_list('k93models')
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


def find_closest_sed(models_teff, models_logg, teff, logg):
    """
    A very simple cost-function to find the closest stellar model
    within a non-regular Teff-log_g grid.

    Since these are not regular grids, the cost function is not an
    absolute science, it depends on what weights more Teff of logg
    for a given case.  The current formula seems to be a good balance.

    Parameters
    ----------
    models_teff: list of floats
        SED model effective-temperature grid.
    models_logg: list of floats
        SED model log(g) grid.
    teff: float
        Target effective temperature.
    logg: float
        Target log(g).

    Returns
    -------
    idx: integer
        index of model with the closest Teff and logg.

    Examples
    --------
    >>> import gen_tso.pandeia as jwst
    >>> # PHOENIX models
    >>> keys, names, p_teff, p_logg = jwst.load_sed_list('phoenix')
    >>> idx = jwst.find_closest_sed(p_teff, p_logg, teff=4143.0, logg=4.66)
    >>> print(f'{keys[idx]}: {repr(names[idx])}')
    k5v: 'K5V 4250K log(g)=4.5'
    """
    cost = (
        np.abs(np.log10(teff/models_teff)) +
        np.abs(logg-models_logg) / 15.0
    )
    idx = np.argmin(cost)
    return idx


def make_scene(sed_type, sed_model, norm_band=None, norm_magnitude=None):
    """
    Create a stellar point-source scene dictionary for use in Pandeia.

    Parameters
    ----------
    sed_type: String
        Type of model: 'phoenix', 'k93models', 'blackbody', or 'flat'
    sed_model:
        The SED model required for each sed_type:
        - phoenix or k93models: the model key (see load_sed_list)
        - blackbody: the effective temperature (K)
        - flat: the unit ('flam' or 'fnu')
    norm_band: String
        Band over which to normalize the spectrum.
    norm_magnitude: float
        Magnitude of the star at norm_band.

    Examples
    --------
    >>> import gen_tso.pandeia_io as jwst

    >>> sed_type = 'phoenix'
    >>> sed_model = 'k5v'
    >>> norm_band = '2mass,ks'
    >>> norm_magnitude = 8.637
    >>> scene = jwst.make_scene(sed_type, sed_model, norm_band, norm_magnitude)
    >>> print(scene)
    {'spectrum': {'sed': {'sed_type': 'phoenix', 'key': 'k5v'},
      'normalization': {'type': 'photsys',
       'bandpass': '2mass,ks',
       'norm_flux': 8.637,
       'norm_fluxunit': 'vegamag'},
      'extinction': {'bandpass': 'j',
       'law': 'mw_rv_31',
       'unit': 'mag',
       'value': 0},
      'lines': [],
      'redshift': 0},
     'position': {'orientation': 0.0, 'x_offset': 0.0, 'y_offset': 0.0},
     'shape': {'geometry': 'point'}}

    >>> sed_type = 'blackbody'
    >>> sed_model = 4250.0
    >>> norm_band = '2mass,ks'
    >>> norm_magnitude = 8.637
    >>> scene = jwst.make_scene(sed_type, sed_model, norm_band, norm_magnitude)
    """
    sed = {'sed_type': sed_type}

    if sed_type == 'flat':
        sed['unit'] = sed_model
    elif sed_type in ['phoenix', 'k93models']:
        sed['key'] = sed_model
    elif sed_type == 'blackbody':
        sed['temp'] = sed_model

    if norm_band is None or norm_band == 'none':
        normalization = {'type': 'none'}
    else:
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


def extract_sed(scene, wl_range=None):
    """
    Extract the flux spectrum array from a given scene dict.

    Parameters
    ----------
    scene: dict
        A pandeia scene dict

    Returns
    -------
    wl: 1D float array
        Scene's wavelength array (um).
    flux: 1D float array
        Scene's flux spectrum (mJy).

    Examples
    --------
    >>> import gen_tso.pandeia_io as jwst
    >>> import matplotlib.pyplot as plt

    >>> sed_type = 'phoenix'
    >>> sed_model = 'k5v'
    >>> norm_band = '2mass,ks'
    >>> norm_magnitude = 8.637
    >>> scene1 = jwst.make_scene(sed_type, sed_model, norm_band, norm_magnitude)
    >>> wl1, phoenix = jwst.extract_sed(scene1)

    >>> sed_type = 'k93models'
    >>> sed_model = 'k7v'
    >>> scene2 = jwst.make_scene(sed_type, sed_model, norm_band, norm_magnitude)
    >>> wl2, kurucz = jwst.extract_sed(scene2)

    >>> sed_type = 'blackbody'
    >>> sed_model = 4250.0
    >>> scene3 = jwst.make_scene(sed_type, sed_model, norm_band, norm_magnitude)
    >>> wl3, bb = jwst.extract_sed(scene3)

    >>> plt.figure(0)
    >>> plt.clf()
    >>> plt.plot(wl1, phoenix, c='b')
    >>> plt.plot(wl2, kurucz, c='darkorange')
    >>> plt.plot(wl3, bb, c='xkcd:green')
    >>> plt.xlim(0.5, 12)
    """
    sed_model = sed.SEDFactory(config=scene['spectrum']['sed'], webapp=True, z=0)
    normalization = NormalizationFactory(
        config=scene['spectrum']['normalization'],
        webapp=True,
    )
    wave, flux = normalization.normalize(sed_model.wave, sed_model.flux)
    if normalization.type == 'none':
        if sed_model.sed_type in ['phoenix', 'k93models']:
            # Convert wavelengths from A to um:
            wave *= 1e-4  # pc.A / pc.um
        if sed_model.sed_type == 'blackbody':
            # Convert blackbody intensity to flux:
            flux *= np.pi

    if wl_range is None:
        wl_mask = np.ones(len(wave.value), bool)
    else:
        wl_mask = (wave.value >= wl_range[0]) & (wave.value <= wl_range[1])
    return wave.value[wl_mask], flux.value[wl_mask]


def set_depth_scene(scene, obs_type, depth_model, wl_range=None):
    """
    Given a stellar point-source scene and an observing geometry,
    generate a new scene with the source flux scaled by a transit depth
        flux_in_transit = flux * (1-depth),
    or scaled by an eclipse depth:
        flux_out_of_eclipse = flux * (1+depth).

    To combine stellar and depth spectra, the lower-resolution
    spectrum will be linearly-interpolated to the sampling of the
    higher-resolution spectrum.

    Also return a scene of the un-scaled stellar source (with the same
    sampling as in the scaled scene, which prevents fringing when
    operating on the spectra produced from these scenes).
    Both output scenes are already normalized (so that a pandeia
    calculation does not undo the eclipse/transit depth scaling).

    Parameters
    ----------
    scene: A pandeia scene dictionary
        An input scene of a stellar point source.
    obs_type: string
        The observing geometry 'transit' or 'eclipse'.
    depth_model: list of two 1D array or a 2D array
        The transit or eclipse depth spectrum where the first item is
        the wavelength (um) and the second is the depth.
    wl_range: float iterable
        A wl_min, wl_max pair where to clip the stellar and depth spectra.

    Examples
    --------
    >>> import gen_tso.pandeia as jwst

    >>> # A pandeia stellar point-source scene:
    >>> scene = jwst.make_scene(
    >>>     sed_type='k93models', sed_model='k7v',
    >>>      norm_band='2mass,ks', norm_magnitude=8.637,
    >>> )
    >>>
    >>> # A transit-depth spectrum:
    >>> depth_model = np.loadtxt(
    >>>     '../planet_spectra/WASP80b_transit.dat', unpack=True)
    >>>
    >>> obs_type = 'transit'
    >>> star_scene, in_transit_scene = jwst.set_depth_scene(
    >>>     scene, obs_type, depth_model,
    >>> )
    """
    # from pandeia_interface import extract_sed
    # Get stellar flux (mJy) from scene, planet depth spectrum from input
    wl_star, star_flux = extract_sed(scene)
    wl_planet, depth = depth_model

    if wl_range is not None:
        wl_mask = (wl_star > wl_range[0]) & (wl_star < wl_range[1])
        wl_star = wl_star[wl_mask]
        star_flux = star_flux[wl_mask]

        wl_mask = (wl_planet > wl_range[0]) & (wl_planet < wl_range[1])
        wl_planet = wl_planet[wl_mask]
        depth = depth[wl_mask]

    # Interpolate to the highest-resolution spectrum
    R_star = np.median(wl_star[1:]/np.abs(np.ediff1d(wl_star)))
    R_planet = np.median(wl_planet[1:]/np.abs(np.ediff1d(wl_planet)))

    if R_star > R_planet:
        wl = wl_star
        interp_func = si.interp1d(
            wl_planet, depth, bounds_error=False, fill_value=0.0,
        )
        depth = interp_func(wl)
    else:
        wl = wl_planet
        interp_func = si.interp1d(
            wl_star, star_flux, bounds_error=False, fill_value=0.0,
        )
        star_flux = interp_func(wl)

    if obs_type == 'transit':
        scaled_flux = star_flux * (1-depth)
    elif obs_type == 'eclipse':
        scaled_flux = star_flux * (1+depth)

    isort = np.argsort(wl)
    depth_scene = copy.deepcopy(scene)
    depth_scene['spectrum']['sed'] = dict(
        sed_type = 'input',
        spectrum = [wl[isort], scaled_flux[isort]],
    )
    # Do not normalize anymore, otherwise we lose the depth correction
    depth_scene['spectrum']['normalization'] = dict(type="none")

    # Now the star:
    star_scene = copy.deepcopy(scene)
    star_scene['spectrum']['sed'] = dict(
        sed_type = 'input',
        spectrum = [wl[isort], star_flux[isort]],
    )
    star_scene['spectrum']['normalization'] = dict(type="none")

    return star_scene, depth_scene


def simulate_tso(
        tso, n_obs=1, resolution=None, bins=None, noiseless=False,
    ):
    """
    Given a TSO dict from a pandeia TSO run, simulate a transit/eclipse
    spectrum scaling the noise by the number of observations and
    resampling over wavelength.

    Parameters
    ----------
    tso: dict
    n_obs: integer
        Number of transit/eclipse observations
    resolution: float
        If not None, resample the spectrum at the given resolution.
    bins: integer
        If not None, bin the spectrum in between the edges given
        by this array.
    noiseless: Bool
        If True, do not add scatter noise to the spectrum.

    Returns
    -------
    bin_wl: 1D array
    bin_spec: 1D array
    bin_err: 1D array
    bin_widths: 1D array

    Examples
    --------
    >>> import gen_tso.pandeia_io as jwst
    >>> import matplotlib.pyplot as plt

    >>> scene = jwst.make_scene(
    >>>     sed_type='phoenix', sed_model='k5v',
    >>>     norm_band='2mass,ks', norm_magnitude=8.637,
    >>> )

    >>> # With NIRSpec
    >>> pando = jwst.PandeiaCalculation('nirspec', 'bots')
    >>> pando.calc['scene'] = [scene]
    >>> disperser='g395h'
    >>> filter='f290lp'
    >>> readout='nrsrapid'
    >>> subarray='sub2048'
    >>> ngroup = 16

    >>> transit_dur = 2.71
    >>> obs_dur = 6.0
    >>> obs_type = 'transit'

    >>> depth_model = np.loadtxt(
    >>>     '../planet_spectra/WASP80b_transit.dat', unpack=True)

    >>> tso = pando.tso_calculation(
    >>>     obs_type, transit_dur, obs_dur, depth_model,
    >>>     ngroup, disperser, filter, subarray, readout,
    >>> )

    >>> bin_wl, bin_spec, bin_err, bin_widths = jwst.simulate_tso(
    >>>    tso, n_obs=1, resolution=300.0, noiseless=False,
    >>> )

    >>> plt.figure(10)
    >>> plt.clf()
    >>> plt.plot(depth_model[0], depth_model[1], c='orange')
    >>> plt.plot(tso['wl'], tso['depth_spectrum'], c='orangered')
    >>> plt.errorbar(bin_wl, bin_spec, bin_err, fmt='ok', ms=4, mfc='w')
    >>> plt.xlim(2.8, 5.3)
    """
    dt_in = tso['time_in']
    dt_out = tso['time_out']
    flux_in = tso['flux_in'] * n_obs
    flux_out = tso['flux_out'] * n_obs
    var_in = tso['var_in'] * n_obs
    var_out = tso['var_out'] * n_obs
    wl = tso['wl']

    # Binning
    wl_min = np.amin(wl)
    wl_max = np.amax(wl)
    if resolution is not None:
        bin_edges = constant_resolution_spectrum(wl_min, wl_max, resolution)
        bin_edges = np.append(bin_edges, wl_max)
    elif bins is not None:
        bin_edges = np.copy(bins)
        if bins[0] > wl_min:
            bin_edges = np.append(wl_min, bin_edges)
        else:
            bin_edges[0] = wl_min
        if bins[-1] < wl_max:
            bin_edges = np.append(bin_edges, wl_max)
        else:
            bin_edges[-1] = wl_max
    else:
        bin_edges = 0.5* (wl[1:] + wl[:-1])

    bin_wl = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    bin_widths = bin_wl - bin_edges[:-1]
    nbins = len(bin_wl)
    bin_out = np.zeros(nbins)
    bin_in = np.zeros(nbins)
    bin_vout = np.zeros(nbins)
    bin_vin = np.zeros(nbins)
    for i in range(nbins):
        bin_mask = (wl>=bin_edges[i]) & (wl<bin_edges[i+1])
        bin_out[i] = np.sum(flux_out[bin_mask])
        bin_in[i] = np.sum(flux_in[bin_mask])
        bin_vout[i] = np.sum(var_out[bin_mask])
        bin_vin[i] = np.sum(var_in[bin_mask])

    bin_err = np.sqrt(
        (dt_out/dt_in/bin_out)**2.0 * bin_vin +
        (bin_in*dt_out/dt_in/bin_out**2.0)**2.0 * bin_vout
    )
    # The numpy random system must have its seed reinitialized in
    # each sub-processes to avoid identical 'random' steps.
    # random.randomint is process- and thread-safe.
    np.random.seed(random.randint(0, 100000))
    rand_noise = np.random.normal(0.0, bin_err)
    bin_spec = 1.0 - (bin_in/dt_in) / (bin_out/dt_out)
    if not noiseless:
        bin_spec += rand_noise

    mask = bin_out > 0
    return bin_wl[mask], bin_spec[mask], bin_err[mask], bin_widths[mask]


def _print_pandeia_report(reports, format=None):
    """
    Return a text summarizing a tso_run() or perform_calculation() output.
    Similar to the results panel in the ETC.

    Examples
    --------
    >>> import gen_tso.pandeia_io as jwst

    >>> wl = np.logspace(0, 2, 1000)
    >>> depth = [wl, np.tile(0.03, len(wl))]
    >>> pando = jwst.PandeiaCalculation('nircam', 'ssgrism')
    >>> pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)
    >>> tso = pando.tso_calculation(
    >>>     'transit', transit_dur=2.1, obs_dur=6.0, depth_model=depth,
    >>>     ngroup=90, readout='rapid', filter='f444w',
    >>> )

    >>> tso_report = jwst._print_pandeia_report(tso, format='rich')
    >>> print(tso_report)
    """
    # This is a TSO dict
    if 'report_in' in reports:
        report_in = reports['report_in']
        report_out = reports['report_out']
    # This is a perform_calculation dict
    else:
        report_in = reports
        report_out = None

    # Put everything into a list to make things easier to handle:
    if not isinstance(report_in, list):
        report_in = [report_in]
    if report_out is not None and not isinstance(report_out, list):
        report_out = [report_out]

    # Exposure
    config = report_in[0]['input']['configuration']
    inst = config['instrument']['instrument']
    subarray = config['detector']['subarray']
    readout = config['detector']['readout_pattern']
    ngroup = config['detector']['ngroup']
    nint = config['detector']['nint']
    if report_out is not None:
        nint += report_out[0]['input']['configuration']['detector']['nint']
    text_report = _print_pandeia_exposure(inst, subarray, readout, ngroup, nint)

    # Saturation
    reports = report_in + report_out
    pixel_rate, full_well = saturation_level(reports, get_max=True)
    saturation_report = _print_pandeia_saturation(
        inst, subarray, readout, ngroup, pixel_rate, full_well,
        format=format,
    )
    text_report = f'{text_report}\n{saturation_report}'
    if format == 'html':
        text_report = text_report.replace('\n', '<br>')

    # Full report

    return text_report


def _print_pandeia_exposure(
        inst=None, subarray=None, readout=None, ngroup=None, nint=None,
        config=None, format=None,
    ):
    """
    Return a text summarizing a tso_run() or perform_calculation() output.
    Similar to the results panel in the ETC.

    Examples
    --------
    >>> import gen_tso.pandeia_io as jwst

    >>> wl = np.logspace(0, 2, 1000)
    >>> depth = [wl, np.tile(0.03, len(wl))]
    >>> pando = jwst.PandeiaCalculation('nircam', 'ssgrism')
    >>> pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)
    >>> tso = pando.tso_calculation(
    >>>     'transit', transit_dur=2.1, obs_dur=6.0, depth_model=depth,
    >>>     ngroup=190, readout='rapid', filter='f444w',
    >>> )

    >>> config = tso['report_out']['input']['configuration']
    >>> text = jwst._print_pandeia_exposure(config=config)

    >>> inst = 'nircam'
    >>> subarray = 'subgrism64'
    >>> readout = 'rapid'
    >>> ngroup = 90
    >>> nint = 150
    >>> text = jwst._print_pandeia_exposure(
    >>>     inst, subarray, readout, ngroup, nint,
    >>> )
    """
    if config is not None:
        inst = config['instrument']['instrument']
        subarray = config['detector']['subarray']
        readout = config['detector']['readout_pattern']
        nint = config['detector']['nint']
        ngroup = config['detector']['ngroup']

    exp_time = exposure_time(inst, subarray, readout, ngroup, nint)
    exposure_hours = exp_time / 3600.0
    exp_text = f'Exposure time: {exp_time:.2f} s ({exposure_hours:.2f} h)'
    return exp_text


def _print_pandeia_saturation(
        inst=None, subarray=None, readout=None, ngroup=None,
        pixel_rate=None, full_well=None, reports=None, format=None,
    ):
    """
    >>> import gen_tso.pandeia_io as jwst

    >>> wl = np.logspace(0, 2, 1000)
    >>> depth = [wl, np.tile(0.03, len(wl))]
    >>> pando = jwst.PandeiaCalculation('nircam', 'ssgrism')
    >>> pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)
    >>> tso = pando.tso_calculation(
    >>>     'transit', transit_dur=2.1, obs_dur=6.0, depth_model=depth,
    >>>     ngroup=190, readout='rapid', filter='f444w',
    >>> )

    >>> pixel_rate, full_well = jwst.saturation_level(tso, get_max=True)
    >>> inst = 'nircam'
    >>> subarray = 'subgrism64'
    >>> readout = 'rapid'
    >>> ngroup = 90
    >>> nint = 150
    >>> text = jwst._print_pandeia_saturation(
    >>>     inst, subarray, readout, ngroup, pixel_rate, full_well,
    >>>     format='html',
    >>> )

    >>> text = jwst._print_pandeia_saturation(
    >>>     reports=[tso], format='html',
    >>> )
    """
    # parse reports
    if reports is not None:
        pixel_rate, full_well = saturation_level(reports, get_max=True)
        # This is a TSO dict
        if 'report_in' in reports[0]:
            report = reports[0]['report_in']
        else:
            report = reports[0]
        config = report['input']['configuration']
        inst = config['instrument']['instrument']
        subarray = config['detector']['subarray']
        readout = config['detector']['readout_pattern']
        ngroup = config['detector']['ngroup']

    sat_time = saturation_time(inst, ngroup, readout, subarray)
    sat_fraction = 100 * pixel_rate * sat_time / full_well
    ngroup_80 = int(80*ngroup/sat_fraction)
    ngroup_max = int(100*ngroup/sat_fraction)

    saturation = format_text(
        f"{sat_fraction:.1f}%", sat_fraction>=81, sat_fraction>=100, format,
    )
    ngroup_80 = format_text(
        f"{ngroup_80:d}", ngroup_80==2, ngroup_80<2, format,
    )
    ngroup_max = format_text(
        f"{ngroup_max:d}", ngroup_max==2, ngroup_max<2, format,
    )
    saturation = f'Max fraction of saturation: {saturation}'
    ngroup_80_sat = f'ngroup below 80% saturation: {ngroup_80}'
    ngroup_max_sat = f'ngroup below 100% saturation: {ngroup_max}'

    sat_text = (
        f'{saturation}\n'
        f'{ngroup_80_sat}\n'
        f'{ngroup_max_sat}'
    )
    if format == 'html':
        sat_text = sat_text.replace('\n', '<br>')
    return sat_text


def print_pandeia_report(report_in, report_out=None, as_html=False):
    r"""
    Get a string summarizing a pandeia's perform_calculation scalar output.
    Skip saturation values (which are covered in another function).

    Parameters
    ----------
    report_in: Dictionary
        A pandeia's perform_calculation() 'scalar' output.
    report_out: Dictionary
        A pandeia's perform_calculation() 'scalar' output.
        If not None, assume that the inputs reports are an in-transit
        out-of-transit pair.
    as_html: Bool
        If True, replace '\n' linebreaks with HTML's <br>
        so that it formats well when printed as HTML.

    Examples
    --------
    >>> import gen_tso.pandeia_io as jwst

    >>> inst = 'nircam'
    >>> readout = 'rapid'
    >>> pando = jwst.PandeiaCalculation(inst, 'ssgrism')
    >>> pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)
    >>> result = pando.perform_calculation(
    >>>     ngroup=92, nint=683, readout='rapid', filter='f444w',
    >>> )
    >>> report = jwst.print_pandeia_report(result['scalar'])
    >>> print(report)

    Signal-to-noise ratio       5998.2
    Extracted flux              2106.2  e-/s
    Flux standard deviation        0.4  e-/s
    Brightest pixel rate        1396.6  e-/s

    Integrations:                         683
    Duty cycle:                          0.98
    Total exposure time:              21638.7  s
    Single exposure time:             21638.7  s
    First--last dt per exposure:      21638.7  s
    Reset--last dt per integration:   21169.9  s

    Reference wavelength:                    4.36  microns
    Area of extraction aperture:             4.76  pixels
    Area of background measurement:           6.3  pixels
    Background surface brightness:            0.3  MJy/sr
    Total sky flux in background aperture:   4.45  e-/s
    Total flux in background aperture:      65.97  e-/s
    Background flux fraction from scene:     0.93
    Number of cosmic rays:      0.0073  events/pixel/read
    """
    if not isinstance(report_in, list):
        report_in = [report_in]
    rate_in = report_in[0]['extracted_flux']
    if report_out is not None:
        if not isinstance(report_out, list):
           report_out = [report_out]
        rate_out = report_out[0]['extracted_flux']

    # Take report with more flux in it
    if report_out is None or rate_in > rate_out:
        reports = report_in
    else:
        reports = report_out

    dt_fmt = '8.1f' if report_in[0]['total_exposure_time'] > 100 else '8.3f'

    snr = ''
    flux = ''
    flux_std = ''
    pixel_rate = ''
    ref_wave = ''
    extract_area = ''
    bkg_area = ''
    bkg_brightness = ''
    sky = ''
    bkg_flux = ''
    bkg_source = ''
    for report in report_in:
        snr += f"{report['sn']:9.1f} "
        flux += f"{report['extracted_flux']:9.1f} "
        flux_std += f"{report['extracted_noise']:9.1f} "
        pixel_rate += f"{report['brightest_pixel']:9.1f} "
        ref_wave += f"{report['reference_wavelength']:6.2f} "
        extract_area += f"{report['extraction_area']:6.2f} "
        if report_in[0]['background_area'] is not None:
            bkg_area += f"{report['background_area']:6.1f} "
            bkg_brightness += f"{report['background']:6.1f} "
            sky += f"{report['background_sky']:6.2f} "
            bkg_flux += f"{report['background_total']:6.2f} "
            bkg_source += f"{report['contamination']:6.2f} "

    background_info = ''
    if report_in[0]['background_area'] is not None:
        background_info = (
            f"Area of background measurement:        {bkg_area} pixels\n"
            f"Background surface brightness:         {bkg_brightness} MJy/sr\n"
            f"Total sky flux in background aperture: {sky} e-/s\n"
            f"Total flux in background aperture:     {bkg_flux} e-/s\n"
            f"Background flux fraction from scene:   {bkg_source}\n"
        )

    cosmic_rays = f"{report_in[0]['cr_ramp_rate']:9.4f}"

    integs = ''
    duty_cycle = ''
    total_time = ''
    exp_time = ''
    dt_exposure = ''
    dt_integ = ''
    reports = report_in[0:1]
    if report_out is not None:
        reports.append(report_out[0])
    for report in reports:
        integs += f"{report['total_integrations']:8d} "
        duty_cycle += f"{report['duty_cycle']:8.2f} "
        total_time += f"{report['total_exposure_time']:{dt_fmt}} "
        exp_time += f"{report['all_dithers_time']:{dt_fmt}} "
        dt_exposure += f"{report['exposure_time']:{dt_fmt}} "
        dt_integ += f"{report['measurement_time']:{dt_fmt}} "

    if report_out is None:
        tso_header = ''
    else:
        tso_header = f"{'in-transit':>41}  out-transit\n"
    if report_in[0]['disperser'] in ['short', 'medium', 'long']:
        channels = "CH1 CH2 CH3 CH4"
        band_header1 = f"{'':31s}{channels.replace(' ', '       ')}\n"
        band_header2 = f"{'':42s}{channels.replace(' ', '    ')}\n"
    else:
        band_header1 = band_header2 = ''

    summary = (
        f"{band_header1}"
        f"Signal-to-noise ratio    {snr}\n"
        f"Extracted flux           {flux} e-/s\n"
        f"Flux standard deviation  {flux_std} e-/s\n"
        f"Brightest pixel rate     {pixel_rate} e-/s\n\n"

        f"{tso_header}"
        f"Integrations:                    {integs}\n"
        f"Duty cycle:                      {duty_cycle}\n"
        f"Total exposure time:             {total_time} s\n"
        f"Single exposure time:            {exp_time} s\n"
        f"First--last dt per exposure:     {dt_exposure} s\n"
        f"Reset--last dt per integration:  {dt_integ} s\n\n"

        f"{band_header2}"
        f"Reference wavelength:                  {ref_wave} microns\n"
        f"Area of extraction aperture:           {extract_area} pixels\n"
        f"{background_info}"
        f"Number of cosmic rays:   {cosmic_rays}  events/pixel/read"
    )

    if as_html:
        summary = summary.replace('\n', '<br>')
    return summary


class PandeiaCalculation():
    """
    A class to interface with the pandeia.engine package.

    Parameters
    ----------
    instrument: string
        The JWST instrument: nircam, niriss, nirspec, miri.
    mode: string
        Observing mode. If not set, default to the first item for
        each instrument from the list below.
        - nircam:
            ssgrism        spectroscopy
            target_acq     aquisition
        - niriss:
            soss           spectroscopy
            target_acq     aquisition
        - nirspec:
            bots           spectroscopy
            target_acq     aquisition
        - miri:
            lrsslitless    spectroscopy
            mrs_ts         spectroscopy
            target_acq     aquisition

    Examples
    --------
    >>> import gen_tso.pandeia_io as jwst
    >>> pando = jwst.PandeiaCalculation('nircam', 'target_acq')
    """
    def __init__(self, instrument, mode=None):
        if mode is None:
            # pick the most popular spectroscopic mode
            pass
        if mode == 'acquisition':
            mode = 'target_acq'
        self.telescope = 'jwst'
        self.instrument = instrument
        self.mode = mode
        self.calc = build_default_calc(
            self.telescope, self.instrument, self.mode,
        )

    def get_configs(self, output=None):
        """
        Print out or return the list of available configurations.

        Parameters
        ----------
        output: String
            The configuration variable to list. Select from:
            readouts, subarrays, filters, or dispersers.

        Returns
        -------
            outputs: 1D list of strings
            The list of available inputs for the requested variable.
        """
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


    def set_scene(
            self, sed_type, sed_model, norm_band, norm_magnitude,
            background='ecliptic_low',
        ):
        """
        Set the stellar point-source scene to observe.

        Parameters
        ----------
        sed_type: String
            Type of model: 'phoenix', 'k93models', 'blackbody', or 'flat'
        sed_model:
            The SED model required for each sed_type:
            - phoenix or k93models: the model key (see load_sed_list)
            - blackbody: the effective temperature (K)
            - flat: the unit ('flam' or 'fnu')
        norm_band: String
            Band over which to normalize the spectrum.
        norm_magnitude: float
            Magnitude of the star at norm_band.
        background: String
            Set the background flux. Select from:
            'ecliptic_low', 'ecliptic_medium', 'ecliptic_high',
            'minzodi_low',  'minzodi_medium',  'minzodi_high'

        Examples
        --------
        >>> import pandeia_interface as jwst

        >>> instrument = 'nircam'
        >>> mode = 'ssgrism'
        >>> pando = jwst.PandeiaCalculation(instrument, mode)
        >>> pando.set_scene(
        >>>     sed_type='phoenix', sed_model='k5v',
        >>>     norm_band='2mass,ks', norm_magnitude=8.637,
        >>> )
        """
        scene = make_scene(sed_type, sed_model, norm_band, norm_magnitude)
        self.calc['scene'] = [scene]
        bkg, bkg_level = background.strip().split('_')
        self.calc['background'] = bkg
        self.calc['background_level'] = bkg_level


    def get_saturation_values(
            self, disperser, filter, subarray, readout, ngroup=2,
            aperture=None, get_max=False,
        ):
        """
        Calculate the brightest-pixel rate (e-/s) and full_well (e-)
        for the current instrument and scene configuration, which once known,
        are sufficient to calculate the saturation level once the
        saturation  time is known.

        Examples
        --------
        >>> import gen_tso.pandeia_io as jwst

        >>> instrument = 'nircam'
        >>> mode = 'ssgrism'
        >>> pando = jwst.PandeiaCalculation(instrument, mode)
        >>> pando.set_scene(
        >>>     sed_type='phoenix', sed_model='k2v',
        >>>     norm_band='2mass,ks', norm_magnitude=8.351,
        >>> )
        >>> brightest_pixel_rate, full_well = pando.get_saturation_values(
        >>>     disperser='grismr', filter='f444w',
        >>>     readout='rapid', subarray='subgrism64',
        >>> )

        >>> # Also works for Target Acquisition:
        >>> instrument = 'nircam'
        >>> mode = 'target_acq'
        >>> pando = jwst.PandeiaCalculation(instrument, mode)
        >>> pando.set_scene(
        >>>     sed_type='phoenix', sed_model='k5v',
        >>>     norm_band='2mass,ks', norm_magnitude=8.351,
        >>> )
        >>> brightest_pixel_rate, full_well = pando.get_saturation_values(
        >>>     disperser=None, filter='f335m',
        >>>     readout='rapid', subarray='sub32tats', ngroup=3,
        >>> )
        """
        # TBD: Automate here the group setting
        reports = self.perform_calculation(
            ngroup=ngroup, nint=1,
            disperser=disperser, filter=filter,
            subarray=subarray, readout=readout,
            aperture=aperture,
        )
        brightest_pixel_rate, full_well = saturation_level(reports, get_max)
        return brightest_pixel_rate, full_well


    def perform_calculation(
            self, ngroup, nint,
            disperser=None, filter=None, subarray=None, readout=None,
            aperture=None,
        ):
        """
        Run pandeia's perform_calculation() for the given configuration
        (or set of configurations, see notes below).

        Parameter
        ----------
        ngroup: Integeer
            Number of groups per integration.  Must be >= 2.
        nint: Integer
            Number of integrations.
        disperser: String
            Disperser/grating for the given instrument.
        filter: String
            Filter for the given instrument.
        subarray: String
            Subarray mode for the given instrument.
        readout: String
            Readout pattern mode for the given instrument.
        aperture: String
            Aperture configuration for the given instrument.

        Returns
        -------
        report: dict
            The Pandeia's report output for the given configuration.
            If there's more than one requested calculation, return a
            list of reports.

        Notes
        -----
        - Provide a list of values for any of these arguments to
          compute a batch of calculations.
        - To leave a config parameter unmodified, leave the respective
          argument as None.
          To set a config parameter as None, set the argument to ''.
        """
        if not isinstance(nint, Iterable):
            nint = [nint]
        if not isinstance(ngroup, Iterable):
            ngroup = [ngroup]
        if not isinstance(disperser, Iterable) or isinstance(disperser, str):
            disperser = [disperser]
        if not isinstance(filter, Iterable) or isinstance(filter, str):
            filter = [filter]
        if not isinstance(subarray, Iterable) or isinstance(subarray, str):
            subarray = [subarray]
        if not isinstance(readout, Iterable) or isinstance(readout, str):
            readout = [readout]
        if not isinstance(aperture, Iterable) or isinstance(aperture, str):
            aperture = [aperture]

        configs = product(
            aperture, disperser, filter, subarray, readout, nint, ngroup,
        )

        reports = [
            self._perform_calculation(config)
            for config in configs
        ]
        if len(reports) == 1:
            return reports[0]
        return reports


    def _perform_calculation(self, params):
        """
        (the real function that) runs pandeia.
        """
        # Unpack configuration parameters
        aperture, disperser, filter, subarray, readout, nint, ngroup = params
        if aperture is not None:
            self.calc['configuration']['instrument']['aperture'] = aperture
        if disperser is not None:
            self.calc['configuration']['instrument']['disperser'] = disperser
        if readout is not None:
            self.calc['configuration']['detector']['readout_pattern'] = readout
        if subarray is not None:
            self.calc['configuration']['detector']['subarray'] = subarray
        if filter == '':
            self.calc['configuration']['instrument']['filter'] = None
        elif filter is not None:
            self.calc['configuration']['instrument']['filter'] = filter

        if self.instrument == 'niriss':
            self.calc['strategy']['order'] = 1
            # DataError: No mask configured for SOSS order 2.

        self.calc['configuration']['detector']['nexp'] = 1 # dither
        self.calc['configuration']['detector']['nint'] = nint
        self.calc['configuration']['detector']['ngroup'] = ngroup

        report = perform_calculation(self.calc)
        self.report = report
        return report

    def calc_noise(
            self, obs_dur, ngroup,
            disperser=None, filter=None, subarray=None, readout=None,
            aperture=None,
        ):
        """
        Run a Pandeia calculation and extract the observed wavelength,
        flux, and variances.

        Parameters
        ----------
        obs_dur: Float
            Duration of the observation.
        ngroup: Integer
            Number of groups per integrations
        disperser: String
        filter: String
        subarray: String
        readout: String
        aperture: String

        Returns
        -------
        TBD

        Examples
        --------
        >>> import gen_tso.pandeia_io as jwst

        >>> instrument = 'nircam'
        >>> mode = 'ssgrism'
        >>> pando = jwst.PandeiaCalculation(instrument, mode)
        >>> pando.set_scene(
        >>>     sed_type='phoenix', sed_model='k5v',
        >>>     norm_band='2mass,ks', norm_magnitude=8.637,
        >>> )
        >>> # Example TBD
        """
        if aperture is None:
            aperture = self.calc['configuration']['instrument']['aperture']
        if disperser is None:
            disperser = self.calc['configuration']['instrument']['disperser']
        if readout is None:
            readout = self.calc['configuration']['detector']['readout_pattern']
        if subarray is None:
            subarray = self.calc['configuration']['detector']['subarray']
        if filter is None:
            filter = self.calc['configuration']['instrument']['filter']

        ins_config = get_instrument_config(self.telescope, self.instrument)
        single_exp_time = exposure_time(
            self.instrument, nint=1, ngroup=ngroup,
            readout=readout, subarray=subarray,
        )

        nint = int(obs_dur*3600/single_exp_time)
        report = self.perform_calculation(
            ngroup, nint, disperser, filter, subarray, readout, aperture,
        )

        # Flux:
        measurement_time = report['scalar']['measurement_time']
        flux = report['1d']['extracted_flux'][1] * measurement_time
        wl = report['1d']['extracted_flux'][0]

        # Background variance:
        background_var = report['1d']['extracted_bg_only'][1] * measurement_time
        # Read noise variance:
        read_noise = read_noise_variance(report, ins_config)
        npix = report['scalar']['extraction_area']
        read_noise_var = 2.0 * read_noise**2.0 * nint * npix
        # Pandeia (multiaccum) noise:
        shot_var = (report['1d']['extracted_noise'][1] * measurement_time)**2.0
        # Last-minus-first (LMF) noise:
        lmf_var = np.abs(flux) + background_var + read_noise_var

        variances = lmf_var, shot_var, background_var, read_noise_var

        return report, wl, flux, variances, measurement_time


    def tso_calculation(
            self, obs_type, transit_dur, obs_dur, depth_model,
            ngroup, disperser=None, filter=None,
            subarray=None, readout=None, aperture=None,
        ):
        """
        Run pandeia to simulate a transit/eclipse time-series observation

        Parameters
        ----------
        obs_type: String
            The observing geometry 'transit' or 'eclipse'.
        transit_dur: Float
            Duration of the transit or eclipse event in hours.
        obs_dur: Float
            Total duration of the observation (baseline plus transit
            or eclipse event) in hours.
        depth_model: list of two 1D array or a 2D array
            The transit or eclipse depth spectrum where the first item is
            the wavelength (um) and the second is the depth.
        ngroup: Integer
            Number of groups per integrations
        disperser: String
        filter: String
        subarray: String
        readout: String
        aperture: String

        Returns
        -------
        tso: dict
            A dictionary containing the time-series observation data:
            - wl: instrumental wavelength sampling (microns)
            - depth_spectrum: Transit/eclipse depth spectrum at instrumental wl
            - time_in: In-transit/eclipse measuring time (seconds)
            - flux_in: In-transit/eclipse flux (e-)
            - var_in:  In-transit/eclipse variance
            - time_out: Out-of-transit/eclipse measuring time (seconds)
            - flux_out: Out-of-transit/eclipse flux (e-)
            - var_out:  Out-of-transit/eclipse
            - report_in:  In-transit/eclipse pandeia output report
            - report_out:  Out-of-transit/eclipse pandeia output report

        Examples
        --------
        >>> import pandeia_interface as jwst
        >>> from pandeia_interface import set_depth_scene

        >>> # Set the stellar scene and transit:
        >>> scene = jwst.make_scene(
        >>>     sed_type='phoenix', sed_model='k5v',
        >>>     norm_band='2mass,ks', norm_magnitude=8.637,
        >>> )
        >>> transit_dur = 2.753
        >>> obs_dur = 7.1
        >>> obs_type = 'transit'
        >>> depth_model = np.loadtxt(
        >>>     '../planet_spectra/WASP80b_transit.dat', unpack=True)

        >>> # Set a NIRSpec observation
        >>> pando = jwst.PandeiaCalculation('nirspec', 'bots')
        >>> pando.calc['scene'] = [scene]
        >>> disperser = 'g395h'
        >>> filter = 'f290lp'
        >>> readout = 'nrsrapid'
        >>> subarray = 'sub2048'
        >>> ngroup = 16

        >>> tso = pando.tso_calculation(
        >>>     obs_type, transit_dur, obs_dur, depth_model,
        >>>     ngroup, disperser, filter, subarray, readout,
        >>> )

        >>> # Fluxes and Flux rates
        >>> col1, col2 = plt.cm.viridis(0.8), plt.cm.viridis(0.25)
        >>> plt.figure(0, (8.5, 4))
        >>> plt.clf()
        >>> plt.subplot(121)
        >>> plt.plot(tso['wl'], tso['flux_out'], c=col2, label='out of transit')
        >>> plt.plot(tso['wl'], tso['flux_in'], c=col1, label='in transit')
        >>> plt.legend(loc='best')
        >>> plt.xlabel('Wavelength (um)')
        >>> plt.ylabel('Collected flux (e-)')
        >>> plt.subplot(122)
        >>> plt.plot(tso['wl'], tso['flux_out']/tso['time_out'], c=col2, label='out of transit')
        >>> plt.plot(tso['wl'], tso['flux_in']/tso['time_in'], c=col1, label='in transit')
        >>> plt.legend(loc='best')
        >>> plt.xlabel('Wavelength (um)')
        >>> plt.ylabel('Flux rate (e-/s)')
        >>> plt.tight_layout()

        >>> # Model and instrument-observed transit depth spectrum
        >>> wl_depth, depth = depth_model
        >>> plt.figure(4)
        >>> plt.clf()
        >>> plt.plot(wl_depth, 100*depth, c='orange', label='model depth')
        >>> plt.plot(tso['wl'], 100*tso['depth_spectrum'], c='b', label='obs depth')
        >>> plt.legend(loc='best')
        >>> plt.xlim(2.75, 5.25)
        >>> plt.xlabel('Wavelength (um)')
        >>> plt.ylabel('Transit depth (%)')
        """
        # Scale in or out transit flux rates
        scene = self.calc['scene'][0]
        star_scene, depth_scene = set_depth_scene(scene, obs_type, depth_model)
        if obs_type == 'eclipse':
            scene_in = star_scene
            scene_out = depth_scene
        elif obs_type == 'transit':
            scene_in = depth_scene
            scene_out = star_scene

        if not isinstance(ngroup, Iterable):
            ngroup = [ngroup]
        if not isinstance(disperser, Iterable) or isinstance(disperser, str):
            disperser = [disperser]
        if not isinstance(filter, Iterable) or isinstance(filter, str):
            filter = [filter]
        if not isinstance(subarray, Iterable) or isinstance(subarray, str):
            subarray = [subarray]
        if not isinstance(readout, Iterable) or isinstance(readout, str):
            readout = [readout]
        if not isinstance(aperture, Iterable) or isinstance(aperture, str):
            aperture = [aperture]

        configs = product(
            aperture, disperser, filter, subarray, readout, ngroup,
        )
        tso = [
            self._tso_calculation(config, scene_in, scene_out, transit_dur, obs_dur)
            for config in configs
        ]
        if len(tso) == 1:
             tso = tso[0]
        self.tso = tso
        # Return scene to its previous state
        self.calc['scene'][0] = scene
        return tso


    def _tso_calculation(
            self, config, scene_in, scene_out, transit_dur, obs_dur,
        ):
        """
        (the real function that) runs a TSO calculation.
        """
        aperture, disperser, filter, subarray, readout, ngroup = config
        if aperture is not None:
            self.calc['configuration']['instrument']['aperture'] = aperture
        if disperser is not None:
            self.calc['configuration']['instrument']['disperser'] = disperser
        if readout is not None:
            self.calc['configuration']['detector']['readout_pattern'] = readout
        if subarray is not None:
            self.calc['configuration']['detector']['subarray'] = subarray
        if filter == '':
            self.calc['configuration']['instrument']['filter'] = None
        elif filter is not None:
            self.calc['configuration']['instrument']['filter'] = filter

        # Compute observed fluxes and noises:
        self.calc['scene'][0] = scene_in
        report_in, wl, flux_in, variances_in, time_in = self.calc_noise(
            transit_dur, ngroup,
        )
        var_lmf_in = variances_in[0]

        out_transit_dur = obs_dur - transit_dur
        self.calc['scene'][0] = scene_out
        report_out, wl, flux_out, variances_out, time_out = self.calc_noise(
            out_transit_dur, ngroup,
        )
        var_lmf_out = variances_out[0]

        # Mask out un-illumnated wavelengths (looking at you, G395H)
        mask = flux_in > 1e-6 * np.median(flux_in)
        wl = wl[mask]
        flux_in = flux_in[mask]
        flux_out = flux_out[mask]
        var_in = var_lmf_in[mask]
        var_out = var_lmf_out[mask]
        obs_depth = 1 - (flux_in/time_in) / (flux_out/time_out)

        tso = {
            'wl': wl,
            'depth_spectrum': obs_depth,
            'time_in': time_in,
            'flux_in': flux_in,
            'var_in': var_in,
            'time_out': time_out,
            'flux_out': flux_out,
            'var_out': var_out,
            'report_in': report_in,
            'report_out': report_out,
        }
        return tso

    def simulate_tso(
            self, n_obs=1, resolution=None, bins=None, noiseless=False,
        ):
        """
        Simulate a time-series observation spectrum with noise
        for the given number of observations and spectral sampling.

        Parameters
        ----------
        TBD

        Returns
        -------
        TBD

        Examples
        --------
        >>> TBD
        """
        return simulate_tso(self.tso, n_obs, resolution, bins, noiseless)

