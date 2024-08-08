# Copyright (c) 2024 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    'read_noise_variance',
    'exposure_time',
    'bin_search_exposure_time',
    'integration_time',
    'saturation_level',
    'load_sed_list',
    'find_closest_sed',
    'extract_sed',
    'make_scene',
    'set_depth_scene',
    'simulate_tso',
    'get_tso_wl_range',
    'get_tso_depth_range',
    '_print_pandeia_exposure',
    '_print_pandeia_saturation',
    '_print_pandeia_stats',
    '_print_pandeia_report',
    'tso_print',
]

import copy
from decimal import Decimal
import json
import random

import numpy as np
import scipy.interpolate as si
import pandeia.engine.sed as sed
from pandeia.engine.calc_utils import get_instrument_config
from pandeia.engine.normalization import NormalizationFactory
from pyratbay.spectrum import constant_resolution_spectrum
from pyratbay.tools import u
import prompt_toolkit

from ..utils import format_text


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
    elif report_config['mode'] == 'sw_tsgrism':
        aperture = report_config['aperture']
        noise = ins_config['detector_config']['sw']['rn']
        if aperture not in noise:
            aperture = 'default'
        read_noise = noise[aperture]
        return read_noise
    else:
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


def integration_time(instrument, subarray, readout, ngroup):
    """
    Compute JWST's integration time for a given instrument configuration.
    Based on pandeia.engine.exposure.

    Examples
    --------
    >>> import gen_tso.pandeia_io as jwst

    >>> # Calculate integration time for a given instrument setup:
    >>> inst = 'nircam'
    >>> ngroup = 90
    >>> readout = 'rapid'
    >>> subarray = 'subgrism64'
    >>> integ_time = jwst.integration_time(inst, subarray, readout, ngroup)
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

    integration_time = tframe * (
        ndrop1 + (ngroup - 1) * (nframe + ndrop2) + nframe
    )
    return integration_time


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
    >>> pando = jwst.PandeiaCalculation(inst, 'lw_tsgrism')
    >>> pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)
    >>> result = pando.perform_calculation(
    >>>     ngroup=2, nint=683, readout='rapid', filter='f444w',
    >>> )
    >>> pixel_rate, full_well = jwst.saturation_level(result)

    >>> # Now I can calculate the saturation level for any integration time:
    >>> # (for the given filter and scene)
    >>> subarray = 'subgrism64'
    >>> for ngroup in [2, 97, 122]:
    >>>     integ_time = jwst.integration_time(inst, subarray, readout, ngroup)
    >>>     sat_level = pixel_rate * integ_time / full_well * 100
    >>>     print(f'Sat. fraction for {ngroup:3d} groups: {sat_level:5.1f}%')
    Sat. fraction for   2 groups:   1.5%
    Sat. fraction for  97 groups:  73.9%
    Sat. fraction for 122 groups:  93.0%
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
    >>> import gen_tso.pandeia_io as jwst
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


def find_closest_sed(teff, logg, models_teff=None, models_logg=None, sed_type='phoenix'):
    """
    A very simple cost-function to find the closest stellar model
    within a non-regular Teff-log_g grid.

    Since these are not regular grids, the cost function is not an
    absolute science, it depends on what weights more Teff of logg
    for a given case.  The current formula seems to be a good balance.

    Parameters
    ----------
    teff: float
        Target effective temperature.
    logg: float
        Target log(g).
    models_teff: list of floats
        SED model effective-temperature grid.
    models_logg: list of floats
        SED model log(g) grid.
    sed_type: String
        Select from 'phoenix' or 'k93models'

    Returns
    -------
    If models_teff or models_logg are None
        sed: String
        The SED key that best matches the teff,logg pair.
    Else
        idx: integer
        index of model with the closest Teff and logg.

    Examples
    --------
    >>> import gen_tso.pandeia_io as jwst
    >>>
    >>> # Kurucz models
    >>> sed = jwst.find_closest_sed(
    >>>     teff=4143.0, logg=4.66, sed_type='k93models',
    >>> )
    >>> print(f'SED: {repr(sed)}')
    SED: 'k7v'
    >>>
    >>> # PHOENIX models when I already have the list of models:
    >>> keys, names, p_teff, p_logg = jwst.load_sed_list('phoenix')
    >>> teff = 4143.0
    >>> logg = 4.66
    >>> idx = jwst.find_closest_sed(teff, logg, p_teff, p_logg)
    >>> print(f'{keys[idx]}: {repr(names[idx])}')
    k5v: 'K5V 4250K log(g)=4.5'
    """
    return_key = False
    if models_teff is None or models_logg is None:
        keys, names, models_teff, models_logg = load_sed_list(sed_type)
        return_key = True
    cost = (
        np.abs(np.log10(teff/models_teff)) +
        np.abs(logg-models_logg) / 15.0
    )
    idx = np.argmin(cost)
    if return_key:
        return keys[idx]
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
    elif sed_type == 'input':
        sed['spectrum'] = sed_model['wl'], sed_model['flux']

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
    >>> import gen_tso.pandeia_io as jwst

    >>> # A pandeia stellar point-source scene:
    >>> scene = jwst.make_scene(
    >>>     sed_type='k93models', sed_model='k7v',
    >>>      norm_band='2mass,ks', norm_magnitude=8.637,
    >>> )
    >>>
    >>> # A transit-depth spectrum:
    >>> depth_model = np.loadtxt('WASP80b_transit.dat', unpack=True)
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


def get_tso_wl_range(tso_run):
    """
    Get the wavelength range covered by a TSO calculation

    Parameters
    ----------
    tso_run: Dictionary
        A TSO calculation output as computed by run_pandeia() in the app.
    wl_scale: String
        Wavelength scale: 'linear' or 'log'.

    Returns
    -------
    min_wl: Float
        Shorter-wavelength boundary.
    max_wl: Float
        Longer-wavelength boundary.
    """
    runs = tso_run['tso']
    if not isinstance(runs, list):
        runs = [runs]

    min_wl = np.amin([np.amin(run['wl']) for run in runs])
    max_wl = np.amax([np.amax(run['wl']) for run in runs])
    # 5% margin
    d_wl = 0.025 * (max_wl-min_wl)
    min_wl = np.round(min_wl-d_wl, decimals=2)
    max_wl = np.round(max_wl+d_wl, decimals=2)
    return min_wl, max_wl


def get_tso_depth_range(tso_run, resolution, units):
    """
    Get the transit/eclipse depth range covered by a TSO calculation

    Parameters
    ----------
    tso_run: Dictionary
        A TSO calculation output as computed by run_pandeia() in the app.
    resolution: Float
        Spectral resolution at which to sample the spectrum.
    units: String
        Depth units, select from: 'none', 'percent', 'ppm'.

    Returns
    -------
    min_depth: Float
        Lower depth boundary (~min depth - 3*sigma_depth).
    max_depth: Float
        Higher depth boundary (~max depth + 3*sigma_depth).
    step: Float
        A quarter of the peak-to-peak depth distance.
    """
    runs = tso_run['tso']
    if not isinstance(runs, list):
        runs = [runs]

    max_depth = []
    min_depth = []
    for tso in runs:
        bin_wl, bin_spec, bin_err, widths = simulate_tso(
            tso, resolution=resolution, noiseless=True,
        )
        err_median = np.median(bin_err)
        d_max1 = np.amax(tso['depth_spectrum'] + 3*err_median)
        d_min1 = np.amin(tso['depth_spectrum'] - 3*err_median)
        max_depth.append(d_max1)
        min_depth.append(d_min1)

    min_depth = np.amin(min_depth) / u(units)
    max_depth = np.amax(max_depth) / u(units)
    step = 0.25*(max_depth - min_depth)
    # Only one significant digit on the step size:
    digits = - Decimal(step).adjusted()
    min_depth = np.round(min_depth, decimals=digits)
    max_depth = np.round(max_depth, decimals=digits)
    step = np.round(step, decimals=digits)
    return min_depth, max_depth, step


def _print_pandeia_exposure(
        inst=None, subarray=None, readout=None, ngroup=None, nint=None,
        config=None, format=None,
    ):
    """
    Return a text showing the total exposure time in seconds and hours.
    Either config or the set of inst, subarray, readout, ngroup, and nint
    values must be defined.

    Examples
    --------
    >>> import gen_tso.pandeia_io as jwst
    >>> import numpy as np
    >>>
    >>> wl = np.logspace(0, 2, 1000)
    >>> depth = [wl, np.tile(0.03, len(wl))]
    >>> pando = jwst.PandeiaCalculation('nircam', 'lw_tsgrism')
    >>> pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)
    >>> tso = pando.tso_calculation(
    >>>     'transit', transit_dur=2.1, obs_dur=6.0, depth_model=depth,
    >>>     ngroup=190, readout='rapid', filter='f444w',
    >>> )
    >>>
    >>> # Print from config dictionary:
    >>> config = tso['report_out']['input']['configuration']
    >>> print(jwst._print_pandeia_exposure(config=config))
    Exposure time: 13988.25 s (3.89 h)
    >>>
    >>> # Print from direct input values:
    >>> inst = 'nircam'
    >>> subarray = 'subgrism64'
    >>> readout = 'rapid'
    >>> ngroup = 90
    >>> nint = 150
    >>> text = jwst._print_pandeia_exposure(
    >>>     inst, subarray, readout, ngroup, nint,
    >>> )
    >>> print(text)
    Exposure time: 4650.09 s (1.29 h)
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
    >>> import numpy as np

    >>> wl = np.logspace(0, 2, 1000)
    >>> depth = [wl, np.tile(0.03, len(wl))]
    >>> pando = jwst.PandeiaCalculation('nircam', 'lw_tsgrism')
    >>> pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)
    >>> tso = pando.tso_calculation(
    >>>     'transit', transit_dur=2.1, obs_dur=6.0, depth_model=depth,
    >>>     ngroup=190, readout='rapid', filter='f444w',
    >>> )

    >>> # Directly print from input values:
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

    >>> # Print from tso_calculation() or perform_calculation() output:
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

    sat_time = integration_time(inst, subarray, readout, ngroup)
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


def _print_pandeia_stats(inst, mode, report_in, report_out=None, format=None):
    r"""
    Return a text summarizing the SNR, timings, and backround info
    from a perform_calculation() or a tso_calculation() output.

    Parameters
    ----------
    inst: String
        Instrument name.
    mode: String
        Instrument's mode.
    report_in: Dictionary
        A tso_calculation() or pandeia's perform_calculation() output.
    report_out: Dictionary
        A pandeia's perform_calculation() output.
        If not None, assume that the inputs reports are an in-transit
        and out-of-transit pair.
    format: String
        If None format as plain text (e.g., for print() calls)
        If 'rich' format as rich/colorful text (e.g., for FormattedText())
        If 'html' format as HTML text (e.g., for browser applications)

    Examples
    --------
    >>> import gen_tso.pandeia_io as jwst

    >>> inst = 'nircam'
    >>> readout = 'rapid'
    >>> pando = jwst.PandeiaCalculation(inst, 'lw_tsgrism')
    >>> pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)
    >>> result = pando.perform_calculation(
    >>>     ngroup=92, nint=683, readout='rapid', filter='f444w',
    >>> )
    >>> report = jwst.print_pandeia_report(inst, mode, result['scalar'])
    >>> print(report)
    """
    if not isinstance(report_in, list):
        report_in = [report_in]
    rate_in = report_in[0]['extracted_flux']
    if report_out == []:
        report_out = None
    if report_out is not None:
        if not isinstance(report_out, list):
           report_out = [report_out]
        rate_out = report_out[0]['extracted_flux']

    # Take report with more flux in it
    if report_out is None or rate_in > rate_out:
        reports = report_in
    else:
        reports = report_out

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
    min_snr = 0.0
    if mode == 'target_acq' and inst in ['miri', 'nirspec']:
        min_snr = 20.0
    if mode == 'target_acq' and inst in ['niriss', 'nircam']:
        min_snr = 30.0

    for report in report_in:
        sn = report['sn']
        snr += format_text(f"{sn:9.1f} ", danger=sn<min_snr, format=format)
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
            f"Background flux fraction from scene:   {bkg_source.rstrip()}\n"
        )

    cosmic_rays = f"{report_in[0]['cr_ramp_rate']:9.4f}"

    integs = ''
    duty_cycle = ''
    total_time = ''
    exp_time = ''
    dt_exposure = ''
    dt_integ = ''
    dt_fmt = '8.1f' if report_in[0]['total_exposure_time'] > 100 else '8.3f'
    min_duty = 0.0 if mode == 'target_acq' else 0.49
    reports = report_in[0:1]
    if report_out is not None:
        reports.append(report_out[0])
    for report in reports:
        integs += f"{report['total_integrations']:8d} "
        duty = report['duty_cycle']
        duty_cycle += format_text(
            f"{duty:8.2f} ", warning=duty<min_duty, format=format,
        )
        total_time += f"{report['total_exposure_time']:{dt_fmt}} "
        exp_time += f"{report['all_dithers_time']:{dt_fmt}} "
        dt_exposure += f"{report['exposure_time']:{dt_fmt}} "
        dt_integ += f"{report['measurement_time']:{dt_fmt}} "

    if report_out is None:
        tso_header = ''
    else:
        tso_header = f"{'in-transit':>41}  out-transit\n"
    if mode == 'miri_ts' and len(reports)==4:
        channels = "CH1 CH2 CH3 CH4"
        band_header1 = f"{'':31s}{channels.replace(' ', '       ')}\n"
        band_header2 = f"{'':42s}{channels.replace(' ', '    ')}\n"
    else:
        band_header1 = band_header2 = ''

    summary = (
        f"{band_header1}"
        f"Signal-to-noise ratio    {snr.rstrip()}\n"
        f"Extracted flux           {flux} e-/s\n"
        f"Flux standard deviation  {flux_std} e-/s\n"
        f"Brightest pixel rate     {pixel_rate} e-/s\n\n"

        f"{tso_header}"
        f"Integrations:                    {integs.rstrip()}\n"
        f"Duty cycle:                      {duty_cycle.rstrip()}\n"
        f"Total exposure time:             {total_time} s\n"
        # Ignore exp_time since it always matches total_time (nexp=1)
        #f"Single exposure time:            {exp_time} s\n"
        f"First--last dt per exposure:     {dt_exposure} s\n"
        f"Reset--last dt per integration:  {dt_integ} s\n\n"

        f"{band_header2}"
        f"Reference wavelength:                  {ref_wave} microns\n"
        f"Area of extraction aperture:           {extract_area} pixels\n"
        f"{background_info}"
        f"Number of cosmic rays:   {cosmic_rays}  events/pixel/read"
    )

    if format=='html':
        summary = summary.replace('\n', '<br>')
    return summary


def _print_pandeia_report(reports, format=None):
    """
    Get a text summarizing tso_calculation() or perform_calculation() output.
    Similar to the results panel in the ETC.

    Parameters
    ----------
    reports: Dictionary
        A tso_calculation() or a perform_calculation() output.
        Or a list of them.
    format: String
        If None format as plain text (e.g., for print() calls)
        If 'rich' format as rich/colorful text (e.g., for FormattedText())
        If 'html' format as HTML text (e.g., for browser applications)

    Examples
    --------
    >>> import gen_tso.pandeia_io as jwst

    >>> wl = np.logspace(0, 2, 1000)
    >>> depth = [wl, np.tile(0.03, len(wl))]
    >>> pando = jwst.PandeiaCalculation('nircam', 'lw_tsgrism')
    >>> pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)
    >>> tso = pando.tso_calculation(
    >>>     'transit', transit_dur=2.1, obs_dur=6.0, depth_model=depth,
    >>>     ngroup=90, readout='rapid', filter='f444w',
    >>> )

    >>> tso_report = jwst._print_pandeia_report(tso, format=None)
    >>> print(tso_report)

    Exposure time: 21545.44 s (5.98 h)
    Max fraction of saturation: 73.7%
    ngroup below 80% saturation: 97
    ngroup below 100% saturation: 122

    Signal-to-noise ratio       3484.8
    Extracted flux              2043.0  e-/s
    Flux standard deviation        0.6  e-/s
    Brightest pixel rate        1354.7  e-/s

                                   in-transit  out-transit
    Integrations:                         243      452
    Duty cycle:                          0.98     0.98
    Total exposure time:               7533.2  14012.3  s
    First--last dt per exposure:       7533.2  14012.3  s
    Reset--last dt per integration:    7366.4  13702.1  s

    Reference wavelength:                    4.36  microns
    Area of extraction aperture:             4.76  pixels
    Area of background measurement:           6.3  pixels
    Background surface brightness:            0.3  MJy/sr
    Total sky flux in background aperture:   4.45  e-/s
    Total flux in background aperture:      64.12  e-/s
    Background flux fraction from scene:     0.93
    Number of cosmic rays:      0.0072  events/pixel/read
    """
    if not isinstance(reports, list):
        reports = [reports]
    # This is a TSO dict
    if 'report_in' in reports[0]:
        report_in = [report['report_in'] for report in reports]
        report_out = [report['report_out'] for report in reports]
    # This is a perform_calculation dict
    else:
        report_in = reports
        report_out = []

    # Put everything into a list to make things easier to handle:
    if not isinstance(report_in, list):
        report_in = [report_in]
    if not isinstance(report_out, list):
        report_out = [report_out]

    # Exposure
    config = report_in[0]['input']['configuration']
    inst = config['instrument']['instrument']
    mode = config['instrument']['mode']
    subarray = config['detector']['subarray']
    readout = config['detector']['readout_pattern']
    ngroup = config['detector']['ngroup']
    nint = config['detector']['nint']
    if report_out != []:
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

    # Full report
    scalar_in = [report['scalar'] for report in report_in]
    scalar_out = [report['scalar'] for report in report_out]
    stats = _print_pandeia_stats(inst, mode, scalar_in, scalar_out, format)
    text_report = f'{text_report}\n\n{stats}'
    if format == 'html':
        text_report = text_report.replace('\n', '<br>')
    return text_report


def tso_print(calculation, format='rich'):
    """
    Print to screen a tso_calculation() output or a Pandeia's
    perform_calculation() output.

    Parameters
    ----------
    calculation: Dictionary or list of dictionaries
        A tso_calculation() or a Pandeia's perform_calculation() output.
    format: String
        If 'rich' print with colourful text when there are warnings
        or errors in values.
        If None, print as plain text.

    Examples
    --------
    >>> import gen_tso.pandeia_io as jwst
    >>> import numpy as np

    >>> wl = np.logspace(0, 2, 1000)
    >>> depth = [wl, np.tile(0.03, len(wl))]
    >>> pando = jwst.PandeiaCalculation('nircam', 'lw_tsgrism')
    >>> pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)
    >>> tso = pando.tso_calculation(
    >>>     'transit', transit_dur=2.1, obs_dur=6.0, depth_model=depth,
    >>>     ngroup=100, readout='rapid', filter='f444w',
    >>> )
    >>> # A perform_calculation() call:
    >>> jwst.tso_print(tso)

    >>> # A tso_calculation() call:
    >>> calc = pando.perform_calculation(ngroup=130, nint=300)
    >>> jwst.tso_print(calc)
    """
    # TBD: get style from style.css file?
    style = prompt_toolkit.styles.Style.from_dict({
        'danger': '#cb2222',
        'warning': '#ffa500',
    })
    report = _print_pandeia_report(calculation, format)
    if format == 'rich':
        report = prompt_toolkit.HTML(report)
    prompt_toolkit.print_formatted_text(report, style=style)

