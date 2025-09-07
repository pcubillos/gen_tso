# Copyright (c) 2025 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    'read_noise_variance',
    'exposure_time',
    'bin_search_exposure_time',
    'integration_time',
    'saturation_level',
    'extract_flux_rate',
    'estimate_flux_rate',
    'groups_below_saturation',
    'get_sed_list',
    'find_closest_sed',
    'find_nearby_seds',
    'make_scene',
    'extract_sed',
    'blackbody_eclipse_depth',
    'set_depth_scene',
    'get_bandwidths',
    'save_tso',
    'simulate_tso',
    '_get_tso_wl_range',
    '_get_tso_depth_range',
    '_print_pandeia_exposure',
    '_print_pandeia_saturation',
    '_print_pandeia_stats',
    '_print_pandeia_report',
    'tso_print',
]

from collections.abc import Iterable
import copy
from decimal import Decimal
import json
import pickle
import random

import numpy as np
import scipy.interpolate as si
from scipy.interpolate import CubicSpline
import pandeia.engine.sed as sed
from pandeia.engine.calc_utils import get_instrument_config
from pandeia.engine.normalization import NormalizationFactory
from pyratbay.spectrum import constant_resolution_spectrum, bin_spectrum
from pyratbay.tools import u
import prompt_toolkit

from ..utils import format_text
from .pandeia_defaults import (
    _photo_modes,
    _load_flux_rate_splines,
    get_sed_types,
    get_throughputs,
    make_saturation_label,
)
sed_types = get_sed_types()


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
    if report_config['mode'] in ['sw_tsgrism', 'sw_ts']:
        aperture = report_config['aperture']
        noise = ins_config['detector_config']['sw']['rn']
        if aperture not in noise:
            aperture = 'default'
        read_noise = noise[aperture]
        return read_noise

    if report_config['mode'] in ['mrs_ts', 'lrsslit']:
        aperture = report_config['aperture']
        aperture = ins_config['aperture_config'][aperture]['detector']
    else:
        aperture = report_config['aperture']

    if aperture not in ins_config['detector_config']:
        aperture = 'default'

    read_noise = ins_config['detector_config'][aperture]['rn']
    if isinstance(read_noise, dict):
        read_noise = read_noise['default']

    return read_noise


# ETC-APT correction for t_exp for SOSS multi-stripe subarrays
log_ngroup = np.log10([
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 50, 100, 300, 1000, 3000,
])
stripe_subarrays = [
    'sub17stripe_soss',
    'sub60stripe_soss',
    'sub204stripe_soss',
    'sub680stripe_soss',
]
log_correction = [
    [
        -0.77956267, -0.95565393, -1.08059267, -1.17750268, -1.25668393,
        -1.32363072, -1.38162266, -1.43277519, -1.47853268, -1.51992536,
        -1.80075197, -1.96989437, -2.18610285, -2.48285405, -2.95709917,
        -3.47896676, -3.95579867,
    ],
    [
        -1.23735383, -1.41345756, -1.53840877, -1.63533126, -1.71452498,
        -1.78142187, -1.83942629, -1.89059129, -1.93636125, -1.97776641,
        -2.25859302, -2.42773542, -2.6439439 , -2.9406951 , -3.41494022,
        -3.9368078 , -4.41363972,
    ],
    [
        -1.74005413, -1.91614539, -2.04108413, -2.13799414, -2.21717539,
        -2.28412218, -2.34211412, -2.39326665, -2.43902414, -2.48041682,
        -2.76124343, -2.93038583, -3.14659431, -3.44334551, -3.91759063,
        -4.43945821, -4.91629013,
    ],
    [
        -2.25430035, -2.42996756, -2.55518895, -2.6523818 , -2.73113886,
        -2.79836839, -2.85593629, -2.90737147, -2.95341179, -2.99438029,
        -3.2752069 , -3.4443493 , -3.66055778, -3.95730898, -4.4315541 ,
        -4.95342169, -5.4302536,
    ],
]

soss_exp_correction = {
    subarray: CubicSpline(
        log_ngroup, log_correction[j], extrapolate=True,
    )
    for j,subarray in enumerate(stripe_subarrays)
}

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
        config = get_instrument_config(telescope, instrument)
    else:
        config = instrument.ins_config

    # When switching instrument, subarray and readout updates are not atomic
    if subarray not in config['subarray_config']['default']:
        return None
    if readout not in config['readout_pattern_config']:
        return None

    subarray_config = config['subarray_config']
    readout_config = config['readout_pattern_config'][readout]

    nframe = readout_config['nframe']
    ndrop2 = readout_config['ndrop2']
    tfffr = subarray_config['default'][subarray]['tfffr']
    tframe = subarray_config['default'][subarray]['tframe']
    has_tframe = (
        readout in subarray_config and
        subarray in subarray_config[readout]
    )
    if has_tframe:
        tframe = subarray_config[readout][subarray]["tframe"]
        tfffr = subarray_config[readout][subarray]["tfffr"]

    ndrop1 = ndrop3 = 0
    if "ndrop1" in readout_config:
        ndrop1 = readout_config["ndrop1"]
    if "ndrop3" in readout_config:
        ndrop3 = readout_config["ndrop3"]

    nreset1 = nreset2 = 1
    if "nreset1" in subarray_config["default"][subarray]:
        nreset1 = subarray_config["default"][subarray]["nreset1"]
        nreset2 = subarray_config["default"][subarray]["nreset2"]
    has_nreset = (
        readout in subarray_config and
        subarray in subarray_config[readout] and
        "nreset1" in subarray_config[readout][subarray]
    )
    if has_nreset:
        nreset1 = subarray_config[readout][subarray]["nreset1"]
        nreset2 = subarray_config[readout][subarray]["nreset2"]

    if "nreset1" in readout_config:
        nreset1 = readout_config["nreset1"]
        nreset2 = readout_config["nreset2"]

    # ETC correction for SOSS substripe subarrays:
    subarray_factors = {
        'sub17stripe_soss': 120.0,
        'sub60stripe_soss':  34.0,
        'sub204stripe_soss': 10.0,
        'sub680stripe_soss':  3.0,
    }
    if subarray in subarray_factors:
        corr = 1.0 + 10.0**soss_exp_correction[subarray](np.log10(ngroup))
        print(f"correction is {corr}")
        t_factor = subarray_factors[subarray] * corr
    else:
        t_factor = 1.0

    def exp_time(nint):
        time = t_factor * nexp * (
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
    Based on pandeia.engine.exposure.get_times()

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
        config = get_instrument_config(telescope, instrument)
    else:
        config = instrument.ins_config

    # When switching instrument, subarray and readout updates are not atomic
    if subarray not in config['subarray_config']['default']:
        return 0.0
    if readout not in config['readout_pattern_config']:
        return 0.0

    nframe = config['readout_pattern_config'][readout]['nframe']
    ndrop2 = config['readout_pattern_config'][readout]['ndrop2']
    tframe = config['subarray_config']['default'][subarray]['tframe']
    has_tframe = (
        readout in config['subarray_config'] and
        subarray in config['subarray_config'][readout]
    )
    if has_tframe:
        tframe = config['subarray_config'][readout][subarray]["tframe"]

    ndrop1 = 0
    if "ndrop1" in config['readout_pattern_config'][readout]:
        ndrop1 = config['readout_pattern_config'][readout]["ndrop1"]

    integration_time = tframe * (
        ndrop1 + (ngroup - 1) * (nframe + ndrop2) + nframe
    )
    return integration_time


def extract_flux_rate(reports, get_max=False):
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
    >>>
    >>> inst = 'nircam'
    >>> readout = 'rapid'
    >>> pando = jwst.PandeiaCalculation(inst, 'lw_tsgrism')
    >>> pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)
    >>> result = pando.perform_calculation(
    >>>     ngroup=2, nint=1, readout=readout, filter='f444w',
    >>> )
    >>> pixel_rate, full_well = jwst.extract_flux_rate(result)
    >>>
    >>> # Now I can calculate the saturation level for any integration time:
    >>> # (for the given filter and scene)
    >>> subarray = 'subgrism64'
    >>> for ngroup in [2, 97, 122]:
    >>>     integ_time = jwst.integration_time(inst, subarray, readout, ngroup)
    >>>     sat_level = 100 * pixel_rate * integ_time / full_well
    >>>     print(f'Sat. fraction for {ngroup:3d} groups: {sat_level:5.1f}%')
    Sat. fraction for   2 groups:   1.5%
    Sat. fraction for  97 groups:  73.9%
    Sat. fraction for 122 groups:  93.0%

    >>> # Calculate maximum number of groups before saturation
    >>> inst = 'nircam'
    >>> subarray = 'subgrism64'
    >>> readout = 'bright2'
    >>> pando = jwst.PandeiaCalculation(inst, 'lw_tsgrism')
    >>> pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)
    >>> result = pando.perform_calculation(
    >>>     ngroup=2, nint=1, readout=readout, filter='f444w',
    >>> )
    >>> pixel_rate, full_well = jwst.extract_flux_rate(result)
    >>>
    >>> # ngroup staying below 80% of saturation:
    >>> req_fraction = 80.0
    >>> dt_integ = (
    >>>     jwst.integration_time(inst, subarray, readout, ngroup=3) -
    >>>     jwst.integration_time(inst, subarray, readout, ngroup=2)
    >>> )
    >>> sat_fraction = 100 * pixel_rate * dt_integ / full_well
    >>> ngroup_req = int(req_fraction/sat_fraction)
    >>> print(f'ngroup below {req_fraction:.1f}% of saturation: {ngroup_req}')
    ngroup below 80.0% of saturation: 52
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


def saturation_level(reports, get_max=False):
    """
    Deprecated. Use `extract_flux_rate()` instead.
    """
    import warnings
    warnings.warn(
        "saturation_level() is deprecated and will be removed in a "
        "future release. Use extract_flux_rate() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return extract_flux_rate(reports, get_max)


def estimate_flux_rate(
        sed_type, sed_model, ks_mag,
        mode, aperture, disperser, filter, subarray, order='',
    ):
    """
    Estimate a detector's brightest-pixel flux rate and the full well
    for a given source and an instrumental configuration,
    based on pre-tabulated Ks-bands magnitudes.

    Parameters
    ----------
    sed_type: String
        Type of model: 'phoenix', 'k93models', or 'bt_settl'.
    sed_model:
        The SED model required for each sed_type, see
        jwst.get_sed_list(), jwst.find_closest_sed() or jwst.find_nearby_seds()
    ks_mag: float
        Magnitude of the star in the Ks band.
    mode: String
        The observing mode for the JWST instrument.
    aperture: String
        Aperture configuration for the given instrument.
    disperser: String
        Disperser/grating for the given instrument.
    filter: String
        Filter for the given instrument.
    subarray: String
        Subarray mode for the given instrument.
    order: Integer
        For NIRISS SOSS only, the spectral order.

    Returns
    -------
    brightest_pixel_rate: Float
        e- per second rate at the brightest pixel.
    full_well: Float
        Number of e- counts to saturate the detector.

    Example
    -------
    >>> import gen_tso.pandeia_io as jwst

    >>> sed_type = 'phoenix'
    >>> sed_model = 'k5v'
    >>> ks_mag = 8.0
    >>> flux_rate, full_well = jwst.estimate_flux_rate(
    >>>     sed_type, sed_model, ks_mag,
    >>>     mode='lw_tsgrism', aperture='lw',
    >>>     disperser='grism', filter='f444w', subarray='subgrism64',
    >>> )
    >>> print(flux_rate, full_well)
    1796.190365030732 58100.0
    """
    sed_label = f'{sed_type}_{sed_model}'
    obs_label = make_saturation_label(
        mode, aperture, disperser, filter, subarray, order, sed_label,
    )
    flux_rate_spline, full_well = _load_flux_rate_splines(obs_label)

    if flux_rate_spline is None:
        return None, None

    pixel_rate = 10**flux_rate_spline(ks_mag)
    return pixel_rate, full_well


def groups_below_saturation(
        req_saturation,
        instrument=None, subarray=None, readout=None,
        flux_rate=None, full_well=None,
        mode=None, aperture=None, disperser=None, filter=None, order=None,
        sed_type=None, sed_model=None, ks_mag=None,
        reports=None,
    ):
    """
    Calculate the maximum number of groups below a give saturation level
    for the given configuration.

    Parameters
    ----------
    req_saturation: Float or 1D float iterable
        Required saturation level in percent.
    instrument: String
        Which instruments (miri, nircam, niriss, or nirspec).
    subarray: String
        Subarray mode for the given instrument.
    readout: String
        Readout pattern mode for the given instrument.
    flux_rate: Float
        e- per second rate at the brightest pixel.
    full_well: Float
        Number of e- counts to saturate the detector.
    mode: String
        The observing mode for the JWST instrument.
    aperture: String
        Aperture configuration for the given instrument.
    disperser: String
        Disperser/grating for the given instrument.
    filter: String
        Filter for the given instrument.
    order: Integer
        For NIRISS SOSS only, the spectral order.
    sed_type: String
        Type of SED model: 'phoenix', 'k93models', or 'bt_settl'.
    sed_model:
        The SED model required for each sed_type, see
        jwst.get_sed_list(), jwst.find_closest_sed() or jwst.find_nearby_seds()
    ks_mag: float
        Magnitude of the star in the Ks band.
    reports: Dictionary or list of dictionaries
        Either a pandeia's perform_calculation() output dictionary
        or a tso_calculation output dictionary.

    Returns
    -------
    ngroup: Integeer
        Number of groups per integration to keep the saturation fraction
        of the brightest pixel rate below req_saturation.

    Examples
    --------
    >>> import gen_tso.pandeia_io as jwst
    >>>
    >>> # Use a pandeia report as input:
    >>> pando = jwst.PandeiaCalculation('nircam', 'lw_tsgrism')
    >>> pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)
    >>> report = pando.perform_calculation(ngroup=2, nint=1)
    >>> ngroup = jwst.groups_below_saturation([80, 100], reports=report)
    >>> print(ngroup)
    [104, 131]

    >>> # Use instrument and source variables as input:
    >>> inst = 'nircam'
    >>> mode = 'lw_tsgrism'
    >>> disperser = 'grismr'
    >>> filter = 'f444w'
    >>> subarray = 'subgrism64'
    >>> readout = 'rapid'
    >>> aperture = 'lw'
    >>>
    >>> sed_type = 'phoenix'
    >>> sed_model = 'k5v'
    >>> ks_mag = 8.351
    >>>
    >>> ngroup = jwst.groups_below_saturation(
    >>>     req_saturation=80.0,
    >>>     instrument=inst, mode=mode, disperser=disperser, filter=filter,
    >>>     subarray=subarray, readout=readout, aperture=aperture,
    >>>     sed_type=sed_type, sed_model=sed_model, ks_mag=ks_mag,
    >>>
    >>> )
    >>> print(ngroup)
    104
    """
    if isinstance(req_saturation, Iterable):
        req_saturations = req_saturation
    else:
        req_saturations = [req_saturation]

    if reports is not None:
        flux_rate, full_well = extract_flux_rate(reports, get_max=True)
        # Unpack instrumental configuration values:
        if not isinstance(reports, list):
            reports = [reports]
        if 'report_in' in reports[0]:
            report = reports[0]['report_in']
        else:
            report = reports[0]
        config = report['input']['configuration']
        instrument = config['instrument']['instrument']
        subarray = config['detector']['subarray']
        readout = config['detector']['readout_pattern']

    if flux_rate is None:
        flux_rate, full_well = estimate_flux_rate(
            sed_type, sed_model, ks_mag,
            mode, aperture, disperser, filter, subarray, order,
        )

    are_undefined = (
        instrument is None,
        subarray is None,
        readout is None,
        flux_rate is None,
        full_well is None,
    )
    if np.any(are_undefined):
        raise ValueError('Not all required inputs are defined')

    dt_1group = integration_time(instrument, subarray, readout, ngroup=1)
    dt_2group = integration_time(instrument, subarray, readout, ngroup=2)
    dt = dt_2group - dt_1group

    sat_1group = flux_rate * dt_1group / full_well
    sat_dt = flux_rate * dt / full_well

    ngroups = []
    for saturation in req_saturations:
        m = 1 + (saturation/100.0 - sat_1group) / sat_dt
        if np.isfinite(m):
            ngroups.append(int(m))
        else:
            ngroups.append(0)

    if not isinstance(req_saturation, Iterable):
        ngroups = ngroups[0]
    return ngroups


def get_sed_list(source):
    """
    Load list of available PHOENIX or Kurucz stellar SED models

    Parameters
    ----------
    source: String
        SED source: 'phoenix', 'k93models', or 'bt_settl'.

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
    >>> # Kurucz models
    >>> keys, names, teff, log_g = jwst.get_sed_list('k93models')
    >>> # PHOENIX models
    >>> keys, names, teff, log_g = jwst.get_sed_list('phoenix')
    >>> # BT-Settl models
    >>> keys, names, teff, log_g = jwst.get_sed_list('bt_settl')
    """
    sed_path = sed.default_refdata_directory
    with open(f'{sed_path}/sed/{source}/spectra.json', 'r') as f:
        info = json.load(f)
    names = np.array([model['display_string'] for model in info.values()])

    if source == 'bt_settl':
        teff = np.array([model.split()[1] for model in names], dtype=float)
        log_g = np.array([model.split()[2] for model in names], dtype=float)
    else:
        teff = np.array([model['teff'] for model in info.values()])
        log_g = np.array([model['log_g'] for model in info.values()])

    keys = np.array(list(info.keys()))
    tsort = np.argsort(teff)[::-1]

    return keys[tsort], names[tsort], teff[tsort], log_g[tsort]


def find_closest_sed(teff, logg, sed_type='phoenix'):
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
    sed_type: String
        Select from 'phoenix' or 'k93models'

    Returns
    -------
    sed: String
        The SED key that best matches the teff,logg pair.

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
    """
    keys, names, models_teff, models_logg = get_sed_list(sed_type)
    cost = (
        np.abs(np.log10(teff/models_teff)) +
        np.abs(logg-models_logg) / 15.0
    )
    idx = np.argmin(cost)
    return keys[idx]


def find_nearby_seds(teff, dt=200.0, sed_type='all'):
    """
    Show all SED models with temperatures in range teff +/- dt.
    Highlight the closest(s) one(s) to teff with asterisks.
    Return the parameters of the closest SED.

    Parameters
    ----------
    teff: float
        Target effective temperature.
    dt: float
        Temperature range around teff to explore.
    sed_type: String or iterable of strings.
        Select one from 'phoenix', 'k93models', or 'bt_settl'.
        Or, select 'all' to consider all SED types.
        Or, provide a list with the SED types to consider.

    Returns
    -------
    model: Tuple
        A tuple with the (sed type, sed key, temperature, log_g) of
        the model closes to teff  Note there can be multiple models at
        same distance from teff, only one is returned.

    Examples
    --------
    >>> import gen_tso.pandeia_io as jwst
    >>>
    >>> # Kurucz models
    >>> sed = jwst.find_nearby_seds(teff=5100.0, dt=700, sed_type='k93models')
    SED key      teff  logg  SED type
    -----------  ----------  -----------
    'k4v'        4560   4.5  'k93models'
    'g5i'        4850   1.1  'k93models'
    'k0v'        5250   4.5  'k93models'  **
    'g8v'        5570   4.5  'k93models'
    'g5v'        5770   4.5  'k93models'

    >>> # All models
    >>> sed = jwst.find_nearby_seds(teff=3600.0, dt=200, sed_type='all')
    SED key      teff  logg  SED type
    -----------  ----------  -----------
    'm3.5-3400'  3400   5.0  'bt_settl'
    'm2v'        3500   4.6  'k93models'  **
    'm2i'        3500   0.0  'phoenix'    **
    'm5v'        3500   5.0  'phoenix'    **
    'm2.5'       3500   5.0  'bt_settl'   **
    'm2v'        3500   4.5  'phoenix'    **
    'k5i'        3750   0.5  'phoenix'
    'm0v'        3750   4.5  'phoenix'
    'm0iii'      3750   1.5  'phoenix'
    'm0i'        3750   0.0  'phoenix'

    >>> # PHOENIX + BT-Settl models
    >>> sed = jwst.find_nearby_seds(
    >>>     teff=3320.0, dt=250, sed_type=['phoenix', 'bt_settl'],
    >>> )
    SED key      teff  logg  SED type
    -----------  ----------  -----------
    'm4.5-3100'  3100   5.0  'bt_settl'
    'm4.5-3200'  3200   5.0  'bt_settl'
    'm3.5-3300'  3300   5.0  'bt_settl'   **
    'm3.5-3400'  3400   5.0  'bt_settl'
    'm2.5'       3500   5.0  'bt_settl'
    'm5v'        3500   5.0  'phoenix'
    'm2i'        3500   0.0  'phoenix'
    'm2v'        3500   4.5  'phoenix'
    """
    sed_types = get_sed_types()
    if sed_type == 'any' or sed_type=='all':
        sed_type = sed_types
    elif isinstance(sed_type, str):
        sed_type = [sed_type]

    for type in sed_type:
        if type not in sed_types:
            raise ValueError(
                f'Invalid sed_type {repr(sed_type)}, select from {sed_types}'
            )

    models = []
    for sed_t in sed_type:
        keys, names, models_teff, logg = get_sed_list(sed_t)
        nmodels = len(keys)
        for i in range(nmodels):
            if np.abs(models_teff[i]-teff) <= dt:
                model = (str(sed_t), keys[i], models_teff[i], logg[i])
                models.append(model)

    temps = [model[2] for model in models]
    i_min = np.argmin(np.abs(np.array(temps)-teff))
    dt_min = np.abs(temps[i_min]-teff)
    closest_model = models[i_min]

    isort = np.argsort(temps)
    models = [models[i] for i in isort]
    print('SED key      teff  logg  SED type')
    print('-----------  ----------  -----------')
    for model in models:
         sed, key, temp, logg = model
         suff = '  **' if np.abs(temp-teff)==dt_min else ''
         print(f'{repr(str(key)):11}  {temp:4.0f}   {logg:.1f}  {repr(sed):11}{suff}')

    return closest_model


def make_scene(sed_type, sed_model, norm_band=None, norm_magnitude=None):
    """
    Create a stellar point-source scene dictionary for use in Pandeia.

    Parameters
    ----------
    sed_type: String
        Type of model:
        - 'phoenix', 'k93models',
        - 'blackbody', 'input', or 'flat'
    sed_model:
        The SED model required for each sed_type:
        - phoenix, k93models: the model key (see get_sed_list)
        - blackbody: the effective temperature (K)
        - input: dict with 'wl' and 'flux' keys containing the
          wavelength (un) and SED spectrum (mJy).
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
    if sed_type == 'kurucz':
        sed_type = 'k93models'

    sed = {'sed_type': sed_type}

    if sed_type == 'flat':
        sed['unit'] = sed_model
    elif sed_type in sed_types:
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
        if sed_model.sed_type in sed_types:
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


def blackbody_eclipse_depth(t_planet, rprs, sed_type, sed_model, return_fluxes=False):
    """
    Compute an eclipse-depth spectrum assuming a blackbody planet emission.

    Parameters
    ----------
    t_planet: Float
        Planet effective temperature (kelvin).
    rprs: Float
        Planet-to-star radius ratio.
    sed_type: String
        Type of stellar SED model: select one from:
        'phoenix', 'k93models', 'blackbody', or 'input'.
    sed_model:
        The SED model required for each sed_type:
        - for phoenix or k93models: the model key (see get_sed_list)
        - for blackbody: the effective temperature (K)
        - for input: a dict including 'wl' and 'flux' keys containing the
              wavelength (um) and SED flux spectrum (mJy).
    return_fluxes: Bool
        If True, also return the planetary and stellar fluxes

    Returns
    -------
    wl: 1D float array
        Eclipse depth's wavelength array (um).
    depth: 1D float array
        Eclipse depth spectrum.
    f_planet: 1D float array
        Planet surface flux (erg s-1 cm-2 Hz-1).
    f_star: 1D float array
        Stellar surface flux (erg s-1 cm-2 Hz-1).

    Examples
    --------
    >>> import gen_tso.pandeia_io as jwst
    >>>
    >>> t_planet = 2000.0
    >>> rprs = 0.1
    >>> sed_type = 'phoenix'
    >>> sed_model = 'k5v'
    >>> wl, depth = jwst.blackbody_eclipse_depth(t_planet, rprs, sed_type, sed_model)
    """
    # Un-normalized planet and star SEDs
    star_scene = make_scene(sed_type, sed_model, norm_band='none')
    planet_scene = make_scene('blackbody', t_planet, norm_band='none')
    wl_star, f_star = extract_sed(star_scene)
    wl_planet, f_planet = extract_sed(planet_scene)
    # Interpolate black body at wl_star
    interp_func = si.interp1d(
        wl_planet, f_planet, bounds_error=False, fill_value=np.nan,
    )
    f_planet = interp_func(wl_star)
    wl_mask = np.isfinite(f_planet)
    wl = wl_star[wl_mask]
    # Eclipse_depth = Fplanet/Fstar * rprs**2
    depth = f_planet[wl_mask] / f_star[wl_mask] * rprs**2
    if return_fluxes:
        return wl, depth, f_planet[wl_mask], f_star[wl_mask]
    return wl, depth


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


def get_bandwidths(inst, mode, aperture, filter):
    """
    Calculate passband bandwidth properties for photometry modes
    See https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-instrumentation/nircam-filters

    Returns
    -------
    wl0: Float
        Pivot wavelength (Tokunaga & Vacca 2005)
    band_width: Float
        See (Rieke et al. 2008)
    min_wl: Float
        Min wavelvelgth where the response is > 0.25 max(response)
    max_wl: Float
        Max wavelvelgth where the response is > 0.25 max(response)

    Examples
    --------
    >>> import gen_tso.pandeia_io as jwst
    >>> inst = 'miri'
    >>> mode = 'imaging_ts'
    >>> aper = 'imager'
    >>> filter = 'f560w'
    >>> pando = jwst.PandeiaCalculation(inst, mode)
    >>> wl0, bw, min_wl, max_wl = jwst.get_bandwidths(inst, mode, aper, filter)
    """
    throughputs = get_throughputs(inst=inst, mode=mode)
    passband = throughputs[aperture][filter]
    band_wl = passband['wl']
    response = passband['response']

    wl0 = np.sqrt(
        np.trapezoid(response*band_wl, band_wl) /
        np.trapezoid(response/band_wl, band_wl)
    )
    band_width = np.trapezoid(response, band_wl) / np.amax(response)

    response_mask = response > 0.25*np.amax(response)
    min_wl = np.amin(band_wl[response_mask])
    max_wl = np.amax(band_wl[response_mask])
    return wl0, band_width, min_wl, max_wl


def save_tso(filename, tso, lightweight=True):
    """
    Save a TSO output to a pickle file.

    Parameters
    ----------
    filename : String
        The path where the TSO object will be saved.
    tso : dict
        The TSO object to be saved. This should be a dictionary-like
        structure as returned by pando.tso_calculation().
    lightweight : bool
        If True, remove the '2d' and '3d' fields from 'report_out' and
        'report_in' to reduce the file size.
        The original `tso` object is not modified.
    """
    tso_copy = copy.deepcopy(tso)
    if lightweight:
        for rep in ['report_out', 'report_in']:
            report = tso_copy[rep]
            report.pop('2d', None)
            report.pop('3d', None)

    with open(filename, 'wb') as handle:
        pickle.dump(tso_copy, handle, protocol=4)


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
        Wavelengths of binned transit/eclipse spectrum.
    bin_spec: 1D array
        Binned simulated transit/eclipse spectrum.
    bin_err: 1D array
        Uncertainties of bin_spec.
    bin_widths: 1D or 2D array
        For spectra, the 1D bin widths of bin_wl.
        For photometry, an array of shape [1,2] with the (lower,upper)
        widths of the passband relative to bin_wl.

    Examples
    --------
    >>> import gen_tso.pandeia_io as jwst
    >>> import matplotlib.pyplot as plt
    >>>
    >>> # A NIRSpec/BOTS simulation
    >>> pando = jwst.PandeiaCalculation('nirspec', 'bots')
    >>> pando.set_scene(
    >>>     sed_type='phoenix', sed_model='k5v',
    >>>     norm_band='2mass,ks', norm_magnitude=8.351,
    >>> )
    >>> disperser='g395h'
    >>> filter='f290lp'
    >>> subarray='sub2048'
    >>> readout='nrsrapid'
    >>> pando.set_config(disperser, filter, subarray, readout)
    >>> ngroup = pando.saturation_fraction(fraction=80.0)
    >>>
    >>> transit_dur = 2.71
    >>> obs_dur = 6.0
    >>> obs_type = 'transit'
    >>> depth_model = np.loadtxt('WASP80b_transit.dat', unpack=True)
    >>>
    >>> tso = pando.tso_calculation(
    >>>     obs_type, transit_dur, obs_dur, depth_model, ngroup,
    >>> )

    >>> bin_wl, bin_spec, bin_err, bin_widths = jwst.simulate_tso(
    >>>    tso, n_obs=1, resolution=300.0, noiseless=False,
    >>> )
    >>>
    >>> plt.figure(10)
    >>> plt.clf()
    >>> plt.plot(depth_model[0], depth_model[1], c='salmon')
    >>> plt.errorbar(bin_wl, bin_spec, bin_err, fmt='o', c='xkcd:blue', ms=4, mfc='w')
    >>> plt.xlim(2.85, 5.2)
    """
    dt_in = tso['time_in']
    dt_out = tso['time_out']
    flux_in = tso['flux_in'] * n_obs
    flux_out = tso['flux_out'] * n_obs
    var_in = tso['var_in'] * n_obs
    var_out = tso['var_out'] * n_obs
    wl = tso['wl']

    # Photometry
    if len(wl) == 1:
        bin_in = flux_in
        bin_out = flux_out
        bin_vin = var_in
        bin_vout = var_out
        # get throughput's bandwidth
        config = tso['report_in']['input']['configuration']['instrument']
        inst = config['instrument']
        mode = config['mode']
        aperture = config['aperture']
        filter = config['filter']
        wl0, bw, min_wl, max_wl = get_bandwidths(inst, mode, aperture, filter)
        bin_wl = wl
        bin_widths = np.array([wl-min_wl, max_wl-wl]).T

    # Spectroscopy binning
    else:
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


def _get_tso_wl_range(tso_run):
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

    min_wl = np.zeros(len(runs))
    max_wl = np.zeros(len(runs))
    for i,tso in enumerate(runs):
        config = tso['report_in']['input']['configuration']['instrument']
        inst = config['instrument']
        mode = config['mode']
        aper = config['aperture']
        filter = config['filter']
        if mode in _photo_modes:
            wl0, bw, wl_min, wl_max = get_bandwidths(inst, mode, aper, filter)
            dwl_lo = wl0 - wl_min
            dwl_hi = wl_max - wl0
            if inst == 'nircam' and 'w' in filter:
                ndw = 3
            else:
                ndw = 4
            min_wl[i] = wl0 - ndw*dwl_lo
            max_wl[i] = wl0 + ndw*dwl_hi
        else:
            min_wl[i] = np.amin(tso['wl'])
            max_wl[i] = np.amax(tso['wl'])

    min_wl = np.amin(min_wl)
    max_wl = np.amax(max_wl)

    # 5% margin
    d_wl = 0.025 * (max_wl-min_wl)
    min_wl = np.round(min_wl-d_wl, decimals=2)
    max_wl = np.round(max_wl+d_wl, decimals=2)
    return min_wl, max_wl


def _get_tso_depth_range(tso_run, resolution, units):
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

    min_wl, max_wl = _get_tso_wl_range(tso_run)
    max_depth = []
    min_depth = []
    for tso in runs:
        bin_wl, bin_spec, bin_err, widths = simulate_tso(
            tso, resolution=resolution, noiseless=True,
        )
        err_median = np.median(bin_err)
        d_min1 = np.amin(tso['depth_spectrum'] - 3*err_median)
        d_max1 = np.amax(tso['depth_spectrum'] + 3*err_median)

        mode = tso['report_in']['input']['configuration']['instrument']['mode']
        if mode in _photo_modes:
            input_wl, input_depth = tso['input_depth']
            wl_min = np.amax([min_wl, np.amin(input_wl)])
            wl_max = np.amin([max_wl, np.amax(input_wl)])
            wl = constant_resolution_spectrum(wl_min, wl_max, resolution)
            depth = bin_spectrum(wl, input_wl, input_depth, gaps='interpolate')
            d_min2 = np.amin(depth)
            d_max2 = np.amax(depth)
        else:
            d_min2 = np.inf
            d_max2 = 0.0

        min_depth.append(np.amin([d_min1, d_min2]))
        max_depth.append(np.amax([d_max1, d_max2]))

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
        req_saturation=80.0,
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
    >>> pixel_rate, full_well = jwst.extract_flux_rate(tso, get_max=True)
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
        pixel_rate, full_well = extract_flux_rate(reports, get_max=True)
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
    sat_fraction = 100.0 * pixel_rate * sat_time / full_well
    saturation = format_text(
        f"{sat_fraction:.1f}%",
        np.round(sat_fraction, decimals=1)>np.round(req_saturation, decimals=1),
        sat_fraction>=100,
        format,
    )

    ngroup_req, ngroup_max = groups_below_saturation(
        [req_saturation, 100.0],
        inst, subarray, readout, pixel_rate, full_well,
    )

    ngroup_req = format_text(
        f"{ngroup_req:d}", ngroup_req==2, ngroup_req<2, format,
    )
    ngroup_max = format_text(
        f"{ngroup_max:d}", ngroup_max==2, ngroup_max<2, format,
    )
    sat_text = (
        f'Max fraction of saturation: {saturation}\n'
        f'ngroup below {req_saturation:.0f}% saturation: {ngroup_req}\n'
        f'ngroup below 100% saturation: {ngroup_max}'
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
    pixel_rate, full_well = extract_flux_rate(reports, get_max=True)
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

