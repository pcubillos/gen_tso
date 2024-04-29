# Copyright (c) 2024 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    'read_noise_variance',
    'exposure_time',
    'saturation_time',
    'load_sed_list',
    'find_closest_sed',
    'extract_sed',
    'make_scene',
    'set_depth_scene',
    'simulate_tso',
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

from ..utils import constant_resolution_spectrum


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


def exposure_time(
        instrument, subarray, readout,
        ngroup=None, nint=None, nexp=1,
    ):
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
    >>> subarray = 'full'
    >>> readout = 'rapid'
    >>> nint = 1
    >>> ngroup = 20
    >>> exp_time = jwst.exposure_time(inst, subarray, readout, ngroup, nint)
    >>> print(exp_time)
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
    >>> import pandeia_interface as jwst
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


class PandeiaCalculation():
    def __init__(self, instrument, mode):
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
            aperture=None,
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
        if not isinstance(reports, list):
            reports = [reports]

        ncalc = len(reports)
        brightest_pixel_rate = np.zeros(ncalc)
        full_well = np.zeros(ncalc)
        for i,report in enumerate(reports):
            results = report['scalar']
            brightest_pixel_rate[i] = results['brightest_pixel']
            full_well[i] = (
                brightest_pixel_rate[i]
                * results['saturation_time']
                / results['fraction_saturation']
            )
        if len(reports) == 1:
            return brightest_pixel_rate[0], full_well[0]
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
            disperser, filter, subarray, readout, aperture,
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
            disperser, filter, subarray, readout, aperture,
        )
        var_lmf_in = variances_in[0]

        out_transit_dur = obs_dur - transit_dur
        self.calc['scene'][0] = scene_out
        report_out, wl, flux_out, variances_out, time_out = self.calc_noise(
            out_transit_dur, ngroup,
            disperser, filter, subarray, readout, aperture,
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

