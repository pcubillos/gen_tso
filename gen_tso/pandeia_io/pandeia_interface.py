# Copyright (c) 2024 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    'exposure_time',
    'saturation_time',
    'load_sed_list',
    'find_closest_sed',
    'extract_sed',
    'make_scene',
    'set_depth_scene',
    'simulate_tso',
    'PandeiaCalculation',
    'generate_all_instruments',
]

import copy
import json
from dataclasses import dataclass
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


def exposure_time(
        instrument, calculation=None,
        nexp=1, nint=None, ngroup=None, readout=None, subarray=None,
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


def make_scene(sed_type, sed_model, norm_band, norm_magnitude):
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
    >>> import gen_tso.pandeia as jwst

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
    >>> import gen_tso.pandeia as jwst
    >>> import matplotlib.pyplot as plt

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
    >>>     ngroup, filter, readout, subarray, disperser,
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


    def set_scene(self, sed_type, sed_model, norm_band, norm_magnitude):
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

    def get_saturation_values(self, filter, readout, subarray, disperser):
        """
        Calculate the brightest-pixel rate (e-/s) and full_well (e-)
        for the current instrument and scene configuration, which once known,
        are sufficient to calculate the saturation level once the
        saturation  time is known.

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
        >>> brightest_pixel_rate, full_well = pando.get_saturation_values(
        >>>     disperser='grismr', filter='f444w',
        >>>     readout='rapid', subarray='subgrism64',
        >>> )
        """
        saturation_run = self.perform_calculation(
            nint=1, ngroup=2,
            readout=readout, subarray=subarray,
            disperser=disperser,
            filter=filter,
        )
        pando_results = saturation_run['scalar']
        brightest_pixel_rate = pando_results['brightest_pixel']

        full_well = (
            brightest_pixel_rate
            * pando_results['saturation_time']
            / pando_results['fraction_saturation']
        )
        return brightest_pixel_rate, full_well


    def perform_calculation(
            self, nint, ngroup,
            filter=None, readout=None, subarray=None, disperser=None,
        ):
        """
        Run pandeia.
        """
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

    def calc_noise(self, obs_dur, ngroup, readout, subarray, disperser, filter):
        """
        Run a Pandeia calculation and extract the observed wavelength,
        flux, and variances.

        Parameters
        ----------
        obs_dur: Float
            Duration of the observation.
        ngroup: Integer
            Number of groups per integrations
        filter: String
        readout: String
        subarray: String
        disperser: String

        Returns
        -------
        TBD

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
        >>> # Example TBD
        """
        ins_config = get_instrument_config(self.telescope, self.instrument)
        single_exp_time = exposure_time(
            self.instrument, nint=1, ngroup=ngroup,
            readout=readout, subarray=subarray,
        )

        nint = int(obs_dur*3600/single_exp_time)
        report = self.perform_calculation(
            nint, ngroup, filter, readout, subarray, disperser,
        )

        # Flux:
        measurement_time = report['scalar']['measurement_time']
        flux = report['1d']['extracted_flux'][1] * measurement_time
        wl = report['1d']['extracted_flux'][0]

        # Background variance:
        background_var = report['1d']['extracted_bg_only'][1] * measurement_time

        # Read noise variance:
        if self.calc['configuration']['instrument']['mode'] == 'bots':
            read_noise = ins_config['detector_config']['default']['rn']['default']
        else:
            aperture = self.calc['configuration']['instrument']['aperture']
            read_noise = ins_config['detector_config'][aperture]['rn']
        npix = report['scalar']['extraction_area']
        read_noise_var = 2.0 * read_noise**2.0 * nint * npix

        # Pandeia (multiaccum) noise:
        shot_var = (report['1d']['extracted_noise'][1] * measurement_time)**2.0

        # Last-minus-first (LMF) noise:
        lmf_var = np.abs(flux) + background_var + read_noise_var

        return (
            report,
            wl, flux,
            lmf_var, shot_var, background_var, read_noise_var,
            measurement_time,
        )


    def tso_calculation(
            self, obs_type, transit_dur, obs_dur, depth_model,
            ngroup, filter, readout, subarray, disperser,
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
        filter: String
        readout: String
        subarray: String
        disperser: String

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
        >>>     ngroup, filter, readout, subarray, disperser,
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
            in_transit_scene = star_scene
            out_transit_scene = depth_scene
        elif obs_type == 'transit':
            in_transit_scene = depth_scene
            out_transit_scene = star_scene

        # Compute observed fluxes and noises:
        self.calc['scene'][0] = in_transit_scene
        in_transit = self.calc_noise(
            transit_dur, ngroup, readout, subarray, disperser, filter,
        )
        self.calc['scene'][0] = out_transit_scene
        out_transit_dur = obs_dur - transit_dur
        out_transit = self.calc_noise(
            out_transit_dur, ngroup, readout, subarray, disperser, filter,
        )
        # report, wl, flux, lmf_var, shot_var, bkg_var, read_var, dt
        self.calc['scene'][0] = scene

        mask = in_transit[2] > 1e-6 * np.median(in_transit[2])
        wl = in_transit[1][mask]
        flux_in = in_transit[2][mask]
        flux_out = out_transit[2][mask]
        var_in = in_transit[3][mask]
        var_out = out_transit[3][mask]
        dt_in = in_transit[7]
        dt_out = out_transit[7]
        obs_depth = 1 - (flux_in/dt_in) / (flux_out/dt_out)

        self.tso = {
            'wl': wl,
            'depth_spectrum': obs_depth,
            'time_in': dt_in,
            'flux_in': flux_in,
            'var_in': var_in,
            'time_out': dt_out,
            'flux_out': flux_out,
            'var_out': var_out,
        }
        return self.tso

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
    cal = PandeiaCalculation(instrument, mode)
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
    cal = PandeiaCalculation(instrument, mode)
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
    cal = PandeiaCalculation(instrument, mode)
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
    cal = PandeiaCalculation(instrument, mode)
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

