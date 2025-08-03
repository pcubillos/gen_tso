# Copyright (c) 2025 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

import numpy as np
import pyratbay.spectrum as ps
from gen_tso import pandeia_io as jwst
from gen_tso.pandeia_io.pandeia_defaults import (
    _load_flux_rate_splines,
    get_detector,
    get_sed_types,
    make_saturation_label,
)


detectors = jwst.generate_all_instruments()
throughputs = jwst.get_throughputs()

# Pre-computed flux rates
flux_rate_splines, full_wells = _load_flux_rate_splines()

# Catalog of stellar SEDs:
sed_dict = {}
for sed_type in get_sed_types():
    sed_keys, sed_models, _, _ = jwst.get_sed_list(sed_type)
    sed_dict[sed_type] = {
        key: model
        for key,model in zip(sed_keys, sed_models)
    }


bands_dict = {
    '2mass,j': 'J mag',
    '2mass,h': 'H mag',
    '2mass,ks': 'Ks mag',
    'gaia,g': 'Gaia mag',
    'johnson,v': 'V mag',
}


def get_throughput(input, evaluate=False):
    config = parse_instrument(
        input, 'instrument', 'mode',
        'aperture', 'disperser', 'filter', 'subarray', 'detector',
    )
    if config is None:
        return None
    inst, mode, aperture, disperser, filter, subarray, detector = config
    obs_type = detector.obs_type

    key = aperture if obs_type == 'photometry' else subarray
    if key not in throughputs[obs_type][inst][mode]:
        return None

    if mode in ['lrsslitless', 'lrsslit']:
        filter = 'None'
    elif mode == 'mrs_ts':
        filter = disperser
    elif mode == 'bots':
        filter = f'{disperser}/{filter}'

    if evaluate:
        return throughputs[obs_type][inst][mode][key][filter]
    config = inst, mode, key, filter
    return throughputs[obs_type], config


def get_auto_sed(input):
    """
    Guess the model closest to the available options given a T_eff
    and log_g pair.
    """
    sed_type = input.sed_type()
    sed_models = sed_dict[sed_type]

    try:
        t_eff = float(input.t_eff.get())
        log_g = float(input.log_g.get())
    except ValueError:
        return sed_models, None
    chosen_sed = jwst.find_closest_sed(t_eff, log_g, sed_type)
    return sed_models, chosen_sed


def get_saturation_values(
        mode, aperture, disperser, filter, subarray, order,
        sed_label, norm_mag,
        cache_saturation,
    ):
    """
    Get pixel_rate and full_well from instrumental settings.
    """
    sat_label = make_saturation_label(
        mode, aperture, disperser, filter, subarray, order, sed_label,
    )

    sed_items = sat_label.split('_')
    band_label = sed_items[-1]
    sat_guess_label = '_'.join(sed_items[0:-2])
    can_guess = band_label == 'Ks' and sat_guess_label in flux_rate_splines
    pixel_rate = None
    full_well = None
    if sat_label in cache_saturation:
        pixel_rate = cache_saturation[sat_label]['brightest_pixel_rate']
        full_well = cache_saturation[sat_label]['full_well']
    elif can_guess:
        cs = flux_rate_splines[sat_guess_label]
        pixel_rate = 10**cs(norm_mag)
        full_well = full_wells[sat_guess_label]
    return pixel_rate, full_well


def draw(tso_list, resolution, n_obs):
    """
    Draw a random noised-up transit/eclipse depth realization from a TSO
    """
    if not isinstance(tso_list, list):
        tso_list = [tso_list]

    sims = []
    for tso in tso_list:
        bin_wl, bin_spec, bin_err, wl_widths = jwst.simulate_tso(
           tso, n_obs=n_obs, resolution=resolution, noiseless=False,
        )
        sims.append({
            'wl': bin_wl,
            'depth': bin_spec,
            'uncert': bin_err,
            'wl_widths': wl_widths,
        })
    return sims


def planet_model_name(input):
    """
    Get the planet model name based on the transit/eclipse depth values.

    Returns
    -------
    depth_label: String
        A string representation of the depth model.
    """
    planet_model_type = input.planet_model_type.get()
    if planet_model_type == 'Input':
        return input.depth.get()
    elif planet_model_type == 'Flat':
        transit_depth = input.transit_depth.get()
        return f'Flat transit ({transit_depth:.3f}%)'
    elif planet_model_type == 'Blackbody':
        eclipse_depth = input.eclipse_depth.get()
        t_planet = input.teq_planet.get()
        return f'Blackbody({t_planet:.0f}K, rprs\u00b2={eclipse_depth:.3f}%)'


def parse_instrument(input, *args):
    """
    Parse instrumental configuration from front-end to back-end.
    Ensure that only the requested parameters are a valid configuration.
    """
    # instrument and mode always checked
    inst = input.instrument.get().lower()
    mode = input.mode.get()
    detector = get_detector(inst, mode, detectors)
    if detector is None:
        return None

    config = {
        'instrument': inst,
        'mode': mode,
        'detector': detector,
    }

    if 'aperture' in args:
        aperture = input.aperture.get()
        has_pupils = mode in ['lw_ts', 'sw_ts']
        if has_pupils and aperture not in detector.pupils:
            return None
        if not has_pupils and aperture not in detector.apertures:
            return None
        if has_pupils:
            aperture = detector.pupil_to_aperture[aperture]
        config['aperture'] = aperture

    if 'disperser' in args:
        disperser = input.disperser.get()
        if disperser not in detector.dispersers:
            return None
        config['disperser'] = disperser

    if 'filter' in args:
        filter = input.filter.get()
        if filter not in detector.filters:
            return None
        config['filter'] = filter

    if 'subarray' in args:
        subarray = input.subarray.get()
        if subarray not in detector.subarrays:
            return None
        config['subarray'] = subarray

    if 'readout' in args:
        readout = input.readout.get()
        if readout not in detector.readouts:
            return None
        config['readout'] = readout

    # Now parse front-end to back-end:
    if 'pairing' in args:
        if mode == 'sw_ts':
            config['pairing'] = input.pairing.get()
        else:
            config['pairing'] = None

    if 'pupil' in args:
        config['pupil'] = input.aperture.get()

    if 'ngroup' in args:
        if mode == 'target_acq':
            ngroup = int(input.ngroup_acq.get())
            config['disperser'] = None
        else:
            ngroup = input.ngroup.get()
        config['ngroup'] = ngroup

    config['nint'] = 1 if mode == 'target_acq' else input.integrations.get()

    if mode == 'mrs_ts':
        config['aperture'] = ['ch1', 'ch2', 'ch3', 'ch4']

    if mode == 'bots' and ('disperser' in args or 'filter' in args):
        if 'filter' not in args:
            filter = input.filter.get()
        config['disperser'], config['filter'] = filter.split('/')

    if 'order' in args:
        if mode == 'soss':
            if filter == 'f277w':
                order = [1]
            else:
                order = input.order.get()
                order = [int(val) for val in order.split()]
        else:
            order = None
        config['order'] = order

    # Return in the same order as requested
    config_list = [config[arg] for arg in args]

    return config_list


def parse_depth_model(input, spectra):
    """
    Parse transit/eclipse model name based on current state.
    Calculate or extract model.
    """
    model_type = input.planet_model_type.get()
    depth_label = planet_model_name(input)
    obs_geometry = input.obs_geometry.get()

    if model_type == 'Input':
        if depth_label is None:
            wl, depth = None, None
        else:
            wl = spectra[obs_geometry][depth_label]['wl']
            depth = spectra[obs_geometry][depth_label]['depth']
    elif model_type == 'Flat':
        transit_depth = input.transit_depth.get() * 0.01
        wl = ps.constant_resolution_spectrum(0.1, 50.0, resolution=300)
        nwave = len(wl)
        depth = np.tile(transit_depth, nwave)
    elif model_type == 'Blackbody':
        rprs = np.sqrt(input.eclipse_depth.get() * 0.01)
        t_planet = input.teq_planet.get()
        sed_type, sed = parse_sed(input, spectra)[0:2]
        wl, depth = jwst.blackbody_eclipse_depth(t_planet, rprs, sed_type, sed)

    return depth_label, wl, depth


def parse_obs(input):
    planet_model_type = input.planet_model_type.get()
    depth_model = None
    rprs_sq = None
    teq_planet = None
    if planet_model_type == 'Input':
        depth_model = input.depth.get()
    elif planet_model_type == 'Flat':
        rprs_sq = input.transit_depth.get()
    elif planet_model_type == 'Blackbody':
        rprs_sq = input.eclipse_depth.get()
        teq_planet = input.teq_planet.get()
    return planet_model_type, depth_model, rprs_sq, teq_planet


def parse_sed(input, spectra, target_acq_mag=None):
    """Extract SED parameters"""
    if target_acq_mag is None:
        sed_type = input.sed_type()
        norm_band = input.magnitude_band.get()
        norm_magnitude = float(input.magnitude.get())
    else:
        sed_type = 'phoenix'
        norm_band = 'gaia,g'
        norm_magnitude = target_acq_mag

    if sed_type in sed_dict:
        if target_acq_mag is None:
            sed_model = input.sed.get()
        else:
            sed_model = input.ta_sed.get()
        if sed_model not in sed_dict[sed_type]:
            return None, None, None, None, None
        model_label = f'{sed_type}_{sed_model}'
    elif sed_type == 'blackbody':
        sed_model = float(input.t_eff.get())
        model_label = f'bb_{sed_model:.0f}K'
    elif sed_type == 'input':
        model_label = input.sed.get()
        if model_label not in spectra['sed']:
            return None, None, None, None, None
        sed_model = spectra['sed'][model_label]

    # Make a label
    band_name = bands_dict[norm_band].split()[0]
    band_label = f'{norm_magnitude:.2f}_{band_name}'
    sed_label = f'{model_label}_{band_label}'

    return sed_type, sed_model, norm_band, norm_magnitude, sed_label

