# Copyright (c) 2025 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)
        
import textwrap

import numpy as np
from gen_tso.app_utils import (
    planet_model_name,
    parse_instrument,
    parse_obs,
    parse_sed,
)


def parse_depth_source(input, user_spectra):
    """
    Parse transit/eclipse model name based on current state.
    Calculate or extract model.
    """
    model_type = input.planet_model_type.get()
    depth_label = planet_model_name(input)
    obs_geometry = input.obs_geometry.get()

    if model_type == 'Input':
        model = user_spectra[obs_geometry][depth_label]
        filename = model['filename']
        units = model['units']
        return filename, units

    if model_type == 'Flat':
        pass
        # nwave = 1000
        # transit_depth = input.transit_depth.get() * 0.01
        # wl = np.linspace(0.6, 50.0, nwave)
        # depth = np.tile(transit_depth, nwave)

    elif model_type == 'Blackbody':
        pass
        # transit_depth = input.eclipse_depth.get() * 0.01
        # t_planet = input.teq_planet.get()
        # # Un-normalized planet and star SEDs
        # sed_type, sed_model, norm_band, norm_mag, sed_label = parse_sed(input, user_spectra)
        # star_scene = jwst.make_scene(sed_type, sed_model, norm_band='none')
        # planet_scene = jwst.make_scene('blackbody', t_planet, norm_band='none')
        # wl, f_star = jwst.extract_sed(star_scene)
        # wl_planet, f_planet = jwst.extract_sed(planet_scene)
        # # Interpolate black body at wl_star
        # interp_func = si.interp1d(
        #     wl_planet, f_planet, bounds_error=False, fill_value=0.0,
        # )
        # f_planet = interp_func(wl)
        # # Eclipse_depth = Fplanet/Fstar * rprs**2
        # depth = f_planet / f_star * transit_depth

    #return depth_label, wl, depth



def export_script_fixed_values(
        input, user_spectra, saturation_fraction, acq_target_list,
    ):
    """
    values: String
        fixed or calculated
    """
    config = parse_instrument(
        input, 'instrument', 'mode', 'aperture', 'disperser', 'filter',
        'subarray', 'readout', 'order', 'ngroup', 'nint',
        'pairing', 'pupil', 'detector',
    )
    inst, mode, aperture, disperser, filter, subarray, readout = config[0:7]
    order, ngroup, nint, pairing, pupil, detector = config[7:]

    req_saturation = saturation_fraction.get()
    name = input.target.get()
    obs_geometry = input.obs_geometry.get()
    transit_dur = float(input.t_dur.get())
    planet_model_type, depth_label, rprs_sq, teq_planet = parse_obs(input)
    print(planet_model_type)

    depth_file, units = parse_depth_source(input, user_spectra)


    target_focus = input.target_focus.get()
    if target_focus == 'acquisition':
        selected = acquisition_targets.cell_selection()['rows'][0]
        target_list = acq_target_list.get()
        target_acq_mag = np.round(target_list[1][selected], 3)
    elif target_focus == 'science':
        #in_transit_integs, in_transit_time = jwst.bin_search_exposure_time(
        #    inst, subarray, readout, ngroup, transit_dur,
        #)
        target_acq_mag = None

    sed_type, sed_model, norm_band, norm_mag, sed_label = parse_sed(
        input, user_spectra, target_acq_mag=target_acq_mag,
    )

    # WRITE SCRIPT
    script = f"""\
    import gen_tso.pandeia_io as jwst
    import gen_tso.catalogs as cat
    import gen_tso.utils as u
    import numpy as np


    # The Pandeia instrumental configuration:
    instrument = {repr(inst)}
    mode = {repr(mode)}
    pando = jwst.PandeiaCalculation(instrument, mode)

    disperser = {repr(disperser)}
    filter = {repr(filter)}
    subarray = {repr(subarray)}
    readout = {repr(readout)}
    aperture = {repr(aperture)}
    order = {repr(order)}

    ngroup = {ngroup}
    nint = {repr(nint)}

    # The star:
    sed_type = {repr(sed_type)}
    sed_model = {repr(sed_model)}
    norm_band = {repr(norm_band)}
    norm_mag = {norm_mag}
    pando.set_scene(sed_type, sed_model, norm_band, norm_mag)
"""

    if mode == 'target_acq':
        script += """\n
    # Target acquisition
    tso = pando.perform_calculation(
        ngroup, nint, disperser, filter, subarray, readout,
        aperture, order,
    )\
"""
    else:
        script += f"""\n
    # The planet:
    # Planet model: wl(um) and transit depth (no units):
    obs_type = {repr(obs_geometry)}
    filename = {repr(depth_file)}
    units = {repr(units)}
    label, wl, model = u.read_spectrum_file(filename, units)
    depth_model = [wl, depth]

    # in-transit and total observation duration times (hours):
    transit_dur = {transit_dur}
    exp_time = jwst.exposure_time(instrument, subarray, readout, ngroup, nint)
    obs_dur = exp_time / 3600.0

    # Run TSO simulation:
    tso = pando.tso_calculation(
        obs_type, transit_dur, obs_dur, depth_model,
        ngroup, disperser, filter, subarray, readout, aperture, order,
    )\
"""
    return textwrap.dedent(script)


def export_script_calculated_values(
        input, user_spectra, saturation_fraction, acq_target_list,
    ):
    """
    values: String
        fixed or calculated
    """
    config = parse_instrument(
        input, 'instrument', 'mode', 'aperture', 'disperser', 'filter',
        'subarray', 'readout', 'order', 'ngroup', 'nint',
        'pairing', 'pupil', 'detector',
    )
    inst, mode, aperture, disperser, filter, subarray, readout = config[0:7]
    order, ngroup, nint, pairing, pupil, detector = config[7:]

    req_saturation = saturation_fraction.get()
    name = input.target.get()
    obs_geometry = input.obs_geometry.get()
    transit_dur = float(input.t_dur.get())
    planet_model_type, depth_label, rprs_sq, teq_planet = parse_obs(input)
    print(planet_model_type)

    target_focus = input.target_focus.get()
    if target_focus == 'acquisition':
        selected = acquisition_targets.cell_selection()['rows'][0]
        target_list = acq_target_list.get()
        target_acq_mag = np.round(target_list[1][selected], 3)
    elif target_focus == 'science':
        #in_transit_integs, in_transit_time = jwst.bin_search_exposure_time(
        #    inst, subarray, readout, ngroup, transit_dur,
        #)
        target_acq_mag = None

    sed_type, sed_model, norm_band, norm_mag, sed_label = parse_sed(
        input, user_spectra, target_acq_mag=target_acq_mag,
    )

    # WRITE SCRIPT
    script = f"""\
    import gen_tso.pandeia_io as jwst
    import gen_tso.catalogs as cat
    import numpy as np


    # The Pandeia instrumental configuration:
    instrument = {repr(inst)}
    mode = {repr(mode)}
    pando = jwst.PandeiaCalculation(instrument, mode)

    disperser = {repr(disperser)}
    filter = {repr(filter)}
    subarray = {repr(subarray)}
    readout = {repr(readout)}
    aperture = {repr(aperture)}
    order = {repr(order)}

    # The star:
    sed_type = {repr(sed_type)}
    sed_model = {repr(sed_model)}
    norm_band = {repr(norm_band)}
    norm_mag = {norm_mag}
    pando.set_scene(sed_type, sed_model, norm_band, norm_mag)

    # Integration timings:
    ngroup = {ngroup}
    nint = {repr(nint)}
    # Automate ngroup below requested saturation fraction:
    # ngroup = pando.saturation_fraction(fraction={req_saturation:.1f})

    # Automate nint to match an observation duration:
    # nint, exp_time = jwst.bin_search_exposure_time(
    #     instrument, subarray, readout, ngroup, obs_dur,
    # )

    # Estimate obs_duration:
    # t_base = np.max([0.5*transit_dur, 1.0])
    # obs_dur = t_start + t_settling + transit_dur + 2*t_base
    obs_dur = jwst.obs_duration(transit_dur, t_base)

    # To automate target properties:
    # catalog = cat.Catalog()
    # target = catalog.get_target({repr(name)})
    # t_eff = target.teff
    # logg_star = target.logg_star
    # sed_model = jwst.find_closest_sed(teff, logg_star, sed_type={repr(sed_type)})
    # norm_band = '2mass,ks'
    # norm_mag = target.ks_mag
    # pando.set_scene(sed_type, sed_model, norm_band, norm_mag)\
"""

    if mode == 'target_acq':
        script += """\n
    # Target acquisition
    tso = pando.perform_calculation(
        ngroup, nint, disperser, filter, subarray, readout,
        aperture, order,
    )\
"""
    else:
        script += f"""\n
    # The planet:
    # Planet model: wl(um) and transit depth (no units):
    obs_type = {repr(obs_geometry)}
    spec_file = 'data/models/WASP80b_transit.dat'
    depth_model = np.loadtxt(spec_file, unpack=True)
    depth_label, wl, depth = parse_depth_model(input)
    depth_model = [wl, depth]

    # in-transit and total observation duration times (hours):
    # transit_dur = target.transit_dur
    transit_dur = {transit_dur}
    exp_time = jwst.exposure_time(instrument, subarray, readout, ngroup, nint)
    obs_dur = exp_time / 3600.0

    # Automate obs_duration:
    # t_start = 1.0
    # t_settling = 0.75
    # t_base = np.max([0.5*transit_dur, 1.0])
    # obs_dur = t_start + t_settling + transit_dur + 2*t_base

    # Run TSO simulation:
    tso = pando.tso_calculation(
        obs_type, transit_dur, obs_dur, depth_model,
        ngroup, disperser, filter, subarray, readout, aperture, order,
    )\
"""
    return textwrap.dedent(script)
