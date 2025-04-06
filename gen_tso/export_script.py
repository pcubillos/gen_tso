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


warning_template = (
    "# NOTE! Need to set the path to the FILE.\n    "
    "# (browsers don't expose paths of uploaded files for security reasons)\n    "
)


def parse_depth_source(input, spectra):
    """
    Parse transit/eclipse model name based on current state.
    Calculate or extract model.
    """
    model_type = input.planet_model_type.get()
    depth_label = planet_model_name(input)
    obs_geometry = input.obs_geometry.get()

    transit_depth_script = ""

    if model_type == 'Input':
        model = spectra[obs_geometry][depth_label]
        filename = model['filename']
        units = model['units']

        warning = ""
        if 'unknown_' in filename:
            filename = filename.replace('unknown_', '')
            warning = warning_template.replace('FILE', f"{obs_geometry} 'depth_file'")

        transit_depth_script = f"""
    # The planet's {obs_geometry} spectrum:
    {warning}obs_type = {repr(obs_geometry)}
    units = {repr(units)}
    depth_file = {repr(filename)}
    label, wl, depth = u.read_spectrum_file(depth_file, units)
    depth_model = [wl, depth]"""

        return transit_depth_script

    if model_type == 'Flat':
        transit_depth = input.transit_depth.get() * 0.01
        transit_depth_script = f"""
    # The planet's {obs_geometry} spectrum:
    obs_type = {repr(obs_geometry)}
    wl = ps.constant_resolution_spectrum(0.1, 50.0, resolution=300)
    nwave = len(wl)
    depth = np.tile({transit_depth:.4e}, nwave)
    depth_model = [wl, depth]"""

        return transit_depth_script

    elif model_type == 'Blackbody':
        rprs = np.sqrt(input.eclipse_depth.get() * 0.01)
        t_planet = input.teq_planet.get()
        transit_depth_script = f"""
    # The planet's {obs_geometry} spectrum:
    t_planet = {t_planet:.1f}
    rprs = {rprs:.5f}
    wl, depth = jwst.blackbody_eclipse_depth(t_planet, rprs, sed_type, sed_model)
    depth_model = [wl, depth]"""

        return transit_depth_script



def export_script_fixed_values(
        input, spectra, saturation_fraction,
        acquisition_targets, acq_target_list,
    ):
    """Translate gen_tso's current app state to a python script"""
    config = parse_instrument(
        input, 'instrument', 'mode', 'aperture', 'disperser', 'filter',
        'subarray', 'readout', 'order', 'ngroup', 'nint',
        'pairing', 'pupil', 'detector',
    )
    inst, mode, aperture, disperser, filter, subarray, readout = config[0:7]
    order, ngroup, nint, pairing, pupil, detector = config[7:]

    name = input.target.get()
    target_focus = input.target_focus.get()
    if target_focus == 'acquisition':
        selected = acquisition_targets.cell_selection()['rows'][0]
        target_list = acq_target_list.get()
        target_acq_mag = np.round(target_list[1][selected], 3)
        name = target_list[0][selected]
    elif target_focus == 'science':
        target_acq_mag = None

    sed_type, sed_model, norm_band, norm_mag, sed_label = parse_sed(
        input, spectra, target_acq_mag=target_acq_mag,
    )
    sed_warning = ""
    if sed_type == 'input':
        sed_units = sed_model['units']
        sed_file = sed_model['filename']
        if 'unknown_' in sed_file:
            sed_file = sed_file.replace('unknown_', '')
            sed_warning = warning_template.replace('FILE', "SED's 'sed_file'")

        sed_script = f"""
    sed_units = {repr(sed_units)}
    sed_file = {repr(sed_file)}
    label, sed_wl, flux = u.read_spectrum_file(sed_file, sed_units)
    sed_model = {{'wl': sed_wl, 'flux': flux}}
    """.strip()
    elif sed_type == 'blackbody':
        t_eff = input.t_eff.get()
        sed_script = f"sed_model = {t_eff}"
    else:
        sed_script = f"sed_model = {repr(sed_model)}"

    # Write the script
    script = f"""
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

    # The star ({name}):
    {sed_warning}sed_type = {repr(sed_type)}
    {sed_script}
    norm_band = {repr(norm_band)}
    norm_mag = {norm_mag}
    pando.set_scene(sed_type, sed_model, norm_band, norm_mag)\
"""

    if mode == 'target_acq':
        script += """\n
    # Run target acquisition
    tso = pando.perform_calculation(
        ngroup, nint, disperser, filter, subarray, readout,
        aperture, order,
    )\
"""
    else:
        # Transit depth
        transit_depth_script = parse_depth_source(input, spectra)
        transit_dur = float(input.t_dur.get())
        script += f"""
    {transit_depth_script}

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

    imports = """\
    import gen_tso.pandeia_io as jwst
    import gen_tso.utils as u
"""

    if 'np.' in script:
        imports += "    import numpy as np\n"
    if 'ps.' in script:
        imports += "    import pyratbay.spectrum as ps\n"
    script = imports + script
    return textwrap.dedent(script)


def export_script_calculated_values(
        input, spectra, saturation_fraction,
        acquisition_targets, acq_target_list,
    ):
    """Translate gen_tso's current app state to a python script"""
    config = parse_instrument(
        input, 'instrument', 'mode', 'aperture', 'disperser', 'filter',
        'subarray', 'readout', 'order', 'ngroup', 'nint',
        'pairing', 'pupil', 'detector',
    )
    inst, mode, aperture, disperser, filter, subarray, readout = config[0:7]
    order, ngroup, nint, pairing, pupil, detector = config[7:]

    name = input.target.get()
    target_focus = input.target_focus.get()
    if target_focus == 'acquisition':
        selected = acquisition_targets.cell_selection()['rows'][0]
        target_list = acq_target_list.get()
        target_acq_mag = np.round(target_list[1][selected], 3)
        name = target_list[0][selected]
    elif target_focus == 'science':
        #in_transit_integs, in_transit_time = jwst.bin_search_exposure_time(
        #    inst, subarray, readout, ngroup, transit_dur,
        #)
        target_acq_mag = None

    req_saturation = saturation_fraction.get()
    obs_geometry = input.obs_geometry.get()
    transit_dur = float(input.t_dur.get())
    planet_model_type, depth_label, rprs_sq, teq_planet = parse_obs(input)

    sed_type, sed_model, norm_band, norm_mag, sed_label = parse_sed(
        input, spectra, target_acq_mag=target_acq_mag,
    )
    sed_warning = ""
    if sed_type == 'input':
        sed_units = sed_model['units']
        sed_file = sed_model['filename']
        if 'unknown_' in sed_file:
            sed_file = sed_file.replace('unknown_', '')
            sed_warning = warning_template.replace('FILE', "SED's 'sed_file'")

        sed_script = f"""
    sed_units = {repr(sed_units)}
    sed_file = {repr(sed_file)}
    label, sed_wl, flux = u.read_spectrum_file(sed_file, sed_units)
    sed_model = {{'wl': sed_wl, 'flux': flux}}
    """.strip()
    elif sed_type == 'blackbody':
        t_eff = input.t_eff.get()
        sed_script = f"sed_model = {t_eff}"
    else:
        sed_script = f"sed_model = {repr(sed_model)}"

    # Write the script
    script = f"""
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

    # The target ({name}):
    catalog = cat.Catalog()
    target = catalog.get_target({repr(name)})
    t_eff = target.teff
    logg_star = target.logg_star

    sed_type = {repr(sed_type)}
    sed_model = jwst.find_closest_sed(teff, logg_star, sed_type)
    norm_band = '2mass,ks'
    norm_mag = target.ks_mag
    pando.set_scene(sed_type, sed_model, norm_band, norm_mag)

    # Set ngroup below requested saturation fraction:
    ngroup = {ngroup}
    ngroup = pando.saturation_fraction(fraction={req_saturation:.1f})

    # Estimate total duration of observation:
    transit_dur = target.transit_dur
    t_start = 1.0
    t_settling = 0.75
    t_base = np.max([0.5*transit_dur, 1.0])
    total_duration = t_start + t_settling + transit_dur + 2*t_base

    # Set nint to match an observation duration:
    nint = {nint}
    nint, exp_time = jwst.bin_search_exposure_time(
        instrument, subarray, readout, ngroup, total_duration,
    )
    # Observation duration times (hours):
    exp_time = jwst.exposure_time(instrument, subarray, readout, ngroup, nint)
    obs_dur = exp_time / 3600.0
"""

    if mode == 'target_acq':
        script += """\n
    nint = 1

    # Target acquisition
    tso = pando.perform_calculation(
        ngroup, nint, disperser, filter, subarray, readout,
        aperture, order,
    )\
"""
    else:
        # Transit depth
        transit_depth_script = parse_depth_source(input, spectra)
        transit_dur = float(input.t_dur.get())
        script += f"""
    {transit_depth_script}

    # Run TSO simulation:
    tso = pando.tso_calculation(
        obs_type, transit_dur, obs_dur, depth_model,
        ngroup, disperser, filter, subarray, readout, aperture, order,
    )\
"""

    imports = """\
    import gen_tso.pandeia_io as jwst
    import gen_tso.catalogs as cat
    import gen_tso.utils as u
"""

    if 'np.' in script:
        imports += "    import numpy as np\n"
    if 'ps.' in script:
        imports += "    import pyratbay.spectrum as ps\n"
    script = imports + script

    return textwrap.dedent(script)
