# Copyright (c) 2025 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

import textwrap

import numpy as np
import gen_tso.pandeia_io as jwst
from gen_tso.app_utils import (
    planet_model_name,
    parse_instrument,
    parse_obs,
    parse_sed,
)
from gen_tso.pandeia_io.pandeia_defaults import (
    make_save_label,
)


warning_template = (
    "# NOTE! Need to set the path to the FILE.\n    "
    "# (browsers don't expose paths of uploaded files for security reasons)\n    "
)


def parse_depth_source(
        input, spectra, teq_text=None, rprs_text=None, depth_text=None,
    ):
    """
    Parse transit/eclipse model name based on current state.
    Calculate or extract model.
    """
    model_type = input.planet_model_type.get()
    depth_label = planet_model_name(input)
    obs_geometry = input.obs_geometry.get()

    transit_depth_script = ""

    if model_type == 'Input':
        if depth_label is None:
            alter = 'flat transit' if obs_geometry=='transit' else 'blackbody'
            return  f"""
    # NOTE!  Need to upload a {obs_geometry}-depth spectrum to simulate a TSO
    # or alternatively select a {alter} model
"""
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
        if depth_text is None:
            depth = input.transit_depth.get() * 0.01
            depth_text = f'{depth:.5e}'
        transit_depth_script = f"""
    # The planet's {obs_geometry} spectrum:
    obs_type = {repr(obs_geometry)}
    wl = ps.constant_resolution_spectrum(0.1, 50.0, resolution=300)
    nwave = len(wl)
    depth = np.tile({depth_text}, nwave)
    depth_model = [wl, depth]"""

        return transit_depth_script

    elif model_type == 'Blackbody':
        if rprs_text is None:
            rprs = np.sqrt(input.eclipse_depth.get() * 0.01)
            rprs_text = f'{rprs:.5f}'
        if teq_text is None:
            teq_text = f'{input.teq_planet.get():.1f}'
        transit_depth_script = f"""
    # The planet's {obs_geometry} spectrum:
    obs_type = {repr(obs_geometry)}
    t_planet = {teq_text}
    rprs = {rprs_text}
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

    # The target ({name}):
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

    filename = make_save_label(
        name.replace(' ','_'), inst, mode, aperture, disperser, filter,
    )
    save_tso = f"""
    # Save to file (set lightweight=False to keep '2d'/'3d' fields)
    filename = {repr(filename)}
    pando.save_tso(filename, lightweight=True)
"""

    if 'np.' in script:
        imports += "    import numpy as np\n"
    if 'ps.' in script:
        imports += "    import pyratbay.spectrum as ps\n"
    script = imports + script + save_tso
    return textwrap.dedent(script)


def export_script_calculated_values(
        input, spectra, saturation_fraction,
        acquisition_targets, acq_target_list, catalog,
    ):
    """
    Translate gen_tso's current app state to a python script
    When values can be computed from variables rather than fixed
    numbers, show that script.
    """
    config = parse_instrument(
        input, 'instrument', 'mode', 'aperture', 'disperser', 'filter',
        'subarray', 'readout', 'order', 'ngroup', 'nint',
        'pairing', 'pupil', 'detector',
    )
    inst, mode, aperture, disperser, filter, subarray, readout = config[0:7]
    order, ngroup, nint, pairing, pupil, detector = config[7:]

    name = input.target.get()
    target = catalog.get_target(name)
    target_focus = input.target_focus.get()
    if target_focus == 'acquisition':
        selected = acquisition_targets.cell_selection()['rows'][0]
        target_list = acq_target_list.get()
        target_acq_mag = np.round(target_list[1][selected], 3)
        name = target_list[0][selected]
    elif target_focus == 'science':
        target_acq_mag = None

    req_saturation = saturation_fraction.get()
    transit_dur = float(input.t_dur.get())
    planet_model_type, depth_label, rprs_sq, teq_planet = parse_obs(input)

    sed_type, sed_model, norm_band, norm_mag, sed_label = parse_sed(
        input, spectra, target_acq_mag=target_acq_mag,
    )
    sed_warning = ""
    t_eff = float(input.t_eff.get())
    log_g = float(input.log_g.get())
    teff_text = 'target.teff' if t_eff == target.teff else f'{t_eff}'
    logg_text = 'target.logg_star' if log_g == target.logg_star else f'{log_g}'

    # Magnitude
    is_target_kmag = (
        target_focus == 'science' and
        norm_band == '2mass,ks' and
        norm_mag == target.ks_mag
    )
    mag_text = 'target.ks_mag' if is_target_kmag else f'{norm_mag:.4f}'

    # SED
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
        sed_script = f"sed_model = {teff_text}"
    else:
        calc_sed_model = jwst.find_closest_sed(t_eff, log_g, sed_type)
        if target_focus == 'acquisition' or calc_sed_model != sed_model:
            sed_script = f"sed_model = {repr(sed_model)}"
        else:
            sed_script = (
                f"t_eff = {teff_text}\n    "
                f"logg_star = {logg_text}\n    "
                "sed_model = jwst.find_closest_sed(t_eff, logg_star, sed_type)"
            )

    sed_script = f"""
    {sed_warning}sed_type = {repr(sed_type)}
    {sed_script}
    norm_band = {repr(norm_band)}
    norm_mag = {mag_text}
    pando.set_scene(sed_type, sed_model, norm_band, norm_mag)
"""

    # ngroup
    pando = jwst.PandeiaCalculation(inst, mode)
    pando.set_config(disperser, filter, subarray, readout, aperture, order)
    pando.set_scene(sed_type, sed_model, norm_band, norm_mag)
    ngroup_script = (
        '# Set ngroup below requested saturation fraction '
        f'(ngroup = {ngroup}):\n    '
        f'ngroup = pando.saturation_fraction(fraction={req_saturation:.1f})'
    )
    if ngroup != pando.saturation_fraction(fraction=req_saturation):
        ngroup_script = (
            '# Number of groups per integration:\n    '
            f'ngroup = {ngroup}'
        )

    # obs_duration
    obs_dur = float(input.obs_dur.get())
    transit_dur = float(input.t_dur.get())
    is_target_tdur = np.abs(transit_dur - target.transit_dur) < 0.01
    tdur_text = 'target.transit_dur' if is_target_tdur else f'{transit_dur}'

    t_start = 1.0
    t_settling = input.settling_time.get()
    t_base = input.baseline_time.get()
    min_baseline = input.min_baseline_time.get()
    t_baseline = np.max([t_base*transit_dur, min_baseline])
    total_duration = t_start + t_settling + transit_dur + 2*t_baseline
    if np.abs(total_duration - obs_dur) < 0.01:
        time_script = f"""
    # Estimate in-transit and total duration of observation:
    transit_dur = {tdur_text}
    t_start = 1.0
    t_settling = {t_settling}
    t_base = np.max([{t_base}*transit_dur, {min_baseline}])
    total_duration = t_start + t_settling + transit_dur + 2*t_base
"""
    else:
        time_script = (
             '\n    # in-transit and total duration of observation:'
            f'\n    transit_dur = {tdur_text}'
            f'\n    total_duration = {obs_dur}'
        )

    # nint
    match_integs = input.integs_switch.get()
    calculated_nint, exp_time = jwst.bin_search_exposure_time(
        inst, subarray, readout, ngroup, obs_dur,
    )
    if match_integs and nint == calculated_nint:
        nint_script = (
            f'{time_script}\n    '
            f'# Set nint to match the observation duration (nint = {nint}):\n'
            '    nint, exp_time = jwst.bin_search_exposure_time(\n'
            '        instrument, subarray, readout, ngroup, total_duration,\n'
            '    )'
        )
    else:
        nint_script = f"\n    # Number of integrations:\n    nint = {nint}"

    # Instrument setup
    inst_script = f"""
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
    pando.set_config(disperser, filter, subarray, readout, aperture, order)

    # The target ({name}):\
"""

    teq = input.teq_planet.get()
    teq_text = 'target.eq_temp' if np.abs(teq-target.eq_temp)<1.0 else f'{teq}'

    rprs = np.sqrt(input.eclipse_depth.get() * 0.01)
    is_target_rprs = np.abs(rprs - target.rprs) < 0.001
    rprs_text = 'target.rprs' if is_target_rprs else f'{rprs:.5f}'

    depth = input.transit_depth.get() * 0.01
    is_target_depth = np.abs(np.sqrt(depth) - target.rprs) < 0.001
    depth_text = 'target.rprs**2' if is_target_depth else f'{depth:.5e}'

    if mode == 'target_acq':
        sed_script += f"""
    # Timings
    ngroup = {ngroup}
    nint = 1

    # Run target acquisition
    tso = pando.perform_calculation(ngroup, nint)\
"""
    else:
        # Transit depth
        transit_depth_script = parse_depth_source(input, spectra, teq_text, rprs_text, depth_text)
        sed_script += f"""
    {ngroup_script}
    {nint_script}
    # Observation duration times (hours):
    exp_time = jwst.exposure_time(instrument, subarray, readout, ngroup, nint)
    obs_dur = exp_time / 3600.0
    {transit_depth_script}

    # Run TSO simulation:
    tso = pando.tso_calculation(
        obs_type, transit_dur, obs_dur, depth_model, ngroup,
    )\
"""

    # Now put everything together:
    target_script = f"""
    catalog = cat.Catalog()
    target = catalog.get_target({repr(name)})
"""
    if 'target.' in sed_script:
        script = inst_script + target_script + sed_script
    else:
        script = inst_script + sed_script

    imports = """\
    import gen_tso.pandeia_io as jwst
    import gen_tso.utils as u
"""

    filename = make_save_label(
        name.replace(' ','_'), inst, mode, aperture, disperser, filter,
    )
    save_tso = f"""\n
    # Save to file (set lightweight=False to keep '2d'/'3d' fields)
    filename = {repr(filename)}
    pando.save_tso(filename, lightweight=True)
"""

    if 'cat.' in script:
        imports += "    import gen_tso.catalogs as cat\n"
    if 'ps.' in script:
        imports += "    import pyratbay.spectrum as ps\n"
    if 'np.' in script:
        imports += "    import numpy as np\n"
    script = imports + script + save_tso

    return textwrap.dedent(script)
