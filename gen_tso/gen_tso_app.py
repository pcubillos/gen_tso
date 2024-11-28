# Copyright (c) 2024 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

import json
import os
from pathlib import Path
import pickle
import sys
import textwrap
from datetime import timedelta, datetime


import faicons as fa
import numpy as np
import pandas as pd
import pandeia.engine
import pyratbay.constants as pc
import pyratbay.tools as pt
import plotly.graph_objects as go
import scipy.interpolate as si
from shiny import ui, render, reactive, req, App
from shinywidgets import output_widget, render_plotly

import gen_tso
from gen_tso import catalogs as cat
from gen_tso import pandeia_io as jwst
from gen_tso import plotly_io as plots
from gen_tso import custom_shiny as cs
from gen_tso.utils import (
    ROOT,
    get_latest_pandeia_versions,
    get_version_advice,
    collect_spectra,
    read_spectrum_file,
    pretty_print_target,
)
import gen_tso.catalogs.utils as u
from gen_tso.pandeia_io.pandeia_defaults import (
    get_detector,
    make_obs_label,
    make_saturation_label,
    make_save_label,
    load_flux_rate_splines,
)
from gen_tso.pandeia_io.pandeia_setup import (
    check_pandeia_ref_data,
    check_pysynphot,
    update_synphot_files,
)
import gen_tso.viewer_popovers as pops


def load_catalog():
    catalog = cat.Catalog()
    is_jwst = np.array([target.is_jwst for target in catalog.targets])
    is_transit = np.array([target.is_transiting for target in catalog.targets])
    is_confirmed = np.array([target.is_confirmed for target in catalog.targets])
    return catalog, is_jwst, is_transit, is_confirmed


# Catalog of known exoplanets (and candidate planets)
catalog, is_jwst, is_transit, is_confirmed = load_catalog()
nplanets = len(catalog.targets)

# Catalog of stellar SEDs:
p_keys, p_models, p_teff, p_logg = jwst.load_sed_list('phoenix')
k_keys, k_models, k_teff, k_logg = jwst.load_sed_list('k93models')

phoenix_dict = {key:model for key,model in zip(p_keys, p_models)}
kurucz_dict = {key:model for key,model in zip(k_keys, k_models)}
sed_dict = {
    'phoenix': phoenix_dict,
    'kurucz': kurucz_dict,
}

bands_dict = {
    '2mass,j': 'J mag',
    '2mass,h': 'H mag',
    '2mass,ks': 'Ks mag',
    'gaia,g': 'Gaia mag',
    'johnson,v': 'V mag',
}
detectors = jwst.generate_all_instruments()
instruments = np.unique([det.instrument for det in detectors])
filter_throughputs = jwst.filter_throughputs()

modes = {}
for inst in instruments:
    spec_modes = {}
    for det in detectors:
        if det.instrument == inst and det.obs_type=='spectroscopy':
            spec_modes[det.mode] = det.mode_label
    choices = {}
    choices['Spectroscopy'] = spec_modes
    #choices['Photometry'] = photo_modes  TBD
    acq_modes = {}
    for det in detectors:
        if det.instrument == inst and det.obs_type=='acquisition':
            acq_modes[det.mode] = 'Target Acquisition'
    choices['Acquisition'] = acq_modes
    modes[inst] = choices

# Pre-computed flux rates
flux_rate_splines, full_wells = load_flux_rate_splines()


depth_choices = {
    'transit': ['Flat', 'Input'],
    'eclipse': ['Blackbody', 'Input']
}

tso_runs = {
    'Transit': {},
    'Eclipse': {},
    'Acquisition': {},
}

def make_tso_labels(tso_runs):
    tso_labels = {
        'Transit': {},
        'Eclipse': {},
        'Acquisition': {},
    }
    for key, runs in tso_runs.items():
        for tso_label, tso_run in runs.items():
            tso_key = f'{key}_{tso_label}'
            tso_labels[key][tso_key] = tso_run['label']
    return tso_labels


cache_target = {}
cache_acquisition = {}
cache_saturation = {}

# Planet and stellar spectra
spectra = {
    'transit': {},
    'eclipse': {},
    'sed': {},
}
bookmarked_spectra = {
    'transit': [],
    'eclipse': [],
    'sed': [],
}
user_spectra = {
    'transit': [],
    'eclipse': [],
    'sed': {},
}

# Load spectra from user-defined folder and/or from default folder
loading_folders = []
argv = [arg for arg in sys.argv if arg != '--debug']
if len(argv) == 2:
    loading_folders.append(os.path.realpath(argv[1]))
loading_folders.append(f'{ROOT}data/models')
current_dir = os.path.realpath(os.getcwd())

for location in loading_folders:
    t_models, e_models, sed_models = collect_spectra(location)
    for label, model in t_models.items():
        spectra['transit'][label] = model
        user_spectra['transit'].append(label)
    for label, model in e_models.items():
        spectra['eclipse'][label] = model
        user_spectra['eclipse'].append(label)
    for label, model in sed_models.items():
        user_spectra['sed'][label] = model


nasa_url = 'https://exoplanetarchive.ipac.caltech.edu/overview'
trexolists_url = 'https://www.stsci.edu/~nnikolov/TrExoLiSTS/JWST/trexolists.html'
stsci_url = 'https://www.stsci.edu/cgi-bin/get-proposal-info?id=PID&observatory=JWST'

depth_units = [
    "none",
    "percent",
    "ppm",
]
sed_units = [
    "erg s\u207b\u00b9 cm\u207b\u00b2 Hz\u207b\u00b9 (frequency space)",
    "erg s\u207b\u00b9 cm\u207b\u00b2 cm (wavenumber space)",
    "erg s\u207b\u00b9 cm\u207b\u00b2 cm\u207b\u00b9 (wavelength space)",
    "mJy",
]

# 2D heatmap plots:
heatmaps = {
    '2d_flux': 'detector',
    '2d_snr': 'snr',
    '2d_saturation': 'saturation',
    '2d_groups': 'ngroups_map',
}

layout_kwargs = dict(
    width=1/2,
    fixed_width=False,
    heights_equal='all',
    gap='7px',
    fill=False,
    fillable=True,
    class_="pb-2 pt-0 m-0",
)


app_ui = ui.page_fluid(
    ui.tags.style(
        """
        .popover {
            --bs-popover-max-width: 500px;
        }
        """
    ),
    ui.layout_columns(
        ui.span(
            ui.HTML(
                "<b>Gen TSO</b>: A general JWST simulator "
                "for exoplanet time-series observations ("
            ),
            ui.tooltip(
                ui.tags.a(
                    fa.icon_svg("book", fill='black'),
                    href='https://pcubillos.github.io/gen_tso',
                    target="_blank",
                ),
                "documentation",
                placement='bottom',
            ),
            ',',
            ui.tooltip(
                ui.input_action_link(
                    id='main_settings',
                    label='',
                    icon=fa.icon_svg("gear", fill='black'),
                ),
                "settings",
                placement='bottom',
            ),
            ')',
            style="font-size: 26px;",
        ),
        ui.output_image("tso_logo", height='50px', inline=True),
        col_widths=(11,1),
        fixed_width=False,
        fill=False,
        fillable=True,
    ),
    ui.tags.script(
        """
        $(function() {
            Shiny.addCustomMessageHandler("update_esasky", function(message) {
                var esaskyFrame = document.getElementById("esasky");
                esaskyFrame.contentWindow.postMessage(
                    JSON.parse(message.command), 'https://sky.esa.int'
                );
            });
        });
        """
    ),

    # Instrument and detector modes:
    ui.layout_columns(
        cs.navset_card_tab_jwst(
            instruments,
            id="instrument",
            selected='NIRCam',
            header="Select an instrument and detector",
            footer=ui.input_select(
                "mode",
                "",
                choices=modes['NIRCam'],
                width='425px',
            ),
        ),
        ui.card(
            # current setup and TSO runs
            ui.layout_columns(
                # Left
                ui.input_select(
                    id="display_tso_run",
                    label=ui.tooltip(
                        "Display TSO run:",
                        "TSO runs will show here after a 'Run Pandeia' call",
                        placement='right',
                    ),
                    choices=make_tso_labels(tso_runs),
                    selected=[''],
                    width='100%',
                ),
                # TBD: Set disabled based on existing TSOs
                ui.layout_column_wrap(
                    ui.input_action_button(
                        id="save_button",
                        label="Save TSO",
                        class_="btn btn-outline-success btn-sm",
                        disabled=False,
                        width='110px',
                    ),
                    ui.input_action_button(
                        id="delete_button",
                        label="Delete TSO",
                        class_='btn btn-outline-danger btn-sm',
                        disabled=False,
                        width='110px',
                    ),
                    width=1,
                    gap='5px',
                    class_="px-0 py-0 mx-0 my-0",
                ),
                col_widths=(9,3),
                fill=True,
                fillable=True,
            ),
            ui.layout_columns(
                ui.input_task_button(
                    id="run_pandeia",
                    label="Run Pandeia",
                    label_busy="processing...",
                    width='100%',
                ),
                ui.panel_conditional(
                    "input.mode == 'target_acq'",
                    ui.input_radio_buttons(
                        id='target_focus',
                        label='',
                        choices={
                            'science': 'sci target',
                            'acquisition': 'acq target',
                        },
                        selected='science',
                    ),
                ),
                col_widths=(9,3),
                gap='10px',
                class_="px-0 py-0 mx-0 my-0",
            ),
        ),
        col_widths=[6,6],
    ),

    ui.layout_columns(
        # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        # The target
        cs.custom_card(
            ui.card_header("Target", class_="bg-primary"),
            ui.panel_well(
                ui.popover(
                    ui.span(
                        fa.icon_svg("gear"),
                        style="position:absolute; top: 5px; right: 7px;",
                    ),
                    ui.input_checkbox_group(
                        id="target_filter",
                        label='',
                        choices={
                            "transit": "transiting",
                            "jwst": "JWST targets",
                            "tess": "TESS candidates",
                            "non_transit": "non-transiting",
                        },
                        selected=['jwst', 'transit'],
                    ),
                    title='Filter targets',
                    placement="right",
                    id="targets_popover",
                ),
                ui.output_ui('target_label'),
                ui.input_selectize(
                    id='target',
                    label='',
                    choices=[target.planet for target in catalog.targets],
                    selected='WASP-80 b',
                    multiple=False,
                ),
                # Target props
                ui.layout_column_wrap(
                    # Row 1
                    ui.p("T_eff (K):"),
                    ui.input_text("t_eff", "", value='1400.0'),
                    # Row 2
                    ui.p("log(g):"),
                    ui.input_text("log_g", "", value='4.5'),
                    # Row 3
                    ui.input_select(
                        id='magnitude_band',
                        label='',
                        choices=bands_dict,
                        selected='2mass,ks',
                    ),
                    ui.input_text(
                        id="magnitude",
                        label="",
                        value='10.0',
                        placeholder="magnitude",
                    ),
                    width=1/2,
                    fixed_width=False,
                    heights_equal='all',
                    gap='7px',
                    fill=False,
                    fillable=True,
                ),
                ui.input_select(
                    id="sed_type",
                    label=ui.output_ui('stellar_sed_label'),
                    choices=[
                        "phoenix",
                        "kurucz",
                        "blackbody",
                        "input",
                    ],
                    selected='phoenix',
                ),
                ui.input_select(
                    id="sed",
                    label="",
                    choices=sed_dict['phoenix'],
                    selected='g0v',
                ),
                class_="px-2 pt-2 pb-0 m-0",
            ),
            # The planet
            ui.panel_well(
                ui.popover(
                    ui.span(
                        fa.icon_svg("gear"),
                        style="position:absolute; top: 5px; right: 7px;",
                    ),
                    # Tdwell = 1.0 + 0.75 + T14 + 2*max(1, T14/2)
                    ui.markdown(
                        '*T*<sub>dur</sub> = *T*<sub>start</sub> + '
                        '*T*<sub>set</sub> + *T*<sub>base</sub> + '
                        '*T*<sub>tran</sub> + *T*<sub>base</sub>',
                    ),
                    ui.markdown(
                        'Start time window (*T*<sub>start</sub>): 1h',
                    ),
                    ui.input_numeric(
                        id="settling_time",
                        label=ui.markdown(
                            'Settling time (*T*<sub>set</sub>, h):',
                        ),
                        value = 0.75,
                        step = 0.25,
                    ),
                    ui.input_numeric(
                        id="baseline_time",
                        label=ui.markdown(
                            'Baseline time (*T*<sub>base</sub>, t_dur):',
                        ),
                        value = 0.5,
                        step = 0.25,
                    ),
                    ui.input_numeric(
                        id="min_baseline_time",
                        label='Minimum baseline time (h):',
                        value = 1.0,
                        step = 0.25,
                    ),
                    title='Observation duration',
                    placement="right",
                    id="obs_popover",
                ),
                ui.markdown("Observation"),
                ui.layout_column_wrap(
                    # Row 1
                    ui.p("Type:"),
                    ui.input_select(
                        id='obs_geometry',
                        label='',
                        choices={
                            'transit': 'Transit',
                            'eclipse': 'Eclipse',
                        }
                    ),
                    # Row 2
                    ui.output_text('transit_dur_label'),
                    ui.input_text("t_dur", "", value='2.0'),
                    # Row 3
                    ui.p("Obs_dur (h):"),
                    ui.input_text("obs_dur", "", value='5.0'),
                    width=1/2,
                    fixed_width=False,
                    heights_equal='all',
                    gap='7px',
                    fill=False,
                    fillable=True,
                ),
                ui.input_select(
                    id="planet_model_type",
                    label=ui.output_ui('depth_label_text'),
                    choices=["Input"],
                ),
                ui.panel_conditional(
                    "input.planet_model_type == 'Input'",
                    ui.tooltip(
                        ui.input_select(
                            id="depth",
                            label="",
                            choices=user_spectra['transit'],
                        ),
                        '',
                        id='depth_tooltip',
                        placement='right',
                    ),
                ),
                ui.panel_conditional(
                    "input.planet_model_type == 'Flat'",
                    ui.layout_column_wrap(
                        ui.p("Depth (%):"),
                        ui.input_numeric(
                            id="transit_depth",
                            label="",
                            value=0.5,
                            step=0.1,
                        ),
                        **layout_kwargs,
                    ),
                ),
                ui.panel_conditional(
                    "input.planet_model_type == 'Blackbody'",
                    ui.layout_column_wrap(
                        ui.HTML("<p>(Rp/Rs)<sup>2</sup> (%):</p>"),
                        ui.input_numeric(
                            id="eclipse_depth",
                            label="",
                            value=0.05,
                            step=0.1,
                        ),
                        ui.p("Temp (K):"),
                        ui.input_numeric(
                            id="teq_planet",
                            label="",
                            value=2000.0,
                            step=100,
                        ),
                        **layout_kwargs,
                    ),
                ),
                class_="px-2 pt-2 pb-0 m-0",
            ),
            body_args=dict(class_="p-2 m-0"),
        ),

        # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        # The detector setup
        cs.custom_card(
            ui.card_header(
                ui.output_ui(id='detector_label'),
                class_="bg-primary",
            ),
            # Grism/filter
            ui.panel_well(
                ui.input_select(
                    id="disperser",
                    label="Disperser",
                    choices={},
                    selected='',
                ),
                ui.input_select(
                    id="filter",
                    label="Filter",
                    choices={},
                    selected='',
                ),
                class_="px-2 pt-2 pb-0 m-0",
            ),
            # subarray / readout
            ui.panel_well(
                ui.input_select(
                    id="subarray",
                    label="Subarray",
                    choices=[''],
                    selected='',
                ),
                ui.input_select(
                    id="readout",
                    label="Readout pattern",
                    choices=[''],
                    selected='',
                ),
                ui.panel_conditional(
                    "input.mode == 'soss' && input.filter == 'clear'",
                    ui.input_select(
                        id="order",
                        label="Order",
                        choices=['1'],
                        selected='1',
                    ),
                ),
                class_="px-2 pt-2 pb-0 m-0",
            ),
            # groups and integrations
            ui.panel_well(
                cs.label_tooltip_button(
                    label='Groups per integration ',
                    icons=fa.icon_svg("circle-play", fill='black'),
                    tooltips='Estimate saturation level',
                    button_ids='calc_saturation',
                    class_='pb-1',
                ),
                ui.output_ui('groups_input'),
                ui.panel_conditional(
                    "input.mode != 'target_acq'",
                    ui.input_numeric(
                        id="integrations",
                        label="Integrations",
                        value=1,
                        min=1, max=100000,
                    ),
                    ui.input_switch(
                        "integs_switch",
                        "Match obs. duration",
                        False,
                    ),
                ),
                class_="px-2 pt-2 pb-0 m-0",
            ),
            # Search nearby Gaia targets for acquisition
            ui.panel_conditional(
                "input.mode == 'target_acq'",
                ui.panel_well(
                    ui.tooltip(
                        ui.markdown('Acquisition targets'),
                        'Gaia targets within 80" of science target',
                        id="gaia_tooltip",
                        placement="top",
                    ),
                    ui.layout_column_wrap(
                        ui.input_task_button(
                            id="search_gaia_ta",
                            label="Search nearby targets",
                            label_busy="processing...",
                            class_='btn btn-outline-secondary btn-sm',
                        ),
                        ui.p("Select TA's SED:"),
                        ui.input_select(
                            id="ta_sed",
                            label="",
                            choices=[],
                            selected='',
                        ),
                        ui.input_action_button(
                            id="get_acquisition_target",
                            label="Print acq. target data",
                            class_='btn btn-outline-secondary btn-sm',
                        ),
                        width=1,
                        heights_equal='row',
                        gap='7px',
                        class_="px-0 py-0 mx-0 my-0",
                    ),
                    class_="px-2 pt-2 pb-2 m-0",
                ),
            ),
            body_args=dict(class_="p-2 m-0"),
        ),

        # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        # Results
        ui.layout_columns(
            ui.navset_card_tab(
                ui.nav_panel(
                    "Filters",
                    pops.filter_popover,
                    cs.custom_card(
                        output_widget("plotly_filters", fillable=True),
                        body_args=dict(class_='m-0 p-0'),
                        full_screen=True,
                        height='250px',
                    ),
                ),
                ui.nav_panel(
                    "Sky view",
                    cs.custom_card(
                        ui.HTML(
                            '<iframe id="esasky" '
                            'height="100%" '
                            'width="100%" '
                            'style="overflow" '
                            'src="https://sky.esa.int/esasky/?target=0.0%200.0'
                            '&fov=0.2&sci=true&hide_welcome=true" '
                            'frameborder="0" allowfullscreen></iframe>',
                        ),
                        body_args=dict(class_='m-0 p-0'),
                        full_screen=True,
                        height='350px',
                    )
                ),
                ui.nav_panel(
                    "Stellar SED",
                    pops.sed_popover,
                    cs.custom_card(
                        output_widget("plotly_sed", fillable=True),
                        body_args=dict(padding='0px'),
                        full_screen=True,
                        height='350px',
                    ),
                ),
                ui.nav_panel(
                    ui.output_text('transit_depth_label'),
                    pops.planet_popover,
                    cs.custom_card(
                        output_widget("plotly_depth", fillable=True),
                        body_args=dict(padding='0px'),
                        full_screen=True,
                        height='350px',
                    ),
                ),
                ui.nav_panel(
                    "TSO",
                    pops.tso_popover,
                    cs.custom_card(
                        output_widget("plotly_tso", fillable=True),
                        body_args=dict(padding='0px'),
                        full_screen=True,
                        height='400px',
                    ),
                ),
                id="tab",
            ),
            ui.navset_card_tab(
                ui.nav_panel(
                    "Results",
                    ui.span(
                        ui.output_ui(id="results"),
                        style="font-family: monospace; font-size:medium;",
                    ),
                ),
                ui.nav_panel(
                    ui.output_ui(id='warnings_label'),
                    ui.output_text_verbatim(id="warnings"),
                ),
                ui.nav_panel(
                    "Acquisition targets",
                    ui.output_data_frame(id="acquisition_targets"),
                ),
            ),
            col_widths=[12, 12],
            fill=False,
        ),
        col_widths=[3, 3, 6],
    ),
    title='Gen TSO',
    theme=f'{ROOT}/data/base_theme.css',
)


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


def get_throughput(input):
    inst = input.instrument.get().lower()
    mode = input.mode.get()
    disperser = input.disperser.get()
    filter = input.filter.get()
    subarray = input.subarray.get()
    if not is_consistent(inst, mode, disperser, filter, subarray):
        return None

    if mode == 'target_acq':
        obs_type = 'acquisition'
    else:
        # TBD: fix when photometry goes live
        obs_type = 'spectroscopy'

    if mode == 'lrsslitless':
        filter = 'None'
    elif mode == 'mrs_ts':
        filter = disperser

    return filter_throughputs[obs_type][inst][mode][subarray][filter]


def get_auto_sed(input):
    """
    Guess the model closest to the available options given a T_eff
    and log_g pair.
    """
    sed_type = input.sed_type()
    sed_models = sed_dict[sed_type]
    if sed_type == 'kurucz':
        m_teff, m_logg = k_teff, k_logg
    elif sed_type == 'phoenix':
        m_teff, m_logg = p_teff, p_logg

    try:
        t_eff = float(input.t_eff.get())
        log_g = float(input.log_g.get())
    except ValueError:
        return sed_models, None
    idx = jwst.find_closest_sed(t_eff, log_g, m_teff, m_logg)
    chosen_sed = list(sed_models)[idx]
    return sed_models, chosen_sed


def parse_sed(input, target_acq_mag=None):
    """Extract SED parameters"""
    if target_acq_mag is None:
        sed_type = input.sed_type()
        norm_band = input.magnitude_band.get()
        norm_magnitude = float(input.magnitude.get())
    else:
        sed_type = 'phoenix'
        norm_band = 'gaia,g'
        norm_magnitude = target_acq_mag

    if sed_type in ['phoenix', 'kurucz']:
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
        if model_label not in user_spectra['sed']:
            return None, None, None, None, None
        sed_model = user_spectra['sed'][model_label]

    if sed_type == 'kurucz':
        sed_type = 'k93models'

    # Make a label
    band_name = bands_dict[norm_band].split()[0]
    band_label = f'{norm_magnitude:.2f}_{band_name}'
    sed_label = f'{model_label}_{band_label}'

    return sed_type, sed_model, norm_band, norm_magnitude, sed_label


def parse_depth_model(input):
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
        nwave = 1000
        transit_depth = input.transit_depth.get() * 0.01
        wl = np.linspace(0.6, 50.0, nwave)
        depth = np.tile(transit_depth, nwave)
    elif model_type == 'Blackbody':
        transit_depth = input.eclipse_depth.get() * 0.01
        t_planet = input.teq_planet.get()
        # Un-normalized planet and star SEDs
        sed_type, sed_model, norm_band, norm_mag, sed_label = parse_sed(input)
        star_scene = jwst.make_scene(sed_type, sed_model, norm_band='none')
        planet_scene = jwst.make_scene('blackbody', t_planet, norm_band='none')
        wl, f_star = jwst.extract_sed(star_scene)
        wl_planet, f_planet = jwst.extract_sed(planet_scene)
        # Interpolate black body at wl_star
        interp_func = si.interp1d(
            wl_planet, f_planet, bounds_error=False, fill_value=0.0,
        )
        f_planet = interp_func(wl)
        # Eclipse_depth = Fplanet/Fstar * rprs**2
        depth = f_planet / f_star * transit_depth

    return depth_label, wl, depth


def is_consistent(
        inst, mode,
        disperser=None, filter=None, subarray=None, readout=None,
    ):
    """
    Check that detector configuration settings are consistent
    between them.
    """
    detector = get_detector(inst, mode, detectors)
    if detector is None:
        return False
    if disperser is not None and disperser not in detector.dispersers:
        return False
    if filter is not None and filter not in detector.filters:
        return False
    if subarray is not None and subarray not in detector.subarrays:
        return False
    if readout is not None and readout not in detector.readouts:
        return False
    return True


def draw(tso_list, resolution, n_obs):
    """
    Draw a random noised-up transit/eclipse depth realization from a TSO
    """
    if not isinstance(tso_list, list):
        tso_list = [tso_list]

    sims = []
    for tso in tso_list:
        bin_wl, bin_spec, bin_err, widths = jwst.simulate_tso(
           tso, n_obs=n_obs, resolution=resolution, noiseless=False,
        )
        sims.append({
            'wl': bin_wl,
            'depth': bin_spec,
            'uncert': bin_err,
        })
    return sims


def server(input, output, session):
    group_starter = reactive.Value(False)
    bookmarked_sed = reactive.Value(False)
    bookmarked_depth = reactive.Value(False)
    saturation_label = reactive.Value(None)
    update_catalog_flag = reactive.Value(False)
    update_sed_flag = reactive.Value(None)
    update_depth_flag = reactive.Value(None)
    uploaded_units = reactive.Value(None)
    warning_text = reactive.Value('')
    machine_readable_info = reactive.Value(False)
    acq_target_list = reactive.Value(None)
    current_acq_science_target = reactive.Value(None)
    preset_ngroup = reactive.Value(None)
    preset_sed = reactive.Value(None)
    preset_obs_dur = reactive.Value(None)
    esasky_command = reactive.Value(None)
    trexo_info = reactive.Value(None)
    tso_draw = reactive.Value(None)

    @reactive.effect
    @reactive.event(input.main_settings)
    def _():
        with open(f'{ROOT}/data/last_updated_trexolist.txt', 'r') as f:
            last_trexo = f.readline().replace('_','-')
        with open(f'{ROOT}/data/last_updated_nea.txt', 'r') as f:
            last_nasa = f.readline().replace('_','-')
        button_width = '95%'

        latest_pandeia = get_latest_pandeia_versions()
        latest_pandeia_jwst = latest_pandeia[0]

        gen_tso_status = get_version_advice(gen_tso)
        pandeia_status = get_version_advice(
            pandeia.engine, latest_pandeia_jwst,
        )
        pandeia_ref_status = check_pandeia_ref_data(latest_pandeia_jwst)
        pysynphot_data = check_pysynphot()

        m = ui.modal(
            ui.markdown(
                'If you see anything in <span style="color:red">red</span>, '
                'click the button to update or follow the instructions.<br>'
                'If you see <span style="color:#ffa500">orange</span>, '
                "you are encouraged to upgrade, but no stress.<br>"
                'If you see <span style="color:#0B980D">green</span>, you '
                'are good to go modeling JWST observations.'
            ),
            ui.hr(),
            gen_tso_status,
            ui.layout_columns(
                # Trexolists
                ui.input_task_button(
                    id='update_trexo',
                    label='Update JWST database',
                    label_busy="Fetching data from trexolists ...",
                    width=button_width,
                    class_="btn btn-sm",
                ),
                ui.HTML(f'Last updated: {last_trexo}'),
                # NASA Archive
                ui.input_task_button(
                    id='update_nasa',
                    label='Update Exoplanet database',
                    label_busy="Fetching data from NASA Archive ...",
                    width=button_width,
                    class_="btn btn-sm",
                ),
                ui.HTML(f"Last updated: {last_nasa}"),
                # pysynphot
                ui.input_task_button(
                    id='update_pysynphot',
                    label='Update Pysynphot',
                    label_busy="Fetching pysynphot data from STScI ...",
                    width=button_width,
                    class_="btn btn-sm",
                ),
                pysynphot_data,
                col_widths=(4,8),
                gap='10px',
                class_="px-0 py-0 mx-0 my-0",
            ),
            pandeia_status,
            pandeia_ref_status,
            ui.hr(),
            title=ui.markdown("**Settings**"),
            easy_close=True,
            size='l',
        )
        ui.modal_show(m)

    @reactive.Effect
    @reactive.event(input.update_trexo)
    def _():
        cat.fetch_trexolist()
        catalog, is_jwst, is_transit, is_confirmed = load_catalog()

    @reactive.Effect
    @reactive.event(input.update_nasa)
    def _():
        cat.update_exoplanet_archive()
        catalog, is_jwst, is_transit, is_confirmed = load_catalog()
        update_catalog_flag.set(~update_catalog_flag.get())

    @reactive.Effect
    @reactive.event(input.update_pysynphot)
    def _():
        status = update_synphot_files()
        for warning in status:
            error_msg = ui.markdown(f"**Error:**<br>{warning}")
            ui.notification_show(error_msg, type="error", duration=8)

    def run_pandeia(input):
        """
        Perform a pandeia calculation on  science or acquisition target.
        This might be a transit/eclipse TSO or a Pandeia perform_calculation
        call.
        """
        # Instrumental setup:
        inst = input.instrument.get().lower()
        mode = input.mode.get()
        disperser = input.disperser.get()
        filter = input.filter.get()
        subarray = input.subarray.get()
        readout = input.readout.get()
        order = input.order.get()
        ngroup = int(input.ngroup.get())
        nint = input.integrations.get()
        aperture = None

        detector = get_detector(inst, mode, detectors)
        inst_label = detector.instrument_label(disperser, filter)

        run_is_tso = True
        # Front-end to back-end exceptions:
        if mode == 'bots':
            disperser, filter = filter.split('/')
        if mode == 'soss':
            if filter == 'f277w':
                order = [1]
            else:
                order = [int(val) for val in order.split()]
        else:
            order = None
        if mode == 'mrs_ts':
            aperture = ['ch1', 'ch2', 'ch3', 'ch4']
        if mode == 'sw_tsgrism':
            aperture = input.disperser.get()
            disperser = 'dhs0'
        if mode == 'target_acq':
            aperture = input.disperser.get()
            disperser = None
            nint = 1
            run_is_tso = False
            run_type = 'Acquisition'

        # Target setup:
        target_focus = input.target_focus.get()
        target_name = input.target.get()
        t_eff = input.t_eff.get()
        log_g = input.log_g.get()
        obs_geometry = input.obs_geometry.get()
        transit_dur = float(input.t_dur.get())
        obs_dur = float(input.obs_dur.get())
        planet_model_type, depth_label, rprs_sq, teq_planet = parse_obs(input)

        if target_focus == 'acquisition':
            selected = acquisition_targets.cell_selection()['rows'][0]
            target_list = acq_target_list.get()
            target_acq_mag = np.round(target_list[1][selected], 3)
        elif target_focus == 'science':
            exp_time = jwst.exposure_time(
                inst, subarray, readout, ngroup, nint,
            )
            in_transit_integs, in_transit_time = jwst.bin_search_exposure_time(
                inst, subarray, readout, ngroup, transit_dur,
            )
            target_acq_mag = None

        sed_type, sed_model, norm_band, norm_mag, sed_label = parse_sed(
            input, target_acq_mag=target_acq_mag,
        )
        if sed_label is None:
            error_msg = ui.markdown("**Error:**<br>No SED model to simulate")
            ui.notification_show(error_msg, type="error", duration=5)
            return

        if run_is_tso and in_transit_integs > nint:
            error_msg = ui.markdown(
                f"**Warning:**<br>observation time for **{nint} integration"
                f"(s)** is less than the {obs_geometry} time.  Running "
                "a perform_calculation()"
            )
            ui.notification_show(error_msg, type="warning", duration=5)
            run_is_tso = False
            run_type = 'Pandeia_SNR'

        pando = jwst.PandeiaCalculation(inst, mode)
        pando.set_scene(sed_type, sed_model, norm_band, norm_mag)

        if sed_label not in bookmarked_spectra['sed']:
            scene = pando.calc['scene'][0]
            wl, flux = jwst.extract_sed(scene, wl_range=[0.3,30.0])
            spectra['sed'][sed_label] = {'wl': wl, 'flux': flux}
            bookmarked_spectra['sed'].append(sed_label)

        if not run_is_tso:
            depth_label = ''
            tso = pando.perform_calculation(
                ngroup, nint,
                disperser, filter, subarray, readout, aperture, order,
            )
        else:
            depth_label, wl, depth = parse_depth_model(input)
            if depth_label is None:
                msg = f"**Error:**<br>No {obs_geometry} depth model to simulate"
                ui.notification_show(ui.markdown(msg), type="error", duration=5)
                return
            run_type = obs_geometry.capitalize()
            if depth_label not in spectra:
                spectra[obs_geometry][depth_label] = {'wl': wl, 'depth': depth}
                bookmarked_spectra[obs_geometry].append(depth_label)
            depth_model = [wl, depth]

            observation_dur = exp_time / 3600.0
            tso = pando.tso_calculation(
                obs_geometry, transit_dur, observation_dur, depth_model,
                ngroup, disperser, filter, subarray, readout, aperture, order,
            )

        if run_is_tso:
            success = "TSO model simulated!"
        else:
            success = "Pandeia calculation done!"
        ui.notification_show(success, type="message", duration=2)
        tso_label = make_obs_label(
            inst, mode, aperture, disperser, filter, subarray, readout, order,
            ngroup, nint, run_type, sed_label, depth_label,
        )

        tso_run = dict(
            is_tso=run_is_tso,
            # The detector
            inst=inst,
            mode=mode,
            inst_label=inst_label,
            label=tso_label,
            # The SED
            target=target_name,
            t_eff=t_eff,
            log_g=log_g,
            transit_dur=transit_dur,
            sed_type=sed_type,
            sed_model=sed_model,
            norm_band=norm_band,
            norm_mag=norm_mag,
            obs_geometry=obs_geometry,
            obs_dur=obs_dur,
            planet_model_type=planet_model_type,
            depth_label=depth_label,
            rprs_sq=rprs_sq,
            teq_planet=teq_planet,
            target_focus=target_focus,
            # The instrumental setting
            aperture=aperture,
            disperser=disperser,
            filter=filter,
            subarray=subarray,
            readout=readout,
            order=order,
            ngroup=ngroup,
            nint=nint,
            # The outputs
            tso=tso,
        )

        if run_is_tso:
            # The planet
            tso_run['depth_model'] = depth_model
            if isinstance(tso, list):
                reports = (
                    [report['report_in']['scalar'] for report in tso],
                    [report['report_out']['scalar'] for report in tso],
                )
                warnings = tso[0]['report_out']['warnings']
            else:
                reports = (
                    tso['report_in']['scalar'],
                    tso['report_out']['scalar'],
                )
                warnings = tso['report_out']['warnings']
        else:
            reports = tso['scalar'], None
            warnings = tso['warnings']
        tso_run['stats'] = jwst._print_pandeia_stats(
            inst, mode, reports[0], reports[1], format='html',
        )
        tso_run['warnings'] = warnings

        if run_is_tso or mode=='target_acq':
            tso_runs[run_type][tso_label] = tso_run
            tso_labels = make_tso_labels(tso_runs)
            ui.update_select(
                'display_tso_run',
                choices=tso_labels,
                selected=f'{run_type}_{tso_label}',
            )

        # Update report
        sat_label = make_saturation_label(
            inst, mode, aperture, disperser, filter, subarray, order, sed_label,
        )
        pixel_rate, full_well = jwst.saturation_level(tso, get_max=True)
        cache_saturation[sat_label] = dict(
            brightest_pixel_rate=pixel_rate,
            full_well=full_well,
            warnings=warnings,
        )
        saturation_label.set(sat_label)
        warning_text.set(warnings)

        #print(inst, mode, aperture, disperser, filter, subarray, readout, order)
        #print(sed_type, sed_model, norm_band, repr(norm_mag))
        print('~~ TSO done! ~~')


    @reactive.Effect
    @reactive.event(input.display_tso_run)
    def update_full_state():
        """
        When a user chooses a run from display_tso_run, update the entire
        front end to match the run setup.
        """
        tso_key = input.display_tso_run.get()
        if tso_key is None:
            return
        key, tso_label = tso_key.split('_', maxsplit=1)
        tso = tso_runs[key][tso_label]

        inst = tso['inst']
        mode = tso['mode']
        detector = get_detector(inst, mode, detectors)
        instrument = detector.instrument

        # The instrumental setting
        filter = tso['filter']
        if mode == 'bots':
            filter = f"{tso['disperser']}/{tso['filter']}"
            disperser = None
        elif mode == 'sw_tsgrism':
            disperser = tso['aperture']
        elif mode == 'target_acq':
            disperser = tso['aperture']
        else:
            disperser = tso['disperser']
        subarray = tso['subarray']
        readout = tso['readout']
        order = tso['order']
        ngroup = tso['ngroup']

        # Schedule preset values for invalidation:
        if mode != input.mode.get() or subarray != input.subarray.get():
            preset_ngroup.set(
                {'mode':mode, 'subarray':subarray, 'ngroup':ngroup}
            )

        ui.update_navs('instrument', selected=instrument)
        mode_choices = modes[instrument]
        ui.update_select('mode', choices=mode_choices, selected=mode)
        ui.update_select(
            'disperser',
            label=detector.disperser_label,
            choices=detector.dispersers,
            selected=disperser,
        )
        ui.update_select(
            'filter',
            label=detector.filter_label,
            choices=detector.filters,
            selected=filter,
        )
        choices = detector.get_constrained_val('subarrays', disperser=disperser)
        ui.update_select('subarray', choices=choices, selected=subarray)
        choices = detector.get_constrained_val('readouts', disperser=disperser)
        ui.update_select('readout', choices=choices, selected=readout)
        if mode == 'soss':
            choices = detector.get_constrained_val('orders', subarray=subarray)
            order = ' '.join([str(val) for val in order])
            ui.update_select('order', choices=choices, selected=order)
        if ngroup != input.ngroup.get():
            if mode == 'target_acq':
                ui.update_select('ngroup', selected=ngroup)
            else:
                ui.update_numeric('ngroup', value=ngroup)

        # The target:
        current_target = input.target.get()
        current_tdur = input.t_dur.get()

        target_focus = tso['target_focus']
        ui.update_radio_buttons('target_focus', selected=target_focus)

        name = tso['target']
        t_dur = str(tso['transit_dur'])
        planet_model_type = tso['planet_model_type']
        ui.update_selectize('target', selected=name)
        norm_band = tso['norm_band']
        norm_mag = str(tso['norm_mag'])
        sed_type = tso['sed_type']
        if sed_type == 'k93models':
            sed_type = 'kurucz'

        if name != current_target:
            if name not in cache_target:
                cache_target[name] = {}
            cache_target[name]['t_eff'] = tso['t_eff']
            cache_target[name]['log_g'] = tso['log_g']
            cache_target[name]['t_dur'] = t_dur
            cache_target[name]['depth_label'] = tso['depth_label']
            cache_target[name]['rprs_sq'] = tso['rprs_sq']
            cache_target[name]['teq_planet'] = tso['teq_planet']
            if target_focus == 'science':
                cache_target[name]['norm_band'] = norm_band
                cache_target[name]['norm_mag'] = norm_mag
        else:
            ui.update_text('t_eff', value=tso['t_eff'])
            ui.update_text('log_g', value=tso['log_g'])
            ui.update_text('t_dur', value=t_dur)
            if target_focus == 'science':
                ui.update_select('magnitude_band', selected=norm_band)
                ui.update_text('magnitude', value=norm_mag)

        # sed_type, sed_model, norm_band, norm_mag, sed_label
        if target_focus == 'science':
            ui.update_select('sed_type', selected=sed_type)
            reset_sed = (
                sed_type != input.sed_type.get() or
                tso['t_eff']!=input.t_eff.get() or
                tso['log_g'] != input.log_g.get()
            )
            if sed_type in ['kurucz', 'phoenix']:
                if reset_sed:
                    preset_sed.set(tso['sed_model'])
                else:
                    choices = sed_dict[sed_type]
                    selected = tso['sed_model']
                    ui.update_select("sed", choices=choices, selected=selected)

        if target_focus == 'acquisition':
            selected = tso['sed_model']
            ui.update_select('ta_sed', choices=phoenix_dict, selected=selected)
            target = catalog.get_target(name, is_transit=None, is_confirmed=None)
            selected = cache_acquisition[target.host]['selected']
        # The observation
        warning_text.set(tso['warnings'])
        obs_geometry = tso['obs_geometry']
        ui.update_select('obs_geometry', selected=obs_geometry)
        if t_dur != current_tdur:
            preset_obs_dur.set(tso['obs_dur'])
        else:
            ui.update_text('obs_dur', value=tso['obs_dur'])

        choices = depth_choices[obs_geometry]
        ui.update_select(
            "planet_model_type", choices=choices, selected=planet_model_type,
        )
        if planet_model_type == 'Input':
            choices = user_spectra[obs_geometry]
            selected = tso['depth_label']
            ui.update_select("depth", choices=choices, selected=selected)
        elif planet_model_type == 'Flat':
            ui.update_numeric("transit_depth", value=tso['rprs_sq'])
        elif planet_model_type == 'Blackbody':
            ui.update_numeric("eclipse_depth", value=tso['rprs_sq'])
            ui.update_numeric("teq_planet", value=tso['teq_planet'])

        # TSO plot popover menu
        if tso['is_tso']:
            min_wl, max_wl = jwst.get_tso_wl_range(tso)
            ui.update_numeric('tso_wl_min', value=min_wl)
            ui.update_numeric('tso_wl_max', value=max_wl)

            resolution = input.tso_resolution.get()
            n_obs = input.n_obs.get()
            tso_draw.set(draw(tso['tso'], resolution, n_obs))
            units = 'percent'  if obs_geometry=='transit' else 'ppm'
            ui.update_select('plot_tso_units', selected=units)
            min_depth, max_depth, step = jwst.get_tso_depth_range(
                tso, resolution, units,
            )
            ui.update_numeric('tso_depth_min', value=min_depth, step=step)
            ui.update_numeric('tso_depth_max', value=max_depth, step=step)


    @render.image
    def tso_logo():
        img = {
            "src": f'{ROOT}data/images/gen_tso_logo.png',
            "height": "50px",
        }
        return img

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Instrument and detector modes
    @reactive.Effect(priority=3)
    @reactive.event(input.instrument)
    def _():
        inst = input.instrument.get()
        #print(f"You selected me: {inst}")
        mode_choices = modes[inst]
        choices = []
        for m in mode_choices.values():
            choices += list(m)

        mode = input.mode.get()
        selected = mode if mode in choices else choices[0]
        ui.update_select('mode', choices=mode_choices, selected=selected)


    @render.ui
    @reactive.event(input.mode)
    def detector_label():
        inst = input.instrument.get()
        mode = input.mode.get()
        if not is_consistent(inst, mode):
            return
        sw_warning = (
            'The SW Grism Time Series mode is still being calibrated; '
            'the SNR and saturation estimates provided by the ETC '
            'may therefore be outside the expected 10% accuracy level'
        )
        if mode == 'sw_tsgrism':
            return ui.tooltip(
                'Detector setup (!)',
                sw_warning,
                placement='top',
            )
        else:
            return 'Detector setup'


    @reactive.Effect(priority=2)
    @reactive.event(input.instrument, input.mode)
    def _():
        inst = input.instrument.get()
        mode = input.mode.get()
        if not is_consistent(inst, mode):
            return
        detector = get_detector(inst, mode, detectors)

        focus = 'science' if mode!='target_acq' else input.target_focus.get()
        ui.update_radio_buttons('target_focus', selected=focus)

        # The disperser
        choices = detector.dispersers
        disperser = input.disperser.get()
        if disperser not in choices:
            disperser = detector.default_disperser

        ui.update_select(
            'disperser',
            label=detector.disperser_label,
            choices=choices,
            selected=disperser,
        )

        # The filter
        choices = detector.filters
        filter = input.filter.get()
        if filter not in choices:
            filter = detector.default_filter
        ui.update_select(
            'filter',
            label=detector.filter_label,
            choices=choices,
            selected=filter,
        )

        selected = input.filter_filter.get()
        if detector.obs_type == 'acquisition':
            choices = [detector.instrument]
        else:
            choices = [detector.instrument, 'all']

        if selected != 'all' or detector.obs_type=='acquisition':
            selected = None
        ui.update_radio_buttons(
            "filter_filter",
            choices=choices,
            selected=selected,
        )


    @reactive.Effect(priority=1)
    @reactive.event(input.instrument, input.mode, input.disperser, input.filter)
    def update_subarray():
        inst = input.instrument.get()
        mode = input.mode.get()
        disperser = input.disperser.get()
        filter = input.filter.get()
        if not is_consistent(inst, mode, disperser, filter):
            return
        detector = get_detector(inst, mode, detectors)

        if mode == 'bots':
            disperser = input.filter.get().split('/')[0]
        else:
            disperser = input.disperser.get()

        if mode == 'sw_tsgrism':
            constraint = {'aperture': disperser}
        else:
            constraint = {'disperser': disperser}
        choices = detector.get_constrained_val('subarrays', **constraint)

        subarray = input.subarray.get()
        if subarray not in choices:
            subarray = detector.default_subarray
        if subarray not in choices:
            subarray = list(choices)[0]
        ui.update_select('subarray', choices=choices, selected=subarray)


    @reactive.Effect(priority=1)
    @reactive.event(
        input.instrument, input.mode,
        input.disperser, input.subarray,
    )
    def update_readout():
        inst = input.instrument.get()
        mode = input.mode.get()
        disperser = input.disperser.get()
        subarray = input.subarray.get()
        if not is_consistent(inst, mode, disperser, subarray=subarray):
            return
        detector = get_detector(inst, mode, detectors)

        if mode == 'soss':
            constraint = {'subarray': subarray}
        else:
            constraint = {'disperser': disperser}
        choices = detector.get_constrained_val('readouts', **constraint)
        readout = input.readout.get()
        if readout not in choices:
            readout = detector.default_readout
        ui.update_select('readout', choices=choices, selected=readout)


    @reactive.Effect
    @reactive.event(input.run_pandeia)
    def _():
        target_focus = input.target_focus.get()
        name = input.target.get()
        target = catalog.get_target(name, is_transit=None, is_confirmed=None)

        error_msg = None
        if target_focus=='acquisition':
            if target is None:
                error_msg = ui.markdown(
                    "Need to select a valid **Science target** before searching "
                    "for nearby acquisition targets"
                )
            elif target.host not in cache_acquisition:
                error_msg = ui.markdown(
                    "First click the '**Search nearby targets**' button, then "
                    "select a target from the '**Acquisition targets**' tab"
                )
            elif input.ta_sed.get() is None:
                error_msg = ui.markdown(
                    "First select a target from the '**Acquisition targets**' tab"
                )

        if error_msg is None:
            run_pandeia(input)
        else:
            ui.notification_show(error_msg, type="warning", duration=5)


    @reactive.effect
    @reactive.event(input.delete_button)
    def _():
        tso_key = input.display_tso_run.get()
        if tso_key is None:
            return
        key, tso_label = tso_key.split('_', maxsplit=1)
        del tso_runs[key][tso_label]
        tso_labels = make_tso_labels(tso_runs)
        ui.update_select('display_tso_run', choices=tso_labels)


    @reactive.effect
    @reactive.event(input.save_button)
    def _():
        # Make a filename from current TSO
        tso_key = input.display_tso_run.get()
        if tso_key is None:
            return
        key, tso_label = tso_key.split('_', maxsplit=1)
        tso = tso_runs[key][tso_label]
        inst = tso['inst']

        filename = make_save_label(
            tso['target'], tso['inst'], tso['mode'],
            tso['aperture'], tso['disperser'], tso['filter'],
        )
        if key != 'Acquisition':
            filename = filename.replace('tso_', f'tso_{key.lower()}_')
        overwrite_warning = ''
        if os.path.exists(f'{current_dir}/{filename}'):
            overwrite_warning = (
                ' (a file with same name already exists, '
                'edit name to avoid overwriting)'
            )
        m = ui.modal(
            ui.input_text(
                id='tso_save_file',
                label=f'Save TSO run to this file{overwrite_warning}:',
                value=filename,
                placeholder=tso_label,
                width='100%',
            ),
            ui.input_text(
                id='tso_save_dir',
                label='Located in this folder:',
                value=current_dir,
                placeholder='select a folder',
                width='100%',
            ),
            # TBD: I wish this could be used to browse a folder :(
            #ui.input_file(
            #    id="save_file_x",
            #    label="Into this folder:",
            #    button_label="Browse",
            #),
            ui.input_action_button(
                id='tso_save_button',
                label='Save to file',
            ),
            title="Download TSO run",
            easy_close=True,
            size='l',
        )
        ui.modal_show(m)

    @reactive.effect
    @reactive.event(input.tso_save_button)
    def _():
        tso_key = input.display_tso_run.get()
        if tso_key is None:
            return
        key, tso_label = tso_key.split('_', maxsplit=1)
        tso_run = tso_runs[key][tso_label]

        folder = input.tso_save_dir.get().strip()
        if folder == '':
            folder = '.'
        filename = input.tso_save_file.get()
        if filename.strip() == '':
            filename = f'tso_{key.lower()}_run.pickle'
        savefile = Path(f'{folder}/{filename}')
        if savefile.suffix == '':
            savefile = savefile.parent / f'{savefile.name}.pickle'

        with open(savefile, 'wb') as handle:
            pickle.dump(tso_run['tso'], handle, protocol=4)
        ui.modal_remove()
        ui.notification_show(
            f"TSO model saved to file: '{savefile}'",
            type="message",
            duration=5,
        )

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Target
    @reactive.Effect
    @reactive.event(input.target_filter, update_catalog_flag)
    def _():
        update_catalog_flag.get()
        mask = np.zeros(nplanets, bool)
        if 'jwst' in input.target_filter.get():
            mask |= is_jwst
        if 'transit' in input.target_filter.get():
            mask |= is_transit
        if 'non_transit' in input.target_filter.get():
            mask |= ~is_transit
        if 'tess' in input.target_filter.get():
            mask |= ~is_confirmed

        targets = [
            target.planet for target,flag in zip(catalog.targets,mask)
            if flag
        ]
        for i,target in enumerate(catalog.targets):
            if mask[i]:
                targets += target.aliases

        # Preserve current target if possible:
        name = input.target.get()
        target = catalog.get_target(name, is_transit=None, is_confirmed=None)
        planet = '' if target is None else target.planet
        ui.update_selectize('target', choices=targets, selected=planet)


    @reactive.effect
    @reactive.event(input.show_info)
    def _():
        """
        Display system parameters
        """
        name = input.target.get()
        target = catalog.get_target(name, is_transit=None, is_confirmed=None)
        planet_info, star_info, aliases = pretty_print_target(target)
        machine_readable_info.set(False)

        info = ui.layout_columns(
            ui.span(planet_info, style="font-family: monospace;"),
            ui.span(star_info, style="font-family: monospace;"),
            width=1/2,
        )

        m = ui.modal(
            info,
            ui.HTML(aliases),
            title=ui.markdown(f'System parameters for: **{target.planet}**'),
            size='l',
            easy_close=True,
            footer=ui.input_action_button(
                id="re_text",
                label="as machine readable",
                class_='btn btn-sm',
            ),
        )
        ui.modal_show(m)


    @reactive.effect
    @reactive.event(input.show_observations)
    def _():
        """
        Display JWST observations
        """
        name = input.target.get()
        target = catalog.get_target(name, is_transit=None, is_confirmed=None)

        trexo_info.set(target.trexo_data)

        keys = ui.HTML(
            'Keys:<br>'
            '<span style="color:#0B980D">Observed, publicly available.</span>'
            '<br><span>Observed, in proprietary period.</span><br>'
            '<span style="color:#FFa500">To be observed, planned window.</span>'
            '<br><span style="color:red">Failed, withdrawn, or skipped.</span>'
        )

        m = ui.modal(
            ui.output_data_frame('trexo_df'),
            ui.hr(),
            keys,
            title=ui.markdown(f'JWST programs for: **{target.host}**'),
            size='xl',
            easy_close=True,
        )
        ui.modal_show(m)


    @render.data_frame
    @reactive.event(trexo_info)
    def trexo_df():
        data = trexo_info.get()
        nobs = len(data['program'])

        today = datetime.today()
        status = data['status']
        date_obs = data['date_start']
        plan_obs = data['plan_window']
        propriety = data['proprietary_period']
        warnings = [
            i for i in range(nobs)
            if status[i] in ['Skipped', 'Failed', 'Withdrawn']
        ]
        available = []
        dates = []
        tbd_dates = []
        for i in range(nobs):
            if isinstance(date_obs[i], datetime):
                release = date_obs[i] + timedelta(days=365.0*propriety[i]/12)
                if release < today:
                    available.append(i)
                dates.append(
                    date_obs[i].strftime('%Y-%m-%d') +
                    f' ({propriety[i]} m)'
                )
            else:
                if isinstance(plan_obs[i], datetime):
                    dates.append(
                        plan_obs[i].strftime('%Y-%m-%d') +
                        f' ({propriety[i]} m)'
                    )
                else:
                    dates.append(f'--- ({propriety[i]} m)')
                if i not in warnings:
                    tbd_dates.append(i)

        styles = [
            {
                'rows': available,
                'style': {"color": "#0B980D"},
            },
            {
                'rows': warnings,
                'style': {"color": "red"},
            },
            {
                'rows': tbd_dates,
                'cols': [10],
                'style': {"color": "#FFa500"},
            },
        ]
        programs = [
            ' '.join(program.split()[0:2])
            for program in data['program']
        ]
        pi = [
            ' '.join(program.split()[2:])
            for program in data['program']
        ]
        hrefs = [stsci_url.replace('PID', pid.split()[1]) for pid in programs]
        programs = [
            ui.tags.a(programs[i], href=hrefs[i], target="_blank")
            for i in range(nobs)
        ]

        data_df = {
            'Program ID': programs,
            'PI': pi,
            'Target': data['trexo_name'],
            # TBD: fetch which planet(s)
            #'planet': ['b' for _ in pi],
            'Event': data['event'],
            'Status': data['status'],
            'Instrument / Mode': data['mode'],
            'Subarray': data['subarray'],
            'Readout': data['readout'],
            'Groups': data['groups'],
            'Duration (h)': data['duration'],
            'Obs date (prop. period)': dates,
        }
        df = pd.DataFrame(data=data_df)

        return render.DataGrid(
            df,
            styles=styles,
            width='100%',
        )


    @reactive.Effect
    @reactive.event(input.re_text)
    def _():
        mri = machine_readable_info.get()
        machine_readable_info.set(~mri)

        name = input.target.get()
        target = catalog.get_target(name, is_transit=None, is_confirmed=None)
        if machine_readable_info.get():
            info = ui.span(
                ui.HTML(target.machine_readable_text().replace('\n','<br>')),
                style="font-family: monospace; font-size:medium;",
            )
            button_label = 'as pretty text'
        else:
            planet_info, star_info, aliases = pretty_print_target(target)
            info = ui.layout_columns(
                ui.span(planet_info, style="font-family: monospace;"),
                ui.span(star_info, style="font-family: monospace;"),
                width=1/2,
            )
            info = [info, ui.HTML(aliases)]
            button_label = 'as machine readable'

        ui.modal_remove()
        m = ui.modal(
            info,
            title=ui.markdown(f'System parameters for: **{target.planet}**'),
            size='l',
            easy_close=True,
            footer=ui.input_action_button(
                id="re_text",
                label=button_label,
                class_='btn btn-sm',
            ),
        )
        ui.modal_show(m)


    @render.ui
    @reactive.event(input.target)
    def target_label():
        name = input.target.get()
        target = catalog.get_target(name, is_transit=None, is_confirmed=None)
        if target is None:
            return ui.span('Science target')

        if len(target.aliases) > 0:
            aliases = ', '.join(target.aliases)
            info_label = f"Also known as: {aliases}"
        else:
            info_label = 'System info'
        info_tooltip = ui.tooltip(
            ui.input_action_link(
                id='show_info',
                label='',
                icon=fa.icon_svg("circle-info", fill='cornflowerblue'),
            ),
            info_label,
            placement='top',
        )

        if target.is_jwst:
            trexolists_tooltip = ui.tooltip(
                ui.input_action_link(
                    id='show_observations',
                    label='',
                    icon=fa.icon_svg("circle-info", fill='goldenrod'),
                ),
                "This target's host is on TrExoLiSTS",
                placement='top',
            )
        else:
            trexolists_tooltip = ui.tooltip(
                fa.icon_svg("circle-info", fill='gray'),
                'not a JWST target (yet)',
                placement='top',
            )

        if target.is_confirmed:
            candidate_tooltip = None
        else:
            candidate_tooltip = ui.tooltip(
                fa.icon_svg("triangle-exclamation", fill='darkorange'),
                ui.markdown("This is a *candidate* planet"),
                placement='top',
            )

        return ui.span(
            'Science target ',
            info_tooltip,
            ui.tooltip(
                ui.tags.a(
                    fa.icon_svg("circle-info", fill='black'),
                    href=f'{nasa_url}/{target.planet}',
                    target="_blank",
                ),
                'See this target on the NASA Exoplanet Archive',
                placement='top',
            ),
            trexolists_tooltip,
            candidate_tooltip,
        )


    @reactive.effect
    @reactive.event(input.target)
    def _():
        """Update target properties"""
        name = input.target.get()
        target = catalog.get_target(name, is_transit=None, is_confirmed=None)
        if target is None:
            return
        if name in target.aliases:
            ui.update_selectize('target', selected=target.planet)

        # Physical properties:
        if target.planet in cache_target:
            t_eff  = cache_target[target.planet]['t_eff']
            log_g = cache_target[target.planet]['log_g']
            t_dur = cache_target[target.planet]['t_dur']
            band = cache_target[target.planet]['norm_band']
            magnitude = cache_target[target.planet]['norm_mag']
        else:
            t_eff = u.as_str(target.teff, '.1f', '')
            log_g = u.as_str(target.logg_star, '.2f', '')
            t_dur = u.as_str(target.transit_dur, '.3f', '')
            band = '2mass,ks'
            magnitude = f'{target.ks_mag:.3f}'

        ui.update_text('t_eff', value=t_eff)
        ui.update_text('log_g', value=log_g)
        ui.update_select('magnitude_band', selected=band)
        ui.update_text('magnitude', value=magnitude)
        ui.update_text('t_dur', value=t_dur)

        delete_catalog = {
            "event": 'deleteCatalogue',
            "content": { 'overlayName': 'Nearby Gaia sources'}
        }
        delete_footprint = {
            "event": 'deleteFootprintsOverlay',
            "content": {'overlayName': 'visit splitting distance'}
        }
        goto = {
            "event": "goToRaDec",
            "content":{"ra": f"{target.ra}", "dec": f"{target.dec}"}
        }
        esasky_command.set([delete_catalog, delete_footprint, goto])

        # Observing properties:
        if name in cache_target and cache_target[name]['rprs_sq'] is not None:
            rprs_square_percent = cache_target[name]['rprs_sq']
            teq_planet = cache_target[name]['teq_planet']
            cache_target[name]['rprs_sq'] = None
            cache_target[name]['teq_planet'] = None
        else:
            teq_planet = np.round(target.eq_temp, decimals=1)
            if np.isnan(teq_planet):
                teq_planet = 0.0
            rprs_square = target.rprs**2.0
            if np.isnan(rprs_square):
                rprs_square = 0.0
            rprs_square_percent = np.round(100*rprs_square, decimals=4)

        if rprs_square_percent is not None:
            ui.update_numeric("transit_depth", value=rprs_square_percent)
            ui.update_numeric("eclipse_depth", value=rprs_square_percent)
        if teq_planet is not None:
            ui.update_numeric('teq_planet', value=teq_planet)


    @reactive.Effect
    @reactive.event(input.sed_type, input.t_eff, input.log_g, update_sed_flag)
    def choose_sed():
        sed_type = input.sed_type.get()
        if sed_type in ['phoenix', 'kurucz']:
            choices, selected = get_auto_sed(input)
            if preset_sed.get() is not None:
                selected = preset_sed.get()
                preset_sed.set(None)
        elif sed_type == 'blackbody':
            if input.t_eff.get() == '':
                t_eff = 0.0
            else:
                t_eff = float(input.t_eff.get())
            selected = f' Blackbody (Teff={t_eff:.0f} K)'
            choices = [selected]
        elif sed_type == 'input':
            choices = list(user_spectra['sed'])
            selected = None

        ui.update_select("sed", choices=choices, selected=selected)


    @render.ui
    @reactive.event(
        bookmarked_sed, input.sed,
        input.t_eff, input.magnitude_band, input.magnitude,
    )
    def stellar_sed_label():
        """Check current SED is bookmarked"""
        sed_type, sed_model, norm_band, norm_mag, sed_label = parse_sed(input)
        is_bookmarked = sed_label in bookmarked_spectra['sed']
        bookmarked_sed.set(is_bookmarked)
        if is_bookmarked:
            sed_icon = fa.icon_svg("star", style='solid', fill='gold')
        else:
            sed_icon = fa.icon_svg("star", style='regular', fill='black')

        icons = [
            sed_icon,
            fa.icon_svg("file-arrow-up", fill='black'),
        ]
        texts = [
            'Bookmark SED',
            'Upload SED',
        ]
        return cs.label_tooltip_button(
            label='Stellar SED model: ',
            icons=icons,
            tooltips=texts,
            button_ids=['sed_bookmark', 'upload_sed']
        )

    @reactive.Effect
    @reactive.event(input.sed_bookmark)
    def _():
        """Toggle bookmarked SED"""
        sed_type, sed_model, norm_band, norm_mag, sed_label = parse_sed(input)
        if sed_type is None:
            msg = ui.markdown("**Error**:<br>No SED model to bookmark")
            ui.notification_show(msg, type="error", duration=5)
            return
        is_bookmarked = not bookmarked_sed.get()
        bookmarked_sed.set(is_bookmarked)
        if is_bookmarked:
            scene = jwst.make_scene(sed_type, sed_model, norm_band, norm_mag)
            wl, flux = jwst.extract_sed(scene, wl_range=[0.3,30.0])
            spectra['sed'][sed_label] = {'wl': wl, 'flux': flux}
            bookmarked_spectra['sed'].append(sed_label)
        else:
            bookmarked_spectra['sed'].remove(sed_label)


    @render.ui
    @reactive.event(
        bookmarked_depth, input.obs_geometry, input.planet_model_type,
        input.depth, input.transit_depth, input.eclipse_depth, input.teq_planet,
    )
    def depth_label_text():
        """Set depth model label"""
        obs_geometry = input.obs_geometry.get()
        depth_label = planet_model_name(input)

        is_bookmarked = depth_label in bookmarked_spectra[obs_geometry]
        bookmarked_depth.set(is_bookmarked)
        fill = 'royalblue' if is_bookmarked else 'gray'
        depth_icon = fa.icon_svg("earth-americas", style='solid', fill=fill)
        icons = [
            depth_icon,
            fa.icon_svg("file-arrow-up", fill='black'),
        ]
        texts = [
            f'Bookmark {obs_geometry} depth model',
            f'Upload {obs_geometry} depth model',
        ]
        return cs.label_tooltip_button(
            label=f"{obs_geometry.capitalize()} depth spectrum: ",
            icons=icons,
            tooltips=texts,
            button_ids=['bookmark_depth', 'upload_depth'],
        )


    @reactive.Effect
    @reactive.event(input.bookmark_depth)
    def _():
        """Toggle bookmarked depth model"""
        obs_geometry = input.obs_geometry.get()
        depth_label = planet_model_name(input)
        if depth_label is None:
            msg = ui.markdown(
                f"**Error:**<br>No {obs_geometry} depth model to bookmark"
            )
            ui.notification_show(msg, type="error", duration=5)
            return
        is_bookmarked = not bookmarked_depth.get()
        bookmarked_depth.set(is_bookmarked)
        if is_bookmarked:
            bookmarked_spectra[obs_geometry].append(depth_label)
            depth_label, wl, depth = parse_depth_model(input)
            spectra[obs_geometry][depth_label] = {'wl': wl, 'depth': depth}
        else:
            bookmarked_spectra[obs_geometry].remove(depth_label)


    @reactive.effect
    @reactive.event(input.obs_geometry, update_depth_flag)
    def _():
        obs_geometry = input.obs_geometry.get()
        choices = depth_choices[obs_geometry]
        model_type = input.planet_model_type.get()
        if model_type not in choices:
            model_type = choices[0]
        ui.update_select(
            "planet_model_type", choices=choices, selected=model_type,
        )

        spectra = user_spectra[obs_geometry]
        name = input.target.get()
        cached = (
            name in cache_target and
            cache_target[name]['depth_label'] is not None
        )
        selected = input.depth.get()
        if cached:
            selected = cache_target[name]['depth_label']
            cache_target[name]['depth_label'] = None
        elif selected not in spectra:
            selected = None if len(spectra) == 0 else spectra[0]
        #print(f'Updating input [cached={cached}]: {repr(selected)}')
        ui.update_select("depth", choices=spectra, selected=selected)

        if len(spectra) > 0:
            tooltip_text = ''
        elif obs_geometry == 'transit':
            tooltip_text = f'Upload a {obs_geometry} depth spectrum'
        elif obs_geometry == 'eclipse':
            tooltip_text = f'Upload an {obs_geometry} depth spectrum'
        ui.update_tooltip('depth_tooltip', tooltip_text)


    @render.text
    @reactive.event(input.obs_geometry)
    def transit_dur_label():
        obs_geometry = input.obs_geometry.get().capitalize()
        return f"{obs_geometry[0]}_dur (h):"

    @render.text
    @reactive.event(input.obs_geometry)
    def transit_depth_label():
        obs_geometry = input.obs_geometry.get().capitalize()
        return f"{obs_geometry} depth"

    @render.ui
    @reactive.event(warning_text)
    def warnings_label():
        warnings = warning_text.get()
        if len(warnings) == 0:
            return "Warnings"
        n_warn = len(warnings)
        return ui.HTML(f'<div style="color:red;">Warnings ({n_warn})</div>')

    @reactive.Effect
    @reactive.event(
        input.t_dur, input.settling_time, input.baseline_time,
        input.min_baseline_time,
    )
    def _():
        """Set observation time based on transit dur and popover settings"""
        if preset_obs_dur.get() is not None:
            obs_dur = preset_obs_dur.get()
            preset_obs_dur.set(None)
            ui.update_text('obs_dur', value=f'{obs_dur:.2f}')
            return
        t_dur = req(input.t_dur).get()
        if t_dur == '':
            ui.update_text('obs_dur', value='0.0')
            return
        transit_dur = float(t_dur)
        settling = req(input.settling_time).get()
        baseline = req(input.baseline_time).get()
        min_baseline = req(input.min_baseline_time).get()
        baseline = np.clip(baseline*transit_dur, min_baseline, np.inf)
        # Tdwell = T_start + T_settle + T14 + 2*max(1, T14/2)
        obs_dur = 1.0 + settling + transit_dur + 2.0*baseline
        ui.update_text('obs_dur', value=f'{obs_dur:.2f}')


    @reactive.effect
    @reactive.event(input.upload_sed)
    def _():
        m = ui.modal(
            ui.markdown(
                "Input files must be plan-text files with two columns, "
                "the first one being the wavelength (microns) and "
                "the second one the stellar SED.<br>**Make sure "
                "the input units are correct before uploading a file!**"
            ),
            ui.input_radio_buttons(
                id="upload_units",
                label='Flux units:',
                choices=sed_units,
                width='100%',
            ),
            ui.input_file(
                id="upload_file",
                label='',
                button_label="Browse",
                multiple=True,
                width='100%',
            ),
            title="Upload stellar spectrum",
            easy_close=True,
        )
        ui.modal_show(m)


    @reactive.effect
    @reactive.event(input.upload_depth)
    def _():
        obs_geometry = input.obs_geometry.get()
        m = ui.modal(
            ui.markdown(
                "Input files must be plan-text files with two columns, "
                "the first one being the wavelength (microns) and "
                f"the second one the {obs_geometry} depth.<br>**Make sure "
                "the input units are correct before uploading a file!**"
            ),
            ui.input_radio_buttons(
                id="upload_units",
                label='Depth units:',
                choices=depth_units,
                width='100%',
            ),
            ui.input_file(
                id="upload_file",
                label='',
                button_label="Browse",
                multiple=True,
                width='100%',
            ),
            title="Upload planetary spectrum",
            easy_close=True,
        )
        ui.modal_show(m)


    @reactive.effect
    @reactive.event(input.upload_units)
    def _():
        uploaded_units.set(input.upload_units.get())


    @reactive.effect
    @reactive.event(input.upload_file)
    def _():
        new_model = input.upload_file.get()
        if not new_model:
            return

        # The units tell this function SED or depth spectrum:
        units = uploaded_units.get()
        label, wl, model = read_spectrum_file(
            new_model[0]['datapath'], on_fail='warning',
        )
        label = new_model[0]['name']
        if wl is None:
            msg = ui.markdown(
                f'**Error:**<br>Invalid format for input file:<br>*{label}*'
            )
            ui.notification_show(msg, type="error", duration=5)
            return

        if label.endswith('.dat') or label.endswith('.txt'):
            label = label[0:-4]

        if units in depth_units:
            obs_geometry = input.obs_geometry.get()
            u = pt.u(units)
            spectra[obs_geometry][label] = {'wl': wl, 'depth': model*u}
            user_spectra[obs_geometry].append(label)
            bookmarked_spectra[obs_geometry].append(label)
            if input.planet_model_type.get() != 'Input':
                return
            # Trigger update choose_depth
            update_depth_flag.set(label)
        elif units in sed_units:
            if 'frequency' in units:
                u = 10**26
            elif 'wavenumber' in units:
                u = 10**26 / pc.c
            elif 'wavelength' in units:
                u = 10**26 / pc.c * (wl*pc.um)**2.0
            elif 'mJy' in units:
                u = 1.0
            user_spectra['sed'][label] = {'wl': wl, 'flux': model*u}
            update_sed_flag.set(label)


    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Detector setup
    @reactive.Effect
    @reactive.event(input.subarray)
    def set_soss_orders():
        inst = input.instrument.get()
        mode = input.mode.get()
        subarray = input.subarray.get()
        if not is_consistent(inst, mode, subarray=subarray):
            return
        if mode != 'soss':
            return
        detector = get_detector(inst, mode, detectors)
        choices = detector.get_constrained_val('orders', subarray=subarray)
        order = input.order.get()
        if order not in choices:
            order = list(choices)[0]
        ui.update_select('order', choices=choices, selected=order)


    @render.ui
    @reactive.event(input.mode, input.subarray)
    def groups_input():
        inst = input.instrument.get()
        mode = input.mode.get()
        subarray = input.subarray.get()
        if group_starter.get():
            current_value = input.ngroup.get()
        else:
            current_value = 2
            group_starter.set(True)
        preset = preset_ngroup.get()
        has_preset = (
            preset is not None and
            preset['mode'] == mode and
            preset['subarray'] == subarray
        )
        if has_preset:
            value = preset['ngroup']
            preset_ngroup.set(None)
        else:
            value = current_value
        if mode == 'target_acq':
            detector = get_detector(inst, mode, detectors)
            choices = detector.get_constrained_val('groups', subarray=subarray)

            value = str(value)
            if value not in choices:
                value = None
            return ui.input_select(
                id="ngroup",
                label="",
                choices=choices,
                selected=value,
            )
        else:
            return ui.input_numeric(
                id="ngroup",
                label='',
                value=int(value),
                min=2, max=10000,
            )

    @reactive.Effect
    @reactive.event(input.calc_saturation)
    def calculate_saturation_level():
        inst = input.instrument.get().lower()
        mode = input.mode.get()
        disperser = input.disperser.get()
        filter = input.filter.get()
        subarray = input.subarray.get()
        readout = input.readout.get()
        order = input.order.get()
        aperture = None
        sed_type, sed_model, norm_band, norm_mag, sed_label = parse_sed(input)

        # Front-end to back-end exceptions:
        if mode == 'sw_tsgrism':
            aperture = input.disperser.get()
            disperser = 'dhs0'
        if mode == 'bots':
            disperser, filter = filter.split('/')
        if mode == 'soss':
            order = [int(val) for val in order.split()]
        if mode == 'mrs_ts':
            aperture = ['ch1', 'ch2', 'ch3', 'ch4']
        if mode == 'target_acq':
            ngroup = int(input.ngroup.get())
            aperture = input.disperser.get()
            disperser = None
        else:
            ngroup = 2

        sat_label = make_saturation_label(
            inst, mode, aperture, disperser, filter, subarray, order, sed_label,
        )

        pando = jwst.PandeiaCalculation(inst, mode)
        pando.set_scene(sed_type, sed_model, norm_band, norm_mag)
        pixel_rate, full_well = pando.get_saturation_values(
            disperser, filter, subarray, readout, ngroup, aperture, order,
            get_max=True,
        )

        cache_saturation[sat_label] = dict(
            brightest_pixel_rate=pixel_rate,
            full_well=full_well,
        )
        # This reactive variable enforces a re-rendering of results
        saturation_label.set(sat_label)


    @reactive.Effect
    @reactive.event(
        input.integs_switch, input.obs_dur, input.mode,
        input.instrument, input.ngroup, input.readout, input.subarray,
    )
    def _():
        """Switch to make the integrations match observation duration"""
        if input.mode.get() == 'target_acq':
            return
        match_dur = input.integs_switch.get()
        if not match_dur:
            ui.update_numeric('integrations', value=1)
            return

        obs_dur = float(req(input.obs_dur).get())
        inst = input.instrument.get().lower()
        ngroup = input.ngroup.get()
        readout = input.readout.get()
        subarray = input.subarray.get()
        if ngroup is None:
            return
        integs, exp_time = jwst.bin_search_exposure_time(
            inst, subarray, readout, int(ngroup), obs_dur,
        )
        if exp_time == 0.0:
            return
        ui.update_numeric('integrations', value=integs)


    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Viewers
    @render_plotly
    def plotly_filters():
        show_all = req(input.filter_filter).get() == 'all'

        inst = input.instrument.get().lower()
        mode = input.mode.get()
        filter = input.filter.get()
        subarray = input.subarray.get()
        if not is_consistent(inst, mode, filter=filter, subarray=subarray):
            return

        if mode == 'lrsslitless':
            filter = 'None'
        if mode == 'mrs_ts':
            filter = input.disperser.get()

        if mode == 'target_acq':
            throughputs = filter_throughputs['acquisition']
        else:
            throughputs = filter_throughputs['spectroscopy']

        fig = plots.plotly_filters(
            throughputs, inst, mode, subarray, filter, show_all,
        )
        return fig


    @render_plotly
    def plotly_sed():
        # Gather bookmarked SEDs
        input.sed_bookmark.get()  # (make panel reactive to sed_bookmark)
        model_names = bookmarked_spectra['sed']
        if len(model_names) == 0:
            fig = go.Figure()
            fig.update_layout(title='Bookmark some SEDs to show them here')
            return fig
        sed_models = [spectra['sed'][model] for model in model_names]

        # Get current SED:
        sed_type, sed_model, norm_band, norm_mag, current_model = parse_sed(input)

        wl_scale = input.plot_sed_xscale.get()
        wl_range = [input.sed_wl_min.get(), input.sed_wl_max.get()]

        throughput = get_throughput(input)
        units = input.plot_sed_units.get()
        resolution = input.plot_sed_resolution.get()
        fig = plots.plotly_sed_spectra(
            sed_models, model_names, current_model,
            units=units,
            wl_range=wl_range, wl_scale=wl_scale,
            resolution=resolution,
            throughput=throughput,
        )
        return fig


    @render_plotly
    def plotly_depth():
        input.bookmark_depth.get()  # (make panel reactive to bookmark_depth)
        update_depth_flag.get()
        obs_geometry = input.obs_geometry.get()
        model_names = bookmarked_spectra[obs_geometry]
        nmodels = len(model_names)
        if nmodels == 0:
            return go.Figure()
        throughput = get_throughput(input)

        current_model = planet_model_name(input)
        units = input.plot_depth_units.get()
        wl_scale = input.plot_depth_xscale.get()
        wl_range = [input.depth_wl_min.get(), input.depth_wl_max.get()]
        resolution = input.depth_resolution.get()

        depth_models = [spectra[obs_geometry][model] for model in model_names]
        fig = plots.plotly_depth_spectra(
            depth_models, model_names, current_model,
            units=units,
            wl_range=wl_range, wl_scale=wl_scale,
            resolution=resolution,
            obs_geometry=obs_geometry,
            throughput=throughput,
        )
        return fig


    @render_plotly
    def plotly_tso():
        tso_key = input.display_tso_run.get()
        if tso_key is None:
            return go.Figure()
        key, tso_label = tso_key.split('_', maxsplit=1)
        tso_run = tso_runs[key][tso_label]
        wl_scale = input.plot_tso_xscale.get()
        wl_range = [input.tso_wl_min.get(), input.tso_wl_max.get()]

        plot_type = input.tso_plot.get()
        if plot_type == 'tso':
            units = input.plot_tso_units.get()
            sim_depths = tso_draw.get()
            depth_range = [input.tso_depth_min.get(), input.tso_depth_max.get()]
            planet = tso_run['depth_label']
            fig = plots.plotly_tso_spectra(
                tso_run['tso'], sim_depths,
                model_label=planet,
                instrument_label=tso_run['inst_label'],
                bin_widths=None,
                units=units, wl_range=wl_range, wl_scale=wl_scale,
                depth_range=depth_range, obs_geometry=tso_run['obs_geometry'],
            )
        elif plot_type == 'fluxes':
            fig = plots.plotly_tso_fluxes(
                tso_run['tso'],
                wl_range=wl_range, wl_scale=wl_scale,
                obs_geometry=tso_run['obs_geometry'],
            )
        elif plot_type == 'snr':
            fig = plots.plotly_tso_snr(
                tso_run['tso'],
                wl_range=wl_range, wl_scale=wl_scale,
                obs_geometry=tso_run['obs_geometry'],
            )
        elif plot_type in heatmaps:
            fig = plots.plotly_tso_2d(tso_run['tso'], heatmaps[plot_type])
        return fig

    @reactive.effect
    @reactive.event(input.plot_tso_units)
    def rescale_tso_depths():
        tso_key = input.display_tso_run.get()
        if tso_key is None:
            return
        key, tso_label = tso_key.split('_', maxsplit=1)
        tso = tso_runs[key][tso_label]
        resolution = input.tso_resolution.get()
        units = input.plot_tso_units.get()

        min_depth, max_depth, step = jwst.get_tso_depth_range(
            tso, resolution, units,
        )
        ui.update_numeric('tso_depth_min', value=min_depth, step=step)
        ui.update_numeric('tso_depth_max', value=max_depth, step=step)


    @reactive.effect
    @reactive.event(input.redraw_tso, input.n_obs, input.tso_resolution)
    def redraw_tso_scatter():
        tso_key = input.display_tso_run.get()
        if tso_key is None:
            return
        key, tso_label = tso_key.split('_', maxsplit=1)
        tso = tso_runs[key][tso_label]

        n_obs = input.n_obs.get()
        resolution = input.tso_resolution.get()
        tso_draw.set(draw(tso['tso'], resolution, n_obs))


    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Results
    @render.ui
    def results():
        warnings = {}
        # Only read for reactivity reasons:
        saturation_label.get()
        input.ta_sed.get()

        inst = input.instrument.get().lower()
        mode = input.mode.get()
        disperser = input.disperser.get()
        filter = input.filter.get()
        subarray = input.subarray.get()
        readout = input.readout.get()
        order = input.order.get()
        ngroup = input.ngroup.get()
        nint = input.integrations.get()
        sed_type, sed_model, norm_band, norm_mag, sed_label = parse_sed(input)

        name = input.target.get()
        target = catalog.get_target(name, is_transit=None, is_confirmed=None)
        depth_label = parse_obs(input)[1]
        transit_dur = float(input.t_dur.get())

        consistent = is_consistent(
            inst, mode, disperser, filter, subarray, readout,
        )
        if ngroup is None or not consistent or sed_label is None:
            warning_text.set(warnings)
            return ui.HTML('<pre> </pre>')

        obs_geometry = input.obs_geometry.get()
        run_type = obs_geometry.capitalize()
        # Front-end to back-end exceptions:
        aperture = None
        if mode == 'bots' and '/' in filter:
            disperser, filter = filter.split('/')
        if mode == 'mrs_ts':
            aperture = ['ch1', 'ch2', 'ch3', 'ch4']
        if mode == 'sw_tsgrism':
            aperture = input.disperser.get()
            disperser = 'dhs0'
        if mode == 'target_acq':
            aperture = input.disperser.get()
            disperser = None
            nint = 1
            run_type = 'Acquisition'
            depth_label = ''

        ngroup = int(ngroup)
        report_text = jwst._print_pandeia_exposure(
            inst, subarray, readout, ngroup, nint,
        )
        target_focus = input.target_focus.get().capitalize()

        if target_focus == 'Science':
            target_acq_mag = None
            target_name = f': {target.planet}' if target is not None else ''
        else:
            no_target = (
                target is None or
                target.host not in cache_acquisition or
                cache_acquisition[target.host]['selected'] is None
            )
            if no_target:
                report_text = f'<b>{target_focus} target</b><br>{report_text}'
                warning_text.set(warnings)
                return ui.HTML(f'<pre>{report_text}</pre>')
            target_list = cache_acquisition[target.host]['targets']
            selected = cache_acquisition[target.host]['selected']
            target_acq_mag = np.round(target_list[1][selected], 3)
            target_name = f': {target_list[0][selected]}'
        report_text = f'<b>{target_focus} target{target_name}</b><br>{report_text}'

        sed_type, sed_model, norm_band, norm_mag, sed_label = parse_sed(
            input, target_acq_mag=target_acq_mag,
        )
        sat_label = make_saturation_label(
            inst, mode, aperture, disperser, filter, subarray, order, sed_label,
        )
        tso_label = make_obs_label(
            inst, mode, aperture, disperser, filter, subarray, readout, order,
            ngroup, nint, run_type, sed_label, depth_label,
        )

        sed_items = sat_label.split('_')
        band_label = sed_items[-1]
        sat_guess_label = '_'.join(sed_items[0:-2])
        can_guess = band_label == 'Ks' and sat_guess_label in flux_rate_splines

        if sat_label in cache_saturation:
            pixel_rate = cache_saturation[sat_label]['brightest_pixel_rate']
            full_well = cache_saturation[sat_label]['full_well']
            saturation_text = jwst._print_pandeia_saturation(
                inst, subarray, readout, ngroup, pixel_rate, full_well,
                format='html',
            )
            report_text += f'<br>{saturation_text}'
        elif can_guess:
            cs = flux_rate_splines[sat_guess_label]
            estimated_rate = 10**cs(norm_mag)
            full_well = full_wells[sat_guess_label]
            saturation_text = jwst._print_pandeia_saturation(
                inst, subarray, readout, ngroup, estimated_rate, full_well,
                format='html',
            )
            report_text += f'<br>{saturation_text}'

        if tso_label in tso_runs[run_type]:
            tso_run = tso_runs[run_type][tso_label]
            warnings = tso_run['warnings']
            if transit_dur == tso_run['transit_dur']:
                report_text += f'<br><br>{tso_run["stats"]}'
        warning_text.set(warnings)
        return ui.HTML(f'<pre>{report_text}</pre>')


    @render.text
    @reactive.event(warning_text)
    def warnings():
        warnings = warning_text.get()
        if len(warnings) == 0:
            return 'No warnings'
        text = ''
        for warn_label, warn_text in warnings.items():
            warn_text = warn_text.replace(
                '<font color=red><b>TA MAY FAIL</b></font>',
                'TA MAY FAIL',
            )
            warn = textwrap.fill(
                f'- {warn_text}',
                subsequent_indent='  ',
                width=60,
            )
            text += warn + '\n\n'
        return text


    @reactive.effect
    @reactive.event(esasky_command)
    async def _():
        commands = esasky_command.get()
        if commands is None:
            return
        if not isinstance(commands, list):
            commands = [commands]
        for command in commands:
            command = json.dumps(command)
            await session.send_custom_message(
                "update_esasky",
                {"command": command},
            )


    @reactive.Effect
    @reactive.event(input.search_gaia_ta)
    def _():
        name = input.target.get()
        target = catalog.get_target(name, is_transit=None, is_confirmed=None)
        if target is None:
            return

        query = cat.fetch_gaia_targets(
            target.ra, target.dec, max_separation=80.0, raise_errors=False,
        )
        if isinstance(query, str):
            msg = ui.HTML(
                "The Gaia astroquery request failed :(<br>"
                "Check the terminal for more info"
            )
            ui.notification_show(msg, type="error", duration=5)
            return

        cache_acquisition[target.host] = {'targets': query, 'selected': None}
        acq_target_list.set(query)
        current_acq_science_target.set(name)
        success = "Nearby targets found!  Open the '*Acquisition targets*' tab"
        ui.notification_show(ui.markdown(success), type="message", duration=5)

        circle = u.esasky_js_circle(target.ra, target.dec, radius=80.0)
        ta_catalog = u.esasky_js_catalog(query)
        esasky_command.set([ta_catalog, circle])


    @render.data_frame
    @reactive.event(acq_target_list, input.target)
    def acquisition_targets():
        """
        Display TA list, gets triggered only when tab is shown
        """
        name = input.target.get()
        target = catalog.get_target(name, is_transit=None, is_confirmed=None)
        if target is None:
            return render.DataGrid(pd.DataFrame([]))
        if target.host not in cache_acquisition:
            # Do I need to? gets wiped anyway
            ui.update_select('ta_sed', choices=[])
            return render.DataGrid(pd.DataFrame([]))

        ta_list = cache_acquisition[target.host]['targets']
        acq_target_list.set(ta_list)
        names, G_mag, t_eff, log_g, ra, dec, separation = ta_list
        data_df = {
            'Gaia DR3 target': [name[9:] for name in names],
            'G_mag': [f'{mag:5.2f}' for mag in G_mag],
            'separation (")': [f'{sep:.3f}' for sep in separation],
            'T_eff (K)': [f'{temp:.1f}' for temp in t_eff],
            'log(g)': [f'{grav:.2f}' for grav in log_g],
            'RA (deg)': [f'{r:.4f}' for r in ra],
            'dec (deg)': [f'{d:.4f}' for d in dec],
        }
        acquisition_df = pd.DataFrame(data=data_df)
        return render.DataGrid(
            acquisition_df,
            selection_mode="row",
            height='370px',
            summary=True,
        )


    @reactive.effect
    def select_ta_row():
        """
        Gets triggrered all the time. Can I limit it to true unser-interactions
        with acquisition_targets
        """
        acq_target_list.get()
        name = input.target.get()
        target = catalog.get_target(name, is_transit=None, is_confirmed=None)

        if target is None or target.host not in cache_acquisition:
            ui.update_select('ta_sed', choices=[])
            return
        target_list = cache_acquisition[target.host]['targets']

        # TBD: if acquisition_targets changed, a previous non-zero cell_selection()
        # will leak into the new dataframe, which will de-synchronize
        # cache_acquisition[target.host]['selected']
        df = acquisition_targets.cell_selection()
        if df is None or len(df['rows'])==0:
            return

        df_data = acquisition_targets.data()
        current_data = [f'Gaia DR3 {id}' for id in df_data["Gaia DR3 target"]]
        if current_data[0] != target_list[0][0]:
            acquisition_targets._reset_reactives()
            return

        cache_acquisition[target.host]['selected'] = idx = df['rows'][0]
        target_name = target_list[0][idx]
        t_eff = target_list[2][idx]
        log_g = target_list[3][idx]
        i = jwst.find_closest_sed(t_eff, log_g, p_teff, p_logg)
        chosen_sed = p_keys[i]
        ui.update_select('ta_sed', choices=phoenix_dict, selected=chosen_sed)

        deselect_targets = {'event': 'deselectAllShapes'}
        select_acq_target = {
            'event': 'selectShape',
            'content': {
                'overlayName': 'Nearby Gaia sources',
                'shapeName': target_name
            }
        }
        esasky_command.set([deselect_targets, select_acq_target])


    # TBD: rename
    @reactive.effect
    @reactive.event(input.get_acquisition_target)
    def _():
        name = input.target.get()
        target = catalog.get_target(name, is_transit=None, is_confirmed=None)
        if target is None:
            return
        if target.host not in cache_acquisition:
            error_msg = ui.markdown(
                "First click the '*Search nearby targets*' button, then select "
                "a target from the '*Acquisition targets*' tab"
            )
            ui.notification_show(error_msg, type="warning", duration=5)
            return

        selected = acquisition_targets.cell_selection()['rows']
        if len(selected) == 0:
            error_msg = ui.markdown(
                "First select a target from the '*Acquisition targets*' tab"
            )
            ui.notification_show(error_msg, type="warning", duration=5)
            return
        target_list = cache_acquisition[target.host]['targets']
        names, G_mag, t_eff, log_g, ra, dec, separation = target_list
        idx = selected[0]
        text = (
            f"\nacq_target = {repr(names[idx])}\n"
            f"gaia_mag = {G_mag[idx]}\n"
            f"separation = {separation[idx]}\n"
            f"t_eff = {t_eff[idx]}\n"
            f"log_g = {log_g[idx]}\n"
            f"ra = {ra[idx]}\n"
            f"dec = {dec[idx]}"
        )
        print(text)


app = App(app_ui, server)

