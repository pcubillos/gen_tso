# Copyright (c) 2024 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

import os
import sys
import pickle
from pathlib import Path
import textwrap

import numpy as np
import scipy.interpolate as si
import pandas as pd
import faicons as fa
from htmltools import HTML
import plotly.graph_objects as go
from shiny import ui, render, reactive, req, App
from shinywidgets import output_widget, render_plotly

from gen_tso import catalogs as cat
from gen_tso import pandeia_io as jwst
from gen_tso import plotly_io as tplots
from gen_tso import custom_shiny as cs
from gen_tso.utils import (
    ROOT, collect_spectra, read_spectrum_file, pretty_print_target,
)
import gen_tso.catalogs.utils as u
from gen_tso.pandeia_io.pandeia_defaults import (
    get_detector,
    make_detector_label,
    make_saturation_label,
)

# Catalog of known exoplanets (and candidate planets)
catalog = cat.Catalog()
nplanets = len(catalog.targets)
is_jwst = np.array([target.is_jwst for target in catalog.targets])
is_transit = np.array([target.is_transiting for target in catalog.targets])
is_confirmed = np.array([target.is_confirmed for target in catalog.targets])


# Catalog of stellar SEDs:
p_keys, p_models, p_teff, p_logg = jwst.load_sed_list('phoenix')
k_keys, k_models, k_teff, k_logg = jwst.load_sed_list('k93models')

phoenix_dict = {model:key for key,model in zip(p_keys, p_models)}
kurucz_dict = {model:key for key,model in zip(k_keys, k_models)}
sed_dict = {
    'phoenix': phoenix_dict,
    'kurucz': kurucz_dict,
}

bands_dict = {
    'J mag': '2mass,j',
    'H mag': '2mass,h',
    'Ks mag': '2mass,ks',
    'Gaia mag': 'gaia,g',
    'V mag': 'johnson,v',
}
detectors = jwst.generate_all_instruments()
instruments = np.unique([det.instrument for det in detectors])
filter_throughputs = jwst.filter_throughputs()

tso_runs = {
    'Transit': {},
    'Eclipse': {},
    'Acquisition': {},
}

def make_tso_labels(tso_runs):
    tso_labels = dict(
        Transit={},
        Eclipse={},
        Acquisition={},
    )
    for key, runs in tso_runs.items():
        for tso_label, tso_run in runs.items():
            tso_key = f'{key}_{tso_label}'
            tso_labels[key][tso_key] = tso_run['label']
    return tso_labels


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
    'sed': [],
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
    # for label, model in sed_models.items():
        # 'depth' --> 'flux'
        #spectra['sed'][label] = {'wl': wl, 'depth': depth}
        #user_spectra['sed'].append(label)


nasa_url = 'https://exoplanetarchive.ipac.caltech.edu/overview'
trexolists_url = 'https://www.stsci.edu/~nnikolov/TrExoLiSTS/JWST/trexolists.html'
css_file = f'{ROOT}data/style.css'

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
    #ui.markdown("""
    #    This app is based on [shiny][0].
    #    [0]: https://shiny.posit.co/py/api/core
    #    """),
    ui.layout_columns(
        ui.markdown(
            "## **Gen TSO**: A general exoplanet ETC for JWST "
            "time-series observations",
        ),
        ui.output_image("tso_logo", height='50px', inline=True),
        col_widths=(11,1),
        fixed_width=False,
        fill=False,
        fillable=True,
    ),
    ui.include_css(css_file),

    # Instrument and detector modes:
    ui.layout_columns(
        cs.navset_card_tab_jwst(
            ['MIRI', 'NIRCam', 'NIRISS', 'NIRSpec'],
            id="instrument",
            selected='NIRCam',
            header="Select an instrument and detector",
            footer=ui.input_select(
                "mode",
                "",
                choices = {},
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
                    #width='450px',
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
            ui.input_task_button(
                id="run_pandeia",
                label="Run Pandeia",
                label_busy="processing...",
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
                    ui.input_text("teff", "", value='1400.0'),
                    # Row 2
                    ui.p("log(g):"),
                    ui.input_text("logg", "", value='4.5'),
                    # Row 3
                    ui.input_select(
                        id='magnitude_band',
                        label='',
                        choices=list(bands_dict.keys()),
                        selected='Ks mag',
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
                ),
                ui.output_ui('choose_sed'),
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
                "Observation",
                ui.layout_column_wrap(
                    # Row 1
                    ui.p("Type:"),
                    ui.input_select(
                        id='geometry',
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
                            id="tplanet",
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
            ui.card_header("Detector setup", class_="bg-primary"),
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
                        choices={},
                    ),
                ),
                class_="px-2 pt-2 pb-0 m-0",
            ),
            # groups / integs
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
                    ui.input_switch("integs_switch", "Match obs. duration", False),
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
                        ui.input_action_button(
                            id="search_gaia_ta",
                            label="Search nearby targets",
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
                            id="perform_ta_calculation",
                            label="Run TA Pandeia",
                            class_='btn btn-outline-secondary btn-sm',
                            # TBD: set a style from CSS file independently of id
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
                    ui.popover(
                        ui.span(
                            fa.icon_svg("gear"),
                            style="position:absolute; top: 5px; right: 7px;",
                        ),
                        "Show filter throughputs",
                        ui.input_radio_buttons(
                            id="filter_filter",
                            label=None,
                            choices=["none", "all"],
                            inline=True,
                        ),
                        placement="right",
                        #placement="top",
                        id="filter_popover",
                    ),
                    cs.custom_card(
                        output_widget("plotly_filters", fillable=True),
                        body_args=dict(class_='m-0 p-0'),
                        full_screen=True,
                        height='250px',
                    ),
                ),
                ui.nav_panel(
                    "Sky view",
                    ui.output_ui('esasky_card'),
                ),
                ui.nav_panel(
                    "Stellar SED",
                    ui.popover(
                        ui.span(
                            fa.icon_svg("gear"),
                            style="position:absolute; top: 5px; right: 7px;",
                        ),
                        ui.input_numeric(
                            id='plot_sed_resolution',
                            label='Resolution:',
                            value=0.0,
                            min=10.0, max=3000.0, step=25.0,
                        ),
                        ui.input_select(
                            "plot_sed_units",
                            "Flux units:",
                            choices = ['mJy'],
                            selected='mJy',
                        ),
                        ui.input_select(
                            "plot_sed_xscale",
                            "Wavelength axis:",
                            choices = ['linear', 'log'],
                            selected='log',
                        ),
                        placement="right",
                        id="sed_popover",
                    ),
                    cs.custom_card(
                        output_widget("plotly_sed", fillable=True),
                        body_args=dict(padding='0px'),
                        full_screen=True,
                        height='300px',
                    ),
                ),
                ui.nav_panel(
                    ui.output_text('transit_depth_label'),
                    ui.popover(
                        ui.span(
                            fa.icon_svg("gear"),
                            style="position:absolute; top: 5px; right: 7px;",
                        ),
                        ui.input_numeric(
                            id='depth_resolution',
                            label='Resolution:',
                            value=250.0,
                            min=10.0, max=3000.0, step=25.0,
                        ),
                        ui.input_select(
                            "plot_depth_units",
                            "Depth units:",
                            choices = depth_units,
                            selected='percent',
                        ),
                        ui.input_select(
                            "plot_depth_xscale",
                            "Wavelength axis:",
                            choices = ['linear', 'log'],
                            selected='log',
                        ),
                        placement="right",
                        id="depth_popover",
                    ),
                    cs.custom_card(
                        output_widget("plotly_depth", fillable=True),
                        body_args=dict(padding='0px'),
                        full_screen=True,
                        height='300px',
                    ),
                ),
                ui.nav_panel(
                    "TSO",
                    ui.popover(
                        ui.span(
                            fa.icon_svg("gear"),
                            style="position:absolute; top: 5px; right: 7px;",
                        ),
                        ui.input_numeric(
                            id='n_obs',
                            label='Number of observations:',
                            value=1.0,
                            min=1.0, max=3000.0, step=1.0,
                        ),
                        ui.input_numeric(
                            id='tso_resolution',
                            label='Observation resolution:',
                            value=250.0,
                            min=25.0, max=3000.0, step=25.0,
                        ),
                        ui.input_select(
                            "plot_tso_units",
                            "Depth units:",
                            choices = depth_units,
                            selected='percent',
                        ),
                        ui.input_select(
                            "plot_tso_xscale",
                            "Wavelength axis:",
                            choices = ['linear', 'log'],
                        ),
                        ui.input_select(
                            "plot_tso_xrange",
                            "Wavelength range:",
                            choices = ['auto', 'JWST (0.6--12.0 um)'],
                        ),
                        placement="right",
                        id="tso_popover",
                    ),
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
                        ui.output_ui(id="exp_time"),
                        style="font-family: monospace; font-size:medium;",
                    ),
                ),
                ui.nav_panel(
                    ui.output_ui('warnings_label'),
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
)


def planet_model_name(input):
    """
    Get the planet model name based on the transit/eclipse depth values.

    Returns
    -------
    depth_label: String
        A string representation of the depth model.
    """
    model_type = input.planet_model_type.get()
    if model_type == 'Input':
        return input.depth.get()
    elif model_type == 'Flat':
        transit_depth = input.transit_depth.get()
        return f'Flat transit ({transit_depth:.3f}%)'
    elif model_type == 'Blackbody':
        eclipse_depth = input.eclipse_depth.get()
        t_planet = input.tplanet.get()
        return f'Blackbody({t_planet:.0f}K, rprs\u00b2={eclipse_depth:.3f}%)'


def get_throughput(input):
    inst = req(input.instrument).get().lower()
    mode = req(input.mode).get()
    if mode == 'target_acq':
        obs_type = 'acquisition'
    else:
        obs_type = 'spectroscopy'

    subarray = input.subarray.get()

    if mode == 'lrsslitless':
        filter = 'None'
    elif mode == 'mrs_ts':
        filter = input.disperser.get()
    else:
        filter = input.filter.get()

    if subarray not in filter_throughputs[obs_type][inst]:
        return None
    if filter not in filter_throughputs[obs_type][inst][subarray]:
        return None
    return filter_throughputs[obs_type][inst][subarray][filter]


def get_auto_sed(input):
    sed_type = input.sed_type()
    if sed_type == 'kurucz':
        m_models, m_teff, m_logg = k_models, k_teff, k_logg
    elif sed_type == 'phoenix':
        m_models, m_teff, m_logg = p_models, p_teff, p_logg

    try:
        teff = float(input.teff.get())
        logg = float(input.logg.get())
    except ValueError:
        return m_models, None
    idx = jwst.find_closest_sed(m_teff, m_logg, teff, logg)
    chosen_sed = m_models[idx]
    return m_models, chosen_sed


def parse_sed(input, target_acq_mag=None):
    """Extract SED parameters"""
    if target_acq_mag is None:
        sed_type = input.sed_type()
        norm_band = bands_dict[input.magnitude_band()]
        norm_magnitude = float(input.magnitude())
    else:
        sed_type = 'phoenix'
        norm_band = 'gaia,g'
        norm_magnitude = target_acq_mag

    if sed_type in ['phoenix', 'kurucz']:
        if target_acq_mag is None:
            sed = input.sed.get()
        else:
            sed = input.ta_sed.get()
        if sed not in sed_dict[sed_type]:
            return None, None, None, None, None
        sed_model = sed_dict[sed_type][sed]
        model_label = f'{sed_type}_{sed_model}'
    elif sed_type == 'blackbody':
        sed_model = float(input.teff.get())
        model_label = f'{sed_type}_{sed_model:.0f}K'
    elif sed_type == 'input':
        model_label = sed_model

    if sed_type == 'kurucz':
        sed_type = 'k93models'

    # Make a label
    for name,band in bands_dict.items():
        if band == norm_band:
            band_label = f'{norm_magnitude:.2f}_{name.split()[0]}'
    sed_label = f'{model_label}_{band_label}'

    return sed_type, sed_model, norm_band, norm_magnitude, sed_label


def parse_depth_model(input):
    """
    Parse transit/eclipse model name based on current state.
    Calculate or extract model.
    """
    model_type = input.planet_model_type.get()
    depth_label = planet_model_name(input)
    obs_geometry = input.geometry.get()

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
        t_planet = input.tplanet.get()
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


def server(input, output, session):
    sky_view_src = reactive.Value('')
    bookmarked_sed = reactive.Value(False)
    bookmarked_depth = reactive.Value(False)
    saturation_label = reactive.Value(None)
    update_depth_flag = reactive.Value(None)
    uploaded_units = reactive.Value(None)
    warning_text = reactive.Value('')
    machine_readable_info = reactive.Value(False)
    acq_target_list = reactive.Value(None)
    current_science_target = reactive.Value(None)

    @render.image
    def tso_logo():
        dir = Path(__file__).resolve().parent.parent
        img = {
            "src": str(dir / "docs/images/gen_tso_logo.png"),
            "height": "50px",
        }
        return img

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Instrument and detector modes
    @reactive.Effect
    @reactive.event(input.instrument)
    def _():
        inst = input.instrument.get()
        print(f"You selected me: {inst}")
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
        ui.update_select('mode', choices=choices)


    @reactive.Effect
    @reactive.event(input.mode)
    def _():
        inst = req(input.instrument).get()
        mode = input.mode.get()
        detector = get_detector(inst, mode, detectors)

        ui.update_select(
            'disperser',
            label=detector.disperser_label,
            choices=detector.dispersers,
            selected=detector.default_disperser,
        )
        ui.update_select(
            'filter',
            label=detector.filter_label,
            choices=detector.filters,
            selected=detector.default_filter,
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

    @reactive.Effect
    @reactive.event(input.disperser, input.filter)
    def update_subarray():
        inst = req(input.instrument).get()
        mode = input.mode.get()
        detector = get_detector(inst, mode, detectors)

        if mode == 'bots':
            disperser = input.filter.get().split('/')[0]
        else:
            disperser = input.disperser.get()
        choices = detector.get_constrained_val('subarrays', disperser=disperser)

        subarray = input.subarray.get()
        if subarray not in choices:
            subarray = detector.default_subarray

        ui.update_select(
            'subarray',
            choices=choices,
            selected=subarray,
        )

    @reactive.Effect
    @reactive.event(input.disperser)
    def update_readout():
        inst = req(input.instrument).get()
        mode = input.mode.get()
        detector = get_detector(inst, mode, detectors)

        disperser = input.disperser.get()
        choices = detector.get_constrained_val('readouts', disperser=disperser)

        readout = input.readout.get()
        if readout not in choices:
            readout = detector.default_readout

        ui.update_select(
            'readout',
            choices=choices,
            selected=readout,
        )

    @reactive.Effect
    @reactive.event(input.run_pandeia)
    def run_pandeia():
        inst = input.instrument.get().lower()
        mode = input.mode.get()
        disperser = input.disperser.get()
        filter = input.filter.get()
        subarray = input.subarray.get()
        readout = input.readout.get()
        order = input.order.get()
        ngroup = int(input.groups.get())
        nint = input.integrations.get()
        aperture = None

        detector = get_detector(inst, mode, detectors)
        inst_label = detector.instrument_label(disperser, filter)

        run_is_tso = True
        # Front-end to back-end exceptions:
        if mode == 'bots':
            disperser, filter = filter.split('/')
        if mode == 'soss':
            order = [int(val) for val in order.split()]
        else:
            order = None
        if mode == 'mrs_ts':
            aperture = ['ch1', 'ch2', 'ch3', 'ch4']
        if mode == 'target_acq':
            aperture = input.disperser.get()
            disperser = None
            nint = 1
            run_is_tso = False

        obs_geometry = input.geometry.get()
        transit_dur = float(input.t_dur.get())
        obs_dur = float(input.obs_dur.get())
        exp_time = jwst.exposure_time(
            inst, subarray, readout, ngroup, nint,
        )
        in_transit_integs, in_transit_time = jwst.bin_search_exposure_time(
            inst, subarray, readout, ngroup, transit_dur,
        )
        if mode != 'target_acq' and in_transit_integs > nint:
            error_msg = ui.markdown(
                f"**Warning:**<br>observation time for **{nint} integration"
                f"(s)** is less than the {obs_geometry} time.  Running "
                "a perform_calculation()"
            )
            ui.notification_show(error_msg, type="warning", duration=5)
            run_is_tso = False

        sed_type, sed_model, norm_band, norm_mag, sed_label = parse_sed(input)
        pando = jwst.PandeiaCalculation(inst, mode)
        pando.set_scene(sed_type, sed_model, norm_band, norm_mag)

        if not run_is_tso:
            tso = pando.perform_calculation(
                ngroup, nint, disperser, filter, subarray, readout, aperture,
            )
            run_type = 'Acquisition'
            depth_label = ''
        else:
            depth_label, wl, depth = parse_depth_model(input)
            if depth_label is None:
                error_msg = ui.markdown(
                    f"**Error:**<br>no {obs_geometry} depth model "
                    "to simulate"
                )
                ui.notification_show(error_msg, type="error", duration=5)
                return
            run_type = obs_geometry.capitalize()
            if depth_label not in spectra:
                spectra[obs_geometry][depth_label] = {'wl': wl, 'depth': depth}
                bookmarked_spectra[obs_geometry].append(depth_label)
            depth_model = [wl, depth]

            obs_dur = exp_time / 3600.0
            tso = pando.tso_calculation(
                obs_geometry, transit_dur, obs_dur, depth_model,
                ngroup, disperser, filter, subarray, readout, aperture, order,
            )

        if run_is_tso:
            success = "TSO model simulated!"
        else:
            success = "Pandeia calculation done!"
        ui.notification_show(success, type="message", duration=2)

        detector_label = make_detector_label(
            inst, mode, disperser, filter, subarray, readout, order,
        )
        group_ints = f'({ngroup} G, {nint} I)'
        tso_label = (
            f'{detector_label} {group_ints} / {sed_label} / {depth_label}'
        )

        tso_run = dict(
            # The detector
            inst=inst,
            mode=mode,
            inst_label=inst_label,
            label=tso_label,
            # The SED
            sed_type=sed_type,
            sed_model=sed_model,
            norm_band=norm_band,
            norm_mag=norm_mag,
            # The instrumental setting
            aperture=aperture,
            disperser=disperser,
            filter=filter,
            subarray=subarray,
            readout=readout,
            order=order,
            ngroup=ngroup,
            # The outputs
            tso=tso,
        )
        if run_is_tso:
            # The planet
            tso_run['t_dur'] = transit_dur
            tso_run['obs_dur'] = obs_dur
            tso_run['depth_model_name'] = depth_label
            tso_run['depth_model'] = depth_model
            if isinstance(tso, list):
                reports = (
                    [report['report_in']['scalar'] for report in tso],
                    [report['report_out']['scalar'] for report in tso],
                )
                warnings = tso[0]['report_in']['warnings']
                # TBD: Consider warnings in other TSO reports?
            else:
                reports = (
                    tso['report_in']['scalar'],
                    tso['report_out']['scalar'],
                )
                warnings = tso['report_in']['warnings']
        else:
            reports = tso['scalar'], None
            warnings = tso['warnings']

        if run_is_tso or mode=='target_acq':
            tso_runs[run_type][tso_label] = tso_run
            tso_labels = make_tso_labels(tso_runs)
            ui.update_select('display_tso_run', choices=tso_labels)

        # Update report
        sat_label = make_saturation_label(
            mode, disperser, filter, subarray, order, sed_label,
        )
        pixel_rate, full_well = jwst.saturation_level(tso, get_max=True)
        cache_saturation[sat_label] = dict(
            brightest_pixel_rate=pixel_rate,
            full_well=full_well,
            inst=inst,
            mode=mode,
            reports=reports,
            warnings=warnings,
        )
        saturation_label.set(sat_label)

        if len(warnings) > 0:
            warning_text.set(warnings)
        else:
            warning_text.set('')

        print(inst, mode, disperser, filter, subarray, readout, order)
        print(sed_type, sed_model, norm_band, repr(norm_mag))
        print('~~ TSO done! ~~')


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
        filename = f'tso_{inst}.pickle'

        m = ui.modal(
            ui.input_text(
                id='tso_save_file',
                label='Save TSO run to file:',
                value=filename,
                placeholder=tso_label,
                width='100%',
            ),
            ui.HTML(f"Located in current folder:<br>'{current_dir}/'<br>"),
            # TBD: I wish this could be used to browse a folder :(
            #ui.input_file(
            #    id="save_file_x",
            #    label="Into this folder:",
            #    button_label="Browse",
            #    multiple=True,
            #    width='100%',
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

        filename = input.tso_save_file.get()
        if filename.strip() == '':
            filename = 'tso_run.pickle'
        savefile = Path(f'{current_dir}/{filename}')
        if savefile.suffix == '':
            savefile = savefile.parent / f'{savefile.name}.pickle'
        if savefile.exists():
            stem = str(savefile.parent / savefile.stem)
            extension = savefile.suffix
            i = 1
            savefile = Path(f'{stem}{i}{extension}')
            while savefile.exists():
                i += 1
                savefile = Path(f'{stem}{i}{extension}')

        with open(savefile, 'wb') as handle:
            pickle.dump(tso_run, handle, protocol=4)
        ui.modal_remove()
        ui.notification_show(
            f"TSO model saved to file: '{savefile}'",
            type="message",
            duration=5,
        )

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Target
    @reactive.Effect
    @reactive.event(input.target_filter)
    def _():
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
            return ui.span('Known target?')

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
            ra, dec = target.trexo_ra_dec
            url = f'{trexolists_url}?ra={ra}&dec={dec}'
            trexolists_tooltip = ui.tooltip(
                ui.tags.a(
                    fa.icon_svg("circle-info", fill='goldenrod'),
                    href=url,
                    target="_blank",
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

    @reactive.Effect
    @reactive.event(input.target)
    def _():
        """Set known-target properties"""
        name = input.target.get()
        target = catalog.get_target(name, is_transit=None, is_confirmed=None)
        if target is None:
            return
        if name in target.aliases:
            ui.update_selectize('target', selected=target.planet)
        teff = u.as_str(target.teff, '.1f', '')
        log_g = u.as_str(target.logg_star, '.2f', '')
        t_dur = u.as_str(target.transit_dur, '.3f', '')

        ui.update_text('teff', value=teff)
        ui.update_text('logg', value=log_g)
        ui.update_select('magnitude_band', selected='Ks mag')
        ui.update_text('magnitude', value=f'{target.ks_mag:.3f}')
        ui.update_text('t_dur', value=t_dur)

        sky_view_src.set(
            f'https://sky.esa.int/esasky/?target={target.ra}%20{target.dec}'
            '&fov=0.2&sci=true'
        )

    @render.ui
    @reactive.event(input.sed_type, input.teff, input.logg)
    def choose_sed():
        sed_type = input.sed_type.get()
        if sed_type in ['phoenix', 'kurucz']:
            m_models, chosen_sed = get_auto_sed(input)
            choices = list(m_models)
            selected = chosen_sed
        elif sed_type == 'blackbody':
            if input.teff.get() == '':
                teff = 0.0
            else:
                teff = float(input.teff.get())
            selected = f' Blackbody (Teff={teff:.0f} K)'
            choices = [selected]
        elif sed_type == 'input':
            choices = list(spectra['sed'])
            selected = None

        return ui.input_select(
            id="sed",
            label="",
            choices=choices,
            selected=selected,
        )

    @render.ui
    @reactive.event(
        bookmarked_sed, input.sed,
        input.teff, input.magnitude_band, input.magnitude,
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
        is_bookmarked = not bookmarked_sed.get()
        bookmarked_sed.set(is_bookmarked)
        sed_type, sed_model, norm_band, norm_mag, sed_label = parse_sed(input)
        if is_bookmarked:
            scene = jwst.make_scene(sed_type, sed_model, norm_band, norm_mag)
            wl, flux = jwst.extract_sed(scene, wl_range=[0.3,30.0])
            spectra['sed'][sed_label] = {'wl': wl, 'flux': flux}
            bookmarked_spectra['sed'].append(sed_label)
        else:
            bookmarked_spectra['sed'].remove(sed_label)
            if sed_label not in user_spectra['sed']:
                spectra['sed'].pop(sed_label)


    @render.ui
    @reactive.event(
        bookmarked_depth, input.geometry, input.planet_model_type,
        input.depth, input.transit_depth, input.eclipse_depth, input.tplanet,
    )
    def depth_label_text():
        """Set depth model label"""
        obs_geometry = input.geometry.get()
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
        obs_geometry = input.geometry.get()
        depth_label = planet_model_name(input)
        if depth_label is None:
            ui.notification_show(
                f"No {obs_geometry} depth model to bookmark",
                type="error",
                duration=5,
            )
            return
        is_bookmarked = not bookmarked_depth.get()
        bookmarked_depth.set(is_bookmarked)
        if is_bookmarked:
            bookmarked_spectra[obs_geometry].append(depth_label)
            depth_label, wl, depth = parse_depth_model(input)
            spectra[obs_geometry][depth_label] = {'wl': wl, 'depth': depth}
        else:
            bookmarked_spectra[obs_geometry].remove(depth_label)
            if depth_label not in user_spectra[obs_geometry]:
                spectra[obs_geometry].pop(depth_label)


    @reactive.effect
    @reactive.event(input.geometry, update_depth_flag)
    def _():
        obs_geometry = input.geometry.get()

        selected = input.planet_model_type.get()
        if obs_geometry == 'transit':
            choices = ['Flat', 'Input']
        elif obs_geometry == 'eclipse':
            choices = ['Blackbody', 'Input']
        if selected not in choices:
            selected = choices[0]
        ui.update_select(
            id="planet_model_type",
            choices=choices,
            selected=selected,
        )

        ui.update_select(id="depth", choices=user_spectra[obs_geometry])

        if len(user_spectra[obs_geometry]) > 0:
            tooltip_text = ''
        elif obs_geometry == 'transit':
            tooltip_text = f'Upload a {obs_geometry} depth spectrum'
        elif obs_geometry == 'eclipse':
            tooltip_text = f'Upload an {obs_geometry} depth spectrum'
        ui.update_tooltip('depth_tooltip', tooltip_text)


    @render.text
    @reactive.event(input.geometry)
    def transit_dur_label():
        obs_geometry = input.geometry.get().capitalize()
        return f"{obs_geometry[0]}_dur (h):"

    @render.text
    @reactive.event(input.geometry)
    def transit_depth_label():
        obs_geometry = input.geometry.get().capitalize()
        return f"{obs_geometry} depth"

    @render.ui
    @reactive.event(warning_text)
    def warnings_label():
        warnings = warning_text.get()
        if warnings == '':
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
        t_total = 1.0 + settling + transit_dur + 2.0*baseline
        ui.update_text('obs_dur', value=f'{t_total:.2f}')


    @reactive.Effect
    @reactive.event(input.target)
    def _():
        name = input.target.get()
        target = catalog.get_target(name, is_transit=None, is_confirmed=None)
        if target is None:
            rprs_square = 0.0
            teq_planet = 1000.0
        else:
            teq_planet = np.round(target.eq_temp, decimals=1)
            if np.isnan(teq_planet):
                teq_planet = 0.0
            rprs_square = target.rprs**2.0
            if np.isnan(rprs_square):
                rprs_square = 0.0
        rprs_square_percent = np.round(100*rprs_square, decimals=4)
        ui.update_numeric(id="transit_depth", value=rprs_square_percent)
        ui.update_numeric(id="eclipse_depth", value=rprs_square_percent)
        ui.update_numeric(id='tplanet', value=teq_planet)


    @reactive.effect
    @reactive.event(input.upload_sed)
    def _():
        m = ui.modal(
            ui.input_file(
                # Need to change the id to avoid conflict with upload_depth
                id="upload_file",
                label=ui.markdown(
                    "Input files must be plan-text files with two columns, "
                    "the first one being the wavelength (microns) and "
                    "the second one the stellar SED. "
                    "**Make sure the input units are correct!**"
                ),
                button_label="Browse",
                multiple=True,
                width='100%',
            ),
            ui.input_radio_buttons(
                id="upload_units",
                label='Flux units:',
                choices=sed_units,
                width='100%',
            ),
            title="Upload Spectrum",
            easy_close=True,
        )
        ui.modal_show(m)


    @reactive.effect
    @reactive.event(input.upload_depth)
    def _():
        obs_geometry = input.geometry.get()
        m = ui.modal(
            ui.input_file(
                id="upload_file",
                label=ui.markdown(
                    "Input files must be plan-text files with two columns, "
                    "the first one being the wavelength (microns) and "
                    f"the second one the {obs_geometry} depth. "
                    "**Make sure the input units are correct!**"
                ),
                button_label="Browse",
                multiple=True,
                width='100%',
            ),
            ui.input_radio_buttons(
                id="upload_units",
                label='Depth units:',
                choices=depth_units,
                width='100%',
            ),
            title="Upload Spectrum",
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
            print('No new model!')
            return

        # The units tell this function SED or depth spectrum:
        units = uploaded_units.get()
        label, wl, depth = read_spectrum_file(
            new_model[0]['datapath'], on_fail='warning',
        )
        if wl is None:
            # TBD: capture and pop up the warning
            return
        # Need to manually handle the label
        label = new_model[0]['name']
        if label.endswith('.dat') or label.endswith('.txt'):
            label = label[0:-4]

        if units in depth_units:
            obs_geometry = input.geometry.get()
            # TBD: convert depth units
            spectra[obs_geometry][label] = {'wl': wl, 'depth': depth}
            user_spectra[obs_geometry].append(label)
            bookmarked_spectra[obs_geometry].append(label)
            if input.planet_model_type.get() != 'Input':
                return
            # Trigger update choose_depth
            update_depth_flag.set(label)
        elif units in sed_units:
            pass


    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Detector setup
    @reactive.Effect
    @reactive.event(input.subarray)
    def set_soss_orders():
        if input.subarray.get() == 'substrip96':
            orders = {'1': '1'}
        else:
            orders = {
                '1': '1',
                '2': '2',
                '1 2': '1 and 2',
            }
        ui.update_select(id='order', choices=orders)


    @render.ui
    @reactive.event(input.mode, input.subarray)
    def groups_input():
        mode = input.mode.get()
        if mode == 'target_acq':
            inst = req(input.instrument).get()
            detector = get_detector(inst, mode, detectors)
            subarray = input.subarray.get()
            choices = detector.get_constrained_val('groups', subarray=subarray)

            return ui.input_select(
                id="groups",
                label="",
                choices=choices,
            )
        else:
            return ui.input_numeric(
                id="groups",
                label='',
                value=2,
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
        if mode == 'bots':
            disperser, filter = filter.split('/')
        if mode == 'soss':
            order = [int(val) for val in order.split()]
        if mode == 'mrs_ts':
            aperture = ['ch1', 'ch2', 'ch3', 'ch4']
        if mode == 'target_acq':
            ngroup = int(input.groups.get())
            aperture = input.disperser.get()
            disperser = None
        else:
            ngroup = 2

        sat_label = make_saturation_label(
            mode, disperser, filter, subarray, order, sed_label,
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
        # This reactive variable enforces a re-rendering of exp_time
        saturation_label.set(sat_label)


    @reactive.Effect
    @reactive.event(
        input.integs_switch, input.obs_dur, input.mode,
        input.instrument, input.groups, input.readout, input.subarray,
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
        ngroup = input.groups.get()
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

        if mode == 'lrsslitless':
            filter = 'None'
        if mode == 'mrs_ts':
            filter = input.disperser.get()

        if mode == 'target_acq':
            throughputs = filter_throughputs['acquisition']
        else:
            throughputs = filter_throughputs['spectroscopy']

        fig = tplots.plotly_filters(
            throughputs, inst, mode, subarray, filter, show_all,
        )
        return fig


    @render.ui
    @reactive.event(sky_view_src)
    def esasky_card():
        src = sky_view_src.get()
        return cs.custom_card(
            HTML(
                '<iframe '
                'height="100%" '
                'width="100%" '
                'style="overflow" '
                f'src="{src}" '
                'frameborder="0" allowfullscreen></iframe>',
                #id=resolve_id(id),
            ),
            body_args=dict(class_='m-0 p-0', id='esasky'),
            full_screen=True,
            height='350px',
        )

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

        throughput = get_throughput(input)
        units = input.plot_sed_units.get()
        wl_scale = input.plot_sed_xscale.get()
        resolution = input.plot_sed_resolution.get()
        fig = tplots.plotly_sed_spectra(
            sed_models, model_names, current_model,
            units=units, wl_scale=wl_scale, resolution=resolution,
            throughput=throughput,
        )
        return fig


    @render_plotly
    def plotly_depth():
        input.bookmark_depth.get()  # (make panel reactive to bookmark_depth)
        obs_geometry = input.geometry.get()
        model_names = bookmarked_spectra[obs_geometry]
        nmodels = len(model_names)
        if nmodels == 0:
            return go.Figure()
        throughput = get_throughput(input)

        current_model = planet_model_name(input)
        units = input.plot_depth_units.get()
        wl_scale = input.plot_depth_xscale.get()
        resolution = input.depth_resolution.get()

        depth_models = [spectra[obs_geometry][model] for model in model_names]
        fig = tplots.plotly_depth_spectra(
            depth_models, model_names, current_model,
            units=units, wl_scale=wl_scale, resolution=resolution,
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
        n_obs = input.n_obs.get()
        resolution = input.tso_resolution.get()
        units = input.plot_tso_units.get()
        wl_scale = input.plot_tso_xscale.get()
        x_range = input.plot_tso_xrange.get()
        if x_range == 'auto':
            wl_range = None
        else:
            wl_range = [0.6, 13.0]

        tso_run = tso_runs[key][tso_label]
        planet = tso_run['depth_model_name']
        fig = tplots.plotly_tso_spectra(
            tso_run['tso'], resolution, n_obs,
            model_label=planet,
            instrument_label=tso_run['inst_label'],
            bin_widths=None,
            units=units, wl_range=wl_range, wl_scale=wl_scale,
            obs_geometry='transit',
        )
        return fig

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Results
    @render.ui
    def exp_time():
        saturation_label.get()  # enforce calc_saturation renders exp_time
        inst = input.instrument.get().lower()
        mode = input.mode.get()
        detector = get_detector(inst, mode, detectors)
        disperser = input.disperser.get()
        filter = input.filter.get()
        subarray = input.subarray.get()
        readout = input.readout.get()
        order = input.order.get()
        ngroup = input.groups.get()
        nint = input.integrations.get()
        sed_type, sed_model, norm_band, norm_mag, sed_label = parse_sed(input)

        if ngroup is None or detector is None or sed_label is None:
            return ui.HTML('<pre> </pre>')

        # Front-end to back-end exceptions:
        if mode == 'bots' and '/' in filter:
            disperser, filter = filter.split('/')
        #if mode == 'mrs_ts':
        #    aperture = ['ch1', 'ch2', 'ch3', 'ch4']
        if mode == 'target_acq':
            #aperture = input.disperser.get()
            disperser = None
            nint = 1

        ngroup = int(ngroup)
        report_text = jwst._print_pandeia_exposure(
            inst, subarray, readout, ngroup, nint,
        )

        sat_label = make_saturation_label(
            mode, disperser, filter, subarray, order, sed_label,
        )
        cached = sat_label in cache_saturation
        if cached:
            pixel_rate = cache_saturation[sat_label]['brightest_pixel_rate']
            full_well = cache_saturation[sat_label]['full_well']
            saturation_text = jwst._print_pandeia_saturation(
                inst, subarray, readout, ngroup, pixel_rate, full_well,
                format='html',
            )
            report_text += f'<br>{saturation_text}'

        if cached and 'reports' in cache_saturation[sat_label]:
            # TBD: check that groups / integs match
            report_in, report_out = cache_saturation[sat_label]['reports']
            inst = cache_saturation[sat_label]['inst']
            mode = cache_saturation[sat_label]['mode']
            stats_text = jwst._print_pandeia_stats(
                inst, mode, report_in, report_out, format='html',
            )
            report_text += f'<br><br>{stats_text}'
        return ui.HTML(f'<pre>{report_text}</pre>')

    @render.text
    @reactive.event(warning_text)
    def warnings():
        warnings = warning_text.get()
        if warnings == '':
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


    @render.data_frame
    @reactive.event(input.search_gaia_ta, input.target)
    def acquisition_targets():
        name = input.target.get()
        # Change of input.target was the trigger:
        if name != current_science_target.get() and name != '':
            current_science_target.set(name)
            acq_target_list.set(None)
            ui.update_select('ta_sed', choices=[])
            return

        target = catalog.get_target(name, is_transit=None, is_confirmed=None)
        if target is None:
            return

        names, G_mag, teff, log_g, ra, dec, separation = cat.fetch_gaia_targets(
            target.ra, target.dec, max_separation=80.0,
        )
        acq_target_list.set([names, G_mag, teff, log_g, ra, dec, separation])
        data_df = {
            'Gaia DR3 target': [name[9:] for name in names],
            'G_mag': [f'{mag:5.2f}' for mag in G_mag],
            'separation (")': [f'{sep:.3f}' for sep in separation],
            'T_eff (K)': [f'{temp:.1f}' for temp in teff],
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
    def rows():
        target_list = acq_target_list.get()
        if target_list is None:
            ui.update_select('ta_sed', choices=[])
            return
        selected = acquisition_targets.cell_selection()['rows']
        if len(selected) == 0:
            ui.update_select('ta_sed', choices=[])
            return

        idx = selected[0]
        teff = target_list[2][idx]
        log_g = target_list[3][idx]
        idx = jwst.find_closest_sed(p_teff, p_logg, teff, log_g)
        chosen_sed = p_models[idx]
        ui.update_select('ta_sed', choices=list(p_models), selected=chosen_sed)


    @reactive.effect
    @reactive.event(input.perform_ta_calculation)
    def _():
        name = input.target.get()
        target = catalog.get_target(name, is_transit=None, is_confirmed=None)
        if target is None:
            return
        target_list = acq_target_list.get()
        if target_list is None:
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

        idx = selected[0]
        gaia_mag = np.round(target_list[1][idx], 3)

        inst = input.instrument.get().lower()
        mode = 'target_acq'
        detector = get_detector(inst, mode, detectors)
        aperture = input.disperser.get()
        disperser = None
        filter = input.filter.get()
        subarray = input.subarray.get()
        readout = input.readout.get()
        order = None
        ngroup = int(input.groups.get())
        nint = 1

        sed_type, sed_model, norm_band, norm_mag, sed_label = parse_sed(
            input, target_acq_mag=gaia_mag,
        )
        pando = jwst.PandeiaCalculation(inst, mode)
        pando.set_scene(sed_type, sed_model, norm_band, norm_mag)
        tso = pando.perform_calculation(
            ngroup, nint, disperser, filter, subarray, readout, aperture,
        )

        success = "Pandeia calculation done!"
        ui.notification_show(success, type="message", duration=2)

        obs_geometry = 'acquisition'
        depth_label = ''
        detector_label = make_detector_label(
            inst, mode, disperser, filter, subarray, readout, order,
        )
        group_ints = f'({ngroup} G, {nint} I)'
        tso_label = (
            f'{detector_label} {group_ints} / {sed_label} / {depth_label}'
        )

        inst_label = detector.instrument_label(disperser, filter)
        tso_run = dict(
            # The detector
            inst=inst,
            mode=mode,
            inst_label=inst_label,
            label=tso_label,
            # The SED
            sed_type=sed_type,
            sed_model=sed_model,
            norm_band=norm_band,
            norm_mag=norm_mag,
            obs_type=obs_geometry,
            # The instrumental setting
            aperture=aperture,
            disperser=disperser,
            filter=filter,
            subarray=subarray,
            readout=readout,
            order=order,
            ngroup=ngroup,
            # The outputs
            tso=tso,
        )

        warnings = tso['warnings']
        tso_runs['Acquisition'][tso_label] = tso_run
        tso_labels = make_tso_labels(tso_runs)
        ui.update_select('display_tso_run', choices=tso_labels)

        if len(warnings) > 0:
            warning_text.set(warnings)
        else:
            warning_text.set('')


    # TBD: rename
    @reactive.effect
    @reactive.event(input.get_acquisition_target)
    def _():
        # TBD: also open ui.modal, or both?
        name = input.target.get()
        target = catalog.get_target(name, is_transit=None, is_confirmed=None)
        if target is None:
            return
        target_list = acq_target_list.get()
        if target_list is None:
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
        names, G_mag, teff, log_g, ra, dec, separation = target_list
        idx = selected[0]
        text = (
            f"acq_target = {repr(names[idx])}\n"
            f"gaia_mag = {G_mag[idx]}\n"
            f"separation = {separation[idx]}\n"
            f"teff = {teff[idx]}\n"
            f"log_g = {log_g[idx]}\n"
            f"ra = {ra[idx]}\n"
            f"dec = {dec[idx]}"
        )
        print(text)

app = App(app_ui, server)

