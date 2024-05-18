# Copyright (c) 2024 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

from collections.abc import Iterable
import os
import sys

import numpy as np
import scipy.interpolate as si

import faicons as fa
from htmltools import HTML
import plotly.graph_objects as go
from shiny import ui, render, reactive, req, App
from shinywidgets import output_widget, render_plotly

from gen_tso import catalogs as cat
from gen_tso import pandeia_io as jwst
from gen_tso import plotly_io as tplots
from gen_tso import custom_shiny as cs
from gen_tso.utils import ROOT, collect_spectra, read_spectrum_file
import gen_tso.catalogs.catalog_utils as u


# Catalog of known exoplanets (and candidate planets)
catalog = cat.Catalog()
planets_array = np.array(catalog.planets)
jwst_targets = list(planets_array[catalog.is_jwst])
transit_planets = list(planets_array[catalog.is_transiting])
non_transit_planets = list(planets_array[~catalog.is_transiting])
candidate_planets = list(planets_array[~catalog.is_confirmed])
planets_aka = u.invert_aliases(catalog.planet_aliases)

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

inst_names = [
    'MIRI',
    'NIRCam',
    'NIRISS',
    'NIRSpec',
]

detectors = jwst.generate_all_instruments()
instruments = np.unique([det.instrument for det in detectors])

def get_detector(mode=None, instrument=None):
    if mode is not None:
        for det in detectors:
            if det.mode == mode:
                if instrument is None or det.instrument==instrument:
                    return det
        return None

filter_throughputs = jwst.filter_throughputs()

tso_runs = {}
tso_labels = {}
tso_labels['Current'] = {'current': 'current'}
tso_labels['Transit'] = {}
tso_labels['Eclipse'] = {}

cache_saturation = {}
spectrum_choices = {
    'transit': [],
    'eclipse': [],
    'sed': [],
}
spectra = {}

# Load spectra from user-defined folder and/or from default folder
loading_folders = []
if len(sys.argv) == 2:
    loading_folders.append(os.path.realpath(sys.argv[1]))
loading_folders.append(f'{ROOT}data/models')

for location in loading_folders:
    t_models, e_models, sed_models = collect_spectra(location)
    for label, model in t_models.items():
        spectra[label] = model
        spectrum_choices['transit'].append(label)
    for label, model in e_models.items():
        spectra[label] = model
        spectrum_choices['eclipse'].append(label)
    # for label, model in sed_models.items():
        # 'depth' --> 'flux'
        #spectra[label] = {'wl': wl, 'depth': depth}
        #spectrum_choices['sed'].append(label)


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


app_ui = ui.page_fluid(
    ui.markdown("## **Gen TSO**: A general ETC for time-series observations"),
    ui.include_css(css_file),
    #ui.markdown("""
    #    This app is based on [shiny][0].
    #    [0]: https://shiny.posit.co/py/api/core
    #    """),

    # Instrument and detector modes:
    ui.layout_columns(
        cs.navset_card_tab_jwst(
            inst_names,
            id="select_instrument",
            selected='NIRCam',
            header="Select an instrument and detector",
            footer=ui.input_select(
                "select_mode",
                "",
                choices = {},
                width='425px',
            ),
        ),
        ui.card(
            # current setup and TSO runs
            ui.input_select(
                id="display_tso_run",
                label=ui.tooltip(
                    "Display TSO run:",
                    "TSO runs will show here after a 'Run Pandeia' call",
                    placement='right',
                ),
                choices=tso_labels,
                selected=['current'],
                width='450px',
            ),
            ui.input_action_button(
                id="run_pandeia",
                label="Run Pandeia",
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
                    choices=catalog.planets,
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
                        choices=['Transit', 'Eclipse'],
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
                ui.output_ui('choose_depth'),
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
                ui.output_ui('integration_input'),
                ui.input_switch("integs_switch", "Match obs. duration", False),
                class_="px-2 pt-2 pb-0 m-0",
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
            ui.card(
                ui.card_header(
                    "Results",
                ),
                ui.output_text_verbatim(id="exp_time")
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
    transit_depth = input.depth.get()
    if model_type == 'Flat':
        return f'Flat transit ({transit_depth:.3f}%)'
    elif model_type == 'Blackbody':
        t_planet = input.tplanet.get()
        return f'Blackbody({t_planet:.0f}K, rprs\u00b2={transit_depth:.3f}%)'
    raise ValueError('Invalid model type')


def get_throughput(input):
    inst = req(input.select_instrument).get().lower()
    mode = req(input.select_mode).get()
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
        teff = float(input.teff())
        logg = float(input.logg())
    except ValueError:
        return m_models, None
    idx = jwst.find_closest_sed(m_teff, m_logg, teff, logg)
    chosen_sed = m_models[idx]
    return m_models, chosen_sed


def parse_sed(input):
    """Extract SED parameters"""
    # Safety to prevent hanging when initalizing the app:
    if not input.sed.is_set():
        return None, None, None, None, None

    sed_type = input.sed_type()
    if sed_type in ['phoenix', 'kurucz']:
        sed = input.sed.get()
        if sed not in sed_dict[sed_type]:
            return None, None, None, None, None
        sed_model = sed_dict[sed_type][sed]
        model_label = f'{sed_type}_{sed_model}'
    elif sed_type == 'blackbody':
        sed_model = float(input.teff.get())
        model_label = f'{sed_type}_{sed_model:.0f}K'
    elif sed_type == 'input':
        model_label = sed_model

    norm_band = bands_dict[input.magnitude_band()]
    norm_magnitude = float(input.magnitude())

    if sed_type == 'kurucz':
        sed_type = 'k93models'

    # Make a label
    for name,band in bands_dict.items():
        if band == norm_band:
            band_label = f'{norm_magnitude}_{name.split()[0]}'
    sed_label = f'{model_label}_{band_label}'

    return sed_type, sed_model, norm_band, norm_magnitude, sed_label


def parse_depth_model(input):
    """
    Parse transit/eclipse model name based on current state.
    Calculate or extract model.
    """
    model_type = input.planet_model_type.get()
    depth_label = planet_model_name(input)

    if model_type == 'Input':
        if depth_label is None:
            wl, depth = None, None
        else:
            wl = spectra[depth_label]['wl']
            depth = spectra[depth_label]['depth']
    elif model_type == 'Flat':
        nwave = 1000
        transit_depth = input.depth.get() * 0.01
        wl = np.linspace(0.6, 50.0, nwave)
        depth = np.tile(transit_depth, nwave)
    elif model_type == 'Blackbody':
        transit_depth = input.depth.get() * 0.01
        t_planet = input.tplanet.get()
        # Un-normalized planet and star SEDs
        sed_type, sed_model, norm_band, norm_mag, sed_label = parse_sed(input)
        star_scene = jwst.make_scene(sed_type, sed_model, norm_band='none')
        planet_scene = jwst.make_scene(
            'blackbody', t_planet, norm_band='none',
        )
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


def make_saturation_label(mode, disperser, filter, subarray, sed_label):
    """
    Make a label of unique saturation setups to identify when and
    when not the saturation level can be estimated.
    """
    sat_label = f'{mode}_{filter}'
    if mode == 'bots':
        sat_label = f'{sat_label}_{subarray}'
    elif mode == 'mrs_ts':
        sat_label = f'{sat_label}_{disperser}'
    sat_label = f'{sat_label}_{sed_label}'
    return sat_label


def server(input, output, session):
    sky_view_src = reactive.Value('')
    bookmarked_sed = reactive.Value(False)
    bookmarked_depth = reactive.Value(False)
    saturation_label = reactive.Value(None)
    update_depth_flag = reactive.Value(None)
    uploaded_units = reactive.Value(None)

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Instrument and detector modes
    @reactive.Effect
    @reactive.event(input.select_instrument)
    def _():
        inst_name = input.select_instrument.get()
        print(f"You selected me: {inst_name}")
        spec_modes = {}
        for det in detectors:
            if det.instrument == inst_name and det.obs_type=='spectroscopy':
                spec_modes[det.mode] = det.mode_label
        choices = {}
        choices['Spectroscopy'] = spec_modes

        #choices['Photometry'] = photo_modes  TBD

        acq_modes = {}
        for det in detectors:
            if det.instrument == inst_name and det.obs_type=='acquisition':
                acq_modes[det.mode] = 'Target Acquisition'
        choices['Acquisition'] = acq_modes

        ui.update_select(
            'select_mode',
            choices=choices,
        )


    @reactive.Effect
    @reactive.event(input.select_mode)
    def _():
        instrument = req(input.select_instrument).get()
        mode = input.select_mode.get()
        det = get_detector(mode, instrument)

        ui.update_select(
            'disperser',
            label=det.disperser_label,
            choices=det.dispersers,
            selected=det.default_disperser,
        )
        ui.update_select(
            'filter',
            label=det.filter_label,
            choices=det.filters,
            selected=det.default_filter,
        )

        selected = input.filter_filter.get()
        if det.obs_type == 'acquisition':
            choices = [det.instrument]
        else:
            choices = [det.instrument, 'all']

        if selected != 'all' or det.obs_type=='acquisition':
            selected = None
        ui.update_radio_buttons(
            "filter_filter",
            choices=choices,
            selected=selected,
        )

    @reactive.Effect
    @reactive.event(input.disperser, input.filter)
    def update_subarray():
        instrument = req(input.select_instrument).get()
        mode = input.select_mode.get()
        det = get_detector(mode, instrument)

        if mode == 'bots':
            disperser = input.filter.get().split('/')[0]
        else:
            disperser = input.disperser.get()
        choices = det.get_constrained_val('subarrays', disperser=disperser)

        subarray = input.subarray.get()
        if subarray not in choices:
            subarray = det.default_subarray

        ui.update_select(
            'subarray',
            choices=choices,
            selected=subarray,
        )

    @reactive.Effect
    @reactive.event(input.disperser)
    def update_readout():
        instrument = req(input.select_instrument).get()
        mode = input.select_mode.get()
        det = get_detector(mode, instrument)

        disperser = input.disperser.get()
        choices = det.get_constrained_val('readouts', disperser=disperser)

        readout = input.readout.get()
        if readout not in choices:
            readout = det.default_readout

        ui.update_select(
            'readout',
            choices=choices,
            selected=readout,
        )

    @reactive.Effect
    @reactive.event(input.run_pandeia)
    def _():
        inst_name = input.select_instrument.get()
        inst = inst_name.lower()
        mode = input.select_mode.get()
        disperser = input.disperser.get()
        filter = input.filter.get()
        subarray = input.subarray.get()
        readout = input.readout.get()
        ngroup = int(input.groups.get())
        nint = int(input.integrations.get())
        aperture = None

        detector = get_detector(mode, inst_name)
        inst_label = jwst.instrument_label(detector, disperser, filter)

        # "Exceptions":
        if mode == 'bots':
            disperser, filter = filter.split('/')
        if mode == 'mrs_ts':
            aperture = ['ch1', 'ch2', 'ch3', 'ch4']
        if mode == 'target_acq':
            aperture = input.disperser.get()

        obs_geometry = input.geometry.get()
        transit_dur = float(input.t_dur.get())
        obs_dur = float(input.obs_dur.get())
        exp_time = jwst.exposure_time(
            inst, subarray, readout, ngroup, nint,
        )
        # TBD: if exp_time << obs_dur, raise warning

        depth_label, wl, depth = parse_depth_model(input)
        if depth_label is None:
            ui.notification_show(
                f"No {obs_geometry.lower()} depth model to simulate",
                type="error",
                duration=5,
            )
            return
        if depth_label not in spectra:
            spectrum_choices[obs_geometry.lower()].append(depth_label)
            spectra[depth_label] = {'wl': wl, 'depth': depth}
        depth_model = [wl, depth]
        sed_type, sed_model, norm_band, norm_mag, sed_label = parse_sed(input)

        pando = jwst.PandeiaCalculation(inst, mode)
        pando.set_scene(sed_type, sed_model, norm_band, norm_mag)
        tso = pando.tso_calculation(
            obs_geometry.lower(), transit_dur, exp_time, depth_model,
            ngroup, disperser, filter, subarray, readout, aperture,
        )

        ui.notification_show(
            "TSO model simulated!",
            type="message",
            duration=2,
        )
        detector_label = jwst.detector_label(
            mode, disperser, filter, subarray, readout,
        )
        group_ints = f'({ngroup} G, {nint} I)'
        pretty_label = (
            f'{detector_label} {group_ints} / {sed_label} / {depth_label}'
        )
        tso_label = f'{obs_geometry} {pretty_label}'
        tso_labels[obs_geometry][tso_label] = pretty_label
        ui.update_select('display_tso_run', choices=tso_labels)

        tso_runs[tso_label] = dict(
            # The detector
            inst=inst,
            mode=mode,
            inst_label=inst_label,
            # The SED
            sed_type=sed_type,
            sed_model=sed_model,
            norm_band=norm_band,
            norm_mag=norm_mag,
            # The planet
            obs_type=obs_geometry,
            t_dur=transit_dur,
            obs_dur=obs_dur,
            depth_model_name=depth_label,
            depth_model=depth_model,
            # The instrumental setting
            aperture=aperture,
            disperser=disperser,
            filter=filter,
            subarray=subarray,
            readout=readout,
            ngroup=ngroup,
            # The outputs
            tso=tso,
            #pandeia_results=pandeia_results,
        )

        print(inst, mode, disperser, filter, subarray, readout)
        print(sed_type, sed_model, norm_band, repr(norm_mag))
        print('~~ TSO done! ~~')


    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Target
    @reactive.Effect
    @reactive.event(input.target_filter)
    def _():
        targets = []
        aliases = []
        if 'jwst' in input.target_filter.get():
            targets += jwst_targets
            aliases += catalog.jwst_aliases
        if 'transit' in input.target_filter.get():
            targets += transit_planets
            aliases += catalog.transit_aliases
        if 'non_transit' in input.target_filter.get():
            targets += non_transit_planets
            aliases += catalog.non_transit_aliases
        if 'tess' in input.target_filter.get():
            targets += candidate_planets
            aliases += catalog.candidate_aliases

        # Remove duplicates, sort, and join:
        targets = list(np.unique(targets))
        aliases = list(np.unique(aliases))
        targets += [alias for alias in aliases if alias not in targets]

        # Preserve current target if possible:
        current_target = input.target.get()
        if current_target not in targets:
            current_target = None
        ui.update_selectize('target', choices=targets, selected=current_target)


    @render.ui
    @reactive.event(input.target)
    def target_label():
        target_name = input.target.get()
        if target_name in planets_aka:
            aliases_text = ', '.join(planets_aka[target_name])
            aka_tooltip = ui.tooltip(
                fa.icon_svg("circle-info", fill='cornflowerblue'),
                f"Also known as: {aliases_text}",
                placement='top',
            )
        else:
            aka_tooltip = None

        if target_name in jwst_targets:
            idx = catalog.planets.index(target_name)
            ra, dec = catalog.trexo_coords[idx]
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

        if target_name in candidate_planets:
            candidate_tooltip = ui.tooltip(
                fa.icon_svg("triangle-exclamation", fill='darkorange'),
                ui.markdown("This is a *candidate* planet"),
                placement='top',
            )
        else:
            candidate_tooltip = None

        return ui.span(
            'Known target? ',
            ui.tooltip(
                ui.tags.a(
                    fa.icon_svg("circle-info", fill='black'),
                    href=f'{nasa_url}/{input.target.get()}',
                    target="_blank",
                ),
                'See this target on the NASA Exoplanet Archive',
                placement='top',
            ),
            trexolists_tooltip,
            aka_tooltip,
            candidate_tooltip,
        )

    @reactive.Effect
    @reactive.event(input.target)
    def _():
        """Set known-target properties"""
        target_name = input.target.get()
        if target_name in catalog.planet_aliases:
            ui.update_selectize(
                'target',
                selected=catalog.planet_aliases[target_name],
            )
            return

        if target_name not in catalog.planets:
            return

        index = catalog.planets.index(target_name)
        teff = u.as_str(catalog.teff[index], '.1f', '')
        log_g = u.as_str(catalog.log_g[index], '.2f', '')
        ui.update_text('teff', value=teff)
        ui.update_text('logg', value=log_g)
        ui.update_select('magnitude_band', selected='Ks mag')
        ui.update_text('magnitude', value=f'{catalog.ks_mag[index]:.3f}')
        t_dur = catalog.tr_dur[index]
        tr_duration = '' if t_dur is None else f'{t_dur:.3f}'
        ui.update_text('t_dur', value=tr_duration)

        ra_planet = catalog.ra[index]
        dec_planet = catalog.dec[index]
        sky_view_src.set(
            f'https://sky.esa.int/esasky/?target={ra_planet}%20{dec_planet}'
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
            choices=spectrum_choices['sed']
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
        sed_type, sed_model, norm_band, norm_mag, label = parse_sed(input)
        is_bookmarked = label in spectrum_choices['sed']
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
        sed_type, sed_model, norm_band, norm_mag, sed_file = parse_sed(input)
        if is_bookmarked:
            scene = jwst.make_scene(sed_type, sed_model, norm_band, norm_mag)
            wl, flux = jwst.extract_sed(scene, wl_range=[0.3,30.0])
            spectrum_choices['sed'].append(sed_file)
            spectra[sed_file] = {'wl': wl, 'flux': flux}
        else:
            spectrum_choices['sed'].remove(sed_file)
            spectra.pop(sed_file)


    # This breaks because tplanet is not always defined
    #@reactive.event(
    #    bookmarked_depth, input.geometry,
    #    input.planet_model_type, input.depth, input.tplanet,
    #)
    @render.ui
    def depth_label_text():
        """Set depth model label"""
        obs_geometry = str(input.geometry.get()).lower()
        depth_label = planet_model_name(input)
        is_bookmarked = not bookmarked_depth.get()
        is_bookmarked = depth_label in spectrum_choices[obs_geometry]
        bookmarked_depth.set(is_bookmarked)
        if is_bookmarked:
            depth_icon = fa.icon_svg("earth-americas", style='solid', fill='royalblue')
        else:
            depth_icon = fa.icon_svg("earth-americas", style='solid', fill='gray')
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
            button_ids=['bookmark_depth', 'upload_depth']
        )

    @reactive.Effect
    @reactive.event(input.bookmark_depth)
    def _():
        """Toggle bookmarked depth model"""
        obs_geometry = input.geometry.get().lower()
        depth_label = planet_model_name(input)
        if depth_label is None:
            ui.notification_show(
                f"No {obs_geometry.lower()} depth model to bookmark",
                type="error",
                duration=5,
            )
            return
        is_bookmarked = not bookmarked_depth.get()
        bookmarked_depth.set(is_bookmarked)
        if is_bookmarked:
            depth_label, wl, depth = parse_depth_model(input)
            spectrum_choices[obs_geometry].append(depth_label)
            spectra[depth_label] = {'wl': wl, 'depth': depth}
        else:
            spectrum_choices[obs_geometry].remove(depth_label)
            spectra.pop(depth_label)


    @reactive.effect
    @reactive.event(input.geometry)
    def _():
        obs_geometry = input.geometry.get()
        selected = input.planet_model_type.get()
        if obs_geometry == 'Transit':
            choices = ['Flat', 'Input']
        elif obs_geometry == 'Eclipse':
            choices = ['Blackbody', 'Input']
        if selected not in choices:
            selected = choices[0]

        ui.update_select(
            id="planet_model_type",
            choices=choices,
            selected=selected,
        )

    @render.text
    @reactive.event(input.geometry)
    def transit_dur_label():
        obs_geometry = input.geometry.get()
        return f"{obs_geometry[0]}_dur (h):"

    @render.text
    @reactive.event(input.geometry)
    def transit_depth_label():
        obs_geometry = input.geometry.get()
        return f"{obs_geometry} depth"

    @reactive.Effect
    @reactive.event(
        input.t_dur, input.settling_time, input.baseline_time,
        input.min_baseline_time,
    )
    def _():
        """Set observation time based on transit dur and popover settings"""
        t_dur = req(input.t_dur).get()
        if t_dur == '':
            ui.update_text('obs_dur', value='')
            return
        transit_dur = float(t_dur)
        settling = req(input.settling_time).get()
        baseline = req(input.baseline_time).get()
        min_baseline = req(input.min_baseline_time).get()
        baseline = np.clip(baseline*transit_dur, min_baseline, np.inf)
        # Tdwell = T_start + T_settle + T14 + 2*max(1, T14/2)
        t_total = 1.0 + settling + transit_dur + 2.0*baseline
        ui.update_text('obs_dur', value=f'{t_total:.2f}')


    @render.ui
    @reactive.event(
        input.target, input.geometry, input.planet_model_type,
        update_depth_flag,
    )
    def choose_depth():
        obs_geometry = input.geometry.get()
        model_type = input.planet_model_type.get()

        target_name = input.target.get()
        if target_name not in catalog.planets:
            rprs_square = 1.0
            teq_planet = 1000.0
        else:
            index = catalog.planets.index(target_name)
            teq_planet = catalog.teq[index]
            if catalog.rprs[index] is None:
                rprs_square = 0.0
            else:
                rprs_square = catalog.rprs[index]**2.0
        rprs_square_percent = np.round(100*rprs_square, decimals=4)

        layout_kwargs = dict(
            width=1/2,
            fixed_width=False,
            heights_equal='all',
            gap='7px',
            fill=False,
            fillable=True,
            class_="pb-2 pt-0 m-0",
        )

        if model_type == 'Flat':
            return ui.layout_column_wrap(
                ui.p("Depth (%):"),
                ui.input_numeric(
                    id="depth",
                    label="",
                    value=rprs_square_percent,
                    step=0.1,
                ),
                **layout_kwargs,
            )
        elif model_type == 'Blackbody':
            return ui.layout_column_wrap(
                ui.HTML("<p>(Rp/Rs)<sup>2</sup> (%):</p>"),
                ui.input_numeric(
                    id="depth",
                    label="",
                    value=rprs_square_percent,
                    step=0.1,
                ),
                ui.p("Temp (K):"),
                ui.input_numeric(
                    id="tplanet",
                    label="",
                    value=teq_planet,
                    step=100,
                ),
                **layout_kwargs,
            )

        if model_type == 'Input':
            choices = spectrum_choices[obs_geometry.lower()]
            input_select = ui.input_select(
                id="depth",
                label="",
                choices=choices,
                #selected=selected,
            )
            if len(choices) > 0:
                return input_select

            if obs_geometry == 'Transit':
                tooltip_text = f"a {obs_geometry.lower()}"
            elif obs_geometry == 'Eclipse':
                tooltip_text = f"an {obs_geometry.lower()}"
            return ui.tooltip(
                input_select,
                f'Upload {tooltip_text} depth spectrum',
                placement='right',
            )


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
        obs_geometry = input.geometry.get().lower()
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
            obs_geometry = input.geometry.get().lower()
            spectrum_choices[obs_geometry].append(label)
            # TBD: convert depth units
            spectra[label] = {'wl': wl, 'depth': depth}
            if input.planet_model_type.get() != 'Input':
                return
            # Trigger update choose_depth
            update_depth_flag.set(label)
        elif units in sed_units:
            pass


    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Detector setup
    @render.ui
    @reactive.event(input.select_mode, input.subarray)
    def groups_input():
        mode = input.select_mode.get()
        if mode == 'target_acq':
            instrument = req(input.select_instrument).get()
            det = get_detector(mode, instrument)
            subarray = input.subarray.get()
            choices = det.get_constrained_val('groups', subarray=subarray)

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

    @render.ui
    @reactive.event(input.select_mode)
    def integration_input():
        mode = input.select_mode.get()
        if mode == 'target_acq':
            return ui.input_select(
                id="integrations",
                label="Integrations",
                choices=[1],
            )
        else:
            return ui.input_numeric(
                id="integrations",
                label="Integrations",
                value=1,
                min=1, max=10000,
            )

    @reactive.Effect
    @reactive.event(input.calc_saturation)
    def calculate_saturation_level():
        inst = input.select_instrument.get().lower()
        mode = input.select_mode.get()
        disperser = input.disperser.get()
        filter = input.filter.get()
        subarray = input.subarray.get()
        readout = input.readout.get()
        aperture = None
        sed_type, sed_model, norm_band, norm_mag, sed_label = parse_sed(input)
        sat_label = make_saturation_label(
            mode, disperser, filter, subarray, sed_label,
        )
        #print(
        #    repr(inst), repr(mode), repr(disperser), repr(filter),
        #    repr(subarray), repr(readout), repr(aperture),
        #)

        if mode == 'bots':
            disperser, filter = filter.split('/')
        if mode == 'mrs_ts':
            aperture = ['ch1', 'ch2', 'ch3', 'ch4']
        if mode == 'target_acq':
            ngroup = int(input.groups.get())
            aperture = input.disperser.get()
            disperser = None
        else:
            ngroup = 2

        pando = jwst.PandeiaCalculation(inst, mode)
        pando.set_scene(sed_type, sed_model, norm_band, norm_mag)
        flux_rate, full_well = pando.get_saturation_values(
            disperser, filter, subarray, readout, ngroup,
            aperture,
        )
        if isinstance(flux_rate, Iterable):
            idx = np.argmax(flux_rate*full_well)
            flux_rate = flux_rate[idx]
            full_well = full_well[idx]

        cache_saturation[sat_label] = dict(
            brightest_pixel_rate = flux_rate,
            full_well = full_well,
        )
        # This reactive variable enforces a re-rendering of exp_time
        saturation_label.set(sat_label)


    @reactive.Effect
    @reactive.event(
        input.integs_switch, input.obs_dur, input.select_mode,
        input.select_instrument, input.groups, input.readout, input.subarray,
    )
    def _():
        """Switch to make the integrations match observation duration"""
        if input.select_mode.get() == 'target_acq':
            return
        match_dur = input.integs_switch.get()
        if not match_dur:
            ui.update_numeric('integrations', value=1)
            return

        obs_dur = float(req(input.obs_dur).get())
        inst = input.select_instrument.get().lower()
        nint = 1
        ngroup = input.groups.get()
        readout = input.readout.get()
        subarray = input.subarray.get()
        if ngroup is None:
            return
        single_exp_time = jwst.exposure_time(
            inst, subarray, readout, int(ngroup), int(nint),
        )
        if single_exp_time == 0.0:
            return
        integs = int(np.round(obs_dur*3600.0/single_exp_time))
        ui.update_numeric('integrations', value=integs)


    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Viewer
    @render_plotly
    def plotly_filters():
        show_all = req(input.filter_filter).get() == 'all'

        inst = req(input.select_instrument).get().lower()
        mode = input.select_mode.get()
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
        model_names = spectrum_choices['sed']
        if len(model_names) == 0:
            fig = go.Figure()
            fig.update_layout(title='Bookmark some SEDs to show them here')
            return fig
        sed_models = [spectra[model] for model in model_names]

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
        model_names = spectrum_choices[obs_geometry.lower()]
        nmodels = len(model_names)
        if nmodels == 0:
            return go.Figure()
        throughput = get_throughput(input)

        current_model = planet_model_name(input)
        units = input.plot_depth_units.get()
        wl_scale = input.plot_depth_xscale.get()
        resolution = input.depth_resolution.get()

        depth_models = [spectra[model] for model in model_names]
        fig = tplots.plotly_depth_spectra(
            depth_models, model_names, current_model,
            units=units, wl_scale=wl_scale, resolution=resolution,
            obs_geometry=obs_geometry,
            throughput=throughput,
        )
        return fig

    @render_plotly
    def plotly_tso():
        tso_label = input.display_tso_run.get()
        n_obs = input.n_obs.get()
        resolution = input.tso_resolution.get()
        units = input.plot_tso_units.get()
        wl_scale = input.plot_tso_xscale.get()
        x_range = input.plot_tso_xrange.get()
        if x_range == 'auto':
            wl_range = None
        else:
            wl_range = [0.6, 13.0]

        if tso_label in tso_runs:
            tso_run = tso_runs[tso_label]
            planet = tso_run['depth_model_name']
            fig = tplots.plotly_tso_spectra(
                tso_run['tso'], resolution, n_obs,
                model_label=planet,
                instrument_label=tso_run['inst_label'],
                bin_widths=None,
                units=units, wl_range=wl_range, wl_scale=wl_scale,
                obs_geometry='Transit',
            )
        else:
            fig = go.Figure()
        return fig

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Results
    @render.text
    def exp_time():
        instrument = req(input.select_instrument).get()
        mode = input.select_mode.get()
        detector = get_detector(mode=mode, instrument=instrument)
        inst = instrument.lower()
        disperser = input.disperser.get()
        filter = input.filter.get()
        subarray = input.subarray.get()
        readout = input.readout.get()
        ngroup = input.groups.get()
        nint = input.integrations.get()
        sed_type, sed_model, norm_band, norm_mag, sed_label = parse_sed(input)

        if ngroup is None or detector is None or sed_label is None:
            return ' '

        ngroup = int(ngroup)
        sat_label = make_saturation_label(
            mode, disperser, filter, subarray, sed_label,
        )

        exp_time = jwst.exposure_time(
            inst, subarray, readout, ngroup, int(nint),
        )
        exposure_hours = exp_time / 3600.0
        exp_text = f'Exposure time: {exp_time:.2f} s ({exposure_hours:.2f} h)'

        saturation_label.get()  # enforce calc_saturation renders exp_time
        if sat_label not in cache_saturation:
            return exp_text

        pixel_rate = cache_saturation[sat_label]['brightest_pixel_rate']
        full_well = cache_saturation[sat_label]['full_well']
        sat_time = jwst.saturation_time(detector, ngroup, readout, subarray)
        sat_fraction = pixel_rate * sat_time / full_well
        ngroup_80 = int(0.8*ngroup/sat_fraction)
        ngroup_max = int(ngroup/sat_fraction)
        return (
            f'{exp_text}\n'
            f'Max. fraction of saturation: {100.0*sat_fraction:.1f}%\n'
            f'ngroup below  80% saturation: {ngroup_80:d}\n'
            f'ngroup below 100% saturation: {ngroup_max:d}'
        )

app = App(app_ui, server)

