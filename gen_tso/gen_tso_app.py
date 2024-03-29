# Copyright (c) 2024 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

import pickle

import faicons as fa
from htmltools import HTML
import numpy as np
import plotly.graph_objects as go
from shiny import ui, render, reactive, req, App
from shinywidgets import output_widget, render_plotly

from gen_tso import catalogs as cat
from gen_tso import pandeia as jwst
from gen_tso import plotly as tplots
from gen_tso import shiny as cs
from gen_tso.utils import ROOT



# Confirmed planets
nea_data = cat.load_nea_targets_table()
planets = nea_data[0]
hosts = nea_data[1]
ra = nea_data[2]
dec = nea_data[3]
ks_mag = nea_data[4]
teff = nea_data[5]
log_g = nea_data[6]
tr_dur = nea_data[7]
rprs = nea_data[8]
teq = nea_data[9]

# JWST targets
jwst_hosts, jwst_aliases, missing = cat.load_trexolits_table()

# TESS candidates TBD

aliases = cat.load_aliases()
aka = {}
for alias, name in aliases.items():
    if name not in aka:
        aka[name] = [alias]
    else:
        aka[name] = aka[name] + [alias]

transit_planets = []
non_transiting = []
jwst_targets = []
transit_aliases = []
non_transiting_aliases = []
jwst_aliases = []
for i,target in enumerate(planets):
    if tr_dur[i] is None:
        non_transiting.append(target)
        if target in aka:
            non_transiting_aliases += aka[target]
    else:
        transit_planets.append(target)
        if target in aka:
            transit_aliases += aka[target]
    if hosts[i] in jwst_hosts:
        jwst_targets.append(target)
        if target in aka:
            jwst_aliases += aka[target]
        # TBD: some jwst_hosts are not (yet) confirmed planets


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

spec_modes = {
    'miri': 'lrsslitless',
    'nircam': 'ssgrism',
    'niriss': 'soss',
    'nirspec': 'bots',
}

detectors = jwst.generate_all_instruments()
instruments = np.unique([det.instrument for det in detectors])

def get_detector(mode=None, instrument=None):
    if mode is not None:
        for det in detectors:
            if det.name == mode:
                return det
        return None


def filter_data_frame():
    """
    To be moved to pandeia_interface.py
    """
    filter_throughputs = {}
    for inst_name,mode in spec_modes.items():
        filter_throughputs[inst_name] = {}

        t_file = f'{ROOT}data/throughputs_{inst_name}_{mode}.pickle'
        with open(t_file, 'rb') as handle:
            data = pickle.load(handle)

        subarrays = list(data.keys())
        if mode not in ['bots', 'soss']:
            subarrays = subarrays[0:1]

        for subarray in subarrays:
            filter_throughputs[inst_name][subarray] = data[subarray]
    return filter_throughputs

filter_throughputs = filter_data_frame()

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

nasa_url = 'https://exoplanetarchive.ipac.caltech.edu/overview'
trexolits_url='https://www.stsci.edu/~nnikolov/TrExoLiSTS/JWST/trexolists.html'

css_file = f'{ROOT}data/style.css'


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
                    choices=planets,
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
                    ui.output_text('transit_dur_label'),
                    ui.input_text("t_dur", "", value='2.0'),
                    # Row 2
                    ui.p("Obs_dur (h):"),
                    ui.input_text("obs_dur", "", value='5.0'),
                    # Row 3
                    width=1/2,
                    fixed_width=False,
                    heights_equal='all',
                    gap='7px',
                    fill=False,
                    fillable=True,
                ),
                ui.input_select(
                    id="planet_model",
                    label="Transit depth spectrum",
                    choices=[],
                ),
                class_="px-2 pt-2 pb-0 m-0",
            ),
            ui.panel_well(
                ui.input_action_button('upload_spectrum', 'Upload spectrum'),
                ui.input_radio_buttons(
                    id="upload_type",
                    label='',
                    choices=['Transit', 'Eclipse', 'SED'],
                    inline=True,
                ),
                class_="px-2 pb-0 m-0",
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
                ui.input_numeric(
                    id="groups",
                    label=cs.label_tooltip_button(
                        label='Groups per integration ',
                        tooltip_text='Click icon to estimate saturation level',
                        icon=fa.icon_svg("circle-play", fill='black'),
                        label_id='ngroup_label',
                        button_id='calc_saturation',
                    ),
                    value=2,
                    min=2, max=10000,
                ),
                ui.input_numeric(
                    id="integrations",
                    label="Integrations",
                    value=1,
                    min=1, max=10000,
                ),
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
                            choices = ['none', 'percent', 'ppm'],
                            selected='percent',
                        ),
                        ui.input_select(
                            "plot_depth_xscale",
                            "Wavelength axis:",
                            choices = ['linear', 'log'],
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
                            choices = ['none', 'percent', 'ppm'],
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


def parse_depth_spectrum(file_path):
    spectrum = np.loadtxt(file_path, unpack=True)
    # TBD: check valid format
    wl, depth = spectrum
    return wl, depth


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
        sed_model = sed_dict[sed_type][input.sed()]
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



def server(input, output, session):
    sky_view_src = reactive.Value('')
    bookmarked_sed = reactive.Value(False)
    saturation_label = reactive.Value(None)

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Instrument and detector modes
    @reactive.Effect
    @reactive.event(input.select_instrument)
    def _():
        inst_name = input.select_instrument.get()
        print(f"You selected me: {inst_name}")
        spec_modes = {
            det.name: det.label
            for det in detectors
            if det.instrument == inst_name
        }
        choices = {}
        choices['Spectroscopy'] = spec_modes
        #choices['Photometry'] = photo_modes  TBD
        choices['Target acquisition'] = {
            f'{inst_name}_target_acq': 'Target Acquisition'
        }
        ui.update_select(
            'select_mode',
            choices=choices,
        )


    @reactive.Effect
    @reactive.event(input.select_mode)
    def _():
        detector = get_detector(mode=input.select_mode.get())
        ui.update_select(
            'disperser',
            label=detector.disperser_title,
            choices=detector.dispersers,
            selected=detector.disperser_default,
        )
        ui.update_select(
            'filter',
            label=detector.filter_title,
            choices=detector.filters,
            selected=detector.filter_default,
        )
        ui.update_select(
            'subarray',
            choices=detector.subarrays,
            selected=detector.subarray_default,
        )
        ui.update_select(
            'readout',
            choices=detector.readouts,
            selected=detector.readout_default,
        )
        detector = get_detector(mode=input.select_mode.get())
        selected = input.filter_filter.get()
        if selected != 'all':
            selected = None
        ui.update_radio_buttons(
            "filter_filter",
            choices=[detector.instrument, 'all'],
            selected=selected,
        )

    @reactive.Effect
    @reactive.event(input.run_pandeia)
    def _():
        inst_name = input.select_instrument.get().lower()
        mode = input.select_mode.get()
        subarray = input.subarray.get().lower()
        readout = input.readout.get().lower()
        filter = input.filter.get().lower()
        disperser = input.disperser.get().lower()
        ngroup = input.groups.get()
        nint = input.integrations.get()
        if mode == 'bots':
            disperser, filter = filter.split('/')

        sed_type, sed_model, norm_band, norm_mag, sed_label = parse_sed(input)

        obs_type = input.geometry.get()
        transit_dur = float(input.t_dur.get())
        obs_dur = float(input.obs_dur.get())
        exp_time = jwst.exposure_time(
            inst_name, nint=nint, ngroup=ngroup,
            readout=readout, subarray=subarray,
        )
        # TBD: if exp_time << obs_dur, raise warning

        depth_model_name = input.planet_model.get()
        if depth_model_name is None:
            ui.notification_show(
                f"Missing {obs_type.lower()} depth model!",
                type="error",
                duration=5,
            )
            return

        depth_model = list(spectra[depth_model_name].values())

        pando = jwst.PandeiaCalculation(inst_name, mode)
        pando.set_scene(sed_type, sed_model, norm_band, norm_mag)
        tso = pando.tso_calculation(
            obs_type.lower(), transit_dur, exp_time, depth_model,
            ngroup, filter, readout, subarray, disperser,
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
            f'{detector_label} {group_ints} / {sed_label} / {depth_model_name}'
        )
        tso_label = f'{obs_type} {pretty_label}'
        tso_labels[obs_type][tso_label] = pretty_label
        ui.update_select('display_tso_run', choices=tso_labels)

        tso_runs[tso_label] = dict(
            # The detector
            inst=inst_name,
            mode=mode,
            # The SED
            sed_type=sed_type,
            sed_model=sed_model,
            norm_band=norm_band,
            norm_mag=norm_mag,
            # The planet
            obs_type=obs_type,
            t_dur=transit_dur,
            obs_dur=obs_dur,
            depth_model_name=depth_model_name,
            depth_model=depth_model,
            # The instrumental setting
            disperser=disperser,
            filter=filter,
            subarray=subarray,
            readout=readout,
            ngroup=ngroup,
            # The outputs
            tso=tso,
            #pandeia_results=pandeia_results,
        )

        print(inst_name, mode, disperser, filter, subarray, readout)
        print(sed_type, sed_model, norm_band, repr(norm_mag))
        print('~~ TSO done! ~~')


    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Target
    @reactive.Effect
    @reactive.event(input.target_filter)
    def _():
        targets = []
        if 'jwst' in input.target_filter.get():
            targets += [p for p in jwst_targets if p not in targets]
            targets += [p for p in jwst_aliases if p not in targets]
        if 'transit' in input.target_filter.get():
            targets += [p for p in transit_planets if p not in targets]
            targets += [p for p in transit_aliases if p not in targets]
        if 'non_transit' in input.target_filter.get():
            targets += [p for p in non_transiting if p not in targets]
            targets += [p for p in non_transiting_aliases if p not in targets]

        # Preserve current target if possible:
        current_target = input.target.get()
        if current_target not in targets:
            current_target = None
        ui.update_selectize('target', choices=targets, selected=current_target)


    @render.ui
    @reactive.event(input.target)
    def target_label():
        target_name = input.target.get()
        if target_name not in aka:
            aka_tooltip = None
        else:
            aliases_text = ', '.join(aka[target_name])
            aka_tooltip = ui.tooltip(
                fa.icon_svg("circle-info", fill='cornflowerblue'),
                f"Also known as: {aliases_text}",
                placement='top',
            )

        if target_name in jwst_targets:
            trexolists_tooltip = ui.tooltip(
                ui.tags.a(
                    fa.icon_svg("circle-info", fill='goldenrod'),
                    href=f'{trexolits_url}',
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
        )

    @reactive.Effect
    @reactive.event(input.target)
    def _():
        """Set known-target properties"""
        target_name = input.target.get()
        if target_name in aliases:
            ui.update_selectize('target', selected=aliases[target_name])
            return

        if target_name not in planets:
            return

        index = planets.index(target_name)
        ui.update_text('teff', value=f'{teff[index]:.1f}')
        ui.update_text('logg', value=f'{log_g[index]:.2f}')
        ui.update_select('magnitude_band', selected='Ks mag')
        ui.update_text('magnitude', value=f'{ks_mag[index]:.3f}')
        t_dur = tr_dur[index]
        tr_duration = '' if t_dur is None else f'{t_dur:.3f}'
        ui.update_text('t_dur', value=tr_duration)

        index = planets.index(target_name)
        ra_planet = ra[index]
        dec_planet = dec[index]
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

        return cs.label_tooltip_button(
            label='Stellar SED model: ',
            tooltip_text='Click star to bookmark SED',
            icon=sed_icon,
            label_id='sed_label',
            button_id='sed_bookmark',
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
            wl, flux = jwst.extract_sed(scene, wl_range=[0.3,21.0])
            spectrum_choices['sed'].append(sed_file)
            spectra[sed_file] = {'wl': wl, 'flux': flux}
        else:
            spectrum_choices['sed'].remove(sed_file)
            spectra.pop(sed_file)

    @reactive.effect
    @reactive.event(input.geometry)
    def _():
        obs_geometry = input.geometry.get()
        ui.update_select(
            id="planet_model",
            label=f"{obs_geometry} depth spectrum:",
            choices=spectrum_choices[obs_geometry.lower()],
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
        transit_dur = float(req(input.t_dur).get())
        settling = req(input.settling_time).get()
        baseline = req(input.baseline_time).get()
        min_baseline = req(input.min_baseline_time).get()
        baseline = np.clip(baseline*transit_dur, min_baseline, np.inf)
        # Tdwell = T_start + T_settle + T14 + 2*max(1, T14/2)
        t_total = 1.0 + settling + transit_dur + 2.0*baseline
        ui.update_text('obs_dur', value=f'{t_total:.2f}')


    @reactive.effect
    @reactive.event(input.upload_file)
    def _():
        new_model = input.upload_file()
        if not new_model:
            print('No new model!')
            return

        current_model = input.planet_model.get()
        upload_type = input.upload_type.get().lower()
        depth_file = new_model[0]['name']
        # TBD: remove file extension?
        spectrum_choices[upload_type].append(depth_file)
        wl, depth = parse_depth_spectrum(new_model[0]['datapath'])
        spectra[depth_file] = {'wl': wl, 'depth': depth}

        ui.update_select(
            id="planet_model",
            #label=f"{obs_geometry} depth spectrum:",
            choices=spectrum_choices[upload_type],
            selected=current_model,
        )

    @reactive.effect
    @reactive.event(input.upload_spectrum)
    def _():
        sed_choices = [
            # TBD: can I get super-scripts?
            "erg s-1 cm-2 Hz-1 (frequency space)",
            "erg s-1 cm-2 cm (wavenumber space)",
            "erg s-1 cm-2 cm-1 (wavelength space)",
            "mJy",
        ]
        depth_choices = [
            "none",
            "percent",
            "ppm",
        ]
        if input.upload_type.get() == 'Transit':
            choices = depth_choices
            label1 = 'transit depth'
            label2 = 'Depth units:'
        elif input.upload_type.get() == 'Eclipse':
            choices = depth_choices
            label1 = 'eclipse depth'
            label2 = 'Depth units:'
        elif input.upload_type.get() == 'SED':
            choices = sed_choices
            label1 = 'stellar SED'
            label2 = 'Flux units:'

        m = ui.modal(
            ui.input_file(
                id="upload_file",
                label=ui.markdown(
                    "Input files must be plan-text files with two columns, "
                    "the first one being the wavelength (microns) and "
                    f"the second one the {label1}. "
                    "**Make sure the input units are correct!**"
                ),
                button_label="Browse",
                multiple=True,
                width='100%',
            ),
            ui.input_radio_buttons(
                id="flux_units",
                label=label2,
                choices=choices,
                width='100%',
            ),
            title="Upload Spectrum",
            easy_close=True,
        )
        ui.modal_show(m)

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Detector setup
    @reactive.Effect
    @reactive.event(input.calc_saturation)
    def _():
        inst_name = input.select_instrument.get().lower()
        mode = input.select_mode.get()
        filter = input.filter.get().lower()
        subarray = input.subarray.get().lower()
        readout = input.readout.get().lower()
        disperser = input.disperser.get().lower()
        if mode == 'bots':
            disperser, filter = filter.split('/')

        pando = jwst.PandeiaCalculation(inst_name, mode)

        sed_type, sed_model, norm_band, norm_mag, sed_label = parse_sed(input)
        pando.set_scene(sed_type, sed_model, norm_band, norm_mag)

        flux_rate, full_well = pando.get_saturation_values(
            filter, readout, subarray, disperser,
        )

        if mode == 'bots':
            filter = f'{disperser}/{filter}'
            sat_label = f'{filter}_{subarray}_{sed_label}'
        else:
            sat_label = f'{filter}_{sed_label}'
        cache_saturation[sat_label] = dict(
            brightest_pixel_rate = flux_rate,
            full_well = full_well,
        )
        # This reactive variable enforces a re-rendering of exp_time
        saturation_label.set(sat_label)

    @reactive.Effect
    @reactive.event(
        input.integs_switch, input.obs_dur,
        input.select_instrument, input.groups, input.readout, input.subarray,
    )
    def _():
        """Switch to make the number of integrations match observation duration"""
        match_dur = input.integs_switch.get()
        if not match_dur:
            ui.update_numeric('integrations', value=1)
            return

        obs_dur = float(req(input.obs_dur).get())
        inst_name = input.select_instrument.get().lower()
        nint = 1
        ngroup = input.groups.get()
        readout = input.readout.get().lower()
        subarray = input.subarray.get().lower()
        if ngroup is None:
            return
        single_exp_time = jwst.exposure_time(
            inst_name, nint=nint, ngroup=ngroup,
            readout=readout, subarray=subarray,
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
        inst_name = req(input.select_instrument).get().lower()
        subarray = req(input.subarray.get()).lower()
        filter_name = input.filter.get().lower()
        if inst_name == 'miri':
            filter_name = 'None'

        fig = tplots.plotly_filters(
            filter_throughputs, inst_name, subarray, filter_name, show_all,
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

        units = input.plot_sed_units.get()
        wl_scale = input.plot_sed_xscale.get()
        resolution = input.plot_sed_resolution.get()
        fig = tplots.plotly_sed_spectra(
            sed_models, model_names, #current_model,
            units=units, wl_scale=wl_scale, resolution=resolution,
        )
        return fig


    @render_plotly
    def plotly_depth():
        obs_geometry = input.geometry.get()
        model_names = spectrum_choices[obs_geometry.lower()]
        nmodels = len(model_names)
        if nmodels == 0:
            return go.Figure()

        current_model = input.planet_model.get()
        units = input.plot_depth_units.get()
        wl_scale = input.plot_depth_xscale.get()
        resolution = input.depth_resolution.get()

        depth_models = [spectra[model] for model in model_names]
        fig = tplots.plotly_depth_spectra(
            depth_models, model_names, current_model,
            units=units, wl_scale=wl_scale, resolution=resolution,
            obs_geometry=obs_geometry,
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
            tso = tso_run['tso']
            bin_wl, bin_spec, bin_err, bin_widths = jwst.simulate_tso(
               tso, n_obs=n_obs, resolution=resolution, noiseless=False,
            )
            fig = tplots.plotly_tso_spectra(
                tso['wl'], tso['depth_spectrum'],
                bin_wl, bin_spec, bin_err,
                label=planet, bin_widths=None,
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
        inst_name = input.select_instrument.get().lower()
        mode = input.select_mode.get()
        detector = get_detector(mode=mode)
        subarray = input.subarray.get().lower()
        readout = input.readout.get().lower()
        ngroup = input.groups.get()
        nint = input.integrations.get()
        if ngroup is None:
            return ''
        exp_time = jwst.exposure_time(
            inst_name, nint=nint, ngroup=ngroup,
            readout=readout, subarray=subarray,
        )
        exposure_hours = exp_time / 3600.0
        exp_text = f'Exposure time: {exp_time:.2f} s ({exposure_hours:.2f} h)'

        filter = str(input.filter.get()).lower()
        sed_type, sed_model, norm_band, norm_mag, sed_label = parse_sed(input)
        saturation_label.get()  # enforce calc_saturation renders exp_time
        if mode == 'bots':
            sat_label = f'{filter}_{subarray}_{sed_label}'
        else:
            sat_label = f'{filter}_{sed_label}'
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
            f'ngroup below 80% and 100% saturation: {ngroup_80:d} / {ngroup_max:d}'
        )

app = App(app_ui, server)

