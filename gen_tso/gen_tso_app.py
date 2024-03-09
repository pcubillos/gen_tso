# Copyright (c) 2024 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

import pickle

import numpy as np

from shiny import ui, render, reactive, App
from shiny.experimental.ui import card_body
from shinywidgets import output_widget, render_plotly
from htmltools import HTML

import plotly.express as px
import plotly.graph_objects as go
import faicons as fa

from pandeia.engine.calc_utils import (
    get_instrument_config,
)

from navset_jwst import navset_card_tab_jwst
import pandeia_interface as jwst
import source_catalog as nea


planets, hosts, teff, log_g, ks_mag, tr_dur = nea.load_nea_table()
p_models, p_teff, p_logg = jwst.load_sed_list('phoenix')
k_models, k_teff, k_logg = jwst.load_sed_list('k93models')


# GEN TSO preamble
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
    To be moved into pandeia_interface.py
    """
    filter_throughputs = {}
    for inst_name,mode in spec_modes.items():
        filter_throughputs[inst_name] = {}
        if inst_name == 'niriss':
            filter_throughputs[inst_name]['none'] = {}
            continue

        t_file = f'../data/throughputs_{inst_name}_{mode}.pickle'
        with open(t_file, 'rb') as handle:
            data = pickle.load(handle)

        subarrays = list(data.keys())
        if mode != 'bots':
            subarrays = subarrays[0:1]

        for subarray in subarrays:
            filter_throughputs[inst_name][subarray] = {}
            filters = list(data[subarray].keys())
            for filter in filters:
                wl = data[subarray][filter]['wl']
                response = data[subarray][filter]['response']
                df = dict(
                    wl=wl,
                    response=response,
                )
                filter_throughputs[inst_name][subarray][filter] = df
    return filter_throughputs

filter_throughputs = filter_data_frame()


# Placeholder
telescope = 'jwst'
instrument = 'nircam'
ins_config = get_instrument_config(telescope, instrument)
mode = spec_modes[instrument]
all_filters = list(ins_config['strategy_config'][mode]['aperture_sizes'])


# Placeholder
ra = 315.02582008947
dec = -5.09445415116
#ra = 10.684
#dec = 41.268
src = f'https://sky.esa.int/esasky/?target={ra}%20{dec}'

# Add main content
gear_icon = fa.icon_svg("gear")


app_ui = ui.page_fluid(
    ui.h2("Gen TSO: General JWST ETC for exoplanet time-series observations"),
    #ui.markdown(
    #    """
    #    This app is based on a [Matplotlib example][0] that displays 2D data
    #    with a user-adjustable colormap. We use a range slider to set the data
    #    range that is covered by the colormap.

    #    [0]: https://matplotlib.org/3.5.3/gallery/userdemo/colormap_interactive_adjustment.html
    #    """
    #),

    # Instrument / detector:
    ui.layout_columns(
        navset_card_tab_jwst(
            ui.nav_panel('MIRI', ''),
            ui.nav_panel('NIRCam', ''),
            ui.nav_panel('NIRISS', ''),
            ui.nav_panel('NIRSpec', ''),
            id="inst_tab",
            header="Select an instrument and detector",
            footer=ui.input_select(
                "select_det",
                "",
                choices = {},
                width='400px'
            ),
        ),
        ui.card(
            ui.input_checkbox_group(
                id="checkbox_group",
                label="Observation type:",
                choices={
                    "spec": "spectroscopy",
                    "photo": "photometry",
                },
                selected=['spec'],
            ),
            ui.input_action_button("run_pandeia", "Run Pandeia"),
        ),
        col_widths=[6,6],
    ),

    ui.layout_columns(
        # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        # The target
        ui.card(
            ui.card_header("Target", class_="bg-primary"),
            ui.input_selectize(
                'target',
                'Known target?',
                planets,
                selected='WASP-80 b',
                multiple=False,
            ),
            ui.p("System properties:"),
            # Target props
            ui.layout_column_wrap(
                # Row 1
                ui.p("T_eff (K):"),
                ui.input_text("teff", "", value='1400.0', placeholder="Teff"),
                # Row 2
                ui.p("log(g):"),
                ui.input_text("logg", "", value='4.5', placeholder="log(g)"),
                # Row 3
                ui.p("Ks mag:"),
                ui.input_text("ksmag", "", value='10.0', placeholder="Ks mag"),
                width=1/2,
                fixed_width=False,
                heights_equal='all',
                fill=False,
                fillable=True,
            ),
            ui.input_select(
                id="star_model",
                label="Stellar SED model",
                choices=[
                    "phoenix (auto)",
                    "kurucz (auto)",
                    "phoenix (select)",
                    "kurucz (select)",
                    "blackbody",
                    "custom",
                ],
            ),
            ui.output_ui('choose_sed'),
            ui.input_action_button("button", "Click me"),
        ),

        # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        # The detector setup
        ui.card(
            ui.card_header("Detector setup", class_="bg-primary"),
            # Grism/filter
            ui.panel_well(
                ui.input_select(
                    id="disperser",
                    label="Disperser",
                    choices={},
                ),
                ui.input_select(
                    id="filter",
                    label="Filter",
                    choices={},
                ),
                class_="pb-0 mb-0",
            ),
            # subarray / readout
            ui.panel_well(
                ui.input_select(
                    id="subarray",
                    label="Subarray",
                    choices=[''],
                ),
                ui.input_select(
                    id="readout",
                    label="Readout pattern",
                    choices=[''],
                ),
                class_="pb-0 mb-0",
            ),
            # groups / integs
            ui.panel_well(
                ui.input_numeric(
                    "groups",
                    "Groups per integration",
                    2,
                    min=2, max=100,
                ),
                ui.input_numeric(
                    "integrations",
                    "Integrations",
                    1,
                    min=1, max=100,
                ),
                ui.input_switch("switch", "Match obs. time", False),
                class_="pb-0 mb-0",
            ),
        ),

        # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        # Results
        ui.layout_columns(
            ui.navset_card_tab(
                ui.nav_panel(
                    "Filters",
                    ui.popover(
                        ui.span(
                            gear_icon,
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
                    ui.card(
                        card_body(
                            output_widget("plotly_filters", fillable=True),
                            padding='1px',
                        ),
                        full_screen=True,
                        height='10px',
                        class_="bg-primary",
                    ),
                ),
                ui.nav_panel(
                    "Sky view",
                    ui.card(
                        card_body(
                            HTML(
                                '<iframe '
                                'height="100%" '
                                'width="100%" '
                                'style="overflow" '
                                f'src="{src}" '
                                'frameborder="0" allowfullscreen></iframe>',
                                #id=resolve_id(id),
                            ),
                            padding='1px',
                            id='esasky_card',
                        ),
                        full_screen=True,
                        height='250px',
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
        ),
        col_widths=[3, 3, 6],
    ),
)


def update_inst_select(input):
    inst_name = input.inst_tab.get()
    print(f"You selected me: {inst_name}")
    modes = {
        det.name: det.label
        for det in detectors
        if det.instrument == inst_name
    }
    obs_types = [
        'spec',
        #'photo',
        'acquisition',
    ]
    #print(f"You clicked this button! {x}  {inst_name}")
    choices = {}
    if 'spec' in obs_types:
        choices['Spectroscopy'] = modes
    if 'photo' in obs_types:
        choices['Photometry'] = {
            val: val
            for val in 'X Y'.split()
        }
    if 'acquisition' in obs_types or True:
        choices['Target acquisition'] = {
            'nircam_target_acq': 'Target Acquisition'
        }

    ui.update_select(
        'select_det',
        choices=choices,
        #selected=x[len(x) - 1] if len(x) > 0 else None,
    )

def update_detector(input):
    detector = get_detector(mode=input.select_det.get())
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


def get_auto_sed(input):
    if 'kurucz' in input.star_model():
        m_models, m_teff, m_logg = k_models, k_teff, k_logg
    elif 'phoenix' in input.star_model():
        m_models, m_teff, m_logg = p_models, p_teff, p_logg
    try:
        teff = float(input.teff())
        logg = float(input.logg())
    except ValueError:
        return m_models, None
    idx = jwst.find_closest_sed(m_teff, m_logg, teff, logg)
    chosen_sed = m_models[idx]
    return m_models, chosen_sed


def server(input, output, session):
    my_sed = reactive.Value(None)

    @render_plotly
    def plotly_filters():
        inst_name = input.inst_tab.get().lower()
        if inst_name == 'miri':
            filter_name = 'None'
        else:
            filter_name = input.filter.get().lower()
        # Eventually, I want to plot-code this by detector
        passbands = filter_throughputs[inst_name]
        if inst_name == 'nirspec':
            subarray = input.subarray.get().lower()
        else:
            subarray = list(filter_throughputs[inst_name].keys())[0]
        #print(f'\n{inst_name}  {subarray}  {filter_name}\n')
        if subarray not in filter_throughputs[inst_name]:
            return go.Figure()
        passbands = filter_throughputs[inst_name][subarray]

        visible = [None for _ in passbands.keys()]
        if inst_name == 'nirspec':
            for i,filter in enumerate(passbands.keys()):
                hide = ('h' in filter_name) is not ('h' in filter)
                if hide and 'prism' not in filter:
                    visible[i] = 'legendonly'
        fig = go.Figure()
        colors = px.colors.sequential.Viridis
        j = 0
        for filter, throughput in passbands.items():
            if filter == filter_name:
                linedict = dict(color='Gold', width=3.0)
            else:
                linedict = dict(color=colors[j])
            fig.add_trace(go.Scatter(
                x=throughput['wl'],
                y=throughput['response'],
                mode='lines',
                name=filter.upper(),
                legendgrouptitle_text=inst_name,
                line=linedict,
                legendrank=j,
                visible=visible[j],
            ))
            j += 1

        fig.update_traces(
            hovertemplate=
                'wl = %{x:.2f}<br>'+
                'throughput = %{y:.3f}'
        )
        fig.update_yaxes(
            title_text=None,
            autorangeoptions=dict(minallowed=0),
        )
        wl_range = [0.5, 13.5] if inst_name=='miri' else [0.5, 6.0]
        fig.update_xaxes(
            title_text='wavelength (um)',
            title_standoff=0,
            range=wl_range,
        )
        fig.update_layout(showlegend=True)
        # Show current filter on top:
        filters = list(passbands.keys())
        if filter_name not in filters:
            return fig
        itop = filters.index(filter_name)
        fig_idx = np.arange(len(fig.data))
        fig_idx[-1] = itop
        fig_idx[itop] = len(fig.data) - 1
        fig.data = tuple(np.array(fig.data)[fig_idx])
        return fig


    @reactive.Effect
    @reactive.event(input.target)
    def _():
        if input.target() not in planets:
            return
        index = planets.index(input.target())
        ui.update_text('teff', value=f'{teff[index]:.1f}')
        ui.update_text('logg', value=f'{log_g[index]:.2f}')
        ui.update_text('ksmag', value=f'{ks_mag[index]:.3f}')

    @render.ui
    @reactive.event(input.star_model, input.teff, input.logg)
    def choose_sed():
        #print(input.star_model())
        if 'auto' in input.star_model():
            # find model
            try:
                teff = float(input.teff())
            except ValueError:
                raise ValueError("Can't select an SED, I need a valid Teff")
            try:
                logg = float(input.logg())
            except ValueError:
                raise ValueError("Can't select an SED, I need a valid log(g)")
            m_models, chosen_sed = get_auto_sed(input)
            my_sed.set(chosen_sed)
            return ui.p(
                chosen_sed,
                style='background:#EBEBEB',
            )
        elif 'select' in input.star_model():
            m_models, chosen_sed = get_auto_sed(input)
            selected = chosen_sed
            return ui.input_select(
                id="sed",
                label="",
                choices=list(m_models),
                selected=selected,
            )
        elif input.star_model() == 'blackbody':
            if input.teff.get() != '':
                teff = float(input.teff.get())
                return ui.p(
                    f'Blackbody(Teff={teff:.0f}K)',
                    style='background:#EBEBEB',
                )


    @reactive.Effect
    @reactive.event(input.sed)
    def _():
        my_sed.set(input.sed())
        print(f'Choose an SED! ({input.sed()})')


    @reactive.Effect
    @reactive.event(input.inst_tab)
    def _():
        update_inst_select(input)


    @reactive.Effect
    @reactive.event(input.select_det)
    def _():
        update_detector(input)

    @reactive.effect
    @reactive.event(input.select_det)
    def _():
        detector = get_detector(mode=input.select_det.get())
        ui.update_radio_buttons(
            "filter_filter",
            choices=[detector.instrument, 'all'],
        )


    @render.text
    def exp_time():
        det_name = input.select_det.get()
        for detector in detectors:
            if detector.name == det_name:
                break
        else:
            return
        subarray = input.subarray.get().lower()
        readout = input.readout.get().lower()
        ngroup = input.groups.get()
        nint = input.integrations.get()
        nexp = 1
        #print(f'\n{subarray}  {readout}  {ngroup}  {nint}\n')
        exp_time = jwst.exposure_time(
            detector, nexp=nexp, nint=nint, ngroup=ngroup,
            readout=readout, subarray=subarray,
        )
        return f'Exposure time: {exp_time:.2f} s'

    #@render.text
    #@reactive.event(input.button)
    #def text():
    #    return f"Last values: {input.selected()}"

    @reactive.Effect
    @reactive.event(input.button)
    def _():
        #print(f"You clicked the button! {dir(input.selector)}")
        #print(f"You clicked the button! {input.inst_tab.get()}")
        #print(f"You clicked the button! {input.selected()}")
        print(input.select_det.get())
        print(f'My favorite SED is: {my_sed.get()}')
        print("You clicked my button!")
        det_name = input.select_det.get()
        for detector in detectors:
            if detector.name == det_name:
                break
        else:
            return
        subarray = input.subarray.get().lower()
        readout = input.readout.get().lower()
        ngroup = input.groups.get()
        nint = input.integrations.get()
        nexp = 1
        print(f'{subarray}  {readout}  {ngroup}  {nint}')
        exp_time = jwst.exposure_time(
            detector, nexp=nexp, nint=nint, ngroup=ngroup,
            readout=readout, subarray=subarray,
        )
        print(f'Exposure time: {exp_time:.2f} s')
        #print(dir(choose_sed))

app = App(app_ui, server)
