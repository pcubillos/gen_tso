# Copyright (c) 2024 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

import matplotlib.pyplot as plt
import numpy as np

from shiny import ui, render, reactive, App
from shiny.experimental.ui import card_body
import plotly.express as px
from shinywidgets import output_widget, render_plotly
from htmltools import HTML

import pandas as pd

from pandeia.engine.calc_utils import (
    build_default_calc,
    get_instrument_config,
)
from pandeia.engine.instrument_factory import InstrumentFactory

from navset_jwst import navset_card_tab_jwst
import pandeia_interface as jwst
import source_catalog as nea


planets, hosts, teff, log_g, ks_mag, tr_dur = nea.load_nea_table()
p_models, p_teff, p_logg = jwst.load_sed_list('phoenix')
k_models, k_teff, k_logg = jwst.load_sed_list('k93models')


# GEN TSO preamble
mode_labels = {
    'lrsslitless': 'Low Resolution Spectroscopy (LRS)',
    'mrs_ts': 'MRS Time Series',
    'ssgrism': 'LW Grism Time Series',
    'soss': 'Single Object Slitless Spectroscopy (SOSS)',
    'bots': 'Bright Object Time Series (BOTS)',
    # Imaging
    'imaging_ts': 'Imaging Time Series',
    'sw_ts': 'SW Time Series',
    'lw_ts': 'LW Time Series',
}

spec_modes = {
    'miri': 'lrsslitless',
    'nircam': 'ssgrism',
    'niriss': 'soss',
    'nirspec': 'bots',
}


detectors = jwst.generate_all_instruments()
instruments = np.unique([det.instrument for det in detectors])

def get_detector(name):
    for det in detectors:
        if det.name == name:
            return det



def filter_data_frame():
    """
    To be moved into pandeia_interface.py
    """
    filters_df = pd.DataFrame()

    telescope = 'jwst'
    for inst_name in instruments:
        print(f'\nThis is {inst_name}')
        instrument = inst_name.lower()
        ins_config = get_instrument_config(telescope, instrument)
        mode = spec_modes[instrument]
        calculation = build_default_calc(telescope, instrument, mode)
        configs = []
        labels = []
        if inst_name == 'NIRISS':
            continue
        if inst_name == 'NIRSpec':
            gratings = ins_config['config_constraints']['dispersers']
            for grating, filters in gratings.items():
                for filter in filters['filters']:
                    configs.append({
                        'disperser': grating,
                        'filter': filter,
                    })
                    labels.append(f'{grating.upper()}/{filter.upper()}')
        if inst_name == 'NIRCam' or inst_name == 'MIRI':
            filters = ins_config['strategy_config'][mode]['aperture_sizes']
            for filter in filters.keys():
                configs.append({'filter': filter})
                labels.append(filter.upper())

        #print(calculation['configuration']['instrument'])
        for config,label in zip(configs,labels):
            calculation['configuration']['instrument'].update(config)
            inst = InstrumentFactory(
                config=calculation['configuration'], webapp=True,
            )
            print(inst_name, config['filter'].upper())
            wl_range = inst.get_wave_range()
            if inst_name != 'MIRI':
                wl_filter = inst.get_wave_filter()
                wl_range['wmin'] = np.amin(wl_filter)
                wl_range['wmax'] = np.amax(wl_filter)
            wl_arr = np.linspace(wl_range['wmin'], wl_range['wmax'], 100)
            qe = inst.get_total_eff(wl_arr)
            if mode == 'bots':
                det = get_detector('bots')
                wl_min, wl_max = det.wl_ranges[label]
                qe[(wl_arr<wl_min) | (wl_arr>wl_max)] = 0.0
            df = pd.DataFrame(dict(
                wl=wl_arr,
                qe=qe,
                filter=label,
                instrument=inst_name,
            ))
            filters_df = pd.concat((filters_df, df))
    return filters_df

filters_df = filter_data_frame()


# Placeholder
telescope = 'jwst'
instrument = 'nircam'
ins_config = get_instrument_config(telescope, instrument)
mode = spec_modes[instrument]
all_filters = list(ins_config['strategy_config'][mode]['aperture_sizes'])


# Placeholder
options = {
    "inferno": "inferno",
    "viridis": "viridis",
    "copper": "copper",
    "prism": "prism (not recommended)",
}

ra = 315.02582008947
dec = -5.09445415116
ra = 10.684
dec = 41.268
src = f'https://sky.esa.int/esasky/?target={ra}%20{dec}'

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
                ui.input_text("teff", "", placeholder="Teff"),
                # Row 2
                ui.p("log(g):"),
                ui.input_text("logg", "", placeholder="log(g)"),
                # Row 3
                ui.p("Ks mag:"),
                ui.input_text("ksmag", "", placeholder="Ks mag"),
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
                    id="grism",
                    label="Grism",
                    choices=options,
                ),
                ui.input_select(
                    id="filter",
                    label="Filter",
                    choices=options,
                ),
                class_="pb-0 mb-0",
            ),
            # subarray / readout
            ui.panel_well(
                ui.input_select(
                    id="subarray",
                    label="Subarray",
                    choices=options,
                ),
                ui.input_select(
                    id="readout",
                    label="Readout pattern",
                    choices=options,
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
            ui.card(
                ui.card_header("Filter throughput"),
                card_body(
                    output_widget("plotly_filters", fillable=True),
                    padding='1px',
                ),
                full_screen=True,
                height='275px',
                class_="bg-primary",
                #class_="bg-primary px-n3 mx-n3 gap-0",
                #class_="bg-primary lead",
            ),
            ui.card(
                HTML(
                    '<iframe '
                    'height="100%" '
                    'width="100%" '
                    'style="overflow" '
                    f'src="{src}" '
                    'frameborder="0" allowfullscreen></iframe>',
                    #id=resolve_id(id),
                ),
                #ui.output_plot("plot_filters"),
                full_screen=True,
                height='350px',
            ),
            col_widths=[12,12],
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
    det_name = input.select_det.get()
    #print(input.select_det.get())
    for detector in detectors:
        if detector.name == det_name:
            break
    else:
        return
    #print(f'I want this one: {detector}')
    ui.update_select(
        'grism',
        label=detector.grism_title,
        choices=detector.grisms,
    )
    ui.update_select(
        'filter',
        label=detector.filter_title,
        choices=detector.filters,
    )
    ui.update_select(
        'subarray',
        choices=detector.subarrays,
    )
    ui.update_select(
        'readout',
        choices=detector.readout,
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
        inst_name = input.inst_tab.get()
        # Eventually, I want to plot-code this by detector
        #det_name = input.select_det.get()
        fig = px.line(
            filters_df.query(f"instrument=='{inst_name}'"),
            x="wl", y="qe", color='filter',
            labels = {'qe': 'QE'},
            #hover_name="filter",
            #hover_data={
            #    'filter': False, # remove species from hover data
            #    'wl':':.2f', # customize hover for column of y attribute
            #    'qe':':.3f', # add other column, customized formatting
            #},
        )
        fig.update_traces(
            hovertemplate=
                'wl = %{x:.2f}<br>'+
                'QE = %{y:.3f}'
        )
        fig.update_yaxes(
            title_text=None,
            autorangeoptions=dict(minallowed=0),
        )
        wl_range = [0.5, 13.5] if inst_name=='MIRI' else [0.5, 6.0]
        fig.update_xaxes(
            title_text='wavelength (um)',
            title_standoff=0,
            range=wl_range,
        )
        return fig


    @output
    @render.plot(alt="A histogram")
    def plot_filters():
        fig = plt.figure()
        fig.set_size_inches(8,3)
        ax = plt.subplot(111)
        #for i,filter in enumerate(all_filters):
        #    lw = 2.0 if i==0 else 1.25
        #    ax.plot(wl[i], qe[i], lw=lw, label=filter)
        ax.set_xlabel('wavelength (um)')
        ax.set_ylabel('throughput')
        ax.tick_params(which='both', direction='in')
        ax.set_ylim(bottom=0.0)
        ax.set_title("Palmer Penguin Masses")
        return fig

    #@render.text
    #@reactive.event(input.target)
    #def teff_nea():
    #    print(input.target())
    #    if input.target() not in planets:
    #        return ""
    #    index = planets.index(input.target())
    #    print(index)
    #    return f'{teff[index]:.1f}'

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

app = App(app_ui, server)
