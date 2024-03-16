# Copyright (c) 2024 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

import pickle

import numpy as np

from shiny import ui, render, reactive, App
from shinywidgets import output_widget, render_plotly
from htmltools import HTML

import plotly.express as px
import plotly.graph_objects as go
import faicons as fa

import pandeia_interface as jwst
import source_catalog as nea
import custom_shiny as cs


# Preamble
planets, hosts, teff, log_g, ks_mag, tr_dur = nea.load_nea_table()
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
    To be moved into pandeia_interface.py
    """
    filter_throughputs = {}
    for inst_name,mode in spec_modes.items():
        filter_throughputs[inst_name] = {}

        t_file = f'../data/throughputs_{inst_name}_{mode}.pickle'
        with open(t_file, 'rb') as handle:
            data = pickle.load(handle)

        subarrays = list(data.keys())
        if mode not in ['bots', 'soss']:
            subarrays = subarrays[0:1]

        for subarray in subarrays:
            filter_throughputs[inst_name][subarray] = data[subarray]
    return filter_throughputs

filter_throughputs = filter_data_frame()


# Placeholder
ra = 315.02582008947
dec = -5.09445415116
src = f'https://sky.esa.int/esasky/?target={ra}%20{dec}'


app_ui = ui.page_fluid(
    ui.markdown("## **Gen TSO**: A general ETC for time-series observations"),
    #ui.markdown("""
    #    This app is based on [shiny][0].
    #    [0]: https://shiny.posit.co/py/api/core
    #    """),

    # Instrument / detector:
    ui.layout_columns(
        cs.navset_card_tab_jwst(
            inst_names,
            id="inst_tab",
            selected='NIRCam',
            header="Select an instrument and detector",
            footer=ui.input_select(
                "select_det",
                "",
                choices = {},
                width='400px'
            ),
        ),
        ui.card(
            # Placeholder / maybe store bookmarked runs in tabs here?
            ui.input_checkbox_group(
                id="checkbox_group",
                label="Observation type:",
                choices={
                    "spec": "spectroscopy",
                    "photo": "photometry",
                },
                selected=['spec'],
            ),
            ui.input_action_button(id="run_pandeia", label="Run Pandeia"),
        ),
        col_widths=[6,6],
    ),

    ui.layout_columns(
        # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        # The target
        cs.custom_card(
            ui.card_header("Target", class_="bg-primary"),
            ui.panel_well(
                ui.input_selectize(
                    'target',
                    'Known target?',
                    planets,
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
                    fill=False,
                    fillable=True,
                ),
                ui.input_select(
                    id="star_model",
                    label=ui.output_ui('stellar_sed_label'),
                    choices=[
                        "phoenix",
                        "kurucz",
                        "blackbody",
                        "custom",
                    ],
                ),
                ui.output_ui('choose_sed'),
                class_="px-2 pt-2 pb-0 m-0",
            ),
            # The planet
            ui.panel_well(
                ui.layout_column_wrap(
                    # Row 1
                    ui.p("Observation:"),
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
                ui.input_radio_buttons(
                    id="upload_type",
                    label='Upload spectrum',
                    choices=["SED", "Transit"],
                    inline=True,
                ),
                ui.input_file(
                    id="upload_depth",
                    label="",
                    button_label="Browse",
                    multiple=True,
                ),
                class_="px-2 py-0 m-0",
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
                ),
                ui.input_select(
                    id="filter",
                    label="Filter",
                    choices={},
                ),
                class_="px-2 pt-2 pb-0 m-0",
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
                class_="px-2 pt-2 pb-0 m-0",
            ),
            # groups / integs
            ui.panel_well(
                ui.input_numeric(
                    "groups",
                    label=cs.label_tooltip_button(
                        label='Groups per integration ',
                        tooltip_text='Click icon to estimate saturation level',
                        icon=fa.icon_svg("circle-play", fill='black'),
                        label_id='ngroup_label',
                        button_id='calc_saturation',
                    ),
                    value=2,
                    min=2, max=100,
                ),
                ui.input_numeric(
                    "integrations",
                    "Integrations",
                    1,
                    min=1, max=100,
                ),
                ui.input_switch("switch", "Match obs. time", False),
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
                    cs.custom_card(
                        HTML(
                            '<iframe '
                            'height="100%" '
                            'width="100%" '
                            'style="overflow" '
                            f'src="{src}" '
                            'frameborder="0" allowfullscreen></iframe>',
                            #id=resolve_id(id),
                        ),
                        body_args=dict(class_='m-0 p-0', id='esasky_card'),
                        full_screen=True,
                        height='350px',
                    ),
                ),
                ui.nav_panel(
                    "Stellar SED",
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
                            id='resolution',
                            label='Resolution:',
                            value=250.0,
                            min=10.0, max=3000.0, step=25.0,
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
    #root = '/Users/pato/Dropbox/IWF/proposals/2023_jwst_light_triad/simulations/'
    #file = 'WASP80b_transmission_spectrum_v2.dat'
    #file_path = root + file
    spectrum = np.loadtxt(file_path, unpack=True)
    # TBD: check valid format
    wl, depth = spectrum
    return wl, depth


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


def parse_sed(input):
    sed_type = input.star_model()
    if sed_type in ['phoenix', 'kurucz']:
        sed_model = sed_dict[sed_type][input.sed()]
    elif sed_type == 'blackbody':
        sed_model = input.teff.get()
    elif sed_type == 'custom':
        pass

    norm_band = bands_dict[input.magnitude_band()]
    norm_magnitude = float(input.magnitude())

    if sed_type == 'kurucz':
        sed_type = 'k93models'
    return sed_type, sed_model, norm_band, norm_magnitude



def server(input, output, session):
    my_sed = reactive.Value(None)
    bookmarked_sed = reactive.Value(False)
    brightest_pix_rate = reactive.Value(None)
    full_well = reactive.Value(None)

    spectrum_choices = {
        'transit': [],
        'eclipse': [],
        'star_sed': [],
    }
    spectra = {}

    @reactive.Effect
    @reactive.event(input.inst_tab)
    def _():
        update_inst_select(input)


    @render.ui
    def stellar_sed_label():
        if bookmarked_sed.get():
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
        print('You did click the star!')
        #print(input.sed_bookmark.get(), input.sed_bookmark.is_set())
        if input.sed_bookmark.get():
            # toggle value
            new_val = not bookmarked_sed.get()
            bookmarked_sed.set(new_val)
            print(f'This is bookmkarked: {bookmarked_sed.get()}')
            #ui.notification_show("Message!", duration=2)


    @reactive.Effect
    @reactive.event(input.calc_saturation)
    def _():
        inst_name = input.inst_tab.get().lower()
        mode = input.select_det.get()
        filter = input.filter.get().lower()
        subarray = input.subarray.get().lower()
        readout = input.readout.get().lower()
        disperser = input.disperser.get().lower()
        if mode == 'bots':
            disperser, filter = filter.split('/')

        #print(inst_name, mode, disperser, filter, subarray, readout)
        pando = jwst.Calculation(inst_name, mode)

        sed_type, sed_model, norm_band, norm_magnitude = parse_sed(input)
        #print(sed_type, sed_model, norm_band, repr(norm_magnitude))
        pando.set_scene(sed_type, sed_model, norm_band, norm_magnitude)

        flux_rate, fullwell = pando.get_saturation_values(
            filter, readout, subarray, disperser,
        )
        brightest_pix_rate.set(flux_rate)
        full_well.set(fullwell)


    @reactive.Effect
    @reactive.event(input.select_det)
    def _():
        update_detector(input)
        detector = get_detector(mode=input.select_det.get())
        ui.update_radio_buttons(
            "filter_filter",
            choices=[detector.instrument, 'all'],
        )

    @render_plotly
    def plotly_filters():
        inst_name = input.inst_tab.get().lower()
        if inst_name == 'miri':
            filter_name = 'None'
        else:
            filter_name = input.filter.get().lower()
        # TBD: Eventually, I want to plot-code this by mode instead of inst
        passbands = filter_throughputs[inst_name]
        if inst_name in ['nirspec', 'niriss']:
            subarray = input.subarray.get().lower()
        else:
            subarray = list(filter_throughputs[inst_name].keys())[0]
        #print(f'\n{inst_name}  {subarray}  {filter_name}\n')
        if subarray not in filter_throughputs[inst_name]:
            print(f'\nGetting out: {subarray} / {filter_throughputs[inst_name].keys()}')
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
            if 'order2' in throughput:
                linedict = dict(color='Orange', width=3.0)
                fig.add_trace(go.Scatter(
                    x=throughput['order2']['wl'],
                    y=throughput['order2']['response'],
                    mode='lines',
                    name='CLEAR Or.2',
                    legendgrouptitle_text=inst_name,
                    line=linedict,
                    legendrank=j,
                ))

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
        # Show current filter on top (TBD: there must be a better way)
        filters = list(passbands.keys())
        if filter_name not in filters:
            return fig
        itop = filters.index(filter_name)
        fig_idx = np.arange(len(fig.data))
        fig_idx[-1] = itop
        fig_idx[itop] = len(fig.data) - 1
        fig.data = tuple(np.array(fig.data)[fig_idx])
        return fig


    @render_plotly
    def plotly_sed():
        # TBD: Same as plotly_depth but with the stellar SEDs
        fig = go.Figure()
        return fig


    @render_plotly
    def plotly_depth():
        obs_geometry = input.geometry.get()
        models = spectrum_choices[obs_geometry.lower()]
        nmodels = len(models)
        current_model = input.planet_model.get()
        units = '%'

        fig = go.Figure()
        if nmodels == 0:
            return fig
        for j,model in enumerate(models):
            if model == current_model:
                linedict = dict(color='Gold', width=3.0)
                rank = j + nmodels
                visible = None
            else:
                linedict = {}
                rank = j
                visible = 'legendonly'
            fig.add_trace(go.Scatter(
                x=spectra[model]['wl'],
                y=spectra[model]['depth']*100.0,
                mode='lines',
                name=model,
                #legendgrouptitle_text=inst_name,
                line=linedict,
                legendrank=rank,
                visible=visible,
            ))

        fig.update_traces(
            hovertemplate=
                'wl = %{x:.2f}<br>'+
                'depth = %{y:.3f}'
        )
        fig.update_yaxes(
            title_text=f'{obs_geometry} depth ({units})',
            title_standoff=0,
        )
        #wl_range = [0.5, 13.5] if inst_name=='miri' else [0.5, 6.0]
        wl_range = [0.5, 6.0]
        fig.update_xaxes(
            title_text='wavelength (um)',
            title_standoff=0,
            range=wl_range,
        )

        fig.update_layout(legend=dict(
            orientation="h",
            entrywidth=0.5,
            entrywidthmode='fraction',
            yanchor="bottom",
            xanchor="right",
            y=1.02,
            x=1
        ))
        fig.update_layout(showlegend=True)
        return fig


    @reactive.Effect
    @reactive.event(input.target)
    def _():
        if input.target() not in planets:
            return
        index = planets.index(input.target())
        ui.update_text('teff', value=f'{teff[index]:.1f}')
        ui.update_text('logg', value=f'{log_g[index]:.2f}')
        ui.update_text('magnitude', value=f'{ks_mag[index]:.3f}')


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
        elif input.star_model() in ['phoenix', 'kurucz']:
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
                    f' Blackbody (Teff={teff:.0f} K)',
                    style='background:#DDDDDD',
                    class_='mb-2 ps-2',
                )


    @reactive.Effect
    @reactive.event(input.sed)
    def _():
        my_sed.set(input.sed())
        print(f'Choose an SED! ({input.sed()})')


    @reactive.effect
    @reactive.event(input.geometry)
    def _():
        obs_geometry = input.geometry.get()
        ui.update_select(
            id="planet_model",
            label=f"{obs_geometry} depth spectrum:",
            choices=spectrum_choices[obs_geometry.lower()],
        )


    @reactive.effect
    @reactive.event(input.upload_depth)
    def _():
        new_model = input.upload_depth()
        if not new_model:
            print('No new model!')
            return

        current_model = input.planet_model.get()
        obs_geometry = input.geometry.get()
        depth_file = new_model[0]['name']
        # TBD: remove file extension?
        spectrum_choices[obs_geometry.lower()].append(depth_file)
        #print(repr(current_model))
        #print(depth_file)
        #print(new_model)
        wl, depth = parse_depth_spectrum(new_model[0]['datapath'])
        spectra[depth_file] = {'wl': wl, 'depth': depth}

        ui.update_select(
            id="planet_model",
            label=f"{obs_geometry} depth spectrum:",
            choices=spectrum_choices[obs_geometry.lower()],
            selected=current_model,
        )


    @render.text
    def transit_dur_label():
        obs_geometry = input.geometry.get()
        return f"{obs_geometry[0]}_dur (h):"

    @render.text
    def transit_depth_label():
        obs_geometry = input.geometry.get()
        return f"{obs_geometry} depth"

    @render.text
    def exp_time():
        mode = input.select_det.get()
        detector = get_detector(mode=mode)
        subarray = input.subarray.get().lower()
        readout = input.readout.get().lower()
        ngroup = input.groups.get()
        nint = input.integrations.get()
        #print(f'\n{subarray}  {readout}  {ngroup}  {nint}\n')
        exp_time = jwst.exposure_time(
            detector, nint=nint, ngroup=ngroup,
            readout=readout, subarray=subarray,
        )
        pixel_rate = brightest_pix_rate.get()
        if pixel_rate is None:
            return f'Exposure time: {exp_time:.2f} s'

        sat_time = jwst.saturation_time(detector, ngroup, readout, subarray)
        sat_fraction = pixel_rate * sat_time / full_well.get()
        ngroup_80 = int(0.8*ngroup/sat_fraction)
        ngroup_max = int(ngroup/sat_fraction)
        return (
            f'Exposure time: {exp_time:.2f} s\n'
            f'Max. fraction of saturation: {100.0*sat_fraction:.1f}%\n'
            f'ngroup below 80% and 100% saturation: {ngroup_80:d} / {ngroup_max:d}'
        )

    @reactive.Effect
    @reactive.event(input.run_pandeia)
    def _():
        print("You clicked my button!")
        #mode = input.select_det.get()
        #detector = get_detector(mode=mode)
        print(f'My favorite SED is: {my_sed.get()}')
        #subarray = input.subarray.get().lower()
        #readout = input.readout.get().lower()
        #ngroup = input.groups.get()
        #nint = input.integrations.get()
        #print(dir(choose_sed))

app = App(app_ui, server)
