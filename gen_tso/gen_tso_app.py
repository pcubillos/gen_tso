import matplotlib.pyplot as plt
import numpy as np
from shiny import ui, render, reactive, App

from pandeia.engine.calc_utils import (
    build_default_calc,
    get_instrument_config,
)
from pandeia.engine.instrument_factory import InstrumentFactory

from navset_jwst import navset_card_tab_jwst
import pandeia_interface as jwst
import source_catalog as nea
planets, hosts, teff, log_g, ks_mag, tr_dur = nea.load_nea_table()



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

def get_detector(label):
    for det in detectors:
        if det.label == label:
            return det

instruments = np.unique([det.instrument for det in detectors])


telescope = 'jwst'
instrument = 'nircam'
ins_config = get_instrument_config(telescope, instrument)
mode = spec_modes[instrument]
all_filters = list(ins_config['strategy_config'][mode]['aperture_sizes'])

calculation = build_default_calc(telescope, instrument, mode)
wl = []
qe = []
for filter in all_filters:
    calculation['configuration']['instrument']['filter'] = filter
    inst = InstrumentFactory(
        config=calculation['configuration'], webapp=True,
    )
    wl_range = inst.get_wave_range()
    wl_arr = np.linspace(wl_range['wmin'], wl_range['wmax'], 100)
    wl.append(wl_arr)
    qe.append(inst.get_total_eff(wl_arr))


# Placeholder
options = {
    "inferno": "inferno",
    "viridis": "viridis",
    "copper": "copper",
    "prism": "prism (not recommended)",
}


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
    ui.card(
        ui.layout_columns(
            navset_card_tab_jwst(
                ui.nav_panel('MIRI', 'Select an instrument and detector'),
                ui.nav_panel('NIRCam', 'Select an instrument and detector'),
                ui.nav_panel('NIRISS', 'Select an instrument and detector'),
                ui.nav_panel('NIRSpec', 'Select an instrument and detector'),
                id="inst_tab",
                placement='below',
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
            col_widths=[9, 3],
            #class_="pb-0 mb-0",
        ),
    ),


    ui.layout_columns(
        # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        # The target
        ui.card(
            #ui.p("Select target"),
            #ui.input_text("x", "", placeholder="Enter target name"),
            ui.input_selectize(
                'target',
                'Select a target',
                planets,
                selected='',
                multiple=False,
            ),
            ui.input_radio_buttons(  
                "star_model",  
                "",  
                {"1": "kurucz", "2": "phoenix", "3": "custom"},  
                inline=True,
            ), 
            ui.p("SED custom"),
            # Target props
            ui.layout_column_wrap(
                # Row 1
                ui.p("T_eff (K):"),
                ui.input_text("teff", "", placeholder="Teff"),
                ui.output_text("teff_nea"),
                # Row 2
                ui.p("log(g):"),
                ui.input_text("logg", "", placeholder="log(g)"),
                ui.p("(NEA val)"),
                # Row 3
                ui.p("Ks mag:"),
                ui.input_text("ksmag", "", placeholder="Ks mag"),
                ui.p("(NEA val)"),
                width=1/3,
                fixed_width=False,
                heights_equal='all',
                fill=False,
                fillable=True,
            ),

            ui.input_action_button("button", "Click me"),
            # Breeds
        ),

        # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        # The detector setup
        ui.card(
            # Grism/filter
            ui.panel_well(
                ui.input_select(
                    "cmap",
                    "Grism",
                    options,
                ),
                ui.input_select(
                    "filter",
                    "Filter",
                    options,
                ),
                class_="pb-0 mb-0",
            ),
            # subarray / readout
            ui.panel_well(
                ui.input_select(
                    "subarray",
                    "Subarray",
                    options,
                ),
                ui.input_select(
                    "readout",
                    "Readout pattern",
                    options,
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

        # Results?
        ui.card(
            ui.p("Results?"),
            ui.output_plot("plot_filters"), 
        ),
        col_widths=[3, 3, 6],
    ),

    #ui.layout_sidebar(
    #    ui.panel_sidebar(
    #        ui.input_numeric(
    #            "resolution", label="Resolution", value=250, min=10,
    #            step=50,
    #        ),
    #        ui.panel_well(
    #                tags.strong("Model parameters"),
    #                par_sliders,
    #            class_="pb-1 mb-3",
    #        ),
    #    ),
    #    ui.panel_main(
    #        ui.output_plot("plot")
    #    )
    #)
)

def update_inst_select(input):
    inst_name = input.inst_tab.get()
    print(f"You selected me: {inst_name}")
    modes = {
        det.name: det.label
        for det in detectors
        if det.instrument == inst_name
    }
    obs_types = input.checkbox_group()
    #print(f"You clicked this button! {x}  {inst_name}")
    choices = {}
    if 'spec' in obs_types:
        choices['Spectroscopy'] = modes
    if 'photo' in obs_types:
        choices['Photometry'] = {
            val: val
            for val in 'X Y'.split()
        }

    ui.update_select(
        'select_det',
        #label="Select input label " + str(len(x)),
        choices=choices,
        #selected=x[len(x) - 1] if len(x) > 0 else None,
    )

def server(input, output, session):

    @output
    @render.plot(alt="A histogram")  
    def plot_filters():  
        fig = plt.figure()
        fig.set_size_inches(8,3)
        ax = plt.subplot(111)
        for i,filter in enumerate(all_filters):
            lw = 2.0 if i==0 else 1.25
            ax.plot(wl[i], qe[i], lw=lw, label=filter)
        ax.set_xlabel('wavelength (um)')
        ax.set_ylabel('throughput')
        ax.tick_params(which='both', direction='in')
        ax.set_ylim(bottom=0.0)
        ax.set_title("Palmer Penguin Masses")
        return fig

    @render.text
    @reactive.event(input.target)
    def teff_nea():
        print(input.target())
        if input.target() not in planets:
            return ""
        index = planets.index(input.target())
        print(index)
        return f'{teff[index]:.1f}'

    @reactive.Effect
    @reactive.event(input.target)
    def _():
        if input.target() not in planets:
            return
        index = planets.index(input.target())
        ui.update_text('teff', value=f'{teff[index]:.1f}')
        ui.update_text('logg', value=f'{log_g[index]:.2f}')
        ui.update_text('ksmag', value=f'{ks_mag[index]:.3f}')

    @render.text
    @reactive.event(input.button)
    def text():
        print("You clicked my button!")
        #return f"Last values: {input.selected()}"
    
    @reactive.Effect
    @reactive.event(input.inst_tab)
    def _():
        update_inst_select(input)

    @reactive.Effect
    @reactive.event(input.checkbox_group)
    def _():
        update_inst_select(input)

    @reactive.Effect
    @reactive.event(input.button)
    def _():
        print(f"You clicked the button! {dir(input.selector)}")
        #print(f"You clicked the button! {input.inst_tab.get()}")
        #print(f"You clicked the button! {input.selected()}")



app = App(app_ui, server)
