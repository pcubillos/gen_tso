# Copyright (c) 2025-2026 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

import json
import os
from pathlib import Path
import textwrap
from datetime import timedelta, datetime

import faicons as fa
import numpy as np
import pandas as pd
import pandeia.engine
import plotly.graph_objects as go
from shiny import ui, render, reactive, req, App
from shinywidgets import output_widget, render_plotly

import gen_tso
from gen_tso import catalogs as cat
from gen_tso import pandeia_io as jwst
from gen_tso import plotly_io as plots
from gen_tso import custom_shiny as cs
from gen_tso.utils import (
    ROOT,
    parser,
    get_latest_pandeia_release,
    get_version_advice,
    get_pandeia_advice,
    collect_spectra,
    read_spectrum_file,
    pretty_print_target,
)
import gen_tso.catalogs.utils as u
from gen_tso.pandeia_io.pandeia_defaults import (
    _load_flux_rate_splines,
    get_detector,
    get_sed_types,
    make_obs_label,
    make_saturation_label,
    make_save_label,
)
from gen_tso.pandeia_io.pandeia_setup import (
    check_pandeia_ref_data,
    check_pysynphot,
    update_synphot_files,
)

from gen_tso.app_utils import (
    detectors,
    get_throughput,
    get_auto_sed,
    get_saturation_values,
    planet_model_name,
    draw,
    parse_instrument,
    parse_depth_model,
    parse_obs,
    parse_sed,
    _safe_num,
)
import gen_tso.viewer_popovers as pops
from gen_tso.export_script import (
    export_script_fixed_values,
    export_script_calculated_values,
)


cli_args = parser()

# Custom targets
default_custom_targets = os.path.join(ROOT, 'data', 'custom_targets.txt')
if cli_args.targets is not None:
    if not os.path.exists(cli_args.targets):
        print(f"\n~ custom targets file not found: {repr(cli_args.targets)} ~\n")
        custom_targets = None
    elif cli_args.targets.lower().endswith('.csv'):
        custom_targets = os.path.join(ROOT, 'data', 'tmp_custom_targets.txt')
        cat.load_csv_targets(cli_args.targets, custom_targets)
    else:
        custom_targets = cli_args.targets
elif os.path.exists(default_custom_targets):
    custom_targets = default_custom_targets
else:
    custom_targets = None


def load_catalog():
    catalog = cat.Catalog(custom_targets)
    is_jwst = np.array([target.is_jwst_planet for target in catalog.targets])
    is_transit = np.array([target.is_transiting for target in catalog.targets])
    is_confirmed = np.array([target.is_confirmed for target in catalog.targets])
    return catalog, is_jwst, is_transit, is_confirmed


# Catalog of known exoplanets (and candidate planets)
catalog, is_jwst, is_transit, is_confirmed = load_catalog()
nplanets = len(catalog.targets)

# Catalog of stellar SEDs:
sed_dict = {}
for sed_type in get_sed_types():
    sed_keys, sed_models, _, _ = jwst.get_sed_list(sed_type)
    sed_dict[sed_type] = {
        key: model
        for key,model in zip(sed_keys, sed_models)
    }

bands_dict = {
    '2mass,j': 'J mag',
    '2mass,h': 'H mag',
    '2mass,ks': 'Ks mag',
    'gaia,g': 'Gaia mag',
    'johnson,v': 'V mag',
}
instruments = np.unique([det.instrument for det in detectors])
pairings = get_detector('nircam', 'sw_ts', detectors).pairings

modes = {}
for inst in instruments:
    choices = {}
    spec_modes = {}
    for det in detectors:
        if det.instrument == inst and det.obs_type=='spectroscopy':
            spec_modes[det.mode] = det.mode_label
    choices['Spectroscopy'] = spec_modes

    photo_modes = {}
    for det in detectors:
        if det.instrument == inst and det.obs_type=='photometry':
            photo_modes[det.mode] = det.mode_label
    if len(photo_modes) > 0:
        choices['Photometry'] = photo_modes

    acq_modes = {}
    for det in detectors:
        if det.instrument == inst and det.obs_type=='acquisition':
            acq_modes[det.mode] = 'Target Acquisition'
    choices['Acquisition'] = acq_modes
    modes[inst] = choices

# Pre-computed flux rates
flux_rate_splines, full_wells = _load_flux_rate_splines()


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

# Load spectra from user-defined folder and/or from default folder
loading_folders = []
if cli_args.models is not None:
    models_path = os.path.realpath(cli_args.models)
    loading_folders.append(models_path)
loading_folders.append(f'{ROOT}data/models')
current_dir = os.path.realpath(os.getcwd())

for location in loading_folders:
    t_models, e_models, sed_models = collect_spectra(location)
    for label, model in t_models.items():
        spectra['transit'][label] = model
    for label, model in e_models.items():
        spectra['eclipse'][label] = model
    for label, model in sed_models.items():
        spectra['sed'][label] = model


nasa_url = 'https://exoplanetarchive.ipac.caltech.edu/overview'
stsci_url = 'https://www.stsci.edu/jwst/science-execution/program-information?id=PID'
cdnjs = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.11.1/'

# Depth and SED units, ensure they are consistent with u.read_spectrum_file()
depth_units = [
    "none",
    "percent",
    "ppm",
]
ergs_s_cm2 = "erg s\u207b\u00b9 cm\u207b\u00b2"
sed_units = {
    "f_freq": f"{ergs_s_cm2} Hz\u207b\u00b9 (frequency space)",
    "f_nu": f"{ergs_s_cm2} cm (wavenumber space)",
    "f_lambda": f"{ergs_s_cm2} cm\u207b\u00b9 (wavelength space)",
    "mJy": "mJy",
}

# 2D heatmap plots:
heatmaps = {
    '2d_flux': 'detector',
    '2d_snr': 'snr',
    '2d_saturation': 'saturation',
    '2d_groups': 'ngroups_map',
}

# Instrument warnings
stripes = [
     'sub17stripe_soss',
     'sub60stripe_soss',
     'sub204stripe_soss',
     'sub680stripe_soss',
]
nircam_warning = (
    'WARNING: The SW Grism Time Series mode is still being calibrated; '
    'the SNR and saturation estimates provided by the ETC '
    'may therefore be outside the expected 10% accuracy level'
)
soss_warning = 'WARNING: The SOSS multi-stripe subarray is still being calibrated'


layout_kwargs = dict(
    width=1/2,
    fixed_width=False,
    heights_equal='all',
    gap='7px',
    fill=False,
    fillable=True,
    class_="pb-2 pt-0 m-0",
)

card_style = "background:#F5F5F5; !important;"

app_ui = ui.page_fluid(
    # ESA Sky
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
    # NASA link
    ui.tags.script("""
        Shiny.addCustomMessageHandler("set_nasa_href", function(msg) {
            const link = document.getElementById("nasa_link");
            if (link) link.href = msg.url;
        });
    """),
    # Copy to clipboard
    ui.HTML("""
        <script>
        Shiny.addCustomMessageHandler('copy_to_clipboard', function(message) {
            navigator.clipboard.writeText(message)
                .then(function() {
                    Shiny.setInputValue("copy_status", "success", {priority: "event"});
                })
                .catch(function(err) {
                    Shiny.setInputValue("copy_status", "failure", {priority: "event"});
                });
        });
        </script>
    """),
    # Syntax highlighting (python)
    ui.HTML(
        f'<link rel="stylesheet" href="{cdnjs}styles/base16/one-light.min.css">'
        f'<script src="{cdnjs}highlight.min.js"></script>'
    ),
    ui.tags.style(
        """
        #run_pandeia{
            font-weight: bold;
            background-color: #EEBA0B;
            color: #FFFFFF;
            border-color: #CC9C00;
        }
        #run_pandeia:hover{
            color:#000000;
            background-color: #ffc60a;
            border-color: #EEBA0B;
        }
        .warning {
            color: #ffa500;
            font-weight: bold;
        }
        .danger {
            color: red;
            font-weight: bold;
        }
        .popover {
            --bs-popover-max-width: 600px;
        }
        .action-link .action-label:empty {
            margin-left: 0.0em !important;
        }
        """
    ),
    ui.layout_columns(
        ui.span(
            ui.HTML(
                "<b>Gen TSO</b>: A general JWST simulator "
                "for exoplanet time series ("
            ),
            ui.tooltip(
                ui.input_action_link(
                    id='main_status',
                    label='',
                    icon=fa.icon_svg("gear", fill='black'),
                ),
                "status",
                placement='bottom',
            ),
            ui.HTML(', '),
            ui.tooltip(
                ui.input_action_link(
                    id='bibtex',
                    label='',
                    icon=fa.icon_svg("book-open-reader", fill='black',
                    ),
                ),
                "citation",
                placement='bottom',
            ),
            ui.HTML(', '),
            ui.tooltip(
                ui.tags.a(
                    fa.icon_svg("book", fill='black'),
                    href='https://pcubillos.github.io/gen_tso',
                    target="_blank",
                ),
                "documentation",
                placement='bottom',
            ),
            ui.HTML(')'),
            style="font-size: 26px;",
        ),
        ui.output_image("tso_logo", inline=True),
        col_widths=(11, 1),
        fixed_width=False,
        fill=False,
        fillable=True,
        class_="p-0 m-0",
    ),
    # Instrument and detector modes:
    ui.layout_columns(
        cs.custom_card(
            ui.card_header("Select instrument and mode"),
            ui.layout_columns(
                ui.layout_columns(
                    cs.navset_card_tab_jwst(
                        instruments,
                        id='instrument',
                        selected='NIRCam',
                    ),
                    ui.input_select(
                        id="mode",
                        label="",
                        choices=modes['NIRCam'],
                        width='100%',
                    ),
                    col_widths=(12,12),
                    fill=True,
                    fillable=True,
                    gap='1px',
                    class_="p-0 m-0",
                ),
                ui.input_action_button(
                    id="export_button",
                    label="Export to notebook",
                    class_="btn btn-outline-success btn-sm",
                ),
                col_widths=(9,3),
                fill=True,
                fillable=True,
                gap='10px',
                class_="p-0 m-0",
            ),
            body_args=dict(class_="p-2 m-0"),
        ),
        cs.custom_card(
            # current setup and TSO runs
            ui.layout_columns(
                ui.layout_column_wrap(
                    ui.card_header("TSO runs"),
                    ui.tooltip(
                        ui.input_select(
                            id="display_tso_run",
                            label='',
                            choices=make_tso_labels(tso_runs),
                            selected=[''],
                            width='100%',
                        ),
                        "TSO runs will show here after a 'Run Pandeia' call",
                        placement='top',
                    ),
                    ui.input_task_button(
                        id="run_pandeia",
                        label="Run Pandeia",
                        label_busy="processing...",
                        width='100%',
                    ),
                    heights_equal=False,
                    width=1,
                    gap='3px',
                    class_="p-0 pb-2 m-0",
                ),
                ui.layout_column_wrap(
                    # TBD: Set disabled based on existing TSOs
                    ui.input_action_button(
                        id="save_button",
                        label="Save TSO",
                        class_="btn btn-outline-success btn-sm",
                        disabled=False,
                        width='100%',
                    ),
                    ui.input_action_button(
                        id="delete_button",
                        label="Delete TSO",
                        class_='btn btn-outline-danger btn-sm',
                        disabled=False,
                        width='100%',
                    ),
                    ui.panel_conditional(
                        "input.mode == 'target_acq'",
                        ui.input_radio_buttons(
                            id='target_focus',
                            label='',
                            choices={
                                'science': 'science target',
                                'acquisition': 'acquisition',
                            },
                            selected='science',
                        ),
                    ),
                    gap='1px',
                    width=1,
                    class_="p-0 pt-2 m-0",
                ),
                col_widths=(9,3),
                heights_equal=False,
                gap='8px',
                class_="p-0 m-0",
            ),
            body_args=dict(class_="py-0 px-2 m-0"),
        ),
        col_widths=[6,6],
    ),

    ui.layout_columns(
        # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        # The target
        cs.custom_card(
            ui.card_header("Target", class_="bg-primary"),
            ui.card(
                ui.card_body(
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
                                "custom": "custom targets",
                                "tess": "TESS candidates",
                                "non_transit": "non-transiting",
                            },
                            selected=['jwst', 'transit'],
                        ),
                        title='Filter targets',
                        placement="right",
                        id="targets_popover",
                    ),
                    # Hidden section to hold switches for conditionals
                    ui.panel_conditional(
                        'false',
                        ui.input_action_button(
                            id="konami_sequence_trigger",
                            label="",
                        ),
                        ui.input_switch(
                            id="is_candidate",
                            label="candidate",
                            value=False,
                        ),
                        ui.input_switch(
                            id="is_custom",
                            label="custom",
                            value=False,
                        ),
                        ui.input_switch(
                            id="has_sed_bookmarks",
                            label="has SED",
                            value=False,
                        ),
                        ui.input_switch(
                            id="has_depth_bookmarks",
                            label="has transits",
                            value=False,
                        ),
                        ui.input_switch(
                            id="raise_detector_warning",
                            label="detector warning",
                            value=False,
                        ),
                    ),
                    # Customizing buttons
                    ui.panel_conditional(
                        # Keep hidden while we fine-tune the customs details
                        #"input.is_custom",
                        "false",
                        ui.layout_column_wrap(
                            ui.input_action_button(
                                id='save_custom_target',
                                label='Save changes',
                                class_='btn btn-outline-success btn-sm',
                            ),
                            #ui.input_action_button(
                            #    id='clear_custom',
                            #    label='Clear changes',
                            #    class_='btn btn-outline-success btn-sm',
                            #),
                            width=1/2,
                            fixed_width=False,
                            heights_equal='all',
                            gap='7px',
                            fill=False,
                            fillable=True,
                        ),
                    ),
                    # The target
                    ui.span(
                        ui.HTML('<b>Science target</b> '),
                        ui.tooltip(
                            ui.input_action_link(
                                id='show_info',
                                label='',
                                icon=fa.icon_svg("circle-info", fill='cornflowerblue'),
                            ),
                            'System info',
                            id='target_info_tooltip',
                            placement='top',
                        ),
                        #url has to be set with javascript, output_ui does not render nicely, ui.input_action_link() does not open in server side.
                        ui.tooltip(
                            ui.tags.a(
                                fa.icon_svg("circle-info", fill='black'),
                                id='nasa_link',
                                href=f'{nasa_url}',
                                target="_blank",
                            ),
                            "Open target's NASA Exoplanet Archive",
                            id='nasa_tooltip',
                            placement='top',
                        ),
                        ui.tooltip(
                            ui.input_action_link(
                                id='show_observations',
                                label='',
                                icon=fa.icon_svg("circle-info", fill='gray'),
                            ),
                            'not a JWST target (yet)',
                            id='jwst_tooltip',
                            placement='top',
                        ),
                        ui.panel_conditional(
                            "input.is_custom",
                            ui.tooltip(
                                fa.icon_svg("circle-info", fill='#15B01A', margin_left='-0.3em'),
                                ui.markdown("This is a custom target"),
                                placement='top',
                            ),
                        ),
                        ui.panel_conditional(
                            "input.is_candidate",
                            ui.tooltip(
                                fa.icon_svg("triangle-exclamation", fill='darkorange', margin_left='-0.3em'),
                                ui.markdown("This is a *candidate* planet"),
                                placement='top',
                            ),
                        ),
                    ),
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
                        ui.input_numeric("t_eff", "", value=1400, min=100, step=100),
                        # Row 2
                        ui.p("log(g):"),
                        ui.input_numeric("log_g", "", value=4.5, min=0, step=0.1),
                        # Row 3
                        ui.input_select(
                            id='magnitude_band',
                            label='',
                            choices=bands_dict,
                            selected='2mass,ks',
                        ),
                        ui.input_numeric(
                            id="magnitude",
                            label="",
                            value=10.0,
                            step=0.1,
                        ),
                        width=1/2,
                        fixed_width=False,
                        heights_equal='all',
                        gap='7px',
                        fill=False,
                        fillable=True,
                    ),
                    ui.span(
                        ui.HTML('<b>Stellar SED</b> '),
                        # bookmarks, upload, and clear
                        ui.tooltip(
                            ui.input_action_link(
                                id='bookmark_sed',
                                label='',
                                icon=fa.icon_svg("star", fill='black'),
                            ),
                            'Bookmark SED',
                            id='sed_book_tooltip',
                            placement='top',
                        ),
                        ui.tooltip(
                            ui.input_action_link(
                                id='upload_sed',
                                label='',
                                icon=fa.icon_svg("file-arrow-up", fill='black'),
                            ),
                            'Upload SED',
                            id='sed_up_tooltip',
                            placement='top',
                        ),
                        ui.panel_conditional(
                            "input.has_sed_bookmarks",
                            ui.tooltip(
                                ui.input_action_link(
                                    id='clear_sed_bookmarks',
                                    label='',
                                    icon=fa.icon_svg("circle-xmark", style='regular', fill='black'),
                                ),
                                'Clear all SED bookmarks',
                                id='sed_clear_tooltip',
                                placement='top',
                            ),
                        ),
                    ),

                    ui.input_select(
                        id="sed_type",
                        label='',
                        choices={
                            "phoenix": "phoenix",
                            "k93models": "kurucz (k93models)",
                            "bt_settl": "BT-Settl MLT (bt_settl)",
                            "blackbody": "blackbody",
                            "input": "input",
                        },
                        selected='phoenix',
                    ),
                    ui.input_select(
                        id="sed",
                        label="",
                        choices=sed_dict['phoenix'],
                        selected='g0v',
                    ),
                    class_="px-2 py-1 pb-2 m-0 gap-2",
                    style=card_style,
                ),
                fill=False,
            ),

            # The planet
            ui.card(
                ui.card_body(
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
                    ui.markdown("**Observation**"),
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
                        ui.input_numeric("t_dur", "", value=2.0, min=0, step=0.1),
                        # Row 3
                        ui.p("Obs_dur (h):"),
                        ui.input_numeric("obs_dur", "", value=5.0, min=0, step=0.1),
                        width=1/2,
                        fixed_width=False,
                        heights_equal='all',
                        gap='7px',
                        fill=False,
                        fillable=True,
                    ),
                    ui.span(
                        ui.output_ui('depth_label'),
                        # bookmarks, upload, and clear
                        ui.tooltip(
                            ui.input_action_link(
                                id='bookmark_depth',
                                label='',
                                icon=None,
                            ),
                            'Bookmark depth model',
                            id='depth_book_tooltip',
                            placement='top',
                        ),
                        ui.tooltip(
                            ui.input_action_link(
                                id='upload_depth',
                                label='',
                                icon=fa.icon_svg("file-arrow-up", fill='black'),
                            ),
                            "Upload depth model",
                            id='depth_up_tooltip',
                            placement='top',
                        ),
                        ui.panel_conditional(
                            "input.has_depth_bookmarks",
                            ui.tooltip(
                                ui.input_action_link(
                                    id='clear_depth_bookmarks',
                                    label='',
                                    icon=fa.icon_svg("circle-xmark", style='regular', fill='black'),
                                ),
                                'Clear all depth bookmarks',
                                id='depth_clear_tooltip',
                                placement='top',
                            ),
                        ),
                    ),
                    ui.input_select(
                        id="planet_model_type",
                        label='',
                        choices=["Input"],
                    ),
                    ui.panel_conditional(
                        "input.planet_model_type == 'Input'",
                        ui.tooltip(
                            ui.input_select(
                                id="depth",
                                label="",
                                choices=list(spectra['transit']),
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
                    class_="px-2 py-1 m-0 gap-2",
                    style=card_style,
                ),
                fill=False,
                class_="p-0 m-0",
            ),
            body_args=dict(class_="p-2 m-0 gap-3"),
        ),

        # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        # The detector setup
        cs.custom_card(
            ui.card_header(
                ui.div(
                    'Detector setup ',
                    ui.panel_conditional(
                        "input.raise_detector_warning",
                        ui.tooltip(
                            fa.icon_svg("triangle-exclamation", fill='gold'),
                            'warning',
                            id='detector_warning_tooltip',
                            placement='top',
                        ),
                    ),
                ),
                class_="bg-primary",
            ),
            # pairing / aperture / disperser / filter
            ui.card(
                ui.card_body(
                    ui.panel_conditional(
                        "input.mode == 'sw_ts'",
                        ui.input_select(
                            id="pairing",
                            label="LW Pairing",
                            choices=pairings,
                            selected=list(pairings)[1],
                        ),
                    ),
                    ui.panel_conditional(
                        "['sw_tsgrism', 'bots', 'sw_ts', 'lw_ts', 'target_acq'].includes(input.mode)",
                        ui.input_select(
                            id="aperture",
                            label="Aperture",
                            choices={},
                            selected='',
                        ),
                    ),
                    ui.panel_conditional(
                        "!['target_acq', 'imaging_ts', 'sw_ts', 'lw_ts', 'bots'].includes(input.mode)",
                        ui.input_select(
                            id="disperser",
                            label="Disperser",
                            choices={},
                            selected='',
                        ),
                    ),
                    ui.panel_conditional(
                        "!['lrsslitless', 'lrsslit', 'mrs_ts'].includes(input.mode)",
                        ui.input_select(
                            id="filter",
                            label="Filter",
                            choices={},
                            selected='',
                        ),
                    ),
                    class_="px-2 py-1 m-0 gap-2",
                    style=card_style,
                ),
                fill=False,
                class_="p-0 m-0",
            ),
            # subarray / readout / order
            ui.card(
                ui.card_body(
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
                    class_="px-2 py-1 m-0 gap-2",
                    style=card_style,
                ),
                fill=False,
                class_="p-0 m-0",
            ),
            # groups and integrations
            ui.card(
                ui.card_body(
                    ui.span(
                        'Groups per integration ',
                        ui.tooltip(
                            ui.input_action_link(
                                id='calc_saturation',
                                label='',
                                icon=fa.icon_svg("circle-play", fill='black'),
                            ),
                            'Estimate saturation level',
                            placement='top',
                        ),
                    ),
                    ui.panel_conditional(
                        "input.mode == 'target_acq'",
                        ui.input_select(
                            id="ngroup_acq",
                            label="",
                            choices=['3'],
                            selected='3',
                        ),
                    ),
                    ui.panel_conditional(
                        "input.mode != 'target_acq'",
                        ui.input_numeric(
                            id="ngroup",
                            label="",
                            value=2,
                            min=2, max=10000,
                        ),
                        ui.layout_columns(
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
                            col_widths=[12,12],
                            gap='4px',
                            class_="px-0 pb-2 m-0",
                        ),
                    ),
                    # Saturation goal
                    ui.p("Saturation fraction (%)", class_='py-0 my-0'),
                    ui.layout_column_wrap(
                        ui.input_numeric(
                            id="saturation_input_text",
                            label="",
                            value=80.0,
                            min=1.0, max=100.0,
                        ),
                        ui.tooltip(
                            ui.input_action_button(
                                id="saturation_button",
                                label="Set groups",
                                class_="btn btn-outline-secondary btn-sm pt-1 mt-1",
                            ),
                            'Update groups up to the saturation fraction',
                            id="saturation_tooltip",
                            placement="top",
                        ),
                        width=1/2,
                        fixed_width=False,
                        heights_equal='all',
                        gap='7px',
                        fill=False,
                        fillable=True,
                        class_="px-0 pt-0 pb-2 m-0",
                    ),
                    class_="px-2 py-1 m-0 gap-2",
                    style=card_style,
                ),
                fill=False,
                class_="p-0 m-0",
            ),
            # Search nearby Gaia targets for acquisition
            ui.panel_conditional(
                "input.mode == 'target_acq'",
                ui.card(
                    ui.card_body(
                        ui.tooltip(
                            ui.markdown('**Acquisition targets**'),
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
                        class_="px-2 py-1 m-0 gap-2",
                        style=card_style,
                    ),
                    fill=False,
                    class_="p-0 m-0",
                ),
            ),
            body_args=dict(class_="p-2 m-0 gap-3"),
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
)


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def server(input, output, session):
    bookmarked_sed = reactive.Value(False)
    bookmarked_depth = reactive.Value(False)
    saturation_fraction = reactive.Value(80)
    saturation_label = reactive.Value(None)
    update_catalog_flag = reactive.Value(False)
    update_sed_flag = reactive.Value(None)
    update_depth_flag = reactive.Value(None)
    uploaded_units = reactive.Value(None)
    warning_text = reactive.Value('')
    acq_target_list = reactive.Value(None)
    current_acq_science_target = reactive.Value(None)
    preset_sed = reactive.Value(None)
    preset_obs_dur = reactive.Value(None)
    esasky_command = reactive.Value(None)
    programs_info = reactive.Value(None)
    tso_draw = reactive.Value(None)
    clipboard = reactive.Value('')
    latest_pandeia = reactive.Value(None)


    # Invisible flags
    @reactive.effect
    @reactive.event(bookmarked_sed)
    def _():
        value = len(bookmarked_spectra['sed']) > 0
        ui.update_switch('has_sed_bookmarks', value=value)

    @reactive.effect
    @reactive.event(bookmarked_depth)
    def _():
        obs_geometry = input.obs_geometry.get()
        value = len(bookmarked_spectra[obs_geometry]) > 0
        ui.update_switch('has_depth_bookmarks', value=value)


    # Track if current target has unsaved changes
    original_values = reactive.value({})

    @reactive.effect
    @reactive.event(input.target)
    def load_target_defaults():
        """Store target's original values when selected"""
        name = input.target.get()
        target = catalog.get_target(name, is_transit=None, is_confirmed=None)
        if target is None:
            return

        original_values.set({
            't_eff': target.teff,
            'log_g': target.logg_star,
            'ks_mag': target.ks_mag,
            't_dur': target.transit_dur,
        })
        ui.update_switch('is_custom', value=False)

    @reactive.effect
    @reactive.event(input.t_eff, input.log_g, input.magnitude, input.magnitude_band, input.t_dur)
    def detect_target_changes():
        """Detect changes in target input fields"""
        orig = original_values.get()
        if not orig:
            return

        orig_t_eff = orig.get('t_eff')
        orig_log_g = orig.get('log_g')
        orig_ks_mag = orig.get('ks_mag')
        orig_t_dur = orig.get('t_dur')

        t_eff = input.t_eff()
        log_g = input.log_g()
        band = input.magnitude_band()
        mag = input.magnitude()
        t_dur = input.t_dur()

        is_custom = bool(
            t_eff != orig_t_eff or
            log_g != orig_log_g or
            (band == '2mass,ks' and mag != orig_ks_mag) or
            t_dur != orig_t_dur
        )
        ui.update_switch('is_custom', value=is_custom)


    @reactive.effect
    @reactive.event(input.save_custom_target)
    def _():
        """Save modified target to custom_targets.txt"""
        name = input.target.get()
        target = catalog.get_target(name, is_transit=None, is_confirmed=None)
        if target is None:
            return

        # Update target with UI values
        target.teff = _safe_num(input.t_eff(), default=np.nan)
        target.logg_star = _safe_num(input.log_g(), default=np.nan)
        target.transit_dur = _safe_num(input.t_dur(), default=np.nan)
        if input.magnitude_band() == '2mass,ks':
            target.ks_mag = _safe_num(input.magnitude(), default=np.nan)

        custom_catalog = f'{ROOT}data/custom_targets.txt'
        custom_targets = cat.load_targets(custom_catalog)
        # Search and replace or add custom target
        custom_planets = [target.planet for target in custom_targets]
        if target.planet in custom_planets:
            idx = custom_planets.index(target.planet)
            custom_targets[idx] = target
        else:
            custom_targets.append(target)
        cat.save_targets(custom_targets, custom_catalog)

        # Update original values to current values after successful save
        original_values.set({
            't_eff': input.t_eff(),
            'log_g': input.log_g(),
            'ks_mag': input.magnitude(),
            't_dur': input.t_dur(),
        })
        ui.update_switch('is_custom', value=False)
        msg = f"Updated {target.planet} into {custom_catalog}"
        ui.notification_show(msg, type='message', duration=6)


    @reactive.effect
    @reactive.event(input.main_status)
    def status_modal():
        with open(f'{ROOT}/data/last_updated_trexolist.txt', 'r') as f:
            last_trexo = f.readline().replace('_','-')
        with open(f'{ROOT}/data/last_updated_nea.txt', 'r') as f:
            last_nasa = f.readline().replace('_','-')
        button_width = '95%'

        if latest_pandeia.get() is None:
            latest_pandeia.set(get_latest_pandeia_release())

        gen_tso_status = get_version_advice(gen_tso)
        pandeia_version_status = get_pandeia_advice(
            pandeia.engine, latest_pandeia.get(),
        )
        pandeia_data_status = check_pandeia_ref_data(latest_pandeia.get())
        pandeia_psf_status = check_pandeia_ref_data(latest_pandeia.get(), 'psf')
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
            pandeia_version_status,
            pandeia_data_status,
            pandeia_psf_status,
            ui.hr(),
            title=ui.markdown("**Status**"),
            easy_close=True,
            size='l',
        )
        ui.modal_show(m)

    @reactive.effect
    @reactive.event(input.bibtex)
    def _():
        bibtex = textwrap.dedent("""\
        @ARTICLE{Cubillos2024paspGenTSO,
            author = {Cubillos, Patricio E.},
            title = "{Gen TSO: A General JWST Simulator for Exoplanet Time-series Observations}",
            journal = {PASP},
            keywords = {Exoplanets, Time series analysis, Astronomy databases, 498, 1916, 83, Astrophysics - Earth and Planetary Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics},
            year = 2024,
            month = dec,
            volume = {136},
            number = {12},
            eid = {124501},
            pages = {124501},
            doi = {10.1088/1538-3873/ad8fd4},
            archivePrefix = {arXiv},
            eprint = {2410.04856},
            primaryClass = {astro-ph.EP},
            adsurl = {https://ui.adsabs.harvard.edu/abs/2024PASP..136l4501C},
            adsnote = {Provided by the SAO/NASA Astrophysics Data System}
        }
        """)

        color_syntax_bibtex = ui.HTML(
            f'<pre><code class="language-java">{bibtex}</code></pre>'
            "<script>hljs.highlightAll();</script>"
        )

        m = ui.modal(
            ui.HTML(f'<pre style=" font-size:13px; margin:0;">{color_syntax_bibtex}</pre>'),
            ui.div(
                ui.input_action_button(
                    id='copy_bibtex',
                    label='Copy to clipboard',
                    class_='btn btn-primary',
                ),
                class_='d-flex justify-content-end mb-2'
            ),
            title="Gen TSO bibtex citation",
            size='l',
            easy_close=True,
            fade=True,
        )
        clipboard.set(bibtex)
        ui.modal_show(m)

    @reactive.effect
    @reactive.event(input.update_trexo)
    def _():
        cat.fetch_trexolist()
        catalog, is_jwst, is_transit, is_confirmed = load_catalog()

    @reactive.effect
    @reactive.event(input.update_nasa)
    def _():
        cat.update_exoplanet_archive()
        catalog, is_jwst, is_transit, is_confirmed = load_catalog()
        update_catalog_flag.set(~update_catalog_flag.get())

    @reactive.effect
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
        config = parse_instrument(
            input, 'instrument', 'mode', 'aperture', 'disperser', 'filter',
            'subarray', 'readout', 'order', 'ngroup', 'nint',
            'pairing', 'pupil', 'detector',
        )
        inst, mode, aperture, disperser, filter, subarray, readout = config[0:7]
        order, ngroup, nint, pairing, pupil, detector = config[7:]

        inst_label = detector.instrument_label(disperser, filter)

        run_is_tso = True
        if mode == 'target_acq':
            run_is_tso = False
            run_type = 'Acquisition'

        # Target setup:
        target_focus = input.target_focus.get()
        target_name = input.target.get()
        t_eff = _safe_num(input.t_eff.get(), default=1400.0, cast=float)
        log_g = _safe_num(input.log_g.get(), default=4.5, cast=float)
        obs_geometry = input.obs_geometry.get()
        transit_dur = _safe_num(input.t_dur.get(), default=2.0, cast=float)
        obs_dur = _safe_num(input.obs_dur.get(), default=1.0, cast=float)

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
            input, spectra, target_acq_mag=target_acq_mag,
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
            spectra['sed'][sed_label] = {'wl': wl, 'flux': flux, 'filename':None}
            bookmarked_spectra['sed'].append(sed_label)

        depth_label, wl, depth = parse_depth_model(input, spectra)
        if not run_is_tso:
            tso = pando.perform_calculation(
                ngroup, nint,
                disperser, filter, subarray, readout, aperture, order,
            )
        else:
            if depth_label is None:
                msg = f"**Error:**<br>No {obs_geometry} depth model to simulate"
                ui.notification_show(ui.markdown(msg), type="error", duration=5)
                return
            run_type = obs_geometry.capitalize()
            if depth_label not in spectra:
                if planet_model_type != 'Input':
                    spectra[obs_geometry][depth_label] = {
                        'wl': wl, 'depth': depth,
                    }
                if depth_label not in bookmarked_spectra[obs_geometry]:
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
            pairing=pairing,
            pupil=pupil,
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
            if isinstance(tso, list):
                reports = [report['scalar'] for report in tso]
                warnings = tso[0]['warnings']
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
            mode, aperture, disperser, filter, subarray, order, sed_label,
        )
        pixel_rate, full_well = jwst.extract_flux_rate(tso, get_max=True)
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


    @reactive.effect
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
        aperture = tso['aperture']
        filter = tso['filter']
        disperser = tso['disperser']
        if mode == 'bots':
            filter = f"{tso['disperser']}/{tso['filter']}"
            disperser = None
        subarray = tso['subarray']
        readout = tso['readout']
        order = tso['order']
        ngroup = tso['ngroup']

        ui.update_navset('instrument', selected=instrument)
        mode_choices = modes[instrument]
        ui.update_select('mode', choices=mode_choices, selected=mode)

        pairing = tso['pairing']
        if mode == 'sw_ts':
            ui.update_select('pairing', selected=pairing)

        apertures = detector.apertures
        if hasattr(detector, 'pupils'):
            aperture = tso['pupil']
            apertures = detector.get_constrained_val('pupils', pairing=pairing)
        ui.update_select(
            'aperture',
            label=detector.aperture_label,
            choices=apertures,
            selected=aperture,
        )

        ui.update_select(
            'disperser',
            label=detector.disperser_label,
            choices=detector.dispersers,
            selected=disperser,
        )

        choices = detector.get_constrained_val('filters', pupil=aperture)
        ui.update_select(
            'filter',
            label=detector.filter_label,
            choices=choices,
            selected=filter,
        )

        if mode in ['sw_tsgrism', 'sw_ts']:
            constraint = {'aperture': aperture}
        else:
            constraint = {'disperser': disperser}
        choices = detector.get_constrained_val('subarrays', **constraint)
        ui.update_select('subarray', choices=choices, selected=subarray)


        if mode in ['soss', 'lw_tsgrism']:
            constraint = {'subarray': subarray}
        elif mode in ['target_acq']:
            constraint = {'aperture': aperture}
        else:
            constraint = {'disperser': disperser}
        choices = detector.get_constrained_val('readouts', **constraint)
        ui.update_select('readout', choices=choices, selected=readout)

        if mode == 'soss':
            choices = detector.get_constrained_val('orders', subarray=subarray)
            order = ' '.join([str(val) for val in order])
            ui.update_select('order', choices=choices, selected=order)

        if mode == 'target_acq':
            if ngroup != input.ngroup_acq.get():
                ngroup = str(ngroup)
                choices = detector.get_constrained_val('groups', subarray=subarray)
                ui.update_select('ngroup_acq', choices=choices, selected=ngroup)
        else:
            if ngroup != input.ngroup.get():
                ui.update_numeric('ngroup', value=ngroup)


        # The target:
        current_target = input.target.get()
        current_tdur = _safe_num(input.t_dur.get(), default=2.0, cast=float)


        target_focus = tso['target_focus']
        ui.update_radio_buttons('target_focus', selected=target_focus)

        name = tso['target']
        t_dur = float(tso['transit_dur'])
        planet_model_type = tso['planet_model_type']
        ui.update_selectize('target', selected=name)
        norm_band = tso['norm_band']
        norm_mag = str(tso['norm_mag'])
        sed_type = tso['sed_type']

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
            ui.update_numeric('t_eff', value=float(tso['t_eff']))
            ui.update_numeric('log_g', value=float(tso['log_g']))
            ui.update_numeric('t_dur', value=float(t_dur))
            if target_focus == 'science':
                ui.update_select('magnitude_band', selected=norm_band)
                ui.update_numeric('magnitude', value=float(norm_mag))

        # sed_type, sed_model, norm_band, norm_mag, sed_label
        if target_focus == 'science':
            ui.update_select('sed_type', selected=sed_type)
            reset_sed = (
                sed_type != input.sed_type.get()
                or float(tso['t_eff']) != _safe_num(input.t_eff.get(), default=float(tso['t_eff']), cast=float)
                or float(tso['log_g']) != _safe_num(input.log_g.get(), default=float(tso['log_g']), cast=float)
            )
            if sed_type in sed_dict:
                if reset_sed:
                    preset_sed.set(tso['sed_model'])
                else:
                    choices = sed_dict[sed_type]
                    selected = tso['sed_model']
                    ui.update_select("sed", choices=choices, selected=selected)

        if target_focus == 'acquisition':
            selected = tso['sed_model']
            choices = sed_dict['phoenix']
            ui.update_select('ta_sed', choices=choices, selected=selected)
            target = catalog.get_target(name, is_transit=None, is_confirmed=None)
            selected = cache_acquisition[target.host]['selected']
        # The observation
        warning_text.set(tso['warnings'])
        obs_geometry = tso['obs_geometry']
        ui.update_select('obs_geometry', selected=obs_geometry)
        if float(t_dur) != float(current_tdur):
            preset_obs_dur.set(tso['obs_dur'])
        else:
            ui.update_numeric('obs_dur', value=float(tso['obs_dur']))

        choices = depth_choices[obs_geometry]
        ui.update_select(
            "planet_model_type", choices=choices, selected=planet_model_type,
        )
        if planet_model_type == 'Input':
            choices = list(spectra[obs_geometry])
            selected = tso['depth_label']
            ui.update_select("depth", choices=choices, selected=selected)
        elif planet_model_type == 'Flat':
            ui.update_numeric("transit_depth", value=tso['rprs_sq'])
        elif planet_model_type == 'Blackbody':
            ui.update_numeric("eclipse_depth", value=tso['rprs_sq'])
            ui.update_numeric("teq_planet", value=tso['teq_planet'])

        # TSO plot popover menu
        if tso['is_tso']:
            min_wl, max_wl = jwst._get_tso_wl_range(tso)
            ui.update_numeric('tso_wl_min', value=min_wl)
            ui.update_numeric('tso_wl_max', value=max_wl)

            resolution = _safe_num(input.tso_resolution.get(), default=250)
            n_obs = _safe_num(input.n_obs.get(), default=1, cast=int)
            tso_draw.set(draw(tso['tso'], resolution, n_obs))
            units = 'percent'  if obs_geometry=='transit' else 'ppm'
            ui.update_select('plot_tso_units', selected=units)
            min_depth, max_depth, step = jwst._get_tso_depth_range(
                tso, resolution, units,
            )
            ui.update_numeric('tso_depth_min', value=min_depth, step=step)
            ui.update_numeric('tso_depth_max', value=max_depth, step=step)


    @render.image
    def tso_logo():
        img = {
            "src": f'{ROOT}data/images/gen_tso_logo.png',
            "height": "45px",
        }
        return img


    @reactive.effect
    @reactive.event(input.export_button)
    def export_to_notebook():
        # For fixed values
        script = export_script_fixed_values(
            input, spectra, saturation_fraction,
            acquisition_targets, acq_target_list,
        )
        fixed_script = ui.HTML(
            f'<pre><code class="language-python">{script}</code></pre>'
            "<script>hljs.highlightAll();</script>"
        )
        # For calculated values
        script = export_script_calculated_values(
            input, spectra, saturation_fraction,
            acquisition_targets, acq_target_list, catalog,
        )
        calculated_script = ui.HTML(
            f'<pre><code class="language-python">{script}</code></pre>'
            "<script>hljs.highlightAll();</script>"
        )

        clipboard.set(script)
        m = ui.modal(
            ui.markdown("**TSO script/notebook**"),
            ui.div(
                ui.input_action_button(
                    id='copy_script',
                    label='Copy to clipboard',
                    class_='btn btn-primary',
                ),
                class_='d-flex justify-content-end mb-2'
            ),
            ui.navset_card_tab(
                ui.nav_panel(
                    "Fixed values",
                    fixed_script,
                ),
                ui.nav_panel(
                    "Calculated values",
                    calculated_script,
                ),
                id="selected_navset_card_tab",
            ),
            easy_close=True,
            size='l',
        )
        ui.modal_show(m)

    @reactive.effect
    @reactive.event(input.copy_script)
    async def copy_clipboard_script():
        print(input.selected_navset_card_tab.get())
        await session.send_custom_message(
            "copy_to_clipboard",
            clipboard.get(),
        )

    @reactive.effect
    @reactive.event(input.copy_bibtex)
    async def copy_bibtex_to_clipboard():
        await session.send_custom_message(
            "copy_to_clipboard",
            clipboard.get(),
        )

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Instrument and detector modes
    @reactive.effect(priority=3)
    @reactive.event(input.instrument)
    def _():
        inst = input.instrument.get()
        mode_choices = modes[inst]
        choices = []
        for m in mode_choices.values():
            choices += list(m)

        mode = input.mode.get()
        selected = mode if mode in choices else choices[0]
        ui.update_select('mode', choices=mode_choices, selected=selected)


    @reactive.effect
    @reactive.event(input.mode, input.subarray)
    def set_detector_label():
        mode = input.mode.get()
        subarray = input.subarray.get()
        if mode == 'sw_tsgrism':
            ui.update_tooltip('detector_warning_tooltip', nircam_warning)
            ui.update_switch('raise_detector_warning', value=True)
        elif subarray in stripes:
            ui.update_tooltip('detector_warning_tooltip', soss_warning)
            ui.update_switch('raise_detector_warning', value=True)
        else:
            ui.update_switch('raise_detector_warning', value=False)


    @reactive.effect(priority=2)
    @reactive.event(input.instrument, input.mode, input.pairing)
    def update_aperture_and_disperser():
        config = parse_instrument(input, 'mode', 'detector')
        if config is None:
            return
        mode, detector = config

        focus = 'science' if mode!='target_acq' else input.target_focus.get()
        ui.update_radio_buttons('target_focus', selected=focus)

        # The aperture
        choices = detector.apertures
        if hasattr(detector, 'pupils'):
            pairing = input.pairing.get()
            choices = detector.get_constrained_val('pupils', pairing=pairing)

        aperture = input.aperture.get()
        if aperture not in choices:
            aperture = detector.default_aperture
        if aperture not in choices:
            aperture = list(choices)[0]
        ui.update_select(
            'aperture',
            label=detector.aperture_label,
            choices=choices,
            selected=aperture,
        )

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


    @reactive.effect(priority=2)
    @reactive.event(input.instrument, input.mode, input.aperture)
    def update_filter():
        config = parse_instrument(input, 'mode', 'aperture', 'detector')
        if config is None:
            return
        mode, aperture, detector = config

        # override pupil-to-aperture correction
        aperture = input.aperture.get()
        choices = detector.get_constrained_val('filters', pupil=aperture)

        filter = input.filter.get()
        if filter not in choices:
            filter = detector.default_filter
        if filter not in choices:
            filter = list(choices)[0]
        ui.update_select(
            'filter',
            label=detector.filter_label,
            choices=choices,
            selected=filter,
        )


    @reactive.effect(priority=1)
    @reactive.event(
        input.instrument, input.mode,
        input.aperture, input.disperser, input.filter,
    )
    def update_subarray():
        config = parse_instrument(
            input, 'mode', 'aperture', 'disperser', 'filter', 'detector',
        )
        if config is None:
            return
        mode, aperture, disperser, filter, detector = config

        if mode in ['sw_tsgrism', 'sw_ts']:
            constraint = {'aperture': aperture}
        else:
            constraint = {'disperser': disperser}
        choices = detector.get_constrained_val('subarrays', **constraint)

        subarray = input.subarray.get()
        if subarray not in choices:
            subarray = detector.default_subarray
        if subarray not in choices:
            subarray = list(choices)[0]
        ui.update_select('subarray', choices=choices, selected=subarray)


    @reactive.effect(priority=1)
    @reactive.event(
        input.instrument, input.mode,
        input.aperture, input.disperser, input.subarray,
    )
    def update_readout():
        config = parse_instrument(
            input, 'mode', 'aperture', 'disperser', 'subarray', 'detector',
        )
        if config is None:
            return
        mode, aperture, disperser, subarray, detector = config

        if mode in ['soss', 'lw_tsgrism']:
            constraint = {'subarray': subarray}
        elif mode in ['target_acq']:
            constraint = {'aperture': aperture}
        else:
            constraint = {'disperser': disperser}
        choices = detector.get_constrained_val('readouts', **constraint)
        readout = input.readout.get()
        if readout not in choices:
            readout = detector.default_readout
        ui.update_select('readout', choices=choices, selected=readout)


    @reactive.effect
    @reactive.event(input.run_pandeia)
    def _():
        target_focus = input.target_focus.get()
        name = input.target.get()
        target = catalog.get_target(name, is_transit=None, is_confirmed=None)

        error_msg = None
        if target_focus=='acquisition':
            if target is None:
                error_msg = ui.markdown(
                    "Need to select a valid **Science target** before "
                    "searching for nearby acquisition targets"
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
            ui.input_checkbox(
                id="is_light",
                label="Remove '2d' and '3d' fields (much smaller file size)",
                value=True,
                width='100%',
            ),
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

        is_light = input.is_light.get()
        jwst.save_tso(savefile, tso_run['tso'], lightweight=is_light)
        ui.modal_remove()
        ui.notification_show(
            f"TSO model saved to file: '{savefile}'",
            type="message",
            duration=5,
        )


    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Target
    @reactive.effect
    @reactive.event(input.target_filter, update_catalog_flag)
    def _():
        update_catalog_flag.get()
        custom_mask = np.array(
            [getattr(t, 'is_custom', False) for t in catalog.targets],
            dtype=bool,
        )

        mask = np.zeros(nplanets, bool)
        if 'jwst' in input.target_filter.get():
            mask |= is_jwst
        if 'transit' in input.target_filter.get():
            mask |= is_transit
        if 'non_transit' in input.target_filter.get():
            mask |= ~is_transit
        if 'tess' in input.target_filter.get():
            mask |= ~is_confirmed
        if 'custom' in input.target_filter.get():
            mask |= custom_mask

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
    @reactive.event(input.target)
    async def _():
        name = input.target.get()
        target = catalog.get_target(name, is_transit=None, is_confirmed=None)
        if target is None:
            return ''

        url = f'{nasa_url}/{target.planet}'
        await session.send_custom_message(
            "set_nasa_href",
            {"url": url}
        )


    @reactive.effect
    @reactive.event(input.show_info)
    def _():
        """
        Display system parameters
        """
        name = input.target.get()
        target = catalog.get_target(name, is_transit=None, is_confirmed=None)
        planet_info, star_info, aliases = pretty_print_target(target)
        clipboard.set(target.machine_readable_text())

        info = ui.layout_columns(
            ui.span(planet_info, style="font-family: monospace;"),
            ui.span(star_info, style="font-family: monospace;"),
            width=1/2,
        )

        m = ui.modal(
            info,
            ui.span(aliases, style="font-family: monospace;"),
            title=ui.markdown(f'System parameters for: **{target.planet}**'),
            size='l',
            easy_close=True,
            footer=ui.input_action_button(
                id="copy_planet",
                label="Copy to clipboard",
                class_='btn btn-sm',
            ),
        )
        ui.modal_show(m)


    @reactive.effect
    @reactive.event(input.copy_planet)
    async def copy_clipboard():
        await session.send_custom_message(
            "copy_to_clipboard",
            clipboard.get(),
        )

    @reactive.effect
    @reactive.event(input.copy_status)
    def notify_copy_status():
        status = input.copy_status()
        msg_type = 'message' if status == "success" else 'warning'
        if status == "success":
            msg = "Copied to clipboard!"
        elif status == "failure":
            msg = "Failed to copy!"
        ui.notification_show(msg, type=msg_type, duration=4)


    @reactive.effect
    @reactive.event(input.show_observations)
    def _():
        """
        Display JWST observations
        """
        name = input.target.get()
        target = catalog.get_target(name, is_transit=None, is_confirmed=None)
        if target is None or not hasattr(target, 'programs'):
            return

        programs_info.set(target.programs)

        keys = ui.HTML(
            'Keys:<br>'
            '<span style="color:#0B980D">Observed, publicly available.</span>'
            '<br><span style="color:#FFa500">Observed, in proprietary period.</span><br>'
            '<span>To be observed, planned window.</span>'
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
    @reactive.event(programs_info)
    def trexo_df():
        data = programs_info.get()
        nobs = len(data)

        today = datetime.today()
        date_obs = [obs['date_start'] for obs in data]
        plan_obs = [obs['plan_window'] for obs in data]
        propriety = [obs['proprietary_period'] for obs in data]
        warnings = [
            i for i,obs in enumerate(data)
            if obs['status'] in ['Skipped', 'Failed', 'Withdrawn']
        ]
        available = []
        private = []
        dates = []
        tbd_dates = []
        for i in range(nobs):
            if isinstance(date_obs[i], datetime):
                release = date_obs[i] + timedelta(days=365.0*propriety[i]/12)
                if release < today:
                    available.append(i)
                else:
                    private.append(i)
                dates.append(
                    date_obs[i].strftime('%Y-%m-%d') +
                    f' ({propriety[i]} m)'
                )
            else:
                if plan_obs[i] is None:
                    dates.append(f'--- ({propriety[i]} m)')
                elif '-' in plan_obs[i]:
                    date = plan_obs[i][0:plan_obs[i].index(' ')]
                    dates.append(f'{date} ({propriety[i]} m)')
                else:
                    dates.append(f'{plan_obs[i]} ({propriety[i]} m)')

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
                'rows': private,
                'style': {"color": "#FFa500"},
            },
        ]
        programs = [
            ui.tags.a(
                f"{obs['category']} {obs['pid']}",
                href=stsci_url.replace('PID', obs['pid']),
                target="_blank",
            )
            for obs in data
        ]
        planets = [', '.join(obs['planets']) for obs in data]

        data_df = {
            'Program ID': programs,
            'PI': [obs['pi'] for obs in data],
            'Target name': [obs['target'] for obs in data],
            'Planet(s)': planets,
            'Event': [obs['event'] for obs in data],
            'Status': [obs['status'] for obs in data],
            'Instrument': [obs['instrument'] for obs in data],
            'Mode': [obs['mode'] for obs in data],
            'Disperser': [obs['disperser'] for obs in data],
            'Filter': [obs['filter'] for obs in data],
            'Subarray': [obs['subarray'] for obs in data],
            'Readout': [obs['readout'] for obs in data],
            'Groups': [obs['groups'] for obs in data],
            'Duration (h)': [obs['duration'] for obs in data],
            'Obs date (prop. period)': dates,
        }
        df = pd.DataFrame(data=data_df)

        return render.DataGrid(
            df,
            styles=styles,
            width='100%',
        )


    @reactive.effect
    @reactive.event(input.target)
    def update_target_icons():
        name = input.target.get()
        target = catalog.get_target(name, is_transit=None, is_confirmed=None)

        # Aliases
        if target is None:
            info_label = 'System info'
        else:
            aliases = [alias for alias in target.aliases if alias != name]
            if len(aliases) > 0:
                aliases = ', '.join(aliases)
                info_label = f"Also known as: {aliases}"
            else:
                info_label = 'System info'
        ui.update_tooltip('target_info_tooltip', info_label)

        # JWST flag
        if target is None:
            icon_color = 'gray'
            tip = 'not a JWST target (yet)'
        elif target.is_jwst_planet:
            icon_color = '#FAC205'
            tip = 'This is a JWST target'
        elif target.is_jwst_host:
            icon_color = 'goldenrod'
            tip = ui.markdown("This *host* is a JWST target")
        else:
            icon_color = 'gray'
            tip = 'not a JWST target (yet)'
        icon = fa.icon_svg("circle-info", fill=icon_color)

        ui.update_action_link('show_observations', icon=icon)
        ui.update_tooltip('jwst_tooltip', tip)

        if target is None:
            return

        # Confirmed or candidate
        ui.update_switch('is_candidate', value=not target.is_confirmed)


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

        def to_float(value):
            """Convert value to float, return NaN for empty/invalid"""
            if value is None:
                return np.nan
            try:
                s = str(value).strip()
                if s == '':
                    return np.nan
                return float(s)
            except Exception:
                return np.nan

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

        ui.update_numeric('t_eff', value=to_float(t_eff))
        ui.update_numeric('log_g', value=to_float(log_g))
        ui.update_select('magnitude_band', selected=band)
        ui.update_numeric('magnitude', value=to_float(magnitude))
        if t_dur == '':
            t_dur = '0.0'
        ui.update_numeric('t_dur', value=to_float(t_dur))

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


    @reactive.effect
    @reactive.event(input.sed_type, input.t_eff, input.log_g, update_sed_flag)
    def choose_sed():
        sed_type = input.sed_type.get()
        if sed_type in sed_dict:
            choices, selected = get_auto_sed(input)
            if preset_sed.get() is not None:
                selected = preset_sed.get()
                preset_sed.set(None)
        elif sed_type == 'blackbody':
            t_eff = _safe_num(input.t_eff.get(), default=0.0, cast=float)
            selected = f' Blackbody (Teff={t_eff:.0f} K)'
            choices = [selected]
        elif sed_type == 'input':
            # bookmarking SED create spectra['sed'] items,
            # a None filename flags them out of the 'input' list
            choices = [
                sed for sed,model in spectra['sed'].items()
                if model['filename'] is not None
            ]
            selected = None

        ui.update_select("sed", choices=choices, selected=selected)


    @reactive.effect
    @reactive.event(
        bookmarked_sed, input.sed,
        input.t_eff, input.magnitude_band, input.magnitude,
    )
    def update_sed_bookmark_icon():
        """Check current SED is bookmarked"""
        sed_type, sed_model, norm_band, norm_mag, sed_label = parse_sed(input, spectra)
        is_bookmarked = sed_label in bookmarked_spectra['sed']
        bookmarked_sed.set(is_bookmarked)
        style = 'solid' if is_bookmarked else 'regular'
        fill = 'gold' if is_bookmarked else 'black'
        sed_icon = fa.icon_svg("star", style=style, fill=fill)
        ui.update_action_link('bookmark_sed', icon=sed_icon)


    @reactive.effect
    @reactive.event(input.bookmark_sed)
    def _():
        """Toggle bookmarked SED"""
        sed_type, sed_model, norm_band, norm_mag, sed_label = parse_sed(input, spectra)
        if sed_type is None:
            msg = ui.markdown("**Error**:<br>No SED model to bookmark")
            ui.notification_show(msg, type="error", duration=5)
            return
        is_bookmarked = not bookmarked_sed.get()
        bookmarked_sed.set(is_bookmarked)
        if is_bookmarked:
            scene = jwst.make_scene(sed_type, sed_model, norm_band, norm_mag)
            wl, flux = jwst.extract_sed(scene, wl_range=[0.3,30.0])
            spectra['sed'][sed_label] = {'wl': wl, 'flux': flux, 'filename':None}
            bookmarked_spectra['sed'].append(sed_label)
        else:
            bookmarked_spectra['sed'].remove(sed_label)

    @reactive.effect
    @reactive.event(input.clear_sed_bookmarks)
    def _():
        """Clear all bookmarked SEDs"""
        bookmarked_spectra['sed'].clear()
        bookmarked_sed.set(False)
        update_sed_flag.set('cleared')
        ui.notification_show("Cleared all SED bookmarks", type="message", duration=3)


    @reactive.effect
    @reactive.event(input.obs_geometry)
    def _():
        """Set depth model label"""
        obs_geometry = f'{input.obs_geometry.get()} depth'
        ui.update_tooltip('depth_up_tooltip', f'Upload {obs_geometry} spectrum')
        ui.update_tooltip('depth_book_tooltip', f'Bookmark {obs_geometry} spectrum')
        ui.update_tooltip(
            'depth_clear_tooltip',
            f'Clear all bookmarked {obs_geometry} spectra',
        )


    @reactive.effect
    @reactive.event(
        bookmarked_depth, input.obs_geometry, input.planet_model_type,
        input.depth, input.transit_depth, input.eclipse_depth, input.teq_planet,
    )
    def update_depth_bookmark_icon():
        """Check current transit-depth is bookmarked"""
        obs_geometry = input.obs_geometry.get()
        depth_label = planet_model_name(input)
        is_bookmarked = depth_label in bookmarked_spectra[obs_geometry]
        bookmarked_depth.set(is_bookmarked)

        fill = 'royalblue' if is_bookmarked else 'gray'
        icon = fa.icon_svg("earth-americas", style='solid', fill=fill)
        ui.update_action_link('bookmark_depth', icon=icon)


    @reactive.effect
    @reactive.event(input.bookmark_depth)
    def _():
        """Toggle bookmarked depth model"""
        obs_geometry = input.obs_geometry.get()
        depth_label = planet_model_name(input)
        planet_model_type = input.planet_model_type.get()
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
            depth_label, wl, depth = parse_depth_model(input, spectra)
            if planet_model_type != 'Input':
                spectra[obs_geometry][depth_label] = {'wl': wl, 'depth': depth}
        else:
            bookmarked_spectra[obs_geometry].remove(depth_label)

    @reactive.effect
    @reactive.event(input.clear_depth_bookmarks)
    def _():
        """Clear bookmarked depth models for the current geometry"""
        obs_geometry = input.obs_geometry.get()
        bookmarked_spectra[obs_geometry].clear()
        bookmarked_depth.set(False)
        update_depth_flag.set('cleared')
        ui.notification_show(f"Cleared all {obs_geometry} depth bookmarks", type="message", duration=3)


    @reactive.effect
    @reactive.event(input.obs_geometry, update_depth_flag)
    def update_depth_types_and_models():
        obs_geometry = input.obs_geometry.get()
        choices = depth_choices[obs_geometry]
        model_type = input.planet_model_type.get()
        if model_type not in choices:
            model_type = choices[0]
        ui.update_select(
            "planet_model_type", choices=choices, selected=model_type,
        )

        models = list(spectra[obs_geometry])
        name = input.target.get()
        cached = (
            name in cache_target and
            cache_target[name]['depth_label'] is not None
        )
        selected = input.depth.get()
        if cached:
            selected = cache_target[name]['depth_label']
            cache_target[name]['depth_label'] = None
        elif selected not in models:
            selected = None if len(models) == 0 else models[0]
        ui.update_select("depth", choices=models, selected=selected)

        if len(models) > 0:
            tooltip_text = ''
        elif obs_geometry == 'transit':
            tooltip_text = f'Upload a {obs_geometry} depth spectrum'
        elif obs_geometry == 'eclipse':
            tooltip_text = f'Upload an {obs_geometry} depth spectrum'
        ui.update_tooltip('depth_tooltip', tooltip_text)


    @render.ui
    @reactive.event(input.obs_geometry)
    def depth_label():
        """Set depth model label"""
        obs_geometry = input.obs_geometry.get()
        return ui.HTML(f'<b>{obs_geometry.capitalize()} spectra</b> '),

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

    @reactive.effect
    @reactive.event(
        input.t_dur, input.settling_time, input.baseline_time,
        input.min_baseline_time,
    )
    def _():
        """Set observation time based on transit dur and popover settings"""
        if preset_obs_dur.get() is not None:
            obs_dur = preset_obs_dur.get()
            preset_obs_dur.set(None)
            ui.update_numeric('obs_dur', value=float(f'{obs_dur:.2f}'))
            return
        t_dur_val = _safe_num(req(input.t_dur).get(), default=0.0, cast=float)
        if t_dur_val == 0.0:
            ui.update_numeric('obs_dur', value=0.0)
            return
        transit_dur = t_dur_val
        settling = req(input.settling_time).get()

        baseline = req(input.baseline_time).get()
        min_baseline = req(input.min_baseline_time).get()
        baseline = np.clip(baseline*transit_dur, min_baseline, np.inf)
        # Tdwell = T_start + T_settle + T14 + 2*max(1, T14/2)
        obs_dur = 1.0 + settling + transit_dur + 2.0*baseline
        ui.update_numeric('obs_dur', value=float(f'{obs_dur:.2f}'))


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
        filename = new_model[0]['name']
        filepath = new_model[0]['datapath']
        units = uploaded_units.get()
        _, wl, model = read_spectrum_file(filepath, units, on_fail='warning')
        if wl is None:
            msg = ui.markdown(
                f'**Error:**<br>Invalid format for input file:<br>*{filename}*'
            )
            ui.notification_show(msg, type="error", duration=7)
            return
        label = os.path.splitext(filename)[0]

        if units in depth_units:
            obs_geometry = input.obs_geometry.get()
            spectra[obs_geometry][label] = {
                'wl': wl,
                'depth': model,
                'units': units,
                'filename': f'unknown_{filename}',
            }
            if label not in bookmarked_spectra[obs_geometry]:
                bookmarked_spectra[obs_geometry].append(label)
            if input.planet_model_type.get() != 'Input':
                return
            # Trigger update choose_depth
            update_depth_flag.set(label)
        elif units in sed_units:
            spectra['sed'][label] = {
                'wl': wl,
                'flux': model,
                'units': units,
                'filename': f'unknown_{filename}',
            }
            update_sed_flag.set(label)


    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Detector setup
    @reactive.effect
    @reactive.event(input.mode, input.subarray)
    def update_orders():
        config = parse_instrument(input, 'mode', 'subarray', 'detector')
        if config is None or config[0] != 'soss':
            return

        mode, subarray, detector = config
        choices = detector.get_constrained_val('orders', subarray=subarray)
        order = input.order.get()
        if order not in choices:
            order = list(choices)[0]
        ui.update_select('order', choices=choices, selected=order)


    @reactive.effect
    @reactive.event(input.mode, input.subarray)
    def update_groups_input():
        config = parse_instrument(
            input, 'mode', 'aperture', 'disperser', 'filter',
            'subarray', 'detector',
        )
        if config is None:
            return
        mode, aperture, disperser, filter, subarray, detector = config

        if mode == 'target_acq':
            ngroup = input.ngroup_acq.get()
        else:
            ngroup = input.ngroup.get()

        if mode == 'target_acq':
            choices = detector.get_constrained_val('groups', subarray=subarray)
            if not ngroup.isnumeric() or int(ngroup) not in choices:
                ngroup = choices[0]
            ngroup = str(ngroup)
            ui.update_select(id="ngroup_acq", choices=choices, selected=ngroup)
        else:
            ui.update_numeric(id="ngroup", value=ngroup)


    @reactive.effect
    @reactive.event(input.calc_saturation)
    def calculate_saturation_level():
        config = parse_instrument(
            input, 'instrument', 'mode', 'aperture', 'disperser', 'filter',
            'subarray', 'readout', 'order', 'ngroup',
        )
        inst, mode, aperture, disperser, filter, subarray, readout = config[0:7]
        order, ngroup = config[7:]

        if mode != 'target_acq':
            ngroup = 2

        sed_type, sed_model, norm_band, norm_mag, sed_label = parse_sed(input, spectra)
        sat_label = make_saturation_label(
            mode, aperture, disperser, filter, subarray, order, sed_label,
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


    @reactive.effect
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

        obs_dur = input.obs_dur.get()
        ngroup = input.ngroup.get()
        if obs_dur is None or ngroup is None:
            return

        inst = input.instrument.get().lower()
        readout = input.readout.get()
        subarray = input.subarray.get()
        integs, exp_time = jwst.bin_search_exposure_time(
            inst, subarray, readout, ngroup, obs_dur,
        )
        if exp_time == 0.0:
            return

        ui.update_numeric('integrations', value=integs)


    @reactive.effect
    def _():
        """
        Update the desired saturation limit only when there is a valid
        input value.
        """
        try:
            value = float(input.saturation_input_text())
            if 1.0 <= value <= 100.0:
                saturation_fraction.set(value)
        except:
            pass

    @reactive.effect
    @reactive.event(input.saturation_button)
    def ngroup_from_saturation_fraction():
        config = parse_instrument(
            input, 'instrument', 'mode', 'aperture', 'disperser', 'filter',
            'subarray', 'readout', 'order', 'detector',
        )
        inst, mode, aperture, disperser, filter, subarray, readout = config[0:7]
        order, detector = config[7:]

        name = input.target.get()
        target = catalog.get_target(name, is_transit=None, is_confirmed=None)
        cached_target = (
            target is not None and
            target.host in cache_acquisition and
            cache_acquisition[target.host]['selected'] is not None
        )

        target_focus = input.target_focus.get().capitalize()
        if target_focus == 'Science':
            target_acq_mag = None
        elif cached_target:
            target_list = cache_acquisition[target.host]['targets']
            selected = cache_acquisition[target.host]['selected']
            target_acq_mag = np.round(target_list[1][selected], 3)
        else:
            return

        norm_mag, sed_label = parse_sed(input, spectra, target_acq_mag)[3:5]
        if inst is None or sed_label is None:
            return

        pixel_rate, full_well = get_saturation_values(
            mode, aperture, disperser, filter, subarray, order,
            sed_label, norm_mag, cache_saturation,
        )
        if pixel_rate is None:
            return

        req_saturation = saturation_fraction.get()
        ngroup_req = jwst.groups_below_saturation(
            req_saturation, inst, subarray, readout, pixel_rate, full_well,
        )

        if mode == 'target_acq':
            choices = detector.get_constrained_val('groups', subarray=subarray)
            masked_choices = np.array(choices) <= ngroup_req
            masked_choices[0] = True
            ngroup = str(np.array(choices)[masked_choices][-1])
            ui.update_select(id='ngroup_acq', choices=choices, selected=ngroup)
        else:
            ngroup_req = int(np.clip(ngroup_req, 2, np.inf))
            ui.update_numeric(id='ngroup', value=ngroup_req)


    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Viewers
    @render_plotly
    @reactive.event(
        input.filter_filter, input.instrument, input.mode,
        input.aperture, input.disperser, input.filter, input.subarray,
    )
    def plotly_filters():
        config = parse_instrument(
            input, 'instrument', 'mode',
            'aperture', 'disperser', 'filter', 'subarray',
        )
        if config is None:
            return

        throughputs, t_config = get_throughput(input)
        inst, mode, key, filter = t_config
        show_all = input.filter_filter.get() == 'all'
        fig = plots.plotly_filters(
            throughputs, inst, mode, key, filter, show_all,
        )
        return fig


    @render_plotly
    def plotly_sed():
        bookmarked_sed.get() # (make panel reactive to remove all bookmarks)
        update_sed_flag.get()

        input.bookmark_sed.get()  # (make panel reactive to bookmark_sed)
        throughput = get_throughput(input, evaluate=True)
        if throughput is None:
            return

        # Gather bookmarked SEDs
        model_names = bookmarked_spectra['sed']
        if len(model_names) == 0:
            fig = go.Figure()
            fig.update_layout(title='Bookmark SEDs models to show them here')
            return fig

        sed_models = [spectra['sed'][model] for model in model_names]
        current_model = parse_sed(input, spectra)[-1]

        wl_scale = input.plot_sed_xscale.get()
        wl_range = [input.sed_wl_min.get(), input.sed_wl_max.get()]
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
    @reactive.event(
        input.bookmark_depth, update_depth_flag, input.clear_depth_bookmarks,
        input.plot_depth_xscale, input.depth_wl_min, input.depth_wl_max,
        input.plot_depth_units, input.depth_resolution, input.obs_geometry,
        input.instrument, input.mode,
        input.aperture, input.disperser, input.filter, input.subarray,
    )
    def plotly_depth():
        input.bookmark_depth.get()  # (make panel reactive to bookmark_depth)
        throughput = get_throughput(input, evaluate=True)
        if throughput is None:
            return

        update_depth_flag.get()
        obs_geometry = input.obs_geometry.get()
        model_names = bookmarked_spectra[obs_geometry]
        nmodels = len(model_names)
        if nmodels == 0:
            fig = go.Figure()
            title = f'Bookmark {obs_geometry.lower()} models to show them here'
            fig.update_layout(title=title)
            return fig

        current_model = planet_model_name(input)
        wl_scale = input.plot_depth_xscale.get()
        wl_range = [input.depth_wl_min.get(), input.depth_wl_max.get()]
        units = input.plot_depth_units.get()
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
                resolution=input.tso_resolution.get(),
                model_label=planet,
                instrument_label=tso_run['inst_label'],
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
        resolution = _safe_num(input.tso_resolution.get(), default=250)
        units = input.plot_tso_units.get()

        min_depth, max_depth, step = jwst._get_tso_depth_range(
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

        n_obs = _safe_num(input.n_obs.get(), default=1, cast=int)
        resolution = _safe_num(input.tso_resolution.get(), default=250)
        tso_draw.set(draw(tso['tso'], resolution, n_obs))


    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Results
    @render.ui
    def results():
        saturation_limit = saturation_fraction.get()
        warnings = {}
        # Only read for reactivity reasons:
        saturation_label.get()
        input.ta_sed.get()

        config = parse_instrument(
            input, 'instrument', 'mode', 'aperture', 'disperser', 'filter',
            'subarray', 'readout', 'order', 'ngroup', 'nint',
        )
        if config is None:
            warning_text.set(warnings)
            return ui.HTML('<pre> </pre>')

        inst, mode, aperture, disperser, filter, subarray, readout = config[0:7]
        order, ngroup, nint = config[7:]

        name = input.target.get()
        target = catalog.get_target(name, is_transit=None, is_confirmed=None)
        cached_target = (
            target is not None and
            target.host in cache_acquisition and
            cache_acquisition[target.host]['selected'] is not None
        )

        if ngroup is None or parse_sed(input, spectra)[-1] is None:
            warning_text.set(warnings)
            return ui.HTML('<pre> </pre>')

        obs_geometry = input.obs_geometry.get()
        if mode == 'target_acq':
            run_type = 'Acquisition'
            depth_label = ''
        else:
            run_type = obs_geometry.capitalize()
            depth_label = parse_obs(input)[1]

        report_text = jwst._print_pandeia_exposure(
            inst, subarray, readout, ngroup, nint,
        )
        target_focus = input.target_focus.get().capitalize()

        if target_focus == 'Science':
            target_acq_mag = None
            target_name = f': {target.planet}' if target is not None else ''
        elif cached_target:
            target_list = cache_acquisition[target.host]['targets']
            selected = cache_acquisition[target.host]['selected']
            target_acq_mag = np.round(target_list[1][selected], 3)
            target_name = f': {target_list[0][selected]}'
        else:
            report_text = f'<b>{target_focus} target</b><br>{report_text}'
            warning_text.set(warnings)
            return ui.HTML(f'<pre>{report_text}</pre>')
        report_text = f'<b>{target_focus} target{target_name}</b><br>{report_text}'

        sed_type, sed_model, norm_band, norm_mag, sed_label = parse_sed(
            input, spectra, target_acq_mag=target_acq_mag,
        )
        pixel_rate, full_well = get_saturation_values(
            mode, aperture, disperser, filter, subarray, order,
            sed_label, norm_mag, cache_saturation,
        )
        if pixel_rate is not None:
            saturation_text = jwst._print_pandeia_saturation(
                inst, subarray, readout, ngroup, pixel_rate, full_well,
                format='html', req_saturation=saturation_limit,
            )
            report_text += f'<br>{saturation_text}'

        tso_label = make_obs_label(
            inst, mode, aperture, disperser, filter, subarray, readout, order,
            ngroup, nint, run_type, sed_label, depth_label,
        )

        if tso_label in tso_runs[run_type]:
            tso_run = tso_runs[run_type][tso_label]
            warnings = tso_run['warnings']
            transit_dur = _safe_num(input.t_dur.get(), default=-1.0)
            stored_dur = tso_run.get('transit_dur', 0.0)
            relative_diff = np.abs(1.0 - transit_dur/stored_dur)
            if relative_diff < 0.01:
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


    @reactive.effect
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
        chosen_sed = jwst.find_closest_sed(t_eff, log_g, sed_type='phoenix')
        choices = sed_dict['phoenix']
        ui.update_select('ta_sed', choices=choices, selected=chosen_sed)

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
