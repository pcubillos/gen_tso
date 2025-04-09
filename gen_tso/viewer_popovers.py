# Copyright (c) 2025 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    'filter_popover',
    'sed_popover',
    'planet_popover',
    'tso_popover',
]

import faicons as fa
from shiny import ui


depth_units = [
    "none",
    "percent",
    "ppm",
]

wl_scales = {
    'Wavelength scale': {
        'linear': 'linear',
        'log': 'log',
    },
}

tso_choices = {
    'tso': 'TSO',
    'fluxes': 'Flux rate',
    'snr': 'S/N',
    '2d_flux': '2D flux',
    '2d_snr': '2D S/N',
    '2d_saturation': '2D saturation',
    '2d_groups': '2D groups',
}


filter_popover = ui.popover(
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
        selected="none",
    ),
    placement="top",
    id="filter_popover",
)


sed_popover = ui.popover(
    ui.span(
        fa.icon_svg("gear"),
        style="position:absolute; top: 5px; right: 7px;",
    ),
    ui.layout_column_wrap(
        ui.input_numeric(
            id='plot_sed_resolution',
            label='Resolution:',
            value=0.0,
            min=10.0, max=3000.0, step=25.0,
        ),
        ui.input_select(
            id="plot_sed_units",
            label="Flux units:",
            choices=['mJy'],
            selected='mJy',
        ),
        width=1/2,
        fixed_width=False,
        gap='5px',
        fill=False,
        fillable=True,
        class_="p-0 pb-1 m-0",
    ),
    ui.layout_column_wrap(
        "Wavelength:",
        ui.input_numeric(
            id='sed_wl_min', label='',
            value=0.45, min=0.3, max=30.0, step=0.15,
        ),
        ui.input_numeric(
            id='sed_wl_max', label='',
            value=28.0, min=0.5, max=30.0, step=1.0,
        ),
        ui.input_select(
            id="plot_sed_xscale",
            label="",
            choices=wl_scales,
            selected='log',
        ),
        width=1/4,
        fixed_width=False,
        gap='5px',
        fill=False,
        fillable=True,
        class_="p-0 m-0",
    ),
    placement="top",
    id="sed_popover",
)


planet_popover = ui.popover(
    ui.span(
        fa.icon_svg("gear"),
        style="position:absolute; top: 5px; right: 7px;",
    ),
    ui.layout_column_wrap(
        ui.input_numeric(
            id='depth_resolution',
            label='Resolution:',
            value=250.0,
            min=10.0, max=3000.0, step=25.0,
        ),
        ui.input_select(
            id="plot_depth_units",
            label="Depth units:",
            choices=depth_units,
            selected='percent',
        ),
        width=1/2,
        fixed_width=False,
        gap='5px',
        fill=False,
        fillable=True,
        class_="p-0 pb-1 m-0",
    ),
    ui.layout_column_wrap(
        "Wavelength:",
        ui.input_numeric(
            id='depth_wl_min', label='',
            value=0.6, min=0.3, max=30.0, step=0.15,
        ),
        ui.input_numeric(
            id='depth_wl_max', label='',
            value=28.0, min=0.5, max=30.0, step=1.0,
        ),
        ui.input_select(
            "plot_depth_xscale",
            label="",
            choices=wl_scales,
            selected='log',
        ),
        width=1/4,
        fixed_width=False,
        gap='5px',
        fill=False,
        fillable=True,
        class_="p-0 m-0",
    ),
    placement="top",
    id="depth_popover",
)


tso_popover = ui.popover(
    ui.span(
        fa.icon_svg("gear"),
        style="position:absolute; top: 5px; right: 7px;",
    ),
    ui.layout_column_wrap(
        ui.input_select(
            id="tso_plot",
            label="Plot:",
            choices=tso_choices,
            selected='tso',
        ),
        ui.panel_conditional(
            "input.tso_plot == 'tso'",
            ui.input_numeric(
                id='n_obs',
                label='Number of obs:',
                value=1.0,
                min=1.0, max=3000.0, step=1.0,
                width='200px',
            ),
        ),
        ui.panel_conditional(
            "input.tso_plot == 'tso'",
            ui.input_numeric(
                id='tso_resolution',
                label='Resolution:',
                value=250.0,
                min=25.0, max=3000.0, step=25.0,
                width='200px',
            ),
        ),
        ui.panel_conditional(
            "input.tso_plot == 'tso'",
            ui.input_select(
                id="plot_tso_units",
                label="Depth units:",
                choices = depth_units,
                selected='percent',
            ),
        ),
        width=1/4,
        fixed_width=False,
        gap='5px',
        fill=False,
        fillable=True,
        class_="px-0 py-1 m-0",
    ),
    ui.layout_column_wrap(
        "Wavelength:",
        ui.input_numeric(
            id='tso_wl_min', label='',
            value=None, min=0.5, max=30.0, step=0.1,
        ),
        ui.input_numeric(
            id='tso_wl_max', label='',
            value=None, min=0.5, max=30.0, step=0.1,
        ),
        ui.input_select(
            id="plot_tso_xscale",
            label='',
            choices=wl_scales,
            selected='linear',
        ),
        width=1/4,
        fixed_width=False,
        gap='5px',
        fill=False,
        fillable=True,
        class_="p-0 m-0",
    ),
    ui.panel_conditional(
        "input.tso_plot == 'tso'",
        ui.layout_column_wrap(
            "Depth:",
            ui.input_numeric(
                id='tso_depth_min',
                label='',
                value=None,
            ),
            ui.input_numeric(
                id='tso_depth_max',
                label='',
                value=None,
            ),
            ui.input_action_button(
                id="redraw_tso",
                label="Re-draw",
                class_="btn btn-outline-primary btn-sm",
            ),
            width=1/4,
            fixed_width=False,
            gap='5px',
            fill=False,
            fillable=True,
            class_="p-0 m-0",
        ),
    ),
    placement="top",
    id="tso_popover",
)
