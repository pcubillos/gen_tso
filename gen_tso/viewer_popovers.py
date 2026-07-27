# Copyright (c) 2025-2026 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
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
