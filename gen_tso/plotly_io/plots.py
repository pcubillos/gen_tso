# Copyright (c) 2024 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    'plotly_filters',
    'plotly_sed_spectra',
    'plotly_depth_spectra',
    'plotly_tso_spectra',
]

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
#import pyratbay.spectrum as ps
#import pyratbay.tools as pt
from ..utils import (
    bin_spectrum,
    constant_resolution_spectrum,
    u,
)


def plotly_filters(passbands, inst_name, subarray, filter_name, show_all):
    """
    Make a plotly figure of the passband filters.
    """
    nirspec_filters = [
        'g140h/f070lp',
        'g140m/f070lp',
        'g140h/f100lp',
        'g140m/f100lp',
        'g235h/f170lp',
        'g235m/f170lp',
        'g395h/f290lp',
        'g395m/f290lp',
        'prism/clear',
    ]

    if inst_name is None:
        return go.Figure()

    instruments = [inst_name]
    if show_all:
        instruments += [
            inst for inst in passbands.keys()
            if inst not in instruments
        ]

    # Parse filters to plot
    all_filters = {}
    nfilters = 0
    for inst in instruments:
        all_filters[inst] = {}

        if inst in ['nircam', 'miri']:
            subarray = list(passbands[inst].keys())[0]
        elif inst != inst_name:
            if inst == 'niriss':
                subarray = 'substrip256'
            elif inst == 'nirspec':
                subarray = 'sub2048'
        elif subarray not in passbands[inst_name]:
            return go.Figure()

        if inst == inst_name:
            filters = list(passbands[inst][subarray].keys())
            if inst == 'nirspec':
                filters = [f for f in nirspec_filters if f in filters]
        else:
            if inst == 'nircam':
                filters = ['f322w2', 'f444w']
            elif inst == 'nirspec':
                filters = ['g140h/f100lp', 'g235h/f170lp', 'g395h/f290lp', 'prism/clear']
            elif inst == 'niriss':
                filters = ['clear']
            elif inst == 'miri':
                filters = ['None']

        for filter in filters:
            band = passbands[inst][subarray][filter]
            all_filters[inst][filter] = band
            if 'order2' in band:
                all_filters[inst]['order2'] = band['order2']
            nfilters += len(all_filters[inst][filter])


    visible = [None for _ in range(nfilters)]
    if inst_name == 'nirspec':
        for i,filter in enumerate(all_filters[inst_name].keys()):
            hide = ('h' in filter_name) is not ('h' in filter)
            if hide and 'prism' not in filter:
                visible[i] = 'legendonly'
    elif inst_name == 'nircam':
        for i,filter in enumerate(all_filters[inst_name].keys()):
            if filter != filter_name and filter not in ['f322w2', 'f444w']:
                visible[i] = 'legendonly'

    if inst_name == 'nirspec':
        primary_colors = px.colors.sample_colorscale(
            'Viridis', np.linspace(0, 0.8, 9),
        )
    else:
        primary_colors = px.colors.sample_colorscale(
            'Viridis', [0.1, 0.3, 0.5, 0.7],
        )
    secondary_colors = [
        px.colors.sequential.gray[3],
        px.colors.sequential.gray[5],
        px.colors.sequential.gray[7],
        px.colors.sequential.gray[8],
        px.colors.sequential.gray[9],
    ]
    secondary_nirspec = [
        px.colors.sequential.Magma[3],
        px.colors.sequential.Magma[5],
        px.colors.sequential.Magma[6],
        px.colors.sequential.Magma[7],
    ]
    sel_cols = iter(primary_colors)
    other_cols = iter(secondary_colors)
    nirspec_cols = iter(secondary_nirspec)


    fig = go.Figure()
    j = 0
    filter_index = -1
    for inst in instruments:
        for filter, throughput in all_filters[inst].items():
            if filter == filter_name:
                color = 'Gold'
                width = 3.5
                filter_index = j
            elif filter == 'order2' and filter_name == 'clear':
                color = 'Orange'
                width = 3.5
            elif inst == inst_name:
                color = next(sel_cols)
                width = 2.0
            else:
                color = next(nirspec_cols) if inst=='nirspec' else next(other_cols)
                width = 1.25
            linedict = dict(color=color, width=width)
            fig.add_trace(go.Scatter(
                x=throughput['wl'],
                y=throughput['response'],
                mode='lines',
                name=filter.upper(),
                legendgroup=inst,
                legendgrouptitle_text=inst,
                line=linedict,
                legendrank=j+nfilters*int(inst != inst_name),
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

    wl_scale = 'log'
    wl_range = [np.log10(0.6), np.log10(13.5)]
    fig.update_xaxes(
        title_text='wavelength (um)',
        title_standoff=0,
        range=wl_range,
        type=wl_scale,
    )
    fig.update_layout(
        showlegend=True,
        legend=dict(groupclick="toggleitem"),
    )

    # Show current filter on top (TBD: there must be a better way)
    if filter_index < 0:
        return fig
    fig_idx = np.arange(len(fig.data))
    fig_idx[-1] = filter_index
    fig_idx[filter_index] = len(fig.data) - 1
    fig.data = tuple(np.array(fig.data)[fig_idx])
    return fig


def plotly_sed_spectra(
        sed_models, labels, highlight_model=None,
        wl_range=[0.5,12], units='mJy', wl_scale='linear', resolution=250.0,
    ):
    """
    Make a plotly figure of stellar SED spectra.
    """
    nmodels = len(sed_models)
    fig = go.Figure()
    for j,model in enumerate(sed_models):
        wl = model['wl']
        flux = model['flux']
        if resolution > 0:
            wl_min = np.amin(wl)
            wl_max = np.amax(wl)
            bin_wl = constant_resolution_spectrum(wl_min, wl_max, resolution)
            bin_flux = bin_spectrum(bin_wl, wl, flux, ignore_gaps=True)
            mask = np.isfinite(bin_flux)
            wl = bin_wl[mask]
            flux = bin_flux[mask]

        if highlight_model is None:
            linedict = dict(width=1.25)
            rank = j
        elif labels[j] == highlight_model:
            linedict = dict(color='Gold', width=2.0)
            rank = j + nmodels
        else:
            linedict = dict(width=1.25)
            rank = j
        fig.add_trace(go.Scatter(
            x=wl,
            y=flux,
            mode='lines',
            opacity=0.75,
            name=labels[j],
            #line=linedict,
            legendrank=rank,
        ))

    fig.update_traces(
        hovertemplate=
            'wl = %{x:.2f}<br>'+
            'flux = %{y:.3f}'
    )
    fig.update_yaxes(
        title_text=f'Flux ({units})',
        title_standoff=0,
    )

    if wl_scale == 'log':
        wl_range = [np.log10(wave) for wave in wl_range]
    fig.update_xaxes(
        title_text='wavelength (um)',
        title_standoff=0,
        range=wl_range,
        type=wl_scale,
    )

    fig.update_layout(legend=dict(
        orientation="h",
        entrywidth=1.0,
        entrywidthmode='fraction',
        yanchor="bottom",
        xanchor="right",
        y=1.02,
        x=1
    ))
    fig.update_layout(showlegend=True)
    return fig


def plotly_depth_spectra(
        depth_models, labels, highlight_model=None,
        wl_range=[0.5,12], units='percent', wl_scale='linear', resolution=250.0,
        obs_geometry='Transit',
    ):
    """
    Make a plotly figure of transit/eclipse depth spectra.
    """
    nmodels = len(depth_models)
    fig = go.Figure()
    for j,model in enumerate(depth_models):
        wl = model['wl']
        depth = model['depth']
        if resolution > 0:
            wl_min = np.amin(wl)
            wl_max = np.amax(wl)
            bin_wl = constant_resolution_spectrum(wl_min, wl_max, resolution)
            depth = bin_spectrum(bin_wl, wl, depth) / u(units)
            wl = bin_wl

        if labels[j] == highlight_model:
            linedict = dict(color='Gold', width=2.0)
            rank = j + nmodels
            visible = None
        else:
            linedict = dict(width=1.25)
            rank = j
            visible = 'legendonly'
        fig.add_trace(go.Scatter(
            x=wl,
            y=depth,
            mode='lines',
            name=labels[j],
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
        title_text=f'{obs_geometry} depth ({units})'.replace('percent','%'),
        title_standoff=0,
    )

    if wl_scale == 'log':
        wl_range = [np.log10(wave) for wave in wl_range]
    fig.update_xaxes(
        title_text='wavelength (um)',
        title_standoff=0,
        range=wl_range,
        type=wl_scale,
    )

    fig.update_layout(legend=dict(
        orientation="h",
        entrywidth=1.0,
        entrywidthmode='fraction',
        yanchor="bottom",
        xanchor="right",
        y=1.02,
        x=1
    ))
    fig.update_layout(showlegend=True)
    return fig


def plotly_tso_spectra(
        wl, spec, bin_wl, bin_spec, bin_err, label,
        bin_widths=None,
        units='percent', wl_range=None, wl_scale='linear', resolution=250.0,
        obs_geometry='Transit',
    ):
    """
    Make a plotly figure of transit/eclipse depth TSO spectra.
    """
    fig = go.Figure()
    obs_col = px.colors.sample_colorscale('Viridis', 0.2)[0]
    model_col = px.colors.sample_colorscale('Viridis', 0.75)[0]

    fig.add_trace(go.Scatter(
            x=wl,
            y=spec/u(units),
            mode='lines',
            name='model',
            line=dict(color=model_col, width=1.5),
        ))
    fig.add_trace(go.Scatter(
            x=bin_wl,
            y=bin_spec/u(units),
            error_y=dict(type='data', array=bin_err/u(units), visible=True),
            mode='markers',
            name=label,
            marker=dict(color=obs_col, size=5),
        ))

    fig.update_traces(
        hovertemplate=
            'wl = %{x:.2f}<br>'+
            'depth = %{y:.3f}'
    )
    ymax = np.amax(spec)/u(units)
    ymin = np.amin(spec)/u(units)
    dy = 0.1 * (ymax-ymin)
    y_range = [ymin-dy, ymax+dy]
    title = f'{obs_geometry} depth ({units})'
    title = title.replace('percent','%').replace(' (none)', '')
    fig.update_yaxes(
        title_text=title,
        title_standoff=0,
        range=y_range,
    )

    if wl_scale == 'log' and wl_range is not None:
        wl_range = [np.log10(wave) for wave in wl_range]
    fig.update_xaxes(
        title_text='wavelength (um)',
        title_standoff=0,
        range=wl_range,
        type=wl_scale,
    )

    fig.update_layout(legend=dict(
        orientation="h",
        entrywidth=1.0,
        entrywidthmode='fraction',
        yanchor="bottom",
        xanchor="right",
        y=1.02,
        x=1
    ))
    fig.update_layout(showlegend=True)
    return fig


