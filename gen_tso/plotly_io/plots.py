# Copyright (c) 2024 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    'plotly_filters',
    'plotly_sed_spectra',
    'plotly_depth_spectra',
]

import numpy as np
import plotly.graph_objects as go
import pyratbay.spectrum as ps
import pyratbay.tools as pt


def plotly_filters():
    pass


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
            bin_wl = ps.constant_resolution_spectrum(wl_min, wl_max, resolution)
            bin_flux = ps.bin_spectrum(bin_wl, wl, flux, ignore_gaps=True)
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
            bin_wl = ps.constant_resolution_spectrum(wl_min, wl_max, resolution)
            depth = ps.bin_spectrum(bin_wl, wl, depth) / pt.u(units)
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


