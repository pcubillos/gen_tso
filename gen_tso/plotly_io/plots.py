# Copyright (c) 2024 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    'plotly_filters',
    'plotly_sed_spectra',
    'plotly_depth_spectra',
    'plotly_tso_spectra',
]

from itertools import groupby
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pyratbay.spectrum import bin_spectrum, constant_resolution_spectrum
from pyratbay.tools import u

from .. import pandeia_io as jwst


COLOR_SEQUENCE = [
    'Royalblue', # blue
    '#15b01a',  # green
    '#000075',  # navy
    '#f032e6',  # magenta
    '#42d4f4',  # cyan
    '#888888',  # grey
    'Red',      # red
    '#9A6324',  # brown
    '#800000',  # maroon
    '#000000',  #  black
    '#469990',  # teal
    '#911eb4',  # purple
    '#808000',  # olive
    'Green',  # green
]


def band_boundaries(band, threshold=0.001):
    """
    Find the wavelength boundaries of a passband where the response
    is greater than the required threshold.

    Parameters
    ----------
    band: dict
        A dictionary of the band response and wavelength.
    threshold: float
        Minimum band response for highlight.

    Returns
    -------
    bounds: list of float pairs
        A list of the wavelength boundaries for each contiguous
        segment with non-zero response.
    """
    wl = band['wl']
    response = band['response']
    bounds = []
    # Contigous ranges where response > threshold:
    for group, indices in groupby(range(len(wl)), lambda x: response[x]>threshold):
        if group:
            indices = list(indices)
            imin = indices[0]
            imax = indices[-1]
            bounds.append((wl[imin], wl[imax]))
    return bounds


def plotly_filters(
        passbands, inst_name, mode_name, subarray_name, filter_name,
        show_all=False,
    ):
    """
    Make a plotly figure of the passband filters.

    Parameters
    ----------
    passbands: Dictionary
        Dictionary of passband arrays. See example below.
    inst_name: String
        Instrument to plot/highlight
    mode_name: String
        Mode to plot/highlight
    subarray: String
        Subarray to plot the passband.
    filter_name: String
        Filter to plot.
    show_all: Bool
        If True, plot a selection from all instruments for better comparison.

    Examples
    --------
    >>> import gen_tso.pandeia_io as jwst
    >>> import gen_tso.plotly_io as plots
    >>>
    >>> passbands = jwst.filter_throughputs()['spectroscopy']
    >>> fig = plots.plotly_filters(
    >>>     passbands, 'nircam', 'lw_tsgrism', 'subgrism64', 'f444w',
    >>>     show_all=True,
    >>> )
    >>> fig.show()
    """
    instruments = [inst_name]
    if show_all:
        instruments += [
            inst for inst in passbands.keys()
            if inst not in instruments
        ]

    show_all_subarrays = {
        'lrsslitless': 'slitlessprism',
        'sw_tsgrism': 'sub40stripe1_dhs',
        'lw_tsgrism': 'subgrism64',
        'bots': 'sub2048',
        'soss': 'substrip256',
    }
    show_all_filters = {
        'lrsslitless': ['None'],
        'lw_tsgrism': ['f322w2', 'f444w'],
        'sw_tsgrism': ['f150w2'],
        'bots': ['g140m/f100lp', 'g235m/f170lp', 'g395m/f290lp', 'prism/clear'],
        'soss': ['clear'],
    }

    # Parse filters to plot
    all_filters = {}
    nfilters = 0
    for inst in instruments:
        modes = list(passbands[inst])
        all_filters[inst] = {}

        for mode in modes:
            if mode == mode_name:
                filters = list(passbands[inst][mode][subarray_name].keys())
            elif show_all and mode in show_all_subarrays:
                subarray_name = show_all_subarrays[mode]
                filters = show_all_filters[mode]
            else:
                filters = []

            for filter in filters:
                band = passbands[inst][mode][subarray_name][filter]
                all_filters[inst][filter] = band
                if 'order2' in band:
                    all_filters[inst]['order2'] = band['order2']
        nfilters += len(all_filters[inst])


    visible = [None for _ in range(nfilters)]
    if mode == 'bots':
        for i,filter in enumerate(all_filters[inst_name].keys()):
            hide = ('h' in filter_name) is not ('h' in filter)
            if hide and 'prism' not in filter:
                visible[i] = 'legendonly'
    elif mode in ['lw_tsgrism', 'sw_tsgrism']:
        nircam_visibles = ['f070w', 'f090w', 'f150w2', 'f322w2', 'f444w']
        for i,filter in enumerate(all_filters[inst_name].keys()):
            if filter != filter_name and filter not in nircam_visibles:
                visible[i] = 'legendonly'

    ncols = len(all_filters[inst_name])
    primary_colors = px.colors.sample_colorscale(
        'Viridis', np.linspace(0, 0.8, ncols),
    )
    secondary_colors = [
        px.colors.sequential.gray[1],
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
                color = next(sel_cols)
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
    wl_range = [np.log10(0.6), np.log10(28.0)]
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
        wl_range=[0.5,28], units='mJy', wl_scale='linear', resolution=250.0,
        throughput=None,
    ):
    """
    Make a plotly figure of stellar SED spectra.
    """
    nmodels = len(sed_models)
    fig = go.Figure(
        #layout={'colorway':px.colors.qualitative.Alphabet},
        layout={'colorway':COLOR_SEQUENCE},
    )

    # Shaded area for filter:
    if throughput is not None:
        if 'order2' in throughput:
            band_bounds = band_boundaries(throughput['order2'], threshold=0.03)
            for bound in band_bounds:
                fig.add_vrect(
                    fillcolor="LightSalmon", opacity=0.4,
                    x0=bound[0], x1=bound[1],
                    layer="below", line_width=0,
                )
        band_bounds = band_boundaries(throughput)
        for bound in band_bounds:
            fig.add_vrect(
                fillcolor="#069af3", opacity=0.4,
                x0=bound[0], x1=bound[1],
                layer="below", line_width=0,
            )

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
            line=linedict,
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
        throughput=None,
    ):
    """
    Make a plotly figure of transit/eclipse depth spectra.
    """
    nmodels = len(depth_models)
    fig = go.Figure(
        layout={'colorway':COLOR_SEQUENCE},
    )
    # Shaded area for filter:
    if throughput is not None:
        if 'order2' in throughput:
            band_bounds = band_boundaries(throughput['order2'], threshold=0.03)
            for bound in band_bounds:
                fig.add_vrect(
                    fillcolor="LightSalmon", opacity=0.4,
                    x0=bound[0], x1=bound[1],
                    layer="below", line_width=0,
                )
        band_bounds = band_boundaries(throughput)
        for bound in band_bounds:
            fig.add_vrect(
                fillcolor="#069af3", opacity=0.4,
                x0=bound[0], x1=bound[1],
                layer="below", line_width=0,
            )

    for j,model in enumerate(depth_models):
        wl = model['wl']
        depth = model['depth']
        if resolution > 0:
            wl_min = np.amin(wl)
            wl_max = np.amax(wl)
            bin_wl = constant_resolution_spectrum(wl_min, wl_max, resolution)
            depth = bin_spectrum(bin_wl, wl, depth, ignore_gaps=True) / u(units)
            mask = np.isfinite(depth)
            wl = bin_wl[mask]
            depth = depth[mask]

        if labels[j] == highlight_model:
            linedict = dict(color='Gold', width=2.0)
            rank = j + nmodels
        else:
            linedict = dict(width=1.25)
            rank = j
        fig.add_trace(go.Scatter(
            x=wl,
            y=depth,
            mode='lines',
            name=labels[j],
            line=linedict,
            legendrank=rank,
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
        tso_list, resolution, n_obs, model_label, instrument_label,
        bin_widths=None,
        units='percent', wl_range=None, wl_scale='linear',
        depth_range=None,
        obs_geometry='Transit',
    ):
    """
    Make a plotly figure of transit/eclipse depth TSO spectra.
    """
    if not isinstance(tso_list, list):
        tso_list = [tso_list]

    if isinstance(instrument_label, str):
        instrument_label = [instrument_label for tso in tso_list]

    fig = go.Figure()
    obs_col = px.colors.sample_colorscale('Viridis', 0.2)[0]
    model_col = px.colors.sample_colorscale('Viridis', 0.75)[0]

    ymax = 0.0
    ymin = np.inf
    legends = []
    for i,tso in enumerate(tso_list):
        bin_wl, bin_spec, bin_err, widths = jwst.simulate_tso(
           tso, n_obs=n_obs, resolution=resolution, noiseless=False,
        )
        wl = tso['wl']
        spec = tso['depth_spectrum']

        show_legend = model_label not in legends
        fig.add_trace(go.Scatter(
            x=wl,
            y=spec/u(units),
            mode='lines',
            name=model_label,
            legendgroup=model_label,
            showlegend=show_legend,
            line=dict(color=model_col, width=1.5),
        ))
        legends.append(model_label)

        show_legend = instrument_label[i] not in legends
        fig.add_trace(go.Scatter(
            x=bin_wl,
            y=bin_spec/u(units),
            error_y=dict(type='data', array=bin_err/u(units), visible=True),
            mode='markers',
            name=instrument_label[i],
            legendgroup=instrument_label[i],
            showlegend=show_legend,
            marker=dict(color=obs_col, size=5),
        ))
        legends.append(instrument_label[i])
        ymax = np.amax([ymax, np.amax(spec)])
        ymin = np.amin([ymin, np.amin(spec)])

    fig.update_traces(
        hovertemplate=
            'wl = %{x:.2f}<br>'+
            'depth = %{y:.3f}'
    )
    if depth_range is None:
        ymax = ymax/u(units)
        ymin = ymin/u(units)
        dy = 0.1 * (ymax-ymin)
        depth_range = [ymin-dy, ymax+dy]
    title = f'{obs_geometry} depth ({units})'
    title = title.replace('percent','%').replace(' (none)', '')
    fig.update_yaxes(
        title_text=title,
        title_standoff=0,
        range=depth_range,
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


