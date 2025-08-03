# Copyright (c) 2025 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    'response_boundaries',
    'plotly_filters',
    'plotly_sed_spectra',
    'plotly_depth_spectra',
    'plotly_tso_spectra',
    'plotly_tso_fluxes',
    'plotly_tso_snr',
    'plotly_tso_2d',
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

depth_units_label = {
    'none': '',
    'percent': ' (%)',
    'ppm': ' (ppm)',
}


def response_boundaries(wl, response, threshold=0.001):
    """
    Find the wavelength boundaries where a response function
    is greater than the required threshold.

    Parameters
    ----------
    wl: 1D float iterable
        Wavelength array where a response function is sampled
    response: 1D float iterable
        Response function.
    threshold: float
        Minimum response value for flagging.

    Returns
    -------
    bounds: list of float pairs
        A list of the wavelength boundaries for each contiguous
        segment with non-zero response.

    Examples
    --------
    >>> import gen_tso.plotly_io as plots
    >>>
    >>> nwave = 21
    >>> wl = np.linspace(0.0, 1.0, nwave)
    >>> response = np.zeros(nwave)
    >>> response[2:6] = response[10:12] = 1.0
    >>> bounds = plots.response_boundaries(wl, response, threshold=0.5)
    >>> print(bounds)
    [(0.1, 0.25), (0.5, 0.55)]
    """
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
    subarray_name: String
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
    >>> passbands = jwst.get_throughputs('spectroscopy')
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
        'sw_tsgrism': 'sub41s1_2-spectra',
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

        for mode in reversed(modes):
            if mode == mode_name and subarray_name in passbands[inst][mode]:
                subarray = subarray_name
                filters = list(passbands[inst][mode][subarray].keys())
            elif show_all and mode in show_all_subarrays:
                subarray = show_all_subarrays[mode]
                filters = show_all_filters[mode]
            else:
                filters = []

            for filter in filters:
                band = passbands[inst][mode][subarray][filter]
                all_filters[inst][filter] = band
                if 'order2' in band:
                    all_filters[inst]['order2'] = band['order2']
        nfilters += len(all_filters[inst])


    visible = [None for _ in range(nfilters)]
    if mode_name == 'bots':
        for i,filter in enumerate(all_filters[inst_name].keys()):
            hide = ('h' in filter_name) is not ('h' in filter)
            if hide and 'prism' not in filter:
                visible[i] = 'legendonly'
    elif mode_name in ['lw_tsgrism', 'sw_tsgrism']:
        nircam_visibles = [
            'f070w', 'f090w', 'f150w2', 'f200w',
            'f322w2', 'f444w',
        ]
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
        wl_range=None, units='mJy', wl_scale='linear', resolution=250.0,
        throughput=None,
    ):
    """
    Make a plotly figure of stellar SED spectra.
    """
    nmodels = len(sed_models)
    fig = go.Figure(
        layout={'colorway': COLOR_SEQUENCE},
    )

    # Shaded area for filter:
    if throughput is not None:
        if 'order2' in throughput:
            wl = throughput['order2']['wl']
            response = throughput['order2']['response']
            band_bounds = response_boundaries(wl, response, threshold=0.03)
            for bound in band_bounds:
                fig.add_vrect(
                    fillcolor="orchid", opacity=0.4,
                    x0=bound[0], x1=bound[1],
                    layer="below", line_width=0,
                )
        band_bounds = response_boundaries(throughput['wl'], throughput['response'])
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
            bin_flux = bin_spectrum(bin_wl, wl, flux, gaps='interpolate')
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
        wl_range = [
            None if wave is None else np.log10(wave)
            for wave in wl_range
        ]
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
        wl_range=None, units='percent', wl_scale='linear', resolution=250.0,
        depth_range=None,
        obs_geometry='transit',
        throughput=None,
    ):
    """
    Make a plotly figure of transit/eclipse depth spectra.
    """
    if depth_range is None:
        depth_range =[None, None]

    nmodels = len(depth_models)
    fig = go.Figure(
        layout={'colorway':COLOR_SEQUENCE},
    )
    # Shaded area for filter:
    if throughput is not None:
        if 'order2' in throughput:
            wl = throughput['order2']['wl']
            response = throughput['order2']['response']
            band_bounds = response_boundaries(wl, response, threshold=0.03)
            for bound in band_bounds:
                fig.add_vrect(
                    fillcolor="orchid", opacity=0.4,
                    x0=bound[0], x1=bound[1],
                    layer="below", line_width=0,
                )
        bounds = response_boundaries(throughput['wl'], throughput['response'])
        for bound in bounds:
            fig.add_vrect(
                fillcolor="#069af3", opacity=0.4,
                x0=bound[0], x1=bound[1],
                layer="below", line_width=0,
            )

    ymax = 0.0
    ymin = np.inf
    for j,model in enumerate(depth_models):
        wl = model['wl']
        depth = model['depth']
        if resolution > 0:
            wl_min = np.amin(wl)
            wl_max = np.amax(wl)
            bin_wl = constant_resolution_spectrum(wl_min, wl_max, resolution)
            depth = bin_spectrum(bin_wl, wl, depth, gaps='interpolate')/u(units)
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
        ymax = np.amax([ymax, np.amax(depth)])
        ymin = np.amin([ymin, np.amin(depth)])


    fig.update_traces(
        hovertemplate=
            'wl = %{x:.2f}<br>'+
            'depth = %{y:.3f}'
    )
    if depth_range[0] is None or depth_range[1] is None:
        dy = 0.05 * (ymax-ymin)
    if depth_range[0] is None:
        depth_range[0] = ymin - dy
    if depth_range[1] is None:
        depth_range[1] = ymax + dy

    ylabel = f'{obs_geometry} depth{depth_units_label[units]}'
    fig.update_yaxes(
        title_text=ylabel,
        title_standoff=0,
        range=depth_range,
    )

    if wl_scale == 'log' and wl_range is not None:
        wl_range = [
            None if wave is None else np.log10(wave)
            for wave in wl_range
        ]
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
        tso_list, sim_depths=None, resolution=250.0, n_obs=1,
        model_label='model', instrument_label=None,
        units='percent', wl_range=None, wl_scale='linear',
        depth_range=None,
        obs_geometry='transit',
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
        if sim_depths is None:
            bin_wl, bin_spec, bin_err, wl_err = jwst.simulate_tso(
               tso, n_obs=n_obs, resolution=resolution, noiseless=False,
            )
        else:
            bin_wl = sim_depths[i]['wl']
            bin_spec = sim_depths[i]['depth']
            bin_err = sim_depths[i]['uncert']
            wl_err = sim_depths[i]['wl_widths']

        mode = tso['report_in']['input']['configuration']['instrument']['mode']
        if mode in jwst._photo_modes:
            input_wl, input_depth = tso['input_depth']
            wl_min = np.amin(input_wl)
            wl_max = np.amax(input_wl)
            wl = constant_resolution_spectrum(wl_min, wl_max, resolution)
            spec = bin_spectrum(wl, input_wl, input_depth, gaps='interpolate')
        else:
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

        if mode in jwst._photo_modes:
            error_x = dict(
                type='data', symmetric=False, visible=True,
                array=wl_err[:,1], arrayminus=wl_err[:,0],
            )
        else:
            error_x = None

        show_legend = instrument_label[i] not in legends
        fig.add_trace(go.Scatter(
            x=bin_wl,
            y=bin_spec/u(units),
            error_y=dict(type='data', array=bin_err/u(units), visible=True),
            error_x=error_x,
            mode='markers',
            name=instrument_label[i],
            legendgroup=instrument_label[i],
            showlegend=show_legend,
            marker=dict(color=obs_col, size=5),
        ))
        legends.append(instrument_label[i])
        ymax = np.amax([ymax, np.amax(spec)])
        ymin = np.amin([ymin, np.amin(spec)])

    # Saturation (take report with highest e-/sec)
    report = tso['report_out'] if obs_geometry=='transit' else tso['report_in']
    wl, partial = report['1d']['n_partial_saturated']
    wl, full = report['1d']['n_full_saturated']
    partial_saturation = response_boundaries(wl, partial, threshold=0)
    for j,bound in enumerate(partial_saturation):
        fig.add_vrect(
            x0=bound[0], x1=bound[1],
            fillcolor="red", opacity=0.3,
            layer="below", line_width=0,
            legendgrouptitle_text="Saturation",
            legendgroup='saturation',
            name='partial',
            showlegend=(j==0),
        )
    full_saturation = response_boundaries(wl, full, threshold=0)
    for j,bound in enumerate(full_saturation):
        fig.add_vrect(
            x0=bound[0], x1=bound[1],
            fillcolor="black", opacity=0.75,
            layer="below", line_width=0,
            legendgroup='saturation',
            name='full',
            showlegend=(j==0),
        )

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
    ylabel = f'{obs_geometry} depth{depth_units_label[units]}'
    fig.update_yaxes(
        title_text=ylabel,
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


def plotly_tso_fluxes(
        tso_list,
        wl_range=None, wl_scale='linear',
        obs_geometry='transit',
    ):
    """
    Plot 1D source and background flux rates
    """
    if not isinstance(tso_list, list):
        tso_list = [tso_list]

    colors = [
        px.colors.sample_colorscale('Viridis', 0.15)[0],
        px.colors.sample_colorscale('Viridis', 0.7)[0],
        px.colors.sample_colorscale('Viridis', 0.9)[0],
    ]
    legends = ['in-transit', 'out-transit', 'background']

    fig = go.Figure()
    for j,tso in enumerate(tso_list):
        wl = tso['report_in']['1d']['extracted_flux'][0]
        fluxes = [
            tso['report_in']['1d']['extracted_flux'][1],
            tso['report_out']['1d']['extracted_flux'][1],
            tso['report_out']['1d']['extracted_bg_only'][1],
        ]
        show_legend = j == 0
        for i in range(len(fluxes)):
            fig.add_trace(go.Scatter(
                x=wl,
                y=fluxes[i],
                mode='lines',
                line=dict(color=colors[i], width=1.75),
                name=legends[i],
                legendgroup=legends[i],
                showlegend=show_legend,
            ))

    # Saturation (take report with highest e-/sec)
    report = tso['report_out'] if obs_geometry=='transit' else tso['report_in']
    wl, partial = report['1d']['n_partial_saturated']
    wl, full = report['1d']['n_full_saturated']
    partial_saturation = response_boundaries(wl, partial, threshold=0)
    for j,bound in enumerate(partial_saturation):
        fig.add_vrect(
            x0=bound[0], x1=bound[1],
            fillcolor="red", opacity=0.3,
            layer="below", line_width=0,
            legendgrouptitle_text="Saturation",
            name='partial',
            legendgroup='saturation',
            showlegend=(j==0),
        )
    full_saturation = response_boundaries(wl, full, threshold=0)
    for j,bound in enumerate(full_saturation):
        fig.add_vrect(
            x0=bound[0], x1=bound[1],
            fillcolor="black", opacity=0.75,
            layer="below", line_width=0,
            name='full',
            legendgroup='saturation',
            showlegend=(j==0),
        )

    fig.update_traces(
        hovertemplate='wl = %{x:.2f}<br>' + 'flux = %{y:.3f}'
    )
    fig.update_yaxes(
        title_text='flux rate (e-/sec)',
        title_standoff=0,
    )
    if wl_scale == 'log' and wl_range is not None:
        wl_range = [np.log10(wave) for wave in wl_range]
    fig.update_xaxes(
        title_text='wavelength (um)',
        title_standoff=0,
        range=wl_range,
        type=wl_scale,
    )
    fig.update_layout(showlegend=True)
    return fig


def plotly_tso_snr(
        tso_list,
        wl_range=None, wl_scale='linear',
        obs_geometry='transit',
    ):
    """
    Plot 1D signal-to-noise ratios for in- and out-of-transit simulations
    """
    if not isinstance(tso_list, list):
        tso_list = [tso_list]

    colors = [
        px.colors.sample_colorscale('Viridis', 0.15)[0],
        px.colors.sample_colorscale('Viridis', 0.7)[0],
    ]
    legends = ['in-transit', 'out-transit']

    fig = go.Figure()
    for j,tso in enumerate(tso_list):
        wl = tso['report_in']['1d']['sn'][0]
        snr = [
            tso['report_in']['1d']['sn'][1],
            tso['report_out']['1d']['sn'][1],
        ]
        show_legend = j == 0
        for i in range(len(snr)):
            fig.add_trace(go.Scatter(
                x=wl,
                y=snr[i],
                mode='lines',
                line=dict(color=colors[i], width=1.75),
                name=legends[i],
                legendgroup=legends[i],
                showlegend=show_legend,
            ))

    # Saturation (take report with highest e-/sec)
    report = tso['report_out'] if obs_geometry=='transit' else tso['report_in']
    wl, partial = report['1d']['n_partial_saturated']
    wl, full = report['1d']['n_full_saturated']
    partial_saturation = response_boundaries(wl, partial, threshold=0)
    for j,bound in enumerate(partial_saturation):
        fig.add_vrect(
            x0=bound[0], x1=bound[1],
            fillcolor="red", opacity=0.3,
            layer="below", line_width=0,
            legendgrouptitle_text="Saturation",
            name='partial',
            legendgroup='saturation',
            showlegend=(j==0),
        )
    full_saturation = response_boundaries(wl, full, threshold=0)
    for j,bound in enumerate(full_saturation):
        fig.add_vrect(
            x0=bound[0], x1=bound[1],
            fillcolor="black", opacity=0.75,
            layer="below", line_width=0,
            name='full',
            legendgroup='saturation',
            showlegend=(j==0),
        )

    fig.update_traces(
        hovertemplate='wl = %{x:.2f}<br>' + 'flux = %{y:.3f}'
    )
    fig.update_yaxes(
        title_text='signal-to-noise ratio',
        title_standoff=0,
    )
    if wl_scale == 'log' and wl_range is not None:
        wl_range = [np.log10(wave) for wave in wl_range]
    fig.update_xaxes(
        title_text='wavelength (um)',
        title_standoff=0,
        range=wl_range,
        type=wl_scale,
    )
    fig.update_layout(showlegend=True)
    return fig


def plotly_tso_2d(tso, heatmap_name):
    """
    Make 2D pandeia plots as in the ETC.

    Parameters
    ----------
    tso: Dictionary
        A TSO output.
    heatmap_name: String
        Heat map property to plot, select from:
        'snr', 'detector', 'saturation', or 'ngroups_map'
    """
    # TBD: Think how to multipanel MRS heatmaps
    if isinstance(tso, list):
        report = tso[0]['report_out']
    else:
        report = tso['report_out']
    inst = report['input']['configuration']['instrument']['instrument']
    mode = report['input']['configuration']['instrument']['mode']
    heatmap = report['2d'][heatmap_name]
    # Shiny can't handle nans
    is_nan = ~np.isfinite(heatmap)
    heatmap[is_nan] = 0.0


    x_min = report['transform']['x_min']
    x_max = report['transform']['x_max']
    nx = report['transform']['x_size']
    x = np.linspace(x_min, x_max, nx)

    y_min = report['transform']['y_min']
    y_max = report['transform']['y_max']
    ny = report['transform']['y_size']
    y = np.linspace(y_min, y_max, ny)

    if inst == 'niriss':
        nx = report['transform']['x_size']
        ny = report['transform']['y_size']
        xlabel = ylabel = 'pixels'
        heatmap = np.flipud(heatmap)
    elif mode == 'mrs_ts':
        xlabel = 'arcsec'
        ylabel = 'arcsec'
    elif mode == 'lrsslit':
        y_min = report['transform']['wave_det_min']
        y_max = report['transform']['wave_det_max']
        y = np.linspace(y_min, y_max, ny)
        xlabel = 'dispersion (arcsec)'
        ylabel = 'wavelength (microns)'
    elif mode == 'lrsslitless':
        y = np.flip(y)
        xlabel = 'dispersion (arcsec)'
        ylabel = 'wavelength (arcsec)'
    else:
        x = report['1d']['sn'][0]
        xlabel = 'wavelength (um)'
        ylabel = 'dispersion (arcsec)'

    # Strategy:
    apertures = []
    if inst != 'niriss':
        aperture = report['input']['strategy']['aperture_size']
        annulus = report['input']['strategy']['sky_annulus']
        apertures = [
            np.tile( 0.5*aperture, 2),
            np.tile(-0.5*aperture, 2),
            np.tile( annulus[0], 2),
            np.tile(-annulus[0], 2),
            np.tile( annulus[1], 2),
            np.tile(-annulus[1], 2),
        ]

    fig = go.Figure(
        data=go.Heatmap(z=heatmap, x=x, y=y, showscale=False),
    )
    if mode == 'mrs_ts':
        t = np.linspace(0.0, 2.0*np.pi, 100)
        for i, aper in enumerate(apertures):
            if i%2 == 1:
                continue
            xx = aper[0]*np.sin(t)
            yy = aper[0]*np.cos(t)
            if i < 2:
                name = 'aperture'
                color= 'gold'
                dash = None
            else:
                name = 'background'
                color= 'limegreen'
                dash = 'dash'
            showlegend = i in [0,2]
            fig.add_trace(go.Scatter(
                x=xx,
                y=yy,
                mode='lines',
                line=dict(color=color, width=2.0, dash=dash),
                name=name, legendgroup=name,
                showlegend=showlegend,
            ))
        fig.update_layout(
            yaxis=dict(scaleanchor='x', scaleratio=1),
        )

    elif inst != 'niriss':
        for i, aper in enumerate(apertures):
            if i < 2:
                name = 'aperture'
                color= 'gold'
                dash = None
            else:
                name = 'background'
                color= 'limegreen'
                dash = 'dash'
            showlegend = i in [0,2]
            if mode in ['lrsslit', 'lrsslitless']:
                xx = aper
                yy = [np.amin(y), np.amax(y)]
            else:
                xx = [np.amin(x), np.amax(x)]
                yy = aper
            fig.add_trace(go.Scatter(
                x=xx,
                y=yy,
                mode='lines',
                line=dict(color=color, width=2.0, dash=dash),
                name=name, legendgroup=name,
                showlegend=showlegend,
            ))

    range = None
    if mode == 'lrsslit':
        range = [np.amax(y), np.amin(y)]

    fig.update_yaxes(
        title_text=ylabel,
        title_standoff=0,
        range=range,
    )
    fig.update_xaxes(
        title_text=xlabel,
        title_standoff=0,
    )

    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5
    ))

    return fig


