# Copyright (c) 2024 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    'get_configs',
    'filter_throughputs',
    'generate_all_instruments',
    'detector_label',
]

import pickle
from pandeia.engine.calc_utils import get_instrument_config
from gen_tso.utils import ROOT


inst_names = {
    'miri': 'MIRI',
    'nircam': 'NIRCam',
    'niriss': 'NIRISS',
    'nirspec': 'NIRSpec',
}
spec_modes = {
    'miri': 'lrsslitless',
    'nircam': 'ssgrism',
    'niriss': 'soss',
    'nirspec': 'bots',
}
acq_modes = {
    'miri': 'target_acq',
    'nircam': 'target_acq',
    'niriss': 'target_acq',
    'nirspec': 'target_acq',
}
all_modes = {
    'spectroscopy': spec_modes,
    'acquisition': acq_modes,
}


def get_constrained_values(inst_config, aper, inst_property, mode):
    """
    Extract instrument properties that might be constrained or not.

    Parameters
    ----------
    inst_config: Dict
        Instrument config dict.
    aper: String
        Instrument's aperture.
    inst_property: String
        The property to extract. Select from: dispersers, filters,
        subarrays, or readout_patterns.
    mode: String
        Observing mode.
    """
    constraints = inst_config['config_constraints']['apertures']
    is_constrained = (
        aper in constraints and
        inst_property in constraints[aper] and
        mode in constraints[aper][inst_property]
    )

    if is_constrained:
        values = constraints[aper][inst_property][mode]
    elif inst_property in inst_config['mode_config'][mode]:
        values = inst_config['mode_config'][mode][inst_property]
    else:
        values = inst_config[inst_property]
    return values


def str_or_dict(info_dict, selected, mode):
    """
    Extract the label of a requested property.

    Parameters
    ----------
    info_dict: Dict
        Instrument's disperser, readout, filter, subarray or slit config
    selected: String
        Selected property.
    mode: String
        Observing mode.
    """
    if 'display_string' not in info_dict[selected]:
        return selected.upper()
    if isinstance(info_dict[selected]['display_string'], str):
        return info_dict[selected]['display_string']
    if mode in info_dict[selected]['display_string']:
        return info_dict[selected]['display_string'][mode]
    return info_dict[selected]['display_string']['default']


def get_configs(instrument=None, obs_type=None):
    """
    Collect the information from the available observing modes
    (Names, modes, dispersers, readout patterns, subarrays, slits).

    Parameters
    ----------
    instrument: String or List
        The JWST instrument(s) to collect. Select one or more
        from miri, nircam, nirspec, or niriss.
        If None collect all instruments.
    obs_type: String
        Type of observation: spectroscopy or acquisition

    Returns
    -------
    A list of dictionaries containing the observing modes info.

    Examples
    --------
    >>> from gen_tso.pandeia_io import get_configs
    >>> insts = get_configs(instrument='niriss', obs_type='spectroscopy')
    >>> insts = get_configs(obs_type='spectroscopy')
    >>> insts = get_configs(obs_type='acquisition')
    """
    ta_apertures = {
        'miri': ['imager'],
        'nircam': ['lw'],
        'nirspec': ['s1600a1'],
        'niriss': ['imager', 'nrm'],
    }
    if instrument is None:
        instrument = 'miri nircam nirspec niriss'.split()
    elif isinstance(instrument, str):
        instrument = [instrument]

    telescope = 'jwst'
    outputs = []
    for inst in instrument:
        inst_config = get_instrument_config(telescope, inst)
        # inst_config['modes']   # To see all modes
        disperser_names = inst_config['disperser_config']
        filter_names = inst_config['filter_config']
        readout_names = inst_config['readout_pattern_config']
        subarray_names = inst_config['subarray_config']['default']
        if 'slit_config' in inst_config:
            slit_names = inst_config['slit_config']

        if obs_type == 'spectroscopy':
            mode = spec_modes[inst]
        elif obs_type == 'acquisition':
            mode = acq_modes[inst]
        #print(f'\n{inst}:  {mode}')
        #print(inst_config['mode_config'][mode]['apertures'])
        apertures = inst_config['mode_config'][mode]['apertures']

        for aper in apertures:
            inst_dict = {}
            inst_dict['instrument'] = inst_names[inst]
            inst_dict['obs_type'] = obs_type
            inst_dict['aperture'] = aper
            if obs_type == 'acquisition':
                if aper not in ta_apertures[inst]:
                    continue
                mode_name = inst_config['aperture_config'][aper]['display_string'][mode]
            else:
                mode_name = inst_config['mode_config'][mode]['display_string']
            #print(f'Aperture: {repr(aper)}\nMode: {mode_name}')
            inst_dict['mode'] = mode
            inst_dict['mode_label'] = mode_name

            dispersers = get_constrained_values(
                inst_config, aper, 'dispersers', mode,
            )
            #print('Dispersers:  ', dispersers)
            inst_dict['dispersers'] = {}
            for disperser in dispersers:
                name = str_or_dict(disperser_names, disperser, mode)
                inst_dict['dispersers'][disperser] = name
                #print(f"   {disperser}: {name}")
            if mode == 'bots':
                constraints = inst_config['config_constraints']['dispersers']
                inst_dict['gratings_constraints'] = constraints

            filters = get_constrained_values(inst_config, aper, 'filters', mode)
            inst_dict['filters'] = {}
            #print('Filters:  ', filters)
            for filter in filters:
                name = str_or_dict(filter_names, filter, mode)
                inst_dict['filters'][filter] = name
                #print(f"   {filter}: {name}")

            readouts = get_constrained_values(
                inst_config, aper, 'readout_patterns', mode,
            )
            inst_dict['readouts'] = {}
            #print('Readouts:', readouts)
            for readout in readouts:
                name = str_or_dict(readout_names, readout, mode)
                inst_dict['readouts'][readout] = name
                #print(f"   {readout}: {name}")

            subarrays = get_constrained_values(
                inst_config, aper, 'subarrays', mode,
            )
            inst_dict['subarrays'] = {}
            #print('Subarrays:', subarrays)
            for subarray in subarrays:
                name = str_or_dict(subarray_names, subarray, mode)
                inst_dict['subarrays'][subarray] = name
                #print(f"   {subarray}: {name}")
            if inst_dict['instrument']=='MIRI' and mode=='target_acq':
                constraints = inst_config['mode_config'][mode]['enum_ngroups']
                inst_dict['subarray_constraints'] = {}
                for subarray in subarrays:
                    key = subarray if subarray in constraints else 'default'
                    groups = constraints[key]
                    inst_dict['subarray_constraints'][subarray] = groups

            slits = get_constrained_values(
                inst_config, aper, 'slits', mode,
            )
            inst_dict['slits'] = {}
            for slit in slits:
                name = str_or_dict(slit_names, slit, mode)
                inst_dict['slits'][slit] = name

            if obs_type == 'acquisition':
                groups = inst_config['mode_config'][mode]['enum_ngroups']
                if not isinstance(groups, list):
                    groups = groups['default']
                #print(inst_config['mode_config'][mode]['enum_nexps'])
                #print(inst_config['mode_config'][mode]['enum_nints'])
            outputs.append(inst_dict)
    return outputs


def print_configs(instrument, mode, output):
    telescope = 'jwst'
    inst_config = get_instrument_config(telescope, instrument)

    if mode is None:
        return(inst_config['modes'])
    config = inst_config['mode_config'][mode]

    subarrays = config['subarrays']
    screen_output = f'subarrays: {subarrays}\n'

    if instrument == 'niriss':
        readouts = inst_config['readout_patterns']
    else:
        readouts = config['readout_patterns']
    screen_output += f'readout patterns: {readouts}\n'

    if instrument == 'nirspec':
        gratings_dict = inst_config['config_constraints']['dispersers']
        gratings = filters = dispersers = []
        for grating, filter_list in gratings_dict.items():
            for filter in filter_list['filters']:
                gratings.append(f'{grating}/{filter}')
        screen_output += f'grating/filter pairs: {gratings}'
    else:
        filters = config['filters']
        dispersers = [disperser for disperser in config['dispersers']]
        screen_output += f'dispersers: {dispersers}\n'
        screen_output += f'filters: {filters}'

    if output is None:
        print(screen_output)
    elif output == 'readouts':
        return readouts
    elif output == 'subarrays':
        return subarrays
    elif output == 'filters':
        return filters
    elif output == 'dispersers':
        return dispersers
    elif instrument=='nirspec' and output=='gratings':
        return gratings
    else:
        raise ValueError(f"Invalid config output: '{output}'")


class Detector:
    def __init__(
            self, mode, label, instrument, obs_type,
            disperser_label, dispersers, filter_label, filters,
            subarrays, readouts, default_indices=None,
        ):
        self.mode = mode
        self.mode_label = label
        self.instrument = instrument
        self.obs_type = obs_type
        self.disperser_label = disperser_label
        self.dispersers = dispersers
        self.filter_label = filter_label
        self.filters = filters
        self.subarrays = subarrays
        self.readouts = readouts

        telescope = 'jwst'
        self.ins_config = get_instrument_config(telescope, instrument.lower())

        if default_indices is None:
            idx_d = idx_f = idx_s = idx_r = 0
        else:
            idx_d, idx_f, idx_s, idx_r = default_indices
        self.default_disperser = list(dispersers)[idx_d]
        self.default_filter = list(filters)[idx_f]
        self.default_subarray = list(subarrays)[idx_s]
        self.default_readout = list(readouts)[idx_r]


def filter_throughputs():
    """
    Collect the throughput response curves for each instrument configuration
    """
    obs_types = [
        'spectroscopy',
        'acquisition',
    ]
    throughputs = {}
    for obs_type in obs_types:
        throughputs[obs_type] = {}
        for inst_name, mode in all_modes[obs_type].items():
            throughputs[obs_type][inst_name] = {}

            t_file = f'{ROOT}data/throughputs_{inst_name}_{mode}.pickle'
            with open(t_file, 'rb') as handle:
                data = pickle.load(handle)

            for subarray in list(data.keys()):
                throughputs[obs_type][inst_name][subarray] = data[subarray]

    return throughputs


def generate_all_instruments():
    """
    A list of Detector() objects to keep the instrument observing mode
    configurations.

    TBD
    ---
    Spectroscopy
        'mrs_ts': 'MRS Time Series',
    Imaging
        'imaging_ts': 'Imaging Time Series',
        'sw_ts': 'SW Time Series',
        'lw_ts': 'LW Time Series',
    Examples
    --------
    >>> from gen_tso.pandeia_io import get_configs
    >>> insts = get_configs(obs_type='spectroscopy')
    >>> insts = get_configs(obs_type='acquisition')
    """
    detectors = []
    # Spectroscopic observing modes
    spec_insts = get_configs(obs_type='spectroscopy')
    for inst in spec_insts:
        mode = inst['mode']
        dispersers = inst['dispersers']
        filters = inst['filters']
        subarrays = inst['subarrays']
        readouts = inst['readouts']

        if mode == 'lrsslitless':
            disperser_label = 'Disperser'
            filter_label = ''
            filters = {'': ''}
            default_indices = 0, 0, 0, 0
        if mode == 'ssgrism':
            disperser_label = 'Grism'
            filter_label = 'Filter'
            default_indices = 0, 3, 3, 0
        if mode == 'bots':
            disperser_label = 'Slit'
            filter_label = 'Grating/Filter'
            inst['mode_label'] = 'Bright Object Time Series (BOTS)'
            constraints = inst['gratings_constraints']
            gratings = {}
            for filter in filters:
                for constraint in constraints:
                    if filter in constraints[constraint]['filters']:
                        label = f"{dispersers[constraint]}/{filters[filter]}"
                        gratings[f'{constraint}/{filter}'] = label
            filters = gratings
            dispersers = inst['slits']
            default_indices = 0, 6, 4, 1
        if mode == 'soss':
            disperser_label = 'Disperser'
            filter_label = 'Filter'
            inst['mode_label'] = 'Single Object Slitless Spectroscopy (SOSS)'
            default_indices = 0, 0, 0, 1

        det = Detector(
            mode,
            inst['mode_label'],
            inst['instrument'],
            inst['obs_type'],
            disperser_label,
            dispersers,
            filter_label,
            filters,
            subarrays,
            readouts,
            default_indices,
        )
        detectors.append(det)

    # Acquisition observing modes
    acq_insts = get_configs(obs_type='acquisition')
    for instrument in inst_names.values():
        insts = [inst for inst in acq_insts if inst['instrument'] == instrument]
        inst = insts[0]
        mode = inst['mode']
        # Use apertures in place of 'disperser'
        dispersers = {inst['aperture']: inst['mode_label']}
        filters = inst['filters']
        subarrays = inst['subarrays']
        readouts = inst['readouts']
        disperser_label = 'Acquisition mode'
        filter_label = 'Filter'

        if inst['instrument'] == 'MIRI':
            default_indices = 0, 0, 5, 0
            # handle groups - subarray
        if inst['instrument'] == 'NIRCam':
            default_indices = 0, 0, 0, 0
        if inst['instrument'] == 'NIRISS':
            default_indices = 0, 0, 0, 1
            dispersers[insts[1]['aperture']] = insts[1]['mode_label']
            # handle readouts
        if inst['instrument'] == 'NIRSpec':
            default_indices = 0, 0, 1, 0

        det = Detector(
            mode,
            inst['mode_label'],
            inst['instrument'],
            inst['obs_type'],
            disperser_label,
            dispersers,
            filter_label,
            filters,
            subarrays,
            readouts,
            default_indices,
        )
        detectors.append(det)


    #mrs_ts = Detector(
    #    'mrs_ts',
    #    'Medium Resolution Spectroscopy (MRS) time series',
    #    'MIRI',
    #    'spectroscopy',
    #)

    #imaging_ts = Detector(
    #    'imaging_ts',
    #    'Imaging time series',
    #    'MIRI',
    #    'photometry',
    #)

    #nircam_grism = Detector(
    #    'lw_ts',
    #    'long wavelength time-series imaging',
    #    'NIRCam',
    #    'photometry',
    #)

    #nircam_grism = Detector(
    #    'sw_ts',
    #    'short wavelength time-series imaging',
    #    'NIRCam',
    #    'photometry',
    #)

    return detectors


def detector_label(mode, disperser, filter, subarray, readout):
    """
    Generate a pretty and (as succinct as possible) label for the
    detector configuration.
    """
    if mode == 'lrsslitless':
        return 'MIRI LRS'
    if mode == 'soss':
        return f'NIRISS {mode.upper()} {subarray}'
    if mode == 'ssgrism':
        subarray = subarray.replace('grism', '')
        return f'NIRCam {filter.upper()} {subarray} {readout}'
    elif mode == 'bots':
        if filter == 'f070lp':
            disperser = f'{disperser}/{filter}'
        return f'NIRSpec {disperser.upper()} {subarray} {readout}'

