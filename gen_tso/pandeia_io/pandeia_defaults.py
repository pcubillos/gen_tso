# Copyright (c) 2024 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    'get_configs',
    'generate_all_instruments',
    'detector_label',
]

from dataclasses import dataclass
from pandeia.engine.calc_utils import get_instrument_config



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
            inst_dict['mode'] = {mode: mode_name}

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

            slits = get_constrained_values(
                inst_config, aper, 'slits', mode,
            )
            inst_dict['slits'] = {}
            for slit in slits:
                name = str_or_dict(slit_names, slit, mode)
                inst_dict['slits'][slit] = name
                #print(f"   {slit}: {name}")

            if obs_type == 'acquisition':
                print(inst_config['mode_config'][mode]['enum_nexps'])
                print(inst_config['mode_config'][mode]['enum_nints'])
                print(inst_config['mode_config'][mode]['enum_ngroups'])
            outputs.append(inst_dict)
    return outputs


@dataclass(order=True)
class Detector:
    name: str
    label: str
    instrument: str
    obs_type: str
    disperser_title: str
    dispersers: list
    filter_title: str
    filters: list
    subarrays: list
    readouts: list
    disperser_default: str
    filter_default: str
    subarray_default: str
    readout_default: str


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
    """
    telescope = 'jwst'

    inst = get_configs(instrument='miri', obs_type='spectroscopy')[0]
    mode = list(inst['mode'])[0]
    dispersers = inst['dispersers']
    filters = inst['filters']
    subarrays = inst['subarrays']
    readouts = inst['readouts']

    disperser_label = 'Disperser'
    filter_label = ''
    filters = {'': ''}
    lrs = Detector(
        mode,
        inst['mode'],
        inst['instrument'],
        inst['obs_type'],
        disperser_label,
        dispersers,
        filter_label,
        filters,
        subarrays,
        readouts,
        disperser_default=list(dispersers)[0],
        filter_default='',
        subarray_default=list(subarrays)[0],
        readout_default=list(readouts)[0],
    )
    lrs.ins_config = get_instrument_config(telescope, inst['instrument'].lower())

    inst = get_configs(instrument='nircam', obs_type='spectroscopy')[0]
    mode = list(inst['mode'])[0]
    dispersers = inst['dispersers']
    filters = inst['filters']
    subarrays = inst['subarrays']
    readouts = inst['readouts']

    disperser_label = 'Grism'
    filter_label = 'Filter'
    nircam_grism = Detector(
        mode,
        inst['mode'],
        inst['instrument'],
        inst['obs_type'],
        disperser_label,
        dispersers,
        filter_label,
        filters,
        subarrays,
        readouts,
        disperser_default=list(dispersers)[0],
        filter_default=list(filters)[3],
        subarray_default=list(subarrays)[3],
        readout_default=list(readouts)[0],
    )
    nircam_grism.ins_config = get_instrument_config(telescope, inst['instrument'].lower())

    inst = get_configs(instrument='niriss', obs_type='spectroscopy')[0]
    mode = list(inst['mode'])[0]
    dispersers = inst['dispersers']
    filters = inst['filters']
    subarrays = inst['subarrays']
    readouts = inst['readouts']

    #mode = 'soss'
    disperser_label = 'Disperser'
    filter_label = 'Filter'
    #'Single Object Slitless Spectroscopy (SOSS)',

    soss = Detector(
        mode,
        inst['mode'],
        inst['instrument'],
        inst['obs_type'],
        disperser_label,
        dispersers,
        filter_label,
        filters,
        subarrays,
        readouts,
        disperser_default=list(dispersers)[0],
        filter_default=list(filters)[0],
        subarray_default=list(subarrays)[0],
        readout_default=list(readouts)[1],
    )
    soss.ins_config = get_instrument_config(telescope, inst['instrument'].lower())

    inst = get_configs(instrument='nirspec', obs_type='spectroscopy')[0]
    mode = list(inst['mode'])[0]
    dispersers = inst['dispersers']
    filters = inst['filters']
    subarrays = inst['subarrays']
    readouts = inst['readouts']

    #mode = 'bots'
    disperser_label = 'Slit'
    filter_label = 'Grating/Filter'
    #'Bright Object Time Series (BOTS)',
    constraints = inst['gratings_constraints']
    gratings = {}
    for filter in filters:
        for constraint in constraints:
            if filter in constraints[constraint]['filters']:
                label = f"{dispersers[constraint]}/{filters[filter]}"
                gratings[f'{constraint}/{filter}'] = label
    filters = gratings
    dispersers = inst['slits']

    bots = Detector(
        mode,
        inst['mode'],
        inst['instrument'],
        inst['obs_type'],
        disperser_label,
        dispersers,
        filter_label,
        filters,
        subarrays,
        readouts,
        disperser_default=list(dispersers)[0],
        filter_default=list(filters)[6],
        subarray_default=list(subarrays)[4],
        readout_default=list(readouts)[1],
    )
    bots.ins_config = get_instrument_config(telescope, inst['instrument'].lower())


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


    #miri_ta = Detector(
    #    'target_acq',
    #    'Target Acquisition',
    #    'MIRI',
    #    'acquisition',
    #)

    #nircam_ta = Detector(
    #    'target_acq',
    #    'Target Acquisition',
    #    'NIRCam',
    #    'acquisition',
    #)

    #niriss_ta = Detector(
    #    'target_acq',
    #    'Target Acquisition',
    #    'NIRISS',
    #    'acquisition',
    #)

    #nirspec_ta = Detector(
    #    'target_acq',
    #    'Target Acquisition',
    #    'NIRSpec',
    #    'acquisition',
    #)


    detectors = [
        lrs,
        #mrs,
        #miri_imaging,
        #miri_ta,
        nircam_grism,
        #nircam_sw,
        #nircam_lw,
        #nircam_ta,
        soss,
        #niriss_ta,
        bots,
        #nirspec_ta,
    ]

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

