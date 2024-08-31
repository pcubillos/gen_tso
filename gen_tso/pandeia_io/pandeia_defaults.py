# Copyright (c) 2024 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    'default_aperture_strategy',
    'get_configs',
    'filter_throughputs',
    'generate_all_instruments',
    'load_flux_rate_splines',
    'Detector',
]

from itertools import product
import pickle

import numpy as np
from scipy.interpolate import CubicSpline
from pandeia.engine.calc_utils import get_instrument_config

from ..utils import ROOT


inst_names = {
    'miri': 'MIRI',
    'nircam': 'NIRCam',
    'niriss': 'NIRISS',
    'nirspec': 'NIRSpec',
}

spec_modes = [
   'lrsslitless',
   'mrs_ts',
   'lw_tsgrism',
   'sw_tsgrism',
   'soss',
   'bots',
]
photo_modes = [
    'imaging_ts',
    'lw_ts',
    'sw_ts',
]
acq_modes = [
    'target_acq',
]


# Spectra extraction apertures (arcsec) based on values reported in:
# Ahrer et al. (2023)    NIRCam/LW
# Alderson et al. (2023) NIRSpec/G395H
# Bouwman et al. (2024)  MIRI/LRS
# Bell et al. (2024)     MIRI/LRS
default_aperture_strategy = {
    'lrsslitless': dict(
        aperture_size = 0.6,
        sky_annulus = [1.0, 2.5],
    ),
    'mrs_ts': dict(
        aperture_size = 0.6,
        sky_annulus = [1.0, 1.5],
    ),
    'lw_tsgrism': dict(
        aperture_size = 0.6,
        sky_annulus = [0.9, 1.5],
    ),
    'sw_tsgrism': dict(
        aperture_size = 0.6,
        sky_annulus = [0.9, 1.5],
    ),
    'bots': dict(
        aperture_size = 0.7,
        sky_annulus = [0.7, 1.5],
    ),
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

    if mode == 'sw_ts' and inst_property == 'filters':
        values = constraints[aper][inst_property]
        values = values[mode] if isinstance(values, dict) else values
    elif is_constrained:
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

    Some properties have special constraints which are specified
    by the 'constraints' item, which is a nested dict of the shape:
        options = constraints[constrained_field][constrainer_field][val]
    For example, for nirspec bots the filters options depend on the disperser,
    then to get the filters available for the 'g140h' disperser do:
        constrained_filters = constraints['filter']['disperser']['g140h']

    Examples
    --------
    >>> from gen_tso.pandeia_io import get_configs

    >>> insts = get_configs(obs_type='spectroscopy')
    >>> insts = get_configs(obs_type='photometry')
    >>> insts = get_configs(obs_type='acquisition')

    >>> insts = get_configs(instrument='miri')
    >>> insts = get_configs(instrument='nircam', obs_type='spectroscopy')
    """
    ta_apertures = {
        'miri': ['imager'],
        'nircam': ['lw'],
        'niriss': ['imager', 'nrm'],
        'nirspec': ['s1600a1'],
    }
    if instrument is None:
        instrument = 'miri nircam nirspec niriss'.split()
    elif isinstance(instrument, str):
        instrument = [instrument]

    if obs_type is None:
        obs_type = ['spectroscopy', 'acquisition']
    elif isinstance(obs_type, str):
        obs_type = [obs_type]

    modes = []
    if 'spectroscopy' in obs_type:
        modes += spec_modes
    if 'acquisition' in obs_type:
        modes += acq_modes
    if 'photometry' in obs_type:
        modes += photo_modes

    telescope = 'jwst'
    outputs = []
    for mode, inst in product(modes, instrument):
        inst_config = get_instrument_config(telescope, inst)
        if mode not in inst_config['modes']:
            continue

        disperser_names = inst_config['disperser_config']
        filter_names = inst_config['filter_config']
        readout_names = inst_config['readout_pattern_config']
        subarray_names = inst_config['subarray_config']['default']
        aperture_names = inst_config['aperture_config']
        if 'slit_config' in inst_config:
            slit_names = inst_config['slit_config']

        if mode in spec_modes:
            obs_type = 'spectroscopy'
        elif mode in acq_modes:
            obs_type = 'acquisition'
        elif mode in photo_modes:
            obs_type = 'photometry'

        inst_dict = {}
        inst_dict['instrument'] = inst_names[inst]
        inst_dict['obs_type'] = obs_type
        mode_name = inst_config['mode_config'][mode]['display_string']
        inst_dict['mode'] = mode
        inst_dict['mode_label'] = mode_name

        apertures = inst_config['mode_config'][mode]['apertures']
        if obs_type == 'acquisition':
            apertures = [
                aper for aper in apertures
                if aper in ta_apertures[inst]
            ]
        inst_dict['apertures'] = {
            aperture: str_or_dict(aperture_names, aperture, mode)
            for aperture in apertures
        }
        aper = apertures[0]

        dispersers = get_constrained_values(
            inst_config, aper, 'dispersers', mode,
        )
        inst_dict['dispersers'] = {
            disperser: str_or_dict(disperser_names, disperser, mode)
            for disperser in dispersers
        }

        filters = get_constrained_values(inst_config, aper, 'filters', mode)
        inst_dict['filters'] = {
            filter: str_or_dict(filter_names, filter, mode)
            for filter in filters
        }

        readouts = get_constrained_values(
            inst_config, aper, 'readout_patterns', mode,
        )
        inst_dict['readouts'] = {
            readout: str_or_dict(readout_names, readout, mode)
            for readout in readouts
        }

        subarrays = get_constrained_values(inst_config, aper, 'subarrays', mode)
        inst_dict['subarrays'] = {
            subarray: str_or_dict(subarray_names, subarray, mode)
            for subarray in subarrays
        }

        slits = get_constrained_values(inst_config, aper, 'slits', mode)
        inst_dict['slits'] = {
            slit: str_or_dict(slit_names, slit, mode)
            for slit in slits
        }

        if inst_dict['mode'] == 'soss':
            inst_dict['orders'] = {
                '1': '1',
                '2': '2',
                '1 2': '1 and 2',
            }
        else:
            inst_dict['orders'] = None

        if obs_type == 'acquisition':
            groups = inst_config['mode_config'][mode]['enum_ngroups']
            if not isinstance(groups, list):
                groups = groups['default']
            inst_dict['groups'] = groups
            # Integrations and exposures are always one (so far)
            #print(inst_config['mode_config'][mode]['enum_nexps'])
            #print(inst_config['mode_config'][mode]['enum_nints'])


        # Special constraints
        inst_dict['constraints'] = {}
        if mode == 'sw_tsgrism':
            aper_constraints = inst_config['config_constraints']['apertures']
            constraints = {
                aperture: aper_constraints[aperture]['subarrays']['default']
                for aperture in apertures
            }
            inst_dict['constraints']['subarrays'] = {'apertures': constraints}

        if mode == 'bots':
            disp_constraints = inst_config['config_constraints']['dispersers']
            constraints = {
                disperser: disp_constraints[disperser]['filters']
                for disperser in dispersers
            }
            inst_dict['constraints']['filters'] = {'dispersers': constraints}
            constraints = {}
            for disperser in dispersers:
                subs = subarrays.copy()
                if disperser == 'prism':
                    subs.remove('sub1024a')
                constraints[disperser] = subs
            inst_dict['constraints']['subarrays'] = {'dispersers': constraints}

        if mode == 'soss':
            constraints = {
                'substrip96': ['1'],
                'substrip256': ['1', '2', '1 2'],
                'sossfull': ['1', '2', '1 2'],
            }
            inst_dict['constraints']['orders'] = {'subarrays': constraints}
            constraints = {
                'substrip96': ['nisrapid'],
                'substrip256': ['nisrapid'],
                'sossfull': readouts,
            }
            inst_dict['constraints']['readouts'] = {'subarrays': constraints}

        if inst_dict['instrument']=='MIRI' and mode=='target_acq':
            group_constraints = inst_config['mode_config'][mode]['enum_ngroups']
            constraints = {}
            for subarray in subarrays:
                key = subarray if subarray in group_constraints else 'default'
                constraints[subarray] = group_constraints[key]
            inst_dict['constraints']['groups'] = {'subarrays': constraints}

        if inst_dict['instrument']=='NIRISS' and mode=='target_acq':
            constraints = {
                aperture: get_constrained_values(
                    inst_config, aperture, 'readout_patterns', mode,
                )
                for aperture in apertures
            }
            inst_dict['constraints']['readouts'] = {'apertures': constraints}

        if mode == 'lw_ts' or mode == 'sw_ts':
            # NIRCam is so special
            inst_dict['double_filter_constraints']  = inst_config['double_filters']
        if mode == 'sw_ts':
            constraints = {
                aperture: get_constrained_values(
                    inst_config, aperture, 'filters', mode,
                )
                for aperture in apertures
            }
            inst_dict['constraints']['filters'] = {'apertures': constraints}
            constraints = {
                aperture: get_constrained_values(
                    inst_config, aperture, 'subarrays', mode,
                )
                for aperture in apertures
            }
            inst_dict['constraints']['subarrays'] = {'apertures': constraints}

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
            subarrays, readouts, apertures, orders=None,
            groups=None, default_indices=None,
            constraints={},
        ):
        """
        An object containing the available instrumental setup options.
        (Intended for the front-end)

        The constraints are a triple-nested dict of the form:
        constraints[constrained_var][constraining_var][var] = constraints
        For example, readout values constrained by the disperser:
        constraints['readouts']['dispersers'] = {
            'imager': ['nis', 'nisrapid'],
            'nrm': ['nisrapid'],
        }
        """
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
        self.apertures = apertures
        self.groups = groups
        self.constraints = constraints
        if self.mode == 'soss':
            self.orders = orders

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

    def get_constrained_val(
            self, var,
            disperser=None, filter=None, subarray=None, readout=None,
            aperture=None,
        ):
        """
        det = generate_all_instruments()[5]
        det.get_constrained_val('filters', disperser='nrm')
        get_constrained_val(det, 'readouts', disperser='nrm')
        get_constrained_val(det, 'readouts', disperser='imager')
        get_constrained_val(det, 'readouts', filter='imager')
        get_constrained_val(det, 'readouts')
        get_constrained_val(det, 'readouts', disperser='lala')
        """
        choices = {
            'dispersers': disperser,
            'filters': filter,
            'subarrays': subarray,
            'readouts': readout,
            'apertures': aperture,
        }

        default_vals = getattr(self, var)
        if var not in self.constraints:
            return default_vals

        for field, constraint in choices.items():
            if constraint is None:
                continue
            is_constrained = (
                field in self.constraints[var] and
                constraint in self.constraints[var][field]
            )
            if is_constrained:
                values = self.constraints[var][field][constraint]
                if isinstance(default_vals, list):
                    return values
                return {
                    val:label
                    for val,label in default_vals.items()
                    if val in values
                }
        return default_vals

    def instrument_label(self, disperser, filter):
        """
        Generate a pretty label with only instrument, mode,
        and sometimes disperser or filter.

        Examples
        --------
        >>> import gen_tso.pandeia_io as jwst
        >>> detectors = jwst.generate_all_instruments()
        >>> bots = detectors[4]
        >>> print(bots.instrument_label(None, 'g395h/f290lp'))
        """
        inst = self.instrument
        mode = self.mode
        if mode == 'mrs_ts':
            disperser_label = self.dispersers[disperser]
            label = f'{inst} / MRS / {disperser_label}'
        elif mode == 'lrsslitless':
            label = f'{inst} / LRS'
        elif mode == 'soss':
            label = f'{inst} / SOSS'
        elif mode == 'lw_tsgrism':
            filter_label = self.filters[filter]
            label = f'{inst} / {filter_label}'
        elif mode == 'sw_tsgrism':
            filter_label = self.filters[filter]
            label = f'{inst} / {filter_label}'
        elif mode == 'bots':
            disperser_label = self.filters[filter]
            disperser_label = disperser_label[:disperser_label.index('/')]
            label = f'{inst} / {disperser_label}'
        elif mode == 'target_acq':
            label = f'{inst} / acquisition'

        return label


def filter_throughputs():
    """
    Collect the throughput response curves for each instrument configuration

    Examples
    --------
    >>> import gen_tso.pandeia_io as jwst
    >>> filter_throughputs = jwst.filter_throughputs()
    """
    detectors = generate_all_instruments()
    throughputs = {}
    for detector in detectors:
        inst = detector.instrument.lower()
        mode = detector.mode
        obs_type = detector.obs_type
        #print(inst, mode, obs_type)
        if obs_type not in throughputs:
            throughputs[obs_type] = {}
        if inst not in throughputs[obs_type]:
            throughputs[obs_type][inst] = {}
        if mode not in throughputs[obs_type][inst]:
            throughputs[obs_type][inst][mode] = {}

        t_file = f'{ROOT}data/throughputs/throughputs_{inst}_{mode}.pickle'
        with open(t_file, 'rb') as handle:
            data = pickle.load(handle)

        for subarray in list(data.keys()):
            throughputs[obs_type][inst][mode][subarray] = data[subarray]

    return throughputs


def generate_all_instruments():
    """
    A list of Detector() objects to keep the instrument observing mode
    configurations.

    TBD
    ---
    Imaging
        'imaging_ts': 'Imaging Time Series'
        'sw_ts': 'SW Time Series'
        'lw_ts': 'LW Time Series'

    Examples
    --------
    >>> from gen_tso.pandeia_io import get_configs, generate_all_instruments
    >>> spec_insts = get_configs(obs_type='spectroscopy')
    >>> acq_insts = get_configs(obs_type='acquisition')
    >>> dets = generate_all_instruments()
    >>> det = dets[5]
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
        apertures = inst['apertures']
        orders = inst['orders']
        constraints = inst['constraints']

        if mode == 'lrsslitless':
            disperser_label = 'Disperser'
            filter_label = ''
            filters = {'': ''}
            default_indices = 0, 0, 0, 0
        if mode == 'mrs_ts':
            disperser_label = 'Wavelength Range'
            filter_label = ''
            filters = {'': ''}
            default_indices = 0, 0, 0, 0
        if mode == 'lw_tsgrism':
            disperser_label = 'Grism'
            filter_label = 'Filter'
            default_indices = 0, 3, 3, 0
        if mode == 'sw_tsgrism':
            # TBD: a hack, in the future, better have an 'aperture' show/hide
            disperser_label = 'PSF Type'
            dispersers = apertures
            filter_label = 'Filter'
            default_indices = 3, 4, 0, 0
        if mode == 'bots':
            disperser_label = 'Slit'
            filter_label = 'Grating/Filter'
            inst['mode_label'] = 'Bright Object Time Series (BOTS)'
            filter_constraints = constraints['filters']['dispersers']
            gratings = {}
            for filter in filters:
                for disperser, c_filters in filter_constraints.items():
                    if filter in c_filters:
                        label = f"{dispersers[disperser]}/{filters[filter]}"
                        gratings[f'{disperser}/{filter}'] = label
            filters = gratings
            dispersers = inst['slits']
            default_indices = 0, 7, 4, 1
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
            apertures,
            orders=orders,
            default_indices=default_indices,
            constraints=constraints,
        )
        detectors.append(det)

    # Photometry observing modes
    # TBD: enable when the front-end app is ready to take these in
    if False:
        photo_insts = get_configs(obs_type='photometry')
        for inst in photo_insts:
            mode = inst['mode']
            # Use pupil in place of 'disperser'
            dispersers = inst['apertures']
            filters = inst['filters']
            subarrays = inst['subarrays']
            readouts = inst['readouts']
            apertures = inst['apertures']

            disperser_label = 'Pupil'
            filter_label = 'Filter'
            default_indices = 0, 0, 0, 0
            constraints = {}

            if mode == 'lw_ts':
                # Constraints for filter depends on pupil
                pupil_constraints = {'lw': []}
                for filter,filter_name in inst['filters'].items():
                    if filter in inst['filter_constraints']:
                        dispersers[filter] = filter_name
                        pupil_constraints[filter] = [inst['filter_constraints'][filter]]
                    else:
                        pupil_constraints['lw'].append(filter)
                constraints['filters'] = {'dispersers': pupil_constraints}
            if mode == 'sw_ts':
                # Constraints for filter depends on pupil
                pupil_constraints = {'sw': []}
                for filter,filter_name in inst['filters'].items():
                    print(filter, filter_name)
                    if filter in inst['filter_constraints']:
                        dispersers[filter] = filter_name
                        pupil_constraints[filter] = [inst['filter_constraints'][filter]]
                    else:
                        pupil_constraints['sw'].append(filter)
                constraints['filters'] = {'dispersers': pupil_constraints}
            if mode == 'imaging_ts':
                disperser_label = ''
                dispersers = {'':''}

    # Acquisition observing modes
    acq_insts = get_configs(obs_type='acquisition')
    for inst in acq_insts:
        mode = inst['mode']
        # Use apertures in place of 'disperser'
        dispersers = inst['apertures']
        filters = inst['filters']
        subarrays = inst['subarrays']
        readouts = inst['readouts']
        apertures = inst['apertures']
        disperser_label = 'Acquisition mode'
        filter_label = 'Filter'
        constraints = inst['constraints']

        if inst['instrument'] == 'MIRI':
            default_indices = 0, 0, 5, 0
        if inst['instrument'] == 'NIRCam':
            default_indices = 0, 0, 0, 0
        if inst['instrument'] == 'NIRISS':
            default_indices = 0, 0, 0, 1
            # Re-label constraint for front-end (aperture-->disperser)
            r_constraint = constraints['readouts'].pop('apertures')
            constraints['readouts']['dispersers'] = r_constraint
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
            apertures,
            groups=inst['groups'],
            default_indices=default_indices,
            constraints=constraints,
        )
        detectors.append(det)

    return detectors


def get_detector(instrument=None, mode=None, detectors=None):
    """
    Find the detector matching instrument and mode.
    """
    if detectors is None:
        detectors = generate_all_instruments()

    if instrument is None and mode is None:
        return None

    if mode is None:
        # Default to first detector for instrument (spectroscopic mode)
        for detector in detectors:
            if detector.instrument.lower() == instrument.lower():
                return detector

    for det in detectors:
        if det.mode == mode:
            if instrument is None or det.instrument.lower()==instrument.lower():
                return det
    return None


def make_save_label(
        target, inst, mode, aperture, disperser, filter,
        subarray=None, readout=None, order=None,
    ):
    """
    A nice short name when saving to file via the application
    """
    if target is not None:
        target = '_' + target.replace(' ', '')
    if mode == 'target_acq':
        return f'tso{target}_{inst}_{mode}.pickle'
    # Shorter miri and nircam mode names:
    mode = mode.replace('lrsslitless', 'lrs').replace('mrs_ts', 'mrs')
    mode = mode.replace('w_tsgrism', 'w')

    if inst in ['miri', 'niriss']:
        return f'tso{target}_{inst}_{mode}.pickle'
    elif inst == 'nirspec':
        return f'tso{target}_{inst}_{mode}_{disperser}.pickle'
    elif mode == 'lw':
        return f'tso{target}_{inst}_{mode}_{filter}.pickle'
    elif mode == 'sw':
        return f'tso{target}_{inst}_{mode}_{filter}_{aperture}.pickle'


def make_detector_label(
        inst, mode, aperture, disperser, filter, subarray, readout, order,
    ):
    """
    Generate a pretty and (as succinct as possible) label for the
    detector configuration.
    """
    if mode == 'target_acq':
        if inst == 'miri':
            label = f'{filter.upper()} {subarray.upper()}'
        if inst == 'nircam':
            label = f'{filter.upper()} {readout.upper()}'
        if inst == 'niriss':
            label = 'Bright' if aperture == 'nrm' else 'Faint'
            label = f'{label} {readout.upper()}'
        if inst == 'nirspec':
            label = f'{filter.upper()} {subarray.upper()} {readout.upper()}'
        return f'{inst_names[inst]} {label}'

    if mode == 'mrs_ts':
        return f'MIRI MRS {disperser.upper()}'
    if mode == 'lrsslitless':
        return 'MIRI LRS'
    if mode == 'soss':
        order = f' O{order[0]}' if len(order)==1 else ''
        return f'NIRISS {mode.upper()} {subarray}{order}'
    if mode == 'lw_tsgrism':
        subarray = subarray.replace('grism', '').replace('_dhs', '')
        return f'NIRCam {filter.upper()} {subarray} {readout}'
    if mode == 'sw_tsgrism':
        return f'NIRCam {filter.upper()} {subarray} {readout}'
    if mode == 'bots':
        if filter == 'f070lp':
            disperser = f'{disperser}/{filter}'
        return f'NIRSpec {disperser.upper()} {subarray} {readout}'


def make_saturation_label(
        inst, mode, aperture, disperser, filter, subarray, order, sed_label,
    ):
    """
    Make a label of unique saturation setups to identify when and
    when not the saturation level can be estimated.
    """
    sat_label = f'{mode}_{filter}'
    if mode == 'bots':
        sat_label = f'{sat_label}_{disperser}_{subarray}'
    elif mode == 'soss':
        order = f'_O{order[0]}' if len(order)==1 else ''
        sat_label = f'{sat_label}{order}'
    elif mode == 'sw_tsgrism':
        sat_label = f'{sat_label}_{aperture}_{subarray}'
    elif mode == 'mrs_ts':
        sat_label = f'{sat_label}_{disperser}'
    elif mode == 'target_acq' and inst == 'niriss':
        sat_label = f'{sat_label}_{aperture}'

    sat_label = f'{sat_label}_{sed_label}'
    return sat_label


def make_obs_label(
        inst, mode, aperture, disperser, filter, subarray, readout, order,
        ngroup, nint, run_type, sed_label, depth_label,
    ):
    detector_label = make_detector_label(
        inst, mode, aperture, disperser, filter, subarray, readout, order,
    )
    if run_type == 'Acquisition':
        group_ints = f'({ngroup} G)'
        depth_label = 'acquisition'
    else:
        group_ints = f'({ngroup} G, {nint} I)'
    label = (
        f'{detector_label} {group_ints} / {sed_label} / {depth_label}'
    )
    return label


def load_flux_rate_splines(obs_label=None):
    """
    Get dictionary of cubic spline functions for pre-calculated
    brightest pixel flux rates at given instrumental and SED config
    combinations.

    Parameters
    ----------
    obs_label: String
         If not None, only return the flux_rate and full_well for the
         instrument and SED configuration matching the input label.
         Return (None,None) if no config matched the obs_label.

    Returns
    -------
    flux_rates: Dictionary of CubicSpline objects
        Spline function that returns the flux rate at a given Ks magnitude.
    full_wells: Dictionary of floats
        e- counts to saturate the detector.

    Examples
    --------
    >>> import gen_tso.pandeia_io as jwst
    >>>
    >>> # Extract all splines:
    >>> flux_rate_splines, full_wells = jwst.load_flux_rate_splines()
    >>> # Evaluate for one config:
    >>> obs_label = 'lw_tsgrism_f444w_phoenix_k5v'
    >>> spline = flux_rate_splines[obs_label]
    >>> print(spline(8.351), full_wells[obs_label])
    3.1140133020415353 58100.001867429375
    >>>
    >>> # Extract a single config:
    >>> obs_label = 'lw_tsgrism_f444w_phoenix_k5v'
    >>> flux_rate_spline, full_well = jwst.load_flux_rate_splines(obs_label)
    >>> print(flux_rate_spline(8.351), full_well)
    3.1140133020415353 58100.001867429375
    """
    if obs_label is not None:
        tokens = obs_label.split('_')
        i_label = '_'.join(tokens[:-2]) + '_'

    flux_rate_data = {}
    for instrument in ['miri', 'nircam', 'niriss', 'nirspec']:
        inst = instrument.lower()
        rate_file = f'{ROOT}data/flux_rates_{inst}.pickle'
        with open(rate_file, 'rb') as handle:
            rates = pickle.load(handle)
        mag = rates.pop('magnitude')
        flux_rate_data[inst] = rates

    flux_rates = {}
    full_wells = {}
    for inst, mode_rates in flux_rate_data.items():
        for mode, disp_rates in mode_rates.items():
            orders = ['1', '1 2'] if mode=='soss' else ['1']
            for disperser, filter_rates in disp_rates.items():
                for filter, sub_rates in filter_rates.items():
                    for subarray, sed_rates in sub_rates.items():
                        for order in orders:
                            if mode in ['sw_tsgrism', 'mrs_ts']:
                                aperture = disperser
                            else:
                                aperture = ''
                            filter = filter.replace('None', '')
                            inst_label = make_saturation_label(
                                inst, mode, aperture, disperser, filter,
                                subarray, order, ''
                            )
                            if obs_label is not None and inst_label != i_label:
                                continue
                            for i,rate in enumerate(sed_rates['phoenix']):
                                log_rate = np.log10(rate)
                                sed = sed_rates['p_names'][i]
                                label = f'{inst_label}phoenix_{sed}'
                                if obs_label is None:
                                    flux_rates[label] = CubicSpline(mag, log_rate)
                                    full_wells[label] = sed_rates['full_well']
                                elif label == obs_label:
                                    return CubicSpline(mag, log_rate), sed_rates['full_well']


                            for i,rate in enumerate(sed_rates['k93models']):
                                log_rate = np.log10(rate)
                                sed = sed_rates['k_names'][i]
                                label = f'{inst_label}kurucz_{sed}'
                                if obs_label is None:
                                    flux_rates[label] = CubicSpline(mag, log_rate)
                                    full_wells[label] = sed_rates['full_well']
                                elif label == obs_label:
                                    return CubicSpline(mag, log_rate), sed_rates['full_well']
    # Model not found
    if obs_label is not None:
        return None, None
    return flux_rates, full_wells


