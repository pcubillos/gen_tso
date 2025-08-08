# Copyright (c) 2025 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    'get_instruments',
    'get_modes',
    'get_sed_types',
    '_spec_modes',
    '_photo_modes',
    '_acq_modes',
    '_default_aperture_strategy',
    '_get_configs',
    'get_throughputs',
    'generate_all_instruments',
    '_load_flux_rate_splines',
    'Detector',
]

from itertools import product
import pickle

import numpy as np
from scipy.interpolate import CubicSpline
from pandeia.engine.calc_utils import get_instrument_config

from ..utils import ROOT


# Dictionaries
inst_names = {
    'miri': 'MIRI',
    'nircam': 'NIRCam',
    'niriss': 'NIRISS',
    'nirspec': 'NIRSpec',
}

spec_dict = {
    'miri': ['lrsslitless', 'lrsslit', 'mrs_ts'],
    'nircam': ['lw_tsgrism', 'sw_tsgrism'],
    'niriss': ['soss'],
    'nirspec': ['bots'],
}

photo_dict = {
    'miri': ['imaging_ts'],
    'nircam': ['lw_ts', 'sw_ts'],
}

# Lists
_spec_modes = []
for modes in spec_dict.values():
    _spec_modes += modes

_photo_modes = []
for modes in photo_dict.values():
    _photo_modes += modes

_acq_modes = [
    'target_acq',
]

def get_instruments():
    """
    Get the list of JWST instruments.

    Returns
    -------
    instruments: 1D list of strings
        JWST instruments
    """
    return list(inst_names)


def get_modes(instrument, type=None):
    """
    Get the list of available TSO modes for a given instrument.

    Parameters
    ----------
    instrument: String
        A JWST instrument.
    type: String
        If set, get only the modes of the specified type:
        spectroscopy, photometry, or acquisition

    Returns
    -------
    modes: 1D list of strings
        JWST TSO modes for instrument
    """
    modes = []
    if type is None:
        types = ['spectroscopy', 'photometry', 'acquisition']
    else:
        types = [type]

    if 'spectroscopy' in types:
        modes += spec_dict[instrument]
    if 'photometry' in types and instrument in photo_dict:
        modes += photo_dict[instrument]
    if 'acquisition' in types:
        modes += _acq_modes

    return modes


def get_sed_types():
    """
    Get the list of SED models.

    Returns
    -------
    instruments: 1D list of strings
        JWST instruments
    """
    return [
        'phoenix',
        'k93models',
        'bt_settl',
    ]


# Spectral extraction apertures (arcsec) based on values reported in:
# Ahrer et al. (2023)    NIRCam/LW
# Alderson et al. (2023) NIRSpec/G395H
# Bouwman et al. (2024)  MIRI/LRS
# Bell et al. (2024)     MIRI/LRS
_default_aperture_strategy = {
    'lrsslitless': dict(
        aperture_size = 0.6,
        sky_annulus = [1.0, 2.5],
    ),
    'lrsslit': dict(
        aperture_size = 0.88,
        sky_annulus = [0.88, 1.4],
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


def get_constraints(config, inst_property, mode, **prop_constraints):
    """
    Extract instrument properties that might be constrained or not.

    Parameters
    ----------
    config: Dict
        Instrument config dict.
    inst_property: String
        The property to extract. Select from: dispersers, filters,
        subarrays, or readout_patterns.
    mode: String
        Observing mode.
    **prop_constraints: String
        Constraining property and value.
    """
    prop_key = list(prop_constraints)[0]
    constraint = prop_constraints[prop_key]

    constraints = {}
    if prop_key in config['config_constraints']:
        constraints = config['config_constraints'][prop_key]
    is_constrained = (
        constraint is not None and
        constraint in constraints and
        inst_property in constraints[constraint]
    )
    if is_constrained:
        constraint_val = constraints[constraint][inst_property]
        if isinstance(constraint_val, list):
            return constraint_val
        if mode in constraint_val:
            return constraint_val[mode]
        if constraint_val['default'] is None:
            return config['mode_config'][mode][inst_property]
        return constraint_val['default']
    if inst_property in config['mode_config'][mode]:
        return config['mode_config'][mode][inst_property]
    return config[inst_property]


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


def _get_configs(instrument=None, obs_type=None):
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
    >>> from gen_tso.pandeia_io import _get_configs

    >>> insts = _get_configs(obs_type='spectroscopy')
    >>> insts = _get_configs(obs_type='photometry')
    >>> insts = _get_configs(obs_type='acquisition')

    >>> insts = _get_configs(instrument='miri')
    >>> insts = _get_configs(instrument='nircam', obs_type='spectroscopy')
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
        modes += _spec_modes
    if 'photometry' in obs_type:
        modes += _photo_modes
    if 'acquisition' in obs_type:
        modes += _acq_modes

    telescope = 'jwst'
    outputs = []
    for mode, inst in product(modes, instrument):
        config = get_instrument_config(telescope, inst)
        if mode not in config['modes']:
            continue

        names = {
            'dispersers': config['disperser_config'],
            'filters': config['filter_config'],
            'readout_patterns': config['readout_pattern_config'],
            'subarrays': config['subarray_config']['default'],
        }
        if 'slit_config' in config:
            names['slits'] = config['slit_config']
        props = list(names)

        if mode in _spec_modes:
            obs_type = 'spectroscopy'
        elif mode in _acq_modes:
            obs_type = 'acquisition'
        elif mode in _photo_modes:
            obs_type = 'photometry'

        inst_dict = {}
        inst_dict['instrument'] = inst_names[inst]
        inst_dict['obs_type'] = obs_type
        mode_name = config['mode_config'][mode]['display_string']
        inst_dict['mode'] = mode
        inst_dict['mode_label'] = mode_name

        aperture_names = config['aperture_config']
        apertures = config['mode_config'][mode]['apertures']
        if obs_type == 'acquisition':
            apertures = [
                aper for aper in apertures
                if aper in ta_apertures[inst]
            ]
        inst_dict['apertures'] = {
            aperture: str_or_dict(aperture_names, aperture, mode)
            for aperture in apertures
        }
        aper = apertures[0] if mode=='target_acq' else None

        for prop in props:
            vals = get_constraints(config, prop, mode, apertures=aper)
            prop_name = 'readouts' if prop=='readout_patterns' else prop
            inst_dict[prop_name] = {
                value: str_or_dict(names[prop], value, mode)
                for value in vals
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
            groups = config['mode_config'][mode]['enum_ngroups']
            if not isinstance(groups, list):
                groups = groups['default']
            inst_dict['groups'] = groups
            # Integrations and exposures are always one (so far)
            #print(config['mode_config'][mode]['enum_nexps'])
            #print(config['mode_config'][mode]['enum_nints'])


        # Special constraints
        inst_dict['constraints'] = {}
        # MIRI
        if inst_dict['instrument']=='MIRI' and mode=='target_acq':
            group_constraints = config['mode_config'][mode]['enum_ngroups']
            constraints = {}
            for subarray in inst_dict['subarrays']:
                key = subarray if subarray in group_constraints else 'default'
                constraints[subarray] = group_constraints[key]
            inst_dict['constraints']['groups'] = {'subarrays': constraints}

        # NIRCam
        if mode == 'sw_tsgrism':
            constraints = {
                aper: get_constraints(config, 'subarrays', mode, apertures=aper)
                for aper in apertures
            }
            inst_dict['constraints']['subarrays'] = {'apertures': constraints}

        if mode == 'lw_tsgrism':
            constraints = {
                subarray: get_constraints(
                    config, 'readout_patterns', mode, subarrays=subarray,
                )
                for subarray in inst_dict['subarrays']
            }
            inst_dict['constraints']['readouts'] = {'subarrays': constraints}

        if inst == 'nircam' and obs_type == 'photometry':
            constraints = {
                aper: get_constraints(
                    config, 'filters', mode, apertures=aper,
                )
                for aper in apertures
            }
            double_filters = config['double_filters']

            pupil_to_aperture = {}
            pupils = {}
            filters = {}
            for aper, a_label in inst_dict['apertures'].items():
                pupils[aper] = a_label
                filters[aper] = []
                for filter in constraints[aper]:
                    if filter in double_filters:
                        filters[filter] = [filter]
                        pupils[filter] = filter.upper()
                        pupil_to_aperture[filter] = aper
                    else:
                        filters[aper] += [filter]
                pupil_to_aperture[aper] = aper
            inst_dict['constraints']['filters'] = {'pupils': filters}
            inst_dict['pupils'] = pupils
            inst_dict['pupil_to_aperture'] = pupil_to_aperture

        if mode == 'sw_ts':
            inst_dict['constraints']['filters']['apertures'] = constraints
            constraints = {
                aper: get_constraints(
                    config, 'subarrays', mode, apertures=aper,
                )
                for aper in apertures
            }
            inst_dict['constraints']['subarrays'] = {'apertures': constraints}
            pairings = {
                'imaging': 'LW Imaging Time Series',
                'grism': 'LW Grism Time Series',
            }
            constraints = {pairing: [] for pairing in pairings}
            for pupil in inst_dict['pupils']:
                if inst_dict['pupil_to_aperture'][pupil] in ['sw', 'wlp8__ts']:
                    constraints['imaging'].append(pupil)
                else:
                    constraints['grism'].append(pupil)
            inst_dict['pairings'] = pairings
            inst_dict['constraints']['pupils'] = {'pairings': constraints}

        # NIRSpec
        if mode == 'bots':
            constraints = {
                disp: get_constraints(config, 'filters', mode, dispersers=disp)
                for disp in inst_dict['dispersers']
            }
            inst_dict['constraints']['filters'] = {'dispersers': constraints}
            constraints = {}
            for disperser in inst_dict['dispersers']:
                subs = list(inst_dict['subarrays'])
                if disperser == 'prism':
                    subs.remove('sub1024a')
                constraints[disperser] = subs
            inst_dict['constraints']['subarrays'] = {'dispersers': constraints}

        # NIRISS
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
                'sossfull': inst_dict['readouts'],
            }
            inst_dict['constraints']['readouts'] = {'subarrays': constraints}

        if inst_dict['instrument']=='NIRISS' and mode=='target_acq':
            constraints = {
                aper: get_constraints(
                    config, 'readout_patterns', mode, apertures=aper,
                )
                for aper in inst_dict['apertures']
            }
            inst_dict['constraints']['readouts'] = {'apertures': constraints}

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
            aperture_label, apertures,
            disperser_label, dispersers,
            filter_label, filters,
            subarrays, readouts, orders=None,
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
        self.aperture_label = aperture_label
        self.apertures = apertures
        self.disperser_label = disperser_label
        self.dispersers = dispersers
        self.filter_label = filter_label
        self.filters = filters
        self.subarrays = subarrays
        self.readouts = readouts
        self.groups = groups
        self.constraints = constraints
        if self.mode == 'soss':
            self.orders = orders

        telescope = 'jwst'
        self.ins_config = get_instrument_config(telescope, instrument.lower())

        if default_indices is None:
            idx_a = idx_d = idx_f = idx_s = idx_r = 0
        else:
            idx_a, idx_d, idx_f, idx_s, idx_r = default_indices
        self.default_aperture = list(apertures)[idx_a]
        if self.mode == 'target_acq':
            self.default_disperser = None
        else:
            self.default_disperser = list(dispersers)[idx_d]
        self.default_filter = list(filters)[idx_f]
        self.default_subarray = list(subarrays)[idx_s]
        self.default_readout = list(readouts)[idx_r]

    def get_constrained_val(
            self, var,
            disperser=None, filter=None, subarray=None, readout=None,
            aperture=None, pupil=None, pairing=None,
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
            'pupils': pupil,
            'pairings': pairing,
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
        >>> bots = detectors[5]
        >>> print(bots.instrument_label('g395h', 'f290lp'))
        """
        inst = self.instrument
        mode = self.mode
        if mode == 'mrs_ts':
            disperser_label = self.dispersers[disperser]
            label = f'{inst} / MRS / {disperser_label}'
        elif mode in ['lrsslit', 'lrsslitless']:
            label = f'{inst} / LRS / {mode[3:].upper()}'
        elif mode == 'imaging_ts':
            filter_label = self.filters[filter]
            label = f'{inst} / {filter_label}'

        elif mode == 'soss':
            label = f'{inst} / SOSS'

        elif mode == 'lw_tsgrism':
            filter_label = self.filters[filter]
            label = f'{inst} / {filter_label}'
        elif mode == 'sw_tsgrism':
            filter_label = self.filters[filter]
            label = f'{inst} / {filter_label}'
        elif mode == 'lw_ts' or mode == 'sw_ts':
            filter_label = self.filters[filter]
            label = f'{inst} / {filter_label}'

        elif mode == 'bots':
            label = self.filters[f'{disperser}/{filter}']
            disperser_label = label[:label.index('/')]
            label = f'{inst} / {disperser_label}'

        elif mode == 'target_acq':
            label = f'{inst} / acquisition'

        return label


def get_throughputs(type=None, inst=None, mode=None):
    """
    Collect the throughput response curves for each instrument configuration

    Parameters
    ----------
    type: String
        If set, get only the modes of the specified type:
        spectroscopy, photometry, or acquisition
    instrument: String
        If set, get only throughputs for the specified instrument.
    mode: String
        If set, get only throughputs for the specified mode and instrument.

    Returns
    -------
    throughputs: Dictionary
        Nested dictionary of shape throughputs[type][inst][mode][key][filter]
        containing the wavelength (um) and response throughputs.
        'key' refers to the apertures for photometry modes, and subarrays
        for others.

    Examples
    --------
    >>> import gen_tso.pandeia_io as jwst
    >>>
    >>> # All available throughputs
    >>> throughputs = jwst.get_throughputs()
    >>> # All throughputs for NIRSpec
    >>> throughputs = jwst.get_throughputs(inst='nirspec')
    """
    if mode is not None:
        if inst is None:
            msg = 'Also need to specify an instrument when requesting a specific mode'
            raise ValueError(msg)
        t_file = f'{ROOT}data/throughputs/throughputs_{inst}_{mode}.pickle'
        with open(t_file, 'rb') as handle:
            throughputs = pickle.load(handle)
        return throughputs

    if inst is not None:
        modes = get_modes(inst)
        throughputs = {}
        for mode in modes:
            t_file = f'{ROOT}data/throughputs/throughputs_{inst}_{mode}.pickle'
            with open(t_file, 'rb') as handle:
                data = pickle.load(handle)
            throughputs[mode] = data
        return throughputs

    if type is None:
        types = ['spectroscopy', 'photometry', 'acquisition']
    else:
        types = [type]
    instruments = get_instruments()

    throughputs = {}
    for obs_type,inst in product(types, instruments):
        modes = get_modes(inst, obs_type)
        if obs_type not in types or len(modes) == 0:
            continue
        if obs_type not in throughputs:
            throughputs[obs_type] = {}
        throughputs[obs_type][inst] = {}

        for mode in modes:
            t_file = f'{ROOT}data/throughputs/throughputs_{inst}_{mode}.pickle'
            with open(t_file, 'rb') as handle:
                data = pickle.load(handle)
            throughputs[obs_type][inst][mode] = data

    if type is not None:
        return throughputs[type]
    return throughputs


def generate_all_instruments():
    """
    A list of Detector() objects to keep the instrument observing mode
    configurations.

    Examples
    --------
    >>> from gen_tso.pandeia_io import _get_configs, generate_all_instruments
    >>> spec_insts = _get_configs(obs_type='spectroscopy')
    >>> photo_insts = _get_configs(obs_type='photometry')
    >>> acq_insts = _get_configs(obs_type='acquisition')
    >>> dets = generate_all_instruments()
    >>> det = dets[5]
    """
    detectors = []
    # Spectroscopic observing modes
    spec_insts = _get_configs(obs_type='spectroscopy')
    for inst in spec_insts:
        mode = inst['mode']
        dispersers = inst['dispersers']
        filters = inst['filters']
        subarrays = inst['subarrays']
        readouts = inst['readouts']
        apertures = inst['apertures']
        orders = inst['orders']
        constraints = inst['constraints']

        aperture_label = 'Aperture'
        if mode in ['lrsslit', 'lrsslitless']:
            disperser_label = 'Disperser'
            filter_label = ''
            filters = {'': ''}
            default_indices = 0, 0, 0, 0, 0
        if mode == 'mrs_ts':
            disperser_label = 'Wavelength Range'
            filter_label = ''
            filters = {'': ''}
            default_indices = 0, 0, 0, 0, 0
        if mode == 'lw_tsgrism':
            disperser_label = 'Grism'
            filter_label = 'Filter'
            default_indices = 0, 0, 3, 3, 0
        if mode == 'sw_tsgrism':
            aperture_label = 'PSF Type'
            disperser_label = 'Grism'
            filter_label = 'Filter'
            default_indices = 3, 0, 4, 0, 0
        if mode == 'bots':
            aperture_label = 'Slit'
            disperser_label = 'Grating'
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
            default_indices = 0, 0, 7, 4, 1
        if mode == 'soss':
            disperser_label = 'Disperser'
            filter_label = 'Filter'
            inst['mode_label'] = 'Single Object Slitless Spectroscopy (SOSS)'
            default_indices = 0, 0, 0, 0, 1

        det = Detector(
            mode,
            inst['mode_label'],
            inst['instrument'],
            inst['obs_type'],
            aperture_label,
            apertures,
            disperser_label,
            dispersers,
            filter_label,
            filters,
            subarrays,
            readouts,
            orders=orders,
            default_indices=default_indices,
            constraints=constraints,
        )
        detectors.append(det)

    # Photometry observing modes
    photo_insts = _get_configs(obs_type='photometry')
    for inst in photo_insts:
        mode = inst['mode']
        apertures = inst['apertures']
        dispersers = inst['dispersers']
        filters = inst['filters']
        subarrays = inst['subarrays']
        readouts = inst['readouts']
        constraints = inst['constraints']

        filter_label = 'Filter'
        if mode == 'lw_ts':
            aperture_label = 'LW Pupil'
            filter_label = 'LW Filter'
        elif mode == 'sw_ts':
            aperture_label = 'SW Pupil'
            filter_label = 'SW Filter'
        disperser_label = 'Disperser'
        dispersers = {'':''}
        default_indices = 0, 0, 0, 0, 0

        det = Detector(
            mode,
            inst['mode_label'],
            inst['instrument'],
            inst['obs_type'],
            aperture_label,
            apertures,
            disperser_label,
            dispersers,
            filter_label,
            filters,
            subarrays,
            readouts,
            default_indices=default_indices,
            constraints=constraints,
        )
        detectors.append(det)

        if mode in ['lw_ts', 'sw_ts']:
            det.pupils = inst['pupils']
            det.pupil_to_aperture = inst['pupil_to_aperture']
        if mode == 'sw_ts':
            det.pairings = inst['pairings']

    # Acquisition observing modes
    acq_insts = _get_configs(obs_type='acquisition')
    for inst in acq_insts:
        mode = inst['mode']
        aperture_label = 'Acquisition mode'
        apertures = inst['apertures']
        disperser_label = 'Disperser'
        dispersers = inst['dispersers']
        dispersers = {'':''}
        filter_label = 'Filter'
        filters = inst['filters']
        subarrays = inst['subarrays']
        readouts = inst['readouts']
        constraints = inst['constraints']

        if inst['instrument'] == 'MIRI':
            default_indices = 0, 0, 0, 5, 0
        if inst['instrument'] == 'NIRCam':
            default_indices = 0, 0, 0, 0, 0
        if inst['instrument'] == 'NIRISS':
            default_indices = 0, 0, 0, 0, 1
        if inst['instrument'] == 'NIRSpec':
            default_indices = 0, 0, 0, 1, 0

        det = Detector(
            mode,
            inst['mode_label'],
            inst['instrument'],
            inst['obs_type'],
            aperture_label,
            apertures,
            disperser_label,
            dispersers,
            filter_label,
            filters,
            subarrays,
            readouts,
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
    mode = mode.replace('mrs_ts', 'mrs')
    mode = mode.replace('w_tsgrism', 'w_grism')
    mode = mode.replace('w_ts', 'w_imaging')

    if mode == 'mrs':
        return f'tso{target}_{inst}_{mode}_{aperture}.pickle'
    elif mode == 'imaging_ts':
        return f'tso{target}_{inst}_{mode}_{filter}.pickle'
    elif mode in ['lrsslit', 'lrsslitless', 'soss']:
        return f'tso{target}_{inst}_{mode}.pickle'
    elif mode == 'bots':
        return f'tso{target}_{inst}_{mode}_{disperser}.pickle'
    elif mode == 'lw_grism':
        return f'tso{target}_{inst}_{mode}_{filter}.pickle'
    elif mode == 'sw_grism':
        return f'tso{target}_{inst}_{mode}_{subarray}_{filter}.pickle'


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
    if mode in ['lrsslit', 'lrsslitless']:
        return f'MIRI {mode.upper()}'
    if mode == 'imaging_ts':
        return f'MIRI {filter.upper()}'

    if mode == 'soss':
        order = f' O{order[0]}' if len(order)==1 else ''
        return f'NIRISS {mode.upper()} {subarray}{order}'

    if mode == 'lw_tsgrism':
        subarray = subarray.replace('grism', '').replace('_dhs', '')
        return f'NIRCam {filter.upper()} {subarray} {readout}'
    if mode == 'sw_tsgrism':
        return f'NIRCam {filter.upper()} {subarray} {readout}'
    if mode == 'lw_ts':
        subarray = subarray.replace('grism', '').replace('_dhs', '')
        return f'NIRCam imaging {filter.upper()} {subarray} {readout}'
    if mode == 'sw_ts':
        return f'NIRCam imaging {aperture} {filter.upper()} {subarray} {readout}'

    if mode == 'bots':
        if filter == 'f070lp':
            disperser = f'{disperser}/{filter}'
        return f'NIRSpec {disperser.upper()} {subarray} {readout}'


def make_saturation_label(
        mode, aperture, disperser, filter, subarray,
        order='', sed_label='',
    ):
    """
    Make a label of unique saturation setups to identify when and
    when not the saturation level can be estimated.
    """
    if mode == 'bots':
        sat_label = f'_{disperser}_{subarray}'
    elif mode == 'soss':
        order = f'_O{order[0]}' if len(order)==1 else ''
        sat_label = f'{order}'
    elif mode == 'sw_tsgrism':
        sat_label = f'_{aperture}_{subarray}'
    elif mode == 'sw_ts':
        sat_label = f'_{aperture}'
    elif mode == 'mrs_ts':
        sat_label = f'_{disperser}'
    elif mode == 'target_acq':
        sat_label = f'_{aperture}'
    else:
         sat_label = ''

    sat_label = f'{mode}_{filter}{sat_label}_{sed_label}'
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


def _load_flux_rate_splines(obs_label=None):
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
    >>> # Extract all splines, evaluate for one config:
    >>> flux_rate_splines, full_wells = jwst._load_flux_rate_splines()
    >>> obs_label = 'lw_tsgrism_f444w_phoenix_k5v'
    >>> spline = flux_rate_splines[obs_label]
    >>> print(spline(8.351), full_wells[obs_label])
    3.1140479065012263 58100.0
    >>>
    >>> # Extract a single config:
    >>> obs_label = 'lw_tsgrism_f444w_phoenix_k5v'
    >>> flux_rate_spline, full_well = jwst._load_flux_rate_splines(obs_label)
    >>> print(flux_rate_spline(8.351), full_well)
    3.1140479065012263 58100.0
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

    aper_modes = _photo_modes + ['sw_tsgrism', 'mrs_ts']

    flux_rates = {}
    full_wells = {}
    for inst, mode_rates in flux_rate_data.items():
        for mode, disp_rates in mode_rates.items():
            orders = ['1', '1 2'] if mode=='soss' else ['1']
            for disperser, filter_rates in disp_rates.items():
                for filter, sub_rates in filter_rates.items():
                    for subarray, sed_rates in sub_rates.items():
                        for order in orders:
                            if mode in aper_modes:
                                aperture = disperser
                            else:
                                aperture = ''
                            filter = filter.replace('None', '')
                            inst_label = make_saturation_label(
                                mode, aperture, disperser, filter,
                                subarray, order, ''
                            )
                            if obs_label is not None and inst_label != i_label:
                                continue
                            for sed_type in get_sed_types():
                                if sed_type not in sed_rates:
                                    continue
                                name = f'{sed_type}_names'
                                for i,rate in enumerate(sed_rates[sed_type]):
                                    log_rate = np.log10(rate)
                                    sed = sed_rates[name][i]
                                    label = f'{inst_label}{sed_type}_{sed}'
                                    if obs_label is None:
                                        flux_rates[label] = CubicSpline(mag, log_rate)
                                        full_wells[label] = sed_rates['full_well']
                                    elif label == obs_label:
                                        return CubicSpline(mag, log_rate), sed_rates['full_well']

    # Model not found
    if obs_label is not None:
        return None, None
    return flux_rates, full_wells


