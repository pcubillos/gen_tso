# Copyright (c) 2025 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    'PandeiaCalculation',
]

from collections.abc import Iterable
from itertools import product

import numpy as np
from pandeia.engine.calc_utils import (
    build_default_calc,
    get_instrument_config,
)
from pandeia.engine.perform_calculation import perform_calculation

from ..plotly_io.plots import response_boundaries
from .pandeia_interface import (
    read_noise_variance,
    bin_search_exposure_time,
    extract_flux_rate,
    estimate_flux_rate,
    groups_below_saturation,
    integration_time,
    make_scene,
    set_depth_scene,
    save_tso,
    simulate_tso,
    tso_print,
)
from .pandeia_defaults import (
    _spec_modes,
    _default_aperture_strategy,
    generate_all_instruments,
    get_sed_types,
    get_throughputs,
    get_detector,
)

try:
    detectors = generate_all_instruments()
    bots_throughputs = get_throughputs(inst='nirspec', mode='bots')
except:
    print(
        "\n~~~  WARNING  ~~~"
        "\n   Could not import pandeia.engine"
        "\n   Check that all databases are correctly installed:"
        "\n   https://pcubillos.github.io/gen_tso/install.html"
        "\n~~~  WARNING  ~~~\n\n"
    )

sed_types = get_sed_types()


class PandeiaCalculation():
    """
    A class to interface with the pandeia.engine package.

    Parameters
    ----------
    instrument: string
        The JWST instrument: nircam, niriss, nirspec, miri.
    mode: string
        Observing mode. If not set, default to the first item for
        each instrument from this list below:

        Spectroscopy: instrument  mode
                      miri        lrsslitless
                      miri        lrsslit
                      miri        mrs_ts
                      nircam      lw_tsgrism
                      nircam      sw_tsgrism
                      niriss      soss
                      nirspec     bots
        Photometry:
                      miri        imaging_ts
                      nircam      lw_ts
                      nircam      sw_ts
        Acquisition:
                      miri        target_acq
                      nircam      target_acq
                      niriss      target_acq
                      nirspec     target_acq

    Examples
    --------
    >>> import gen_tso.pandeia_io as jwst

    >>> pando = jwst.PandeiaCalculation('nirspec')
    >>> pando = jwst.PandeiaCalculation('nircam')
    >>> pando = jwst.PandeiaCalculation('nircam', 'sw_tsgrism')
    >>> pando = jwst.PandeiaCalculation('nircam', 'target_acq')
    """
    def __init__(self, instrument, mode=None):
        if mode is None:
            # Default to the most popular spectroscopic mode
            for detector in detectors:
                if detector.instrument.lower() == instrument:
                    mode = detector.mode
                    break
        if mode == 'acquisition':
            mode = 'target_acq'

        self.telescope = 'jwst'
        self.instrument = instrument
        self.mode = mode
        self.calc = build_default_calc(
            self.telescope, self.instrument, self.mode,
        )
        # Set default config for TSO:
        detector = get_detector(self.instrument, self.mode, detectors)
        self._detector = detector
        aperture = detector.default_aperture
        disperser = detector.default_disperser
        filter = detector.default_filter
        subarray = detector.default_subarray
        readout = detector.default_readout
        if self.mode == 'bots':
            disperser, filter = filter.split('/')

        if self.mode == 'target_acq':
            self.calc['configuration']['instrument']['aperture'] = aperture
        else:
            self.calc['configuration']['instrument']['disperser'] = disperser
        self.calc['configuration']['instrument']['filter'] = filter
        self.calc['configuration']['detector']['subarray'] = subarray
        self.calc['configuration']['detector']['readout_pattern'] = readout
        self._ensure_wl_reference_in_range()
        # Default aperture/sky annuli:
        if self.mode in _default_aperture_strategy:
            strat = _default_aperture_strategy[self.mode]
            self.calc['strategy']['aperture_size'] = strat['aperture_size']
            self.calc['strategy']['sky_annulus'] = strat['sky_annulus']


    def get_configs(
        self, output=None,
        # *, aperture=None, disperser=None,
        # filter=None, subarray=None, readout=None,
    ):
        """
        Print out or return the list of available configurations.

        Parameters
        ----------
        output: String
            The configuration variable to list. Select from:
            apertures, readouts, subarrays, filters, or dispersers.

        Returns
        -------
            outputs: 1D list of strings
            The list of available inputs for the requested variable.
        """
        detector = self._detector

        # TBD: collect constraints:
        #if output in detector.constraints:
        #    if constraint in detector.constraints[output]
        #        TBD

        screen_output = ''
        apertures = list(detector.apertures)
        screen_output += f'apertures: {apertures}\n'

        if self.mode == 'bots':
            gratings_dict = detector.constraints['filters']['dispersers']
            gratings = filters = dispersers = []
            for grating, filter_list in gratings_dict.items():
                for filter in filter_list:
                    gratings.append(f'{grating}/{filter}')
            screen_output += f'grating/filter pairs: {sorted(gratings)}\n'
        else:
            dispersers = list(detector.dispersers)
            filters = list(detector.filters)
            screen_output += f'dispersers: {dispersers}\n'
            screen_output += f'filters: {filters}\n'

        subarrays = list(detector.subarrays)
        screen_output += f'subarrays: {subarrays}\n'

        readouts = list(detector.readouts)
        screen_output += f'readout patterns: {readouts}\n'

        if output is None:
            print(screen_output)
        elif output == 'apertures':
            return apertures
        elif output == 'readouts':
            return readouts
        elif output == 'subarrays':
            return subarrays
        elif output == 'filters':
            return filters
        elif output == 'dispersers':
            return dispersers
        elif self.instrument=='nirspec' and output=='gratings':
            return gratings
        else:
            raise ValueError(f"Invalid config output: '{output}'")

    def set_config(
        self, disperser=None, filter=None,
        subarray=None, readout=None, aperture=None, order=None,
    ):
        """
        Set the instrumental configuration.

        Parameters
        ----------
        disperser: String
            Disperser/grating for the given instrument.
        filter: String
            Filter for the given instrument.
        subarray: String
            Subarray mode for the given instrument.
        readout: String
            Readout pattern mode for the given instrument.
        aperture: String
            Aperture configuration for the given instrument.
        order: Integer
            For NIRISS SOSS only, the spectral order.
            Other modes will ignore this argument.
        """
        # TBD: check the inputs are valid options
        config = self.calc['configuration']
        if aperture is not None:
            config['instrument']['aperture'] = aperture

        if disperser is not None:
            config['instrument']['disperser'] = disperser
        if config['instrument']['disperser'] == '':
            config['instrument']['disperser'] = None

        if filter is not None:
            config['instrument']['filter'] = filter
        if config['instrument']['filter'] == '':
            config['instrument']['filter'] = None

        if subarray is not None:
            config['detector']['subarray'] = subarray

        if readout is not None:
            config['detector']['readout_pattern'] = readout

        if order is not None and self.mode == 'soss':
            self.calc['strategy']['order'] = order


    def wl_ranges(self):
        """
        Get wavelength range covered by the instrument/mode
        """
        aperture = self.calc['configuration']['instrument']['aperture']
        disperser = self.calc['configuration']['instrument']['disperser']
        filter = self.calc['configuration']['instrument']['filter']
        conf = get_instrument_config('jwst', self.instrument)

        if self.mode == 'bots':
            subarray = self.calc['configuration']['detector']['subarray']
            filter = f'{disperser}/{filter}'
            throughput = bots_throughputs[subarray][filter]
            wl = throughput['wl']
            response = throughput['response']
            band_bounds = response_boundaries(wl, response, threshold=0.03)
            bounds = [tuple(np.round(bounds, 3)) for bounds in band_bounds]
            if len(bounds) == 1:
                bounds = bounds[0]
            return bounds

        if self.mode in ['lw_tsgrism', 'target_acq', 'lw_ts', 'sw_ts']:
            config = conf['range'][aperture][filter]
        elif self.mode in ['sw_tsgrism']:
            ranges = conf['range'][aperture]['dhs0_2']
            ranges.update(conf['range'][aperture]['dhs0_1'])
            config = ranges[filter]
        elif self.mode in ['lrsslit']:
            config = conf['range'][aperture]
        elif self.mode in ['lrsslitless', 'mrs_ts']:
            config = conf['range'][aperture][disperser]
        elif self.mode in ['imaging_ts']:
            config = conf['range'][aperture][filter]
        elif self.mode == 'soss':
            order = 1
            disperser = f'{disperser}_{order}'
            config = conf['range'][aperture][disperser][filter]

        wl_min = config['wmin']
        wl_max = config['wmax']
        return (wl_min, wl_max)

    def _ensure_wl_reference_in_range(self):
        """
        Make sure that reference wavelength is in the range of the detector
        """
        if self.mode not in _spec_modes:
            return
        if 'reference_wavelength' not in self.calc['strategy']:
            self.calc['strategy']['reference_wavelength'] = -1.0

        ref_wl = self.calc['strategy']['reference_wavelength']
        wl_ranges = self.wl_ranges()
        if isinstance(wl_ranges, tuple):
            wl_ranges = [wl_ranges]
        in_range = np.any([
            ran[0]<ref_wl and ref_wl<ran[1]
            for ran in wl_ranges
        ])
        if in_range:
            return

        subarray = self.calc['configuration']['detector']['subarray']
        if self.mode == 'bots' and subarray != 'sub2048':
            ref_wl = np.mean(wl_ranges[0])
        else:
            ref_wl = 0.5*(np.amax(wl_ranges) + np.amin(wl_ranges))
        self.calc['strategy']['reference_wavelength'] = np.round(ref_wl, 2)


    def set_scene(
            self, sed_type, sed_model, norm_band, norm_magnitude,
            background='ecliptic_low',
        ):
        """
        Set the stellar point-source scene to observe.

        Parameters
        ----------
        sed_type: String
            Type of model: 'phoenix', 'k93models', 'blackbody', or 'flat'
        sed_model:
            The SED model required for each sed_type:
            - phoenix or k93models: the model key (see get_sed_list)
            - blackbody: the effective temperature (K)
            - flat: the unit ('flam' or 'fnu')
        norm_band: String
            Band over which to normalize the spectrum.
        norm_magnitude: float
            Magnitude of the star at norm_band.
        background: String
            Set the background flux. Select from:
            'ecliptic_low', 'ecliptic_medium', 'ecliptic_high',
            'minzodi_low',  'minzodi_medium',  'minzodi_high'

        Examples
        --------
        >>> import gen_tso.pandeia_io as jwst

        >>> instrument = 'nircam'
        >>> mode = 'lw_tsgrism'
        >>> pando = jwst.PandeiaCalculation(instrument, mode)
        >>> pando.set_scene(
        >>>     sed_type='phoenix', sed_model='k5v',
        >>>     norm_band='2mass,ks', norm_magnitude=8.637,
        >>> )
        """
        scene = make_scene(sed_type, sed_model, norm_band, norm_magnitude)
        self.calc['scene'] = [scene]
        bkg, bkg_level = background.strip().split('_')
        self.calc['background'] = bkg
        self.calc['background_level'] = bkg_level


    def get_scene(self):
        """
        Get a flattened copy of the scene containing the SED and
        normalization properties.

        Returns
        -------
        scene_args: Dictionary
            Scene arguments.
        """
        scene = self.calc['scene'][0]['spectrum']
        scene_args = scene['sed'].copy()
        normalization = scene['normalization'].copy()
        scene_args['normalization'] = normalization.pop('type')
        scene_args.update(normalization)

        if 'spectrum' in scene_args:
            scene_args.pop('spectrum')

        return scene_args


    def get_saturation_values(
            self, disperser, filter, subarray, readout, ngroup=2,
            aperture=None, order=None, get_max=False,
        ):
        """
        Calculate the brightest-pixel rate (e-/s) and full_well (e-)
        for the current instrument and scene configuration, which once known,
        are sufficient to calculate the saturation level once the
        saturation  time is known.

        Examples
        --------
        >>> import gen_tso.pandeia_io as jwst

        >>> instrument = 'nircam'
        >>> mode = 'lw_tsgrism'
        >>> pando = jwst.PandeiaCalculation(instrument, mode)
        >>> pando.set_scene(
        >>>     sed_type='phoenix', sed_model='k2v',
        >>>     norm_band='2mass,ks', norm_magnitude=8.351,
        >>> )
        >>> brightest_pixel_rate, full_well = pando.get_saturation_values(
        >>>     disperser='grismr', filter='f444w',
        >>>     readout='rapid', subarray='subgrism64',
        >>> )

        >>> # Also works for Target Acquisition:
        >>> instrument = 'nircam'
        >>> mode = 'target_acq'
        >>> pando = jwst.PandeiaCalculation(instrument, mode)
        >>> pando.set_scene(
        >>>     sed_type='phoenix', sed_model='k5v',
        >>>     norm_band='2mass,ks', norm_magnitude=8.351,
        >>> )
        >>> brightest_pixel_rate, full_well = pando.get_saturation_values(
        >>>     disperser=None, filter='f335m',
        >>>     readout='rapid', subarray='sub32tats', ngroup=3,
        >>> )
        """
        # TBD: Automate here the group setting
        reports = self.perform_calculation(
            ngroup=ngroup, nint=1,
            disperser=disperser, filter=filter,
            subarray=subarray, readout=readout,
            aperture=aperture, order=order,
        )
        brightest_pixel_rate, full_well = extract_flux_rate(reports, get_max)
        return brightest_pixel_rate, full_well


    def saturation_fraction(
            self, fraction=None, ngroup=None,
            flux_rate=None, full_well=None,
        ):
        """
        Estimate the number of groups below a given saturation fraction or
        the saturation level for a given number of groups.

        Parameters
        ----------
        fraction: Float
            If not None, estimate the maximum number of groups below
            the given saturation fraction (percentage units).
        ngroup: Interger
            If not None, estimate the saturation fraction (%) for ngroup.
        flux_rate: Float
            e- per second rate at the brightest pixel.
        full_well: Float
            Number of e- counts to saturate the detector.

        Returns
        -------
        (if fraction argument is not None)
        ngroup: Integer
             Maximum number of groups to remain below the saturation fraction

        (if ngroup argument is not None)
        sat_fraction: Float
             Saturation level reached for given ngroup.

        Examples
        --------
        >>> import gen_tso.pandeia_io as jwst

        >>> instrument = 'nircam'
        >>> mode = 'lw_tsgrism'
        >>> pando = jwst.PandeiaCalculation(instrument, mode)
        >>> pando.set_scene(
        >>>     sed_type='phoenix', sed_model='k5v',
        >>>     norm_band='2mass,ks', norm_magnitude=8.351,
        >>> )
        >>>
        >>> # Get number of groups below 80% saturation:
        >>> ngroup = pando.saturation_fraction(fraction=80.0)
        >>> print(ngroup)
        104
        >>>
        >>> # Get saturation fraction (%) for 90 groups:
        >>> fraction = pando.saturation_fraction(ngroup=90)
        >>> print(fraction)
        68.6075132017307
        """
        if fraction is not None and ngroup is not None:
            raise ValueError('Only one of fraction and ngroup must be defined')
        if fraction is None and ngroup is None:
            raise ValueError('At least one of fraction and ngroup must be defined')

        config = self.calc['configuration']
        scene = self.calc['scene'][0]['spectrum']
        inst = config['instrument']['instrument']
        readout = config['detector']['readout_pattern']
        subarray = config['detector']['subarray']
        mode = config['instrument']['mode']
        aperture = config['instrument']['aperture']
        disperser = config['instrument']['disperser']
        filter = config['instrument']['filter']
        order = self.calc['strategy']['order'] if mode=='soss' else None
        if not isinstance(order, list):
            order = [order]

        from_report = (
            (flux_rate is None or full_well is None) and
            hasattr(self, 'report')
        )
        if from_report:
            report = self.report
            flux_rate, full_well = extract_flux_rate(report, get_max=True)

        can_guess = (
            (flux_rate is None or full_well is None) and
            scene['sed']['sed_type'] in sed_types and
            scene['normalization']['bandpass'] == '2mass,ks'
        )
        # If flux rate values are not known, try to guess them:
        if can_guess:
            sed_type = scene['sed']['sed_type']
            sed_model = scene['sed']['key']
            ks_mag = scene['normalization']['norm_flux']
            flux_rate, full_well = estimate_flux_rate(
                sed_type, sed_model, ks_mag,
                mode, aperture, disperser, filter, subarray, order,
            )

        if flux_rate is None or full_well is None:
            return

        # The calculation
        if fraction is not None:
            ngroup = groups_below_saturation(
                req_saturation=fraction,
                instrument=inst, subarray=subarray, readout=readout,
                flux_rate=flux_rate, full_well=full_well,
            )
            return ngroup

        if ngroup is not None:
            sat_time = integration_time(inst, subarray, readout, ngroup)
            sat_fraction = 100 * flux_rate * sat_time / full_well
            return sat_fraction


    def show_config(self):
        """
        Display a summary of the instrumental and scene configuration
        """
        config = self.calc['configuration']
        inst = config['instrument']['instrument']
        mode = config['instrument']['mode']
        aperture = config['instrument']['aperture']
        disperser = config['instrument']['disperser']
        filter = config['instrument']['filter']
        readout = config['detector']['readout_pattern']
        subarray = config['detector']['subarray']
        order_str = ''
        if mode == 'soss':
            order = self.calc['strategy']['order']
            order_str = f'    order = {order}\n'

        print(
            'Instrument configuration:\n'
            f'    instrument = {repr(inst)}\n'
            f'    mode = {repr(mode)}\n'
            f'    aperture = {repr(aperture)}\n'
            f'    disperser = {repr(disperser)}\n'
            f'    filter = {repr(filter)}\n'
            f'    readout pattern = {repr(readout)}\n'
            f'    subarray = {repr(subarray)}\n'
            f'{order_str}'
        )

        scene = self.get_scene()
        print('Scene configuration:')
        for key, val in scene.items():
            print(f'    {key} = {repr(val)}')


    def perform_calculation(
            self, ngroup, nint,
            disperser=None, filter=None, subarray=None, readout=None,
            aperture=None, order=None,
        ):
        """
        Run pandeia's perform_calculation() for the given configuration
        (or set of configurations, see notes below).

        Parameter
        ----------
        ngroup: Integer
            Number of groups per integration.  Must be >= 2.
        nint: Integer
            Number of integrations.
        disperser: String
            Disperser/grating for the given instrument.
        filter: String
            Filter for the given instrument.
        subarray: String
            Subarray mode for the given instrument.
        readout: String
            Readout pattern mode for the given instrument.
        aperture: String
            Aperture configuration for the given instrument.
        order: Integer
            For NIRISS SOSS only, the spectral order.
            Other modes will ignore this argument.

        Returns
        -------
        report: dict
            The Pandeia's report output for the given configuration.
            If there's more than one requested calculation, return a
            list of reports.

        Notes
        -----
        - Provide a list of values for any of these arguments to
          compute a batch of calculations.
        - To leave a config parameter unmodified, leave the respective
          argument as None.
          To set a config parameter as None, set the argument to ''.
        """
        if not isinstance(nint, Iterable):
            nint = [nint]
        if not isinstance(ngroup, Iterable):
            ngroup = [ngroup]
        if not isinstance(disperser, Iterable) or isinstance(disperser, str):
            disperser = [disperser]
        if not isinstance(filter, Iterable) or isinstance(filter, str):
            filter = [filter]
        if not isinstance(subarray, Iterable) or isinstance(subarray, str):
            subarray = [subarray]
        if not isinstance(readout, Iterable) or isinstance(readout, str):
            readout = [readout]
        if not isinstance(aperture, Iterable) or isinstance(aperture, str):
            aperture = [aperture]
        if not isinstance(order, Iterable) or isinstance(order, str):
            order = [order]

        configs = product(
            aperture, disperser, filter, subarray, readout, order, nint, ngroup,
        )

        reports = [
            self._perform_calculation(config)
            for config in configs
        ]
        if len(reports) == 1:
            return reports[0]
        return reports


    def _perform_calculation(self, params):
        """
        (the real function that) runs pandeia.
        """
        # Unpack configuration parameters
        aperture, disperser, filter, subarray, readout, order, nint, ngroup = params
        self.set_config(disperser, filter, subarray, readout, aperture, order)

        self.calc['configuration']['detector']['nexp'] = 1 # dither
        self.calc['configuration']['detector']['nint'] = nint
        self.calc['configuration']['detector']['ngroup'] = ngroup
        self._ensure_wl_reference_in_range()

        self.report = perform_calculation(self.calc)
        # In pandeia ver 2024.9+, NIRISS/SOSS returns float32 for wl arrays
        # need to convert to double for safety
        wl = np.array(self.report['1d']['extracted_flux'][0], dtype=float)
        self.report['1d']['extracted_flux'][0] = wl
        return self.report


    def calc_noise(
            self, obs_dur=None, ngroup=None,
            disperser=None, filter=None, subarray=None, readout=None,
            aperture=None, nint=None,
        ):
        """
        Run a Pandeia calculation and extract the observed wavelength,
        flux, and variances.

        Parameters
        ----------
        obs_dur: Float
            Duration of the observation.
        ngroup: Integer
            Number of groups per integrations
        disperser: String
            Disperser/grating for the given instrument.
        filter: String
            Filter for the given instrument.
        subarray: String
            Subarray mode for the given instrument.
        readout: String
            Readout pattern mode for the given instrument.
        aperture: String
            Aperture configuration for the given instrument.
        nint: Integer
            Number of integrations.

        Examples
        --------
        >>> import gen_tso.pandeia_io as jwst

        >>> instrument = 'nircam'
        >>> mode = 'lw_tsgrism'
        >>> pando = jwst.PandeiaCalculation(instrument, mode)
        >>> pando.set_scene(
        >>>     sed_type='phoenix', sed_model='k5v',
        >>>     norm_band='2mass,ks', norm_magnitude=8.637,
        >>> )
        >>> # Example TBD
        """
        if aperture is None:
            aperture = self.calc['configuration']['instrument']['aperture']
        if disperser is None:
            disperser = self.calc['configuration']['instrument']['disperser']
        if filter is None:
            filter = self.calc['configuration']['instrument']['filter']
        if subarray is None:
            subarray = self.calc['configuration']['detector']['subarray']
        if readout is None:
            readout = self.calc['configuration']['detector']['readout_pattern']

        if ngroup is None:
            raise TypeError("Missing required argument: 'ngroup'")
        # Check timing inputs are properly defined:
        if obs_dur is None and nint is None:
            raise ValueError(
                'Neither of obs_dur nor nint were provided'
            )
        if obs_dur is not None and nint is not None:
            raise ValueError(
                'Either provide obs_dur or nint, but not both'
            )
        if obs_dur is not None:
            nint, _ = bin_search_exposure_time(
                self.instrument, subarray, readout, ngroup, obs_dur,
            )
        if nint == 0:
            nint = 1
            # TBD: and raise a warning

        report = self.perform_calculation(
            ngroup, nint, disperser, filter, subarray, readout, aperture,
        )

        # Flux:
        measurement_time = report['scalar']['measurement_time']
        flux = report['1d']['extracted_flux'][1] * measurement_time
        wl = report['1d']['extracted_flux'][0]

        # Background variance:
        background_var = report['1d']['extracted_bg_only'][1] * measurement_time
        # Read noise variance:
        ins_config = get_instrument_config(self.telescope, self.instrument)
        read_noise = read_noise_variance(report, ins_config)
        npix = report['scalar']['extraction_area']
        read_noise_var = 2.0 * read_noise**2.0 * nint * npix
        # Pandeia (multiaccum) noise:
        shot_var = (report['1d']['extracted_noise'][1] * measurement_time)**2.0
        # Last-minus-first (LMF) noise:
        lmf_var = np.abs(flux) + background_var + read_noise_var

        variances = lmf_var, shot_var, background_var, read_noise_var

        return report, wl, flux, variances, measurement_time


    def tso_calculation(
            self, obs_type, transit_dur, obs_dur, depth_model,
            ngroup=None, disperser=None, filter=None,
            subarray=None, readout=None, aperture=None, order=None,
        ):
        """
        Run pandeia to simulate a transit/eclipse time-series observation

        Parameters
        ----------
        obs_type: String
            The observing geometry 'transit' or 'eclipse'.
        transit_dur: Float
            Duration of the transit or eclipse event in hours.
        obs_dur: Float
            Total duration of the observation (baseline plus transit
            or eclipse event) in hours.
        depth_model: list of two 1D array or a 2D array
            The transit or eclipse depth spectrum where the first item is
            the wavelength (um) and the second is the depth.
        ngroup: Integer
            Number of groups per integrations
        disperser: String
            Disperser/grating for the given instrument.
        filter: String
            Filter for the given instrument.
        subarray: String
            Subarray mode for the given instrument.
        readout: String
            Readout pattern mode for the given instrument.
        aperture: String
            Aperture configuration for the given instrument.
        order: Integer
            For NIRISS SOSS only, the spectral order.
            Other modes will ignore this argument.

        Returns
        -------
        tso: dict
            A dictionary containing the time-series observation data:
            - wl: instrumental wavelength sampling (microns)
            - depth_spectrum: Transit/eclipse depth spectrum at instrumental wl
            - time_in: In-transit/eclipse measuring time (seconds)
            - flux_in: In-transit/eclipse flux (e-)
            - var_in:  In-transit/eclipse variance
            - time_out: Out-of-transit/eclipse measuring time (seconds)
            - flux_out: Out-of-transit/eclipse flux (e-)
            - var_out:  Out-of-transit/eclipse
            - report_in:  In-transit/eclipse pandeia output report
            - report_out:  Out-of-transit/eclipse pandeia output report

        Examples
        --------
        >>> import gen_tso.pandeia_io as jwst
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> import pyratbay.constants as pc
        >>>
        >>> transit_dur = 2.131
        >>> obs_dur = 6.01
        >>> # Planet model: wl(um) and transit depth (no units):
        >>> depth_model = np.loadtxt('WASP80b_transit.dat', unpack=True)

        >>> # Set a NIRCam observation
        >>> pando = jwst.PandeiaCalculation('nircam', 'lw_tsgrism')
        >>> # The star:
        >>> pando.set_scene(
        >>>     sed_type='phoenix', sed_model='k5v',
        >>>     norm_band='2mass,ks', norm_magnitude=8.351,
        >>> )
        >>> # Take a look at the default cofiguration:
        >>> pando.calc['configuration']
        >>>
        >>> # Edit disperser, filter, readout, subarray, or aperture if needed
        >>> # See options with pando.get_configs()
        >>> ngroup = 90

        >>> # Run TSO:
        >>> obs_type = 'transit'
        >>> tso = pando.tso_calculation(
        >>>     obs_type, transit_dur, obs_dur, depth_model, ngroup,
        >>> )
        >>>
        >>> # Draw a simulated transit spectrum at selected resolution
        >>> obs_wl, obs_depth, obs_error, band_widths = jwst.simulate_tso(
        >>>     tso, resolution=250.0,
        >>> )
        >>> plt.figure(4)
        >>> plt.clf()
        >>> plt.plot(
        >>> tso['wl'], tso['depth_spectrum']/pc.percent,
        >>>     c='salmon', label='depth at instrumental resolution',
        >>> )
        >>> plt.errorbar(
        >>>     obs_wl, obs_depth/pc.percent, yerr=obs_error/pc.percent,
        >>>     fmt='o', ms=5, color='xkcd:blue', mfc=(1,1,1,0.85),
        >>>     label='simulated (noised up) transit spectrum',
        >>> )
        >>> plt.legend(loc='best')
        >>> plt.xlim(3.6, 5.05)
        >>> plt.ylim(2.88, 3.00)
        >>> plt.xlabel('Wavelength (um)')
        >>> plt.ylabel('Transit depth (%)')
        >>> plt.title('WASP-80 b / NIRCam F444W')
        >>>
        >>> # Fluxes and Flux rates
        >>> col1, col2 = plt.cm.viridis(0.8), plt.cm.viridis(0.25)
        >>> plt.figure(0, (8.5, 4))
        >>> plt.clf()
        >>> plt.subplot(121)
        >>> plt.plot(tso['wl'], tso['flux_out'], c=col2, label='out of transit')
        >>> plt.plot(tso['wl'], tso['flux_in'], c=col1, label='in transit')
        >>> plt.legend(loc='best')
        >>> plt.xlabel('Wavelength (um)')
        >>> plt.ylabel('Total collected flux (e-)')
        >>> plt.subplot(122)
        >>> plt.plot(tso['wl'], tso['flux_out']/tso['time_out'], c=col2, label='out of transit')
        >>> plt.plot(tso['wl'], tso['flux_in']/tso['time_in'], c=col1, label='in transit')
        >>> plt.legend(loc='best')
        >>> plt.xlabel('Wavelength (um)')
        >>> plt.ylabel('Flux rate (e-/s)')
        >>> plt.tight_layout()

        """
        if transit_dur >= obs_dur:
            raise ValueError(
                f'{obs_type} duration is longer than the observation duration'
            )
        # Scale in or out transit flux rates
        scene = self.calc['scene'][0]
        star_scene, depth_scene = set_depth_scene(scene, obs_type, depth_model)
        if obs_type == 'eclipse':
            scene_in = star_scene
            scene_out = depth_scene
        elif obs_type == 'transit':
            scene_in = depth_scene
            scene_out = star_scene

        if not isinstance(aperture, Iterable) or isinstance(aperture, str):
            aperture = [aperture]
        if not isinstance(disperser, Iterable) or isinstance(disperser, str):
            disperser = [disperser]
        if not isinstance(filter, Iterable) or isinstance(filter, str):
            filter = [filter]
        if not isinstance(subarray, Iterable) or isinstance(subarray, str):
            subarray = [subarray]
        if not isinstance(readout, Iterable) or isinstance(readout, str):
            readout = [readout]
        if not isinstance(order, Iterable) or isinstance(order, str):
            order = [order]
        if not isinstance(ngroup, Iterable):
            ngroup = [ngroup]

        configs = product(
            aperture, disperser, filter, subarray, readout, order, ngroup,
        )
        tso = []
        for config in configs:
            tso_run = self._tso_calculation(
                config, scene_in, scene_out, transit_dur, obs_dur,
            )
            tso_run['input_depth'] = depth_model
            tso.append(tso_run)
        if len(tso) == 1:
             tso = tso[0]
        self.tso = tso
        # Return scene to its previous state
        self.calc['scene'][0] = scene
        return tso


    def _tso_calculation(
            self, config, scene_in, scene_out, transit_dur, obs_dur,
        ):
        """
        (the real function that) runs a TSO calculation.
        """
        aperture, disperser, filter, subarray, readout, order, ngroup = config
        self.set_config(disperser, filter, subarray, readout, aperture, order)

        # Now that everything is defined I can turn durations into integs:
        inst = self.instrument
        subarray = self.calc['configuration']['detector']['subarray']
        readout = self.calc['configuration']['detector']['readout_pattern']
        if transit_dur is not None:
            transit_integs, _ = bin_search_exposure_time(
                inst, subarray, readout, ngroup, transit_dur,
            )
        if obs_dur is not None:
            obs_integs, _ = bin_search_exposure_time(
                inst, subarray, readout, ngroup, obs_dur,
            )
        if obs_integs <= transit_integs:
            raise ValueError(
                "Number of integrations for the total observation duration "
                "is <= in-transit integrations"
            )

        # Compute observed fluxes and noises:
        self.calc['scene'][0] = scene_in
        report_in, wl, flux_in, variances_in, time_in = self.calc_noise(
            nint=transit_integs, ngroup=ngroup,
        )
        var_lmf_in = variances_in[0]

        out_transit_integs = obs_integs - transit_integs
        self.calc['scene'][0] = scene_out
        report_out, wl, flux_out, variances_out, time_out = self.calc_noise(
            nint=out_transit_integs, ngroup=ngroup,
        )
        var_lmf_out = variances_out[0]

        # Mask out un-illumnated wavelengths (looking at you, G395H)
        mask = flux_in > 1e-6 * np.median(flux_in)
        wl = wl[mask]
        flux_in = flux_in[mask]
        flux_out = flux_out[mask]
        var_in = var_lmf_in[mask]
        var_out = var_lmf_out[mask]
        obs_depth = 1 - (flux_in/time_in) / (flux_out/time_out)

        tso = {
            'wl': wl,
            'depth_spectrum': obs_depth,
            'time_in': time_in,
            'flux_in': flux_in,
            'var_in': var_in,
            'time_out': time_out,
            'flux_out': flux_out,
            'var_out': var_out,
            'report_in': report_in,
            'report_out': report_out,
        }
        return tso


    def tso_print(self, format='rich'):
        """
        Print to screen a summary of the latest tso_calculation() ran.

        Parameters
        ----------
        format: String
            If 'rich' print with colourful text when there are warnings
            or errors in values.
            If None, print as plain text.

        Examples
        --------
        >>> import gen_tso.pandeia_io as jwst
        >>> import numpy as np

        >>> wl = np.logspace(0, 2, 1000)
        >>> depth = [wl, np.tile(0.03, len(wl))]
        >>> pando = jwst.PandeiaCalculation('nircam', 'lw_tsgrism')
        >>> pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)
        >>> tso = pando.tso_calculation(
        >>>     'transit', transit_dur=2.1, obs_dur=6.0, depth_model=depth,
        >>>     ngroup=130, readout='rapid', filter='f444w',
        >>> )
        >>> pando.tso_print()
        """
        tso_print(self.tso, format)


    def simulate_tso(
            self, n_obs=1, resolution=None, bins=None, noiseless=False,
        ):
        """
        Simulate a time-series observation spectrum with noise
        for the given number of observations and spectral sampling.

        Parameters
        ----------
        n_obs: integer
            Number of transit/eclipse observations
        resolution: float
            If not None, resample the spectrum at the given resolution.
        bins: integer
            If not None, bin the spectrum in between the edges given
            by this array.
        noiseless: Bool
            If True, do not add scatter noise to the spectrum.

        Returns
        -------
        bin_wl: 1D array
            Wavelengths of binned transit/eclipse spectrum.
        bin_spec: 1D array
            Binned simulated transit/eclipse spectrum.
        bin_err: 1D array
            Uncertainties of bin_spec.
        bin_widths: 1D or 2D array
            For spectra, the 1D bin widths of bin_wl.
            For photometry, an array of shape [1,2] with the (lower,upper)
            widths of the passband relative to bin_wl.
        """
        return simulate_tso(self.tso, n_obs, resolution, bins, noiseless)


    def save_tso(self, filename, tso=None, lightweight=True):
        """
        Save a TSO output to a pickle file.

        Parameters
        ----------
        filename: String
            The path where the TSO object will be saved.
        tso: dict
            The TSO object to be saved.  If not specified, assume the latest
            TSO computed by this PandeiaCalculation object.
        lightweight: bool
            If True, remove the '2d' and '3d' fields from 'report_out' and
            'report_in' to reduce the file size.
        """
        if tso is None:
            tso = self.tso
        save_tso(filename, tso, lightweight)

