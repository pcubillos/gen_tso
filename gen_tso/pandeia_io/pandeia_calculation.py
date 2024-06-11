# Copyright (c) 2024 Patricio Cubillos
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

from .pandeia_interface import (
    read_noise_variance,
    bin_search_exposure_time,
    saturation_level,
    make_scene,
    set_depth_scene,
    simulate_tso,
    tso_print,
)

class PandeiaCalculation():
    """
    A class to interface with the pandeia.engine package.

    Parameters
    ----------
    instrument: string
        The JWST instrument: nircam, niriss, nirspec, miri.
    mode: string
        Observing mode. If not set, default to the first item for
        each instrument from the list below.
        - nircam:
            ssgrism        spectroscopy
            target_acq     aquisition
        - niriss:
            soss           spectroscopy
            target_acq     aquisition
        - nirspec:
            bots           spectroscopy
            target_acq     aquisition
        - miri:
            lrsslitless    spectroscopy
            mrs_ts         spectroscopy
            target_acq     aquisition

    Examples
    --------
    >>> import gen_tso.pandeia_io as jwst
    >>> pando = jwst.PandeiaCalculation('nircam', 'target_acq')
    """
    def __init__(self, instrument, mode=None):
        if mode is None:
            # pick the most popular spectroscopic mode
            pass
        if mode == 'acquisition':
            mode = 'target_acq'
        self.telescope = 'jwst'
        self.instrument = instrument
        self.mode = mode
        self.calc = build_default_calc(
            self.telescope, self.instrument, self.mode,
        )

    def get_configs(self, output=None):
        """
        Print out or return the list of available configurations.

        Parameters
        ----------
        output: String
            The configuration variable to list. Select from:
            readouts, subarrays, filters, or dispersers.

        Returns
        -------
            outputs: 1D list of strings
            The list of available inputs for the requested variable.
        """
        ins_config = get_instrument_config(self.telescope, self.instrument)
        config = ins_config['mode_config'][self.mode]

        subarrays = config['subarrays']
        screen_output = f'subarrays: {subarrays}\n'

        if self.instrument == 'niriss':
            readouts = ins_config['readout_patterns']
        else:
            readouts = config['readout_patterns']
        screen_output += f'readout patterns: {readouts}\n'

        if self.instrument == 'nirspec':
            gratings_dict = ins_config['config_constraints']['dispersers']
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
        elif self.instrument=='nirspec' and output=='gratings':
            return gratings
        else:
            raise ValueError(f"Invalid config output: '{output}'")


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
            - phoenix or k93models: the model key (see load_sed_list)
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
        >>> import pandeia_interface as jwst

        >>> instrument = 'nircam'
        >>> mode = 'ssgrism'
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


    def get_saturation_values(
            self, disperser, filter, subarray, readout, ngroup=2,
            aperture=None, get_max=False,
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
        >>> mode = 'ssgrism'
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
            aperture=aperture,
        )
        brightest_pixel_rate, full_well = saturation_level(reports, get_max)
        return brightest_pixel_rate, full_well


    def perform_calculation(
            self, ngroup, nint,
            disperser=None, filter=None, subarray=None, readout=None,
            aperture=None,
        ):
        """
        Run pandeia's perform_calculation() for the given configuration
        (or set of configurations, see notes below).

        Parameter
        ----------
        ngroup: Integeer
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

        configs = product(
            aperture, disperser, filter, subarray, readout, nint, ngroup,
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
        aperture, disperser, filter, subarray, readout, nint, ngroup = params
        if aperture is not None:
            self.calc['configuration']['instrument']['aperture'] = aperture
        if disperser is not None:
            self.calc['configuration']['instrument']['disperser'] = disperser
        if readout is not None:
            self.calc['configuration']['detector']['readout_pattern'] = readout
        if subarray is not None:
            self.calc['configuration']['detector']['subarray'] = subarray
        if filter == '':
            self.calc['configuration']['instrument']['filter'] = None
        elif filter is not None:
            self.calc['configuration']['instrument']['filter'] = filter

        if self.instrument == 'niriss':
            self.calc['strategy']['order'] = 1
            # DataError: No mask configured for SOSS order 2.

        self.calc['configuration']['detector']['nexp'] = 1 # dither
        self.calc['configuration']['detector']['nint'] = nint
        self.calc['configuration']['detector']['ngroup'] = ngroup

        report = perform_calculation(self.calc)
        self.report = report
        return report

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
        filter: String
        subarray: String
        readout: String
        aperture: String

        Returns
        -------
        TBD

        Examples
        --------
        >>> import gen_tso.pandeia_io as jwst

        >>> instrument = 'nircam'
        >>> mode = 'ssgrism'
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
        if readout is None:
            readout = self.calc['configuration']['detector']['readout_pattern']
        if subarray is None:
            subarray = self.calc['configuration']['detector']['subarray']
        if filter is None:
            filter = self.calc['configuration']['instrument']['filter']

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
            subarray=None, readout=None, aperture=None,
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
        filter: String
        subarray: String
        readout: String
        aperture: String

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
        >>> import pandeia_interface as jwst
        >>> from pandeia_interface import set_depth_scene

        >>> # Set the stellar scene and transit:
        >>> scene = jwst.make_scene(
        >>>     sed_type='phoenix', sed_model='k5v',
        >>>     norm_band='2mass,ks', norm_magnitude=8.637,
        >>> )
        >>> transit_dur = 2.753
        >>> obs_dur = 7.1
        >>> obs_type = 'transit'
        >>> depth_model = np.loadtxt(
        >>>     '../planet_spectra/WASP80b_transit.dat', unpack=True)

        >>> # Set a NIRSpec observation
        >>> pando = jwst.PandeiaCalculation('nirspec', 'bots')
        >>> pando.calc['scene'] = [scene]
        >>> disperser = 'g395h'
        >>> filter = 'f290lp'
        >>> readout = 'nrsrapid'
        >>> subarray = 'sub2048'
        >>> ngroup = 16

        >>> tso = pando.tso_calculation(
        >>>     obs_type, transit_dur, obs_dur, depth_model,
        >>>     ngroup, disperser, filter, subarray, readout,
        >>> )

        >>> # Fluxes and Flux rates
        >>> col1, col2 = plt.cm.viridis(0.8), plt.cm.viridis(0.25)
        >>> plt.figure(0, (8.5, 4))
        >>> plt.clf()
        >>> plt.subplot(121)
        >>> plt.plot(tso['wl'], tso['flux_out'], c=col2, label='out of transit')
        >>> plt.plot(tso['wl'], tso['flux_in'], c=col1, label='in transit')
        >>> plt.legend(loc='best')
        >>> plt.xlabel('Wavelength (um)')
        >>> plt.ylabel('Collected flux (e-)')
        >>> plt.subplot(122)
        >>> plt.plot(tso['wl'], tso['flux_out']/tso['time_out'], c=col2, label='out of transit')
        >>> plt.plot(tso['wl'], tso['flux_in']/tso['time_in'], c=col1, label='in transit')
        >>> plt.legend(loc='best')
        >>> plt.xlabel('Wavelength (um)')
        >>> plt.ylabel('Flux rate (e-/s)')
        >>> plt.tight_layout()

        >>> # Model and instrument-observed transit depth spectrum
        >>> wl_depth, depth = depth_model
        >>> plt.figure(4)
        >>> plt.clf()
        >>> plt.plot(wl_depth, 100*depth, c='orange', label='model depth')
        >>> plt.plot(tso['wl'], 100*tso['depth_spectrum'], c='b', label='obs depth')
        >>> plt.legend(loc='best')
        >>> plt.xlim(2.75, 5.25)
        >>> plt.xlabel('Wavelength (um)')
        >>> plt.ylabel('Transit depth (%)')
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

        configs = product(
            aperture, disperser, filter, subarray, readout, ngroup,
        )
        tso = [
            self._tso_calculation(
                config, scene_in, scene_out, transit_dur, obs_dur,
            )
            for config in configs
        ]
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
        aperture, disperser, filter, subarray, readout, ngroup = config
        if aperture is not None:
            self.calc['configuration']['instrument']['aperture'] = aperture
        if disperser is not None:
            self.calc['configuration']['instrument']['disperser'] = disperser
        if readout is not None:
            self.calc['configuration']['detector']['readout_pattern'] = readout
        if subarray is not None:
            self.calc['configuration']['detector']['subarray'] = subarray
        if filter == '':
            self.calc['configuration']['instrument']['filter'] = None
        elif filter is not None:
            self.calc['configuration']['instrument']['filter'] = filter

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
        >>> pando = jwst.PandeiaCalculation('nircam', 'ssgrism')
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
        TBD

        Returns
        -------
        TBD

        Examples
        --------
        >>> TBD
        """
        return simulate_tso(self.tso, n_obs, resolution, bins, noiseless)

