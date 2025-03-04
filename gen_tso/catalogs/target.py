# Copyright (c) 2025 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    'Target',
]

import numpy as np
import pyratbay.constants as pc
import pyratbay.atmosphere as pa

from . import utils as u


class Target():
    """
    A handy exoplanet target object.
    """
    def __init__(
        self, entry=None,
        host=None, mstar=np.nan, rstar=np.nan, teff=np.nan, logg_star=np.nan,
        metal_star=np.nan, ks_mag=np.nan, ra=np.nan, dec=np.nan,
        planet=None, mplanet=np.nan, rplanet=np.nan, period=np.nan, sma=np.nan,
        transit_dur=np.nan, ars=np.nan, rprs=np.nan, eq_temp=np.nan,
        is_confirmed=np.nan, is_min_mass=False, aliases=[],
    ):
        # Turn on if planet mass is RV's minimum mass: M*sin(i)
        self.is_min_mass = False
        if entry is None:
            if host is None or planet is None:
                raise ValueError('Must at least specify planet and host names')
            self.host = host
            self.mstar = mstar
            self.rstar = rstar
            self.teff = teff
            self.logg_star = logg_star
            self.metal_star = metal_star
            self.ks_mag = ks_mag
            self.ra = ra
            self.dec = dec

            self.planet = planet
            self.mplanet = mplanet
            self.rplanet = rplanet
            self.sma = sma
            self.period = period
            self.ars = ars
            self.rprs = rprs
            self.eq_temp = eq_temp
            self.transit_dur = transit_dur
            self.is_min_mass = is_min_mass

        else:
            self.host = entry['hostname']
            self.mstar = entry['st_mass']
            self.rstar = entry['st_rad']
            self.teff = entry['st_teff']
            self.logg_star = entry['st_logg']
            self.metal_star = entry['st_met']
            self.ks_mag = entry['sy_kmag']
            self.ra = entry['ra']
            self.dec = entry['dec']

            self.planet = entry['pl_name']
            self.mplanet = entry['pl_masse']
            if np.isnan(entry['pl_masse']) and np.isfinite(entry['pl_msinie']):
                self.mplanet = entry['pl_msinie']
                self.is_min_mass = True
            self.rplanet = entry['pl_rade']
            self.sma = entry['pl_orbsmax']
            self.period = entry['pl_orbper']
            self.ars = entry['pl_ratdor']
            self.rprs = entry['pl_ratror']
            self.transit_dur = entry['pl_trandur']
            self.eq_temp = entry['pl_eqt']

        self.is_transiting = np.isfinite(self.transit_dur)
        self.is_confirmed = is_confirmed
        self.aliases = aliases
        self._complete_values()

    def _complete_values(self):
        # Rank groups
        ars = (
            np.isnan(self.sma) +
            np.isnan(self.ars) +
            np.isnan(self.rstar)
        )
        rprs = (
            np.isnan(self.rplanet) +
            np.isnan(self.rprs) +
            np.isnan(self.rstar)
        )
        aperiod = (
            np.isnan(self.period) +
            np.isnan(self.sma) +
            np.isnan(self.mstar)
        )
        solve_order = np.argsort([ars, rprs, aperiod])
        for i in solve_order:
            if i == 0:
                solution = solve_a_rs(self.sma, self.rstar, self.ars)
                self.sma, self.rstar, self.ars = solution
            if i == 1:
                solution = solve_rprs(self.rplanet, self.rstar, self.rprs)
                self.rplanet, self.rstar, self.rprs = solution
            if i == 2:
                solution = solve_sma_period(self.period, self.sma, self.mstar)
                self.period, self.sma, self.mstar = solution
        self.equilibrium_temp()


    def equilibrium_temp(self, A=0.0, recirculation=1.0, force_update=False):
        """
        Parameters
        ----------
        A: Scalar
            Planetary bond albedo.
        recirculation: Scalar
            Day--night energy recirculation factor:
            recirculation = 0.5  no redistribution (total dayside reemission)
            recirculation = 1.0  good redistribution (4pi reemission)
        """
        eq_temp, _ = pa.equilibrium_temp(
            self.teff,
            self.rstar*pc.rsun,
            self.sma*pc.au,
            A=A, f=recirculation,
        )
        update = (
            np.isfinite(eq_temp) and
            (np.isnan(self.eq_temp) or force_update)
        )
        if update:
            self.eq_temp = eq_temp
        return eq_temp

    def copy_star(self, star):
        props = [
            'rstar',
            'mstar',
            'teff',
            'metal_star',
            'logg_star',
            'ks_mag',
        ]
        update_rplanet = (
            np.isfinite(self.rprs) and
            np.isfinite(self.rstar) and
            np.isfinite(star.rstar) and
            self.rstar != star.rstar
        )
        for prop in props:
            if np.isfinite(getattr(star, prop)):
                setattr(self, prop, getattr(star, prop))
        if update_rplanet:
            self.rplanet = self.rprs * self.rstar*pc.rsun / pc.rearth
        # Now complement all other values
        self._complete_values()

    def machine_readable_text(self):
        is_jwst_planet = is_jwst_host = ''
        if hasattr(self, 'is_jwst_planet'):
            is_jwst_planet = f'is_jwst_planet = {self.is_jwst_planet}\n'
            is_jwst_host = f'is_jwst_host = {self.is_jwst_host}\n'
        status = 'confirmed planet' if self.is_confirmed else 'candidate'
        mplanet_label = 'm_sini' if self.is_min_mass else 'mplanet'

        rstar = u.as_str(self.rstar, '.3f', 'np.nan')
        mstar = u.as_str(self.mstar, '.3f', 'np.nan')
        teff = u.as_str(self.teff, '.1f', 'np.nan')
        log_g = u.as_str(self.logg_star, '.2f', 'np.nan')
        metal = u.as_str(self.metal_star, '.2f', 'np.nan')
        ks_mag = u.as_str(self.ks_mag, '.2f', 'np.nan')
        ra = u.as_str(self.ra, '.4f', 'np.nan')
        dec = u.as_str(self.dec, '.4f', 'np.nan')

        rplanet = u.as_str(self.rplanet, '.3f', 'np.nan')
        mplanet = u.as_str(self.mplanet, '.3f', 'np.nan')
        rprs = u.as_str(self.rprs, '.3f', 'np.nan')
        a_rstar = u.as_str(self.ars, '.3f', 'np.nan')
        sma = u.as_str(self.sma, '.3f', 'np.nan')
        period = u.as_str(self.period, '.3f', 'np.nan')
        transit_dur = u.as_str(self.transit_dur, '.3f', 'np.nan')
        eq_temp = u.as_str(self.eq_temp, '.1f', 'np.nan')

        report = (
            f"host = {repr(self.host)}\n"
            f"planet = {repr(self.planet)}\n"
            f'aliases = {self.aliases}\n\n'

            f"rstar = {rstar}  # r_sun\n"
            f"mstar = {mstar}  # m_sun\n"
            f"teff = {teff} # K\n"
            f"log_g = {log_g}\n"
            f"metallicity = {metal}\n"
            f"ks_mag = {ks_mag}\n"
            f"ra = {ra}  # deg\n"
            f"dec = {dec}  # deg\n\n"

            f"rplanet = {rplanet}  # r_earth\n"
            f"{mplanet_label} = {mplanet}  # m_earth\n"
            f"transit_dur = {transit_dur}  # h\n"
            f"sma = {sma}  # AU\n"
            f"period = {period}  # d\n"
            f"eq_temp = {eq_temp}  # K\n"
            f"rprs = {rprs}\n"
            f"a_rstar = {a_rstar}\n"

            f'\nis_transiting = {self.is_transiting}\n'
            f'{is_jwst_planet}'
            f'{is_jwst_host}'
            f'status = {repr(status)}\n'
        )
        return report


    def __str__(self):
        rstar = u.as_str(self.rstar, '.3f', '---')
        mstar = u.as_str(self.mstar, '.3f', '---')
        teff = u.as_str(self.teff, '.1f', '---')
        logg = u.as_str(self.logg_star, '.2f', '---')
        metal = u.as_str(self.metal_star, '.2f', '---')
        ks_mag = u.as_str(self.ks_mag, '.2f', '---')
        ra = u.as_str(self.ra, '.3f', '---')
        dec = u.as_str(self.dec, '.3f', '---')

        rplanet = u.as_str(self.rplanet, '.3f', '---')
        mplanet = u.as_str(self.mplanet, '.3f', '---')
        rprs = u.as_str(self.rprs, '.3f', '---')
        ars = u.as_str(self.ars, '.3f', '---')
        sma = u.as_str(self.sma, '.3f', '---')
        period = u.as_str(self.period, '.3f', '---')
        t14 = u.as_str(self.transit_dur, '.3f', '---')
        eq_temp = u.as_str(self.eq_temp, '.1f', '---')

        mplanet_label = 'M*sin(i)' if self.is_min_mass else 'mplanet'
        report = (
            f"planet = {self.planet}\n"
            f"host = {self.host}\n\n"
            f"rstar = {rstar} r_sun\n"
            f"mstar = {mstar} m_sun\n"
            f"teff = {teff} K\n"
            f"log_g = {logg}\n"
            f"metallicity = {metal}\n"
            f"Ks_mag = {ks_mag}\n"
            f"RA = {ra} deg\n"
            f"dec = {dec} deg\n\n"

            f"rplanet = {rplanet} r_earth\n"
            f"{mplanet_label} = {mplanet} m_earth\n"
            f"transit_dur = {t14} h\n"
            f"sma = {sma} AU\n"
            f"period = {period} d\n"
            f"eq_temp = {eq_temp} K\n"
            f"rplanet/rstar = {rprs}\n"
            f"a/rstar = {ars}\n"
        )
        report += f'\nis transiting = {self.is_transiting}\n'
        if hasattr(self, 'is_jwst'):
            report += f'is JWST host = {self.is_jwst}\n'
        if np.isfinite(self.is_confirmed):
            status = 'confirmed' if self.is_confirmed else 'candidate'
            report += f'status = {status} planet\n'
        if len(self.aliases) > 0:
            report += f'aliases: {self.aliases}'
        return report


def format_nea_entry(entry):
    # Have TOI entries the same keys as PS entries:
    if 'toi' in entry.keys():
        entry['hostname'] = f"TOI-{entry['toipfx']}"
        entry['st_met'] = np.nan
        entry['sy_kmag'] = np.nan
        logg = entry['st_logg'] if entry['st_logg'] is not None else np.nan
        rstar = entry['st_rad'] if entry['st_rad'] is not None else np.nan
        entry['st_mass'] = 10**logg * (rstar*pc.rsun)**2 / pc.G / pc.msun

        entry['pl_name'] = f"TOI-{entry['toi']}"
        entry['pl_masse'] = np.nan
        entry['pl_orbsmax'] = np.nan
        entry['pl_ratdor'] = np.nan
        entry['pl_ratror'] = np.sqrt(entry.pop('pl_trandep')*pc.ppm)
        entry['pl_trandur'] = entry.pop('pl_trandurh')
    else:
        entry['pl_eqt'] = np.nan

    # Patch
    if entry['st_mass'] == 0.0:
        entry['st_mass'] = np.nan

    # Replace None with np.nan
    for key in entry.keys():
        if entry[key] is None:
            entry[key] = np.nan
    return entry


def solve_sma_period(period, sma, mstar):
    """
    Solve period-sma-mstar system of equations.

    Parameters
    ----------
    period: Float
        Orbital period (days).
    sma: Float
        Orbital semi-major axis (AU).
    mstar: Float
        Stellar mass (m_sun).
    """
    missing = (
        np.isnan(period) +
        np.isnan(sma) +
        np.isnan(mstar)
    )
    # Know everything or not enough:
    if missing != 1:
        return period, sma, mstar

    two_pi_G = 2.0*np.pi / np.sqrt(pc.G)
    if np.isnan(mstar):
        mstar = (sma*pc.au)**3.0 / (period*pc.day/two_pi_G)**2.0 / pc.msun
    elif np.isnan(period):
        period = np.sqrt((sma*pc.au)**3.0 / (mstar*pc.msun)) * two_pi_G / pc.day
    elif np.isnan(sma):
        sma = ((period*pc.day/two_pi_G)**2.0 * (mstar*pc.msun))**(1/3) / pc.au

    return period, sma, mstar


def solve_rprs(rplanet, rstar, rprs):
    """
    Solve planet radius -- stellar radius system of equations.

    Parameters
    ----------
    rplanet: Float
        Planet radius (r_earth).
    rstar: Float
        Stellar radius (r_sun).
    rprs: Float
        Planet--star radius ratio.
    """
    missing = (
        np.isnan(rplanet) +
        np.isnan(rstar) +
        np.isnan(rprs)
    )
    # Know everything or not enough:
    if missing != 1:
        return rplanet, rstar, rprs

    if np.isnan(rplanet):
        rplanet = rprs * (rstar*pc.rsun) / pc.rearth
    elif np.isnan(rstar):
        rstar = rplanet*pc.rearth / rprs / pc.rsun
    elif np.isnan(rprs):
        rprs = rplanet*pc.rearth / (rstar*pc.rsun)

    return rplanet, rstar, rprs


def solve_a_rs(sma, rstar, ars):
    """
    Solve semi-major axis -- stellar radius system of equations.

    Parameters
    ----------
    sma: Float
        Orbital semi-major axis (AU).
    rstar: Float
        Stellar radius (r_sun).
    ars: Float
        sma--rstar ratio.
    """
    missing = (
        np.isnan(sma) +
        np.isnan(ars) +
        np.isnan(rstar)
    )
    # Know everything or not enough:
    if missing != 1:
       return sma, rstar, ars

    if np.isnan(sma):
        sma = ars * rstar*pc.rsun / pc.au
    elif np.isnan(rstar):
        rstar = sma*pc.au / ars / pc.rsun
    elif np.isnan(ars):
        ars = sma*pc.au / (rstar*pc.rsun)

    return sma, rstar, ars


def solve_host(targets, n_dups):
    star = targets[0]
    # Identify conflicts
    masses = [target.mstar for target in targets]
    u_masses = np.unique([mass for mass in masses if np.isfinite(mass)])

    radii = [target.rstar for target in targets]
    u_radii = np.unique([radius for radius in radii if np.isfinite(radius)])

    teff = [target.teff for target in targets]
    u_teff = np.unique([temp for temp in teff if np.isfinite(temp)])

    log_g = [target.logg_star for target in targets]
    u_logg = np.unique([grav for grav in log_g if np.isfinite(grav)])

    metal = [target.metal_star for target in targets]
    u_metal = np.unique([met for met in metal if np.isfinite(met)])

    conflicts = (
        len(u_masses) > 1,
        len(u_radii) > 1,
        len(u_teff) > 1,
        len(u_logg) > 1,
        len(u_metal) > 1,
    )
    # Solve conflicts
    if np.any(conflicts):
        props = [
            'rstar',
            'mstar',
            'teff',
            'metal_star',
            'logg_star',
        ]
        completeness = [np.sum(missing_mask(target)) for target in targets]
        # Sort by completeness then by duplicity
        rank = np.lexsort((-np.array(n_dups), completeness))
        # Set a unique set of stellar properties, complement if needed
        star = targets[rank[0]]
        for i in rank[1:]:
            target = targets[i]
            for prop in props:
                update = (
                    np.isnan(getattr(star, prop)) and
                    np.isfinite(getattr(target, prop))
                )
                if update:
                    setattr(star, prop, getattr(target, prop))
    return star


def missing_mask(target):
    missing = np.array([
        np.isnan(target.rstar),
        np.isnan(target.mstar),
        np.isnan(target.teff),
        np.isnan(target.logg_star),
        np.isnan(target.metal_star),

        np.isnan(target.rplanet),
        np.isnan(target.mplanet),
        np.isnan(target.transit_dur),
        np.isnan(target.sma),
        np.isnan(target.period),
        np.isnan(target.ars),
        np.isnan(target.rprs),
    ])
    return missing


def rank_planets(target, alt_targets):
    rank = np.zeros(len(alt_targets))

    props = [
        'rprs',
        'ars',
        'rplanet',
        'mplanet',
        'rstar',
        'mstar',
        'transit_dur',
        'period',
        'sma',
        'teff',
        'metal_star',
        'logg_star',
    ]
    t = 0
    for t in range(len(alt_targets)):
        missing = missing_mask(target)
        # rank by fill as many missing values as possible
        for i,alt in enumerate(alt_targets):
            alt_miss = missing_mask(alt)
            rank[i] = np.sum(~alt_miss & missing)
        if np.all(rank==0):
            break

        # loop by rank
        i_rank = np.argsort(-rank)[0]
        alt = alt_targets[i_rank]
        # update props
        for prop in props:
            update = (
                np.isnan(getattr(target, prop)) and
                np.isfinite(getattr(alt, prop))
            )
            if update:
                setattr(target, prop, getattr(alt, prop))
                target._complete_values()
    return t


