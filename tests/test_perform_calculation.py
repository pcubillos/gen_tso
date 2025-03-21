# Copyright (c) 2025 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

import numpy as np
import gen_tso.pandeia_io as jwst


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Spectroscopy
def test_perform_calculation_miri_lrs():
    pando = jwst.PandeiaCalculation('miri', 'lrsslitless')
    pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)

    ngroup = 34
    nint = 1379

    report = pando.perform_calculation(
        ngroup, nint,
    )

    rep = report['scalar']
    # Differ from ETC because background is not exactly the same
    # ETC background 2572 e/s
    np.testing.assert_allclose(rep['sn'], 2216.8585)
    np.testing.assert_allclose(rep['extracted_flux'], 2789.8679)
    np.testing.assert_allclose(rep['extracted_noise'], 1.258478, rtol=1e-6)
    np.testing.assert_allclose(rep['brightest_pixel'], 28412.383)
    np.testing.assert_allclose(rep['fraction_saturation'], 0.793349, rtol=1e-6)


def test_perform_calculation_miri_mrs():
    pando = jwst.PandeiaCalculation('miri', 'mrs_ts')
    pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)

    ngroup = 50
    nint = 450

    report = pando.perform_calculation(
        ngroup, nint,
    )

    rep = report['scalar']
    # Differ from ETC because background is not exactly the same
    np.testing.assert_allclose(rep['sn'], 2133.817392)
    np.testing.assert_allclose(rep['extracted_flux'], 256.712868)
    np.testing.assert_allclose(rep['extracted_noise'], 0.120307, rtol=1e-5)
    np.testing.assert_allclose(rep['brightest_pixel'], 172.1364)
    np.testing.assert_allclose(rep['fraction_saturation'], 0.123334, rtol=1e-6)


def test_perform_calculation_nirspec_bots():
    pando = jwst.PandeiaCalculation('nirspec', 'bots')
    pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)
    pando.calc['background'] = 'ecliptic'

    ngroup = 14
    nint = 550

    report = pando.perform_calculation(
        ngroup, nint,
    )

    rep = report['scalar']
    # Differ from ETC because background is not exactly the same
    np.testing.assert_allclose(rep['sn'], 5909.431001)
    np.testing.assert_allclose(rep['extracted_flux'], 6184.164104)
    np.testing.assert_allclose(rep['extracted_noise'], 1.046491, rtol=1e-6)
    np.testing.assert_allclose(rep['brightest_pixel'], 3975.6643)
    np.testing.assert_allclose(rep['fraction_saturation'], 0.77238, rtol=1e-6)


def test_perform_calculation_niriss_soss():
    pando = jwst.PandeiaCalculation('niriss', 'soss')
    pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)

    subarray = 'substrip96'
    ngroup = 8
    nint = 350

    report = pando.perform_calculation(
        ngroup, nint, subarray=subarray,
    )

    rep = report['scalar']
    # Differ from ETC because background is not exactly the same
    np.testing.assert_allclose(rep['sn'], 11269.248655)
    np.testing.assert_allclose(rep['extracted_flux'], 28856.874712)
    np.testing.assert_allclose(rep['extracted_noise'], 2.560674, rtol=1e-6)
    np.testing.assert_allclose(rep['brightest_pixel'], 3191.468469)
    np.testing.assert_allclose(rep['fraction_saturation'], 0.785101, rtol=1e-6)


def test_perform_calculation_nircam_lw_tsgrism():
    pando = jwst.PandeiaCalculation('nircam', 'lw_tsgrism')
    pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)
    pando.calc['background'] = 'ecliptic'

    filter = 'f322w2'
    readout = 'rapid'
    subarray = 'subgrism64'
    ngroup = 50
    nint = 442

    report = pando.perform_calculation(
        ngroup, nint,
        filter=filter, subarray=subarray, readout=readout,
    )

    rep = report['scalar']
    # Differ from ETC because background is not exactly the same
    np.testing.assert_allclose(rep['sn'], 5663.236)
    np.testing.assert_allclose(rep['extracted_flux'], 5197.943)
    np.testing.assert_allclose(rep['extracted_noise'], 0.91784, rtol=1e-6)
    np.testing.assert_allclose(rep['brightest_pixel'], 2370.5251)
    np.testing.assert_allclose(rep['fraction_saturation'], 0.694858, rtol=1e-6)


def test_perform_calculation_nircam_sw_tsgrism():
    pando = jwst.PandeiaCalculation('nircam', 'sw_tsgrism')
    pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)

    aperture = 'dhs0spec8'
    filter = 'f150w'
    readout = 'rapid'
    subarray = 'sub260s4_8-spectra'
    ngroup = 80
    nint = 90

    report = pando.perform_calculation(
        ngroup, nint,
        filter=filter, subarray=subarray, readout=readout,
        aperture=aperture,
    )

    rep = report['scalar']
    # Differ from ETC because background is not exactly the same
    np.testing.assert_allclose(rep['sn'], 4108.861511)
    np.testing.assert_allclose(rep['extracted_flux'], 2326.930432)
    np.testing.assert_allclose(rep['extracted_noise'], 0.56632, rtol=1e-6)
    np.testing.assert_allclose(rep['brightest_pixel'], 485.7249)
    np.testing.assert_allclose(rep['fraction_saturation'], 0.718261, rtol=1e-6)



# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Photometry
def test_perform_calculation_nircam_sw_ts():
    pando = jwst.PandeiaCalculation('nircam', 'sw_ts')
    pando.set_scene('phoenix', 'k2v', '2mass,ks', 10.055)

    # pando.show_config()
    # pando.calc['strategy']
    filter = 'f212n'
    readout = 'rapid'
    subarray = 'sub160p'
    ngroup = 11
    nint = 500

    report = pando.perform_calculation(
        ngroup, nint,
        filter=filter, subarray=subarray, readout=readout,
    )

    rep = report['scalar']
    # Differ from ETC because background is not exactly the same
    np.testing.assert_allclose(rep['sn'], 10907.382669)
    np.testing.assert_allclose(rep['extracted_flux'], 95399.94145)
    np.testing.assert_allclose(rep['extracted_noise'], 8.746364, rtol=1e-6)
    np.testing.assert_allclose(rep['brightest_pixel'], 15469.766)
    np.testing.assert_allclose(rep['fraction_saturation'], 0.640836, rtol=1e-6)


def test_perform_calculation_nircam_lw_ts():
    pando = jwst.PandeiaCalculation('nircam', 'lw_ts')
    pando.set_scene('phoenix', 'k2v', '2mass,ks', 10.055)

    filter = 'f480m'
    readout = 'rapid'
    subarray = 'sub160p'
    ngroup = 11
    nint = 500

    report = pando.perform_calculation(
        ngroup, nint,
        filter=filter, subarray=subarray, readout=readout,
    )

    rep = report['scalar']
    # Differ from ETC because background is not exactly the same
    np.testing.assert_allclose(rep['sn'], 9885.335196)
    np.testing.assert_allclose(rep['extracted_flux'], 73134.053834)
    np.testing.assert_allclose(rep['extracted_noise'], 7.398237, rtol=1e-6)
    np.testing.assert_allclose(rep['brightest_pixel'], 14831.356)
    np.testing.assert_allclose(rep['fraction_saturation'], 0.782422, rtol=1e-6)


def test_perform_calculation_miri_imaging_ts():
    pando = jwst.PandeiaCalculation('miri', 'imaging_ts')
    pando.set_scene('phoenix', 'k2v', '2mass,ks', 10.055)

    filter = 'f560w'
    subarray = 'sub256'
    ngroup = 11
    nint = 500

    report = pando.perform_calculation(
        ngroup, nint,
        filter=filter, subarray=subarray,
    )

    rep = report['scalar']
    # Differ from ETC because background is not exactly the same
    np.testing.assert_allclose(rep['sn'], 9137.029043)
    np.testing.assert_allclose(rep['extracted_flux'], 156471.800829)
    np.testing.assert_allclose(rep['extracted_noise'], 17.12502, rtol=1e-6)
    np.testing.assert_allclose(rep['brightest_pixel'], 39420.387)
    np.testing.assert_allclose(rep['fraction_saturation'], 0.670673, rtol=1e-6)


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Acquisition
def test_perform_calculation_miri_target_acq():
    pando = jwst.PandeiaCalculation('miri', 'target_acq')
    pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)

    filter = 'f1000w'
    subarray = 'slitlessprism'
    ngroup = 10
    nint = 1

    report = pando.perform_calculation(
        ngroup, nint,
        filter=filter, subarray=subarray,
    )

    rep = report['scalar']
    # Differ from ETC because background is not exactly the same
    np.testing.assert_allclose(rep['sn'], 512.337046)
    np.testing.assert_allclose(rep['extracted_flux'], 451308.813965)
    np.testing.assert_allclose(rep['extracted_noise'], 880.882648, rtol=1e-6)
    np.testing.assert_allclose(rep['brightest_pixel'], 47220.953)
    np.testing.assert_allclose(rep['fraction_saturation'], 0.387804, rtol=1e-6)


def test_perform_calculation_nircam_target_acq():
    pando = jwst.PandeiaCalculation('nircam', 'target_acq')
    pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)

    filter = 'f335m'
    ngroup = 5
    nint = 1

    report = pando.perform_calculation(
        ngroup, nint,
        filter=filter, 
    )

    rep = report['scalar']
    # Differ from ETC because background is not exactly the same
    np.testing.assert_allclose(rep['sn'], 281.089612)
    np.testing.assert_allclose(rep['extracted_flux'], 2667698.826546)
    np.testing.assert_allclose(rep['extracted_noise'], 9490.563538, rtol=1e-6)
    np.testing.assert_allclose(rep['brightest_pixel'], 510198.7)
    np.testing.assert_allclose(rep['fraction_saturation'], 0.483993, rtol=1e-6)


def test_perform_calculation_niriss_target_acq():
    pando = jwst.PandeiaCalculation('niriss', 'target_acq')
    pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)

    readout = 'nis'
    ngroup = 3
    nint = 1
    report = pando.perform_calculation(ngroup, nint, readout=readout)

    rep = report['scalar']
    # Differ from ETC because background is not exactly the same
    np.testing.assert_allclose(rep['sn'], 406.413349)
    np.testing.assert_allclose(rep['extracted_flux'], 469783.817383)
    np.testing.assert_allclose(rep['extracted_noise'], 1155.926148, rtol=1e-6)
    np.testing.assert_allclose(rep['brightest_pixel'], 69039.62)
    np.testing.assert_allclose(rep['fraction_saturation'], 0.52355, rtol=1e-6)


def test_perform_calculation_nirspec_target_acq():
    pando = jwst.PandeiaCalculation('nirspec', 'target_acq')
    pando.set_scene('phoenix', 'k2v', 'gaia,g', 14.71)

    filter = 'f110w'
    subarray = 'sub32'
    readout = 'nrsrapid'
    ngroup = 3
    nint = 1

    report = pando.perform_calculation(
        ngroup, nint,
        filter=filter, subarray=subarray, readout=readout
    )

    rep = report['scalar']
    # Differ from ETC because background is not exactly the same
    np.testing.assert_allclose(rep['sn'], 76.970847)
    np.testing.assert_allclose(rep['extracted_flux'], 526950.164062)
    np.testing.assert_allclose(rep['extracted_noise'], 6846.100645, rtol=1e-6)
    np.testing.assert_allclose(rep['brightest_pixel'], 283694.34)
    np.testing.assert_allclose(rep['fraction_saturation'], 0.19588, rtol=1e-6)


