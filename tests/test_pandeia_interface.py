# Copyright (c) 2025 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

import os
import pickle
import pytest

import numpy as np
from pandeia.engine.calc_utils import get_instrument_config

import gen_tso.pandeia_io as jwst
from gen_tso.utils import ROOT

os.chdir(ROOT+'../tests')
# See tests/mocks/make_mocks.py for mocked data setup.


#@pytest.mark.skip(reason='TBD')
@pytest.mark.parametrize(
    'configs',
    (
        # inst,  mode, aperture, expected_rn
        ('miri', 'mrs_ts', 'ch1', 32.6),
        ('miri', 'mrs_ts', 'ch2', 32.6),
        ('miri', 'mrs_ts', 'ch3', 32.6),
        ('miri', 'mrs_ts', 'ch4', 32.6),
        ('miri', 'lrsslitless', 'imager', 32.6),
        ('miri', 'imaging_ts', 'imager', 32.6),
        ('nircam', 'lw_tsgrism', 'lw', 8.98),
        ('nircam', 'sw_tsgrism', 'dhs0spec2', 21.3192),
        ('nircam', 'sw_tsgrism', 'dhs0spec4', 42.6384),
        ('nircam', 'sw_tsgrism', 'dhs0spec8', 85.2768),
        ('nircam', 'sw_tsgrism', 'dhs0bright', 10.6596),
        ('nircam', 'lw_ts', 'lw', 8.98),
        ('nircam', 'sw_ts', 'sw', 10.6596),
        ('nircam', 'sw_ts', 'wlp4', 10.6596),
        ('nircam', 'sw_ts', 'wlp8__ts', 10.6596),
        ('nircam', 'sw_ts', 'wlp8__tsgrism', 10.6596),
        ('nirspec', 'bots',  's1600a1', 9.799),
        ('niriss', 'soss', 'soss', 11.55),
    )
)
def test_read_noise_variance(configs):
    inst, mode, aperture, expected_rn = configs
    ins_config = get_instrument_config('jwst', inst)
    # mock report
    instrument = {
        'mode': mode,
        'aperture': aperture,
    }
    configuration = {'instrument': instrument}
    input = {'configuration': configuration}
    report = {'input': input}

    read_noise = jwst.read_noise_variance(report, ins_config)
    np.testing.assert_almost_equal(read_noise, expected_rn)


@pytest.mark.parametrize('nint', [1, 10, 100])
def test_exposure_time_nircam(nint):
    inst = 'nircam'
    subarray = 'subgrism64'
    readout = 'rapid'
    ngroup = 90
    exp_time = jwst.exposure_time(inst, subarray, readout, ngroup, nint)
    if nint == 1:
        expected_exposure = 31.00063
    elif nint == 10:
        expected_exposure = 310.0063
    elif nint == 100:
        expected_exposure = 3100.063
    np.testing.assert_allclose(exp_time, expected_exposure)


@pytest.mark.parametrize('nint', [1, 10, 100])
def test_exposure_time_miri_fastr1(nint):
    inst = 'miri'
    subarray = 'slitlessprism'
    readout = 'fastr1'
    ngroup = 30
    exp_time = jwst.exposure_time(inst, subarray, readout, ngroup, nint)
    if nint == 1:
        expected_exposure = 4.7712
    elif nint == 10:
        expected_exposure = 49.14336
    elif nint == 100:
        expected_exposure = 492.86496
    np.testing.assert_allclose(exp_time, expected_exposure)


@pytest.mark.parametrize('nint', [1, 10, 100])
def test_exposure_time_miri_slowr1(nint):
    inst = 'miri'
    subarray = 'full'
    readout = 'slowr1'
    ngroup = 30
    exp_time = jwst.exposure_time(inst, subarray, readout, ngroup, nint)
    if nint == 1:
        expected_exposure = 716.6976
    elif nint == 10:
        expected_exposure = 7381.98528
    elif nint == 100:
        expected_exposure = 74034.86208
    np.testing.assert_allclose(exp_time, expected_exposure)


def test_exposure_time_bad_subarray():
    inst = 'nircam'
    subarray = 'nope'
    readout = 'rapid'
    ngroup = 90
    nint = 1
    exp_time = jwst.exposure_time(inst, subarray, readout, ngroup, nint)
    assert exp_time == 0.0


def test_exposure_time_bad_readout():
    inst = 'nircam'
    subarray = 'subgrism64'
    readout = 'nope'
    ngroup = 90
    nint = 1
    exp_time = jwst.exposure_time(inst, subarray, readout, ngroup, nint)
    assert exp_time == 0.0


def test_bin_search_exposure_time_nircam():
    inst = 'nircam'
    subarray = 'subgrism64'
    readout = 'rapid'
    ngroup = 90
    obs_time = 6.0
    nint, exp_time = jwst.bin_search_exposure_time(
        inst, subarray, readout, ngroup, obs_time,
    )
    assert nint == 697
    np.testing.assert_allclose(exp_time, 21607.43911)


def test_bin_search_exposure_time_miri():
    inst = 'miri'
    subarray = 'slitlessprism'
    readout = 'fastr1'
    ngroup = 30
    obs_time = 6.0
    nint, exp_time = jwst.bin_search_exposure_time(
        inst, subarray, readout, ngroup, obs_time,
    )
    assert nint == 4382
    np.testing.assert_allclose(exp_time, 21604.15264)


def test_bin_search_exposure_time_bad_subarray():
    inst = 'nircam'
    subarray = 'nope'
    readout = 'rapid'
    ngroup = 90
    obs_time = 6.0
    nint, exp_time = jwst.bin_search_exposure_time(
        inst, subarray, readout, ngroup, obs_time,
    )
    assert exp_time == 0.0


def test_bin_search_exposure_time_bad_readout():
    inst = 'nircam'
    subarray = 'subgrism64'
    readout = 'nope'
    ngroup = 90
    obs_time = 6.0
    nint, exp_time = jwst.bin_search_exposure_time(
        inst, subarray, readout, ngroup, obs_time,
    )
    assert exp_time == 0.0


@pytest.mark.skip(reason='TBD')
def test_integration_time():
    pass


def test_extract_flux_rate_perform_calculation_single():
    with open('mocks/perform_calculation_nircam_lw_tsgrism.pkl', 'rb') as f:
        result = pickle.load(f)

    pixel_rate, full_well = jwst.extract_flux_rate(result)
    expected_rate = 1300.2144775390625
    expected_well = 58100.00422843957
    np.testing.assert_allclose(pixel_rate, expected_rate)
    np.testing.assert_allclose(full_well, expected_well)


def test_extract_flux_rate_perform_calculation_multi():
    with open('mocks/perform_calculation_miri_mrs_ts.pkl', 'rb') as f:
        result = pickle.load(f)

    pixel_rate, full_well = jwst.extract_flux_rate(result)
    expected_rate = [
        204.37779236, 109.17165375,  32.75593948,   4.55676889,
    ]
    expected_well = [
        193654.99564498, 193655.00540975, 193655.004117  , 193654.99798687,
    ]
    np.testing.assert_allclose(pixel_rate, expected_rate)
    np.testing.assert_allclose(full_well, expected_well)


def test_extract_flux_rate_tso_calculation_single():
    with open('mocks/tso_calculation_nircam_lw_tsgrism.pkl', 'rb') as f:
        tso = pickle.load(f)

    pixel_rate, full_well = jwst.extract_flux_rate(tso)
    expected_rate = [1261.23925781, 1300.21447754]
    expected_well = [58100.00016363, 58100.00129109]
    np.testing.assert_allclose(pixel_rate, expected_rate)
    np.testing.assert_allclose(full_well, expected_well)


def test_extract_flux_rate_tso_calculation_multi():
    with open('mocks/tso_calculation_miri_mrs_ts.pkl', 'rb') as f:
        tso = pickle.load(f)

    pixel_rate, full_well = jwst.extract_flux_rate(tso)
    expected_rate = [
        198.24740601, 105.90003204,  31.80119514,   4.46783495,
        204.37779236, 109.17165375,  32.75593948,   4.55676889,
    ]
    expected_well = [
        193654.99147199, 193654.99886718, 193654.98493823, 193654.99820043,
        193654.99564498, 193655.00540975, 193655.004117  , 193654.99798687,
    ]
    np.testing.assert_allclose(pixel_rate, expected_rate)
    np.testing.assert_allclose(full_well, expected_well)


@pytest.mark.skip(reason='TBD')
def test_extract_flux_rate_tso_calculation_get_max():
    pass


@pytest.mark.skip(reason='TBD')
def test_extract_flux_rate():
    pass


@pytest.mark.skip(reason='TBD')
def test_estimate_flux_rate():
    pass


@pytest.mark.skip(reason='TBD')
def test_groups_below_saturation():
    pass


@pytest.mark.skip(reason='TBD')
def test_get_sed_list():
    pass


@pytest.mark.skip(reason='TBD')
def test_find_closest_sed():
    pass


@pytest.mark.skip(reason='TBD')
def test_find_nearby_seds():
    pass


@pytest.mark.skip(reason='TBD')
def test_make_scene():
    pass


@pytest.mark.skip(reason='TBD')
def test_extract_sed():
    pass


@pytest.mark.skip(reason='TBD')
def test_blackbody_eclipse_depth():
    pass


@pytest.mark.skip(reason='TBD')
def test_set_depth_scene():
    pass


def test_get_bandwidths_nirspec_spectro():
    inst = 'nircam'
    mode = 'lw_ts'
    aper = 'lw'
    filter = 'f250m'
    wl0, bw, min_wl, max_wl = jwst.get_bandwidths(inst, mode, aper, filter)
    np.testing.assert_allclose(wl0, 2.5032492904902526)
    np.testing.assert_allclose(bw, 0.18093647144369174)
    np.testing.assert_allclose(min_wl, 2.4099999999997896)
    np.testing.assert_allclose(max_wl, 2.5949999999997693)


def test_get_bandwidths_miri_photo():
    inst = 'miri'
    mode = 'imaging_ts'
    aper = 'imager'
    filter = 'f560w'
    wl0, bw, min_wl, max_wl = jwst.get_bandwidths(inst, mode, aper, filter)
    np.testing.assert_allclose(wl0, 5.635006830413706)
    np.testing.assert_allclose(bw, 1.0006342537921697)
    np.testing.assert_allclose(min_wl, 5.029999999999501)
    np.testing.assert_allclose(max_wl, 6.204999999999371)


@pytest.mark.skip(reason='TBD')
def test_save_tso():
    pass


def test_simulate_tso_spectroscopy():
    pando = jwst.PandeiaCalculation('nircam', 'lw_tsgrism')
    pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)

    disperser='grismr'
    filter='f444w'
    readout='rapid'
    subarray='subgrism64'
    ngroup = 14

    transit_dur = 2.71
    obs_dur = 6.01
    obs_type = 'transit'
    model_path = f'{ROOT}data/models/WASP80b_transit.dat'
    depth_model = np.loadtxt(model_path, unpack=True)

    tso = pando.tso_calculation(
        obs_type, transit_dur, obs_dur, depth_model,
        ngroup, disperser, filter, subarray, readout,
    )

    sim = jwst.simulate_tso(tso, n_obs=1, resolution=50.0, noiseless=True)
    bin_wl, bin_spec, bin_err, bin_widths = sim

    #print(' '.join([f'{val:.10e},' for val in bin_spec]))
    #print(' '.join([f'{val:.10e},' for val in bin_err]))
    expected_wl = [
        3.76373232, 3.83976732, 3.91733838, 3.99647653, 4.07721343,
        4.15958137, 4.24361332, 4.32934288, 4.41680435, 4.50603273,
        4.59706369, 4.68993366, 4.7846798 , 4.88134   , 4.9642892,
    ]
    expected_depths = [
        2.9196140488e-02, 2.9133293446e-02, 2.9080333256e-02, 2.9072499494e-02,
        2.9099833873e-02, 2.9181661142e-02, 2.9334961283e-02, 2.9401298674e-02,
        2.9380711497e-02, 2.9308732552e-02, 2.9250589981e-02, 2.9310169251e-02,
        2.9331140031e-02, 2.9383422601e-02, 2.9416677476e-02,

    ]
    expected_errors = [
        2.5216620191e-03, 9.7522860111e-05, 3.0456858027e-05, 2.8706507253e-05,
        2.9208368639e-05, 2.9868139512e-05, 3.0767740663e-05, 3.3243328518e-05,
        3.5617857052e-05, 3.8147765558e-05, 4.0227910565e-05, 4.3083460408e-05,
        4.5791244642e-05, 4.9872653719e-05, 7.4568353154e-05,
    ]
    np.testing.assert_allclose(bin_wl, expected_wl)
    np.testing.assert_allclose(bin_spec, expected_depths)
    np.testing.assert_allclose(bin_err, expected_errors)

    # In case you want to see:
    if False:
        import matplotlib.pyplot as plt
        plt.figure(10)
        plt.clf()
        plt.plot(depth_model[0], depth_model[1], c='salmon')
        plt.xlim(2.0, 15)
        plt.errorbar(
            bin_wl, bin_spec, yerr=bin_err, xerr=bin_widths,
            fmt='o', color='xkcd:blue', mec='k', mew=1.0,
        )
        plt.xlim(3.7, 5.1)
        plt.ylim(0.0287, 0.0298)


def test_simulate_tso_photometry_nircam():
    pando = jwst.PandeiaCalculation('nircam', 'lw_ts')
    scene = jwst.make_scene('phoenix', 'k5v', '2mass,ks', 13.5)
    pando.calc['scene'] = [scene]

    filters = pando.get_configs('filters')
    subarray = 'sub160p'
    ngroup = 6
    model_path = f'{ROOT}data/models/WASP80b_transit.dat'
    depth_model = np.loadtxt(model_path, unpack=True)

    transit_dur = 2.71
    obs_dur = 6.0
    obs_type = 'transit'

    photo = []
    depths = np.zeros(len(filters))
    errors = np.zeros(len(filters))
    wl0 = np.zeros(len(filters))
    for i,filter in enumerate(filters):
        tso = pando.tso_calculation(
            obs_type, transit_dur, obs_dur, depth_model,
            ngroup, filter=filter, subarray=subarray,
        )
        sim = jwst.simulate_tso(tso, n_obs=1, noiseless=True)
        bin_wl, bin_spec, bin_err, bin_widths = sim
        wl0[i] = bin_wl[0]
        depths[i] = bin_spec[0]
        errors[i] = bin_err[0]
        photo.append(sim)

    #print(' '.join([f'{val:.10e},' for val in depths]))
    #print(' '.join([f'{val:.10e},' for val in errors]))
    expected_wl = [
       2.50043643, 2.74658693, 2.98917097, 3.10777101, 3.23666866,
       3.35436689, 3.52803571, 3.61482424, 4.05288935, 4.07232474,
       4.27644969, 4.33293162, 4.62826397, 4.65428413, 4.7078885 ,
       4.80988318,
    ]
    expected_depths = [
        2.9602355004e-02, 2.9673371794e-02, 2.9704446586e-02, 2.9605678087e-02,
        2.9750499043e-02, 2.9726816999e-02, 2.9526299697e-02, 2.9459430372e-02,
        2.9076237238e-02, 2.9157603316e-02, 2.9342234571e-02, 2.9243180952e-02,
        2.9275989763e-02, 2.9279518701e-02, 2.9312241722e-02, 2.9349633790e-02,

    ]
    expected_errors = [
        1.3840358487e-04, 7.4326836871e-05, 1.2111463701e-04, 5.6637595030e-05,
        5.8597305401e-04, 1.2942307791e-04, 8.9283843176e-05, 1.4150529160e-04,
        7.6607387286e-04, 1.6205714300e-04, 2.6322708714e-04, 1.1749776751e-04,
        3.5311120708e-04, 1.3152440611e-03, 1.4757125324e-03, 3.2309278153e-04,
    ]
    np.testing.assert_allclose(wl0, expected_wl)
    np.testing.assert_allclose(depths, expected_depths)
    np.testing.assert_allclose(errors, expected_errors)

    # In case you want to see:
    if False:
        import matplotlib.pyplot as plt
        plt.figure(10)
        plt.clf()
        plt.plot(depth_model[0], depth_model[1], c='0.6')
        plt.xlim(2.3, 6.0)
        plt.ylim(0.0283, 0.0304)
        for i,sim in enumerate(photo):
            bin_wl, bin_spec, bin_err, bin_widths = sim
            col = plt.cm.rainbow(i/len(filters))
            plt.errorbar(
                bin_wl, bin_spec, yerr=bin_err, xerr=bin_widths.T/2,
                fmt='o', color=col, mec='k', mew=1.0, label=filters[i],
            )
        plt.legend(loc='best')



def test_simulate_tso_photometry_miri():
    # With MIRI / imaging
    pando = jwst.PandeiaCalculation('miri', 'imaging_ts')
    scene = jwst.make_scene('phoenix', 'k5v', '2mass,ks', 13.5)
    pando.calc['scene'] = [scene]

    model_path = f'{ROOT}data/models/WASP80b_transit.dat'
    depth_model = np.loadtxt(model_path, unpack=True)
    transit_dur = 2.71
    obs_dur = 6.0
    obs_type = 'transit'
    ngroup = 6
    photo = []
    filters = ['f560w', 'f770w', 'f1000w', 'f1130w', 'f1280w']

    depths = np.zeros(len(filters))
    errors = np.zeros(len(filters))
    wl0 = np.zeros(len(filters))
    for i,filter in enumerate(filters):
        tso = pando.tso_calculation(
            obs_type, transit_dur, obs_dur, depth_model,
            ngroup, filter=filter,
        )
        sim = jwst.simulate_tso(tso, n_obs=1, noiseless=True)
        bin_wl, bin_spec, bin_err, bin_widths = sim
        wl0[i] = bin_wl[0]
        depths[i] = bin_spec[0]
        errors[i] = bin_err[0]
        photo.append(sim)

    #print(' '.join([f'{val:.10e},' for val in depths]))
    #print(' '.join([f'{val:.10e},' for val in errors]))
    expected_wl = 5.60136086,  7.53420869,  9.88160076, 11.2961649 , 12.70594686
    expected_depths = [
        2.9531941371e-02, 2.9716732792e-02, 2.9197826341e-02,
        2.9140093469e-02, 2.9250072922e-02,
    ]
    expected_errors = [
        2.5261903335e-04, 2.5860955286e-04, 5.1512651158e-04,
        1.6119916090e-03, 1.1052531125e-03,
    ]
    np.testing.assert_allclose(wl0, expected_wl)
    np.testing.assert_allclose(depths, expected_depths)
    np.testing.assert_allclose(errors, expected_errors)

    # In case you want to see:
    if False:
        import matplotlib.pyplot as plt
        plt.figure(10)
        plt.clf()
        plt.plot(depth_model[0], depth_model[1], c='0.6')
        plt.xlim(2.0, 15)
        for i,sim in enumerate(photo):
            bin_wl, bin_spec, bin_err, bin_widths = sim
            col = plt.cm.rainbow(i/len(filters))
            plt.errorbar(
                bin_wl, bin_spec, yerr=bin_err, xerr=bin_widths.T/2,
                fmt='o', color=col, mec='k', mew=1.0, label=filters[i],
            )
        plt.legend(loc='best')


def test__print_pandeia_exposure_config():
    with open('mocks/perform_calculation_nircam_lw_tsgrism.pkl', 'rb') as f:
        result = pickle.load(f)
    config = result['input']['configuration']
    text = jwst._print_pandeia_exposure(config=config)
    assert text == 'Exposure time: 701.41 s (0.19 h)'


def test__print_pandeia_exposure_vals():
    inst = 'nircam'
    subarray = 'subgrism64'
    readout = 'rapid'
    ngroup = 90
    nint = 150
    text = jwst._print_pandeia_exposure(inst, subarray, readout, ngroup, nint)
    assert text == 'Exposure time: 4650.09 s (1.29 h)'


@pytest.mark.parametrize('format', [None, 'html', 'rich'])
def test__print_pandeia_exposure_format(format):
    with open('mocks/perform_calculation_nircam_lw_tsgrism.pkl', 'rb') as f:
        result = pickle.load(f)
    config = result['input']['configuration']
    text = jwst._print_pandeia_exposure(config=config, format=format)
    assert text == 'Exposure time: 701.41 s (0.19 h)'


def test__print_pandeia_saturation_perform_calc():
    # HERE!
    with open('mocks/perform_calculation_nircam_lw_tsgrism.pkl', 'rb') as f:
        result = pickle.load(f)

    text = jwst._print_pandeia_saturation(reports=[result], format=None)
    expected_text = """Max fraction of saturation: 1.5%
ngroup below 80% saturation: 104
ngroup below 100% saturation: 131"""
    assert text == expected_text


def test__print_pandeia_saturation_tso_calc():
    with open('mocks/tso_calculation_miri_mrs_ts.pkl', 'rb') as f:
        tso = pickle.load(f)

    text = jwst._print_pandeia_saturation(reports=tso, format=None)
    expected_text = """Max fraction of saturation: 73.2%
ngroup below 80% saturation: 273
ngroup below 100% saturation: 341"""
    assert text == expected_text


def test__print_pandeia_saturation_values():
    pixel_rate, full_well = 1396.5528564453125, 58100.002280515175
    inst = 'nircam'
    subarray = 'subgrism64'
    readout = 'rapid'
    ngroup = 90
    text = jwst._print_pandeia_saturation(
        inst, subarray, readout, ngroup, pixel_rate, full_well,
        format=None,
    )
    expected_text = """Max fraction of saturation: 73.7%
ngroup below 80% saturation: 97
ngroup below 100% saturation: 122"""
    assert text == expected_text


def test__print_pandeia_saturation_req_saturation():
    pixel_rate, full_well = 1396.5528564453125, 58100.002280515175
    inst = 'nircam'
    subarray = 'subgrism64'
    readout = 'rapid'
    ngroup = 90
    text = jwst._print_pandeia_saturation(
        inst, subarray, readout, ngroup, pixel_rate, full_well,
        format=None,
        req_saturation=70.0,
    )
    expected_text = """Max fraction of saturation: 73.7%
ngroup below 70% saturation: 85
ngroup below 100% saturation: 122"""
    assert text == expected_text


@pytest.mark.skip(reason='TBD')
def test__print_pandeia_stats():
    pass


def test__print_pandeia_report_perform_calculation_single():
    with open('mocks/perform_calculation_nircam_lw_tsgrism.pkl', 'rb') as f:
        result = pickle.load(f)

    report = jwst._print_pandeia_report([result], format=None)
    expected_report = """Exposure time: 701.41 s (0.19 h)
Max fraction of saturation: 1.5%
ngroup below 80% saturation: 104
ngroup below 100% saturation: 131

Signal-to-noise ratio        363.4
Extracted flux              1815.8  e-/s
Flux standard deviation        5.0  e-/s
Brightest pixel rate        1300.2  e-/s

Integrations:                         683
Duty cycle:                          0.66
Total exposure time:                701.4  s
First--last dt per exposure:        701.4  s
Reset--last dt per integration:     232.6  s

Reference wavelength:                    4.46  microns
Area of extraction aperture:             4.76  pixels
Area of background measurement:           6.3  pixels
Background surface brightness:            0.3  MJy/sr
Total sky flux in background aperture:   4.94  e-/s
Total flux in background aperture:      59.59  e-/s
Background flux fraction from scene:     0.92
Number of cosmic rays:      0.0002  events/pixel/read"""
    assert report == expected_report



def test__print_pandeia_report_perform_calculation_multi():
    with open('mocks/perform_calculation_miri_mrs_ts.pkl', 'rb') as f:
        result = pickle.load(f)

    report = jwst._print_pandeia_report(result, format=None)
    expected_report = """Exposure time: 27858.63 s (7.74 h)
Max fraction of saturation: 73.2%
ngroup below 80% saturation: 273
ngroup below 100% saturation: 341

Signal-to-noise ratio       1334.2    1265.2     960.1     179.2
Extracted flux               225.1     147.4      46.2       2.0  e-/s
Flux standard deviation        0.2       0.1       0.0       0.0  e-/s
Brightest pixel rate         204.4     109.2      32.8       4.6  e-/s

Integrations:                          40
Duty cycle:                          0.99
Total exposure time:              27858.6  s
First--last dt per exposure:      27858.6  s
Reset--last dt per integration:   27639.4  s

Reference wavelength:                    5.35   8.15  12.50  19.29  microns
Area of extraction aperture:             8.20   5.21   2.98   1.61  pixels
Area of background measurement:          79.4   50.5   28.9   15.6  pixels
Background surface brightness:            1.0   10.9   43.0  165.5  MJy/sr
Total sky flux in background aperture:   0.29   0.82   2.98   2.44  e-/s
Total flux in background aperture:       0.85   1.47   3.19   2.54  e-/s
Background flux fraction from scene:     0.66   0.45   0.06   0.04
Number of cosmic rays:      0.3122  events/pixel/read"""
    assert report == expected_report


def test__print_pandeia_report_tso_calculation_single():
    with open('mocks/tso_calculation_nircam_lw_tsgrism.pkl', 'rb') as f:
        result = pickle.load(f)

    report = jwst._print_pandeia_report(result, format=None)
    expected_report = """Exposure time: 21607.44 s (6.00 h)
Max fraction of saturation: 68.6%
ngroup below 80% saturation: 104
ngroup below 100% saturation: 131

Signal-to-noise ratio       3241.5
Extracted flux              1761.3  e-/s
Flux standard deviation        0.5  e-/s
Brightest pixel rate        1261.2  e-/s

                               in-transit  out-transit
Integrations:                         244      453
Duty cycle:                          0.99     0.99
Total exposure time:               7564.2  14043.3  s
First--last dt per exposure:       7564.2  14043.3  s
Reset--last dt per integration:    7396.7  13732.4  s

Reference wavelength:                    4.46  microns
Area of extraction aperture:             4.76  pixels
Area of background measurement:           6.3  pixels
Background surface brightness:            0.3  MJy/sr
Total sky flux in background aperture:   4.94  e-/s
Total flux in background aperture:      57.95  e-/s
Background flux fraction from scene:     0.91
Number of cosmic rays:      0.0072  events/pixel/read"""
    assert report == expected_report


def test__print_pandeia_report_tso_calculation_multi():
    with open('mocks/tso_calculation_miri_mrs_ts.pkl', 'rb') as f:
        tso = pickle.load(f)

    report = jwst._print_pandeia_report(tso, format=None)
    expected_report = """Exposure time: 20893.28 s (5.80 h)
Max fraction of saturation: 73.2%
ngroup below 80% saturation: 273
ngroup below 100% saturation: 341

Signal-to-noise ratio        657.0     623.0     472.3      87.5
Extracted flux               218.4     142.9      44.8       1.9  e-/s
Flux standard deviation        0.3       0.2       0.1       0.0  e-/s
Brightest pixel rate         198.2     105.9      31.8       4.5  e-/s

                               in-transit  out-transit
Integrations:                          10       20
Duty cycle:                          0.99     0.99
Total exposure time:               6962.6  13927.9  s
First--last dt per exposure:       6962.6  13927.9  s
Reset--last dt per integration:    6909.8  13819.7  s

Reference wavelength:                    5.35   8.15  12.50  19.29  microns
Area of extraction aperture:             8.20   5.21   2.98   1.61  pixels
Area of background measurement:          79.4   50.5   28.9   15.6  pixels
Background surface brightness:            1.0   10.9   43.0  165.5  MJy/sr
Total sky flux in background aperture:   0.29   0.82   2.98   2.44  e-/s
Total flux in background aperture:       0.83   1.45   3.18   2.54  e-/s
Background flux fraction from scene:     0.65   0.44   0.06   0.04
Number of cosmic rays:      0.3122  events/pixel/read"""
    assert report == expected_report


def test_tso_print_plain(capsys):
    with open('mocks/tso_calculation_nircam_lw_tsgrism.pkl', 'rb') as f:
        result = pickle.load(f)
    # Make some troublesome figures
    result['report_in']['scalar']['duty_cycle'] = 0.25
    result['report_in']['input']['configuration']['detector']['ngroup'] = 130
    result['report_out']['scalar']['duty_cycle'] = 0.25
    result['report_out']['input']['configuration']['detector']['ngroup'] = 130

    jwst.tso_print(result, format=None)
    captured = capsys.readouterr()
    expected_report = (
        "Exposure time: 31103.65 s (8.64 h)\r\n"
        "Max fraction of saturation: 99.1%\r\n"
        "ngroup below 80% saturation: 104\r\n"
        "ngroup below 100% saturation: 131\r\n"
        "\r\n"
        "Signal-to-noise ratio       3241.5\r\n"
        "Extracted flux              1761.3  e-/s\r\n"
        "Flux standard deviation        0.5  e-/s\r\n"
        "Brightest pixel rate        1261.2  e-/s\r\n"
        "\r\n"
        "                               in-transit  out-transit\r\n"
        "Integrations:                         244      453\r\n"
        "Duty cycle:                          0.25     0.25\r\n"
        "Total exposure time:               7564.2  14043.3  s\r\n"
        "First--last dt per exposure:       7564.2  14043.3  s\r\n"
        "Reset--last dt per integration:    7396.7  13732.4  s\r\n"
        "\r\n"
        "Reference wavelength:                    4.46  microns\r\n"
        "Area of extraction aperture:             4.76  pixels\r\n"
        "Area of background measurement:           6.3  pixels\r\n"
        "Background surface brightness:            0.3  MJy/sr\r\n"
        "Total sky flux in background aperture:   4.94  e-/s\r\n"
        "Total flux in background aperture:      57.95  e-/s\r\n"
        "Background flux fraction from scene:     0.91\r\n"
        "Number of cosmic rays:      0.0072  events/pixel/read\r\n"
    )
    assert captured.out == expected_report


@pytest.mark.skip(reason='Having some issues with capsys IO')
def test_tso_print_rich(capsys):
    with open('mocks/tso_calculation_nircam_lw_tsgrism.pkl', 'rb') as f:
        result = pickle.load(f)
    # Make some troublesome figures
    result['report_in']['scalar']['duty_cycle'] = 0.25
    result['report_in']['input']['configuration']['detector']['ngroup'] = 130
    result['report_out']['scalar']['duty_cycle'] = 0.25
    result['report_out']['input']['configuration']['detector']['ngroup'] = 130

    jwst.tso_print(result)
    captured = capsys.readouterr()
    expected_report = (
        "Exposure time: 31014.40 s (8.62 h)\r\n"
        "Max fraction of saturation: 106.4%\r\n"
        "ngroup below 80% saturation: 97\r\n"
        "ngroup below 100% saturation: 122\r\n"
        "\r\n"
        "Signal-to-noise ratio       3484.8\r\n"
        "Extracted flux              2043.0  e-/s\r\n"
        "Flux standard deviation        0.6  e-/s\r\n"
        "Brightest pixel rate        1354.7  e-/s\r\n"
        "\r\n"
        "                               in-transit  out-transit\r\n"
        "Integrations:                         243      452\r\n"
        "Duty cycle:                          0.25     0.25\r\n"
        "Total exposure time:               7533.2  14012.3  s\r\n"
        "First--last dt per exposure:       7533.2  14012.3  s\r\n"
        "Reset--last dt per integration:    7366.4  13702.1  s\r\n"
        "\r\n"
        "Reference wavelength:                    4.36  microns\r\n"
        "Area of extraction aperture:             4.76  pixels\r\n"
        "Area of background measurement:           6.3  pixels\r\n"
        "Background surface brightness:            0.3  MJy/sr\r\n"
        "Total sky flux in background aperture:   4.45  e-/s\r\n"
        "Total flux in background aperture:      64.12  e-/s\r\n"
        "Background flux fraction from scene:     0.93\r\n"
        "Number of cosmic rays:      0.0072  events/pixel/read\r\n"
    )
    assert captured.out == expected_report




