# Copyright (c) 2024 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

import os
import pickle
import pytest

import numpy as np

import gen_tso.pandeia_io as jwst
from gen_tso.utils import ROOT

os.chdir(ROOT+'../tests')
# See tests/mocks/make_mocks.py for mocked data setup.


def test_saturation_level_perform_calculation_single():
    with open('mocks/perform_calculation_nircam_lw_tsgrism.pkl', 'rb') as f:
        result = pickle.load(f)

    pixel_rate, full_well = jwst.saturation_level(result)
    expected_rate = 1396.5528564453125
    expected_well = 58100.00422843957
    np.testing.assert_allclose(pixel_rate, expected_rate)
    np.testing.assert_allclose(full_well, expected_well)


def test_saturation_level_perform_calculation_multi():
    with open('mocks/perform_calculation_miri_mrs_ts.pkl', 'rb') as f:
        result = pickle.load(f)

    pixel_rate, full_well = jwst.saturation_level(result)
    expected_rate = [
        204.37779236, 109.17165375,  32.75593948,   4.55676889,
    ]
    expected_well = [
        193654.99564498, 193655.00540975, 193655.004117  , 193654.99798687,
    ]
    np.testing.assert_allclose(pixel_rate, expected_rate)
    np.testing.assert_allclose(full_well, expected_well)


def test_saturation_level_tso_calculation_single():
    with open('mocks/tso_calculation_nircam_lw_tsgrism.pkl', 'rb') as f:
        tso = pickle.load(f)

    pixel_rate, full_well = jwst.saturation_level(tso)
    expected_rate = [1354.6842041 , 1396.55285645]
    expected_well = [58100.00016363, 58100.00129109]
    np.testing.assert_allclose(pixel_rate, expected_rate)
    np.testing.assert_allclose(full_well, expected_well)


def test_saturation_level_tso_calculation_multi():
    with open('mocks/tso_calculation_miri_mrs_ts.pkl', 'rb') as f:
        tso = pickle.load(f)

    pixel_rate, full_well = jwst.saturation_level(tso)
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
def test_saturation_level_tso_calculation_get_max():
    pass


def test_exposure_time_nircam():
    inst = 'nircam'
    subarray = 'subgrism64'
    readout = 'rapid'
    ngroup = 90
    nint = 1
    exp_time = jwst.exposure_time(inst, subarray, readout, ngroup, nint)
    np.testing.assert_allclose(exp_time, 31.00063)


def test_exposure_time_miri():
    inst = 'miri'
    subarray = 'slitlessprism'
    readout = 'fastr1'
    ngroup = 30
    nint = 1
    exp_time = jwst.exposure_time(inst, subarray, readout, ngroup, nint)
    np.testing.assert_allclose(exp_time, 4.7712)


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
    with open('mocks/perform_calculation_nircam_lw_tsgrism.pkl', 'rb') as f:
        result = pickle.load(f)

    text = jwst._print_pandeia_saturation(reports=[result], format=None)
    expected_text = """Max fraction of saturation: 1.6%
ngroup below 80% saturation: 97
ngroup below 100% saturation: 122"""
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


@pytest.mark.skip(reason='TBD')
def test__print_pandeia_stats():
    pass


def test__print_pandeia_report_perform_calculation_single():
    with open('mocks/perform_calculation_nircam_lw_tsgrism.pkl', 'rb') as f:
        result = pickle.load(f)

    report = jwst._print_pandeia_report([result], format=None)
    expected_report = """Exposure time: 701.41 s (0.19 h)
Max fraction of saturation: 1.6%
ngroup below 80% saturation: 97
ngroup below 100% saturation: 122

Signal-to-noise ratio        410.9
Extracted flux              2106.2  e-/s
Flux standard deviation        5.1  e-/s
Brightest pixel rate        1396.6  e-/s

Integrations:                         683
Duty cycle:                          0.33
Total exposure time:                701.4  s
First--last dt per exposure:        701.4  s
Reset--last dt per integration:     232.6  s

Reference wavelength:                    4.36  microns
Area of extraction aperture:             4.76  pixels
Area of background measurement:           6.3  pixels
Background surface brightness:            0.3  MJy/sr
Total sky flux in background aperture:   4.45  e-/s
Total flux in background aperture:      65.97  e-/s
Background flux fraction from scene:     0.93
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
    expected_report = """Exposure time: 21545.44 s (5.98 h)
Max fraction of saturation: 73.7%
ngroup below 80% saturation: 97
ngroup below 100% saturation: 122

Signal-to-noise ratio       3484.8
Extracted flux              2043.0  e-/s
Flux standard deviation        0.6  e-/s
Brightest pixel rate        1354.7  e-/s

                               in-transit  out-transit
Integrations:                         243      452
Duty cycle:                          0.98     0.98
Total exposure time:               7533.2  14012.3  s
First--last dt per exposure:       7533.2  14012.3  s
Reset--last dt per integration:    7366.4  13702.1  s

Reference wavelength:                    4.36  microns
Area of extraction aperture:             4.76  pixels
Area of background measurement:           6.3  pixels
Background surface brightness:            0.3  MJy/sr
Total sky flux in background aperture:   4.45  e-/s
Total flux in background aperture:      64.12  e-/s
Background flux fraction from scene:     0.93
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

