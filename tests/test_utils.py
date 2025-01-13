# Copyright (c) 2025 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)
 
import gen_tso.utils as u
from prompt_toolkit.formatted_text import FormattedText


def test_format_text_plain():
    text = 'WASP-80 b'
    formatted = u.format_text(text, danger=True)
    assert formatted == text


def test_format_text_normal():
    text = 'WASP-80 b'
    formatted = u.format_text(text, warning=False, danger=False, format='html')
    assert formatted == text


def test_format_text_html_warning():
    text = 'WASP-80 b'
    formatted = u.format_text(text, warning=True, format='html')
    assert formatted == '<span class="warning">WASP-80 b</span>'


def test_format_text_html_danger():
    text = 'WASP-80 b'
    formatted = u.format_text(text, danger=True, format='html')
    assert formatted == '<span class="danger">WASP-80 b</span>'


def test_format_text_rich_warning():
    text = 'WASP-80 b'
    formatted = u.format_text(text, warning=True, format='rich')
    assert formatted == '<warning>WASP-80 b</warning>'


def test_format_text_rich_danger():
    text = 'WASP-80 b'
    formatted = u.format_text(text, danger=True, format='rich')
    assert formatted == '<danger>WASP-80 b</danger>'


def test_format_text_warning():
    text = 'WASP-80 b'
    formatted = u.format_text(text, warning=True, format='html')
    assert formatted == '<span class="warning">WASP-80 b</span>'


def test_format_text_danger1():
    text = 'WASP-80 b'
    formatted = u.format_text(text, danger=True, format='html')
    assert formatted == '<span class="danger">WASP-80 b</span>'


def test_format_text_danger2():
    text = 'WASP-80 b'
    formatted = u.format_text(text, warning=True, danger=True, format='html')
    assert formatted == '<span class="danger">WASP-80 b</span>'


def test_read_spectrum_file_success():
    pass


def test_read_spectrum_file_fail_none():
    pass


def test_read_spectrum_file_fail_warning_format():
    pass


def test_read_spectrum_file_fail_warning_wrong_file():
    pass


def test_read_spectrum_file_fail_error_format():
    pass


def test_read_spectrum_file_fail_error_wrong_file():
    pass


def test_collect_spectra_success():
    pass


def test_collect_spectra_success_and_fail_none():
    pass


def test_collect_spectra_success_and_fail_warning():
    pass


def test_collect_spectra_fail_error():
    pass
