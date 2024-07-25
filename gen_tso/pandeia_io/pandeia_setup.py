# Copyright (c) 2024 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    'check_latest_pandeia_version',
    'check_pandeia_ref_data',
    'check_pysynphot',
    'fetch_vega',
    'fetch_seds',
    'fetch_throughputs',
]

import os
from pathlib import Path
import requests
from shiny import ui
import shutil
import tarfile


stsci_url = 'https://archive.stsci.edu/hlsps/reference-atlases/'
syn1 = 'hlsp_reference-atlases_hst_multi_everything_multi_v16_sed.tar'
syn4 = 'hlsp_reference-atlases_hst_multi_kurucz-1993-atlas_multi_v2_synphot4.tar'
syn5 = 'hlsp_reference-atlases_hst_multi_pheonix-models_multi_v3_synphot5.tar'


def check_latest_pandeia_version():
    response = requests.get('https://pypi.org/pypi/pandeia.engine/json')
    last_pandeia = response.json()['info']['version']
    return last_pandeia


def check_pandeia_ref_data(engine_version=None):
    """
    Check that the Pandeia reference data environment variable exists.
    Check that the path exists.
    Check that the reference data version is consistent with Pandeia.engine

    Returns
    -------
    ui.HTML() formatted text with state of the pandeia reference data.
    """
    pandeia_url = (
        "https://outerspace.stsci.edu/display/PEN/Pandeia+Engine+Installation"
    )

    if "pandeia_refdata" not in os.environ:
        output = "Missing 'pandeia_refdata' environment variable"
    else:
        refdata_path = os.environ['pandeia_refdata']
        try:
            with open(f"{refdata_path}/VERSION_PSF") as fp:
                data_version = fp.readline().strip()
            if engine_version is None or engine_version==data_version:
                output = (
                    f"Pandeia reference data version: {data_version} is up to "
                    f"date<br>$pandeia_refdata={refdata_path}"
                )
            else:
                output = (
                    f"Pandeia reference data version: {data_version} < "
                    f"pandeia.engine version {engine_version}"
                )
        except OSError:
            output = f"Invalid 'pandeia_refdata' path: {repr(refdata_path)}"

    if 'up to date' in output:
        return ui.HTML(f'<span style="color:#0B980D">{output}</span>')

    return ui.span(
        ui.HTML(
            f'<span style="color:red">{output}.</span> '
            'Please follow the instructions in section 2.1 of '
        ),
        ui.tags.a(pandeia_url, href=pandeia_url, target="_blank"),
    )


def check_pysynphot():
    """
    Check pysynphot environment variable exists.
    Check that the path exists.
    Check that the required files are there.

    Returns
    -------
    ui.HTML() formatted text with state of pysynphot.
    """
    if "PYSYN_CDBS" not in os.environ:
        return "Missing 'PYSYN_CDBS' path"

    cdbs_path = os.environ['PYSYN_CDBS']
    if not os.path.exists(cdbs_path):
        return f'"PYSYN_CDBS" path does not exist:<br>{cdbs_path}'

    # check files
    vega_path = f'{cdbs_path}/calspec/alpha_lyr_stis_010.fits'
    atlases = {
        'throughput': os.path.exists(f'{cdbs_path}/comp'),
        'Kurucz SED': os.path.exists(f'{cdbs_path}/grid/k93models/'),
        'PHOENIX SED': os.path.exists(f'{cdbs_path}/grid/phoenix/'),
        'Vega SED': os.path.exists(vega_path),
    }

    lost = []
    found = []
    for atlas, is_there in atlases.items():
        if is_there:
            found.append(atlas)
        else:
            lost.append(atlas)
    found_txt = ', '.join(found)
    lost_txt = ', '.join(lost)

    if len(lost) == 0:
        return ui.HTML(
            f'<span style="color:#0B980D">All {found_txt} files are '
            'in place</span>'
        )
    if len(found) == 0:
        return ui.HTML(
            f'<span style="color:red">Could not find {lost_txt} files</span>'
        )

    return ui.HTML(
        f'<p><span style="color:#0B980D">Found {found_txt} files.</span> '
        f'<span style="color:red">But {lost_txt} files are missing.</span></p>'
    )


def update_pysynphot_files():
    if "PYSYN_CDBS" not in os.environ:
        raise ValueError('$PYSYN_CDBS environment variable is not defined')

    cdbs_path = os.environ['PYSYN_CDBS']
    if not os.path.exists(cdbs_path):
        Path(cdbs_path).mkdir(parents=True, exist_ok=True)
        if not os.path.exists(cdbs_path):
            raise ValueError(
                f'Could not create "PYSYN_CDBS" directory: {cdbs_path}'
            )

    # check files
    vega_path = f'{cdbs_path}/calspec/alpha_lyr_stis_010.fits'
    atlases = {
        'throughput': os.path.exists(f'{cdbs_path}/comp'),
        'Kurucz SED': os.path.exists(f'{cdbs_path}/grid/k93models/'),
        'PHOENIX SED': os.path.exists(f'{cdbs_path}/grid/phoenix/'),
        'Vega SED': os.path.exists(vega_path),
    }

    warnings = []
    if not atlases['Vega SED']:
        warnings.append(fetch_vega())
    if not atlases['Kurucz SED']:
        warnings.append(fetch_seds('k93models'))
    if not atlases['PHOENIX SED']:
        warnings.append(fetch_seds('phoenix'))
    if not atlases['throughput']:
        warnings.append(fetch_throughputs())

    warnings = [warn for warn in warnings if warn is not None]
    return warnings


def fetch_vega():
    """
    Fetch Vega reference spectrum and place it in the rigth folder.

    Returns
    -------
    On success, returns None.
    If there was an error, return a string describing the error.
    """
    if "PYSYN_CDBS" not in os.environ:
        return '$PYSYN_CDBS environment variable is not defined'

    vega_url = 'https://ssb.stsci.edu/trds/calspec/alpha_lyr_stis_010.fits'
    synphot_path = os.environ['PYSYN_CDBS']
    path_idx = vega_url.find('calspec')
    vega_path = f'{synphot_path}/{vega_url[path_idx:]}'

    query_parameters = {}
    response = requests.get(vega_url, params=query_parameters)
    if not response.ok:
        error = (
            'Could not download Vega reference spectrum\n'
            f'You may try downloading it manually from:\n   {vega_url}\n'
            f'And put the file in:\n    {vega_path}'
        )
        print('\n' + error)
        return error.replace('\n','<br>')

    if not os.path.exists(os.path.dirname(vega_path)):
        os.mkdir(os.path.dirname(vega_path))
    with open(vega_path, mode="wb") as file:
        file.write(response.content)


def fetch_seds(sed):
    """
    Download Kurucz or PHOENIX SED tar files from STScI.
    Unzip tar file.
    Place k93models/ or phoenix/ folder into $PYSYN_CDBS/trds/grid/

    Parameters
    ----------
    sed: String
        Select from 'k93models' or 'phoenix'.

    Returns
    -------
    On success, returns None.
    If there was an error, return a string describing the error.
    """
    if "PYSYN_CDBS" not in os.environ:
        return '$PYSYN_CDBS environment variable is not defined'
    synphot_path = os.environ['PYSYN_CDBS']

    if sed == 'k93models':
        url = f'{stsci_url}{syn4}'
    elif sed == 'phoenix':
        url = f'{stsci_url}{syn5}'
    else:
        return f'Invalid SED type: {sed}'
    model_dir = f'{synphot_path}/grid/{sed}'

    print(f'Downloading {sed} models ...')
    tmp_tar_file = f'{synphot_path}/synphot_file.tar'
    with requests.get(url, stream=True) as response:
        if not response.ok:
            return (
                'HTTP request failed. '
                f'Could not download {sed} reference spectrum'
            )
        with open(tmp_tar_file, mode="wb") as file:
            for chunk in response.iter_content(chunk_size=10 * 1024):
                file.write(chunk)

    print('Saving to PYSYN_CDBS grid folder')
    grid_dir = f'{synphot_path}/grid/'
    if not os.path.exists(grid_dir):
        Path(grid_dir).mkdir(parents=True, exist_ok=True)
    tmp_dir = f'{synphot_path}/tmp_sed'

    with tarfile.open(tmp_tar_file, "r") as tar:
        tar.extractall(path=tmp_dir)
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    shutil.move(f'{tmp_dir}/grp/redcat/trds/grid/{sed}', grid_dir)
    shutil.rmtree(tmp_dir)
    os.remove(tmp_tar_file)
    print(f'{sed} files updated!')


def fetch_throughputs():
    url = f'{stsci_url}{syn1}'
    return None
