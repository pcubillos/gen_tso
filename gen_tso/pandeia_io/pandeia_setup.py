# Copyright (c) 2024 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    'check_latest_pandeia_version',
    'check_pandeia_ref_data',
    'check_pysynphot',
    'update_synphot_files',
    'fetch_vega',
    'fetch_synphot_files',
]

import os
from pathlib import Path
import shutil
import tarfile

import prompt_toolkit as ptk
import requests
from shiny import ui


stsci_url = 'https://archive.stsci.edu/hlsps/reference-atlases/'
syn1 = 'hlsp_reference-atlases_hst_multi_everything_multi_v16_sed.tar'
syn4 = 'hlsp_reference-atlases_hst_multi_kurucz-1993-atlas_multi_v2_synphot4.tar'
syn5 = 'hlsp_reference-atlases_hst_multi_pheonix-models_multi_v3_synphot5.tar'


def check_latest_pandeia_version():
    response = requests.get('https://pypi.org/pypi/pandeia.engine/json')
    last_pandeia = response.json()['info']['version']
    return last_pandeia


def check_pandeia_ref_data(latest_version):
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
        output = "Missing '$pandeia_refdata' environment variable"
    else:
        refdata_path = os.environ['pandeia_refdata']
        try:
            with open(f"{refdata_path}/VERSION_PSF") as fp:
                data_version = fp.readline().strip()
            if data_version == latest_version:
                output = (
                    f"Pandeia reference data version {data_version} is up to "
                    f"date<br>$pandeia_refdata={refdata_path}"
                )
            else:
                output = (
                    f"Pandeia reference data version: {data_version} < "
                    f"pandeia.engine version {latest_version}"
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


def update_synphot_files(force_update=False):
    """
    Check which synphot files are needed, fetch them from STScI,
    https://archive.stsci.edu/hlsp/reference-atlases
    and save them to the right folder pointed by the $PYSYN_CDBS
    environment variable.

    Parameters
    ----------
    force_update: Bool
        If True, enforce updating files even if they already exist.

    Returns
    -------
    warnings: List of strings
        List of any error during the file requests (if any).

    Examples
    --------
    >>> from gen_tso.pandeia_io.pandeia_setup import fetch_synphot_files
    >>> fetch_synphot_files('k93models')
    """
    unset_pysyn = (
        "'$PYSYN_CDBS' environment variable was not set. "
        "Could not setup pysynphot reference data"
    )
    if "PYSYN_CDBS" not in os.environ:
        completer = ptk.completion.PathCompleter(
            only_directories=True,
            expanduser=True,
        )
        synphot_dir = ptk.prompt(
            "The '$PYSYN_CDBS' environment variable is not defined.\nPlease "
            "set it to a dir where to store the Pandeia synphot data "
            "(~3GB of data)\n(and make sure the top folder is named 'trds')\n"
            "$PYSYN_CDBS = ",
            completer=completer,
        )

        if synphot_dir.strip() == '':
            print(unset_pysyn)
            return

        path = Path(os.path.realpath(os.path.expanduser(synphot_dir)))
        if path.name != 'trds':
            go_on = ptk.prompt(
                "The '$PYSYN_CDBS' top dir is not named 'trds', what about:"
                f"\n{path / 'trds'}"
                '\nProceed (Y/n)? '
            )
            if go_on.strip().lower() not in ['', 'y', 'yes']:
                print(unset_pysyn)
                return
            path = path / 'trds'
        print(
            "\nNow, add this line to your bash file:\n"
            f"export PYSYN_CDBS={path}\nand then 'source' "
            "the bash file or launch the app from a new terminal"
        )
        synphot_path = str(path)
    else:
        synphot_path = os.environ['PYSYN_CDBS']

    if not os.path.exists(synphot_path):
        Path(synphot_path).mkdir(parents=True, exist_ok=True)
        if not os.path.exists(synphot_path):
            raise ValueError(
                f'Could not create "PYSYN_CDBS" directory: {synphot_path}'
            )

    # Check files to update
    vega_path = f'{synphot_path}/calspec/alpha_lyr_stis_010.fits'
    atlases = {
        'throughput': os.path.exists(f'{synphot_path}/comp'),
        'Kurucz SED': os.path.exists(f'{synphot_path}/grid/k93models/'),
        'PHOENIX SED': os.path.exists(f'{synphot_path}/grid/phoenix/'),
        'Vega SED': os.path.exists(vega_path),
    }

    warnings = []
    if not atlases['Vega SED'] or force_update:
        warnings.append(fetch_vega(synphot_path))
    if not atlases['Kurucz SED'] or force_update:
        warnings.append(fetch_synphot_files('k93models', synphot_path))
    if not atlases['PHOENIX SED'] or force_update:
        warnings.append(fetch_synphot_files('phoenix', synphot_path))
    if not atlases['throughput'] or force_update:
        warnings.append(fetch_synphot_files('throughput', synphot_path))

    warnings = [warn for warn in warnings if warn is not None]
    return warnings


def fetch_vega(synphot_path=None):
    """
    Fetch Vega reference spectrum and place it in the rigth folder.

    Returns
    -------
    On success, returns None.
    If there was an error, return a string describing the error.
    """
    if synphot_path is None:
        if "PYSYN_CDBS" not in os.environ:
            return '$PYSYN_CDBS environment variable is not defined'
        synphot_path = os.environ['PYSYN_CDBS']

    vega_url = 'https://ssb.stsci.edu/trds/calspec/alpha_lyr_stis_010.fits'
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


def fetch_synphot_files(synphot_file, synphot_path=None):
    """
    Download Kurucz or PHOENIX SED tar files from STScI.
    Unzip tar file.
    Place k93models/ or phoenix/ folder into $PYSYN_CDBS/trds/grid/

    Parameters
    ----------
    sed: String
        Select from 'k93models', 'phoenix', or 'throughput'.

    Returns
    -------
    On success, returns None.
    If there was an error, return a string describing the error.

    Examples
    --------
    >>> from gen_tso.pandeia_io.pandeia_setup import fetch_synphot_files
    >>> fetch_synphot_files('k93models')
    """
    if synphot_path is None:
        if "PYSYN_CDBS" not in os.environ:
            return '$PYSYN_CDBS environment variable is not defined'
        synphot_path = os.environ['PYSYN_CDBS']

    if synphot_file == 'k93models':
        url = f'{stsci_url}{syn4}'
    elif synphot_file == 'phoenix':
        url = f'{stsci_url}{syn5}'
    elif synphot_file == 'throughput':
        url = f'{stsci_url}{syn1}'
    else:
        return f'Invalid Synphot file type: {synphot_file}'

    if not os.path.exists(synphot_path):
        Path(synphot_path).mkdir(parents=True, exist_ok=True)

    print(f'Downloading {synphot_file} files ...')
    tmp_tar_file = f'{synphot_path}/synphot_file_{synphot_file}.tar'
    with requests.get(url, stream=True) as response:
        if not response.ok:
            return (
                'HTTP request failed. '
                f'Could not download {synphot_file} reference files'
            )
        with open(tmp_tar_file, mode="wb") as file:
            for chunk in response.iter_content(chunk_size=10 * 1024):
                file.write(chunk)

    print('Saving to PYSYN_CDBS grid folder')
    tmp_dir = f'{synphot_path}/tmp_synphot'
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    with tarfile.open(tmp_tar_file, "r") as tar:
        tar.extractall(path=tmp_dir)
    shutil.copytree(
        f'{tmp_dir}/grp/redcat/trds/', synphot_path,
        dirs_exist_ok=True,
    )
    shutil.rmtree(tmp_dir)
    os.remove(tmp_tar_file)
    print(f'{synphot_file} files updated!')

