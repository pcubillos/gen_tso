# Copyright (c) 2025 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

import os
import sys
from shiny import run_app

from gen_tso.pandeia_io.pandeia_setup import update_synphot_files
import gen_tso.catalogs as cat
from gen_tso.utils import ROOT


def main():
    """
    Launch the Gen TSO application.

    Usage
    -----
    # Run the application
    tso [--debug] [models_folder]

    # Check/update the pandeia reference data
    tso --update_db

    # Update the NASA Exoplanet Archive data
    tso --update_exo

    Optional commands
    -----------------
    --update_db:
        If set, update the pysynphot database
    --update_exo:
        If set, update the NASA exoplanet archive
    --debug:
        If set, run the app with reload=True, which reloads a live
        app if the code is updated.
    models_folder:
        If set, the app will attempt to load transit, eclipse, and SED
        models from the specified folder.
    """
    if '--update_db' in sys.argv:
        status = update_synphot_files()

    if '--update_exo' in sys.argv:
        cat.update_exoplanet_archive()

    if '--update_programs' in sys.argv:
        cat.update_jwst_programs()

    #if '--add_custom' in sys.argv:
    if '--update_custom' in sys.argv:
        try:
            i = sys.argv.index('--update_custom')
            csv_file = os.path.realpath(sys.argv[i+1])
        except Exception:
            print("ERROR: --update_custom requires a path argument", file=sys.stderr)
            sys.exit(2)
        cat.update_custom_targets(csv_file)
        sys.exit(0)

    if '--load_custom' in sys.argv:
        try:
            i = sys.argv.index('--load_custom')
            targets_path = os.path.realpath(sys.argv[i + 1])
        except Exception:
            print("ERROR: --load_custom requires a path argument", file=sys.stderr)
            sys.exit(2)
        os.environ['GEN_TSO_CUSTOM_TARGETS'] = targets_path
        print(f"Loading custom targets from: {targets_path}")

        if targets_path.lower().endswith('.csv'):
            session_txt = os.path.join(ROOT, 'data', 'custom_targets_session.txt')
            cat.load_csv_targets(targets_path, session_txt)
            print(f"Converted CSV to: {session_txt}")

    if (
        '--update_db' not in sys.argv and
        '--update_exo' not in sys.argv and
        '--update_programs' not in sys.argv
    ):
        reload = '--debug' in sys.argv
        app = os.path.realpath(os.path.dirname(__file__)) + '/gen_tso_app.py'
        run_app(app, reload=reload, launch_browser=True, dev_mode=False)


if __name__ == "__main__":
    main()

