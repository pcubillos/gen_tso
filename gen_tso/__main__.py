# Copyright (c) 2025 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

import os
import sys
from shiny import run_app



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
        # Import here, otherwise shiny breaks (bc of grequests?)
        from gen_tso.pandeia_io.pandeia_setup import update_synphot_files
        status = update_synphot_files()

    if '--update_exo' in sys.argv:
        import gen_tso.catalogs as cat
        cat.update_exoplanet_archive()

    if '--update_programs' in sys.argv:
        import gen_tso.catalogs as cat
        cat.update_jwst_programs()

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

