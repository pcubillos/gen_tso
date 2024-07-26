# Copyright (c) 2024 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

import os
import sys
from shiny import run_app



def main():
    """
    Launch the Gen TSO application.

    Usage
    -----
    tso [--debug] [models_folder]
    tso --setup_db

    Optional commands
    -----------------
    --setup_db:
        If set, update the NASA exoplanet archive and pysynphot databases
        Otherwise, launch the TSO app
    --debug:
        If set, run the app with reload=True, which reloads a live
        app if the code is updated.
    models_folder:
        If set, the app will attempt to load transit, eclipse, and SED
        models from the specified folder.
    """
    if '--setup_db' in sys.argv:
        # Import here, otherwise shiny breaks (bc of grequests?)
        import gen_tso.catalogs as cat
        from gen_tso.pandeia_io.pandeia_setup import update_synphot_files

        cat.update_exoplanet_archive()
        status = update_synphot_files()

    else:
        reload = '--debug' in sys.argv
        app = os.path.realpath(os.path.dirname(__file__)) + '/gen_tso_app.py'
        run_app(app, reload=reload, launch_browser=True, dev_mode=False)


if __name__ == "__main__":
    main()

