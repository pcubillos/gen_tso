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

    Optional commands
    -----------------
    --debug:
        If set, run the app with reload=True, which reloads a live
        app if the code is updated.
    models_folder:
        If set, the app will attempt to load transit, eclipse, and SED
        models from the specified folder.
    """
    reload = '--debug' in sys.argv
    app = os.path.realpath(os.path.dirname(__file__)) + '/gen_tso_app.py'
    run_app(app, reload=reload, launch_browser=True)


if __name__ == "__main__":
    main()

