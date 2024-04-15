# Copyright (c) 2024 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

import os
from shiny import run_app


def main():
    app = os.path.realpath(os.path.dirname(__file__)) + '/gen_tso_app.py'
    run_app(app, reload=True, launch_browser=True)


if __name__ == "__main__":
    main()

