# Copyright (c) 2024 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

import subprocess
import os

def main():
    app = os.path.realpath(os.path.dirname(__file__)) + '/gen_tso_app.py'
    try:
        subprocess.call(f'shiny run --reload --launch-browser {app}'.split())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

