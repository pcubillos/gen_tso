# Copyright (c) 2025 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

import os
import sys
from shiny import run_app

from gen_tso.pandeia_io.pandeia_setup import update_synphot_files
import gen_tso.catalogs as cat
from gen_tso.utils import ROOT, parser


def main():
    """Launch the Gen TSO application"""
    args = parser()

    if args.update_exo:
        cat.update_exoplanet_archive()
        sys.exit(0)

    if args.update_programs:
        cat.update_jwst_programs()
        sys.exit(0)

    if args.update_db:
        update_synphot_files()
        sys.exit(0)

    if args.update_custom is not None:
        cat.update_custom_targets(args.update_custom, mode='replace')
        sys.exit(0)

    if args.add_custom is not None:
        cat.update_custom_targets(args.add_custom, mode='add')
        sys.exit(0)

    if args.targets is not None:
        os.environ['GEN_TSO_CUSTOM_TARGETS'] = args.targets
        #if targets_path.lower().endswith('.csv'):
        #    session_txt = os.path.join(ROOT, 'data', 'custom_targets_session.txt')
        #    cat.load_csv_targets(targets_path, session_txt)
        #    print(f"Converted CSV to: {session_txt}")

    reload = args.debug
    app = os.path.join(ROOT, 'gen_tso_app.py')
    run_app(app, reload=reload, launch_browser=True, dev_mode=False)


if __name__ == "__main__":
    main()

