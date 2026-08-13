"""PyInstaller entry script for the Content Factory desktop executable."""

from multiprocessing import freeze_support


if __name__ == "__main__":
    # PyInstaller's override diverts multiprocessing/resource-tracker children
    # before application imports can mistake them for another desktop launch.
    freeze_support()
    from app.frozen_entrypoint import main

    raise SystemExit(main())
