# find-asteroids


[![DOI](https://zenodo.org/badge/1058178358.svg)](https://doi.org/10.5281/zenodo.17139782)


Find moving objects in detection catalogs.

<img width="960" height="720" alt="image" src="https://github.com/user-attachments/assets/1ce78a36-8a80-4db7-97bc-27c4a7e2d67f"/>

Installation via PyPI: 
```bash
$ python -m pip install find-asteroids
```

Usage ([docs/notebooks/search.ipynb](https://github.com/stevenstetzler/find_asteroids/tree/main/docs/notebooks/search.ipynb)):
```
usage: find-asteroids [-h] --catalog CATALOG [--psfs PSFS] --velocity VELOCITY
                      VELOCITY --angle ANGLE ANGLE --dx DX --num-results
                      NUM_RESULTS --results-dir RESULTS_DIR [--precompute]
                      [--gpu] [--gpu-kernels] [--device DEVICE]
                      [--output-format OUTPUT_FORMAT]

optional arguments:
  -h, --help            show this help message and exit
  --catalog CATALOG     The detection catalog to search. An astropy-readable
                        table containing at least 'ra', 'dec', and 'time'
                        columns (with units). (default: None)
  --psfs PSFS           An astropy-readable table containing a 'psf' column
                        (with units) that specifies the PSF-widths of the
                        images from which the detection catalog is derived. If
                        not provided, a value of 1 arcsec is assumed.
                        (default: None)
  --velocity VELOCITY VELOCITY
                        The velocity range over which to search, in units of
                        deg/day. (default: None)
  --angle ANGLE ANGLE   The on-sky angles over which to search, in units of
                        deg. (default: None)
  --dx DX               Search bin-width, in units of the PSF-width. (default:
                        None)
  --num-results NUM_RESULTS
                        Number of results to produce. (default: None)
  --results-dir RESULTS_DIR
                        The directory into which to write results. (default:
                        None)
  --precompute          Precompute projected positions of detections for all
                        trial velocities (uses more memory, but may be
                        faster). (default: False)
  --gpu                 Run the core-search components of the algorithm on
                        GPU. (default: False)
  --gpu-kernels         Run the entirety of the search algorithm on the GPU.
                        (default: False)
  --device DEVICE       The GPU device number to use. (default: -1)
  --output-format OUTPUT_FORMAT
                        The astropy.table supported format for writing
                        results. (default: ecsv)
```

References:
- Stetzler, S. et al. (2025) An Efficient Shift-and-Stack Algorithm Applied to Detection Catalogs
