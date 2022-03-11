## FISHingRod – Utilities

- __`FISH_analysis_notebook_[...].ipynb`__ <br> A notebook to analyze 3D localization data (i.e. `.loc` and `.loc2` files created from `../FISH_pipeline.py` and `../assignSpotsToNucl.py`).<br><br>
- __`estimateChromaticCorrM_[...].ipynb`__ <br> A notebook to compute transformation matrices for correcting chromatic aberration in microscopy images. Such matrices can be used in `FISH_analysis_notebook_[...].ipynb` to correct the coordinates of spots detected on raw images, or in `channel_corrections_[...].ipynb` to correct images directly. <br><br>
- __`channel_corrections_[...].ipynb`__ <br> A notebook to correct chromatic aberrations on microscopy images and (optionally) fluorescence bleedthrough between channnels by image compensation.<br><br>
