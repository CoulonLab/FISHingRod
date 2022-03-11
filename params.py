# Parameter file to use with FISH_pipeline.py and assignSpotsToNucl.py
#

FISHchannels=[[1,'_Q670',20.],[2,'_CAL610',35.],[3,'_Q570',25.],[4,'_FAM',250.]]
#fieldMask='out/20210331_MK_test_tmp-nuclei.tif'

#FISHchannels=[[0,'_AF750',10.]]
#fieldMask='out/20210331_MK_test_FAM_fieldMask.tif'


pxSize=.1078      # Pixel size in XY (in um)
zStep=.3          # Distance between z slices (in um)

psfXY=.13         # PSF width in XY (in um)
psfZ=.36          # PSF depth in Z (in um)

maxDist=2.;       # Maximal distance tolerated between guess and fit (in PSF width)
minSeparation=1.; # Minimal distance tolerated between two spots (in PSF width)

