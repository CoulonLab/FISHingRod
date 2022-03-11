################################################################################
##
## FISHingRod -- Image analysis software suite for fluorescence in situ
##               hybridization (FISH) data.
##
## Antoine Coulon, 2022
## Contact: <antoine.coulon@curie.fr>
##
##
##    This program is free software: you can redistribute it and/or modify
##    it under the terms of the GNU General Public License as published by
##    the Free Software Foundation, either version 3 of the License, or
##    (at your option) any later version.
##
##    This program is distributed in the hope that it will be useful,
##    but WITHOUT ANY WARRANTY; without even the implied warranty of
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##    GNU General Public License for more details.
##
##    You should have received a copy of the GNU General Public License
##    along with this program.  If not, see <https://www.gnu.org/licenses/>.
##        
################################################################################

__version__='1.0.1'


import os, imp
from scipy import *
from skimage import io


#################################
##### EXECUTION PARAMETERS ######
#################################

parentDir='D:/Coulon_data/analysis'

inclFn=['_nucl.tif']
exclFn=[]

verbose=1

#################################
#################################


lDir=[parentDir+'/'+d for d in os.listdir(parentDir)
        if os.path.isdir(os.path.join(parentDir,d))
        and d[:2]!='__']

for dirIn in lDir:
    dirOut=dirIn+'/out'

    if os.path.isfile(dirIn+'/params.py'): prmFile=dirIn+'/params.py'
    elif os.path.isfile(parentDir+'/params.py'): prmFile=parentDir+'/params.py'
    elif os.path.isfile(os.getcwd()+'/params.py'): prmFile=os.getcwd()+'/params.py'
    else: raise ValueError("Could not locate 'params.py' file.")

    if verbose>=1:
      print("\nWorking folder: '%s'"%dirOut)
      print(  "Parameter file: '%s'"%prmFile)
    
    prm=imp.load_source('prm',prmFile)
    
    lFnTifNucl=[f for f in os.listdir(dirOut) if os.path.isfile(os.path.join(dirOut,f))
                and f[-4:]=='.tif'
                and all([a in f for a in inclFn])
                and not any([a in f for a in exclFn])]
    lFnTifNucl.sort()

    for ch in range(len(prm.FISHchannels)):
      if verbose>=1: print("\nChannel %d: '%s'"%(prm.FISHchannels[ch][0],prm.FISHchannels[ch][1]))

      loc2Cat=[]
      fnLoc2Cat=os.path.basename(dirIn)+prm.FISHchannels[ch][1]+'.loc2'

      for i,fnTifNucl in enumerate(lFnTifNucl):
        if verbose>=2: print(" └ Processing '%s'"%fnTifNucl)      
      
        if verbose>=3: print("    └ Reading image...",end='')
        im=io.imread(os.path.join(dirOut,fnTifNucl));
        if verbose>=3: print(" --> done.")

        if verbose>=3: print("    └ Assigning spots...",end='')
        loc2=[]
        fnLoc=fnTifNucl.split('_nucl.tif')[0].split('.tif')[0]+prm.FISHchannels[ch][1]+'.loc'
        fnLoc2=fnLoc+'2'
        if os.path.exists(os.path.join(dirOut,fnLoc)):
          loc=loadtxt(os.path.join(dirOut,fnLoc));
          loc=loc.reshape((-1,8))
          for l in loc:
            #l[2]=im.shape[0]-l[2]; l[6]=-l[6] # flip-y !!!!!!!!!!!!!!!!!!!!
            cellId=im[int(l[2].round()),int(l[1].round())]
            if cellId: loc2.append(r_[l,i,cellId])
        loc2Cat.extend(loc2)
        loc2=array(loc2)
        savetxt(os.path.join(dirOut,fnLoc2),loc2,fmt='%.5e',header='cols: intensity, x pos, y pos, z pos, bg offset, bg x tilt, bg y tilt, bg z tilt, frame, cellId')
        if verbose>=2: print(" --> done.")

      loc2Cat=array(loc2Cat)
      savetxt(os.path.join(dirOut,fnLoc2Cat),loc2Cat,fmt='%.5e',header='cols: intensity, x pos, y pos, z pos, bg offset, bg x tilt, bg y tilt, bg z tilt, frame, cellId')

