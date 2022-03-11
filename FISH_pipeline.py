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

__version__='1.0.8'



import os, imp
from scipy import *
from scipy import fftpack
from skimage import io, filters
from skimage.external import tifffile
from matplotlib import pyplot as plt
import warnings
                    

def GaussianMaskFit2(im,coo,s,optLoc=1,bgSub=2,winSize=13,convDelta=.05,nbIter=20):
  """Applies the algorithm from [Thompson et al. (2002) PNAS, 82:2775].
Parameters:
- im: a numpy array with the image
- coo: approximate coordinates (in pixels) of the spot to localize and measure
- s: width of the PSF in pixels
- optLoc: If 1, applied the iterative localization refinement algorithm, starting with the coordinates provided in coo. If 0, only measures the spot intensity at the coordinates provided in coo.
- bgSub: 0 -> no background subtraction. 1 -> constant background subtraction. 2 -> tilted plane background subtraction.
- winSize: Size of the window (in pixels) around the position in coo, used for the iterative localization and for the background subtraction.
- convDelta: cutoff to determine convergence, i.e. the distance (in pixels) between two iterations
- nbIter: the maximal number of iterations.

Returns
- the intensity value of the spot.
- the corrdinates of the spot.

If convergence is not found after nbIter iterations, return 0 for both intensity value and coordinates.
"""
  coo=array(coo);
  for i in range(nbIter):
    if not prod(coo-winSize/2.>=0)*prod(coo+winSize/2.<=im.shape[::-1]): return 0.,r_[0.,0.], 0.
    winOrig=(coo-(winSize-1)/2).astype(int)
    ix,iy=meshgrid(winOrig[1]+r_[:winSize],winOrig[0]+r_[:winSize]);
    N=exp(-(iy-coo[0])**2/(2*s**2)-(ix-coo[1])**2/(2*s**2))/(2*pi*s**2)
    S=im[winOrig[0]:winOrig[0]+winSize][:,winOrig[1]:winOrig[1]+winSize]*1.
    if bgSub==2:
      xy=r_[:2*winSize]%winSize-(winSize-1)/2.
      bgx=polyfit(xy,r_[S[0],S[-1]],1); S=(S-xy[:winSize]*bgx[0]).T;
      bgy=polyfit(xy,r_[S[0],S[-1]],1); S=(S-xy[:winSize]*bgy[0]).T;
      bg=mean([S[0],S[-1],S[:,0],S[:,-1],]); S-=bg
      bg=r_[bg,bgx[0],bgy[0]]
    if bgSub==1:
      bg=mean([S[0],S[-1],S[:,0],S[:,-1],]); S-=bg
    S=S.clip(0) # Prevent negative values !!!!
    if optLoc:
      SN=S*N; ncoo=r_[sum(iy*SN),sum(ix*SN)]/sum(SN)
      ncoo=2*ncoo-coo # Extrapolation of localization step !!!!
      if abs(coo-ncoo).max()<convDelta: return sum(SN)/sum(N**2),coo,bg
      else: coo=ncoo
    else: return sum(S*N)/sum(N**2),coo,bg
  return 0.,r_[0.,0.], 0.


def GaussianMaskFit3D(im,coo,s,sZ=None,optLoc=1,bgSub=2,winSize=None,winSizeZ=None,convDelta=.05,nbIter=20):
  """Applies the algorithm from [Thompson et al. (2002) PNAS, 82:2775] adapted to 3D images.
Parameters:
- im: a numpy array with the image
- coo: approximate z,y,x coordinates (in pixels) of the spot to localize and measure
- s: width of the PSF in x,y in pixels
- sZ: width of the PSF in z in pixels. Defaults to the same value as s.
- optLoc: If 1, applied the iterative localization refinement algorithm, starting with the coordinates provided in coo. If 0, only measures the spot intensity at the coordinates provided in coo.
- bgSub: 0 -> no background subtraction. 1 -> constant background subtraction. 2 -> tilted plane background subtraction.
- winSize: Size of the x,y window (in pixels) around the position in coo, used for the iterative localization and for the background subtraction.
- winSizeZ: Same as winSize, in the z dimension.
- convDelta: cutoff to determine convergence, i.e. the distance (in pixels) between two iterations
- nbIter: the maximal number of iterations.

Returns
- the intensity value of the spot.
- the coordinates of the spot (z,y,x).
- the background level:
   - either a constant value if bgSub=1
   - or [offset, tilt in z, tilt in y, tilt in x] if bgSub=2

If convergence is not found after nbIter iterations, return 0 for both intensity value and coordinates.
"""
  coo=array(coo);
  if sZ==None: sZ=s
  if winSize ==None: winSize =int(ceil(s*8./2))*2+1
  if winSizeZ==None: winSizeZ=int(ceil(sZ*4./2))*2+1
  for i in range(nbIter):
    if not (winSizeZ/2.<=coo[0]<=im.shape[0]-winSizeZ/2.)*prod([winSize/2.<=coo[j]<=im.shape[j]-winSize/2. for j in [1,2]]):
      return 0.,r_[0.,0.,0.], 0.
    winOrig=r_[coo[0]-int(winSizeZ/2),coo[1:]-int(winSize/2)].astype(int)
    iy,iz,ix=meshgrid(winOrig[1]+r_[:winSize],winOrig[0]+r_[:winSizeZ],winOrig[2]+r_[:winSize]);
    N=exp(-(iz-coo[0])**2/(2*sZ**2)-(iy-coo[1])**2/(2*s**2)-(ix-coo[2])**2/(2*s**2))/((2*pi)**1.5*s*s*sZ)
    S=im[winOrig[0]:winOrig[0]+winSizeZ][:,winOrig[1]:winOrig[1]+winSize][:,:,winOrig[2]:winOrig[2]+winSize]*1.
    if bgSub==2:
      cxy=r_[:winSize]-(winSize-1)/2.
      cz=r_[:winSizeZ]-(winSizeZ-1)/2.
      bgx=polyfit(cxy,mean(r_[S[:,0],S[:,-1]],0),1)[0];
      bgy=polyfit(cxy,mean(r_[S[:,:,0],S[:,:,-1]],0),1)[0];
      bgz=polyfit(cz,mean(c_[S[:,0],S[:,-1],S[:,1:-1,0],S[:,1:-1,-1]],1),1)[0];
      S=rollaxis(rollaxis(rollaxis(S-cxy*bgx,2)-cxy*bgy,2)-cz*bgz,2)
      bg=mean([S[:,0],S[:,-1],S[:,:,0],S[:,:,-1],]); S-=bg
      bg=r_[bg,bgz,bgy,bgx]
    if bgSub==1:
      bg=mean([S[:,0],S[:,-1],S[:,:,0],S[:,:,-1],]); S-=bg
    #S=S.clip(0) # Prevent negative values !!!!
    if optLoc:
      SN=S*N; ncoo=r_[sum(iz*SN),sum(iy*SN),sum(ix*SN)]/sum(SN)
      #ncoo+=ncoo-coo # Extrapolation of localization step !!!!
      #ncoo+=(ncoo-coo)*.7 # Extrapolation of localization step !!!!
      #print(i,ncoo,abs(coo-ncoo).max())
      if abs(coo-ncoo).max()<convDelta: return sum(SN)/sum(N**2),coo,bg
      else: coo=ncoo
    else: return sum(S*N)/sum(N**2),coo,bg
  return 0.,r_[0.,0.,0.], 0.


def bpass(im,r1=1.,r2=1.7):
  x=r_[r_[:(1+im.shape[1])//2],r_[(1-im.shape[1])//2:0]]
  y=r_[r_[:(1+im.shape[0])//2],r_[(1-im.shape[0])//2:0]]
  ker1x=exp(-(x/r1)**2/2); ker1x/=sum(ker1x); fker1x=fft(ker1x);
  ker1y=exp(-(y/r1)**2/2); ker1y/=sum(ker1y); fker1y=fft(ker1y);
  ker2x=exp(-(x/r2)**2/2); ker2x/=sum(ker2x); fker2x=fft(ker2x);
  ker2y=exp(-(y/r2)**2/2); ker2y/=sum(ker2y); fker2y=fft(ker2y);
  fim=fftpack.fftn(im)
  return fftpack.ifftn((fim*fker1x).T*fker1y-(fim*fker2x).T*fker2y).real.T

def bpass3D(im,r1=1.,r2=1.7,rz1=1.,rz2=1.7,zMirror=False):
  return filters.gaussian(im,r_[rz1,r1,r1],mode='mirror')-filters.gaussian(im,r_[rz2,r2,r2],mode='mirror')


def locFileToFieldMask(maskDimensions, fnLoc2, distPx, cutoffLow=None):
  """maskDimensions: either a tuple (F,Y,X) (where F is the number of frames (field of view) and Y and X are the image dimensions in px), or the name of a TIFF file with the desired dimensions (e.g. the nuclear mask file).
  """
  if type(maskDimensions) not in [tuple, list]:
    maskDimensions=io.imread(maskDimensions).T.swapaxes(1,2).shape;
  fieldMask=zeros(maskDimensions,dtype=uint16)
  loc2=loadtxt(fnLoc2)
  for a in loc2:
    for y in range(max(0,int(a[2]-distPx)),min(1+int(a[2]+distPx),maskDimensions[1])):
      for x in range(max(0,int(a[1]-distPx)),min(1+int(a[1]+distPx),maskDimensions[2])):
        if (x-a[1])**2+(y-a[2])**2<distPx**2: fieldMask[int(a[8]),y,x]=1
  #io.imsave(fnLoc2.replace('.loc2','')+'_fieldMask.tif',fieldMask.swapaxes(1,2).T)
  io.imsave(fnLoc2.replace('.loc2','')+'_fieldMask.tif',fieldMask)


#######################
def batchAnalysis(parentDir,inclFn=[],exclFn=['_nucl'],saveGuessLoc=True,saveFailedLoc=True,showPixValHisto=True,saveIntermediateImages=False,paramFile='params.py',outputFolder='out',verbose=2):
  
  lDir=[parentDir+'/'+d for d in os.listdir(parentDir)
          if os.path.isdir(os.path.join(parentDir,d))
          and d[:2]!='__']
  
  for dirIn in lDir:
      dirOut=os.path.join(dirIn,outputFolder)
      if not os.path.isdir(dirOut): os.mkdir(dirOut)
      if os.path.isfile(os.path.join(dirIn,paramFile)): prmFile=os.path.join(dirIn,paramFile)
      elif os.path.isfile(os.path.join(parentDir,paramFile)): prmFile=os.path.join(parentDir,paramFile)
      elif os.path.isfile(os.path.join(os.getcwd(),paramFile)): prmFile=os.path.join(os.getcwd(),paramFile)
      else: raise ValueError("Could not locate '%s' file."%paramFile)
  
      if verbose>=1:
        print("\nInput folder:   '%s'"%dirIn)
        print( " Output folder:  '%s'"%dirOut)
        print( " Parameter file: '%s'"%prmFile)
      
      prm=imp.load_source('prm',prmFile)
      
      psfPx=prm.psfXY/prm.pxSize
      psfZPx=prm.psfZ/prm.zStep
      
      if hasattr(prm,'fieldMask'):
        if verbose>=2: print("\nReading field mask image '%s'."%(prm.fieldMask))
        #fieldMask=io.imread(os.path.join(dirIn,prm.fieldMask)).T.swapaxes(1,2);
        fieldMask=io.imread(os.path.join(dirIn,prm.fieldMask));
        # /!\ Dimensions should be FYX /!\
        useFieldMask=True
      else: useFieldMask=False

      lFnTif=[f for f in os.listdir(dirIn) if os.path.isfile(os.path.join(dirIn,f))
                  and f[-4:]=='.tif'
                  and all([a in f for a in inclFn])
                  and not any([a in f for a in exclFn])]
      lFnTif.sort()
      #print(lFnTif);
      
      for iTif,fnTif in enumerate(lFnTif):
        if verbose>=1: print("\nProcessing (%d) '%s'"%(iTif,fnTif))
        #if iTif<0: continue # !!!!!!!!!!!!!!!!
                
        for ch in range(len(prm.FISHchannels)):
          if verbose>=1: print(" └ Channel %d: '%s'"%(prm.FISHchannels[ch][0],prm.FISHchannels[ch][1]))
        
          if verbose>=2: print("    └ Reading image...",end='')
          if tifffile.TiffFile(os.path.join(dirIn,fnTif)).is_micromanager:
            imFile=tifffile.TiffFile(os.path.join(dirIn,fnTif),is_ome=False)
            mmMetadata=imFile.micromanager_metadata
            im=imFile.asarray()
            im=im.reshape(im.shape[:-2][::-1]+im.shape[-2:]).swapaxes(0,1) #!!!! Use this line or not depending on micromanager version used
          else: im=io.imread(os.path.join(dirIn,fnTif));
          #im=rollaxis(im, axis=3, start=1) #!!!! Use this line or not depending on micromanager version used
          #print("\n==> Dimensions of image are %s and should be ZCYX. <=="%(im.shape,))
          # /!\ Dimensions should be ZCYX /!\
          
          ##### DEBUG #######
          if 1:
            fig,ax=plt.subplots(1,2)
            plt.sca(ax[0]);
            if useFieldMask: plt.imshow(fieldMask[iTif]);
            plt.sca(ax[1]); plt.imshow(im[13,4]);
            plt.show()
          #break;
          ##### DEBUG #######
          
          #imCh=im[:,:,:,prm.FISHchannels[ch][0]]
          imCh=im[:,prm.FISHchannels[ch][0]]
          if verbose>=2: print(" --> done.")
        
          ### Compute band-pass image
          if verbose>=2: print("    └ Computing band-pass image...",end='')

          imBp=bpass3D(imCh,1.,psfPx,1.,psfZPx,zMirror=4)*65535 # !!!!!!!!!!
          if saveIntermediateImages:
            io.imsave(os.path.join(dirOut,fnTif)[:-4]+prm.FISHchannels[ch][1]+'_bp.tif',float32(imBp))
            print(" --> Image saved.")
          elif verbose>=2: print(" --> done.")
            
          ### Find local maxima
          if verbose>=2: print("    └ Finding local maxima...",end='')
          locMax=rollaxis(array([roll(imBp,i,-1)  *r_[0,ones(imBp.shape[-1]  -2),0] for i in r_[-1:2]]).max(0),-1);
          locMax=rollaxis(array([roll(locMax,i,-1)*r_[0,ones(locMax.shape[-1]-2),0] for i in r_[-1:2]]).max(0),-1);
          locMax=rollaxis(array([roll(locMax,i,-1)*r_[0,ones(locMax.shape[-1]-2),0] for i in r_[-1:2]]).max(0),-1);
          locMax=(locMax==imBp)*1
          if useFieldMask: locMax*=fieldMask[iTif]
          locMaxCoo=array(where(locMax)); locMaxCoo=tuple(locMaxCoo.T[where((locMaxCoo[0]!=0)*(locMaxCoo[0]!=imBp.shape[0]-1)*(locMaxCoo[1]!=0)*(locMaxCoo[1]!=imBp.shape[1]-1)*(locMaxCoo[2]!=0)*(locMaxCoo[2]!=imBp.shape[2]-1))].T)
          locMaxVal=imBp[locMaxCoo];
          if saveIntermediateImages:
            with warnings.catch_warnings():
              warnings.simplefilter("ignore")
              io.imsave(os.path.join(dirOut,fnTif)[:-4]+prm.FISHchannels[ch][1]+'_locMax.tif',float32(locMax))
            print(" --> Image saved.")
          elif verbose>=2: print(" --> done.")
        
           ### Show pixel value histogram
          if showPixValHisto and locMaxVal.shape[0]:
            fig=plt.figure(figsize=(8,3))
            fig.add_subplot(121)
            h,x=histogram(log10(maximum(1e-12,locMaxVal)),bins=r_[-1:log10(locMaxVal.max()):.01]);
            plt.xscale('log'); plt.xlabel('Pixel intensity');
            plt.yscale('log'); plt.ylabel('Count');
            plt.axvline(median(locMaxVal),ls='--'); plt.axvline(mean(locMaxVal),ls='-'); plt.axvline(mean(locMaxVal)+var(locMaxVal)**.5,ls=':'); plt.axvline(prm.FISHchannels[ch][2],c='r',ls='-');
            plt.plot(10**x[:-1],h+1);
            fig.add_subplot(122)
            plt.xscale('log'); plt.xlabel('Pixel intensity');
            plt.yscale('log'); plt.ylabel('Inv. cumul.'); plt.xlim(.1,locMaxVal.max()); plt.ylim(.8/locMaxVal.shape[0],2.);
            plt.axvline(median(locMaxVal),ls='--',label='median'); plt.axvline(mean(locMaxVal),ls='-',label='mean'); plt.axvline(mean(locMaxVal)+var(locMaxVal)**.5,ls=':',label='mean + sd'); plt.axvline(prm.FISHchannels[ch][2],c='r',ls='-',label='Your threshold');
            plt.scatter(sort(locMaxVal)[::-1],r_[0:1:1j*locMaxVal.shape[0]],s=5); plt.legend(loc='upper left', bbox_to_anchor=(1.1, 1.))
            plt.tight_layout(); plt.show()
        
          ### Fit each local maximum with a Gaussian mask
          if verbose>=2: print("    └ Fitting Gaussian masks...            ",end='')
          zG,yG,xG,valG=tuple(array(locMaxCoo+(locMaxVal,))[:,where(locMaxVal>prm.FISHchannels[ch][2])])
          cooGuess=r_[zG,yG,xG].T
          if saveGuessLoc:
            fnGuess=fnTif[:-4]+prm.FISHchannels[ch][1]+'_guess.loc'
            savetxt(os.path.join(dirOut,fnGuess),r_[valG,xG,yG,zG].T,fmt='%.5e',header='cols: value of local maximum, x pos, y pos, z pos')
          fitResults=[]
          failedLoc=[]
          for i in r_[:cooGuess.shape[0]]:
            if verbose==2: print('\b\b\b\b\b\b\b\b\b\b\b%5d/%5d'%(i+1,cooGuess.shape[0]),end='')
            # Eliminate localization on image border
            if r_[cooGuess[i][-2:],imCh.shape[-2:]-cooGuess[i][-2:]-1].min()>1:
              intensity,coo,tilt=GaussianMaskFit3D(imCh,cooGuess[i],psfPx)
              # Keep only if it converged close to the inital guess (and inside field mask)
              if intensity!=0 and sum(((coo-cooGuess[i])/r_[psfZPx,psfPx,psfPx])**2)<prm.maxDist**2 and ((not useFieldMask) or (lambda cooInt: fieldMask[iTif,cooInt[1],cooInt[2]])((coo+.5).round().astype(int))):
                # Remove duplicates
                if sum([sum(((coo-a[1:4])/r_[psfZPx,psfPx,psfPx])**2)<prm.minSeparation**2 for a in fitResults])==0:
                  fitResults.append(r_[intensity,coo,tilt])
                  if verbose>=3: print('\n    loc %d: Ok           (guess x=%d y=%d z=%d -- fit x=%.1f y=%.1f z=%.1f)'%((i,)+tuple(cooGuess[i][::-1])+tuple(coo[::-1])),end='')
                else:
                  failedLoc.append(r_[intensity,cooGuess[i],3])
                  if verbose>=3: print('\n    loc %d: Duplicate    (guess x=%d y=%d z=%d)'%((i,)+tuple(cooGuess[i][::-1])),end='')
              else:
                failedLoc.append(r_[intensity,cooGuess[i],2])
                if verbose>=3:   print('\n    loc %d: Out of frame (guess x=%d y=%d z=%d)'%((i,)+tuple(cooGuess[i][::-1])),end='')
            else:
              failedLoc.append(r_[0.,cooGuess[i],1])
              if verbose>=3:     print('\n    loc %d: Image border (guess x=%d y=%d z=%d)'%((i,)+tuple(cooGuess[i][::-1])),end='')
          if len(fitResults): fitResults=array(fitResults)[:,[0,3,2,1,4,7,6,5]]; # re-order columns
          else: fitResults=zeros((0,8));
          if len(failedLoc): failedLoc=array(failedLoc)[:,[0,3,2,1,4]]; # re-order columns
          else: failedLoc=zeros((0,5));
      
          # Save results in text files
          fnLoc=fnTif[:-4]+prm.FISHchannels[ch][1]+'.loc'
          if verbose>=2: print(' --> successful: %d (failed: %d - on border: %d)'%(len(fitResults),sum(failedLoc[:,-1]!=1),sum(failedLoc[:,-1]==1)))
          if verbose>=3: print('      Saved "%s"'%(fnLoc)+(saveFailedLoc!=False)*(' and ".failedLoc"'))
          savetxt(os.path.join(dirOut,fnLoc),fitResults,fmt='%.5e',header='cols: intensity, x pos, y pos, z pos, bg offset, bg x tilt, bg y tilt, bg z tilt')
          if saveFailedLoc:
            fnFail=fnTif[:-4]+prm.FISHchannels[ch][1]+'_failed.loc'
            savetxt(os.path.join(dirOut,fnFail),failedLoc,fmt='%.5e',header='cols: intensity, x pos, y pos, z pos, fail code (1: close to image border, 2: converged out of frame, 3: duplicate)')
  


#################################
#####         MAIN         ######
#################################


##########
# 1. Put all the datasets (i.e. micromanager folders) you want to analyze in a given folder (e.g. in D:/Coulon_data/analysis)
# 2. Add the params.py file in this folder if you want to use the same parameters for all datasets. Else put a different params.py file in each subfolder
# 3. Put the correct path in the line below
# 4. Run this FISH_pipeline.py file

batchAnalysis('D:/Coulon_data/analysis')

##########

#dataSetName="20210602_pool2coreDel_E2_coreDel-AF750_JDP2-in1in2-Q670_FOS-CAL610_BATF-in2-Q570_FJB-FAM_20210705KB_5"
#dirOut='D:/a/g/'+dataSetName+'/out/';
#locFileToFieldMask(maskDimensions=(128,2048,2048),
#                   fnLoc2=dirOut+dataSetName+'_FAM.loc2',
#                   distPx=2/0.108,cutoffLow=2.5e5)







