################################################################################
##
## FISHingRod -- Image analysis software suite for fluorescence in situ
##               hybridization (FISH) data.
##
## Antoine Coulon, 2024
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


from numpy import *
from scipy import *
from scipy import ndimage, special, optimize
import scipy.stats
import pandas as pd
from matplotlib import pyplot as plt
import os, sys, imp
import copy
from skimage import io
#from skimage.external import tifffile
import warnings
import json

__version__='1.2.0'


toBold=lambda s: '\033[1m%s\033[0m'%(s)


#########################
#########################
class scLoc(ndarray): #https://numpy.org/doc/stable/user/basics.subclassing.html
    """ Single-channel localization dataset."""
    
    def __reduce__(s): #https://stackoverflow.com/questions/26598109/preserve-custom-attributes-when-pickling-subclass-of-numpy-array
        # Get the parent's __reduce__ tuple
        pickled_state = super(scLoc, s).__reduce__()
        # Create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (s.__dict__,)
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(s, state):
        s.__dict__ = state[-1]
        # Call the parent's __setstate__ with the other tuple elements.
        super(scLoc, s).__setstate__(state[0:-1])
        
    def makecopy(s):
        a=s.copy()
        a.__dict__=s.__dict__.copy()
        return a

    def show(s,verboseLevel=1,dataAll=None):
        if verboseLevel>=1: print(("%s – "%toBold(s.name) if type(s.name)!=type(None) else "")+s.fn)
        if s.norm: xlabel='Normalized intensity (%s)'%s.unit
        else: xlabel='Intensity (%s)'%s.unit
        fig,ax=plt.subplots(1,3,sharex='col',figsize=(10,1.5))
        def plotAll(bins):
            if type(dataAll)!=type(None): plt.hist(dataAll[:,0],bins=bins,color='gray');
            plt.hist(s[:,0],bins=bins,color=s.color);
            if type(s.norm)!=type(None):  plt.axvline(1,c='lightgray', ls=':')
            if type(s.cutoffLow)!=type(None):  plt.axvline(s.cutoffLow,c='k',ls=':')
            if type(s.cutoffHigh)!=type(None): plt.axvline(s.cutoffHigh,c='k',ls=':')
        #plt.sca(ax[0]); plt.ylabel('Counts'); plt.xlabel(xlabel);                    plotAll(s.bins)
        plt.sca(ax[0]); plt.ylabel('Counts'); plt.xlabel(xlabel); plt.yscale('log'); plotAll(s.bins)     
        plt.sca(ax[1]); plt.ylabel('Counts'); plt.xlabel(xlabel); plt.xscale('log'); plotAll(s.binsLog)
        plt.sca(ax[2]); plt.ylabel('Counts'); plt.xlabel(xlabel); plt.xscale('log'); plt.yscale('log'); plotAll(s.binsLog)
        plt.tight_layout(); plt.show()
        if verboseLevel>=2:
            print('file name:',      s.fn);
            print('unit:',           s.unit);
            print('norm:',           s.norm);
            print('cutoffLow:',      s.cutoffLow);
            print('cutoffHigh:',     s.cutoffHigh);
            print('equalizeSubPix:', s.equalizeSubPix);
            print('chromaticCorrM:', s.chromaticCorrM);
            print('color:',          s.color);

    def showFrameBias(s):
        frames=sort(list(set(s[:,-2])))
        plt.figure(figsize=(4,3))
        plt.scatter(frames,
                    array([median(s[s[:,-2]==f,0]) for f in frames]),  marker='.', c=s.color)
        #plt.yscale('log')
        plt.ylim(0,None)
        plt.xlabel("Frame number"); plt.ylabel("Median spot intensity (%s)"%s.unit)
        plt.show()

    def showPositionBias(s, nBins=30, xMax=2048, yMax=2048):    
        x=s[:,1]*nBins/xMax; y=s[:,2]*nBins/yMax

        heatmaps=array([[ (lambda a:[a.shape[0],median(a[:,0])])(s[(ix<x)*(x<=(ix+1))*(iy<y)*(y<=(iy+1))]) for ix in r_[:nBins]] for iy in r_[:nBins]])

        fig,ax=plt.subplots(1,3,figsize=(14,3.5))
        plt.sca(ax[0])
        plt.scatter(s[:,1],s[:,2],  marker='.', alpha=0.05, c=s.color)
        plt.xlim(0,2048); plt.ylim(0,2048); plt.gca().set_aspect('equal')
        plt.xlabel("x position (px)"); plt.ylabel("y position (px)")

        plt.sca(ax[1])
        plt.imshow(heatmaps[:,:,0],origin='lower')
        plt.clim(0,None); plt.colorbar(label='Spot density')

        plt.sca(ax[2])
        plt.imshow(heatmaps[:,:,1],origin='lower')
        plt.clim(0,None); plt.colorbar(label="Median spot intensity (%s)"%s.unit)

        plt.show()

    def showFrameAndPositionBiases(s, nBins=30, xMax=2048, yMax=2048):
        
        fig,ax=plt.subplots(1,4,figsize=(13,2.6),gridspec_kw={'width_ratios':[1.2,1,1,1]})

        ### Frame bias
        plt.sca(ax[0])      
        frames=sort(list(set(s[:,-2])))
        plt.scatter(frames,
                    array([median(s[s[:,-2]==f,0]) for f in frames]),  marker='.', c=s.color)
        #plt.yscale('log')
        plt.ylim(0,None)
        plt.xlabel("Frame number"); plt.ylabel("Median spot intensity (%s)"%s.unit)

        ### Position bias
        x=s[:,1]*nBins/xMax; y=s[:,2]*nBins/yMax
        heatmaps=array([[ (lambda a:[a.shape[0],median(a[:,0])])(s[(ix<x)*(x<=(ix+1))*(iy<y)*(y<=(iy+1))]) for ix in r_[:nBins]] for iy in r_[:nBins]])

        plt.sca(ax[1])
        plt.scatter(s[:,1],s[:,2],  marker='.', alpha=0.05, c=s.color)
        plt.xlim(0,2048); plt.ylim(0,2048); plt.gca().set_aspect('equal')
        plt.xlabel("x position (px)"); plt.ylabel("y position (px)")

        plt.sca(ax[2])
        plt.imshow(heatmaps[:,:,0],origin='lower')
        plt.clim(0,None); plt.colorbar(label='Spot density')

        plt.sca(ax[3])
        plt.imshow(heatmaps[:,:,1],origin='lower')
        plt.clim(0,None); plt.colorbar(label="Median spot intensity (%s)"%s.unit)

        plt.tight_layout()
        plt.show()
        

#########################
#########################

def loadLoc2(fn, name=None, unit='a.u.', norm=None, cutoffLow=None, cutoffHigh=None, equalizeSubPix=False, chromaticCorrM=None,
             disp=True, bins=None, binsLog=None, color=None):

    ### Load data
    data=loadtxt(fn)
    #data=pd.read_csv(fn, sep=' ', header=1, names=["intensity", "x_px", "y_px", "z_pl", "bg_offset", "bg_x_tilt", "bg_y_tilt", "bg_z_tilt", "frame", "cellId"])

    ### Apply chromatic shift correction
    if type(chromaticCorrM)==list: chromaticCorrM=array(chromaticCorrM)
    if type(chromaticCorrM)!=type(None):
      if chromaticCorrM.shape in [(16,), (12,)]: chromaticCorrM=chromaticCorrM.reshape((-1,4))
      if chromaticCorrM.shape==(3,4): chromaticCorrM=c_[chromaticCorrM.T,r_[0,0,0,1]].T
      if chromaticCorrM.shape!=(4,4): raise valueError("Chromatin correction matrix not understood.")
      else: data=chromaticCorrData(data,chromaticCorrM)

    ### Normalize intensities
    if type(norm) in [int, float]: data[:,0]/=norm
    elif norm!=None:
      if type(norm)==str: norm=[norm]
      if norm[0] in ['median','med','m']:
        norm=median(data[:,0])        
      elif norm[0] in ['quantile','quant','q']:
        norm=sort(data[:,0])[int((data.shape[0]-1)*norm[1])]
      else: raise ValueError("'norm' paramter not understood.");
      data[:,0]/=norm
      if disp: print('  => Intensity normalization factor: ',norm)
    
    ### Filter based on intensities
    dataAll=data.copy() # Keep copy of the data before filtering
    if cutoffLow:  data=data[where(data[:,0]>cutoffLow)[0]]
    if cutoffHigh: data=data[where(data[:,0]<cutoffHigh)[0]]
    if equalizeSubPix not in [False, None]: ### !!!!!! Assumes voxel size 1x1x1
      if equalizeSubPix in [True,1,'XY','xy','XYZ','xyz']:
        for i in [1,2]: data[:,i]=data[:,i]//1+(scipy.stats.rankdata(data[:,i]%1)-1)/data.shape[0]
      if equalizeSubPix in [True,1,'Z','z','XYZ','xyz']:
        data[:,3]=data[:,3]//1+(scipy.stats.rankdata(data[:,3]%1)-1)/data.shape[0]
        
    ### Prepare display parameter (and displays graphs if `disp` is True)
    if type(bins) in [int,float,ndarray]:
        if type(bins) in [int,float]: bins=r_[:bins]-.5
    else: bins=(r_[0:1:.01]-.005)*1.1*max(data[:,0])
    if type(binsLog)==type(None): binsLog=(10**r_[-1.5:1.5:.02])*median(data[:,0])
    if color==None: color='black'
        
    # Make `scLoc` instance and keep parameters as attributes
    data=data.view(scLoc)
    data.fn            =fn
    data.name          =name
    data.unit          =unit
    data.norm          =norm
    data.cutoffLow     =cutoffLow
    data.cutoffHigh    =cutoffHigh
    data.bins          =bins
    data.binsLog       =binsLog
    data.color         =color
    data.equalizeSubPix=equalizeSubPix
    data.chromaticCorrM=chromaticCorrM
        
    if disp: data.show(dataAll=dataAll)
        
    return data



def chromaticCorrData(d,M):
    return c_[d[:,0], dot(M,c_[d[:,1:4],d[:,1]*0+1].T).T[:,:3] ,d[:,4:]]


def filterLoc2(data,incl=None,excl=None):
    if type(incl)==type(None): cellIds=r_[1:1+int(array(data).T[-1].max())]
    else:
        if type(incl)==str:  incl=loadtxt(incl)
        if type(incl)==list: incl=array(incl)        
        if len(incl.shape)==2: incl=incl[:,0]
        cellIds=incl
    cellIds=set(cellIds)
    if type(excl)!=type(None):
        if type(excl)==str:  excl=loadtxt(excl)
        if type(excl)==list: excl=array(excl)        
        if len(excl.shape)==2: excl=excl[:,0]
        cellIds=cellIds-set(excl)
    if type(data) not in [list,tuple]:
        data2=array([d for d in data if d[-1] in cellIds]).view(scLoc)
        data2.__dict__=data.__dict__
    else:
        data2=[] 
        for dd in data:
            data2.append(array([d for d in dd if d[-1] in cellIds]).view(scLoc))
            data2[-1].__dict__=dd.__dict__
    cellIds=array(sort(list(cellIds))).astype(int)
    return data2,cellIds


    
######################################
######################################
class mcLoc:
    """ Multi-channel localization dataset."""
    def __init__(s, lData, cellIds=None, verbose=True, voxelSize=None, nuclei_csv=None):
        """Initialize `mcLoc()` object.
        - lData: list of numpy arrays from `loadLoc2()`
        - cellIds: list of cell Ids in the data set.
        - verbose: boolean
        - voxelSize: numpy array with 3 elements the pixel size (in um) in x and y and voxel depth. If None, the `params.py` file in the current directory is used, if any. Else the default value is used and a warning is printed.
        """
        s.lData=lData;
        s.nCh=len(s.lData)
        for i,d in enumerate(s.lData):
            if type(d.name)==type(None): d.name='ch%d'%i 
        s.cellIds=cellIds if type(cellIds)!=type(None) else r_[1:1+int(max([d[:,-1].max() for d in s.lData]))]
        if verbose:
            for i,d in enumerate(s.lData): print('%d – %s: %d spots'%(i,toBold(d.name),d.shape[0]))
            print('Cells: %d\n'%len(s.cellIds))
        if type(voxelSize)!=type(None): s.voxelSize=voxelSize
        else:
            if os.path.isfile('./params.py'):
                prm=imp.load_source('prm','./params.py')
                s.voxelSize=r_[prm.pxSize,prm.pxSize,prm.zStep]
                if verbose: print('Using voxel size from `./params.py` (%s)'%repr(s.voxelSize))
            else:
                s.voxelSize=r_[.1075,.1075,.3]
                print(" /!\ Warning: voxel size not provided. "+
                      "Using default values (%s)"%repr(s.voxelSize))
        if type(nuclei_csv)!=type(None):
            if type(nuclei_csv)==pd.core.frame.DataFrame:
                s.nuclei_csv=nuclei_csv
            elif type(nuclei_csv)==str:
                s.nuclei_csv=pd.read_csv(nuclei_csv).rename(columns={' ':'cellIds'});
            else: raise AttributeError("Parameter `nuclei_csv` not understood.")
        s.listSpotsInCells()
    
    def listSpotsInCells(s):
        """Generates a dictionnary 'spotsInCells' where
        - keys are cellIds
        - values are a list with one element per channel, where each element is a list of all the spotIds
        """
        s.spotsInCells=dict([[ cId, [[] for i in range(s.nCh)] ] for cId in s.cellIds])
        for iCh,d in enumerate(s.lData):
            for sId,sp in enumerate(d):
                if int(sp[-1]) in s.cellIds: s.spotsInCells[int(sp[-1])][iCh].append(sId)
        return s.spotsInCells
    
    def generateCtrlLoc(s,nbSpots,mask):
        """ TO DO. Generate a randomly distributed set of localizations.
        The mask can be the nuclear masks (=> uniform distrib within nuclei) or can
        be any grayscale image (=> non-homogenous Poisson distr.)
        Note: add an option to re-use the spot intensity distribution from an
        existing channel.
        """
        pass;
    
    def calcDist(s,channelPairs=None,excludeSelfDirac=True):
        """Calculates distances between spots within cells.
        If `channelPairs` is None, all channels are evaluated against all channels.
        Else, `channelPairs` can be
          - a list of tuples, e.g. [(0,1), (0,2)], specifying the channels pairs
            that have to be evaluated.
          - a square matrix with one raw and column par channel and where a boolean
            value indicate pairs of channels to be evaluated.
        Returns a matrix where res[i,j] is the result for channel i vs channel j:
        an array where columns are:
        distance, intensity in channel i, intensity in channel j,
                     spotId in channel i,    spotId in channel j"""

        if not hasattr(s,'spotsInCells'): s.listSpotsInCells()
            
        if type(channelPairs)==type(None): toCalc=ones((s.nCh,s.nCh),dtype=bool)
        else:
            toCalc=zeros((s.nCh,s.nCh),dtype=bool)
            if type(channelPairs) in [tuple,list]:
                for a in channelPairs: toCalc[a[0],a[1]]=True
            else: toCalc=channelPairs.astype(bool)
        toCalc=toCalc+toCalc.T
        
        dist       =             ndarray((s.nCh,s.nCh),dtype=ndarray)
        distInCells=dict([[ cId, ndarray((s.nCh,s.nCh),dtype=ndarray) ]
                                                          for cId in s.cellIds])
        
        for c0 in range(s.nCh):
            for c1 in range(c0,s.nCh):
                if not toCalc[c0,c1]: continue
                res=[]; d0=s.lData[c0]; d1=s.lData[c1]
                for cId,llSpots in s.spotsInCells.items():
                    resC=[[sum(((d1[i1][1:4]-d0[i0][1:4])*s.voxelSize)**2)**.5,
                            d0[i0][0], d1[i1][0],
                            i0,        i1        ]
                                  for i0 in llSpots[c0] for i1 in llSpots[c1]
                                    if not (excludeSelfDirac and c0==c1 and i0==i1)]
                    distInCells[cId][c0,c1]=array(resC).reshape(-1,5)
                    if c0!=c1: distInCells[cId][c1,c0]=\
                               distInCells[cId][c0,c1][:,[0,2,1,4,3]]
                    res.extend(resC);
                dist[c0,c1]=array(res).reshape(-1,5)
                if c0!=c1: dist[c1,c0]=dist[c0,c1][:,[0,2,1,4,3]]
        s.dist       =dist
        s.distInCells=distInCells
        return dist
    
    def showDist(s, c0, c1, dThr=None, bins=None, binsLog=None):
        fig,ax=plt.subplots(1,2,figsize=(6,2.2));
        if type(bins)   ==type(None): bins=r_[:20:.01]
        if type(binsLog)==type(None): binsLog=10**r_[-2:1.5:.01]
        
        h,d  =histogram(s.dist[c0,c1][:,0],bins=bins); d=(d[1:]+d[:-1])/2
        plt.sca(ax[0]); plt.title(s.lData[c0].name+(" – %s"%(s.lData[c1].name) if c0!=c1 else ""))
        #plt.gca().spines['left'].set_color(s.lData[c0].color)
        plt.plot(d,h,c=s.lData[c1].color);
        #plt.plot(d,h,c=s.lData[c0].color,lw=3,zorder=0);
        #plt.plot(d,h,c=s.lData[c1].color,lw=1);
        plt.xlim(0,None); plt.ylim(0,None);
        plt.xlabel('Distance (µm)'); plt.ylabel('Counts');
        if type(dThr)!=type(None):  plt.axvline(dThr,c='k',ls=':')

        h,d  =histogram(s.dist[c0,c1][:,0],bins=binsLog); d=(d[1:]+d[:-1])/2
        plt.sca(ax[1]); plt.title(s.lData[c0].name+(" – %s"%(s.lData[c1].name) if c0!=c1 else ""))
        #plt.gca().spines['left'].set_color(s.lData[c0].color)
        plt.plot(d,h,c=s.lData[c1].color);
        #plt.plot(d,h,c=s.lData[c0].color,lw=3,zorder=0);
        #plt.plot(d,h,c=s.lData[c1].color,lw=1);
        plt.xlim(.01,None); plt.ylim(1.,None); plt.xscale('log'); plt.yscale('log')
        plt.xlabel('Distance (µm)'); plt.ylabel('Counts');
        if type(dThr)!=type(None):  plt.axvline(dThr,c='k',ls=':')

        plt.tight_layout(); plt.show()

    def listLociInCells(s,cRef,dThr,lCh=None,method=None,verbose=True):
        """For each cell, makes a list of loci defined as the spots in channel `cRef`
        and pair them with spots in all the channels in `lCh`, using the distance
        thresholds in `dThr` (can be a single value or a list with the same length as
        `lCh`) and the method in `method` ('nearest', 'brightest'; can be a single
        value or a list with the same length as `lCh`).
        
        Return: a dictionnary (also stored as attribute `lociInCells`) where
        - keys are cellIds
        - values are:
             lociInCells[cId]
              => [# first locus:
                  [[{sId in channel 0}, {intensity in channel 0}, {distance to ref}],
                   [{sId in channel 1}, {intensity in channel 1}, {distance to ref}],
                   [-1, 0., -1.],               # => If no spot was found for pairing
                   [{sId in channel 3}, {intensity in channel 3}, {distance to ref}],
                   None,                        # => If channel not listed in `lCh`
                  ],
                  [...], # second locus
                  [...], # third locus
                  [...], # fourth locus
                  ...
                 ]
        e.g. > lociInCells[cId][i][c] gives the spotId, intensity and distance (to
               ref spot) of the spot in channel c at the i-th locus in cell cId.
"""
        if type(lCh)==type(None): lCh=range(s.nCh)
        if type(dThr) not in [ndarray, list, tuple]: dThr=[dThr]*len(lCh)
        if type(method)==type(None): method='brightest'
        if type(method)==str:        method=[method]*len(lCh)
        dThr  =[dThr  [lCh.index(c)] if (c in lCh) else None for c in range(s.nCh)]
        method=[method[lCh.index(c)] if (c in lCh) else None for c in range(s.nCh)]
                
        s.cRef=cRef
        s.lociInCells=dict()
        if not hasattr(s,'spotsInCells'): s.listSpotsInCells()

        for cId,llSpots in s.spotsInCells.items(): ### For all cells
            
            # Pre-fill with default values
            s.lociInCells[cId]=loci=\
                [[[-1, 0., -1.] if (c in lCh) else None
                                     for c in range(s.nCh)] for a in llSpots[cRef]]
            if len(llSpots[cRef])==0: continue
            
            for c in range(s.nCh):                 ### For all channels
                dRef=s.lData[cRef]; d=s.lData[c]
                
                if c==cRef:                         # Reference channel
                    for iRef,sIdRef in enumerate(llSpots[cRef]):
                        loci[iRef][c]=[sIdRef,
                                       dRef[sIdRef,0],
                                       0.]
                        
                elif c in lCh and len(llSpots[c]):  # Channel to pair
                    # Inverse of square distances
                    invDistSq=1/maximum(1e-20,array([[
                        sum(((dRef[sIdRef][1:4]-d[sId][1:4])*s.voxelSize)**2)
                               for sId in llSpots[c]] for sIdRef in llSpots[cRef]]))
                    # Intensities in channel c
                    intensity=array([d[sId,0] for sId in llSpots[c]])

                    if   method[c] in ['brightest', 'bright', 'b']:
                        # Remove  spots that are too far by putting them to 0 in:
                        intensity=intensity*(invDistSq.max(0)>=1/dThr[c]**2)

                        nbAssigned=0; # Counter
                        while 1:
                            # Find brightest spot
                            i=argmax(intensity) 
                            if intensity[i]==0.: break

                            # Find nearest reference spot (not already assigned)
                            iRef=argmax(invDistSq[:,i])
                            if invDistSq[iRef,i]>=1/dThr[c]**2:
                                sId=llSpots[c][i]
                                loci[iRef][c]=[sId,
                                               d[sId,0],
                                               1/invDistSq[iRef,i]**.5]
                                # Eliminate assigned spots by putting them to 0 in:
                                invDistSq[iRef,:]=0.; invDistSq[:,i]=0.;
                                nbAssigned+=1
                                if nbAssigned==len(llSpots[cRef]): break
                            # Eliminate the spot by putting it to 0 in:
                            intensity[i]=0.
                            

                    elif method[c] in ['nearest', 'near', 'n']:
                        while 1:
                            # Find the closest pair
                            iRef,i=divmod(argmax(invDistSq),invDistSq.shape[1]);
                            if invDistSq[iRef,i]<1/dThr[c]**2: break
                            sId=llSpots[c][i]
                            loci[iRef][c]=[sId,
                                           d[sId,0],
                                           1/invDistSq[iRef,i]**.5]
                            # Eliminate the assigned spots by putting them to 0 in:
                            invDistSq[iRef,:]=0.; invDistSq[:,i]=0.; 

    def listSpots_pairedAndUnpaired(s,c):
        """Returns two lists of spotIds for spots in channel `c`, which are:
          - all the spots that are paired with a locus (-1 if a locus is not paired with a spot in channel `c`)
          - all the spots that are not paired with a locus,
          followed by the two corresponding filtered versions of the data table."""
        if not hasattr(s,'lociInCells'): raise AttributeError("Missing attribute `lociInCells`. Please run `listLociInCells()` first.")
        
        paired=array([l[c][0] for cId,ll in s.lociInCells.items() for l in ll]).astype(int)
        unpaired=set(range(0,s.lData[c].shape[0]))-set(paired);
        unpaired=array(list(unpaired)).astype(int)

        # Add entry for missing spots (i.e. spotId -1) 
        data=c_[s.lData[c].T,zeros(10)].T
            
        return paired, unpaired, data[paired], data[unpaired]


    def listSpots_nearAndFar(s,cRef,c,dThr):
        """Returns two lists of spotIds for spots in channel `c`, which are:
          - all the spots that are within a distance `dThr` of a spot in channel `cRef`
          - all the spots that are beyond a distance `dThr` of a spot in channel `cRef`,
          followed by the two corresponding filtered versions of the data table."""
        if not hasattr(s,'distInCells'): raise AttributeError("Missing attribute `distInCells`. Please run `calcDist()` first.")
        
        near= array(list(set(concatenate([(lambda d: d[d[:,0]<= dThr ,4])(allDists[cRef,c]) for cId,allDists in s.distInCells.items()]).astype(int))))
        far=set(range(0,s.lData[c].shape[0]))-set(near);
        far=array(list(far)).astype(int)

        return near, far, s.lData[c][near], s.lData[c][far]

                            
                            
                            
######################################
######################################
  
  
  
#######################
def inspectLoc(raw=None,
               cellId=None,
               bbox=None,
               loc=[],
               locVal=None,
               rawCCMs=None,
               dirIn=None,
               dirOut=None,
               includeRaw=True,
               maxProj=True,
               fnTifInspect=None,
               reIndexNuclei=False,
               verbose=None):
  """e.g. inspectLoc('FISH-MK20180206_MCF7-ss_BATF-in1-Cy3_JDP2-in1-Cy5_0.1uM_2_MMStack_1-Pos_009_002.ome.tif',loc=['_Cy3.loc','_Cy5.loc'],locVal=3,includeRaw=1,maxProj=1)
          inspectLoc('FISH-MK20180206_MCF7-ss_BATF-in1-Cy3_JDP2-in1-Cy5_0.1uM_2_MMStack_1-Pos_009_002.ome.tif',loc='_Cy3.loc',locVal='_Cy3_bp.tif')
          inspectLoc('FISH-MK20180206_MCF7-ss_BATF-in1-Cy3_JDP2-in1-Cy5_0.1uM_2_MMStack_1-Pos_009_002.ome.tif',loc=['_Cy3_guess.loc','_Cy3.loc','_Cy3_failed.loc'],locVal=[0,0,4],includeRaw=1,maxProj=1)
          inspectLoc('20180702_IND-E2-40min_BATF-in1-Cy3-0.1uM_FOS-mRNA-Cy5-0.1uM_20180630MK_2_MMStack_2-Pos_001_000.ome.tif',loc=['_Cy3_guess.loc','_Cy3.loc','_Cy3.loc2'],locVal=[0,0,-1],dirIn='D:/Coulon_data/analyzed/20180702_IND-E2-40min_BATF-in1-Cy3-0.1uM_FOS-mRNA-Cy5-0.1uM_20180630MK_2',includeRaw=1,maxProj=1)
          """
  
  gl=globals()
  
  if verbose==None:
    if type(cellId) in [list, ndarray]: verbose=1
    else: verbose=0
  
  # Resolve input and output directories
  if dirIn==None:
    if 'dirIn' in gl.keys(): dirIn=gl['dirIn']
    else: dirIn=os.getcwd()
    #else: raise ValueError("'dirIn' not specified and not defined as a global variable.")
  if dirOut==None: 
    if 'dirOut' in gl.keys(): dirOut=gl['dirOut']
    else: dirOut=os.path.join(dirIn,'out')
    #else: dirOut=os.getcwd()
    #else: raise ValueError("'dirOut' not specified and not defined as a global variable.")
    
  if fnTifInspect==None:
    if 'dirSrc' in gl.keys(): fnTifInspect=os.path.join(gl['dirSrc'],'tmp-inspect.tif')
    else: fnTifInspect='./tmp-inspect.tif'

  #if type(mask)!=list: mask=[mask]
  if type(loc)!=list: loc=[loc]
  if locVal==None: locVal=[None]*len(loc)
  if type(locVal)!=list: locVal=[locVal]*len(loc)
  
  allImRes=[]
  
  if (raw==None) == (type(cellId)==type(None)): raise ValueError("Need either 'raw' or 'cellId' to be defined.")
  if type(cellId)!=type(None):
    if not type(cellId) in [ndarray, list]: lCellIds=[cellId]
    else: lCellIds=cellId
  else: lCellIds=[None]

  # Building and loading cellIdToFileName index
  if os.path.exists(os.path.join(dirOut,'.cellIdToFileName.json')) and not reIndexNuclei:
    cellIdToFileName=dict(json.load(open(os.path.join(dirOut,'.cellIdToFileName.json'))))
  else:
    if verbose>=1: print('Indexing nuclei...')
    nuclFiles=[d for d in os.listdir(dirOut) if d[-9:]=='_nucl.tif']; nuclFiles.sort()
    cellIdToFileName={}
    for nf in nuclFiles:
      cellIdToFileName.update(dict.fromkeys([int(aa) for aa in (set(io.imread(os.path.join(dirOut,nf)).flatten())-{0})],nf))
      if len(lCellIds)==1 and lCellIds[0] in cellIdToFileName.keys(): break
    if len(lCellIds)!=1:
      json.dump(list(cellIdToFileName.items()),open(os.path.join(dirOut,'.cellIdToFileName.json'),'w'))

  for cellId in lCellIds:
    if verbose>=1 and cellId!=None: print('Computing cell %d ...'%cellId)
    bbox2=bbox
    if cellId!=None:
      nf=cellIdToFileName[cellId]
      nuclCoo=c_[where(io.imread(os.path.join(dirOut,nf))==cellId)]
      if nuclCoo.shape[0]==0: raise ValueError("Could not find cell %d."%cellId)
      raw=nf[:-9]+'.tif'
      if type(bbox2)==type(None): bbox2=20.
      if type(bbox2) in [int, float]:
        bbox2=r_[nuclCoo.min(0)[::-1],nuclCoo.max(0)[::-1]]+r_[-1,-1,1,1]*bbox2      

    # locate raw image file
    if os.path.isfile(raw): rawPath=raw
    elif os.path.isfile(os.path.join(dirIn,raw)): rawPath=os.path.join(dirIn,raw)
    else: raise ValueError("Could not locate file '%s'."%raw)
  
    # Open raw image
    if verbose>=2: print('Opening raw image...')
    im=io.imread(rawPath,is_ome=False)
    im=im.reshape(im.shape[:-2][::-1]+im.shape[-2:]).swapaxes(0,1) #!!!!!!!!!!!!
    # /!\ Dimensions should be ZCYX /!\
    
    if type(bbox2)==list: bbox2=array(bbox2)
    if type(bbox2)==type(None): bbox2=r_[0,0,im.shape[-1],im.shape[-2]]
    elif not (type(bbox2)==ndarray and bbox2.shape[0]==4):
        raise ValueError("Parameter 'bbox' not understood.")
    
    if type(rawCCMs)==type(None):
      # Reduce bounding box if outside of image
      bbox2=r_[maximum(0,bbox2[:2]), minimum(r_[im.shape[-1], im.shape[-2]],bbox2[2:])]
    else:
      if len(rawCCMs)!=im.shape[1]:
        raise ValueError("Parameter 'rawCCMs' should be a list with a number of elements equal to the number of channels in the raw image.")
    bbox2=bbox2.round().astype(int)
  
    if includeRaw:
      if type(rawCCMs)==type(None):
        imRes=c_[(im[:,:,bbox2[1]:bbox2[3],bbox2[0]:bbox2[2]]*1.).swapaxes(0,1).T,
                 zeros((len(loc),im.shape[0],bbox2[3]-bbox2[1],bbox2[2]-bbox2[0])).T].T
      else:
        imRes=zeros((im.shape[1]+len(loc),im.shape[0],bbox2[3]-bbox2[1],bbox2[2]-bbox2[0]))
        rawCCMs=[array(ccm).reshape(4,4) if type(ccm)!=type(None) else eye(4) for ccm in rawCCMs]
        cooRaw=(lambda y,z,x: array([x,y,z,z*0+1]))(*meshgrid(r_[bbox2[1]:bbox2[3]],r_[:im.shape[0]],r_[bbox2[0]:bbox2[2]]))
        for i in range(len(rawCCMs)):
          imC=im.swapaxes(0,1)[i]
          cooCorr=dot(cooRaw.T,scipy.linalg.inv(rawCCMs[i]).T).T[:3][::-1]
          imRes[i]=ndimage.interpolation.map_coordinates(imC,cooCorr)
        
    else: imRes=zeros((len(loc),im.shape[0],bbox2[3]-bbox2[1],bbox2[2]-bbox2[0]))
  
  
    for i in range(len(loc)):
      l=loc[i]; lv=locVal[i]
      
      # Locate .loc file
      if type(l) in [ndarray, scLoc]: fitResults=l
      else:
        if os.path.isfile(l): locPath=l
        elif os.path.isfile(raw)     and os.path.isfile(raw[:-4]    +l): locPath=raw[:-4]+l
        elif os.path.isfile(rawPath) and os.path.isfile(rawPath[:-4]+l): locPath=rawPath[:-4]+l
        elif os.path.isfile(os.path.join(dirOut,l)): locPath=os.path.join(dirOut,l)
        elif os.path.isfile(os.path.join(dirOut,raw[:-4]+l)): locPath=os.path.join(dirOut,raw[:-4]+l)
        elif os.path.isfile(os.path.join(dirOut,os.path.basename(raw[:-4]))+l): locPath=os.path.join(dirOut,os.path.basename(raw[:-4]))+l
        elif os.path.isfile(os.path.join(dirOut,os.path.basename(dirIn))+l): locPath=os.path.join(dirOut,os.path.basename(dirIn))+l
        else: raise ValueError("Could not locate loc file from '%s'."%os.path.join(dirOut,raw[:-4]+l))
        if verbose>=2: print("Using loc file '%s'"%locPath);
        fitResults=loadtxt(locPath)
      if fitResults.shape[0]==0: fitResults=zeros((0,10))
      if len(fitResults.shape)==1: fitResults=fitResults.reshape(1,10)
      if type(bbox2)!=type(None):
        fitResults=fitResults[where((bbox2[0]<=fitResults[:,1])
                                    * (fitResults[:,1]<bbox2[2])
                                    * (bbox2[1]<=fitResults[:,2])
                                    * (fitResults[:,2]<bbox2[3]))[0]]
        #fitResults=array([p for p in fitResults if
        #              bbox2[0]<=p[1]<bbox2[2] and bbox2[1]<=p[2]<bbox2[3]])
      if cellId!=None:
        fitResults=fitResults[where(fitResults[:,9]==cellId)[0]]
      else:
        fitResults=array([a for a in fitResults if os.path.basename(raw)[:-4] in cellIdToFileName[int(a[9])]])

      x,y,z=fitResults[:,1:4].T.round().astype(int)
      
      # Locate locVal file
      if type(lv)==int: lv=fitResults[:,lv]
      elif lv==None: lv=1+i
      else:
        if verbose>=2: print('Opening locVal file...')
        if os.path.isfile(lv): lvPath=lv
        elif os.path.isfile(raw) and os.path.isfile(raw[:-4]+lv): lvPath=raw[:-4]+lv
        elif os.path.isfile(os.path.join(dirOut,lv)): lvPath=os.path.join(dirOut,lv)
        elif os.path.isfile(os.path.join(dirOut,raw[:-4]+lv)): lvPath=os.path.join(dirOut,raw[:-4]+lv)
        elif os.path.isfile(os.path.join(dirIn,lv)): lvPath=os.path.join(dirIn,lv)
        elif os.path.isfile(os.path.join(dirIn,raw[:-4]+lv)): lvPath=os.path.join(dirIn,raw[:-4]+lv)
        else: raise ValueError("Could not locate locVal file from '%s'."%lv)
        imLv=io.imread(lvPath)
        lv=imLv[z,y,x]
      
      if verbose>=2: print('Drawing "box" image...')
      for dx,dy in [[-4,-4],[-4,-3],[-4,-2],[-4,2],[-4,3],[-4,4],[-3,4],[-2,4],[2,4],[3,4],[4,4],[4,3],[4,2],[4,-2],[4,-3],[4,-4],[3,-4],[2,-4],[-2,-4],[-3,-4]]:
        imRes[i-len(loc)][z, (y+dy-bbox2[1])%imRes.shape[-2],
                             (x+dx-bbox2[0])%imRes.shape[-1]]=lv
  
    imRes=imRes.swapaxes(0,1)
    if maxProj: imRes=imRes.max(0)
    
    allImRes.append(imRes)


  if len(allImRes)==1: imRes=allImRes[0]
  else:
    if verbose>=2: print('Concatenating all images...')
    imShape=tuple(array([im.shape for im in allImRes]).max(0))
    imRes=zeros((len(allImRes),)+imShape)
    for i,im in enumerate(allImRes):
      offsets=r_[imRes.shape[-1]-im.shape[-1],imRes.shape[-2]-im.shape[-2]]//2
      imRes[i].T[offsets[0]:offsets[0]+im.shape[-1]][:,offsets[1]:offsets[1]+im.shape[-2]]+=im.T
  
  if verbose>=2: print('Saving...')
  if fnTifInspect==None:
    fnTif=os.path.basename(raw)
    fnTifInspect=os.path.join(dirOut,fnTif)[:-4]+'_inspect.tif'
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    #return float32(imRes)
    io.imsave(fnTifInspect,float32(imRes),imagej=True)
    #io.imsave(fnTifInspect,float32(imRes),compress=6)
    #return fnTifInspect
  if verbose>=1: print('Done.')

    
    
#######################
def spotImg(spotId=None,
            bbox=None,
            loc=[],
            locVal=None,
            rawCCMs=None,
            dirIn=None,
            dirOut=None,
            averageProj=True,
            fnTifOutput=None,
            reIndexNuclei=False,
            verbose=None):
  
  gl=globals()
  
  if verbose==None:
    if type(cellId) in [list, ndarray]: verbose=1
    else: verbose=0
  
  # Resolve input and output directories
  if dirIn==None:
    if 'dirIn' in gl.keys(): dirIn=gl['dirIn']
    else: dirIn=os.getcwd()
    #else: raise ValueError("'dirIn' not specified and not defined as a global variable.")
  if dirOut==None: 
    if 'dirOut' in gl.keys(): dirOut=gl['dirOut']
    else: dirOut=os.path.join(dirIn,'out')
    #else: dirOut=os.getcwd()
    #else: raise ValueError("'dirOut' not specified and not defined as a global variable.")
    
  if fnTifInspect==None:
    if 'dirSrc' in gl.keys(): fnTifInspect=os.path.join(gl['dirSrc'],'tmp-inspect.tif')
    else: fnTifInspect='./tmp-inspect.tif'

  #if type(mask)!=list: mask=[mask]
  if type(loc)!=list: loc=[loc]
  if locVal==None: locVal=[None]*len(loc)
  if type(locVal)!=list: locVal=[locVal]*len(loc)
  
  allImRes=[]
  
  if (raw==None) == (type(cellId)==type(None)): raise ValueError("Need either 'raw' or 'cellId' to be defined.")
  if type(cellId)!=type(None):
    if not type(cellId) in [ndarray, list]: lCellIds=[cellId]
    else: lCellIds=cellId
  else: lCellIds=[None]

  # Building and loading cellIdToFileName index
  if os.path.exists(os.path.join(dirOut,'.cellIdToFileName.json')) and not reIndexNuclei:
    cellIdToFileName=dict(json.load(open(os.path.join(dirOut,'.cellIdToFileName.json'))))
  else:
    if verbose>=1: print('Indexing nuclei...')
    nuclFiles=[d for d in os.listdir(dirOut) if d[-9:]=='_nucl.tif']; nuclFiles.sort()
    cellIdToFileName={}
    for nf in nuclFiles:
      cellIdToFileName.update(dict.fromkeys([int(aa) for aa in (set(io.imread(os.path.join(dirOut,nf)).flatten())-{0})],nf))
      if len(lCellIds)==1 and lCellIds[0] in cellIdToFileName.keys(): break
    if len(lCellIds)!=1:
      json.dump(list(cellIdToFileName.items()),open(os.path.join(dirOut,'.cellIdToFileName.json'),'w'))

  for cellId in lCellIds:
    if verbose>=1 and cellId!=None: print('Computing cell %d ...'%cellId)
    bbox2=bbox
    if cellId!=None:
      nf=cellIdToFileName[cellId]
      nuclCoo=c_[where(io.imread(os.path.join(dirOut,nf))==cellId)]
      if nuclCoo.shape[0]==0: raise ValueError("Could not find cell %d."%cellId)
      raw=nf[:-9]+'.tif'
      if type(bbox2)==type(None): bbox2=20.
      if type(bbox2) in [int, float]:
        bbox2=r_[nuclCoo.min(0)[::-1],nuclCoo.max(0)[::-1]]+r_[-1,-1,1,1]*bbox2      

    # locate raw image file
    if os.path.isfile(raw): rawPath=raw
    elif os.path.isfile(os.path.join(dirIn,raw)): rawPath=os.path.join(dirIn,raw)
    else: raise ValueError("Could not locate file '%s'."%raw)
  
    # Open raw image
    if verbose>=2: print('Opening raw image...')
    im=io.imread(rawPath,is_ome=False)
    im=im.reshape(im.shape[:-2][::-1]+im.shape[-2:]).swapaxes(0,1) #!!!!!!!!!!!!
    # /!\ Dimensions should be ZCYX /!\
    
    if type(bbox2)==list: bbox2=array(bbox2)
    if type(bbox2)==type(None): bbox2=r_[0,0,im.shape[-1],im.shape[-2]]
    elif not (type(bbox2)==ndarray and bbox2.shape[0]==4):
        raise ValueError("Parameter 'bbox' not understood.")
    
    if type(rawCCMs)==type(None):
      # Reduce bounding box if outside of image
      bbox2=r_[maximum(0,bbox2[:2]), minimum(r_[im.shape[-1], im.shape[-2]],bbox2[2:])]
    else:
      if len(rawCCMs)!=im.shape[1]:
        raise ValueError("Parameter 'rawCCMs' should be a list with a number of elements equal to the number of channels in the raw image.")
    bbox2=bbox2.round().astype(int)
  
    if includeRaw:
      if type(rawCCMs)==type(None):
        imRes=c_[(im[:,:,bbox2[1]:bbox2[3],bbox2[0]:bbox2[2]]*1.).swapaxes(0,1).T,
                 zeros((len(loc),im.shape[0],bbox2[3]-bbox2[1],bbox2[2]-bbox2[0])).T].T
      else:
        imRes=zeros((im.shape[1]+len(loc),im.shape[0],bbox2[3]-bbox2[1],bbox2[2]-bbox2[0]))
        rawCCMs=[array(ccm).reshape(4,4) if type(ccm)!=type(None) else eye(4) for ccm in rawCCMs]
        cooRaw=(lambda y,z,x: array([x,y,z,z*0+1]))(*meshgrid(r_[bbox2[1]:bbox2[3]],r_[:im.shape[0]],r_[bbox2[0]:bbox2[2]]))
        for i in range(len(rawCCMs)):
          imC=im.swapaxes(0,1)[i]
          cooCorr=dot(cooRaw.T,scipy.linalg.inv(rawCCMs[i]).T).T[:3][::-1]
          imRes[i]=ndimage.interpolation.map_coordinates(imC,cooCorr)
        
    else: imRes=zeros((len(loc),im.shape[0],bbox2[3]-bbox2[1],bbox2[2]-bbox2[0]))
  
  
    for i in range(len(loc)):
      l=loc[i]; lv=locVal[i]
      
      # Locate .loc file
      if type(l) in [ndarray, scLoc]: fitResults=l
      else:
        if os.path.isfile(l): locPath=l
        elif os.path.isfile(raw)     and os.path.isfile(raw[:-4]    +l): locPath=raw[:-4]+l
        elif os.path.isfile(rawPath) and os.path.isfile(rawPath[:-4]+l): locPath=rawPath[:-4]+l
        elif os.path.isfile(os.path.join(dirOut,l)): locPath=os.path.join(dirOut,l)
        elif os.path.isfile(os.path.join(dirOut,raw[:-4]+l)): locPath=os.path.join(dirOut,raw[:-4]+l)
        elif os.path.isfile(os.path.join(dirOut,os.path.basename(raw[:-4]))+l): locPath=os.path.join(dirOut,os.path.basename(raw[:-4]))+l
        elif os.path.isfile(os.path.join(dirOut,os.path.basename(dirIn))+l): locPath=os.path.join(dirOut,os.path.basename(dirIn))+l
        else: raise ValueError("Could not locate loc file from '%s'."%os.path.join(dirOut,raw[:-4]+l))
        if verbose>=2: print("Using loc file '%s'"%locPath);
        fitResults=loadtxt(locPath)
      if fitResults.shape[0]==0: fitResults=zeros((0,10))
      if len(fitResults.shape)==1: fitResults=fitResults.reshape(1,10)
      if type(bbox2)!=type(None):
        fitResults=fitResults[where((bbox2[0]<=fitResults[:,1])
                                    * (fitResults[:,1]<bbox2[2])
                                    * (bbox2[1]<=fitResults[:,2])
                                    * (fitResults[:,2]<bbox2[3]))[0]]
        #fitResults=array([p for p in fitResults if
        #              bbox2[0]<=p[1]<bbox2[2] and bbox2[1]<=p[2]<bbox2[3]])
      if cellId!=None:
        fitResults=fitResults[where(fitResults[:,9]==cellId)[0]]
      else:
        fitResults=array([a for a in fitResults if os.path.basename(raw)[:-4] in cellIdToFileName[int(a[9])]])

      x,y,z=fitResults[:,1:4].T.round().astype(int)
      
      # Locate locVal file
      if type(lv)==int: lv=fitResults[:,lv]
      elif lv==None: lv=1+i
      else:
        if verbose>=2: print('Opening locVal file...')
        if os.path.isfile(lv): lvPath=lv
        elif os.path.isfile(raw) and os.path.isfile(raw[:-4]+lv): lvPath=raw[:-4]+lv
        elif os.path.isfile(os.path.join(dirOut,lv)): lvPath=os.path.join(dirOut,lv)
        elif os.path.isfile(os.path.join(dirOut,raw[:-4]+lv)): lvPath=os.path.join(dirOut,raw[:-4]+lv)
        elif os.path.isfile(os.path.join(dirIn,lv)): lvPath=os.path.join(dirIn,lv)
        elif os.path.isfile(os.path.join(dirIn,raw[:-4]+lv)): lvPath=os.path.join(dirIn,raw[:-4]+lv)
        else: raise ValueError("Could not locate locVal file from '%s'."%lv)
        imLv=io.imread(lvPath)
        lv=imLv[z,y,x]
      
      if verbose>=2: print('Drawing "box" image...')
      for dx,dy in [[-4,-4],[-4,-3],[-4,-2],[-4,2],[-4,3],[-4,4],[-3,4],[-2,4],[2,4],[3,4],[4,4],[4,3],[4,2],[4,-2],[4,-3],[4,-4],[3,-4],[2,-4],[-2,-4],[-3,-4]]:
        imRes[i-len(loc)][z, (y+dy-bbox2[1])%imRes.shape[-2],
                             (x+dx-bbox2[0])%imRes.shape[-1]]=lv
  
    imRes=imRes.swapaxes(0,1)
    if maxProj: imRes=imRes.max(0)
    
    allImRes.append(imRes)


  if len(allImRes)==1: imRes=allImRes[0]
  else:
    if verbose>=2: print('Concatenating all images...')
    imShape=tuple(array([im.shape for im in allImRes]).max(0))
    imRes=zeros((len(allImRes),)+imShape)
    for i,im in enumerate(allImRes):
      offsets=r_[imRes.shape[-1]-im.shape[-1],imRes.shape[-2]-im.shape[-2]]//2
      imRes[i].T[offsets[0]:offsets[0]+im.shape[-1]][:,offsets[1]:offsets[1]+im.shape[-2]]+=im.T
  
  if verbose>=2: print('Saving...')
  if fnTifInspect==None:
    fnTif=os.path.basename(raw)
    fnTifInspect=os.path.join(dirOut,fnTif)[:-4]+'_inspect.tif'
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    #return float32(imRes)
    io.imsave(fnTifInspect,float32(imRes),imagej=True)
    #io.imsave(fnTifInspect,float32(imRes),compress=6)
    #return fnTifInspect
  if verbose>=1: print('Done.')

    
    
    
    
####################################
####################################
#### Evaluate FDR and FNR from fluorescence intensity distributions ####

## Convertions
logToPow=lambda muLog,sdLog: [10**(muLog+sdLog**2/2), ((10**(sdLog**2)-1)*10**(2*muLog+sdLog**2))**.5]
powToLog=lambda muPow,sdPow: [log10(muPow**2/(muPow**2+sdPow**2)**.5), log10(1+sdPow**2/muPow**2)**.5]

## Probability density functions
pdf_Poisson_log10=lambda z,lm,cv: exp(-(2*10**(z-lm)*log(cv)+1)/cv**2-special.gammaln(1+10**(z-lm)/cv**2))/cv**2*log(10)*10**(z-lm)
pdf_Poisson_log10_weighted=lambda x,p: p[4]*(lambda a: a/(sum(a)*(x[1]-x[0])))(  p[4]*pdf_Poisson_log10(x,p[0],p[1])  *exp(-minimum(0,x-p[2])**2/2/p[3]**2))
ccdf=lambda x,m,sd: (1-special.erf((x-m)/(sd*2**.5)))/2

## Models to fit fluo. distributions
twoGaussians=lambda x,p: p[4]/(p[1]*(2*pi)**.5)*exp(-(x-p[0])**2/2/p[1]**2)+(1-p[4])/(p[3]*(2*pi)**.5)*exp(-(x-p[2])**2/2/p[3]**2)
oneGaussian=lambda x,p: p[2]/(p[1]*(2*pi)**.5)*exp(-(x-p[0])**2/2/p[1]**2)
#Poission_and_Gaussian=lambda x,p: p[4]*pdf_Poisson_log10(x,p[0],p[1]) +(1-p[4])/(p[3]*(2*pi)**.5)*exp(-(x-p[2])**2/2/p[3]**2)
Poission_and_Gaussian=lambda x,p: pdf_Poisson_log10_weighted(x,p)  +  (1-p[4])/(p[3]*(2*pi)**.5)*exp(-(x-p[2])**2/2/p[3]**2)


def evalSingleFluoDistr(d, guessLog10_sig=6, guessLog10_sig_sd=0.2,
                           guessLog10_bg=4.5, guessLog10_bg_sd=0.2, guess_fracSignalSpots=0.5,
                           log10_bins=r_[3:7:.01], performFit=True, disp=True, fnPDF=None):

    #twoGaussians_mSq=lambda p: mean((pdf-twoGaussians(fluoLog10,p))**2)
    #oneGaussian_WmSq=lambda p: mean((10**fluoLog10)*(pdf-oneGaussian(fluoLog10,p))**2)
    #twoGaussians_WmSq=lambda p: mean((  fluoLog10  )*(pdf-twoGaussians(fluoLog10,p))**2)
    Poission_and_Gaussian_mSq=lambda p: mean((pdf-Poission_and_Gaussian(fluoLog10,p))**2)
    
    
    ## If a fluo. distribution is provided:
    if type(d) not in [int, float, type(None)]:
        pdf,fluoLog10=histogram(log10(d[:,0]),density=1,bins=log10_bins);
        fluoLog10=(fluoLog10[1:]+fluoLog10[:-1])/2
        if performFit:
            fit = optimize.fmin(Poission_and_Gaussian_mSq,
                                r_[guessLog10_sig, guessLog10_sig_sd,
                                   guessLog10_bg,  guessLog10_bg_sd,  guess_fracSignalSpots],disp=False)
        else:
            fit=r_[guessLog10_sig,guessLog10_sig_sd,guessLog10_bg,guessLog10_bg_sd,guess_fracSignalSpots]
    ## If a fluo. distribution not is provided:
    else:                                      
        fluoLog10=(log10_bins[1:]+log10_bins[:-1])/2
        pdf=fluoLog10*0
        fit=r_[guessLog10_sig,guessLog10_sig_sd,guessLog10_bg,guessLog10_bg_sd,guess_fracSignalSpots]

    ## Calculate background spot distribution by subtracting the true positive
    #pdf_background=pdf-fit[4]*pdf_Poisson_log10(fluoLog10,fit[0],fit[1])
    pdf_background = pdf-pdf_Poisson_log10_weighted(fluoLog10,fit) 
    
    ## Fit right decay of background spot distribution with a power law
    #i0=where(fluoLog10>(.15+sum(fluoLog10*pdf_background)/sum(pdf_background)))[0][0]
    i0=where(fluoLog10>(fit[2]+fit[3]))[0][0]
    bg_fit=optimize.fmin(lambda p: sum((pdf_background[i0:]-10**(-p[0]*(fluoLog10[i0:]-p[1])))**2),
                         r_[4,5],disp=False)
    
    ## FNR is the probability that a 'signal' spot is missed
    #      is the cumulative distribution of the 'signal' distribution
    FNR =cumsum(pdf_Poisson_log10_weighted(fluoLog10,fit)/fit[4])*(fluoLog10[1]-fluoLog10[0])
   
    ## FDR is the probability that a selected spot is a background spot
    #      is the ccdf of the background divided by the ccdf of the full distribution.
    FDR =  cumsum(  (10**(-bg_fit[0]*(fluoLog10-bg_fit[1]))      )[::-1])[::-1]\
          /cumsum(  (pdf_Poisson_log10_weighted(fluoLog10,fit)\
                     +10**(-bg_fit[0]*(fluoLog10-bg_fit[1]))     )[::-1])[::-1]
    
    #https://en.wikipedia.org/wiki/Precision_and_recall
    

    intercept=where((FDR-FNR<0)*1)[0]
    if intercept.shape[0]: FDR_eq_FNR=((FNR+FDR)/2)[intercept[0]]
    else:            FDR_eq_FNR=NaN
    area_FDR_FNR =sum((FDR[1:]+FDR[:-1])/2*diff(FNR))

    if disp:
        print("----------------------------------------------")
        print("Mean & s.d. of log10(signal)     = %.2f ± %.2f"%(fit[0],fit[1]))
        print("Slope of background power-law    = %.2f"%(-bg_fit[0]))
        print("Fraction of 'signal' spots       = %.1f %%"%(100*fit[4]))
        print("----------------------------------------------")
        print("Value where FDR = FNR (log10)    = %.2f"%log10(FDR_eq_FNR))
        print("Area under FDR-vs-FNR curve (log10) = %.2f"%log10(area_FDR_FNR))
        print("----------------------------------------------")
        
        
        fig,ax=plt.subplots(3,2,sharex='col',figsize=(8,6),gridspec_kw={'width_ratios':[3,2],'height_ratios':[2.5,1,1]})

        ax_=ax[0,0]
        ax_.plot(10**fluoLog10,pdf)
        #ax_.plot(10**fluoLog10,Poission_and_Gaussian(fluoLog10,fit))
        ax_.plot(10**fluoLog10[i0:],(pdf_Poisson_log10_weighted(fluoLog10,fit)+10**(-bg_fit[0]*(fluoLog10-bg_fit[1])))[i0:])
        #ax_.plot(10**fluoLog10,fit[4]*pdf_Poisson_log10(fluoLog10,fit[0],fit[1]))
        ax_.plot(10**fluoLog10,pdf_background,zorder=0,c='lightgray')
        ax_.set_xscale('log'); #ax_.set_xlim(1e3,1e7)
        ax_.set_ylabel('Probability density')

        ax_=ax[1,0]
        ax_.plot(10**fluoLog10,pdf_background,c='lightgray',zorder=0)
        ax_.plot(10**fluoLog10[i0:],pdf_background[i0:],c='C02')
        ax_.plot(10**fluoLog10,10**(-bg_fit[0]*(fluoLog10-bg_fit[1])),c='k',ls=':')        
        ax_.set_xscale('log'); #ax_.set_xlim(1e3,1e7)
        ax_.set_yscale('log'); ax_.set_ylim(1e-3,1e1)
        ax_.set_ylabel('Probability density')

        ax_=ax[2,0]
        ax_.plot(10**fluoLog10,FDR,label='FDR')
        ax_.plot(10**fluoLog10,FNR,label='FNR')
        ax_.set_xscale('log'); #ax_.set_xlim(1e3,1e7)
        #ax_.set_yscale('log'); ax_.set_ylim(1e-5,1e0)
        ax_.set_ylim(0,1)
        ax_.set_xlabel('Fluorescence intensity (a.u.)'); ax_.set_ylabel('Probability')
        ax_.legend()#loc='upper left')

        ax[0,1].remove()
        ax_=fig.add_subplot(fig.gca().get_gridspec()[0,1])
        #ax_.set_xscale('log'); ax_.set_xlim(1e-4,1e0); ax_.set_yscale('log'); ax_.set_ylim(1e-4,1e0); ax_.grid()
        ax_.set_xlim(-.02,1); ax_.set_ylim(-.02,1); ax_.grid()
        ax_.plot(FNR,FNR,c='lightgray')
        ax_.plot(FNR,FDR,label='')
        ax_.set_ylabel('FDR')

        ax[1,1].remove()
        ax[2,1].remove()
        ax_=fig.add_subplot(fig.gca().get_gridspec()[1:,1])
        ax_.set_xscale('log'); ax_.set_xlim(1e-4,1e0); ax_.set_yscale('log'); ax_.set_ylim(1e-4,1e0); ax_.grid()
        #ax_.set_xlim(-.02,1); ax_.set_ylim(-.02,1); 
        ax_.plot(FNR,FNR,c='lightgray')
        ax_.plot(FNR,FDR,label='')
        ax_.set_xlabel('FNR'); ax_.set_ylabel('FDR')

        plt.tight_layout()
        if fnPDF is None: plt.show()
        else: plt.savefig(fnPDF)
        
    return fit, fluoLog10, pdf, pdf_background, FNR, FDR, FDR_eq_FNR, area_FDR_FNR



def evalPositiveAndNegativeFluoDistr(d_signal, d_background, fracSignalSpots=.5,
                                     log10_bins=r_[3:7:.01], disp=True, fnPDF=None):
    
    pdf_signal,    fluoLog10 = histogram(log10(d_signal[:,0]),    density=1,bins=log10_bins);
    pdf_background,fluoLog10 = histogram(log10(d_background[:,0]),density=1,bins=log10_bins);
    fluoLog10=(fluoLog10[1:]+fluoLog10[:-1])/2
    
    ## FNR is the probability that a 'signal' spot is missed
    #      is the cumulative distribution of the 'signal' distribution
    FNR =cumsum(pdf_signal)*(fluoLog10[1]-fluoLog10[0])
    
    ## FDR is the probability that a selected spot is a background spot
    #      is the ccdf of the background divided by the ccdf of the full distribution.
    FDR =  cumsum(( (1-fracSignalSpots)*pdf_background                              )[::-1])[::-1]\
          /cumsum(( (1-fracSignalSpots)*pdf_background + fracSignalSpots*pdf_signal )[::-1])[::-1]
    
    #https://en.wikipedia.org/wiki/Precision_and_recall
    

    intercept=where((FDR-FNR<0)*1)[0]
    if intercept.shape[0]: FDR_eq_FNR=((FNR+FDR)/2)[intercept[0]]
    else:            FDR_eq_FNR=NaN
    area_FDR_FNR =sum((lambda a:(a[1:]+a[:-1])/2)(nan_to_num(FDR,nan=1.))*diff(FNR))

    if disp:
        print("----------------------------------------------")
        print("Mean & s.d. of log10(signal)        = %.2f ± %.2f"%(
            mean(log10(d_signal[d_signal[:,0]!=0,0])),
            var( log10(d_signal[d_signal[:,0]!=0,0]))**.5))
        print("Assuming fraction of 'signal' spots = %.1f %%"%(100*fracSignalSpots))
        print("----------------------------------------------")
        print("Value where FDR = FNR (log10)       = %.2f"%log10(FDR_eq_FNR))
        print("Area under FDR-vs-FNR curve (log10) = %.2f"%log10(area_FDR_FNR))
        print("----------------------------------------------")
        
        
        fig,ax=plt.subplots(3,2,sharex='col',figsize=(8,6),gridspec_kw={'width_ratios':[3,2],'height_ratios':[2.5,1,1]})

        ax_=ax[0,0]
        ax_.plot(10**fluoLog10,pdf_signal,    c='C1')
        ax_.plot(10**fluoLog10,pdf_background,c='C0')
        ax_.set_xscale('log'); #ax_.set_xlim(1e3,1e7)
        ax_.set_ylabel('Probability density')

        ax_=ax[1,0]
        ax_.plot(10**fluoLog10,pdf_signal,    c='C1')
        ax_.plot(10**fluoLog10,pdf_background,c='C0')
        ax_.set_xscale('log'); #ax_.set_xlim(1e3,1e7)
        ax_.set_yscale('log'); ax_.set_ylim(1e-3,1e1)
        ax_.set_ylabel('Probability density')

        ax_=ax[2,0]
        ax_.plot(10**fluoLog10,FDR,label='FDR')
        ax_.plot(10**fluoLog10,FNR,label='FNR')
        ax_.set_xscale('log'); #ax_.set_xlim(1e3,1e7)
        #ax_.set_yscale('log'); ax_.set_ylim(1e-5,1e0)
        ax_.set_ylim(0,1)
        ax_.set_xlabel('Fluorescence intensity (a.u.)'); ax_.set_ylabel('Probability')
        ax_.legend()#loc='upper left')

        ax[0,1].remove()
        ax_=fig.add_subplot(fig.gca().get_gridspec()[0,1])
        ax_.set_xlim(-.02,1); ax_.set_ylim(-.02,1); ax_.grid()
        ax_.plot(FNR,FNR,c='lightgray')
        ax_.plot(FNR,FDR,label='')
        ax_.set_ylabel('FDR')

        ax[1,1].remove()
        ax[2,1].remove()
        ax_=fig.add_subplot(fig.gca().get_gridspec()[1:,1])
        ax_.set_xscale('log'); ax_.set_xlim(1e-4,1e0); ax_.set_yscale('log'); ax_.set_ylim(1e-4,1e0); ax_.grid()
        ax_.plot(FNR,FNR,c='lightgray')
        ax_.plot(FNR,FDR,label='')
        ax_.set_xlabel('FNR'); ax_.set_ylabel('FDR')

        plt.tight_layout()
        if fnPDF is None: plt.show()
        else: plt.savefig(fnPDF)
        
    return fracSignalSpots, fluoLog10, pdf_signal, pdf_background, FNR, FDR, FDR_eq_FNR, area_FDR_FNR



