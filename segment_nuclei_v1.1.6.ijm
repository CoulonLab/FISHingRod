////////////////////////////////////////////////////////////////////////////////
//
// FISHingRod -- Image analysis software suite for fluorescence in situ
//               hybridization (FISH) data.
//
// Antoine Coulon, 2022
// Contact: <antoine.coulon@curie.fr>
//
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <https://www.gnu.org/licenses/>.
//        
////////////////////////////////////////////////////////////////////////////////

// version 1.1.6

//////////////////////////////////////
///////////// PARAMETERS /////////////

inclFnRE=".*.*"
exclFnRE=".*_nucl.*"

listChannels="5" // (1-based) //Segmentation will be performed on the first one. All will be measured
zSliceStart=8 // (1-based)
zSliceStop =15 // (1-based)

DAPIthreshold=1500 //
separateNuclei=false;

nuclAreaMin=110. // in µm^2
nuclAreaMax=430. // in µm^2

batchMode=false;
displayResult=true;
multipleDatasets=false;
UseDefaultOutputDirectory=true;

//////////////////////////////////////
//////////////////////////////////////

if (!multipleDatasets) {
  dirIn=getDirectory("Choose directory with raw images");
  nbDatasets=1; }
else {
  if (!UseDefaultOutputDirectory) {
  	print("Warning: overriding 'UseDefaultOutputDirectory'. Default output directories are used when analyzing multiple datasets.");
  	UseDefaultOutputDirectory=true; }
  rootDirIn=getDirectory("Choose directory with subdirectories containing raw images");
  lDirIn=getFileList(rootDirIn);
  nbDatasets=lDirIn.length;
}

if (batchMode) { setBatchMode(true); }

for (ii=0; ii<nbDatasets; ii++) {
  if (multipleDatasets) { dirIn=rootDirIn+lDirIn[ii]; }
  print(dirIn);
  if (endsWith(dirIn, "/") | endsWith(dirIn, "\\")) {

    if (!UseDefaultOutputDirectory) { dirOut=getDirectory("Choose output directory"); }
    else { dirOut=dirIn+"out/"; }

    if (!File.exists(dirOut)) { File.makeDirectory(dirOut); }

    list=getFileList(dirIn);
    Array.sort(list);
    //run("Set Measurements...", "area mean standard min perimeter shape integrated median redirect=None decimal=3");
    run("Set Measurements...", "area mean standard min center perimeter shape integrated median skewness kurtosis redirect=None decimal=3");
    run("Clear Results");
    listDAPIimages="";
    clearCounter="clear";
    for (i=0; i<list.length; i++) {
      if (matches(list[i],inclFnRE)
          & !matches(list[i],exclFnRE)
          & endsWith(list[i], ".tif")) {

        
        open(""+dirIn+list[i]);
        //run("TIFF Virtual Stack...", "open="+dirIn+list[i]);

        getPixelSize (unit, pixelWidth, pixelHeight);
        if ((unit!='micron') & (unit!="µm") & (unit!="um")) {
    	    print("Warning: no pixel size information. Using default values.");
    	    pixelWidth=0.108;
    	    pixelHeight=pixelWidth;
        }
        title=getTitle();
        if (endsWith(title,".tif")) title2=substring(title,0,lengthOf(title)-4);
        else title2=title;
        rename('tmp1');

        //run("Duplicate...", "title=tmp2 duplicate channels="+DAPIchannel+" slices="+zSliceStart+"-"+zSliceStop);
        run("Make Substack...", "channels="+listChannels+" slices="+zSliceStart+"-"+zSliceStop); rename("tmp2");

        if (zSliceStart!=zSliceStop) {
          run("Z Project...", "projection=[Max Intensity]");
          rename("tmp"); close("tmp2"); close("tmp1");
        } else { rename("tmp"); close("tmp1"); }
        run("Grays");

        if (displayResult) {
          nameDAPI="tmpDAPI_"+IJ.pad(i,4);
          run("Duplicate...", "duplicate title="+nameDAPI);
          listDAPIimages=listDAPIimages+"image"+(i+1)+"="+nameDAPI+" ";
        }

        run("Duplicate...", "title=tmp3");
        run("Subtract Background...", "rolling=50 sliding");
        run("Gaussian Blur...", "sigma=2 stack");

	    ///// Choose here between manual and automatic thresholding
        //run("Convert to Mask", "method=Li background=Dark");
        setOption("BlackBackground", true); setThreshold(DAPIthreshold,65535); run("Convert to Mask","background=Dark");

        run("Fill Holes", "stack");
        if (separateNuclei) { run("Watershed", "stack");
 }        
		print("size="+(nuclAreaMin/(pixelWidth*pixelHeight))+"-"+(nuclAreaMax/(pixelWidth*pixelHeight))+"   w:"+pixelWidth+" h:"+pixelHeight);
        //run("Analyze Particles...", "size="+(nuclAreaMin)+"-"+(nuclAreaMax)+" show=[Count Masks] exclude "+clearCounter+" display add stack");
        //run("Analyze Particles...", "size="+(nuclAreaMin/(pixelWidth*pixelHeight))+"-"+(nuclAreaMax/(pixelWidth*pixelHeight))+" show=[Count Masks] exclude "+clearCounter+" display add stack");
        run("Analyze Particles...", "size="+(nuclAreaMin/(pixelWidth*pixelHeight))+"-"+(nuclAreaMax/(pixelWidth*pixelHeight))+" pixel show=[Count Masks] exclude "+clearCounter+" display add stack");
        clearCounter="";
        run("3-3-2 RGB");
        resetMinAndMax();
        //setMinAndMax(0,1);
        rename(title2+"_nucl.tif");
        save(dirOut+title2+"_nucl.tif");
        //if (!displayResult) { close(); }
        close();
        
        //close("tmp3");
        //selectWindow("tmp3"); roiManager("Show None"); close(); roiManager("Show All");
        selectWindow("tmp3"); close();

        selectWindow("tmp");
        //List.setMeasurements(); run("Subtract...", "value="+List.getValue("Median"));
        //roiManager("multi-measure measure_all append"); // !!!!! /!\ Comment out if using microscope 1 computer
        if (roiManager("count")!=0) { roiManager("Delete"); }
        close("tmp");
      }
    }

    close("ROI Manager");
    saveAs("Results", dirOut+File.getName(dirIn)+"_segmented_nuclei.csv");

    
if (batchMode) { setBatchMode(false); }

    if (!displayResult) { close("Results"); }
    if (displayResult) {
      run("Image Sequence...", "open="+dirOut+" file=ome_nucl sort");
      rename("all_Count_masks");

      if (!batchMode) {
        run("Concatenate...", " title=all_DAPI "+listDAPIimages);

        if (0) { run("Merge Channels...", "c1=all_DAPI c2=all_Count_masks create"); }
        else {
          selectWindow("all_Count_masks");
          run("Duplicate...", "duplicate title=Outlines");
          setThreshold(0.5,65535); run("Convert to Mask","background=Dark");
          setThreshold(0.5,65535); run("Convert to Mask","background=Dark");
          setOption("BlackBackground", false); run("Outline", "stack"); run("Green"); run("16-bit");
          selectWindow("all_DAPI");
          getDimensions(w, h, channels, slices, frames);
          if (channels==1) { run("Merge Channels...", "c1=all_DAPI c2=all_Count_masks c3=Outlines create"); }
          if (channels==2) { run("Split Channels");
              run("Merge Channels...", "c1=C1-all_DAPI c2=all_Count_masks c3=Outlines c4=C2-all_DAPI create"); }
          if (channels==3) { run("Split Channels");
              run("Merge Channels...", "c1=C1-all_DAPI c2=all_Count_masks c3=Outlines c4=C2-all_DAPI c5=C3-all_DAPI create"); }
          //Stack.setActiveChannels("101"); run("Next Slice [>]");
          save(dirOut+File.getName(dirIn)+"_segmented_nuclei.tif");
        }
      }
    }
    if (multipleDatasets) { close(); }
  }
}
