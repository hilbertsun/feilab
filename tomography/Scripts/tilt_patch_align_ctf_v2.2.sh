#!/bin/sh

###########################################
#  This Script is used to do tomographic  #
#  reconstruction automatically starting  #
#  from raw tilt series with fudicial     #
#  markers. CTF correction is performed   #
#  before reconstruction by using WBP     #
#  method.                                #
#                                         #
#  Author : Fei Sun && Yuchen Deng        #
#  @ Fei Sun'lab, Institute of Biophysics #
#  Chinese Academy of Sciences            #
#  Date : Aug. 30th, 2020                 #
#  Contact : feisun@ibp.ac.cn             #
#  Version : 2.0                          #
#  Changes(1.3): Considering the rotation #
#            angle inversion (inv_angle). #
#            Need Tilt_series_CTF_correc  #
#            tion_v1.2.m matlab code      #
#  Changes(1.4): Add markererase function #
#            into the workflow (erase =   #
#            0 or 1 )                     #
#  Changes(1.5): Update markerauto to ver #
#            sion 1.5. In this version,   #
#            geometry parameters will be  #
#            output and used in the foll- #
#            -owing reconstruction.       #
#  Change(1.51): Update Gctf to v0.34 and #
#            change --do_ef_rotave to     #
#            --do_EPA                     #
#  Change(1.6): Update markerauto to ver  #
#            sion 1.5.1. In this version, #
#            geometry parameters will be  #
#            calculated by a separated    #
#            program autogen.             #
#  Change(1.7): Update to fit Gctf_v1.06  #
#  Change(2.0): Replace markerauto by IMOD#
#               patch tracker to perform  #
#               fiducial_free alignment   #
#               of tilt serials.          #
#  Change(2.1): Add geometry determination#
#  Change(2.2): Conver CTF_correct matlab #
#            code to executable program   #
#            that can run under MCR_v92   #
#            enviroment.                  #
#                                         #
###########################################

apix=2.65
ac_volt=300
Cs=2.7
AC=0.07
Defocus_avg=5.0
Defocus_range=3.0

#inv_angle: 1 represents no inversion and -1 for inversion

inv_angle=1

### Setting IMOD ###
source /work/program/imod/IMOD-linux.sh

### Setting EMAN2 ###
source /work/program/EMAN2.11/EMAN2.11/eman2.bashrc

### Setting Gctf ###
export PATH=/usr/local/cuda-7.5/bin:/work/program/Gctf:$PATH; 
export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib:/usr/local/cuda-7.5/lib64/:$LD_LIBRARY_PATH

### ICON ###
GICON_ver="ICON_GPU_v1.2.9"
export PATH=/usr/local/cuda-7.5/bin:/work/program/ICON/${GICON_ver}/build/CentOS64/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64/:/work/program/ICON/${GICON_ver}/build/CentOS64/lib:$LD_LIBRARY_PATH


### AutoGDeterm ###
AUTOGD_PATH="/work/program/AutoGDeterm"
export PATH=${AUTOGD_PATH}/ver1.0/Ubuntu64/:$PATH

### Setting Yuchen Deng's TiltCTFCorrection code path ###
TiltCTFCorr_PATH="/work/program/Tilt_CTF_Corr_v1.0"

### Setting Matlab path ###
#alias usematlab="export PATH=/usr/local/MATLAB/R2012b/bin:$PATH;export LD_LIBRARY_PATH=/usr/local/MATLAB/R2012b/toolbox/dip/Linuxa64/lib"
#alias usematlab="export PATH=/usr/local/MATLAB/R2017a/bin:$PATH"
#usematlab

#useMCR_v92
source /home/fei/bin/matlab_runtime/MCR_v92_setup.bashrc

##################################### Do not change below #################################

if [ $# -lt 6 ] ; then
    cat << EOF
USAGE: sh $0 <tilt_series.mrc> <tilt_angle.rawtlt> <pre-defined rotation angle> <rotation axis : x is 1 or y is 0> <patch size> <bin: 1,2 or 4>
EOF
    exit 0
fi


Def_Low=`echo $Defocus_avg $Defocus_range | gawk '//{print ($1-$2)*10000}'`
Def_High=`echo $Defocus_avg $Defocus_range | gawk '//{print ($1+$2)*10000}'`

file=$1
rawtlt=$2
preangle=$3
ro_axis=$4
patchsize=$5
bin=$6

#thickness=$7
#rotate_v=$8
#lp_filter=$9

if [ -f "$file" ] ; then

        name=`echo "$file" | sed "s/.mrc//g"`

        if test ! -f "$rawtlt" ; then

                echo "ERROR1: $rawtlt does not exist."
                exit 0

        fi

        rm ${name}_autorec.log

        echo "Automatic Tomographic Reconstruction @ Fei Sun's Lab, Version 2.0" | tee -a ${name}_autorec.log

else
        echo "ERROR2: $file does not exist."
        exit 0
fi


###########DO Tilt Serial Alignment Using IMOD Patch Tracker######################

tiltxcorr -StandardInput << eof | tee -a ${name}_autorec.log
InputFile       $file
OutputFile      ${name}.prexf
TiltFile        $rawtlt
RotationAngle   $preangle
ViewsWithMagChanges     1
FilterSigma1    0.03
FilterRadius2   0.25
FilterSigma2    0.05
eof

xftoxg -in ${name}.prexf -g ${name}.prexg -n 0

newstack -StandardInput << eof | tee -a ${name}_autorec.log
InputFile       $file
OutputFile      ${name}.preali
TransformFile   ${name}.prexg
ModeToOutput    0
FloatDensities  2
BinByFactor     1
ImagesAreBinned 2.0
eof

tiltxcorr -StandardInput <<eof | tee -a ${name}_autorec.log
BordersInXandY  102,102
IterateCorrelations     2
SizeOfPatchesXandY      $patchsize, $patchsize
ImagesAreBinned 1
InputFile       ${name}.preali
OutputFile      ${name}_pt.fid
PrealignmentTransformFile       ${name}.prexg
TiltFile        $rawtlt
RotationAngle   $preangle
ViewsWithMagChanges     1
FilterSigma1    0.03
FilterRadius2   0.125
FilterSigma2    0.03
OverlapOfPatchesXandY   0.33,0.33
eof

imodchopconts -StandardInput <<eof | tee -a ${name}_autorec.log
InputModel ${name}_pt.fid
OutputModel ${name}.fid
MinimumOverlap  4
AssignSurfaces 1
eof

tiltalign -StandardInput <<eof | tee -a ${name}_autorec.log
ModelFile       ${name}.fid
ImageFile       ${name}.preali
#ImageSizeXandY 2048,2048
ImagesAreBinned 1
OutputModelFile ${name}.3dmod
OutputResidualFile      ${name}.resid
OutputFidXYZFile        ${name}fid.xyz
OutputTiltFile  ${name}_ali.tlt
OutputXAxisTiltFile     ${name}.xtilt
OutputTransformFile     ${name}.tltxf
OutputFilledInModel     ${name}_nogaps.fid
RotationAngle   $preangle
SeparateGroup   1-16
TiltFile        $rawtlt

#
# ADD a recommended tilt angle change to the existing AngleOffset value
#
AngleOffset     0
RotOption       1
RotDefaultGrouping      5
#
# TiltOption 0 fixes tilts, 2 solves for all tilt angles; change to 5 to solve
# for fewer tilts by grouping views by the amount in TiltDefaultGrouping
#
TiltOption      0
TiltDefaultGrouping     5
MagReferenceView        1
MagOption       0
MagDefaultGrouping      4
#
# To solve for distortion, change both XStretchOption and SkewOption to 3;
# to solve for skew only leave XStretchOption at 0
#
XStretchOption  0
SkewOption      0
XStretchDefaultGrouping 7
SkewDefaultGrouping     11
BeamTiltOption  0
#
# To solve for X axis tilt between two halves of a dataset, set XTiltOption to 4
#
XTiltOption     0
XTiltDefaultGrouping    2000
#
# Criterion # of S.D's above mean residual to report (- for local mean)
#
ResidualReportCriterion 3.0
SurfacesToAnalyze       1
MetroFactor     0.25
MaximumCycles   1000
KFactorScaling  1.0
NoSeparateTiltGroups    1
#
# ADD a recommended amount to shift up to the existing AxisZShift value

#
AxisZShift      0.0
ShiftZFromOriginal     1 
#
# Set to 1 to do local alignments
#
LocalAlignments 0
OutputLocalFile ${name}local.xf
#
# Target size of local patches to solve for in X and Y
#
TargetPatchSizeXandY    700,700
MinSizeOrOverlapXandY   0.5,0.5

#
# Minimum fiducials total and on one surface if two surfaces
#
MinFidsTotalAndEachSurface      8,3
FixXYZCoordinates       0
LocalOutputOptions      1,0,1
LocalRotOption  3
LocalRotDefaultGrouping 6
LocalTiltOption 5
LocalTiltDefaultGrouping        6
LocalMagReferenceView   1
LocalMagOption  3
LocalMagDefaultGrouping 7
LocalXStretchOption     0
LocalXStretchDefaultGrouping    7
LocalSkewOption 0
LocalSkewDefaultGrouping        11
eof


#
# COMBINE TILT TRANSFORMS WITH PREALIGNMENT TRANSFORMS
#
xfproduct -StandardInput <<eof | tee -a ${name}_autorec.log
InputFile1      ${name}.prexg
InputFile2      ${name}.tltxf
OutputFile      ${name}_ali.xf
eof


########### DO CTF estimation Using Kai Zhang's Gctf program ######################
 
	mkdir tmp
	cd tmp
	ln -s ../$file tmp_orig.mrcs

	mrcs_to_mrc.py tmp_orig.mrcs tmp_orig_series 1 3

	Gctf --apix $apix  --kV $ac_volt --Cs $Cs --ac $AC --defL $Def_Low --defH $Def_High --defS 100 --astm 500 --resL 100 --resH 6  --do_EPA 1 *.mrc  | tee -a ${name}_autorec.log		

	gawk '//{if (NF > 3) {print ($3+$4)/2/10000}}' micrographs_all_gctf.star > tmp_orig.defocus

	ctf_num=`ls *.ctf | wc -l`
	ctf_to_hdf_st.py tmp_orig_series_ 3 1 $ctf_num .ctf tmp.hdf
	rm *.ctf *.log *.star *.mrc tmp_orig.mrcs
	mv tmp.hdf ../${name}_ctf.hdf
	mv tmp_orig.defocus ../${name}_defocus.txt
	cd ..
	rm -r -f tmp

######### DO Tilt CTF Correction and Normalization Using YuChen Deng's matlab code #####################

#	fullfile=$PWD/$file
#	fullname=$PWD/$name	
#	echo "cd ${TiltCTFCorr_PATH};" >> tmp.m
#	echo "fun=Tilt_series_CTF_correction();" >> tmp.m
#	echo "fun.TiltSeriesCTFCorr('$fullfile', '${fullname}_ctfCorr.mrcs', '${fullname}_ali.xf', '${fullname}_ali.tlt', '${fullname}_defocus.txt', $inv_angle, $ac_volt, $apix, $Cs, 30, $AC);" >> tmp.m
#	matlab -nodesktop -r "run('tmp.m'), quit" | tee -a  ${name}_autorec.log
#	rm tmp.m

	tiltCTFcorrect $file ${name}_ctfCorr.mrcs ${name}_ali.xf ${name}_ali.tlt ${name}_defocus.txt $inv_angle, $ac_volt, $apix, $Cs, 30, $AC

######### Generating CTF corrected accuratly aligned Tilt series stack ################################

	newstack -input ${name}_ctfCorr.mrcs -output ${name}_ctfCorr_ali_bin${bin}.st -xf ${name}_ali.xf -bin $bin | tee -a ${name}_autorec.log

######## Determination of geometry parameter of section (tilt offset and pitch angle) #################

	AutoGDeterm -i ${name}_ctfCorr_ali_bin${bin}.st -t ${name}_ali.tlt -o ${name}_geo.txt


