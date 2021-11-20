#!/bin/sh

###########################################
#  This Script is used to do tomographic  #
#  reconstruction using ICON.             #
#  The input needs a CTF corrected stack  #
#  and an xf file with precise alignment  #
#  parameters.                            #
#                                         #
#                                         #
#  Author : Fei Sun                       #
#  @ Fei Sun'lab, Institute of Biophysics #
#  Chinese Academy of Sciences            #
#  Date : Aug. 18th, 2015                 #
#  Contact : feisun@ibp.ac.cn             #
#                                         #
###########################################

apix=2.27

###For cryoET dataset###
datatype=1
###For nsEM dataset###
#datatype=0


### Setting IMOD ###
source /work/program/imod/IMOD-linux.sh

### AutoGDeterm ###
AUTOGD_PATH="/work/program/AutoGDeterm"
export PATH=${AUTOGD_PATH}/ver1.0/Ubuntu64/:$PATH

### Setting GICON ###
GICON_ver="ICON_GPU_v1.2.9"
alias useGICON="export PATH=/usr/local/cuda-7.5/bin:/work/program/ICON/${GICON_ver}/build/CentOS64/bin:$PATH;export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64/:/work/program/ICON/${GICON_ver}/build/CentOS64/lib:$LD_LIBRARY_PATH"
useGICON

##################################### Do not change below #################################

if [ $# -lt 5 ] ; then
    cat << EOF
USAGE: sh $0 <ctf_corrected_tilt_series.st> <tilt_ali.xf> <tilt_ali.tlt> <thickness_add in pixel> <bin 2,4,8>
EOF
    exit 0
fi

input=$1
ali_xf=$2
ali_tlt=$3
thickness_add=$4
bin=$5

	if test ! -f "$ali_xf" ; then
		
		echo "ERROR: $ali_xf does not exist."
	        exit 0

	fi
	
	if test ! -f "$ali_tlt"; then

		echo "ERROR: $ali_tlt does not exist."
		exit 0
	fi

	if test ! -f "$input"; then

		echo "ERROR: $input does not exist."
		exit 0

	fi

	name=`echo "$input" | sed "s/.st//g"`
	rm ${name}_icon.log	

	echo "Tomographic Reconstruction using ICON @ Fei Sun's Lab, Version 1.1" | tee -a ${name}_icon.log

######### Preprocessing before ICON reconstruction #######################

	ICONPreProcess -i $input -t $ali_tlt -th $thickness_add -o ${name}_pre.mrc | tee -a ${name}_icon.log


######### Generating aligned Tilt series stack for ICON ################################

	newstack -input ${name}_pre.mrc -output ${name}_pre_ali_bin${bin}.mrc -xf $ali_xf -bin $bin | tee -a ${name}_icon.log
        

######## Get Information of Image Size of Each Section ##############################

	sizeinfo=`header --size ${name}_pre_ali_bin${bin}.mrc`
	size_x=`echo $sizeinfo | gawk '//{print $1}'`
	size_y=`echo $sizeinfo | gawk '//{print $2}'`
	size_z=`echo $sizeinfo | gawk '//{print $3}'`
	slice=`echo $sizeinfo | gawk '//{print $1-1}'`


######## Determination of geometry parameter of section (tilt offset and pitch angle) #################

        AutoGDeterm -i ${name}_pre_ali_bin${bin}.mrc -t ${ali_tlt} -o ${name}_geo.txt

	thickness=`gawk '/thickness/{print int($3)}' ${name}_geo.txt`
	geo_thickness=`echo $thickness $thickness_add | gawk '//{print int($1+$2)}'`
	geo_zshift=`gawk '/zshift/{print $3}' ${name}_geo.txt`

	geo_pitch=`gawk '/azimuthal angle/{print $4}' ${name}_geo.txt`
	geo_tiltoffset=`gawk '/tilt angle offset/{print $5}' ${name}_geo.txt`

	rot_x=${geo_pitch} 
	rot_y=`echo ${geo_tiltoffset} | gawk '//{print -1*$1}'`
	center_x=`echo ${slice} | gawk '//{print int(($1+1)/2)}'`
	center_y=${center_x}
	center_z=`echo ${slice} ${geo_zshift} | gawk '//{print int(($1+1)/2+$2)}'`

######### Tomographic Reconstruction using ICON method ##############################

	mkdir ${name}_tmp

	ICON-GPU -i ${name}_pre_ali_bin${bin}.mrc -t $ali_tlt -o ${name}_tmp -s 0,$slice -iter 20,100,20 -d $datatype -thr 0 -g -1 | tee -a ${name}_icon.log

	ICONMask3 -i ${name}_tmp/reconstruction -t $ali_tlt -o ${name}.tmp.rec -s 0,$slice -th `echo $slice+1 | bc` -cf ${name}_tmp/crossValidation/crossV.frc -ff ${name}_tmp/crossValidation/fullRec.frc | tee -a ${name}_icon.log

	rm -r -f ${name}_tmp

	newstack -meansd 0,1  ${name}.tmp.rec  ${name}_icon_norm.rec | tee -a ${name}_icon.log	

	trimvol -rx ${name}_icon_norm.rec ${name}_icon_rot.rec | tee -a ${name}_icon.log
        
	rm ${name}.tmp.rec

	rotatevol -i ${name}_icon_rot.rec -ou ${name}.tmp.rec -c ${center_x},${center_y},${center_z} -a 0,${rot_y},0
	rotatevol -i ${name}.tmp.rec -ou ${name}_icon_rot_geocorr.rec -a 0,0,${rot_x}

	rm ${name}.tmp.rec

	clip resize -cx ${center_x} -cy ${center_y} -cz ${center_y} -ox ${size_x} -oy ${size_y} -oz ${geo_thickness} ${name}_icon_rot_geocorr.rec ${name}_icon_rot_geocorr_clip.rec





