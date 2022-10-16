#!/bin/bash

# ----- usage ------ #
function usage()
{
	echo "fastAF2 v0.01 [May-22-2022] "
	echo "    An extremely fast version of AF2 (just using 1 model and NO refinement)  "
	echo ""
	echo "USAGE:  ./fastAF2.sh <-i query_seq> [-o out_root] [-m method] [-g GPU] [-H home] "
	echo "Options:"
	echo ""
	echo "***** required arguments *****"
	echo "-i query_seq    : Query protein sequence in FASTA format. "
	echo ""
	echo "-o out_root     : Output directory. [default = './\${input_name}_fAF2'] "
	echo ""
	echo "***** optional arguments *****"
	echo "-m method       : method to generate MSA (qjackhmmer or mmseq2). [default = 'qjackhmmer'] "
	echo ""
	echo "-g GPU          : gpu_device. [default = '0' ] "
	echo ""
	echo "***** home relevant **********"
	echo "-H home         : home directory of fastAF2."
	echo "                 [default = `dirname $0`]"
	echo ""
	exit 1
}


#-------------------------------------------------#
##### ===== get pwd and check suage====== #########
#-------------------------------------------------#

#------ current directory ------#
curdir="$(pwd)"

#-------- check usage -------#
if [ $# -lt 1 ];
then
	usage
fi


#---------------------------------------------------------#
##### ===== All arguments are defined here ====== #########
#---------------------------------------------------------#


# ----- get arguments ----- #
#-> required arguments
query_seq=""
out_root=""     #-> output to current directory
#-> optional arguments
method="qjackhmmer"
gpu_device="0"
#--| home relevant
home=`dirname $0`  #-> home directory

#-> parse arguments
while getopts ":i:o:m:g:H:" opt;
do
	case $opt in
	#-> required arguments
	i)
		query_seq=$OPTARG
		;;
	o)
		out_file=$OPTARG
		;;
	#-> optional arguments
	m)
		method=$OPTARG
		;;
	g)
		gpu_device=$OPTARG
		;;
	#-> home relevant
	H)
		home=$OPTARG
		;;
	#-> default
	\?)
		echo "Invalid option: -$OPTARG" >&2
		exit 1
		;;
	:)
		echo "Option -$OPTARG requires an argument." >&2
		exit 1
		;;
	esac
done


#---------------------------------------------------------#
##### ===== Part 0: initial argument check ====== #########
#---------------------------------------------------------#

#----------- check home -----------------#
if [ ! -d "$home" ]
then
	echo "home directory $home not exist " >&2
	exit 1
fi
home=`readlink -f $home`

# ------ check query_seq ------#
if [ -z "$query_seq" ]
then
	echo "input query_seq is null !!" >&2
	exit 1
fi
if [ ! -s "$query_seq" ]
then
	echo "query_seq $query_seq not found !!" >&2
	exit 1
fi
query_seq=`readlink -f $query_seq`
#-> get query_name
fulnam=`basename $query_seq`
relnam=${fulnam%.*}

# ------ check output directory -------- #
if [ "$out_root" == "" ]
then
	out_root=$curdir/${relnam}_fAF2
fi
mkdir -p $out_root
out_root=`readlink -f $out_root`



#================== Part 0.2 initialization =======================#

#-------- mkdir tmp ------#
temp_root="/tmp"
DATE=`date '+%s%N' | cut -b10-19`
folder="TMP_fastAF2_${relnam}_${RANDOM}_${DATE}"
tmp="$temp_root/$folder"
echo "$tmp"
mkdir -p $tmp

#------------- fastMSA -----------#
begin=$SECONDS
begi=$SECONDS
echo "#---------- step 1: fastMSA ---------#"
/share/wangsheng/GitBucket/fastMSA/fastMSA.sh -i $query_seq -m $method -o $tmp/$relnam.a3m
duration=$(( SECONDS - begi ))
echo "timer of fastMSA: $duration"

#------------- fastAF2 -----------#
#begi=$SECONDS
#echo "#---------- step 2.1: fastAF2 (features.pkl) ---------#"
#export BREAK_AFTER_TEMPLATES="True"
#/share/wangsheng/GitBucket/alphafold2_sheng/alphafold2/run_local.sh \
#	-i $tmp/$relnam.a3m -o $out_root -g $gpu_device -T -1 -m model_1 -E /share/wangsheng/miniconda/envs/af2 \
#	1> $tmp/$relnam.ws1 2> $tmp/$relnam.ws2
#duration=$(( SECONDS - begi ))
#echo "timer of fastAF2 (features.pkl): $duration"
begi=$SECONDS
echo "#---------- step 2.2: fastAF2 (pytorch E2E) ---------#"
/user/wangsheng/miniconda/envs/af2torch/bin/python /share/wangsheng/GitBucket/af2torch/run_alphafold.py \
	--device cuda:$gpu_device --model_name model_1 --model_dir /share/yanghuan/params_torch/ --data_path $tmp/$relnam.a3m --a3m_input --output_dir $out_root \
	1> $tmp/$relnam.ww1 2> $tmp/$relnam.ww2
#	--device cuda:$gpu_device --model_name model_1 --model_dir /share/yanghuan/params_torch/ --data_path ${out_root}/features.pkl --output_dir $out_root \
duration=$(( SECONDS - begi ))
echo "timer of fastAF2 (pytorch E2E): $duration"

#---- total timer -----#
duration=$(( SECONDS - begin ))
echo "timer: $duration"

#-------- remove tmp folder -------#
rm -rf $tmp

#======= exit =======#
exit 0
