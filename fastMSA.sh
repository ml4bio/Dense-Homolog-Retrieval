#!/bin/bash

# ----- usage ------ #
function usage()
{
	echo "fastMSA v0.02 [May-21-2022] "
	echo "    An extremely fast search engine for Multiple Sequence Alignment (MSA)  "
	echo ""
	echo "USAGE:  ./fastMSA.sh <-i query_seq> [-o out_file] [-S server] [-m method] [-H home] "
	echo "             	[-k topk] [-d database] [-n iter] [-e evalue] [-M min_cut] [-N max_num] "
	echo "Options:"
	echo ""
	echo "***** required arguments *****"
	echo "-i query_seq    : Query protein sequence in FASTA format. "
	echo ""
	echo "-o out_file     : Output MSA in A3M format. "
	echo "                  [default = './\${input_name}.a3m'] "
	echo ""
	echo "-S server       : local http server IP for curl command. "
	echo "                  [default = 'http://172.16.20.190/fastmsa' ]"
	echo ""
	echo "-m method       : method to generate MSA (qjackhmmer or mmseq2). [default = 'qjackhmmer']"
	echo ""
	echo "***** optional arguments *****"
	echo "#--| dense retrieval"
	echo "-k topk         : topk retrieved sequences. [default=40K] "
	echo ""
	echo "-d database     : the database to be searched. [default=uniref90] "
	echo ""
	echo "#--| MSA generation"
	echo "-n iter         : Maximal iteration to run the seleced package. [default = 2] "
	echo ""
	echo "-e evalue       : E-value cutoff for the selected package. [default = 0.001] "
	echo ""
	echo "#--| MSA filter"
	echo "-M min_cut      : Minimal coverage of sequences in the generated MSA. [default = -1] "
	echo "                  -1 indicates that we DON'T perform any filtering. Please set from 50 to 70. "
	echo ""
	echo "-N max_num      : Maximal number of sequences in the generated MSA. [default = -1] "
	echo "                  -1 indicates that we DON'T perform any filtering. For example, set 20000 here. "
	echo ""
	echo "***** home relevant **********"
	echo "-H home         : home directory of fastMSA."
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
out_file=""     #-> output to current directory
server="http://172.16.20.190/fastmsa"
method="qjackhmmer"
#--| dense retrieval
topk=400000
database=uniref90
#--| MSA generation
iter=2          #-> default is 2 iterations, for threading purpose
e_value=0.001   #-> default is 0.001, for threading purpose
#--| MSA filter
min_cut=-1      #-> default is -1. If set, then run Cov_Filter
max_num=-1      #-> default is -1. If set, then run Meff_Filter
#--| home relevant
home=`dirname $0`  #-> home directory

#-> parse arguments
while getopts ":i:o:S:m:k:d:n:e:M:N:H:" opt;
do
	case $opt in
	#-> required arguments
	i)
		query_seq=$OPTARG
		;;
	o)
		out_file=$OPTARG
		;;
	S)
		server=$OPTARG
		;;
	m)
		method=$OPTARG
		;;
	#-> optional arguments
	#--| dense retrieval
	k)
		topk=$OPTARG
		;;
	d)
		database=$OPTARG
		;;
	#--| MSA generation
	n)
		iter=$OPTARG
		;;
	e)
		e_value=$OPTARG
		;;
	#--| MSA filter
	M)
		min_cut=$OPTARG
		;;
	N)
		max_num=$OPTARG
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

#----------- check out_file -----------#
if [ "$out_file" == "" ]
then
	out_file=${relnam}.a3m
fi

#================== Part 0.2 initialization =======================#

#-------- mkdir tmp ------#
temp_root="/tmp"
output_root="/share/linmingzhi/fm_output"
DATE=`date '+%s%N' | cut -b10-19`
folder="TMP_fastMSA_${relnam}_${RANDOM}_${DATE}"
txp="$output_root/$folder"
tmp="$temp_root/$folder"
echo "$txp"
echo "$tmp"
mkdir -p $txp
mkdir -p $tmp
chmod 777 $txp
$home/bin/Verify_FASTA $query_seq $txp/$relnam.fasta
#-> sleep 
echo "sleep 5s"
sleep 5s

#-------- dense retrieval ------#
echo "curl --location --request POST '$server' \
	--header 'Content-Type: application/x-www-form-urlencoded' \
	--data-urlencode 'input=$txp/$relnam.fasta' \
	--data-urlencode 'output=$txp' \
	--data-urlencode 'tarnum=$topk'"
curl --location --request POST "$server" \
        --header "Content-Type: application/x-www-form-urlencoded" \
        --data-urlencode "input=$txp/$relnam.fasta" \
        --data-urlencode "output=$txp" \
	--data-urlencode "tarnum=$topk"

#-> wait
timer=0
run_time=100
while true
do
	#--| check the existance of the generated file
	if [ -s "$txp/db/$relnam.fasta" ]
	then
		num=`grep "^>" $txp/db/$relnam.fasta | wc | awk '{print $1}'`
		if [ $num -eq $topk ]
		then
			break
		else
			echo "db file $txp/db/$relnam.fasta contains $num < $topk sequences"
		fi
	fi
	#--| sleep
	echo "fastMSA job $query_seq still not finished : $timer / $run_time X 5s"
	sleep 5s
	((timer++))
	if [ $timer -gt $run_time ]
	then
		echo "running time limit $run_time min is up, still not finished" >&2
		exit
	fi
done

#===== different appraoch to obtain MSA ======#
if [ "$method" == "qjackhmmer" ]
then
	#-------- qjackhmmer ------#
	cpu_num=4
	echo "qjackhmm start with $database with evalue $e_value and iteration $iter with cpu $cpu_num"
	#-> run jackhmmer
	$home/bin/qjackhmmer -B $tmp/$relnam.a3m -N $iter --cpu $cpu_num -E $e_value --domE $e_value --incE $e_value --incdomE $e_value \
		--noali --notextw --F1 0.0005 --F2 5e-05 --F3 5e-07 --tblout $tmp/$relnam.tblout --domtblout $tmp/$relnam.domtblout -o $tmp/$relnam.output \
		$txp/seq/$relnam.fasta $txp/db/$relnam.fasta
	OUT=$?
	if [ $OUT -ne 0 ]
	then
		echo "failed in $home/bin/qjackhmmer -B $tmp/$relnam.a3m $txp/seq/$relnam.fasta $txp/db/$relnam.fasta"
		exit 1
	fi
	echo "jackhammer done"
elif [ "$method" == "mmseq2" ]
then
	#-------- mmseq2 ---------#
	echo "run mmseqs easy-search"
	$home/bin/mmseqs easy-search $txp/seq/$relnam.fasta $txp/db/$relnam.fasta $tmp/$relnam.mmseq2 $tmp/tmp --max-seqs 5000 -s 7.5 1>$tmp/ws1 2>$tmp/ws2
	#-------- head_retrieve ---#
	awk '{print $2}' $tmp/$relnam.mmseq2 > $tmp/$relnam.mmseq2_header
	$home/bin/Retrieve_FASTA $tmp/$relnam.mmseq2_header $txp/db/$relnam.fasta > $tmp/$relnam.mmseq2_fa
	#-------- qjackhmmer ------#
	cpu_num=4
	echo "run qjackhmmer on head_retrieve a3m"
	$home/bin/qjackhmmer -B $tmp/$relnam.a3m -N $iter --cpu $cpu_num -E $e_value --domE $e_value --incE $e_value --incdomE $e_value \
		--noali --notextw --F1 0.0005 --F2 5e-05 --F3 5e-07 --tblout $tmp/$relnam.tblout --domtblout $tmp/$relnam.domtblout -o $tmp/$relnam.output \
		$txp/seq/$relnam.fasta $tmp/$relnam.mmseq2_fa
	OUT=$?
	if [ $OUT -ne 0 ]
	then
		echo "failed in $home/bin/qjackhmmer -B $tmp/$relnam.a3m $txp/seq/$relnam.fasta $tmp/$relnam.mmseq2_fa"
		exit 1
	fi
	echo "mmseq2 done"
else
	echo "method $method not support yet"
	exit 1
fi
#-> post process
$home/bin/A3M_ReFormat $tmp/$relnam.a3m $tmp/$relnam.a3m_
mv $tmp/$relnam.a3m_ $tmp/$relnam.a3m

#-------- MSA filter ------#
#-> coverage filter
if [ $min_cut -ne -1 ]
then
	#-> calculate the number of sequences in A3M
	numLines=`grep "^>" $tmp/$relnam.a3m | wc | awk '{print $1}'`
	echo "number of lines in the original A3M before CovFilt is $numLines"
	#-> coverage filter
	cp $tmp/$relnam.a3m $tmp/$relnam.a3m_prev
	$home/bin/MSA_CovFilter $tmp/$relnam.a3m_prev $tmp/$relnam.a3m $min_cut
	#-> calculate the number of sequences in A3M
	numLines=`grep "^>" $tmp/$relnam.a3m | wc | awk '{print $1}'`
	echo "number of lines in the cov_filt A3M before CovFilt is $numLines"
fi
#-> maximal number filter
if [ $max_num -ne -1 ]
then
	#-> calculate the number of sequences in A3M
	numLines=`grep "^>" $tmp/$relnam.a3m | wc | awk '{print $1}'`
	echo "number of lines in the original A3M is $numLines"
	#-> judge numLines
	if [ $numLines -gt $max_num ]
	then
		#--| copy to prefilter
		cp $tmp/$relnam.a3m $tmp/$relnam.a3m_prefilt
		#--| filter
		$home/bin/meff_filter -i $tmp/$relnam.a3m_prefilt -o $tmp/$relnam.a3m -n $max_num -C $cpu_num
		#--| filter result
		numLines=`grep "^>" $tmp/$a3m_file | wc | awk '{print $1}'`
		echo "number of lines in the filtered A3M is $numLines"
	fi
fi

#------ final copy -------#
cp $tmp/$relnam.a3m $out_file

#-------- remove tmp folder -------#
rm -rf $txp
rm -rf $tmp

#======= exit =======#
exit 0
