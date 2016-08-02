#!/bin/bash
# takes arguments: filename of pred, filename of gold

if [[ $# -eq 0 ]]
then
  echo "Usage: ./eval_rouge.sh /path/to/pred.txt /path/to/gold.txt"
  exit 1
fi

pred=/scratch/tmp_PRED
gold=/scratch/tmp_GOLD

rm -rf $pred
rm -rf $gold
mkdir $pred
mkdir $gold

echo "Preparing files..."
source activate my_root
python eval_rouge.py $1 $2
source deactivate

perl prepare4rouge-simple.pl

export ROUGE=/scratch/RELEASE-1.5.5

cd "/scratch/tmp_OUTPUT"

echo "FULL LENGTH"
perl $ROUGE/ROUGE-1.5.5.pl -m -n 2 -w 1.2 -e $ROUGE -a settings.xml


#echo "LIMITED LENGTH"
#perl $ROUGE/ROUGE-1.5.5.pl -m -b 75 -n 2 -w 1.2 -e $ROUGE -a settings.xml
