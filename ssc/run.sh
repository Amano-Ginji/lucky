#!/bin/bash

date="20170130"

mode=0

function random_range() {
  local beg=$1
  local end=$2
  echo $((RANDOM % ($end - $beg) + $beg))
}

function get_data() {
  for((i=0; i<=3650; ++i)); do
    dt=$(date -d "$date -$i days" +"%Y-%m-%d")
    url="http://baidu.lecai.com/lottery/draw/list/200?d=$dt"
    echo "crawl $url ..."
    python get_data.py $url > data/result.$dt
    rnd=$(random_range 1 100)
    tm=$(echo "$rnd*0.01"|bc)
    sleep $tm
  done
}

function make_dataset() {
  input=$1;
  output1=$2;
  output2=$3;
  ratio=$4;
  mode=$5;

  cat /dev/null > $output1
  cat /dev/null > $output2

  cat $input | python make_dataset.py $output1 $output2 $ratio $mode
}

function train() {
  return ""
}

function evaluate() {
  return ""
}

echo ">>> get data ..."
###get_data

echo ">>> make dataset ..."
###make_dataset "data/result.200* data/result.201[0-5]*" data/samples.train data/samples.train-dev 0.9 $mode
###make_dataset "data/result.2016*" data/samples.dev data/samples.test 0.6 $mode
make_dataset "data/result.200* data/result.201[0-5]* data/result.2016*" data/samples data/samples.null 1.0 $mode

cat /dev/null > data/samples.new.train
cat /dev/null > data/samples.new.train-dev
cat /dev/null > data/samples.new.dev
cat /dev/null > data/samples.new.test
cat data/samples | awk -F"\t" -v r1=0.9 -v r2=0.6 '
NR>45000 && NR<369000 {
  if (rand() < r1) print $0 >> "data/samples.new.train";
  else print $0 >> "data/samples.new.train-dev";
}
NR>=369000 {
  if (rand() < r2) print $0 >> "data/samples.new.dev";
  else print $0 >> "data/samples.new.test";
}'

echo ">>> train model ..."
./bin/train -s 2 -c 0.000000001 -e 0.000001 data/samples.new.train result/model

echo ">>> test model ..."
./bin/predict data/samples.new.train result/model result/output.train
./bin/predict data/samples.new.train-dev result/model result/output.train-dev
./bin/predict data/samples.new.dev result/model result/output.dev
./bin/predict data/samples.new.test result/model result/output.test

### case ###
#make_dataset "data/result.2016-01-26" data/samples.20160126 data/samples.null.20160126 1.0
