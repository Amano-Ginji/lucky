#!/bin/bash

samples="data/samples"

samples_num=$(wc -l $samples | awk -F" " '{print $1;}')
steps=120
iter_num=100
start_pos=$((${samples_num}-$steps*${iter_num}))
end_pos=${samples_num}

mkdir -p data
mkdir -p result

function run_model() {
  echo ">>>>>> run model ..."

  for ((i=${iter_num}; i>=1; --i)); do
    echo ">>> make dataset ..."
    start_pos=$((${samples_num}-$steps*$i))
    end_pos=$((${start_pos}+$steps))
    echo "from: "${start_pos}", to: "${end_pos}
  
    train_file=data/samples.train.${start_pos}.${end_pos}
    train_dev_file=data/samples.train-dev.${start_pos}.${end_pos}
    dev_file=data/samples.dev.${start_pos}.${end_pos}
    test_file=data/samples.test.${start_pos}.${end_pos}
  
    cat /dev/null > ${train_file}
    cat /dev/null > ${train_dev_file}
    cat /dev/null > ${dev_file}
    cat /dev/null > ${test_file}
  
    cat $samples | awk -F"\t" -v r1=0.9 -v r2=0.0 -v start_pos=${start_pos} -v end_pos=${end_pos} '
      NR<start_pos {
        if (rand() < r1) print $0 >> "'${train_file}'";
        else print $0 >> "'${train_dev_file}'";
      }
      NR>=start_pos && NR<end_pos {
        if (rand() < r2) print $0 >> "'${dev_file}'";
        else print $0 >> "'${test_file}'";
      }'
  
    echo ">>> train model ..."
    ./bin/train -s 2 -c 0.000000001 -e 0.000001 ${train_file} result/model.${start_pos}.${end_pos}
  
    echo ">>> test model ..."
    ./bin/predict ${test_file} result/model.${start_pos}.${end_pos} result/output.${start_pos}.${end_pos} > result/eval.${start_pos}.${end_pos}
  
    echo ">>> clean data ..."
    rm -rf ${train_file}
    rm -rf ${train_dev_file}
    rm -rf result/model.${start_pos}.${end_pos}
  done
  
  cat result/eval.3* | awk -F"[(/)]" '{cnt+=$(NF-1);cnt_t+=$(NF-2);} END{print cnt,cnt_t,cnt_t/cnt;}'
}

function run_random() {
  echo ">>>>>> run random ..."

  for ((i=${iter_num}; i>=1; --i)); do
    echo ">>> make dataset ..."
    start_pos=$((${samples_num}-$steps*$i))
    end_pos=$((${start_pos}+$steps))
    echo "from: "${start_pos}", to: "${end_pos}
  
    train_file=data/samples.train.${start_pos}.${end_pos}
    train_dev_file=data/samples.train-dev.${start_pos}.${end_pos}
    dev_file=data/samples.dev.${start_pos}.${end_pos}
    test_file=data/samples.test.${start_pos}.${end_pos}
  
    cat /dev/null > ${train_file}
    cat /dev/null > ${train_dev_file}
    cat /dev/null > ${dev_file}
    cat /dev/null > ${test_file}
  
    cat $samples | awk -F"\t" -v r1=0.9 -v r2=0.0 -v start_pos=${start_pos} -v end_pos=${end_pos} '
      NR<start_pos {
        if (rand() < r1) print $0 >> "'${train_file}'";
        else print $0 >> "'${train_dev_file}'";
      }
      NR>=start_pos && NR<end_pos {
        if (rand() < r2) print $0 >> "'${dev_file}'";
        else print $0 >> "'${test_file}'";
      }'
  
    echo ">>> train model ..."
  
    echo ">>> test model ..."
    awk -F" " '{label=$1;pred=int(10*rand());++cnt;if(label==pred)++cnt_t;} END{print cnt_t,cnt,cnt_t/cnt;}' ${test_file} > result/eval.rand.${start_pos}.${end_pos}

    echo ">>> clean data ..."
    rm -rf ${train_file}
    rm -rf ${train_dev_file}
    rm -rf result/model.${start_pos}.${end_pos}
  done
  
  cat result/eval.rand.* | awk -F"[ (/)]" '{cnt+=$(NF-1);cnt_t+=$(NF-2);} END{print cnt,cnt_t,cnt_t/cnt;}'
}

function run_hot() {
  echo ">>>>>> run random ..."
  period=$1

  for ((i=${iter_num}; i>=1; --i)); do
  #for ((i=1; i>=1; --i)); do
    echo ">>> make dataset ..."
    start_pos=$((${samples_num}-$steps*$i))
    end_pos=$((${start_pos}+$steps))
    echo "from: "${start_pos}", to: "${end_pos}
  
    train_file=data/samples.train.${start_pos}.${end_pos}
    train_dev_file=data/samples.train-dev.${start_pos}.${end_pos}
    dev_file=data/samples.dev.${start_pos}.${end_pos}
    test_file=data/samples.test.${start_pos}.${end_pos}
  
    cat /dev/null > ${train_file}
    cat /dev/null > ${train_dev_file}
    cat /dev/null > ${dev_file}
    cat /dev/null > ${test_file}
  
    cat $samples | awk -F"\t" -v r1=0.9 -v r2=0.0 -v start_pos=${start_pos} -v end_pos=${end_pos} '
      NR<start_pos {
        if (rand() < r1) print $0 >> "'${train_file}'";
        else print $0 >> "'${train_dev_file}'";
      }
      NR>=start_pos && NR<end_pos {
        if (rand() < r2) print $0 >> "'${dev_file}'";
        else print $0 >> "'${test_file}'";
      }'
  
    echo ">>> train model ..."
    lines=$(cat ${train_file} ${train_dev_file} | wc -l)
    start_lines=$(($lines-120*$period))
    echo "lines: $lines"
    echo "start lines: ${start_lines}"
    cat ${train_file} ${train_dev_file} | awk -F"\t" -v start_lines=${start_lines} '
    NR>start_lines {
      label = $1;
      ++cnt[$1];
    }
    END {
      max_cnt = -1;
      pred = "";
      for(r in cnt) {
        if(cnt[r] > max_cnt) {
          max_cnt = cnt[r];
          pred = r;
        }
      }
      print pred,max_cnt;
    }' > result/model.p${period}.${start_pos}.${end_pos}

    echo ">>> test model ..."
    awk -F" " '
    FILENAME~/model/{
      pred = $1;
    } 
    FILENAME~/samples/{
      ++cnt;
      label = $1;     
      if(label == pred) {
        ++cnt_t;
      }
    }
    END {
      print cnt_t,cnt,cnt_t/cnt;
    }' result/model.p${period}.${start_pos}.${end_pos} ${test_file} > result/eval.hot.p${period}.${start_pos}.${end_pos}

    echo ">>> clean data ..."
    rm -rf ${train_file}
    rm -rf ${train_dev_file}
    #rm -rf result/model.p{period}${start_pos}.${end_pos}
  done
  
  cat result/eval.hot.* | awk -F"[ (/)]" '{cnt+=$(NF-1);cnt_t+=$(NF-2);} END{print cnt_t,cnt,cnt_t/cnt;}'
 
}

#run_model
#run_random
#run_hot 7
run_hot 30
#run_hot 60
#run_hot 90
