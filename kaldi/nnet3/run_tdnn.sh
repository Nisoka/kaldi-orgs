#!/bin/bash

# This script is based on swbd/s5c/local/nnet3/run_tdnn.sh

# this is the standard "tdnn" system, built in nnet3; it's what we use to
# call multi-splice.

# At this script level we don't support not running on GPU, as it would be painfully slow.
# If you want to run without GPU you'd have to call train_tdnn.sh with --gpu false,
# --num-threads 16 and --minibatch-size 128.
set -e

stage=0
train_stage=-10
affix=
common_egs_dir=

# training options
initial_effective_lrate=0.0015
final_effective_lrate=0.00015
num_epochs=4
num_jobs_initial=2
num_jobs_final=12
remove_egs=true

# feature options
use_ivectors=true

# End configuration section.

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

# ${affix:+_$affix} 表示有affix时候加上affix, 为空就什么都没有.

# tdnn训练空间目录
dir=exp/nnet3/tdnn_sp${affix:+_$affix}
# exp/tri5a 最新训练结果模型
gmm_dir=exp/tri5a
# 构造 ivector特征数据目录 以及 MFCC特征目录
train_set=train_sp
# 最新训练结果模型 在训练数据增加扰动后 decode对齐结果目录
# (在run_ivector_common.sh 中进行的 对训练数据增加扰动,以及decode align)
ali_dir=${gmm_dir}_sp_ali
# exp/tri5a/graph 最新模型构建的图 HCLG.fst
graph_dir=$gmm_dir/graph






# ====================== 1 在训练样本MFCC上增加扰动 得到 sp 数据                     ==> data/train_sp_hires 
# ====================== 2 在train_sp_hires 进行    decode align                     ==> exp/tri5a_sp_ali
# ====================== 3 训练 UBM extractor, 对sp 数据提取100-dim ivector特征      ==> exp/nnet3/ivectors_train_sp

# data/train_sp   扰动数据(data/train --> data/train_sp)
# mfcc_perturbed  扰动数据特征(data/train_sp --> mfcc_perturbed   (exp/make_mfcc/train_sp is the log))
#                 run.sh 相同目录下 
# exp/tri5a_sp_ali 扰动数据对齐结果 (mfcc_perturbed --> exp/tri5a_sp_ali)
# data/train_sp_hires   高分辨率数据 (data/train_sp --> data/train_sp_hires)
# mfcc_perturbed_hires  高分辨率特征 (data/train_sp_hires --> mfcc_perturbed_hires)
# data/train_sp_hires_nopitch   get the nopitch hires mfcc-perturbed-feature to extra the ivectors
# exp/nnet3/diag_ubm
# exp/nnet3/pca_transform
# exp/nnet3/extractor
# exp/nnet3/ivectors_train_sp/train_sp_hires_nopitch_max2
#                                修改spkerinfo, 每个spk 最多两个utterance. ???
#                                (data/train_sp_hires_nopitch --> exp/nnet3/ivectors_train_sp/train_sp_hires_nopitch_max2)
# exp/nnet3/ivectors_train  提取ivector特征 (exp/nnet3/ivectors_train_sp/train_sp_hires_nopitch_max2 --> exp/nnet3/ivectors_train)
# exp/nnet3/ivectors_test dev.

local/nnet3/run_ivector_common.sh --stage $stage || exit 1;





if [ $stage -le 7 ]; then
  echo "$0: creating neural net configs";

  num_targets=$(tree-info $ali_dir/tree |grep num-pdfs|awk '{print $2}')

  mkdir -p $dir/configs

  # 注意 有一个name=input的输入层十分重要, 因为这个层会直接在 fixed-affine-layer修正仿射层之前.
  # fixed-affine-layer 修正仿射层 name=lda, 是做lda变换的lda层,不需要更新参数?
  
  # 将如下内容 写入 network.xconfig
  # input 一个ivector的 100维输入
  # input 一个input     43维输入
  
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=43 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-2,-1,0,1,2,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-layer name=tdnn1 dim=850
  relu-batchnorm-layer name=tdnn2 dim=850 input=Append(-1,0,2)
  relu-batchnorm-layer name=tdnn3 dim=850 input=Append(-3,0,3)
  relu-batchnorm-layer name=tdnn4 dim=850 input=Append(-7,0,2)
  relu-batchnorm-layer name=tdnn5 dim=850 input=Append(-3,0,3)
  relu-batchnorm-layer name=tdnn6 dim=850
  output-layer name=output input=tdnn6 dim=$num_targets max-change=1.5
EOF

  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

# tdnn_sp/                                                                                                                  
# └── configs                                                                                                               
#     ├── final.config                                                                                                      
#     ├── init.config                                                                                                       
#     ├── init.raw                                                                                                          
#     ├── network.xconfig                                                                                                   
#     ├── ref.config
#     ├── ref.raw
#     ├── vars
#     ├── xconfig
#     ├── xconfig.expanded.1
#     └── xconfig.expanded.2




if [ $stage -le 8 ]; then
  # if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
  #   utils/create_split_dir.pl \
  #    /export/b0{5,6,7,8}/$USER/kaldi-data/egs/aishell-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  # fi

  steps/nnet3/train_dnn.py --stage=$train_stage \
    --cmd="$decode_cmd" \
    --feat.online-ivector-dir exp/nnet3/ivectors_${train_set} \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.num-jobs-initial $num_jobs_initial \
    --trainer.optimization.num-jobs-final $num_jobs_final \
    --trainer.optimization.initial-effective-lrate $initial_effective_lrate \
    --trainer.optimization.final-effective-lrate $final_effective_lrate \
    --egs.dir "$common_egs_dir" \
    --cleanup.remove-egs $remove_egs \
    --cleanup.preserve-model-interval 500 \
    --use-gpu true \
    --feat-dir=data/${train_set}_hires \
    --ali-dir $ali_dir \
    --lang data/lang \
    --reporting.email="$reporting_email" \
    --dir=$dir  || exit 1;
fi
















# debug: liujunnan
echo "over the training train_tdnn.py"
exit 0;





if [ $stage -le 9 ]; then
  # this version of the decoding treats each utterance separately
  # without carrying forward speaker information.
  for decode_set in dev test; do
    num_jobs=`cat data/${decode_set}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`
    decode_dir=${dir}/decode_$decode_set
    steps/nnet3/decode.sh --nj $num_jobs --cmd "$decode_cmd" \
       --online-ivector-dir exp/nnet3/ivectors_${decode_set} \
       $graph_dir data/${decode_set}_hires $decode_dir || exit 1;
  done
fi

wait;
exit 0;
