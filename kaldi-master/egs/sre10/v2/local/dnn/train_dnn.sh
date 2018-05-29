#!/bin/bash

# This script is based on egs/fisher_english/s5/run.sh. It trains a
# multisplice time-delay neural network used in the DNN-based speaker
# recognition recipes.

# It's best to run the commands in this one by one.

. ./cmd.sh
. ./path.sh
mfccdir=`pwd`/mfcc
set -e

# ---------------------------------------------------------
# 准备数据

# 1 统一数据 连接到 links=data/local/data/links之下
# 2 生成数据信息文件
#   utt-id:  spker-[AB]-start-end
#   spker:   spker-[AB]
#   gender:  [fm]
#   data/train_in_all_asr/wav.scp       uttid  sph2pip- sph-path
#   data/train_in_all_asr/utt2spk       uttid  spker
#   data/train_in_all_asr/text          uttid  text
#   data/train_in_all_asr/spk2gender    spker  gender
#   data/train_in_all_asr/segments ????
# ---------------------------------------------------------
# the next command produces the data in local/train_all_asr
local/dnn/fisher_data_prep.sh /export/corpora3/LDC/LDC2004T19 /export/corpora3/LDC/LDC2005T19 \
                              /export/corpora3/LDC/LDC2004S13 /export/corpora3/LDC/LDC2005S13



# You could also try specifying the --calldata argument to this command as below.
# If specified, the script will use actual speaker personal identification
# numbers released with the dataset, i.e. real speaker IDs. Note: --calldata has
# to be the first argument of this script.
# local/fisher_data_prep.sh --calldata /export/corpora3/LDC/LDC2004T19 /export/corpora3/LDC/LDC2005T19 \
#    /export/corpora3/LDC/LDC2004S13 /export/corpora3/LDC/LDC2005S13

# at BUT:
# local/fisher_data_prep.sh /mnt/matylda6/jhu09/qpovey/FISHER/LDC2005T19 /mnt/matylda2/data/FISHER/




# ---------------------------------------------------------
# 生成 dict数据
# [in data/local/dict/ ]
# lexicon.txt
# extra_questions.txt
# nonsilence_phones.txt
# optional_silence.txt
# silence_phones.txt
# ---------------------------------------------------------
local/dnn/fisher_prepare_dict.sh


# ---------------------------------------------------------
# 预处理 lang数据 准备 一些 sets.txt  oov.txt  等等
# ---------------------------------------------------------
utils/prepare_lang.sh data/local/dict "<unk>" data/local/lang data/lang

# ---------------------------------------------------------
# 生成训练语言模型
# data/local/lm/3gram-mincount/lm_unpruned.gz
# ---------------------------------------------------------
local/dnn/fisher_train_lms.sh


# ---------------------------------------------------------
# 创建 G.fst  L.fst 并测试是否是确定化的
# ---------------------------------------------------------
local/dnn/fisher_create_test_lang.sh

# ---------------------------------------------------------
# 创建 dev目录, 当训练完成LM模型之后, 我们使用了前10k utt 作为dev, 所以
# 所以 前4kutt 不能用户LM训练, 然而他们还是在lexicon, 以及spkers会重复,
# 所以仍然不能作为一个test set
# ---------------------------------------------------------
# Use the first 4k sentences as dev set.  Note: when we trained the LM, we used
# the 1st 10k sentences as dev set, so the 1st 4k won't have been used in the
# LM training data.   However, they will be in the lexicon, plus speakers
# may overlap, so it's still not quite equivalent to a test set.
utils/fix_data_dir.sh data/train_all_asr



# ---------------------------------------------------------
# 计算 mfcc特征  数据为 data/train_all_asr中的数据信息文件
#  ==> mfcc
# ---------------------------------------------------------
steps/make_mfcc.sh --nj 40 --cmd "$train_cmd" --mfcc-config conf/mfcc_asr.conf \
                   data/train_all_asr \
                   exp/make_mfcc/train_all_asr \
                   $mfccdir || exit 1;


utils/fix_data_dir.sh data/train_all_asr
utils/validate_data_dir.sh data/train_all_asr

# ---------------------------------------------------------
# 划分数据 feats.scp wav.scp utt2spk 等等  生成
# 5k data/dev_asr
# 5k data/test_asr
# leave data/train_asr
# ---------------------------------------------------------
# dev  test数据集每个都3.3小时.
# 这些数据没有仔细梳理, 可能会有speaker信息与train中数据重叠
# 在进行LM训练时 我们剔除了前10k的utt, 所以LM训练并没有训练到dev test数据的
# The dev and test sets are each about 3.3 hours long.  These are not carefully
# done; there may be some speaker overlap with each other and with the training
# set.  Note: in our LM-training setup we excluded the first 10k utterances (they
# were used for tuning but not for training), so the LM was not (directly) trained
# on either the dev or test sets.
utils/subset_data_dir.sh --first data/train_all_asr 10000 data/dev_and_test_asr
utils/subset_data_dir.sh --first data/dev_and_test_asr 5000 data/dev_asr
utils/subset_data_dir.sh --last data/dev_and_test_asr 5000 data/test_asr
rm -r data/dev_and_test_asr

steps/compute_cmvn_stats.sh data/dev_asr exp/make_mfcc/dev_asr $mfccdir
steps/compute_cmvn_stats.sh data/test_asr exp/make_mfcc/test_asr $mfccdir

n=$[`cat data/train_all_asr/segments | wc -l` - 10000]
utils/subset_data_dir.sh --last data/train_all_asr $n data/train_asr
steps/compute_cmvn_stats.sh data/train_asr exp/make_mfcc/train_asr $mfccdir


# Now-- there are 1.6 million utterances, and we want to start the monophone training
# on relatively short utterances (easier to align), but not only the very shortest
# ones (mostly uh-huh).  So take the 100k shortest ones, and then take 10k random
# utterances from those.

utils/subset_data_dir.sh --shortest data/train_asr 100000 data/train_asr_100kshort
utils/subset_data_dir.sh  data/train_asr_100kshort 10000 data/train_asr_10k

local/dnn/remove_dup_utts.sh 100 data/train_asr_10k data/train_asr_10k_nodup

utils/subset_data_dir.sh --speakers data/train_asr 30000 data/train_asr_30k
utils/subset_data_dir.sh --speakers data/train_asr 100000 data/train_asr_100k





# The next commands are not necessary for the scripts to run, but increase
# efficiency of data access by putting the mfcc's of the subset
# in a contiguous place in a file.
( . ./path.sh;
  # make sure mfccdir is defined as above..
  cp data/train_asr_10k_nodup/feats.scp{,.bak}
  copy-feats scp:data/train_asr_10k_nodup/feats.scp  ark,scp:$mfccdir/kaldi_fish_10k_nodup.ark,$mfccdir/kaldi_fish_10k_nodup.scp \
  && cp $mfccdir/kaldi_fish_10k_nodup.scp data/train_asr_10k_nodup/feats.scp
)
( . ./path.sh;
  # make sure mfccdir is defined as above..
  cp data/train_asr_30k/feats.scp{,.bak}
  copy-feats scp:data/train_asr_30k/feats.scp  ark,scp:$mfccdir/kaldi_fish_30k.ark,$mfccdir/kaldi_fish_30k.scp \
  && cp $mfccdir/kaldi_fish_30k.scp data/train_asr_30k/feats.scp
)
( . ./path.sh;
  # make sure mfccdir is defined as above..
  cp data/train_asr_100k/feats.scp{,.bak}
  copy-feats scp:data/train_asr_100k/feats.scp  ark,scp:$mfccdir/kaldi_fish_100k.ark,$mfccdir/kaldi_fish_100k.scp \
  && cp $mfccdir/kaldi_fish_100k.scp data/train_asr_100k/feats.scp
)


# ---------------------------------------------------------
# mono 训练
# 并使用 mono mdl 进行对齐 train_asr_30k
# ---------------------------------------------------------
steps/train_mono.sh --nj 10 --cmd "$train_cmd" \
  data/train_asr_10k_nodup data/lang exp/mono0a

steps/align_si.sh --nj 30 --cmd "$train_cmd" \
   data/train_asr_30k data/lang exp/mono0a exp/mono0a_ali || exit 1;



# ---------------------------------------------------------
# deltas 训练
# 使用 train_asr_30k 的align 进行训练 delta mdl
# out:
#    exp/tri1/mdl   exp/tri1_ali/align.gz
#    exp/tri2/mdl   exp/tri2_ali/align.gz( 对train_asr_100k 生成对齐align.gz )
# ---------------------------------------------------------
steps/train_deltas.sh --cmd "$train_cmd" \
    2500 20000 data/train_asr_30k data/lang exp/mono0a_ali exp/tri1 || exit 1;

# 重新构图 解码
(utils/mkgraph.sh data/lang_test exp/tri1 exp/tri1/graph
 steps/decode.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
   exp/tri1/graph data/dev_asr exp/tri1/decode_dev)&

# 对齐
steps/align_si.sh --nj 30 --cmd "$train_cmd" \
   data/train_asr_30k data/lang exp/tri1 exp/tri1_ali || exit 1;


steps/train_deltas.sh --cmd "$train_cmd" \
    2500 20000 data/train_asr_30k data/lang exp/tri1_ali exp/tri2 || exit 1;

(
  utils/mkgraph.sh data/lang_test exp/tri2 exp/tri2/graph || exit 1;
  steps/decode.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
   exp/tri2/graph data/dev_asr exp/tri2/decode_dev || exit 1;
)&

steps/align_si.sh --nj 30 --cmd "$train_cmd" \
  data/train_asr_100k data/lang exp/tri2 exp/tri2_ali || exit 1;



# ---------------------------------------------------------
# lda mllt 特征空间变换后训练
# 重新生成 train_asr_100k 的对齐  exp/tri3a_ali/align.gz
# ---------------------------------------------------------
# Train tri3a, which is LDA+MLLT, on 100k data.
steps/train_lda_mllt.sh --cmd "$train_cmd" \
   --splice-opts "--left-context=3 --right-context=3" \
   5000 40000 data/train_asr_100k data/lang exp/tri2_ali exp/tri3a || exit 1;
(
  utils/mkgraph.sh data/lang_test exp/tri3a exp/tri3a/graph || exit 1;
  steps/decode.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
   exp/tri3a/graph data/dev_asr exp/tri3a/decode_dev || exit 1;
)&

# Next we'll use fMLLR and train with SAT (i.e. on
# fMLLR features)

steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
  data/train_asr_100k data/lang exp/tri3a exp/tri3a_ali || exit 1;



# ---------------------------------------------------------
# sat 说话人自适应训练
# 使用 lda mllt的对齐 进行训练
# 第一迭代 sat 生成对齐 exp/tri4a_ali/align.gz
# 第二迭代 sat 生成最终模型 exp/tri5a
# ---------------------------------------------------------
steps/train_sat.sh  --cmd "$train_cmd" \
  5000 100000 data/train_asr_100k data/lang exp/tri3a_ali  exp/tri4a || exit 1;

(
  utils/mkgraph.sh data/lang_test exp/tri4a exp/tri4a/graph
  steps/decode_fmllr.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
   exp/tri4a/graph data/dev_asr exp/tri4a/decode_dev
)&

steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
  data/train_asr data/lang exp/tri4a exp/tri4a_ali || exit 1;


steps/train_sat.sh  --cmd "$train_cmd" \
  7000 300000 data/train_asr data/lang exp/tri4a_ali  exp/tri5a || exit 1;

# 最终构图, 但是后面没有对齐??
(
  utils/mkgraph.sh data/lang_test exp/tri5a exp/tri5a/graph
  steps/decode_fmllr.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
    exp/tri5a/graph data/dev_asr exp/tri5a/decode_dev
)&



# this will help find issues with the lexicon.
# steps/cleanup/debug_lexicon.sh --nj 300 --cmd "$train_cmd" data/train_asr_100k data/lang exp/tri5a data/local/dict/lexicon.txt exp/debug_lexicon_100k

# The following is based on an older nnet2 recipe.
local/dnn/run_nnet2_multisplice.sh
