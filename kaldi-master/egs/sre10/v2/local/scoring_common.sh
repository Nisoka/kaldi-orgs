#!/bin/bash
# Copyright 2015   David Snyder
# Apache 2.0.
#

# local/scoring_common.sh \
#     data/sre    data/sre10_train    data/sre10_test \
#     exp/ivectors_sre   exp/ivectors_sre10_train   exp/ivectors_sre10_test

# 就是 将
#  train     data/sre
#  enroll    data/sre10_train
#  test      data/sre10_test
#  其中train  enroll 按照male female 划分数据 (包括 wav.scp  utt2spk , ivector  spk_ivector等)
#  test 当然只需要保留每一句的 utt-ivector 等待与其中的每个spk_ivector进行相似度比较,
#       但是还是先进行了 gender 的划分, 这样的识别是不是有点不自动了?


if [ $# != 6 ]; then
  echo "Usage: $0 <plda-data-dir> <enroll-data-dir> <test-data-dir> <plda-ivec-dir> <enroll-ivec-dir> <test-ivec-dir>"
fi
# data 路径(wav.scp utt2spk spk2gender)
plda_data_dir=${1%/}
enroll_data_dir=${2%/}
test_data_dir=${3%/}

# ivector 路径
plda_ivec_dir=${4%/}
enroll_ivec_dir=${5%/}
test_ivec_dir=${6%/}

if [ ! -f ${test_data_dir}/trials ]; then
  echo "${test_data_dir} needs a trial file."
  exit;
fi

mkdir -p local/.tmp

# 根据 spk2gender 将data 分为 female male 两部分数据
# 划分sre test enroll train 数据为 male female
# Partition the SRE data into male and female subsets.
cat ${plda_data_dir}/spk2gender | grep -w f > local/.tmp/female_spklist
utils/subset_data_dir.sh --spk-list local/.tmp/female_spklist ${plda_data_dir} ${plda_data_dir}_female
cat ${plda_data_dir}/spk2gender | grep -w m > local/.tmp/male_spklist
utils/subset_data_dir.sh --spk-list local/.tmp/male_spklist ${plda_data_dir} ${plda_data_dir}_male

cat ${test_data_dir}/spk2gender | grep -w f > local/.tmp/female_spklist
utils/subset_data_dir.sh --spk-list local/.tmp/female_spklist ${test_data_dir} ${test_data_dir}_female
cat ${test_data_dir}/spk2gender | grep -w m > local/.tmp/male_spklist
utils/subset_data_dir.sh --spk-list local/.tmp/male_spklist ${test_data_dir} ${test_data_dir}_male

cat ${enroll_data_dir}/spk2gender | grep -w f > local/.tmp/female_spklist
utils/subset_data_dir.sh --spk-list local/.tmp/female_spklist ${enroll_data_dir} ${enroll_data_dir}_female
cat ${enroll_data_dir}/spk2gender | grep -w m > local/.tmp/male_spklist
utils/subset_data_dir.sh --spk-list local/.tmp/male_spklist ${enroll_data_dir} ${enroll_data_dir}_male




# trials 文件, 以test_data_dir 中的trials 经过 female male的 utt2spk 进行过滤
# 准备 female male 的 trials
# Prepare female and male trials.

trials_female=${test_data_dir}_female/trials
trials_male=${test_data_dir}_male/trials

cat ${test_data_dir}/trials | awk '{print $2, $0}' | \
  utils/filter_scp.pl ${test_data_dir}_female/utt2spk | cut -d ' ' -f 2- \
  > $trials_female

cat ${test_data_dir}/trials | awk '{print $2, $0}' | \
  utils/filter_scp.pl ${test_data_dir}_male/utt2spk | cut -d ' ' -f 2- \
  > $trials_male



mkdir -p ${test_ivec_dir}_male
mkdir -p ${test_ivec_dir}_female
mkdir -p ${enroll_ivec_dir}_male
mkdir -p ${enroll_ivec_dir}_female
mkdir -p ${plda_ivec_dir}_male
mkdir -p ${plda_ivec_dir}_female




# 将ivector.scp 划分为 male female
# Partition the i-vectors into male and female subsets.
# enroll_data_dir 过滤ivector   为 male female
utils/filter_scp.pl ${enroll_data_dir}_male/utt2spk \
                    ${enroll_ivec_dir}/ivector.scp > ${enroll_ivec_dir}_male/ivector.scp
utils/filter_scp.pl ${enroll_data_dir}_female/utt2spk \
                    ${enroll_ivec_dir}/ivector.scp > ${enroll_ivec_dir}_female/ivector.scp

# enroll_data 过滤spk_ivector.scp spk均值ivector  为male female
utils/filter_scp.pl ${enroll_data_dir}_male/spk2utt \
                    ${enroll_ivec_dir}/spk_ivector.scp > ${enroll_ivec_dir}_male/spk_ivector.scp
utils/filter_scp.pl ${enroll_data_dir}_female/spk2utt \
                    ${enroll_ivec_dir}/spk_ivector.scp > ${enroll_ivec_dir}_female/spk_ivector.scp

# enroll_data 过滤num_utts.ark 为 male female
utils/filter_scp.pl ${enroll_data_dir}_male/spk2utt \
                    ${enroll_ivec_dir}/num_utts.ark > ${enroll_ivec_dir}_male/num_utts.ark
utils/filter_scp.pl ${enroll_data_dir}_female/spk2utt \
                    ${enroll_ivec_dir}/num_utts.ark > ${enroll_ivec_dir}_female/num_utts.ark


utils/filter_scp.pl ${plda_data_dir}_female/utt2spk \
                    ${plda_ivec_dir}/ivector.scp > ${plda_ivec_dir}_female/ivector.scp
utils/filter_scp.pl ${plda_data_dir}_male/utt2spk \
                    ${plda_ivec_dir}/ivector.scp > ${plda_ivec_dir}_male/ivector.scp

utils/filter_scp.pl ${plda_data_dir}_male/spk2utt \
                    ${plda_ivec_dir}/spk_ivector.scp > ${plda_ivec_dir}_male/spk_ivector.scp
utils/filter_scp.pl ${plda_data_dir}_female/spk2utt \
                    ${plda_ivec_dir}/spk_ivector.scp > ${plda_ivec_dir}_female/spk_ivector.scp
utils/filter_scp.pl ${plda_data_dir}_male/spk2utt \
                    ${plda_ivec_dir}/num_utts.ark > ${plda_ivec_dir}_male/num_utts.ark
utils/filter_scp.pl ${plda_data_dir}_female/spk2utt \
                    ${plda_ivec_dir}/num_utts.ark > ${plda_ivec_dir}_female/num_utts.ark


# test_data 过滤为 male female 的 utt-ivector, 每句utt 都会得到ivector 然后进行与enroll 相似度比较进行识别
utils/filter_scp.pl ${test_data_dir}_male/utt2spk \
                    ${test_ivec_dir}/ivector.scp > ${test_ivec_dir}_male/ivector.scp
utils/filter_scp.pl ${test_data_dir}_female/utt2spk \
                    ${test_ivec_dir}/ivector.scp > ${test_ivec_dir}_female/ivector.scp


# plda_ivec_dir 训练数据的 normalize-length 然后计算 mean
# 以及 分别的 male mean  female mean
# 长度归整 然后计算 mean.vec (分别计算每个data/sre  data/sre_male  data/sre_femal)
# Compute gender independent and dependent i-vector means.

run.pl ${plda_ivec_dir}/log/compute_mean.log \
  ivector-normalize-length scp:${plda_ivec_dir}/ivector.scp \
  ark:- \| ivector-mean ark:- ${plda_ivec_dir}/mean.vec || exit 1;

run.pl ${plda_ivec_dir}_male/log/compute_mean.log \
  ivector-normalize-length scp:${plda_ivec_dir}_male/ivector.scp \
  ark:- \| ivector-mean ark:- ${plda_ivec_dir}_male/mean.vec || exit 1;

run.pl ${plda_ivec_dir}_female/log/compute_mean.log \
  ivector-normalize-length scp:${plda_ivec_dir}_female/ivector.scp \
  ark:- \| ivector-mean ark:- ${plda_ivec_dir}_female/mean.vec || exit 1;

rm -rf local/.tmp
