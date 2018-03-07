s#!/bin/bash

set -euo pipefail

# This script is modified based on mini_librispeech/s5/local/nnet3/run_ivector_common.sh

# This script is called from local/nnet3/run_tdnn.sh and
# local/chain/run_tdnn.sh (and may eventually be called by more
# scripts).  It contains the common feature preparation and
# iVector-related parts of the script.  See those scripts for examples
# of usage.



# local/nnet3/run_ivector_common.sh --stage 0 || exit 1;

stage=0
train_set=train
test_sets="dev test"
gmm=tri5a

nnet3_affix=

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

# sat 最终获得得到 结果模型
gmm_dir=exp/${gmm}

# out: 对齐结果目录
ali_dir=exp/${gmm}_sp_ali

# data/train/feats.scp  exp/tri5a/final.mdl
for f in data/${train_set}/feats.scp ${gmm_dir}/final.mdl; do
  if [ ! -f $f ]; then
    echo "$0: expected file $f to exist"
    exit 1
  fi
done


# perturb 扰动正常数据, 获得alignment   _sp 代表速度扰动过的结果.


# out:
# data/train_sp   speed perturbed data's feats.scp
# mfcc_perturbed  sp data's mfcc-perturbed-feature.ark (mfcc_perturbed_features cmvn.)

if [ $stage -le 1 ]; then
  # Although the nnet will be trained by high resolution data, we still have to
  # perturb the normal data to get the alignment _sp stands for speed-perturbed
  echo "$0: preparing directory for low-resolution speed-perturbed data (for alignment)"
  utils/data/perturb_data_dir_speed_3way.sh data/${train_set} data/${train_set}_sp

  echo "$0: making MFCC features for low-resolution speed-perturbed data"
  steps/make_mfcc_pitch.sh --cmd "$train_cmd" --nj 70 data/${train_set}_sp \
    exp/make_mfcc/train_sp      mfcc_perturbed || exit 1;

  steps/compute_cmvn_stats.sh data/${train_set}_sp \
    exp/make_mfcc/train_sp      mfcc_perturbed || exit 1;

  utils/fix_data_dir.sh data/${train_set}_sp
fi


# 使用 tri5a中的模型 对 sp(speed_perturbed)数据进行 对齐 输出 ==> exp/tri5a_sp_ali.
# use the model in tri5a, align the data speed_perturbed to get the ali output to the dir tri5a_sp_ali.

# out:
# exp/tri5a_sp_ali    use the final.mdl in tri5a to generate the ali-result of data/train_sp.
if [ $stage -le 2 ]; then
  echo "$0: aligning with the perturbed low-resolution data"
  steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
    data/${train_set}_sp data/lang $gmm_dir $ali_dir || exit 1
fi




# usefor:

# out:
#    data/train_sp_hires   copy the sp data from data/train_sp to data/train_sp_hires for generate the hires data.
#    mfcc_perturbed_hires   extra the hires mfcc-perturbed-feature from the data/train_sp_hires(data/train_sp)
#    data/train_sp_hires_nopitch   get the nopitch hires mfcc-perturbed-feature to extra the ivectors

if [ $stage -le 3 ]; then
  # Create high-resolution MFCC features (with 40 cepstra instead of 13).
  # this shows how you can split across multiple file-systems.
    
  echo "$0: creating high-resolution MFCC features"
  mfccdir=mfcc_perturbed_hires
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    utils/create_split_dir.pl /export/b0{5,6,7,8}/$USER/kaldi-data/mfcc/aishell-$(date +'%m_%d_%H_%M')/s5/$mfccdir/storage $mfccdir/storage
  fi

  for datadir in ${train_set}_sp ${test_sets}; do
    utils/copy_data_dir.sh data/$datadir data/${datadir}_hires
  done

  # do volume-perturbation on the training data prior to extracting hires
  # features; this helps make trained nnets more invariant to test data volume.
  utils/data/perturb_data_dir_volume.sh data/${train_set}_sp_hires || exit 1;

  for datadir in ${train_set}_sp ${test_sets}; do
    steps/make_mfcc_pitch.sh --nj 10 --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/${datadir}_hires exp/make_hires/$datadir $mfccdir || exit 1;
    steps/compute_cmvn_stats.sh data/${datadir}_hires exp/make_hires/$datadir $mfccdir || exit 1;
    
    utils/fix_data_dir.sh data/${datadir}_hires || exit 1;
    
    # create MFCC data dir without pitch to extract iVector
    utils/data/limit_feature_dim.sh 0:39 data/${datadir}_hires data/${datadir}_hires_nopitch || exit 1;
    steps/compute_cmvn_stats.sh data/${datadir}_hires_nopitch exp/make_hires/$datadir $mfccdir || exit 1;
  done
fi




# out:
#    THE UBM
if [ $stage -le 4 ]; then
  echo "$0: computing a subset of data to train the diagonal UBM."
  # We'll use about a quarter of the data.
  mkdir -p exp/nnet3${nnet3_affix}/diag_ubm
  temp_data_root=exp/nnet3${nnet3_affix}/diag_ubm

  num_utts_total=$(wc -l <data/${train_set}_sp_hires_nopitch/utt2spk)
  num_utts=$[$num_utts_total/4]
  utils/data/subset_data_dir.sh data/${train_set}_sp_hires_nopitch \
     $num_utts ${temp_data_root}/${train_set}_sp_hires_nopitch_subset

  echo "$0: computing a PCA transform from the hires data."
  steps/online/nnet2/get_pca_transform.sh --cmd "$train_cmd" \
      --splice-opts "--left-context=3 --right-context=3" \
      --max-utts 10000 --subsample 2 \
       ${temp_data_root}/${train_set}_sp_hires_nopitch_subset \
       exp/nnet3${nnet3_affix}/pca_transform

  echo "$0: training the diagonal UBM."
  # Use 512 Gaussians in the UBM.
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 30 \
    --num-frames 700000 \
    --num-threads 8 \
    ${temp_data_root}/${train_set}_sp_hires_nopitch_subset 512 \
    exp/nnet3${nnet3_affix}/pca_transform exp/nnet3${nnet3_affix}/diag_ubm
fi



# OUT:
#     TRAIN THE IVECTOR-EXTRACTOR.
if [ $stage -le 5 ]; then
  # Train the iVector extractor.  Use all of the speed-perturbed data since iVector extractors
  # can be sensitive to the amount of data.  The script defaults to an iVector dimension of
  # 100.
  echo "$0: training the iVector extractor"
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 10 \
     data/${train_set}_sp_hires_nopitch exp/nnet3${nnet3_affix}/diag_ubm \
     exp/nnet3${nnet3_affix}/extractor || exit 1;
fi







train_set=train_sp
# USEFOR:
#     extract the ivectors-features.
# out:
#     exp/nnet3/ivectors_train_sp  extract the ivectors-features from the data/train_sp_hires_nopitch.


if [ $stage -le 6 ]; then
  # We extract iVectors on the speed-perturbed training data after combining
  # short segments, which will be what we train the system on.  With
  # --utts-per-spk-max 2, the script pairs the utterances into twos, and treats
  # each of these pairs as one speaker; this gives more diversity in iVectors..
  # Note that these are extracted 'online'.

  # note, we don't encode the 'max2' in the name of the ivectordir even though
  # that's the data we extract the ivectors from, as it's still going to be
  # valid for the non-'max2' data, the utterance list is the same.

  ivectordir=exp/nnet3${nnet3_affix}/ivectors_${train_set}
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $ivectordir/storage ]; then
    utils/create_split_dir.pl /export/b0{5,6,7,8}/$USER/kaldi-data/ivectors/aishell-$(date +'%m_%d_%H_%M')/s5/$ivectordir/storage $ivectordir/storage
  fi

  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  temp_data_root=${ivectordir}
  utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
    data/${train_set}_hires_nopitch ${temp_data_root}/${train_set}_sp_hires_nopitch_max2
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 30 \
    ${temp_data_root}/${train_set}_sp_hires_nopitch_max2 \
    exp/nnet3${nnet3_affix}/extractor $ivectordir

  # Also extract iVectors for the test data, but in this case we don't need the speed
  # perturbation (sp).
  for data in $test_sets; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 8 \
      data/${data}_hires_nopitch exp/nnet3${nnet3_affix}/extractor \
      exp/nnet3${nnet3_affix}/ivectors_${data}
  done
fi

exit 0