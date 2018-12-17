#!/bin/bash


# This script dumps egs with several frames of labels, controlled by the
# frames_per_eg config variable (default: 8).  This takes many times less disk
# space because typically we have 4 to 7 frames of context on the left and
# right, and this ends up getting shared.  This is at the expense of slightly
# higher disk I/O while training.

set -o pipefail
trap "" PIPE

# Begin configuration section.
cmd=run.pl
frame_subsampling_factor=1
frames_per_eg=8   # number of frames of labels per example.  more->less disk space and
                  # less time preparing egs, but more I/O during training.
                  # Note: may in general be a comma-separated string of alternative
                  # durations (more useful when using large chunks, e.g. for BLSTMs);
                  # the first one (the principal num-frames) is preferred.
left_context=4    # amount of left-context per eg (i.e. extra frames of input features
                  # not present in the output supervision).
right_context=4   # amount of right-context per eg.

left_context_initial=-1    # if >=0, left-context for first chunk of an utterance
right_context_final=-1     # if >=0, right-context for last chunk of an utterance
compress=true   # set this to false to disable compression (e.g. if you want to see whether
                # results are affected).

num_utts_subset=300     # number of utterances in validation and training
                        # subsets used for shrinkage and diagnostics.
num_valid_frames_combine=0 # #valid frames for combination weights at the very end.
num_train_frames_combine=60000 # # train frames for the above.
num_frames_diagnostic=10000 # number of frames for "compute_prob" jobs
samples_per_iter=400000 # this is the target number of egs in each archive of egs
                        # (prior to merging egs).  We probably should have called
                        # it egs_per_iter. This is just a guideline; it will pick
                        # a number that divides the number of samples in the
                        # entire data.

transform_dir=     # If supplied, overrides alidir as the place to find fMLLR transforms

stage=0
nj=6         # This should be set to the maximum number of jobs you are
             # comfortable to run in parallel; you can increase it if your disk
             # speed is greater and you have more machines.
srand=0     # rand seed for nnet3-copy-egs and nnet3-shuffle-egs

online_ivector_dir=  # can be used if we are including speaker information as iVectors.
cmvn_opts=  # can be used for specifying CMVN options, if feature type is not lda (if lda,
            # it doesn't make sense to use different options than were used as input to the
            # LDA transform).  This is used to turn off CMVN in the online-nnet experiments.
generate_egs_scp=false # If true, it will generate egs.JOB.*.scp per egs archive

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: $0 [opts] <data> <ali-dir> <egs-dir>"
  echo " e.g.: $0 data/train exp/tri3_ali exp/tri4_nnet/egs"
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config file containing options"
  echo "  --nj <nj>                                        # The maximum number of jobs you want to run in"
  echo "                                                   # parallel (increase this only if you have good disk and"
  echo "                                                   # network speed).  default=6"
  echo "  --cmd (utils/run.pl;utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --samples-per-iter <#samples;400000>             # Target number of egs per archive (option is badly named)"
  echo "  --frames-per-eg <frames;8>                       # number of frames per eg on disk"
  echo "                                                   # May be either a single number or a comma-separated list"
  echo "                                                   # of alternatives (useful when training LSTMs, where the"
  echo "                                                   # frames-per-eg is the chunk size, to get variety of chunk"
  echo "                                                   # sizes).  The first in the list is preferred and is used"
  echo "                                                   # when working out the number of archives etc."
  echo "  --left-context <int;4>                           # Number of frames on left side to append for feature input"
  echo "  --right-context <int;4>                          # Number of frames on right side to append for feature input"
  echo "  --left-context-initial <int;-1>                  # If >= 0, left-context for first chunk of an utterance"
  echo "  --right-context-final <int;-1>                   # If >= 0, right-context for last chunk of an utterance"
  echo "  --num-frames-diagnostic <#frames;4000>           # Number of frames used in computing (train,valid) diagnostics"
  echo "  --num-valid-frames-combine <#frames;10000>       # Number of frames used in getting combination weights at the"
  echo "                                                   # very end."
  echo "  --stage <stage|0>                                # Used to run a partially-completed training process from somewhere in"
  echo "                                                   # the middle."

  exit 1;
fi




# data=args.feat_dir,                          # data/train_sp_hires    ---- high soluation feats.
# alidir=args.ali_dir,                         # exp/tri5a_sp_ali       ---- the hires-sp-features alignment.
# egs_dir=default_egs_dir,                     # exp/nnet3/tdnn_sp/egs  ???? output.
# frames_per_eg                                # 8  in train_tdnn.py's 

data=$1
alidir=$2
dir=$3

# Check some files.
# 不为空, 增加 ivector特征.
[ ! -z "$online_ivector_dir" ] && \
  extra_files="$online_ivector_dir/ivector_online.scp $online_ivector_dir/ivector_period"

for f in $data/feats.scp $alidir/ali.1.gz $alidir/final.mdl $alidir/tree $extra_files; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

# nj=30  split data ==> data/train_sp_hires/split30
sdata=$data/split$nj
utils/split_data.sh $data $nj

mkdir -p $dir/log $dir/info
cp $alidir/tree $dir

num_ali_jobs=$(cat $alidir/num_jobs) || exit 1;


# 数量不能低于 num_utts_subset*4, 否则认为数量不够训练一个 nnet
num_utts=$(cat $data/utt2spk | wc -l)
if ! [ $num_utts -gt $[$num_utts_subset*4] ]; then
  echo "$0: number of utterances $num_utts in your training data is too small versus --num-utts-subset=$num_utts_subset"
  echo "... you probably have so little data that it doesn't make sense to train a neural net."
  exit 1
fi

# utils/filter_scp.pl -f 1 $dir/utt.list $aishell_text > $dir/transcripts.txt
# 从 aishell_text 中过滤 得到 transcripts.txt 过滤规则是 第-f n个filed 在 utt.list中存在.

# utils/shuffle_list.pl  input.file   > output.file
# 随机化input.file 内line的顺序.

# utils/apply_map.pl map.txt  <input.file   >output.file
# 将input.file 中逐行 向map.txt中定义的map进行映射,

# eg
# map.txt
# A   a
# B   b
# C   c
# input.file
# A
# C
# output.file
# a
# c


# ----------------------------------------------
# 在train的utt中, 随机化,  取 300个 作为 valid_utt
# 用来验证什么?
awk '{print $1}' $data/utt2spk | utils/shuffle_list.pl | head -$num_utts_subset \
    > $dir/valid_uttlist || exit 1;

# 验证用utt 经过 apply_map.pl sort uniq 
if [ -f $data/utt2uniq ]; then  # this matters if you use data augmentation.
  echo "File $data/utt2uniq exists, so augmenting valid_uttlist to"
  echo "include all perturbed versions of the same 'real' utterances."
  
  mv $dir/valid_uttlist $dir/valid_uttlist.tmp
  
  utils/utt2spk_to_spk2utt.pl $data/utt2uniq > $dir/uniq2utt

  # 1 有效utt 经过apply_map utt2uniq 得到 n 个 spker
  # 2 apply_map map=uniq2utt 获得n个spker 的所有utt
  # 3 这些utt 最终得到 有效uttlist.
  cat $dir/valid_uttlist.tmp | utils/apply_map.pl $data/utt2uniq | \
    sort | uniq | utils/apply_map.pl $dir/uniq2utt | \
    awk '{for(n=1;n<=NF;n++) print $n;}' | sort  > $dir/valid_uttlist
  
  rm $dir/uniq2utt $dir/valid_uttlist.tmp
fi

# ----------------------------------------------
# 剩余的 乱序后 取 300 用作 train_subset_uttlist ???
# filter_scp  filter=utt2spk 将所有的valid_uttlist 去掉, 剩余的shuffle 乱序
# 取300 作为 train_subset_uttlist.
awk '{print $1}' $data/utt2spk | utils/filter_scp.pl --exclude $dir/valid_uttlist | \
   utils/shuffle_list.pl | head -$num_utts_subset > $dir/train_subset_uttlist || exit 1;





# alidir 
[ -z "$transform_dir" ] && transform_dir=$alidir


# false ============ no need
# because we'll need the features with a different number of jobs than $alidir, copy to ark,scp.
if [ -f $transform_dir/raw_trans.1 ]; then
  echo "$0: using raw transforms from $transform_dir"
  if [ $stage -le 0 ]; then
    $cmd $dir/log/copy_transforms.log \
      copy-feats "ark:cat $transform_dir/raw_trans.* |" "ark,scp:$dir/trans.ark,$dir/trans.scp"
  fi
fi




# ------------------------ 设置 数据 feats -------------
## Set up features.
echo "$0: feature type is raw"

# filter 过滤掉valid_uttlist中存在的utt 剩余的 作为feats
feats="ark,s,cs:utils/filter_scp.pl --exclude $dir/valid_uttlist $sdata/JOB/feats.scp      | apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:- ark:- |"

# filter 过滤获得valid_uttlist 作为valid_feats
valid_feats="ark,s,cs:utils/filter_scp.pl $dir/valid_uttlist $data/feats.scp               | apply-cmvn $cmvn_opts --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp scp:- ark:- |"

# filter 过滤获得train_subset_uttlist 作为train_subset_feats.
train_subset_feats="ark,s,cs:utils/filter_scp.pl $dir/train_subset_uttlist $data/feats.scp | apply-cmvn $cmvn_opts --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp scp:- ark:- |"



echo $cmvn_opts >$dir/cmvn_opts # caution: the top-level nnet training script should copy this to its own dir now.






# nil
if [ -f $dir/trans.scp ]; then
  feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk scp:$dir/trans.scp ark:- ark:- |"
  valid_feats="$valid_feats transform-feats --utt2spk=ark:$data/utt2spk scp:$dir/trans.scp ark:- ark:- |"
  train_subset_feats="$train_subset_feats transform-feats --utt2spk=ark:$data/utt2spk scp:$dir/trans.scp ark:- ark:- |"
fi


# ------------------------ ivector feats --------------
# t
if [ ! -z "$online_ivector_dir" ]; then
  ivector_dim=$(feat-to-dim scp:$online_ivector_dir/ivector_online.scp -) || exit 1;
  echo $ivector_dim > $dir/info/ivector_dim
  steps/nnet2/get_ivector_id.sh $online_ivector_dir > $dir/info/final.ie.id || exit 1
  ivector_period=$(cat $online_ivector_dir/ivector_period) || exit 1;
  ivector_opts="--online-ivectors=scp:$online_ivector_dir/ivector_online.scp --online-ivector-period=$ivector_period"
else
  ivector_opts=""
  echo 0 >$dir/info/ivector_dim
fi





# ---------------------- 计算 data/train_sp_hires 的数据帧frame总数 frame-dim---------------
# exp/nnet3/tdnn_sp/info/num_frames  frames_cnt
# feats_one
# feat_dim

if [ $stage -le 1 ]; then

  echo "$0: working out number of frames of training data"
  num_frames=$(steps/nnet2/get_num_frames.sh $data)
  echo $num_frames > $dir/info/num_frames

  
  echo "$0: working out feature dim"
  # sed s/target/result/g  查找target替换为result.
  feats_one="$(echo $feats | sed s/JOB/1/g)"
  if feat_dim=$(feat-to-dim "$feats_one" - 2>/dev/null); then
    echo $feat_dim > $dir/info/feat_dim
  fi
  
else
  num_frames=$(cat $dir/info/num_frames) || exit 1;
  feat_dim=$(cat $dir/info/feat_dim) || exit 1;
fi





# sample -- eg???

# frames = num_archives * sample_per_iter * frames_per_eg_principal
# what? num_archives
#       samplle_per_iter???
#       frames_per_eg     is example's frames??
# a archives is a batch.

# the first field in frames_per_eg (which is a comma-separated list of numbers)
# is the 'principal' frames-per-eg, and for purposes of working out the number
# of archives we assume that this will be the average number of frames per eg.
frames_per_eg_principal=$(echo $frames_per_eg | cut -d, -f1)

# num_frames = 
# the + 1 is to round up, not down... we assume it doesn't divide exactly.
num_archives=$[$num_frames/($frames_per_eg_principal*$samples_per_iter)+1]

if [ $num_archives -eq 1 ]; then
  echo "*** $0: warning: the --frames-per-eg is too large to generate one archive with"
  echo "*** as many as --samples-per-iter egs in it.  Consider reducing --frames-per-eg."
  sleep 4
fi






# We may have to first create a smaller number of larger archives, with number
# $num_archives_intermediate, if $num_archives is more than the maximum number
# of open filehandles that the system allows per process (ulimit -n).
# This sometimes gives a misleading answer as GridEngine sometimes changes that
# somehow, so we limit it to 512.
max_open_filehandles=$(ulimit -n) || exit 1
[ $max_open_filehandles -gt 512 ] && max_open_filehandles=512

# 修改archives_multiple 档案乘子
# 至少满足 num_archives_intermediate 内部数量 应该小于而不能近似为 max_opend_filehandles.
num_archives_intermediate=$num_archives
archives_multiple=1
while [ $[$num_archives_intermediate+4] -gt $max_open_filehandles ]; do
  archives_multiple=$[$archives_multiple+1]
  num_archives_intermediate=$[$num_archives/$archives_multiple+1];
done

# num_archives_intermediate???
# archives_multiple ???
# now make sure num_archives is an exact multiple of archives_multiple.
num_archives=$[$archives_multiple*$num_archives_intermediate]




# frames = num_archives * sample_per_iter * frames_per_eg_principal 
# frames = num_archives * egs_per_archive * frames_per_eg_principal
# Work out the number of egs per archive
# num_egs = num_frames/frames_per_eg_principal
# egs_per_archive = num_egs / num_archives
egs_per_archive=$[$num_frames/($frames_per_eg_principal*$num_archives)]
! [ $egs_per_archive -le $samples_per_iter ] && \
  echo "$0: script error: egs_per_archive=$egs_per_archive not <= samples_per_iter=$samples_per_iter" \
  && exit 1;


# --------------- 经过计算 得到 num_archives egs_per_archive frames_per_eg num_frames
# 52
echo $num_archives >$dir/info/num_archives
# 394272
echo $egs_per_archive > $dir/info/egs_per_archive
# 8  is equal to frames_per_eg_princepal
echo $frames_per_eg >$dir/info/frames_per_eg

# 164017282
# echo $num_frames > $dir/info/num_frames




# num_archives of archives,
#         egs_per_archives of egs,
#               frames_per_eg of [labels, (l, r) context]
echo "$0: creating $num_archives archives, each with $egs_per_archive egs, with"
echo "$0:   $frames_per_eg labels per example, and (left,right) context = ($left_context,$right_context)"
if [ $left_context_initial -ge 0 ] || [ $right_context_final -ge 0 ]; then
  echo "$0:   ... and (left-context-initial,right-context-final) = ($left_context_initial,$right_context_final)"
fi




# false
if [ -e $dir/storage ]; then
  # Make soft links to storage directories, if distributing this way..  See utils/create_split_dir.pl.
  echo "$0: creating data links"
  utils/create_data_link.pl $(for x in $(seq $num_archives); do echo $dir/egs.$x.ark; done)
  
  for x in $(seq $num_archives_intermediate); do
    utils/create_data_link.pl $(for y in $(seq $nj); do echo $dir/egs_orig.$y.$x.ark; done)
  done
fi





# ------------------ copy 数据的 对齐结果 --------------
# copy the alidir/ali.n.gz => dir(exp/nnet3/tdnn_sp/egs)ali.ark & ali.scp
if [ $stage -le 2 ]; then
  echo "$0: copying data alignments"
  for id in $(seq $num_ali_jobs); do gunzip -c $alidir/ali.$id.gz; done | \
    copy-int-vector ark:- ark,scp:$dir/ali.ark,$dir/ali.scp || exit 1;
fi


# ------------------ 上下文信息 ---------------
egs_opts="--left-context=$left_context --right-context=$right_context --compress=$compress --num-frames=$frames_per_eg"
[ $left_context_initial -ge 0 ] && egs_opts="$egs_opts --left-context-initial=$left_context_initial"
[ $right_context_final -ge 0 ] && egs_opts="$egs_opts --right-context-final=$right_context_final"

# 写入上下文信息
echo $left_context > $dir/info/left_context
echo $right_context > $dir/info/right_context
echo $left_context_initial > $dir/info/left_context_initial
echo $right_context_final > $dir/info/right_context_final






# get num_pdfs from the tree-info.
num_pdfs=$(tree-info --print-args=false $alidir/tree | grep num-pdfs | awk '{print $2}')



if [ $stage -le 3 ]; then
  # 1:
  # get validation and training subset examples .
  echo "$0: Getting validation and training subset examples."
  rm $dir/.error 2>/dev/null

  # 2:
  # extracting the validation and training-subset alignments.
  echo "$0: ... extracting validation and training-subset alignments."
  # do the filtering just once, as ali.scp may be long.
  utils/filter_scp.pl <(cat $dir/valid_uttlist $dir/train_subset_uttlist) \
                      <$dir/ali.scp >$dir/ali_special.scp

  
  # ================= nnet3-get-egs  构建 tdnn训练用样本结构 ===========
  #                Train  ---- train_subset_all.egs
  #                Validation -- valid_all.egs
  # 3:
  # and
  #     filter get the validation alignments. (trans-id)
  #     ali-to-pdf  ali-to-post  translate the tid to post
  #     nnet3-get-egs
  $cmd $dir/log/create_valid_subset.log \
    utils/filter_scp.pl $dir/valid_uttlist $dir/ali_special.scp \| \
    ali-to-pdf $alidir/final.mdl scp:- ark:- \| \
    ali-to-post ark:- ark:- \| \
    nnet3-get-egs --num-pdfs=$num_pdfs --frame-subsampling-factor=$frame_subsampling_factor \
      $ivector_opts $egs_opts "$valid_feats" \
      ark,s,cs:- "ark:$dir/valid_all.egs" || touch $dir/.error &

  
  $cmd $dir/log/create_train_subset.log \
    utils/filter_scp.pl $dir/train_subset_uttlist $dir/ali_special.scp \| \
    ali-to-pdf $alidir/final.mdl scp:- ark:- \| \
    ali-to-post ark:- ark:- \| \
    nnet3-get-egs --num-pdfs=$num_pdfs --frame-subsampling-factor=$frame_subsampling_factor \
      $ivector_opts $egs_opts "$train_subset_feats" \
      ark,s,cs:- "ark:$dir/train_subset_all.egs" || touch $dir/.error &
  wait;



  
  [ -f $dir/.error ] && echo "Error detected while creating train/valid egs" && exit 1







  # =================== nnet3-subset-egs  获取n个 样本 egs ===========
  # 1 valid_combine.egs
  # 2 valid_diagnostic.egs
  # 3 train_combine.egs
  # 4 train_diagnostic.egs
  
  echo "... Getting subsets of validation examples for diagnostics and combination."
  # false..
  if $generate_egs_scp; then
    valid_diagnostic_output="ark,scp:$dir/valid_diagnostic.egs,$dir/valid_diagnostic.scp"
    train_diagnostic_output="ark,scp:$dir/train_diagnostic.egs,$dir/train_diagnostic.scp"
  else
    valid_diagnostic_output="ark:$dir/valid_diagnostic.egs"
    train_diagnostic_output="ark:$dir/train_diagnostic.egs"
  fi

  
  $cmd $dir/log/create_valid_subset_combine.log \
    nnet3-subset-egs --n=$[$num_valid_frames_combine/$frames_per_eg_principal] ark:$dir/valid_all.egs \
    ark:$dir/valid_combine.egs || touch $dir/.error &

  
  $cmd $dir/log/create_valid_subset_diagnostic.log \
    nnet3-subset-egs --n=$[$num_frames_diagnostic/$frames_per_eg_principal] ark:$dir/valid_all.egs \
    $valid_diagnostic_output || touch $dir/.error &

  $cmd $dir/log/create_train_subset_combine.log \
    nnet3-subset-egs --n=$[$num_train_frames_combine/$frames_per_eg_principal] ark:$dir/train_subset_all.egs \
    ark:$dir/train_combine.egs || touch $dir/.error &
  

  $cmd $dir/log/create_train_subset_diagnostic.log \
    nnet3-subset-egs --n=$[$num_frames_diagnostic/$frames_per_eg_principal] ark:$dir/train_subset_all.egs \
    $train_diagnostic_output || touch $dir/.error &
  wait


  
  sleep 5  # wait for file system to sync.
  
  cat $dir/valid_combine.egs $dir/train_combine.egs > $dir/combine.egs

  if $generate_egs_scp; then
    cat $dir/valid_combine.egs $dir/train_combine.egs  | \
    nnet3-copy-egs ark:- ark,scp:$dir/combine.egs,$dir/combine.scp
    rm $dir/{train,valid}_combine.scp
  else
    cat $dir/valid_combine.egs $dir/train_combine.egs > $dir/combine.egs
  fi

  
  for f in $dir/{combine,train_diagnostic,valid_diagnostic}.egs; do
    [ ! -s $f ] && echo "No examples in file $f" && exit 1;
  done

  # 移除tmp文件
  rm $dir/valid_all.egs $dir/train_subset_all.egs $dir/{train,valid}_combine.egs
fi










# ==================== nnet3-get-egs 构建 tdnn-nnet3 训练用结构样本数据 ===========
# ==================== 划分训练样本, 将egs 分给不同的并行任务job ==================

# ---- 得到 exp/nnet3/tdnn_sp/egs/egs_orig.JOB.1:52.ark

if [ $stage -le 4 ]; then
  # create egs_orig.*.*.ark; the first index goes to $nj,
  # the second to $num_archives_intermediate.

  # 生成 num_archives_intermediate 那么多的 egs.JOB.$n(1:52).ark
  egs_list=
  for n in $(seq $num_archives_intermediate); do
    egs_list="$egs_list ark:$dir/egs_orig.JOB.$n.ark"
  done

  # =====================  !!!! 使用全部数据, 生成 egs

  echo "$0: Generating training examples on disk"
  # -------- 从全部数据中 获得egs > 生成 egs_orig.JOB.$n.ark  JOB: 1:16
  # -------- 并行JOB数量抽取数据. 得到 egs_orig.JOB(1:16).$n(1:52).ark
  $cmd JOB=1:$nj $dir/log/get_egs.JOB.log \
    nnet3-get-egs --num-pdfs=$num_pdfs --frame-subsampling-factor=$frame_subsampling_factor \
    $ivector_opts $egs_opts "$feats" \
    "ark,s,cs:filter_scp.pl $sdata/JOB/utt2spk $dir/ali.scp | ali-to-pdf $alidir/final.mdl scp:- ark:- | ali-to-post ark:- ark:- |" ark:- \| \
    nnet3-copy-egs --random=true --srand=\$[JOB+$srand] ark:- $egs_list || exit 1;
fi






# num_archives  表示压缩文件数量
# egs_per_archives 表示一个压缩文件内 egs样本数量
# frames_per_eg    表示一个样本内 frame数量

# ==============> num_frames = num_archives * egs_per_archives * frames_per_eg
# liujunnan@innovem:/data/home/liujunnan/aishell/s5/exp/nnet3/tdnn_sp/egs/info$ cat num_frames 
# 164017282
# liujunnan@innovem:/data/home/liujunnan/aishell/s5/exp/nnet3/tdnn_sp/egs/info$ cat num_archives 
# 52
# liujunnan@innovem:/data/home/liujunnan/aishell/s5/exp/nnet3/tdnn_sp/egs/info$ cat egs_per_archive 
# 394272
# liujunnan@innovem:/data/home/liujunnan/aishell/s5/exp/nnet3/tdnn_sp/egs/info$ cat frames_per_eg 
# 8

# >>> 394272 * 52 * 8 - 164017282
# -130 (这个误差是 chunk 之间有重叠)
# >>> 



# =================== 将 egs_orig.(1:16).(1:52).ark 归档到 egs.JOB(1:52).ark中.
# ------------------- 1:52 是目标并行进行TDNN训练的并行数
# ------------------- 1:16 是原本抽取特征时并行抽取的进程数.
# ------------------- 因此和上面看着好像有不同 实际上是对应的.
if [ $stage -le 5 ]; then
  # 重新组合 并乱序 这些egs的压缩文件
  echo "$0: recombining and shuffling order of archives on disk"
  # combine all the "egs_orig.*.JOB.scp" (over the $nj splits of the data) and
  # shuffle the order, writing to the egs.JOB.ark

  # the input is a concatenation over the input jobs.
  egs_list=
  for n in $(seq $nj); do
    egs_list="$egs_list $dir/egs_orig.$n.JOB.ark"
  done

  # true
  if [ $archives_multiple == 1 ]; then # normal case.
    if $generate_egs_scp; then
      output_archive="ark,scp:$dir/egs.JOB.ark,$dir/egs.JOB.scp"
    else
      output_archive="ark:$dir/egs.JOB.ark"
    fi

    #  JOB 1: 52, 将egs_list exp/nnet3/tdnn_sp/egs/egs_orig.$n.JOB.ark 归档 > egs.JOB.ark中.
    $cmd --max-jobs-run $nj JOB=1:$num_archives_intermediate $dir/log/shuffle.JOB.log \
      nnet3-shuffle-egs --srand=\$[JOB+$srand] "ark:cat $egs_list|" $output_archive  || exit 1;

    # false
    if $generate_egs_scp; then
      #concatenate egs.JOB.scp in single egs.scp
      rm $dir/egs.scp 2> /dev/null || true
      for j in $(seq $num_archives_intermediate); do
        cat $dir/egs.$j.scp || exit 1;
      done > $dir/egs.scp || exit 1;
      for f in $dir/egs.*.scp; do rm $f; done
    fi
  else
    # we need to shuffle the 'intermediate archives' and then split into the
    # final archives.  we create soft links to manage this splitting, because
    # otherwise managing the output names is quite difficult (and we don't want
    # to submit separate queue jobs for each intermediate archive, because then
    # the --max-jobs-run option is hard to enforce).
      
    # 太长没用的 删掉了 具体看源码中的代码.......
  fi
fi



# =====================  后期删除一些 tmp文件 ==============
if [ $frame_subsampling_factor -ne 1 ]; then
  echo $frame_subsampling_factor > $dir/info/frame_subsampling_factor
fi


if [ $stage -le 6 ]; then
    
  echo "$0: removing temporary archives"
  for x in $(seq $nj); do
    for y in $(seq $num_archives_intermediate); do
      file=$dir/egs_orig.$x.$y.ark
      [ -L $file ] && rm $(utils/make_absolute.sh $file)
      rm $file
    done
  done

  # false
  if [ $archives_multiple -gt 1 ]; then
    # there are some extra soft links that we should delete.
    for f in $dir/egs.*.*.ark; do rm $f; done
  fi
  
  echo "$0: removing temporary alignments and transforms"
  # Ignore errors below because trans.* might not exist.
  rm $dir/{ali,trans}.{ark,scp} 2>/dev/null
fi

echo "$0: Finished preparing training examples"
