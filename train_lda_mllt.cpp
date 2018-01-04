// this is a org
// 原始MFCC、语言模型、对齐数据。
void shellParamters(){

 // Begin configuration.
// cmd=run.pl
// config=
// stage=-5
// scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
// realign_iters="10 20 30";
// mllt_iters="2 4 6 12";
// num_iters=35    # Number of iterations of training
// max_iter_inc=25  # Last iter to increase #Gauss on.
// dim=40
// beam=10
// retry_beam=40
// careful=false
// boost_silence=1.0 # Factor by which to boost silence likelihoods in alignment
// power=0.25 # Exponent for number of gaussians according to occurrence counts
// randprune=4.0 # This is approximately the ratio by which we will speed up the
//               # LDA and MLLT calculations via randomized pruning.
// splice_opts=
// cluster_thresh=-1  # for build-tree control final bottom-up clustering of leaves
// norm_vars=false # deprecated.  Prefer --cmvn-opts "--norm-vars=false"
// cmvn_opts=
// context_opts=   # use "--context-width=5 --central-position=2" for quinphone.
// // End configuration.

// train_tree=true

// steps/train_lda_mllt.sh --cmd "$train_cmd" --splice-opts "--left-context=3 --right-context=3"


    // 2500
    // 15000
    // data/mfcc/train
    // data/lang
    // exp/tri1_ali
    // exp/tri2b
  
    // numleaves=$1  目标叶子节点数
    // totgauss=$2   高斯总数
    // data=$3       训练用特征
    // lang=$4       语言模型的fst
    // alidir=$5     上步对齐结果
    // dir=$6        目标输出

  
}



// splicedfeats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- |"
// # Note: $feats gets overwritten later in the script.
// feats="$splicedfeats transform-feats $dir/0.mat ark:- ark:- |"







