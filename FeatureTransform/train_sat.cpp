
// #sat
// steps/train_sat.sh --cmd "$train_cmd" 2500 15000 data/mfcc/train data/lang exp/tri2b_ali exp/tri3b || exit 1;

// #sat_ali
// steps/align_fmllr.sh --nj $n --cmd "$train_cmd" data/mfcc/train data/lang exp/tri3b exp/tri3b_ali



// # This does Speaker Adapted Training (SAT), i.e. train on
// # fMLLR-adapted features.  It can be done on top of either LDA+MLLT, or
// # delta and delta-delta features.  If there are no transforms supplied
// # in the alignment directory, it will estimate transforms itself before
// # building the tree (and in any case, it estimates transforms a number
// # of times during training).


/* # Begin configuration section.
stage=-5
exit_stage=-100 # you can use this to require it to exit at the
                # beginning of a specific stage.  Not all values are
                # supported.

fmllr_update_type=full
cmd=run.pl
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
beam=10
retry_beam=40
careful=false

boost_silence=1.0 # Factor by which to boost silence likelihoods in alignment

context_opts=  # e.g. set this to "--context-width 5 --central-position 2" for quinphone.

realign_iters="10 20 30";

fmllr_iters="2 4 6 12";
silence_weight=0.0 # Weight on silence in fMLLR estimation.
num_iters=35   # Number of iterations of training
max_iter_inc=25 # Last iter to increase #Gauss on.
power=0.2 # Exponent for number of gaussians according to occurrence counts
cluster_thresh=-1  # for build-tree control final bottom-up clustering of leaves

phone_map=
train_tree=true
tree_stats_opts=
cluster_phones_opts=
compile_questions_opts=
# End configuration section. 



if [ $# != 6 ]; then
  echo "Usage: steps/train_sat.sh <#leaves> <#gauss> <data> <lang> <ali-dir> <exp-dir>"
  echo " e.g.: steps/train_sat.sh 2500 15000 data/train_si84 data/lang exp/tri2b_ali_si84 exp/tri3b"
  echo "Main options (for others, see top of script file)"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --config <config-file>                           # config containing options"
  echo "  --stage <stage>                                  # stage to do partial re-run from."
  exit 1;
fi

numleaves=$1  // 2500
totgauss=$2   // 15000
data=$3       // data/mfcc/train
lang=$4       // data/lang
alidir=$5     // exp/tri2b_ali   =====================  last ali result
dir=$6        // exp/tri3b       =====================  this output


numgauss=$numleaves
incgauss=$[($totgauss-$numgauss)/$max_iter_inc]  # per-iter #gauss increment

*/


/*
oov=`cat $lang/oov.int`
nj=`cat $alidir/num_jobs` || exit 1;
silphonelist=`cat $lang/phones/silence.csl`
ciphonelist=`cat $lang/phones/context_indep.csl` || exit 1;
sdata=$data/split$nj;
splice_opts=`cat $alidir/splice_opts 2>/dev/null` # frame-splicing options.
cmvn_opts=`cat $alidir/cmvn_opts 2>/dev/null`
delta_opts=`cat $alidir/delta_opts 2>/dev/null`
phone_map_opt=
[ ! -z "$phone_map" ] && phone_map_opt="--phone-map='$phone_map'"

*/



void MainInHere(){
  
}


// !!!!!!!!!!!!!!!!!!!!!!!!! # Set up speaker-independent features.
// sifeats  --- 经过lda_mllt(final.mat) 变换后的特征-  不考虑说话人的特征

// sifeats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $alidir/final.mat ark:- ark:- |"

// cp $alidir/final.mat $dir
// cp $alidir/full.mat $dir 2>/dev/null





// ======================================== gmm-est-fmllr =========================
// ## Get initial fMLLR transforms (possibly from alignment dir)
//   echo "$0: obtaining initial fMLLR transforms since not present in $alidir"
// // 根据 trans-id 对齐序列 得到 probablly<pdf-id, prob> 对齐序列
//   ali-to-post "ark:gunzip -c $alidir/ali.JOB.gz|" ark:- \|              \
//   weight-silence-post $silence_weight $silphonelist $alidir/final.mdl ark:- ark:- \| \

//   gmm-est-fmllr --fmllr-update-type=$fmllr_update_type --spk2utt=ark:$sdata/JOB/spk2utt $alidir/final.mdl "$sifeats" \
//                             ark:- ark:$dir/trans.JOB || exit 1;



//                                 lda_mllt.mdl   sifeats           <posteriors,prob>对齐   多个转移矩阵
// "Usage: gmm-est-fmllr [options] <model-in> <feature-rspecifier> <post-rspecifier> <transform-wspecifier>\n";
int gmm_est_fmllr(int argc, char *argv[]) {


    const char *usage =
        "Estimate global fMLLR transforms, either per utterance or for the supplied\n"
        "set of speakers (spk2utt option).  Reads posteriors (on transition-ids).  Writes\n"
        "to a table of matrices.\n"

    ParseOptions po(usage);
    FmllrOptions fmllr_opts;
    string spk2utt_rspecifier;
    po.Register("spk2utt", &spk2utt_rspecifier, "rspecifier for speaker to "
                "utterance-list map");
    
    
    string
        model_rxfilename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        post_rspecifier = po.GetArg(3),
        trans_wspecifier = po.GetArg(4);

    TransitionModel trans_model;
    AmDiagGmm am_gmm;
    {
      bool binary;
      Input ki(model_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_gmm.Read(ki.Stream(), binary);
    }

    RandomAccessPosteriorReader post_reader(post_rspecifier);

    double tot_impr = 0.0, tot_t = 0.0;

    
    BaseFloatMatrixWriter transform_writer(trans_wspecifier);

    int32 num_done = 0, num_no_post = 0, num_other_error = 0;
    if (spk2utt_rspecifier != "") {  // per-speaker adaptation
      SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
      RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);

      // foreach spker
      for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
        // 构造一个统计量对象 用来估计 fmllr 变换的矩阵
        FmllrDiagGmmAccs spk_stats(am_gmm.Dim(), fmllr_opts);
        // spker and it's utts
        string spk = spk2utt_reader.Key();
        const vector<string> &uttlist = spk2utt_reader.Value();
        // foreach utt in a spker
        for (size_t i = 0; i < uttlist.size(); i++) {
          std::string utt = uttlist[i];
          // utt feats 和 <pdf-id, prob> 对齐信息
          const Matrix<BaseFloat> &feats = feature_reader.Value(utt);
          const Posterior &post = post_reader.Value(utt);

          AccumulateForUtterance(feats, post, trans_model, am_gmm, &spk_stats);
          num_done++;
        }  // end looping over all utterances of the current speaker


        // 计算 fmllr 转移矩阵， 对于一个人实现一个特定的转移矩阵.
        BaseFloat impr, spk_tot_t;
        {  // Compute the transform and write it out.
          Matrix<BaseFloat> transform(am_gmm.Dim(), am_gmm.Dim()+1);
          transform.SetUnit();
          // 计算 fmllr 转移矩阵.
          spk_stats.Update(fmllr_opts, &transform, &impr, &spk_tot_t);
          transform_writer.Write(spk, transform);
        }

        tot_impr += impr;
        tot_t += spk_tot_t;
      }  // end looping over speakers
    } else {  // per-utterance adaptation

    }
    return (num_done != 0 ? 0 : 1);
}


/*
  in:
  feats  utt的特征
  post   <pdf-id, prob> 
  trans-mdl
  am-gmm 所有gmm pdf参数
  out:
  spk_stats 某个spk统计量
  
*/
void AccumulateForUtterance(const Matrix<BaseFloat> &feats,
                            const Posterior &post,
                            const TransitionModel &trans_model,
                            const AmDiagGmm &am_gmm,
                            FmllrDiagGmmAccs *spk_stats) {
  Posterior pdf_post;
  ConvertPosteriorToPdfs(trans_model, post, &pdf_post);

  // foreach frame
  for (size_t i = 0; i < post.size(); i++) {
    // foreach probably <pdf-id, prob>.
    for (size_t j = 0; j < pdf_post[i].size(); j++) {
      int32 pdf_id = pdf_post[i][j].first;
      // 向spker统计量中 增加统计量
      spk_stats->AccumulateForGmm(am_gmm.GetPdf(pdf_id),
                                  feats.Row(i),
                                  pdf_post[i][j].second);
    }
  }
}

/*
  in:
  pdf  DiagGmm
  data MFCC特征
  weight  pdf权重
  
*/
BaseFloat FmllrDiagGmmAccs::AccumulateForGmm(const DiagGmm &pdf,
                                             const VectorBase<BaseFloat> &data,
                                             BaseFloat weight) {
  int32 num_comp = pdf.NumGauss();
  // 分量模型概率
  Vector<BaseFloat> posterior(num_comp);
  BaseFloat loglike;
  // 分量后验概率（用来更新GMM参数的统计量）,
  loglike = pdf.ComponentPosteriors(data, &posterior);
  
  posterior.Scale(weight);
  // 根据 刚刚得到的后验概率 以及 pdf各分量的方差协方差均值参数 获得统计量
  AccumulateFromPosteriors(pdf, data, posterior);
  return loglike;
}

// 计算估计 fmllr用的 统计量 (某个pdf-id 的均值矩阵、协方差矩阵、pdf内分量后验概率等信息)
void FmllrDiagGmmAccs:: AccumulateFromPosteriors(
    const DiagGmm &pdf,
    const VectorBase<BaseFloat> &data,
    const VectorBase<BaseFloat> &posterior) {
  
  if (this->DataHasChanged(data)) {
    CommitSingleFrameStats();
    InitSingleFrameStats(data);
  }
  
  // 统计量 保存在 this->single_frame_stats_ 中.
  SingleFrameStats &stats = this->single_frame_stats_;
  stats.count += posterior.Sum();
  stats.a.AddMatVec(1.0, pdf.means_invvars(), kTrans, posterior, 1.0);
  stats.b.AddMatVec(1.0, pdf.inv_vars(), kTrans, posterior, 1.0);
}










//   对每个spk 执行对应的转移矩阵, 将sifeats 通过转移矩阵转换特征 实现特征
//   feats="$sifeats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark,s,cs:$dir/trans.JOB ark:- ark:- |"
//   cur_trans_dir=$dir









//  汇总决策树用统计量
//   # Get tree stats.
//   acc-tree-stats $context_opts $tree_stats_opts $phone_map_opt --ci-phones=$ciphonelist $alidir/final.mdl "$feats" \
//     "ark:gunzip -c $alidir/ali.JOB.gz|" $dir/JOB.treeacc || exit 1;

//   sum-tree-stats $dir/treeacc $dir/*.treeacc || exit 1;



//  构建决策树
//   echo "$0: Getting questions for tree clustering."
//   # preparing questions, roots file...
//   cluster-phones $cluster_phones_opts $context_opts $dir/treeacc $lang/phones/sets.int $dir/questions.int 2>$dir/log/questions.log || exit 1;
//   cat $lang/phones/extra_questions.int >> $dir/questions.int
//   compile-questions $context_opts $compile_questions_opts $lang/topo $dir/questions.int $dir/questions.qst 2>$dir/log/compile_questions.log || exit 1;


//   echo "$0: Building the tree"
//     build-tree $context_opts --verbose=1 --max-leaves=$numleaves \
//     --cluster-thresh=$cluster_thresh $dir/treeacc $lang/phones/roots.int \
//     $dir/questions.qst $lang/topo $dir/tree || exit 1;





// 根据决策树，初始化模型
//   echo "$0: Initializing the model"
//     gmm-init-model  --write-occs=$dir/1.occs  \
//       $dir/tree $dir/treeacc $lang/topo $dir/1.mdl 

//   # Convert the alignments.
//   echo "$0: Converting alignments from $alidir to use current tree"
//   convert-ali $phone_map_opt $alidir/final.mdl $dir/1.mdl $dir/tree \
//     "ark:gunzip -c $alidir/ali.JOB.gz|" "ark:|gzip -c >$dir/ali.JOB.gz" 


//   根据对齐信息 构建utt转换图 fst.
//   echo "$0: Compiling graphs of transcripts"
//     compile-train-graphs --read-disambig-syms=$lang/phones/disambig.int $dir/tree $dir/1.mdl  $lang/L.fst  \
//      "ark:utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt < $sdata/JOB/text |" \
//       "ark:|gzip -c >$dir/fsts.JOB.gz" 




// x=1
// while [ $x -lt $num_iters ]; do
//     echo Aligning data
//     根据fst图 生成 对齐序列
//       gmm-align-compiled $scale_opts --beam=$beam --retry-beam=$retry_beam --careful=$careful "$mdl" \
//       "ark:gunzip -c $dir/fsts.JOB.gz|" "$feats" \
//       "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1;


//       echo Estimating fMLLR transforms
//       对应用了fmllr的特征 在继续估计 fmllr转移矩阵, 然后 下次迭代 将多次的fmllr变换矩阵compose点乘一下.
//       # We estimate a transform that's additional to the previous transform;
//       # we'll compose them.

//         ali-to-post "ark:gunzip -c $dir/ali.JOB.gz|" ark:-  \| \
//         weight-silence-post $silence_weight $silphonelist $dir/$x.mdl ark:- ark:- \| \
//         gmm-est-fmllr --fmllr-update-type=$fmllr_update_type
//                       --spk2utt=ark:$sdata/JOB/spk2utt $dir/$x.mdl "$feats" ark:- ark:$dir/tmp_trans.JOB || exit 1;



//       对所有的 并行任务 进行对应组合(因为一个spk 一定在一个任务中)，这里compose-transforms 使用的是 affine模式
//       并不是标准点乘
//       for n in `seq $nj`; do
//         compose-transforms --b-is-affine=true \
//           ark:$dir/tmp_trans.$n ark:$cur_trans_dir/trans.$n ark:$dir/composed_trans.$n \
//           && mv $dir/composed_trans.$n $dir/trans.$n 
//       done

//     特征变换， 应用刚刚est 得到的fmllr变换矩阵 进行特征变换 得到新的特征.
//     feats="$sifeats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark:$dir/trans.JOB ark:- ark:- |"
//     cur_trans_dir=$dir
//   fi

//     因为特征发生变化, 重新est GMM参数, 然后循环迭代进行 估计 fmllr, 最终得到一个（累积的）fmllr变换矩阵.
//     并且通过每次迭代 最终得到一个 经过了 fmllr 变换得到的   模型 final.mdl
//       gmm-acc-stats-ali $dir/$x.mdl "$feats" \
//       "ark,s,cs:gunzip -c $dir/ali.JOB.gz|" $dir/$x.JOB.acc || exit 1;
//       gmm-est --power=$power --write-occs=$dir/$[$x+1].occs --mix-up=$numgauss $dir/$x.mdl \
//       "gmm-sum-accs - $dir/$x.*.acc |" $dir/$[$x+1].mdl || exit 1;


//   x=$[$x+1];
// done






    
// if [ $stage -le $x ]; then
//   # Accumulate stats for "alignment model"-- this model is
//   # computed with the speaker-independent features, but matches Gaussian-for-Gaussian
//   # with the final speaker-adapted model.


//   从新对齐结果 两种特征 计算重新估计gmm 和 trans的转移概率 的统计量 
//     ali-to-post "ark:gunzip -c $dir/ali.JOB.gz|" ark:-  \| \
//     gmm-acc-stats-twofeats $dir/$x.mdl "$feats" "$sifeats" ark,s,cs:- $dir/$x.JOB.acc 



//     ali-to-post "ark:gunzip -c $dir/ali.JOB.gz|" ark:-  \| \
//     gmm-acc-stats-twofeats $dir/$x.mdl "$feats" "$sifeats" ark,s,cs:- $dir/$x.JOB.acc
//    feats 用来获得后验概率
//    sifeats 用来计算统计量的
//    计算 包括 gmm参数估计用统计量、 trans-model 状态转移概率用统计量.
int gmm_acc_stats_twofeats(int argc, char *argv[]) {
  using namespace kaldi;
  const char *usage =
      "Accumulate stats for GMM training, computing posteriors with one set of features\n"        
      "but accumulating statistics with another.\n"
      "First features are used to get posteriors, second to accumulate stats\n"        
      "Usage:  gmm-acc-stats-twofeats [options] <model-in> <feature1-rspecifier> <feature2-rspecifier> <posteriors-rspecifier> <stats-out>\n"


  std::string
      model_filename = po.GetArg(1),
      feature1_rspecifier = po.GetArg(2),
      feature2_rspecifier = po.GetArg(3),
      posteriors_rspecifier = po.GetArg(4),
      accs_wxfilename = po.GetArg(5);

  using namespace kaldi;
  typedef kaldi::int32 int32;

  AmDiagGmm am_gmm;
  TransitionModel trans_model;
  {
    bool binary;
    Input ki(model_filename, &binary);
    trans_model.Read(ki.Stream(), binary);
    am_gmm.Read(ki.Stream(), binary);
  }

  // 从trans-model 获得转移统计量??
  // 开始只是为所有trans-id 申请一个统计变量
  Vector<double> transition_accs;
  trans_model.InitStats(&transition_accs);
    // void InitStats(Vector<double> *stats) const { stats->Resize(NumTransitionIds()+1); } 
   
  
  int32 new_dim = 0;
  AccumAmDiagGmm gmm_accs;
  // will initialize once we know new_dim.

  double tot_like = 0.0;
  double tot_t = 0.0;

  SequentialBaseFloatMatrixReader feature1_reader(feature1_rspecifier);
  RandomAccessBaseFloatMatrixReader feature2_reader(feature2_rspecifier);
  RandomAccessPosteriorReader posteriors_reader(posteriors_rspecifier);

  int32 num_done = 0, num_no2ndfeats = 0, num_no_posterior = 0, num_other_error = 0;
  // foreach utt feats( fmllr )
  for (; !feature1_reader.Done(); feature1_reader.Next()) {
    
    std::string key = feature1_reader.Key();
    if (!feature2_reader.HasKey(key)) {

    } else {
      // frame x everyFeature  === (M X N)
      const Matrix<BaseFloat> &mat1 = feature1_reader.Value();
      const Matrix<BaseFloat> &mat2 = feature2_reader.Value(key);

      if (new_dim == 0) {
        new_dim = mat2.NumCols();
        gmm_accs.Init(am_gmm, new_dim, kGmmAll);
      }
      
      const Posterior &posterior = posteriors_reader.Value(key);


      num_done++;
      BaseFloat
          tot_like_this_file = 0.0,
          tot_weight_this_file = 0.0;

      Posterior pdf_posterior;
      // 将posterior保存的 <trans-id, prob> 转化为 <pdf-id, prob>
      // 不同的trans-id 可能是相同的 pdf-id 
      ConvertPosteriorToPdfs(trans_model, posterior, &pdf_posterior);

      // foreach frame
      for (size_t i = 0; i < posterior.size(); i++) {
        // Accumulates for GMM.  GMM累积量.
        // foreach frame probable <pdf-id, prob>.
        for (size_t j = 0; j <pdf_posterior[i].size(); j++) {
          int32 pdf_id = pdf_posterior[i][j].first;
          BaseFloat weight = pdf_posterior[i][j].second;
          
          tot_like_this_file += weight *
              gmm_accs.AccumulateForGmmTwofeats(am_gmm,
                                                mat1.Row(i),
                                                mat2.Row(i),
                                                pdf_id,
                                                weight);
          tot_weight_this_file += weight;
        }

        // Accumulates for transitions.
        // foreach frame probable <trans-id, prob>.
        for (size_t j = 0; j < posterior[i].size(); j++) {
          int32 tid = posterior[i][j].first;
          BaseFloat weight = posterior[i][j].second;
          // 统计每个trans-id 的权重和(一般weight = 1, 如果是可能概率的话会<1 训练是不会出现).
          trans_model.Accumulate(weight, tid, &transition_accs);
        }
      }
      tot_like += tot_like_this_file;
      tot_t += tot_weight_this_file;

    }
  }

  {
    Output ko(accs_wxfilename, binary);
    transition_accs.Write(ko.Stream(), binary);
    gmm_accs.Write(ko.Stream(), binary);
  }
  

}


//  某frame feature1 feature2, 可能的pdf-id， 以及权重
// tot_like_this_file += weight *
//     gmm_accs.AccumulateForGmmTwofeats(am_gmm,
//                                       mat1.Row(i),
//                                       mat2.Row(i),
//                                       pdf_id,
//                                       weight);

BaseFloat AccumAmDiagGmm::AccumulateForGmmTwofeats(
    const AmDiagGmm &model,
    const VectorBase<BaseFloat> &data1,
    const VectorBase<BaseFloat> &data2,
    int32 gmm_index,
    BaseFloat weight) {

  // 通过pdf-id获得gmm参数.
  const DiagGmm &gmm = model.GetPdf(gmm_index);
  // 获得pdf的全部统计量
  AccumDiagGmm &acc = *(gmm_accumulators_[gmm_index]);
  
  Vector<BaseFloat> posteriors;
  // 计算GMM各分量后验概率, 将GMM中各分量的后验概率 -> posteriors
  BaseFloat log_like = gmm.ComponentPosteriors(data1, &posteriors);
  posteriors.Scale(weight);
  // feature2 经过lda_mllt变换的特征, 该gmm的分量后验概率
  // 计算一些统计量.
  acc.AccumulateFromPosteriors(data2, posteriors);
  
  total_log_like_ += log_like * weight;
  total_frames_ += weight;
  return log_like;
}

// in:
// data 某帧特征
// gmm 的分量后验概率
// 汇总 一些统计量 后验概率的平方 协方差 等.
void AccumDiagGmm::AccumulateFromPosteriors(
    const VectorBase<BaseFloat> &data,
    const VectorBase<BaseFloat> &posteriors) {
  
  Vector<double> post_d(posteriors);  // Copy with type-conversion
  
  // accumulate 汇总统计量occupancy
  occupancy_.AddVec(1.0, post_d);
  
  if (flags_ & kGmmMeans) {
    Vector<double> data_d(data);  // Copy with type-conversion
    mean_accumulator_.AddVecVec(1.0, post_d, data_d);
    if (flags_ & kGmmVariances) {
      data_d.ApplyPow(2.0);
      variance_accumulator_.AddVecVec(1.0, post_d, data_d);
    }
  }
}




//   最后根据 两种特征 得到的统计量 进行GMM参数估计, 得到  final.alimdl
//   # Update model.
//     gmm-est --power=$power --remove-low-count-gaussians=false $dir/$x.mdl \
//     "gmm-sum-accs - $dir/$x.*.acc|"   $dir/$x.alimdl  || exit 1;
//   rm $dir/$x.*.acc
// fi






// $x.mdl  是循环内经过fmllr变换后的MFCC特征 进行估计的  模型  ---> final.mdl
// $x.alimdl 是经过fmllr变换后, 同时使用 fmllr特征 以及 原本的lda_mllt特征 进行估计得到的  模型 ---> final.alimdl

// 以后选择使用那个mdl作为声学特征 更好点.

// rm $dir/final.{mdl,alimdl,occs} 2>/dev/null
// ln -s $x.mdl $dir/final.mdl
// ln -s $x.occs $dir/final.occs
// ln -s $x.alimdl $dir/final.alimdl













