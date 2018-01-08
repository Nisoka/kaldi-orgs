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


/* splicedfeats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- |"

 # Note: $feats gets overwritten later in the script.
 feats="$splicedfeats transform-feats $dir/0.mat ark:- ark:- |" */

// splice-feats 扩展一句utt特征向量, 将原本每帧D维向量, 变为 (1+8) = 9xD维.  eg frame=4 (0,1,2,3 + 4 + 5,6,7,8)
void SpliceFrames(const MatrixBase<BaseFloat> &input_features,
                  int32 left_context,  //4
                  int32 right_context, //4
                  Matrix<BaseFloat> *output_features) {
  int32 T = input_features.NumRows(), D = input_features.NumCols();
  
  int32 N = 1 + left_context + right_context;
  
  output_features->Resize(T, D*N);
  
  for (int32 t = 0; t < T; t++) {
    // 将output_features 的 t帧数据空间 使用SubVecotor 引用
    SubVector<BaseFloat> dst_row(*output_features, t);
    
    // N = 9 left_context = 4;  t = 4,  t2 = 0
    for (int32 j = 0; j < N; j++) {
      int32 t2 = t + j - left_context;
      if (t2 < 0) t2 = 0;
      if (t2 >= T) t2 = T-1;
      // 使用output_features的空间, 每帧 N*D大小,
      // 通过 指针dst 每次选择一个D的数据, 从原feat上下文context 加入到output_features中.
      SubVector<BaseFloat> dst(dst_row, j*D, D),
          src(input_features, t2);
      dst.CopyFromVec(src);
    }
  }
}


// transform-feats $dir/0.mat ark:- ark:-
// 利用0.mat 将输入的feat 进行lda 降维 得到 40维特征,
// 最后进行了一个叫什么 LogDet的计算 不知道具体作用.
int transform_feats(int argc, char *argv[]) {
    using namespace kaldi;

    const char *usage =
        "Apply transform (e.g. LDA; HLDA; fMLLR/CMLLR; MLLT/STC)\n"
        "Linear transform if transform-num-cols == feature-dim, affine if\n"
        "transform-num-cols == feature-dim+1 (->append 1.0 to features)\n"
        "Per-utterance by default, or per-speaker if utt2spk option provided\n"
        "Global if transform-rxfilename provided.\n"
        "Usage: transform-feats [options] (<transform-rspecifier>|<transform-rxfilename>) <feats-rspecifier> <feats-wspecifier>\n"
        "See also: transform-vec, copy-feats, compose-transforms\n";
        
    ParseOptions po(usage);
    std::string utt2spk_rspecifier;

    std::string transform_rspecifier_or_rxfilename = po.GetArg(1);
    std::string feat_rspecifier = po.GetArg(2);
    std::string feat_wspecifier = po.GetArg(3);

    SequentialBaseFloatMatrixReader feat_reader(feat_rspecifier);
    BaseFloatMatrixWriter feat_writer(feat_wspecifier);

    RandomAccessBaseFloatMatrixReaderMapped transform_reader;
    bool use_global_transform;

    //通过 ark:- 读取的splicedfeats  是 所有utt特征
    Matrix<BaseFloat> global_transform;

    if (ClassifyRspecifier(transform_rspecifier_or_rxfilename, NULL, NULL) == kNoRspecifier) {
      // not an rspecifier -> interpret as rxfilename....
      use_global_transform = true;
      ReadKaldiObject(transform_rspecifier_or_rxfilename, &global_transform);
    } else {  // an rspecifier -> not a global transform.

    }

    enum { Unknown, Logdet, PseudoLogdet, DimIncrease };
    int32 logdet_type = Unknown;
    double tot_t = 0.0, tot_logdet = 0.0;  // to compute average logdet weighted by time...
    int32 num_done = 0, num_error = 0;
    BaseFloat cached_logdet = -1;

    // foreach utt
    for (;!feat_reader.Done(); feat_reader.Next()) {
      
      std::string utt = feat_reader.Key();

      // utt old-feats
      const Matrix<BaseFloat> &feat(feat_reader.Value());
      // utt lda-0.mat
      const Matrix<BaseFloat> &trans =
          (use_global_transform ? global_transform : transform_reader.Value(utt));
      
      int32
          // lda 0.mat M
          transform_rows = trans.NumRows(),
          // lda 0.mat N
          transform_cols = trans.NumCols(),
          // old-feats dim-M
          feat_dim = feat.NumCols();

      // old-feats -- [T, 0.mat-M]
      // 0.mat        [N', M']
      
      // ==> 构造 lda-feats -- [T, 0.mat-N]
      Matrix<BaseFloat> feat_out(feat.NumRows(), transform_rows);

      if (transform_cols == feat_dim) {
        feat_out.AddMatMat(1.0, feat, kNoTrans, trans, kTrans, 0.0);
      } else if (transform_cols == feat_dim + 1) {
        // append the implicit 1.0 to the input features.
        SubMatrix<BaseFloat> linear_part(trans, 0, transform_rows, 0, feat_dim);
        feat_out.AddMatMat(1.0, feat, kNoTrans, linear_part, kTrans, 0.0);
        Vector<BaseFloat> offset(transform_rows);
        offset.CopyColFromMat(trans, feat_dim);
        feat_out.AddVecToRows(1.0, offset);
      }
      
      num_done++;

      if (logdet_type == Unknown) {
        // 这里没有发生降维, no
        if (transform_rows == feat_dim) logdet_type = Logdet;  // actual logdet.
        // 降维 yes logdet_type = PseudoLogdet
        else if (transform_rows < feat_dim) logdet_type = PseudoLogdet;  // see below 伪logdet计算
        else logdet_type = DimIncrease;  // makes no sense to have any logdet.
        
        // PseudoLogdet is if we have a dimension-reducing transform T, we compute
        // 1/2 logdet(T T^T).  Why does this make sense?

        // Imagine we do MLLT after LDA and compose the transforms;
        // the MLLT matrix is A and the LDA matrix is L,
        // so T = A L.  T T^T = A L L^T A, so 1/2 logdet(T T^T) = logdet(A) + 1/2 logdet(L L^T).
        // since L L^T is a constant, this is valid for comparing likelihoods if we're
        // just trying to see if the MLLT is converging.
      }

      // trans --  40x91, feat_dim - 91
      if (logdet_type != DimIncrease) {
        // Accumulate log-determinant stats. 统计log 统计量  N-out，            N-in
        SubMatrix<BaseFloat> linear_transform(trans, 0, trans.NumRows(), 0, feat_dim);
        //                                     M      ro       r          co  c
        // 进行了一遍拷贝 trans
        // 1645   // point to the begining of window
        // 1646   MatrixBase<Real>::num_rows_ = r;
        // 1647   MatrixBase<Real>::num_cols_ = c;

        // stride_ = M.stride_ >= cols--91
        // 1648   MatrixBase<Real>::stride_ = M.Stride();  ???

        // data_  = M.data_
        // 1649   MatrixBase<Real>::data_ = M.Data_workaround() +
        // 1650       static_cast<size_t>(co) +
        // 1651       static_cast<size_t>(ro) * static_cast<size_t>(M.Stride());
        // 1652 }
        
        // "linear_transform" is just the linear part of any transform, ignoring
        // any affine (offset) component.
        SpMatrix<BaseFloat> TT(trans.NumRows());
        // TT = linear_transform * linear_transform^T
        // 40x91 X 91x40 --> TT = 40X40
        TT.AddMat2(1.0, linear_transform, kNoTrans, 0.0);
        BaseFloat logdet;
        
        if (use_global_transform) {  // true
          if (cached_logdet == -1)
            cached_logdet = 0.5 * TT.LogDet(NULL);
          
          // Real SpMatrix<Real>::LogDet(Real *det_sign) const {
          //   Real log_det;
          //   SpMatrix<Real> tmp(*this);
          //   // false== output not needed (saves some computation).
          //   求矩阵的逆
          //   tmp.Invert(&log_det, det_sign, false);
          //   return log_det;
          // }          
          logdet = cached_logdet;
        }
        
        {
          tot_t += feat.NumRows();
          tot_logdet += feat.NumRows() * logdet;
        }
      }
      
      feat_writer.Write(utt, feat_out);
    }
}

// feat_out 输出特征 保存的data [T, N]  == feat_in [T, 13*9] X [13*9, 40]
//  trans 保存的是转置矩阵 -- [40, 13*9]
void MatrixBase<Real>::AddMatMat(const Real alpha,
                                  const MatrixBase<Real>& A,  // feature in
                                  MatrixTransposeType transA,
                                  const MatrixBase<Real>& B,
                                  MatrixTransposeType transB,
                                  const Real beta) {

  ASSERT(transA == kNoTrans && transB == kTrans  // A 非转置   B 转置
         && A.num_cols_ == B.num_cols_   // 13*9
         && A.num_rows_ == num_rows_     //feat_in-T, feat_out-T
         && B.num_rows_ == num_cols_);   //trans-N, feat_out-N
  // C = alpha*A*B + beta*C
  // 实际就是通过 feat-in X trans0.mat ---> feat_out.
  // blas 库中的代码
  cblas_Xgemm(alpha, transA, A.data_, A.num_rows_, A.num_cols_, A.stride_,
              transB, B.data_, B.stride_, beta, data_, num_rows_, num_cols_, stride_);
}







// ali-to-post "ark:gunzip -c $alidir/ali.JOB.gz|" ark:- \|\
//   weight-silence-post 0.0 $silphonelist $alidir/final.mdl ark:- ark:- \|  \
//   acc-lda --rand-prune=$randprune $alidir/final.mdl "$splicedfeats" ark,s,cs:- \
//   $dir/lda.JOB.acc || exit 1;

// est-lda --write-full-matrix=$dir/full.mat --dim=$dim $dir/0.mat $dir/lda.*.acc \
//   2>$dir/log/lda_est.log || exit 1;
//   rm $dir/lda.*.acc





// 1 ali-to-post
//   in: 对齐结果 每帧用对应trans-id结果表示
//   out: 修改为 每帧用 对应的 多个可能的vector<trans-id, probability> 来表示
/** @brief Convert alignments to viterbi style posteriors. The aligned
    symbol gets a weight of 1.0 */
int ali_to_post(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  
    const char *usage =
        "Convert alignments to posteriors.  This is simply a format change\n"
        "from integer vectors to Posteriors, which are vectors of lists of\n"
        "pairs (int, float) where the float represents the posterior.  The\n"
        "floats would all be 1.0 in this case.\n"
        "The posteriors will still be in terms of whatever integer index\n"
        "the input contained, which will be transition-ids if they came\n"
        "directly from decoding, or pdf-ids if they were processed by\n"
        "ali-to-post.\n"
        "Usage:  ali-to-post [options] <alignments-rspecifier> <posteriors-wspecifier>\n"
        "e.g.:\n"
        " ali-to-post ark:1.ali ark:1.post\n"
        
        "See also: ali-to-pdf, ali-to-phones, show-alignments, post-to-weights\n";

    ParseOptions po(usage);


    std::string alignments_rspecifier = po.GetArg(1);
    std::string posteriors_wspecifier = po.GetArg(2);

    int32 num_done = 0;
    SequentialInt32VectorReader alignment_reader(alignments_rspecifier);
    PosteriorWriter posterior_writer(posteriors_wspecifier);

    // 每句utt的对齐信息
    for (; !alignment_reader.Done(); alignment_reader.Next()) {
      num_done++;
      // utt -- vector<trans-id>
      const std::vector<int32> &alignment = alignment_reader.Value();
      
      // Posterior is vector<vector<pair<int32, BaseFloat> > >
      Posterior post;
      AlignmentToPosterior(alignment, &post);
      posterior_writer.Write(alignment_reader.Key(), post);
    }
}

// 将一句utt 对齐信息
// 从 trans-id 对齐方式  --->  vector<trains-id, probability>

// Posterior is vector<vector<pair<int32, BaseFloat> > >
// Posterior 保存的是每个utt的每个时间帧的vector多个可能pair<trans-id, probability>
void AlignmentToPosterior(const std::vector<int32> &ali,
                          Posterior *post) {
  post->clear();
  post->resize(ali.size());  // 设置为utt对齐 大小

  for (size_t i = 0; i < ali.size(); i++) {
    (*post)[i].resize(1);
    (*post)[i][0].first = ali[i];
    (*post)[i][0].second = 1.0;
  }
}



// 2 weight-silence-post
//   weight-silence-post 0.0 $silphonelist $alidir/final.mdl ark:- ark:- |
// //in:
// 0.0 ?
// silphonelist --- sil 只保存了这个音素
// final.mdl    train_deltas的模型结果
// ark:- ali-to-post 输出的Posterior类型对齐结果
// //out:
// ark:-  为sil 音素 增加了silence_scale 权重变换的 postritor。

int weight_silence_post(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
    const char *usage =
        "Apply weight to silences in posts\n"
        "Usage:  weight-silence-post [options] <silence-weight> <silence-phones> "
        "<model> <posteriors-rspecifier> <posteriors-wspecifier>\n"
        "e.g.:\n"
        " weight-silence-post 0.0 1:2:3 1.mdl ark:1.post ark:nosil.post\n";

    ParseOptions po(usage);

    bool distribute = false;

    std::string
        silence_weight_str = po.GetArg(1), //0.0
        silence_phones_str = po.GetArg(2),
        model_rxfilename = po.GetArg(3),
        posteriors_rspecifier = po.GetArg(4),
        posteriors_wspecifier = po.GetArg(5);

    BaseFloat silence_weight = 0.0;
    // silence_weight_str = 0.0
    // 所以 silence_weight   0.0
    // silence_set  <1>
    if (!ConvertStringToReal(silence_weight_str, &silence_weight))
      KALDI_ERR << "Invalid silence-weight parameter: expected float, got \""
                 << silence_weight_str << '"';
    
    std::vector<int32> silence_phones;
    if (!SplitStringToIntegers(silence_phones_str, ":", false, &silence_phones))
      KALDI_ERR << "Invalid silence-phones string " << silence_phones_str;
    if (silence_phones.empty())
      KALDI_WARN <<"No silence phones, this will have no effect";
    // 使用set集合保存数组,能够去重 快速查找.
    ConstIntegerSet<int32> silence_set(silence_phones);  // faster lookup.

    TransitionModel trans_model;
    ReadKaldiObject(model_rxfilename, &trans_model);

    int32 num_posteriors = 0;
    SequentialPosteriorReader posterior_reader(posteriors_rspecifier);
    PosteriorWriter posterior_writer(posteriors_wspecifier);

    // 每utt
    for (; !posterior_reader.Done(); posterior_reader.Next()) {
      num_posteriors++;
      // Posterior is vector<vector<pair<int32, BaseFloat> > >
      Posterior post = posterior_reader.Value();

      // 对sil音素增加0.0的权重
      if (distribute)
        WeightSilencePostDistributed(trans_model, silence_set,
                                     silence_weight, &post);
      else
        WeightSilencePost(trans_model, silence_set,
                          silence_weight, &post);

      posterior_writer.Write(posterior_reader.Key(), post);
    }
}

// trans_model 是用来判断一个tid 属于那个phone用的？
void WeightSilencePost(const TransitionModel &trans_model,
                       const ConstIntegerSet<int32> &silence_set,
                       BaseFloat silence_scale,
                       Posterior *post) {
  // foreach frame i
  for (size_t i = 0; i < post->size(); i++) {
    // 所有可能的trans-id, probability
    std::vector<std::pair<int32, BaseFloat> > this_post;
    this_post.reserve((*post)[i].size());

    // 所有可能trans-id
    for (size_t j = 0; j < (*post)[i].size(); j++) {
      int32 tid = (*post)[i][j].first,
          phone = trans_model.TransitionIdToPhone(tid);
      BaseFloat weight = (*post)[i][j].second;

      // if a sli
      if (silence_set.count(phone) != 0) {  // is a silence.
        if (silence_scale != 0.0)
          this_post.push_back(std::make_pair(tid, weight*silence_scale));
      } else {
        this_post.push_back(std::make_pair(tid, weight));
      }
    }
    
    (*post)[i].swap(this_post);
  }
}





// 3 acc-lda
//   acc-lda --rand-prune=$randprune $alidir/final.mdl "$splicedfeats" ark,s,cs:- \
// in:
//     rand_prune
//     final.mdl
//     splicefeats ----- 经过增加前后帧数据的 特征
// out:
//     ark:- 输出统计量

/** @brief
    Accumulate LDA statistics based on pdf-ids.
    Inputs are the source models, that serve as the input
    (and may potentially contain the current transformation),
    the un-transformed features and state posterior probabilities
*/
int acc_lda(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  
  const char *usage =
      "Accumulate LDA statistics based on pdf-ids.\n"
      "Usage:  acc-lda [options] <transition-gmm/model> <features-rspecifier> <posteriors-rspecifier> <lda-acc-out>\n"
      "Typical usage:\n"
      " ali-to-post ark:1.ali ark:- | lda-acc 1.mdl \"ark:splice-feats scp:train.scp|\"  ark:- ldaacc.1\n";

    bool binary = true;
    BaseFloat rand_prune = 0.0;
    ParseOptions po(usage);

    std::string model_rxfilename = po.GetArg(1);
    std::string features_rspecifier = po.GetArg(2);
    std::string posteriors_rspecifier = po.GetArg(3);
    
    std::string acc_wxfilename = po.GetArg(4);

    TransitionModel trans_model;
    
    LdaEstimate lda;
    SequentialBaseFloatMatrixReader feature_reader(features_rspecifier);
    RandomAccessPosteriorReader posterior_reader(posteriors_rspecifier);

    int32 num_done = 0, num_fail = 0;

    // foreach utt
    for (;!feature_reader.Done(); feature_reader.Next()) {
      std::string utt = feature_reader.Key();

      // utt posterior & feats
      const Posterior &post (posterior_reader.Value(utt));
      const Matrix<BaseFloat> &feats(feature_reader.Value());

      // 所有可能的pdf-id, 特征维度(9 * 13)???
      if (lda.Dim() == 0)
        lda.Init(trans_model.NumPdfs(), feats.NumCols());
      
      // void LdaEstimate::Init(int32 num_classes, int32 dimension) {
      //   zero_acc_.Resize(num_classes); 每个pdf-id 统计量
      //   first_acc_.Resize(num_classes, dimension); 每个pdf-id 每维度 统计量
      //   total_second_acc_.Resize(dimension);  每维度统计量？？SpMatrix<double> SpMatrix 是个什么矩阵呢？
      // }

      // pdf-post 将每帧 Vector<trans-id, probability> 转为每帧 Vector<pdf-id, probability>
      Posterior pdf_post;
      ConvertPosteriorToPdfs(trans_model, post, &pdf_post);

      // foreach frame i
      for (int32 i = 0; i < feats.NumRows(); i++) {
        // 保存 当前帧 feats
        SubVector<BaseFloat> feat(feats, i);
        // 当前帧 所有可能pdf-id, 进行剪枝, 加入lda.Accumulate 统计量.
        // 统计量包括  feat, 对应可能的<pdf-id, weight>
        for (size_t j = 0; j < pdf_post[i].size(); j++) {
          int32 pdf_id = pdf_post[i][j].first;          
          BaseFloat weight = RandPrune(pdf_post[i][j].second, rand_prune);
          if (weight != 0.0) {
            lda.Accumulate(feat, pdf_id, weight);
          }
        }
      }
      num_done++;
    }

    //  save lda acc
    Output ko(acc_wxfilename, binary);
    lda.Write(ko.Stream(), binary);
}

// 将vector<tid, prob> --> vector<pdf-id, prob>
void ConvertPosteriorToPdfs(const TransitionModel &tmodel,
                            const Posterior &post_in,
                            Posterior *post_out) {
  post_out->clear();
  post_out->resize(post_in.size());
  // foreach frame i
  for (size_t i = 0; i < post_out->size(); i++) {
    
    unordered_map<int32, BaseFloat> pdf_to_post;
    // foreach prob tid
    for (size_t j = 0; j < post_in[i].size(); j++) {
      
      int32 tid = post_in[i][j].first,
          pdf_id = tmodel.TransitionIdToPdf(tid);
      BaseFloat post = post_in[i][j].second;
      if (pdf_to_post.count(pdf_id) == 0)
        pdf_to_post[pdf_id] = post;
      else
        pdf_to_post[pdf_id] += post;
    }
    // 传递给调用
    (*post_out)[i].reserve(pdf_to_post.size());
    // 去掉 = 0.0
    for (unordered_map<int32, BaseFloat>::const_iterator iter =
             pdf_to_post.begin(); iter != pdf_to_post.end(); ++iter) {
      if (iter->second != 0.0)
        (*post_out)[i].push_back(
            std::make_pair(iter->first, iter->second));
    }
  }
}

void LdaEstimate::Accumulate(const VectorBase<BaseFloat> &data,
                             int32 class_id, BaseFloat weight) {

  Vector<double> data_d(data);

  // zero_acc_ vector<pdf-id weight> 保存pdf-id 的权重累和
  zero_acc_(class_id) += weight;
  
  // first_acc_ pdf-id 的权重以及特征 对每个pdf-id 的 对应特征 [features ] + [weight x data_d]
  first_acc_.Row(class_id).AddVec(weight, data_d);
  
  // total_second_acc 所有pdf-id feature的二次统计量 value += weight x features x features‘ -- 1.0 x [1x3] [3x1]
  // 保存的并不是个向量.???  SpMatrix
  total_second_acc_.AddVec2(weight, data_d);
}

// void SpMatrix<Real>::AddVec2(const Real alpha, const VectorBase<OtherReal> &v) {

//   Real *data = this->data_;
//   const OtherReal *v_data = v.Data();
//   MatrixIndexT nr = this->num_rows_;
//   for (MatrixIndexT i = 0; i < nr; i++)
//     for (MatrixIndexT j = 0; j <= i; j++, data++)
//       *data += alpha * v_data[i] * v_data[j];
// }

          





// 4 est-lda 根据统计量 使用lda降维 输入特征得到 降维矩阵0.mat
// est-lda --write-full-matrix=$dir/full.mat --dim=$dim $dir/0.mat $dir/lda.*.acc \
//                      full-mat                          out-mat     lda-acc
// in: lda.acc
// out: 0.mat
// full.mat???

int est_lda(int argc, char *argv[]) {  
  using namespace kaldi;
  typedef kaldi::int32 int32;
    const char *usage =
        "Estimate LDA transform using stats obtained with acc-lda.\n"
        "Usage:  est-lda [options] <lda-matrix-out> <lda-acc-1> <lda-acc-2> ...\n";

    bool binary = true;
    std::string full_matrix_wxfilename;
    
    LdaEstimateOptions opts;
    
    po.Register("write-full-matrix", &full_matrix_wxfilename,
                "Write full LDA matrix to this location.");

    LdaEstimate lda;
    std::string lda_mat_wxfilename = po.GetArg(1);

    // 读取lda 统计量  pdf-id 的特征、权重、权重累和、特征^2等.
    for (int32 i = 2; i <= po.NumArgs(); i++) {
      bool binary_in, add = true;
      Input ki(po.GetArg(i), &binary_in);
      lda.Read(ki.Stream(), binary_in, add);
    }

    Matrix<BaseFloat> lda_mat;
    Matrix<BaseFloat> full_lda_mat;

    // 估计 lda_mat, full_lda_mat 参数.
    lda.Estimate(opts, &lda_mat, &full_lda_mat);

    // 写入 lda_mat  
    WriteKaldiObject(lda_mat, lda_mat_wxfilename, binary);
    // 写入 full_lda_mat（一般不写入）
    if (full_matrix_wxfilename != "") {
      Output ko(full_matrix_wxfilename, binary);
      full_lda_mat.Write(ko.Stream(), binary);
    }
}


// lda.Estimate(opts, &lda_mat, &full_lda_mat)
// 根据统计量 计算得到实现lda的变换投影矩阵 lda_mat.
// 通过投影变换矩阵能够实现 类间差距最大, 同时类内差距最小.  sb(类间)/sw(类内) 到达最大值.

// lda_mat --- 0.mat ---> 是 (13*9 X 40) 将特征降维为40的变换矩阵.




// feats="$splicedfeats transform-feats $dir/0.mat ark:- ark:- |"

// splicedfeats --- 是 13x9 维度的特征
// splicedfeats 通过ark:- 利用transform-feats 通过0.mat变换 得到lda变换后的特征 --> ark:-| ===> feats
// feats 是经过上面lda变换矩阵 变换后的特征, 已经实现了lda降维.  feats的特征此时为 dim = 40
//  具体 看 上面 transform-feats 的分析过程








// ===================================
// 按照新的 feats 重新构建决策树.
// ===================================
// 此时 feats特征已经 与 原本gmm参数的特征不同, 原本GMM适用的特征
// 已经经过升维后 lda进行降维,
// 特征发生改变, 原本的音素聚类问题需要改变, 状态绑定决策树需要修改, GMM需要修改.

// 计算 状态-MFCC统计量  in final.mdl  MFCC   ali_gz  --->  状态-MFCC统计量
// acc-tree-stats $context_opts --ci-phones=$ciphonelist $alidir/final.mdl "$feats" 
//     "ark:gunzip -c $alidir/ali.JOB.gz|"  $dir/JOB.treeacc
// 将*.treeacc 总和一下 --> treeacc
// sum-tree-stats $dir/treeacc $dir/*.treeacc || exit 1;


// 重新应用一遍决策树构建过程.
//   # preparing questions, roots file... 聚类音素, 获得音素簇集合 -- 问题
//   cluster-phones $context_opts $dir/treeacc $lang/phones/sets.int \
//     $dir/questions.int 2> $dir/log/questions.log || exit 1;

//   # 将问题 转化为 qst形式..... 没啥用.
//   compile-questions $context_opts $lang/topo $dir/questions.int \
//     $dir/questions.qst 2>$dir/log/compile_questions.log || exit 1;

//   echo "$0: Building the tree"  构建决策树
//     build-tree $context_opts --verbose=1 --max-leaves=$numleaves \
//     --cluster-thresh=$cluster_thresh $dir/treeacc $lang/phones/roots.int \
//     $dir/questions.qst $lang/topo $dir/tree || exit 1;








// gmm-init-model 是根据状态绑定决策树的 pdf-id情况, 对每个pdf-id进行GMM参数估计
// $dir/tree 是 决策树状态绑定之后的决策树
//                                           状态绑定决策树  统计量  topo      ---> model
// gmm-init-model  --write-occs=$dir/1.occs  $dir/tree $dir/treeacc $lang/topo $dir/1.mdl 


//转换原对齐 trans-id --> pdf-id
// Converting alignments from $alidir to use current tree"
//     convert-ali $alidir/final.mdl $dir/1.mdl $dir/tree      "ark:gunzip -c $alidir/ali.JOB.gz|" "ark:|gzip -c >$dir/ali.JOB.gz"
// in:
//     old model
//     new model
//     状态绑定决策树
//     输入对齐pdf-id
// out:
//     输出对齐 pdf-id


int convert_ali(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
    const char *usage =
        "Convert alignments from one decision-tree/model to another\n"
        "Usage:  convert-ali  [options] <old-model> <new-model> <new-tree> "
        "<old-alignments-rspecifier> <new-alignments-wspecifier>\n"
        "e.g.: \n"
        " convert-ali old/final.mdl new/0.mdl new/tree ark:old/ali.1 ark:new/ali.1\n";

    int32 frame_subsampling_factor = 1;
    bool reorder = true;
    bool repeat_frames = false;

    std::string phone_map_rxfilename;

    std::string old_model_filename = po.GetArg(1);
    std::string new_model_filename = po.GetArg(2);
    std::string new_tree_filename = po.GetArg(3);
    std::string old_alignments_rspecifier = po.GetArg(4);
    std::string new_alignments_wspecifier = po.GetArg(5);

    std::vector<int32> phone_map;
    if (phone_map_rxfilename != "") {  // read phone map.
      ReadPhoneMap(phone_map_rxfilename,
                   &phone_map);
    }

    SequentialInt32VectorReader alignment_reader(old_alignments_rspecifier);
    Int32VectorWriter alignment_writer(new_alignments_wspecifier);

    TransitionModel old_trans_model;
    ReadKaldiObject(old_model_filename, &old_trans_model);

    TransitionModel new_trans_model;
    ReadKaldiObject(new_model_filename, &new_trans_model);

    // check 要求topo结构不能改变. topo 结构是描述一种 声学模型的结构, 必须保持一致
    if (!(old_trans_model.GetTopo() == new_trans_model.GetTopo()))
      KALDI_WARN << "Toplogies of models are not equal: "
                 << "conversion may not be correct or may fail.";


    ContextDependency new_ctx_dep;  // the tree.
    ReadKaldiObject(new_tree_filename, &new_ctx_dep);

    int num_success = 0, num_fail = 0;

    // foreach utt ali 
    for (; !alignment_reader.Done(); alignment_reader.Next()) {
      std::string key = alignment_reader.Key();
      
      const std::vector<int32> &old_alignment = alignment_reader.Value();
      std::vector<int32> new_alignment;

      //根据旧的对齐Vector <trans-id> 通过old_model 和 new_model new_tree 得到 新的对齐 Vector<trans-id>
      if (ConvertAlignment(old_trans_model,
                           new_trans_model,
                           new_ctx_dep,
                           old_alignment,
                           frame_subsampling_factor,  // 1
                           repeat_frames,             // false
                           reorder,                   // true
                           (phone_map_rxfilename != "" ? &phone_map : NULL), // NULL 
                           &new_alignment)) {
        alignment_writer.Write(key, new_alignment);
        num_success++;
      } else {
      }
    }
}



bool ConvertAlignment(const TransitionModel &old_trans_model,
                      const TransitionModel &new_trans_model,
                      const ContextDependencyInterface &new_ctx_dep,
                      const std::vector<int32> &old_alignment,
                      int32 subsample_factor,  // 1
                      bool repeat_frames,      // false
                      bool new_is_reordered,   // true
                      const std::vector<int32> *phone_map,  // NULL
                      std::vector<int32> *new_alignment) {
  if (!repeat_frames || subsample_factor == 1) {
    return ConvertAlignmentInternal(old_trans_model,
                                    new_trans_model,
                                    new_ctx_dep,
                                    old_alignment,
                                    subsample_factor - 1,
                                    subsample_factor,
                                    new_is_reordered,
                                    phone_map,
                                    new_alignment);
    // The value "subsample_factor - 1" for conversion_shift above ensures the
    // alignments have the same length as the output of 'subsample-feats'
  } else {
    std::vector<std::vector<int32> > shifted_alignments(subsample_factor);
    for (int32 conversion_shift = subsample_factor - 1;
         conversion_shift >= 0; conversion_shift--) {
      if (!ConvertAlignmentInternal(old_trans_model,
                                    new_trans_model,
                                    new_ctx_dep,
                                    old_alignment,
                                    conversion_shift,
                                    subsample_factor,
                                    new_is_reordered,
                                    phone_map,
                                    &shifted_alignments[conversion_shift]))
        return false;
    }
    KALDI_ASSERT(new_alignment != NULL);
    new_alignment->clear();
    new_alignment->reserve(old_alignment.size());
    int32 max_shifted_ali_length = (old_alignment.size() / subsample_factor)
        + (old_alignment.size() % subsample_factor);
    for (int32 i = 0; i < max_shifted_ali_length; i++)
      for (int32 conversion_shift = subsample_factor - 1;
           conversion_shift >= 0; conversion_shift--)
        if (i < static_cast<int32>(shifted_alignments[conversion_shift].size()))
          new_alignment->push_back(shifted_alignments[conversion_shift][i]);
  }
  KALDI_ASSERT(new_alignment->size() == old_alignment.size());
  return true;
}



static bool ConvertAlignmentInternal(const TransitionModel &old_trans_model,
                      const TransitionModel &new_trans_model,
                      const ContextDependencyInterface &new_ctx_dep,
                      const std::vector<int32> &old_alignment,
                      int32 conversion_shift,  // 0
                      int32 subsample_factor,  // 1
                      bool new_is_reordered,   // true
                      const std::vector<int32> *phone_map,  // NULL
                      std::vector<int32> *new_alignment) {

  bool old_is_reordered = IsReordered(old_trans_model, old_alignment);
  new_alignment->clear();
  new_alignment->reserve(old_alignment.size());

  // 根据oldmodel oldalignment --- > 按照phones 序列进行分组了的对齐结果
  std::vector<std::vector<int32> > old_split;  // split into phones.
  if (!SplitToPhones(old_trans_model, old_alignment, &old_split))
    return false;

  // phone_cnt
  int32 phone_sequence_length = old_split.size();
  std::vector<int32> mapped_phones(phone_sequence_length);
  for (size_t i = 0; i < phone_sequence_length; i++) {
    // 获得对应音素
    mapped_phones[i] = old_trans_model.TransitionIdToPhone(old_split[i][0]);
    if (phone_map != NULL) {  // Map the phone sequence.
      int32 sz = phone_map->size();
      if (mapped_phones[i] < 0 || mapped_phones[i] >= sz ||
          (*phone_map)[mapped_phones[i]] == -1)
        KALDI_ERR << "ConvertAlignment: could not map phone " << mapped_phones[i];
      mapped_phones[i] = (*phone_map)[mapped_phones[i]];
    }
  }

  // the sizes of each element of 'new_split' indicate the length of alignment
  // that we want for each phone in the new sequence.
  std::vector<std::vector<int32> > new_split(phone_sequence_length);
  if (subsample_factor == 1 &&
      old_trans_model.GetTopo() == new_trans_model.GetTopo()) {
    // we know the old phone lengths will be fine.
    for (size_t i = 0; i < phone_sequence_length; i++)
      new_split[i].resize(old_split[i].size());
  } else {
    // .. they may not be fine.

  }

  int32
      N = new_ctx_dep.ContextWidth(),
      P = new_ctx_dep.CentralPosition();

  // by starting at -N and going to phone_sequence_length + N,
  // we're being generous and not bothering to work out the exact array bounds.
  // foreach 三音素窗中的每个音素
  for (int32 win_start = -N;
       win_start < static_cast<int32>(phone_sequence_length + N);
       win_start++) {  // start of a context window.

    // central_pos 中心音素 --实际音素  cetral_pos是逐个增加的
    int32 central_pos = win_start + P;
    if (static_cast<size_t>(central_pos) < phone_sequence_length) {
      // i.e. if (central_pos >= 0 && central_pos < phone_sequence_length)

      // 构建一个三音素窗,利用三音素结构 通过决策树找到对应的新的pdf-id
      std::vector<int32> new_phone_window(N, 0);
      for (int32 offset = 0; offset < N; offset++)
        if (static_cast<size_t>(win_start+offset) < phone_sequence_length)
          new_phone_window[offset] = mapped_phones[win_start+offset];

      // old_alignment_for_phone -- 保存centralphone内 旧的trans-id
      const std::vector<int32> &old_alignment_for_phone = old_split[central_pos];
      // 保存新的 音素内trans-id
      std::vector<int32> &new_alignment_for_phone = new_split[central_pos];

      // 每个音素的 根据旧的 对齐 transi-id 获得 对齐 trans-id
      ConvertAlignmentForPhone(old_trans_model, new_trans_model, new_ctx_dep,
                               old_alignment_for_phone, new_phone_window,
                               old_is_reordered, new_is_reordered,
                               &new_alignment_for_phone);
      // 
      new_alignment->insert(new_alignment->end(),
                            new_alignment_for_phone.begin(),
                            new_alignment_for_phone.end());
    }
  }
  KALDI_ASSERT(new_alignment->size() ==
               (old_alignment.size() + conversion_shift)/subsample_factor);
  return true;
}

// 获得按照内部音素 为边界划分的对齐
// in:
// trans-mdl   转移模型, 保存的 state  pdf-id phone 之间关系
// aligenment  原本一句utt的对齐pdf-id
// out:
// split_output  通过音素边界状态 IsFinal 的判断, 得到将对齐按照phone进行设置组的 Vector<phoneVecotr<pdf-id, pdf-id>>
static bool SplitToPhonesInternal(const TransitionModel &trans_model,
                                  const std::vector<int32> &alignment,
                                  bool reordered,
                                  std::vector<std::vector<int32> > *split_output) {
  if (alignment.empty()) return true;  // nothing to split.
  std::vector<size_t> end_points;  // points at which phones end [in an
  // stl iterator sense, i.e. actually one past the last transition-id within
  // each phone]..

  bool was_ok = true;
  // foreach trans-id, 得到每个phone的终止帧 frame
  // end_points 保存每个phone的音素最后一帧
  for (size_t i = 0; i < alignment.size(); i++) {
    int32 trans_id = alignment[i];
    if (trans_model.IsFinal(trans_id)) {  // is final-prob
      if (!reordered) end_points.push_back(i+1);
      else {  // reordered.
        while (i+1 < alignment.size() &&
              trans_model.IsSelfLoop(alignment[i+1])) {
          KALDI_ASSERT(trans_model.TransitionIdToTransitionState(alignment[i]) ==
                 trans_model.TransitionIdToTransitionState(alignment[i+1]));
          i++;
        }
        end_points.push_back(i+1);
      }
    } else if (i+1 == alignment.size()) {
      // need to have an end-point at the actual end.
      // but this is an error- should have been detected already.
      was_ok = false;
      end_points.push_back(i+1);
    } else {
      int32 this_state = trans_model.TransitionIdToTransitionState(alignment[i]),
          next_state = trans_model.TransitionIdToTransitionState(alignment[i+1]);
      if (this_state == next_state) continue;  // optimization.
      int32 this_phone = trans_model.TransitionStateToPhone(this_state),
          next_phone = trans_model.TransitionStateToPhone(next_state);
      if (this_phone != next_phone) {
        // The phone changed, but this is an error-- we should have detected this via the
        // IsFinal check.
        was_ok = false;
        end_points.push_back(i+1);
      }
    }
  }


  //cur_point 保存当前待识别音素的 起始帧, end_points[i] 保存当前phone的终止帧
  size_t cur_point = 0;
  // 根据phone 终止帧 序列 获得phone
  // foreach phoneframes!!!!! 
  for (size_t i = 0; i < end_points.size(); i++) {
    split_output->push_back(std::vector<int32>());
    // The next if-statement checks if the initial trans-id at the current end
    // point is the initial-state of the current phone if that initial-state
    // is emitting (a cursory check that the alignment is plausible).
    int32 trans_state =
      trans_model.TransitionIdToTransitionState(alignment[cur_point]);
    int32 phone = trans_model.TransitionStateToPhone(trans_state);
    int32 forward_pdf_class = trans_model.GetTopo().TopologyForPhone(phone)[0].forward_pdf_class;
    
    if (forward_pdf_class != kNoPdf)  // initial-state of the current phone is emitting
      if (trans_model.TransitionStateToHmmState(trans_state) != 0)
        was_ok = false;
    // phone frames, 按音素划分对齐的trans-id
    for (size_t j = cur_point; j < end_points[i]; j++)
      split_output->back().push_back(alignment[j]);
    
    cur_point = end_points[i];
  }
  return was_ok;
}


static inline void ConvertAlignmentForPhone(
    const TransitionModel &old_trans_model,
    const TransitionModel &new_trans_model,
    const ContextDependencyInterface &new_ctx_dep,
    const std::vector<int32> &old_phone_alignment,
    const std::vector<int32> &new_phone_window,
    bool old_is_reordered,
    bool new_is_reordered,
    std::vector<int32> *new_phone_alignment) {
  
  int32 alignment_size = old_phone_alignment.size();
  static bool warned_topology = false;
  
  int32 P = new_ctx_dep.CentralPosition(),
      old_central_phone = old_trans_model.TransitionIdToPhone(old_phone_alignment[0]),
      new_central_phone = new_phone_window[P];
  
  const HmmTopology &old_topo = old_trans_model.GetTopo(),
      &new_topo = new_trans_model.GetTopo();



  int32 new_num_pdf_classes = new_topo.NumPdfClasses(new_central_phone);
  std::vector<int32> pdf_ids(new_num_pdf_classes);  // Indexed by pdf-class

  // 根据三音素窗 以及 状态绑定决策树 计算实际音素的不同 pdf-class 的 new pdf-id.
  for (int32 pdf_class = 0; pdf_class < new_num_pdf_classes; pdf_class++) {
    if (!new_ctx_dep.Compute(new_phone_window, pdf_class,
                             &(pdf_ids[pdf_class]))) {
      std::ostringstream ss;
      WriteIntegerVector(ss, false, new_phone_window);
      KALDI_ERR << "tree did not succeed in converting phone window "
                << ss.str();
    }
  }

  // the topologies and lengths match -> we can directly transfer the alignment.
  // aligenment 中保存的是 trans-id. 通过trans-id 得到对应的 pdf-class
  for (int32 j = 0; j < alignment_size; j++) {
    
    int32 old_tid = old_phone_alignment[j],
        old_tstate = old_trans_model.TransitionIdToTransitionState(old_tid);
    // 内部通过topo结果获得 pdf-class
    int32 forward_pdf_class =
        old_trans_model.TransitionStateToForwardPdfClass(old_tstate),
        self_loop_pdf_class =
        old_trans_model.TransitionStateToSelfLoopPdfClass(old_tstate);
    int32 hmm_state = old_trans_model.TransitionIdToHmmState(old_tid);
    int32 trans_idx = old_trans_model.TransitionIdToTransitionIndex(old_tid);

    // 根据pdf-class 获得新的pdf-id
    int32 new_forward_pdf = pdf_ids[forward_pdf_class];
    int32 new_self_loop_pdf = pdf_ids[self_loop_pdf_class];

    // 根据phone, hmm-state, pdf-id 通过超照对应的tuple 获得新的trans-state
    int32 new_trans_state =
        new_trans_model.TupleToTransitionState(new_central_phone, hmm_state,
                                               new_forward_pdf, new_self_loop_pdf);
    
    // 一个确定的trans-state 可能会具有多个trans-index 进而得到不同的trans-id.
    // 完成一个音素的所有状态的对齐结果 --- new_aligenment  verctor<trans-id>.
    int32 new_tid =
        new_trans_model.PairToTransitionId(new_trans_state, trans_idx);
    
    (*new_phone_alignment)[j] = new_tid;
  }

  if (new_is_reordered != old_is_reordered)
    ChangeReorderingOfAlignment(new_trans_model, new_phone_alignment);
}






//为每个utt  构建 phone -> word fst图
// in:
// 决策树
// trans-model
// L.fst
// utt word标注
// :
// 根据 状态绑定决策树 转移模型 L.fst 以及utt的word标注, 找到对应每个 utt的phone为基本状态的fst图.
// out:
// utt 解码fst

// Compiling graphs of transcripts
// compile-train-graphs --read-disambig-syms=$lang/phones/disambig.int $dir/tree $dir/1.mdl  $lang/L.fst \
//      "ark:utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt < $data/split$nj/JOB/text |" \
//       "ark:|gzip -c >$dir/fsts.JOB.gz"



// while [ $x -lt $num_iters ]; do
//   echo Training pass $x
//       // in:
//       // 转移模型
//       // utt的fst图
//       // feats特征
//       // out:
//       // 对齐的trans-id 序列
//   gmm-align-compiled $scale_opts --beam=$beam --retry-beam=$retry_beam --careful=$careful "$mdl" \
//     "ark:gunzip -c $dir/fsts.JOB.gz|" "$feats" \
//     "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1;

//   // 2 根据trans-id对齐信息, 转化为probable<pdf-id, prob>对齐信息， 然后增加silence权重.
//    echo "$0: Estimating MLLT"
   
//    ali-to-post "ark:gunzip -c $dir/ali.JOB.gz|" ark:- \| \
//    weight-silence-post 0.0 $silphonelist $dir/$x.mdl ark:- ark:- \| \
//    gmm-acc-mllt --rand-prune=$randprune  $dir/$x.mdl "$feats" ark:- $dir/$x.JOB.macc || exit 1;

//    est-mllt $dir/$x.mat.new $dir/$x.*.macc 2> $dir/log/mupdate.$x.log || exit 1;


//    计算新的DiagGmm 的所有均值 以及 Gconst
//    gmm-transform-means  $dir/$x.mat.new $dir/$x.mdl $dir/$x.mdl

//    组合LDA($cur_lda_iter.mat) 和 MLLT($x.mat.new)矩阵 得到 $x.mat
//    compose-transforms --print-args=false $dir/$x.mat.new $dir/$cur_lda_iter.mat $dir/$x.mat

//    多次迭代的 mllt mat矩阵 $x.mat. 每次计算了mllt矩阵之后 都将MFCC重新变换, 每次都是在新的MFCC上进行的MLLT, 这样每次
//    的MLLT估计都只需要通过初始化一个unitMat 进行更新一个MLLT矩阵,然后 compose以前的matrix即可.
//    feats="$splicedfeats transform-feats $dir/$x.mat ark:- ark:- |"
//    cur_lda_iter=$x


//     // 重估 gmm 参数.
//     gmm-acc-stats-ali  $dir/$x.mdl "$feats" \
//       "ark,s,cs:gunzip -c $dir/ali.JOB.gz|" $dir/$x.JOB.acc || exit 1;

//     gmm-est --write-occs=$dir/$[$x+1].occs --mix-up=$numgauss --power=$power \
//       $dir/$x.mdl "gmm-sum-accs - $dir/$x.*.acc |" $dir/$[$x+1].mdl || exit 1;
  
//   [ $x -le $max_iter_inc ] && numgauss=$[$numgauss+$incgauss];
//   x=$[$x+1];
// done



// 将对齐trans-id 转化为 多个可能的<trans-id, prob> 这里实际上只有一个可能 <trans-id, prob=1.0> .
// ali-to-post "ark:gunzip -c $dir/ali.JOB.gz|" ark:- \| \
// weight-silence-post 0.0 $silphonelist $dir/$x.mdl ark:- ark:- \|        \ 
// gmm-acc-mllt --rand-prune=$randprune  $dir/$x.mdl "$feats" ark:- $dir/$x.JOB.macc || exit 1;

int gmm_acc_mllt(int argc, char *argv[]) {
  using namespace kaldi;
  const char *usage =
      "Accumulate MLLT (global STC) statistics\n"
      "Usage:  gmm-acc-mllt [options] <model-in> <feature-rspecifier> <posteriors-rspecifier> <stats-out>\n"
      "e.g.: \n"
      " gmm-acc-mllt 1.mdl scp:train.scp ark:1.post 1.macc\n";

  ParseOptions po(usage);
  bool binary = true;
  BaseFloat rand_prune = 0.25;
  po.Register("binary", &binary, "Write output in binary mode");
  po.Register("rand-prune", &rand_prune, "Randomized pruning parameter to speed up "
              "accumulation (larger -> more pruning.  May exceed one).");
  po.Read(argc, argv);


  std::string
      model_filename = po.GetArg(1),
      feature_rspecifier = po.GetArg(2),
      posteriors_rspecifier = po.GetArg(3),
      accs_wxfilename = po.GetArg(4);

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

  // 内部通过Vecotr<SpMatrix> 保存多个对称矩阵, 保存的是协方差么？
  // amm_gmm.Dim() 保存的是 内部的DiaGmm 的Dim() 就是特征维度.
  MlltAccs mllt_accs(am_gmm.Dim(), rand_prune);

  double tot_like = 0.0;
  double tot_t = 0.0;

  SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
  RandomAccessPosteriorReader posteriors_reader(posteriors_rspecifier);

  int32 num_done = 0, num_no_posterior = 0, num_other_error = 0;
  // foreach utt
  for (; !feature_reader.Done(); feature_reader.Next()) {
    std::string key = feature_reader.Key();
    if (!posteriors_reader.HasKey(key)) {
      num_no_posterior++;
    } else {
      const Matrix<BaseFloat> &mat = feature_reader.Value();
      const Posterior &posterior = posteriors_reader.Value(key);

      num_done++;
      BaseFloat tot_like_this_file = 0.0, tot_weight = 0.0;

      Posterior pdf_posterior;
      ConvertPosteriorToPdfs(trans_model, posterior, &pdf_posterior);

      // foreach frame i 
      for (size_t i = 0; i < posterior.size(); i++) {
        // foreach probable pdf-id--- DiaGmm. 训练时应该都是只对应一个DiaGmm因为有标注
        for (size_t j = 0; j < pdf_posterior[i].size(); j++) {
          int32 pdf_id = pdf_posterior[i][j].first;
          BaseFloat weight = pdf_posterior[i][j].second;

          tot_like_this_file += mllt_accs.AccumulateFromGmm(am_gmm.GetPdf(pdf_id),
                                                            mat.Row(i),
                                                            weight) * weight;
          tot_weight += weight;
        }
      }
    }

  }
  // 保存所有统计量
  WriteKaldiObject(mllt_accs, accs_wxfilename, binary);
}


// 根据transmodel , 将可能的tid, prob 转化为 可能的 pdf-id, prob.
void ConvertPosteriorToPdfs(const TransitionModel &tmodel,
                            const Posterior &post_in,
                            Posterior *post_out) {
  post_out->clear();
  post_out->resize(post_in.size());

  // foreach frame
  for (size_t i = 0; i < post_out->size(); i++) {
    unordered_map<int32, BaseFloat> pdf_to_post;
    // foreach probable t-id
    for (size_t j = 0; j < post_in[i].size(); j++) {
      
      int32 tid = post_in[i][j].first,
          pdf_id = tmodel.TransitionIdToPdf(tid);
      BaseFloat post = post_in[i][j].second;
      if (pdf_to_post.count(pdf_id) == 0)
        pdf_to_post[pdf_id] = post;
      else
        pdf_to_post[pdf_id] += post;
    }
    // 将当前frame i的所有可能pdf-id保存到 post_out[i].
    (*post_out)[i].reserve(pdf_to_post.size());  // 修改某个vector大小
    for (unordered_map<int32, BaseFloat>::const_iterator iter =
             pdf_to_post.begin(); iter != pdf_to_post.end(); ++iter) {
      if (iter->second != 0.0)
        (*post_out)[i].push_back(
            std::make_pair(iter->first, iter->second));
    }
  }
}



// 计算了什么东西反正是 类似 更新参数用的 统计量.  与 各个GMM参数有关、与MFCC有关,
// 计算MLLT转换矩阵 用的统计量.
void MlltAccs::AccumulateFromPosteriors(const DiagGmm &gmm,
                                        const VectorBase<BaseFloat> &data,
                                        const VectorBase<BaseFloat> &posteriors) {
  // 确保 数据维度与高斯参数维度相同、后验概率维度 与 gmm高斯分量数目相同
  KALDI_ASSERT(data.Dim() == gmm.Dim());
  KALDI_ASSERT(data.Dim() == Dim());
  KALDI_ASSERT(posteriors.Dim() == gmm.NumGauss());

  // Matrix<>   --- row-高斯分量数   col MFCC维度 参数
  const Matrix<BaseFloat> &means_invvars = gmm.means_invvars();
  const Matrix<BaseFloat> &inv_vars = gmm.inv_vars();
  
  Vector<BaseFloat> mean(data.Dim());
  SpMatrix<double> tmp(data.Dim());  // 对称矩阵， 虽然是矩阵样子 内部保存的数据实际上是一个数组.
  Vector<double> offset_dbl(data.Dim());
  
  double this_beta_ = 0.0;

  // foreach mixcomp..
  for (int32 i = 0; i < posteriors.Dim(); i++) {  

    BaseFloat posterior = RandPrune(posteriors(i), rand_prune_);
    if (posterior == 0.0) continue;
    
    SubVector<BaseFloat> mean_invvar(means_invvars, i);
    SubVector<BaseFloat> inv_var(inv_vars, i);
    
    mean.AddVecDivVec(1.0, mean_invvar, inv_var, 0.0);  // get mean.

    mean.AddVec(-1.0, data);  // get offset
    
    offset_dbl.CopyFromVec(mean);  // make it double.
    
    tmp.SetZero();
    tmp.AddVec2(1.0, offset_dbl);
    for (int32 j = 0; j < data.Dim(); j++)
      G_[j].AddSp(inv_var(j)*posterior, tmp);
    this_beta_ += posterior;
  }
  beta_ += this_beta_;
  Vector<double> data_dbl(data);
}

BaseFloat MlltAccs::AccumulateFromGmm(const DiagGmm &gmm,  // frame probable pdf-id gmm 参数
                                      const VectorBase<BaseFloat> &data,  // frame mfcc
                                      BaseFloat weight) {  // e.g. weight = 1.0  pdf-id, prob
  // 多个高斯分量
  Vector<BaseFloat> posteriors(gmm.NumGauss());
  BaseFloat ans = gmm.ComponentPosteriors(data, &posteriors);
  posteriors.Scale(weight);
  AccumulateFromPosteriors(gmm, data, posteriors);
  return ans;
}

// Gets likelihood of data given this. Also provides per-Gaussian posteriors.
// 获得每个高斯分量的 后验概率？？
BaseFloat DiagGmm::ComponentPosteriors(const VectorBase<BaseFloat> &data,
                                       Vector<BaseFloat> *posterior) const {

  Vector<BaseFloat> loglikes;
  LogLikelihoods(data, &loglikes);
  BaseFloat log_sum = loglikes.ApplySoftMax();
  posterior->CopyFromVec(loglikes);
  return log_sum;
}

// 计算 某个MFCC属于 DiagGmm中某个高斯分量的概率？
// 应该是的, 这里means_invvars_都是多个高斯分量的 vector
void DiagGmm::LogLikelihoods(const VectorBase<BaseFloat> &data,
                             Vector<BaseFloat> *loglikes) const {
  // gconst_  是一个计算用的部件
  loglikes->Resize(gconsts_.Dim(), kUndefined);
  loglikes->CopyFromVec(gconsts_);

  // x^2
  Vector<BaseFloat> data_sq(data);
  data_sq.ApplyPow(2.0);

  // loglikes +=  means * inv(vars) * data.
  loglikes->AddMatVec(1.0, means_invvars_, kNoTrans, data, 1.0);
  // loglikes += -0.5 * inv(vars) * data_sq.
  loglikes->AddMatVec(-0.5, inv_vars_, kNoTrans, data_sq, 1.0);
}



// est-mllt $dir/$x.mat.new $dir/$x.*.macc
// in:
// $x.*.macc 刚刚 gmm-acc-mllt 统计得到的统计量
// out:
// $x.mat.new 计算得到的MFCC转移矩阵, 通过在特征域进行mat.new变换，实现MLLT将Diag参数 扩展一下. 提高表达能力.
void est_mllt(){
  // 获得update mllt统计量
  // 根据统计量的维度 初始化一个 matrix
  Matrix<BaseFloat> mat(mllt_accs.Dim(), mllt_accs.Dim());
  mat.SetUnit();
    
  BaseFloat objf_impr, count;
  // 根据统计量计算mllt 转换矩阵, 病
  mllt_accs.Update(&mat, &objf_impr, &count);

  // 写入到 $x.mat.new.
  writeToFile();
}

// static version of the Update function.
// 更新算法的计算过程.
void MlltAccs::Update(double beta,
                      const std::vector<SpMatrix<double> > &G,
                      MatrixBase<BaseFloat> *M_ptr,
                      BaseFloat *objf_impr_out,
                      BaseFloat *count_out) {
  int32 dim = G.size();
  KALDI_ASSERT(dim != 0 && M_ptr != NULL
               && M_ptr->NumRows() == dim
               && M_ptr->NumCols() == dim);
  if (beta < 10*dim) {  // not really enough data to estimate.
    // don't bother with min-count parameter etc., as MLLT is typically
    // global.
    if (beta > 2*dim)
      KALDI_WARN << "Mllt:Update, very small count " << beta;
    else
      KALDI_WARN << "Mllt:Update, insufficient count " << beta;
  }
  int32 num_iters = 200;  // may later make this an option.
  Matrix<double> M(dim, dim), Minv(dim, dim);
  M.CopyFromMat(*M_ptr);
  std::vector<SpMatrix<double> > Ginv(dim);
  for (int32 i = 0; i < dim;  i++) {
    Ginv[i].Resize(dim);
    Ginv[i].CopyFromSp(G[i]);
    Ginv[i].Invert();
  }

  double tot_objf_impr = 0.0;
  for (int32 p = 0; p < num_iters; p++) {
    for (int32 i = 0; i < dim; i++) {  // for each row
      SubVector<double> row(M, i);
      // work out cofactor (actually cofactor times a constant which
      // doesn't affect anything):
      Minv.CopyFromMat(M);
      Minv.Invert();
      Minv.Transpose();
      SubVector<double> cofactor(Minv, i);
      // Objf is: beta log(|row . cofactor|) -0.5 row^T G[i] row
      // optimized by (c.f. Mark Gales's techreport "semitied covariance matrices
      // for hidden markov models, eq.  (22)),
      // row = G_i^{-1} cofactor sqrt(beta / cofactor^T G_i^{-1} cofactor). (1)
      // here, "row" and "cofactor" are considered as column vectors.
      double objf_before = beta * Log(std::abs(VecVec(row, cofactor)))
          -0.5 * VecSpVec(row, G[i], row);
      // do eq. (1) above:
      row.AddSpVec(std::sqrt(beta / VecSpVec(cofactor, Ginv[i], cofactor)),
                   Ginv[i], cofactor, 0.0);
      double objf_after = beta * Log(std::abs(VecVec(row, cofactor)))
          -0.5 * VecSpVec(row, G[i], row);
      if (objf_after < objf_before - fabs(objf_before)*0.00001)
        KALDI_ERR << "Objective decrease in MLLT update.";
      tot_objf_impr += objf_after - objf_before;
    }
    if (p < 10 || p % 10 == 0)
      KALDI_LOG << "MLLT objective improvement per frame by " << p
                << "'th iteration is " << (tot_objf_impr/beta) << " per frame "
                << "over " << beta << " frames.";
  }
  if (objf_impr_out)
    *objf_impr_out = tot_objf_impr;
  if (count_out)
    *count_out = beta;
  M_ptr->CopyFromMat(M);
}




// gmm-transform-mean
// in:
// 转换矩阵
// old mdl  -- trans-model + GMM参数
// out
// new mdl  -- trans-model + 修改均值以及Gconst 的GMM参数
int gmm_transform_means(int argc, char *argv[]) {

    const char *usage =
        "Transform GMM means with linear or affine transform\n"
        "Usage:  gmm-transform-means <transform-matrix> <model-in> <model-out>\n"
        "e.g.: gmm-transform-means 2.mat 2.mdl 3.mdl\n";


    std::string
        mat_rxfilename = po.GetArg(1),
        model_in_rxfilename = po.GetArg(2),
        model_out_wxfilename = po.GetArg(3);

    Matrix<BaseFloat> mat;
    ReadKaldiObject(mat_rxfilename, &mat);

    AmDiagGmm am_gmm;
    TransitionModel trans_model;
    {
      bool binary_read;
      Input ki(model_in_rxfilename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_gmm.Read(ki.Stream(), binary_read);
    }
    
    // 实际上是MFCC 维度
    int32 dim = am_gmm.Dim();

    // foreach DiagGmm
    for (int32 i = 0; i < am_gmm.NumPdfs(); i++) {
      DiagGmm &gmm = am_gmm.GetPdf(i);

      // 多个分量的 means
      Matrix<BaseFloat> means;
      gmm.GetMeans(&means);
      Matrix<BaseFloat> new_means(means.NumRows(), means.NumCols());

      if (mat.NumCols() == dim) {  // linear case
        // Right-multiply means by mat^T (equivalent to left-multiplying each
        // row by mat).
        new_means.AddMatMat(1.0, means, kNoTrans, mat, kTrans, 0.0);
      } else { // affine case
        Matrix<BaseFloat> means_ext(means.NumRows(), means.NumCols()+1);
        means_ext.Set(1.0);  // set all elems to 1.0
        SubMatrix<BaseFloat> means_part(means_ext, 0, means.NumRows(),
                                        0, means.NumCols());
        means_part.CopyFromMat(means);  // copy old part...
        new_means.AddMatMat(1.0, means_ext, kNoTrans, mat, kTrans, 0.0);
      }

      // 使用新的均值
      gmm.SetMeans(new_means);
      // 计算Gconst
      gmm.ComputeGconsts();
    }

    // 写入文件
    {
      Output ko(model_out_wxfilename, binary);
      trans_model.Write(ko.Stream(), binary);
      am_gmm.Write(ko.Stream(), binary);
    }
}


// compose-transforms --print-args=false $dir/$x.mat.new $dir/$cur_lda_iter.mat $dir/$x.mat
// 合并 上轮变换矩阵 和 本次更新的变换矩阵.
int compose_transforms(int argc, char *argv[]) {

  const char *usage =
      "Compose (affine or linear) feature transforms\n"
      "Usage: compose-transforms [options] (<transform-A-rspecifier>|<transform-A-rxfilename>) "
      "(<transform-B-rspecifier>|<transform-B-rxfilename>) (<transform-out-wspecifier>|<transform-out-wxfilename>)\n"
      " Note: it does matrix multiplication (A B) so B is the transform that gets applied\n"
      "  to the features first.  If b-is-affine = true, then assume last column of b corresponds to offset\n"
      " e.g.: compose-transforms 1.mat 2.mat 3.mat\n"
      "   compose-transforms 1.mat ark:2.trans ark:3.trans\n"
      "   compose-transforms ark:1.trans ark:2.trans ark:3.trans\n"
      " See also: transform-feats, transform-vec, extend-transform-dim, est-lda, est-pca\n";

  bool b_is_affine = false;
  bool binary = true;
  std::string utt2spk_rspecifier;
  ParseOptions po(usage);


  // new mllt mat
  std::string transform_a_fn = po.GetArg(1);
  // old lda+mllt mat
  std::string transform_b_fn = po.GetArg(2);
  // ----> out new lda+mllt mat
  std::string transform_c_fn = po.GetArg(3);

  // all these "fn"'s are either rspecifiers or filenames.

  bool a_is_rspecifier =
      (ClassifyRspecifier(transform_a_fn, NULL, NULL)
       != kNoRspecifier),
      b_is_rspecifier =
      (ClassifyRspecifier(transform_b_fn, NULL, NULL)
       != kNoRspecifier),
      c_is_wspecifier =
      (ClassifyWspecifier(transform_c_fn, NULL, NULL, NULL)
       != kNoWspecifier);


   
  if (a_is_rspecifier || b_is_rspecifier) {
    BaseFloatMatrixWriter c_writer(transform_c_fn);
    if (a_is_rspecifier) {
      SequentialBaseFloatMatrixReader a_reader(transform_a_fn);
      if (b_is_rspecifier) {  // both are rspecifiers.
        RandomAccessBaseFloatMatrixReader b_reader(transform_b_fn);
        
        for (;!a_reader.Done(); a_reader.Next()) {
          if (utt2spk_rspecifier != "") {  // assume a is per-utt, b is per-spk.

          } else {  // Normal case: either both per-utterance or both per-speaker.
            if (!b_reader.HasKey(a_reader.Key())) {
              KALDI_WARN << "Second table does not have key " << a_reader.Key();
            } else {
              Matrix<BaseFloat> c;
              if (!ComposeTransforms(a_reader.Value(), b_reader.Value(a_reader.Key()),
                                     b_is_affine, &c))
                continue;  // warning will have been printed already.
              c_writer.Write(a_reader.Key(), c);
            }
          }
        }
      } else {  // a is rspecifier,  b is rxfilename
        Matrix<BaseFloat> b;
        ReadKaldiObject(transform_b_fn, &b);
        for (;!a_reader.Done(); a_reader.Next()) {
          Matrix<BaseFloat> c;
          if (!ComposeTransforms(a_reader.Value(), b,
                                 b_is_affine, &c))
            continue;  // warning will have been printed already.
          c_writer.Write(a_reader.Key(), c);
        }
      }
    } else {
      Matrix<BaseFloat> a;
      ReadKaldiObject(transform_a_fn, &a);
      SequentialBaseFloatMatrixReader b_reader(transform_b_fn);
      for (; !b_reader.Done(); b_reader.Next()) {
        Matrix<BaseFloat> c;
        if (!ComposeTransforms(a, b_reader.Value(),
                               b_is_affine, &c))
          continue;  // warning will have been printed already.
        c_writer.Write(b_reader.Key(), c);
      }
    }
  } else {  // all are just {rx, wx}filenames.
    Matrix<BaseFloat> a;
    ReadKaldiObject(transform_a_fn, &a);
    Matrix<BaseFloat> b;
    ReadKaldiObject(transform_b_fn, &b);
    Matrix<BaseFloat> c;
    
    if (!ComposeTransforms(a, b, b_is_affine, &c)) exit (1);

    WriteKaldiObject(c, transform_c_fn, binary);
  }
  return 0;
}


// 组合矩阵, 内部实际上就是 矩阵点乘。
bool ComposeTransforms(const Matrix<BaseFloat> &a, const Matrix<BaseFloat> &b,
                       bool b_is_affine,
                       Matrix<BaseFloat> *c) {

  // 实际上就是 矩阵点乘.
  if (a.NumCols() == b.NumRows()) {
    c->Resize(a.NumRows(), b.NumCols());
    c->AddMatMat(1.0, a, kNoTrans, b, kNoTrans, 0.0);  // c = a * b.
    return true;
  }
  
}




void question(){
  // 1 pdf-id  and trans-id ??? 区别
  // pdf-id 是绑定了的 trans-state（phone, HMM-state, pdf-id）, 而 trans-id， 是所有trans-state的所有可能输出转移的id
  // trans-id 才是对齐、识别过程中使用的基本单位. 因为trans-id 包含的信息量够多.
  
  // 2 DiagGmm 保存的是多个高斯分量 的模型
  // LogLikelihoods() 函数 计算的是 高斯分量后验概率, 是在EM算法中用来更新参数的统计量.


  
}






// ///////////////////////////////////////////////////////////////////////////
// 执行了 train_lda_mllt.sh之后会进行 align_si.sh 
// #lda_mllt_ali
// steps/align_si.sh  --nj $n --cmd "$train_cmd" --use-graphs true data/mfcc/train data/lang exp/tri2b exp/tri2b_ali || exit 1;


# Begin configuration section.

nj=4
cmd=run.pl
use_graphs=false
# Begin configuration.
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
beam=10
retry_beam=40
careful=false

boost_silence=1.0 # Factor by which to boost silence during alignment.

# End configuration options.

echo "$0 $@"  # Print the command line for logging


if [ $# != 4 ]; then
   echo "usage: steps/align_si.sh <data-dir> <lang-dir> <src-dir> <align-dir>"
   echo "e.g.:  steps/align_si.sh data/train data/lang exp/tri1 exp/tri1_ali"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --use-graphs true                                # use graphs in src-dir"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi


data=$1  // data/mfcc/train/
lang=$2  // data/lang
srcdir=$3  // exp/tri2b
dir=$4     // exp/tri2_ali


for f in $data/text $lang/oov.int $srcdir/tree $srcdir/final.mdl; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1;
done


cp $lang/phones.txt $dir || exit 1;
cp $srcdir/{tree,final.mdl} $dir || exit 1;
cp $srcdir/final.occs $dir;



if [ -f $srcdir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "$0: feature type is $feat_type"

// use lda
case $feat_type in
  delta) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas $delta_opts ark:- ark:- |";;

  lda) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |"
                                          
    cp $srcdir/final.mat $srcdir/full.mat $dir
   ;;
  *) echo "$0: invalid feature type $feat_type" && exit 1;
esac




echo "$0: aligning data in $data using model from $srcdir, putting alignments in $dir"

mdl="gmm-boost-silence --boost=$boost_silence `cat $lang/phones/optional_silence.csl` $dir/final.mdl - |"

if $use_graphs; then
  [ $nj != "`cat $srcdir/num_jobs`" ] && echo "$0: mismatch in num-jobs" && exit 1;
  [ ! -f $srcdir/fsts.1.gz ] && echo "$0: no such file $srcdir/fsts.1.gz" && exit 1;

  $cmd JOB=1:$nj $dir/log/align.JOB.log \
    gmm-align-compiled $scale_opts --beam=$beam --retry-beam=$retry_beam --careful=$careful "$mdl" \
      "ark:gunzip -c $srcdir/fsts.JOB.gz|" "$feats" "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1;

fi

steps/diagnostic/analyze_alignments.sh --cmd "$cmd" $lang $dir

echo "$0: done aligning data."
