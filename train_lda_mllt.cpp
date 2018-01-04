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

// splice-feats 扩展特征向量, 将原本每帧D维向量, 变为 (1+8) = 9xD维.  eg frame=4 (0,1,2,3 + 4 + 5,6,7,8)
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
// used in the bottom
int main(int argc, char *argv[]) {
  try {
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
    po.Register("utt2spk", &utt2spk_rspecifier, "rspecifier for utterance to speaker map");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string transform_rspecifier_or_rxfilename = po.GetArg(1);
    std::string feat_rspecifier = po.GetArg(2);
    std::string feat_wspecifier = po.GetArg(3);

    SequentialBaseFloatMatrixReader feat_reader(feat_rspecifier);
    BaseFloatMatrixWriter feat_writer(feat_wspecifier);

    RandomAccessBaseFloatMatrixReaderMapped transform_reader;
    bool use_global_transform;
    Matrix<BaseFloat> global_transform;

    // read to global_transform ???
    if (ClassifyRspecifier(transform_rspecifier_or_rxfilename, NULL, NULL)
       == kNoRspecifier) {
      // not an rspecifier -> interpret as rxfilename....
      use_global_transform = true;
      ReadKaldiObject(transform_rspecifier_or_rxfilename, &global_transform);
    } else {  // an rspecifier -> not a global transform.
      use_global_transform = false;
      if (!transform_reader.Open(transform_rspecifier_or_rxfilename,
                                 utt2spk_rspecifier)) {
        KALDI_ERR << "Problem opening transforms with rspecifier "
                  << '"' << transform_rspecifier_or_rxfilename << '"'
                  << " and utt2spk rspecifier "
                  << '"' << utt2spk_rspecifier << '"';
      }
    }

    enum { Unknown, Logdet, PseudoLogdet, DimIncrease };
    int32 logdet_type = Unknown;
    double tot_t = 0.0, tot_logdet = 0.0;  // to compute average logdet weighted by time...
    int32 num_done = 0, num_error = 0;
    BaseFloat cached_logdet = -1;
    
    for (;!feat_reader.Done(); feat_reader.Next()) {
      std::string utt = feat_reader.Key();
      const Matrix<BaseFloat> &feat(feat_reader.Value());

      if (!use_global_transform && !transform_reader.HasKey(utt)) {
        KALDI_WARN << "No fMLLR transform available for utterance "
                   << utt << ", producing no output for this utterance";
        num_error++;
        continue;
      }
      const Matrix<BaseFloat> &trans =
          (use_global_transform ? global_transform : transform_reader.Value(utt));
      int32 transform_rows = trans.NumRows(),
          transform_cols = trans.NumCols(),
          feat_dim = feat.NumCols();

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
      } else {
        KALDI_WARN << "Transform matrix for utterance " << utt << " has bad dimension "
                   << transform_rows << "x" << transform_cols << " versus feat dim "
                   << feat_dim;
        if (transform_cols == feat_dim+2)
          KALDI_WARN << "[perhaps the transform was created by compose-transforms, "
              "and you forgot the --b-is-affine option?]";
        num_error++;
        continue;
      }
      num_done++;

      if (logdet_type == Unknown) {
        if (transform_rows == feat_dim) logdet_type = Logdet;  // actual logdet.
        else if (transform_rows < feat_dim) logdet_type = PseudoLogdet;  // see below
        else logdet_type = DimIncrease;  // makes no sense to have any logdet.
        // PseudoLogdet is if we have a dimension-reducing transform T, we compute
        // 1/2 logdet(T T^T).  Why does this make sense?  Imagine we do MLLT after
        // LDA and compose the transforms; the MLLT matrix is A and the LDA matrix is L,
        // so T = A L.  T T^T = A L L^T A, so 1/2 logdet(T T^T) = logdet(A) + 1/2 logdet(L L^T).
        // since L L^T is a constant, this is valid for comparing likelihoods if we're
        // just trying to see if the MLLT is converging.
      }

      if (logdet_type != DimIncrease) { // Accumulate log-determinant stats.
        SubMatrix<BaseFloat> linear_transform(trans, 0, trans.NumRows(), 0, feat_dim);
        // "linear_transform" is just the linear part of any transform, ignoring
        // any affine (offset) component.
        SpMatrix<BaseFloat> TT(trans.NumRows());
        // TT = linear_transform * linear_transform^T
        TT.AddMat2(1.0, linear_transform, kNoTrans, 0.0);
        BaseFloat logdet;
        if (use_global_transform) {
          if (cached_logdet == -1)
            cached_logdet = 0.5 * TT.LogDet(NULL);
          logdet = cached_logdet;
        } else {
          logdet = 0.5 * TT.LogDet(NULL);
        }
        if (logdet != logdet || logdet-logdet != 0.0) // NaN or info.
          KALDI_WARN << "Matrix has bad logdet " << logdet;
        else {
          tot_t += feat.NumRows();
          tot_logdet += feat.NumRows() * logdet;
        }
      }
      feat_writer.Write(utt, feat_out);
    }
    if (logdet_type != Unknown && logdet_type != DimIncrease)
      KALDI_LOG << "Overall average " << (logdet_type == PseudoLogdet ? "[pseudo-]":"")
                << "logdet is " << (tot_logdet/tot_t) << " over " << tot_t
                << " frames.";
    KALDI_LOG << "Applied transform to " << num_done << " utterances; " << num_error
              << " had errors.";

    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}





// ali-to-post "ark:gunzip -c $alidir/ali.JOB.gz|" ark:- \|                \
//   weight-silence-post 0.0 $silphonelist $alidir/final.mdl ark:- ark:- \|  \
//   acc-lda --rand-prune=$randprune $alidir/final.mdl "$splicedfeats" ark,s,cs:- \
//   $dir/lda.JOB.acc || exit 1;

// est-lda --write-full-matrix=$dir/full.mat --dim=$dim $dir/0.mat $dir/lda.*.acc \
//   2>$dir/log/lda_est.log || exit 1;
//   rm $dir/lda.*.acc





// 1 ali-to-post
//   in: 对齐结果 trans-id
//   out: 输出给weight-slience-post
/** @brief Convert alignments to viterbi style posteriors. The aligned
    symbol gets a weight of 1.0 */
int main(int argc, char *argv[]) {
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

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

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

int main(int argc, char *argv[]) {
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

    po.Register("distribute", &distribute, "If true, rather than weighting the "
                "individual posteriors, apply the weighting to the whole frame: "
                "i.e. on time t, scale all posterior entries by "
                "p(sil)*silence-weight + p(non-sil)*1.0");

    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }


    std::string
        silence_weight_str = po.GetArg(1),
        silence_phones_str = po.GetArg(2),
        model_rxfilename = po.GetArg(3),
        posteriors_rspecifier = po.GetArg(4),
        posteriors_wspecifier = po.GetArg(5);

    BaseFloat silence_weight = 0.0;
    // silence_weight_str = 1
    // 所以 silence_weight   1.0
    // silence_set  <1>
    if (!ConvertStringToReal(silence_weight_str, &silence_weight))
      KALDI_ERR << "Invalid silence-weight parameter: expected float, got \""
                 << silence_weight_str << '"';
    
    std::vector<int32> silence_phones;
    if (!SplitStringToIntegers(silence_phones_str, ":", false, &silence_phones))
      KALDI_ERR << "Invalid silence-phones string " << silence_phones_str;
    if (silence_phones.empty())
      KALDI_WARN <<"No silence phones, this will have no effect";
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
  // frame
  for (size_t i = 0; i < post->size(); i++) {
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
int main(int argc, char *argv[]) {
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
    po.Register("binary", &binary, "Write accumulators in binary mode.");
    po.Register("rand-prune", &rand_prune,
                "Randomized pruning threshold for posteriors");
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_rxfilename = po.GetArg(1);
    std::string features_rspecifier = po.GetArg(2);
    std::string posteriors_rspecifier = po.GetArg(3);
    std::string acc_wxfilename = po.GetArg(4);

    TransitionModel trans_model;
    {
      bool binary_read;
      Input ki(model_rxfilename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      // discard rest of file.
    }


    
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

      // pdf-post 将每帧 Vector<trans-id, probability> 转为每帧 Vector<pdf-id, probability>
      Posterior pdf_post;
      ConvertPosteriorToPdfs(trans_model, post, &pdf_post);

      // foreach frame
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
      
      if (num_done % 100 == 0)
        KALDI_LOG << "Done " << num_done << " utterances.";
    }
     
    Output ko(acc_wxfilename, binary);
    lda.Write(ko.Stream(), binary);
}

void LdaEstimate::Accumulate(const VectorBase<BaseFloat> &data,
                             int32 class_id, BaseFloat weight) {

  Vector<double> data_d(data);

  // zero_acc_ vector<pdf-id - weight> 保存pdf-id 的权重累和
  zero_acc_(class_id) += weight;
  // first_acc_ pdf-id 的权重以及特征 Matrix<pdf-id < weight, feats>>
  first_acc_.Row(class_id).AddVec(weight, data_d);
  // total_second_acc 二次统计量??? 特征×特征.
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

          
void ConvertPosteriorToPdfs(const TransitionModel &tmodel,
                            const Posterior &post_in,
                            Posterior *post_out) {
  post_out->clear();
  post_out->resize(post_in.size());

  // foreach frame
  for (size_t i = 0; i < post_out->size(); i++) {
    
    unordered_map<int32, BaseFloat> pdf_to_post;
    // foreach frame's maybe <trans-id, probability>
    for (size_t j = 0; j < post_in[i].size(); j++) {

      int32 tid = post_in[i][j].first,
          pdf_id = tmodel.TransitionIdToPdf(tid);
      
      BaseFloat post = post_in[i][j].second;

      // 多个trans-id 会对应一个 pdf-id
      if (pdf_to_post.count(pdf_id) == 0)
        pdf_to_post[pdf_id] = post;
      else
        pdf_to_post[pdf_id] += post;
    }
    
    (*post_out)[i].reserve(pdf_to_post.size());

    // 使用 Posterior post_out 保存 每帧的可能<pdf-id, probability>
    for (unordered_map<int32, BaseFloat>::const_iterator iter =
             pdf_to_post.begin(); iter != pdf_to_post.end(); ++iter) {
      
      if (iter->second != 0.0)
        (*post_out)[i].push_back(
            std::make_pair(iter->first, iter->second));
    }
  }
}





// 4 est-lda
// est-lda --write-full-matrix=$dir/full.mat --dim=$dim $dir/0.mat $dir/lda.*.acc \

int main(int argc, char *argv[]) {  
  using namespace kaldi;
  typedef kaldi::int32 int32;
    const char *usage =
        "Estimate LDA transform using stats obtained with acc-lda.\n"
        "Usage:  est-lda [options] <lda-matrix-out> <lda-acc-1> <lda-acc-2> ...\n";

    bool binary = true;
    std::string full_matrix_wxfilename;
    
    LdaEstimateOptions opts;
    
    ParseOptions po(usage);
    po.Register("binary", &binary, "Write matrix in binary mode.");
    po.Register("write-full-matrix", &full_matrix_wxfilename,
                "Write full LDA matrix to this location.");
    opts.Register(&po);
    po.Read(argc, argv);

    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }

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

    // 写入 lda_mat  full_lda_mat。
    WriteKaldiObject(lda_mat, lda_mat_wxfilename, binary);
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
//     此时 feats 是经过上面lda变换矩阵 变换后的特征, 已经实现了lda降维.  feats的特征此时为 dim = 40



// 按照新的 feats 重新构建决策树.

// 注意此时 feats特征已经 与 原本gmm参数的特征不同, 原本GMM适用的特征 已经经过升维后 lda进行降维, 需要重新估计GMM参数.
// 统计量, 
// if [ $stage -le -4 ] && $train_tree; then
//     acc-tree-stats $context_opts \
//     --ci-phones=$ciphonelist $alidir/final.mdl "$feats" \
//     "ark:gunzip -c $alidir/ali.JOB.gz|" $dir/JOB.treeacc || exit 1;

//     sum-tree-stats $dir/treeacc $dir/*.treeacc || exit 1;

//   rm $dir/*.treeacc
// fi


// 重新应用一遍决策树构建过程.
// if [ $stage -le -3 ] && $train_tree; then
//   echo "$0: Getting questions for tree clustering."
//   # preparing questions, roots file...
//   cluster-phones $context_opts $dir/treeacc $lang/phones/sets.int \
//     $dir/questions.int 2> $dir/log/questions.log || exit 1;
//   cat $lang/phones/extra_questions.int >> $dir/questions.int
//   compile-questions $context_opts $lang/topo $dir/questions.int \
//     $dir/questions.qst 2>$dir/log/compile_questions.log || exit 1;

//   echo "$0: Building the tree"
//   $cmd $dir/log/build_tree.log \
//     build-tree $context_opts --verbose=1 --max-leaves=$numleaves \
//     --cluster-thresh=$cluster_thresh $dir/treeacc $lang/phones/roots.int \
//     $dir/questions.qst $lang/topo $dir/tree || exit 1;
// fi




// $dir/tree 是 决策树状态绑定之后的决策树, 在gmm-init-model中是用来做topo的, 但是、
// 此时的topo与初始化的topo完全不同, 包含了很重要的状态pdf-id绑定信息.

// if [ $stage -le -2 ]; then
//   echo "$0: Initializing the model"
//   if $train_tree; then
//     gmm-init-model  --write-occs=$dir/1.occs  \
//       $dir/tree $dir/treeacc $lang/topo $dir/1.mdl 2> $dir/log/init_model.log || exit 1;
//     grep 'no stats' $dir/log/init_model.log && echo "This is a bad warning.";
//     rm $dir/treeacc
// fi

// gmm-init-model  --write-occs=$dir/1.occs                              \
//     $dir/tree $dir/treeacc $lang/topo $dir/1.mdl 2> $dir/log/init_model.log || exit 1;


//转换原对齐 trans-id --> pdf-id

// if [ $stage -le -1 ]; then
//   # Convert the alignments.
//   echo "$0: Converting alignments from $alidir to use current tree"
//   $cmd JOB=1:$nj $dir/log/convert.JOB.log \
//     convert-ali $alidir/final.mdl $dir/1.mdl $dir/tree \
//      "ark:gunzip -c $alidir/ali.JOB.gz|" "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1;
// fi



//为每个utt  构建 phone -> word fst图
// if [ $stage -le 0 ] && [ "$realign_iters" != "" ]; then
//   echo "$0: Compiling graphs of transcripts"
//   $cmd JOB=1:$nj $dir/log/compile_graphs.JOB.log \
//     compile-train-graphs --read-disambig-syms=$lang/phones/disambig.int $dir/tree $dir/1.mdl  $lang/L.fst  \
//      "ark:utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt < $data/split$nj/JOB/text |" \
//       "ark:|gzip -c >$dir/fsts.JOB.gz" || exit 1;
// fi



while [ $x -lt $num_iters ]; do
  echo Training pass $x
  // 1 根据utt的图, feats特征  ==> 生成trans-id对齐信.息
  gmm-align-compiled $scale_opts --beam=$beam --retry-beam=$retry_beam --careful=$careful "$mdl" \
    "ark:gunzip -c $dir/fsts.JOB.gz|" "$feats" \
    "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1;

  // 2 根据trans-id对齐信息, 转化为 pdf-id对齐信息， 然后增加silence权重.
    if [ $stage -le $x ]; then
      echo "$0: Estimating MLLT"
      $cmd JOB=1:$nj $dir/log/macc.$x.JOB.log \
        ali-to-post "ark:gunzip -c $dir/ali.JOB.gz|" ark:- \| \
        weight-silence-post 0.0 $silphonelist $dir/$x.mdl ark:- ark:- \| \
                              
        gmm-acc-mllt --rand-prune=$randprune  $dir/$x.mdl "$feats" ark:- $dir/$x.JOB.macc \

      est-mllt $dir/$x.mat.new $dir/$x.*.macc 2> $dir/log/mupdate.$x.log || exit 1;

      gmm-transform-means  $dir/$x.mat.new $dir/$x.mdl $dir/$x.mdl \

      compose-transforms --print-args=false $dir/$x.mat.new $dir/$cur_lda_iter.mat $dir/$x.mat || exit 1;
      rm $dir/$x.*.macc
    fi

      // 多次迭代的 mllt mat矩阵 $x.mat.
    feats="$splicedfeats transform-feats $dir/$x.mat ark:- ark:- |"
    cur_lda_iter=$x

  if [ $stage -le $x ]; then
    $cmd JOB=1:$nj $dir/log/acc.$x.JOB.log \
      gmm-acc-stats-ali  $dir/$x.mdl "$feats" \
      "ark,s,cs:gunzip -c $dir/ali.JOB.gz|" $dir/$x.JOB.acc || exit 1;
    $cmd $dir/log/update.$x.log \
      gmm-est --write-occs=$dir/$[$x+1].occs --mix-up=$numgauss --power=$power \
        $dir/$x.mdl "gmm-sum-accs - $dir/$x.*.acc |" $dir/$[$x+1].mdl || exit 1;
    rm $dir/$x.mdl $dir/$x.*.acc $dir/$x.occs
  fi
  [ $x -le $max_iter_inc ] && numgauss=$[$numgauss+$incgauss];
  x=$[$x+1];
done













void question(){
  // 1 pdf-id  and trans-id ??? 区别


  
}
