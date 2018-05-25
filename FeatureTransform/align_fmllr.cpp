

// ## Set up model and alignment model.
// mdl=$srcdir/final.mdl
// 判断是否存在,确定是否使用alimdl
// if [ -f $srcdir/final.alimdl ]; then
//   alimdl=$srcdir/final.alimdl
// else
//   alimdl=$srcdir/final.mdl
// fi
// [ ! -f $mdl ] && echo "$0: no such model $mdl" && exit 1;
// alimdl_cmd="gmm-boost-silence --boost=$boost_silence `cat $lang/phones/optional_silence.csl` $alimdl - |"
// mdl_cmd="gmm-boost-silence --boost=$boost_silence `cat $lang/phones/optional_silence.csl` $mdl - |"


// sifeats 是经过lda_mllt_fmllr变换后的特征.
// 这里还使用不同的 mdl  --- $alimdl, $mdl

//   echo "$0: computing fMLLR transforms"  怎么又计算 fmllr 转换矩阵

//   if [ "$alimdl" != "$mdl" ]; then
//       ali-to-post "ark:gunzip -c $dir/pre_ali.JOB.gz|" ark:- \| \
//       weight-silence-post 0.0 $silphonelist $alimdl ark:- ark:- \| \

//       这里使用了 alimdl, mdl.
//       先使用 alimdl(是同时使用 fmllr特征 和 lda_mllt特征 得到的转移模型)得到 新形式的对齐信息 <pdf-id, list-高斯分量后验概率>
//       然后利用得到的高斯分量后验概率 再利用 fmllr变换后特征 重新计算得到一个 fmllr变换矩阵.... 没什么特别的作用吧.??
//       感觉纯粹毫无道理 是实验出来的结果么.

//       gmm-post-to-gpost $alimdl "$sifeats" ark:- ark:- \| \
//       gmm-est-fmllr-gpost --fmllr-update-type=$fmllr_update_type \
//       --spk2utt=ark:$sdata/JOB/spk2utt $mdl "$sifeats" \
//       ark,s,cs:- ark:$dir/trans.JOB
//   else
//       ali-to-post "ark:gunzip -c $dir/pre_ali.JOB.gz|" ark:- \| \
//       weight-silence-post 0.0 $silphonelist $alimdl ark:- ark:- \| \

//       这里是判断alimdl == mdl 这里只用mdl. 正常重新更新fmllr.
//       gmm-est-fmllr --fmllr-update-type=$fmllr_update_type \
//       --spk2utt=ark:$sdata/JOB/spk2utt $mdl "$sifeats" \
//       ark,s,cs:- ark:$dir/trans.JOB
//   fi


int gmm_post_to_gpost(int argc, char *argv[]) {
  using namespace kaldi;
  const char *usage =
        "Convert state-level posteriors to Gaussian-level posteriors\n"
        "Usage:  gmm-post-to-gpost [options] <model-in> <feature-rspecifier> <posteriors-rspecifier> "
        "<gpost-wspecifier>\n"
        "e.g.: \n"
        " gmm-post-to-gpost 1.mdl scp:train.scp ark:1.post ark:1.gpost\n";

    ParseOptions po(usage);
    bool binary = true;
    BaseFloat rand_prune = 0.0;

    std::string
        model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        posteriors_rspecifier = po.GetArg(3),
        gpost_wspecifier = po.GetArg(4);

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

    double tot_like = 0.0;
    double tot_t = 0.0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessPosteriorReader posteriors_reader(posteriors_rspecifier);

    GaussPostWriter gpost_writer(gpost_wspecifier);

    int32 num_done = 0, num_no_posterior = 0, num_other_error = 0;
    // foreach utt feat
    for (; !feature_reader.Done(); feature_reader.Next()) {
      
      std::string key = feature_reader.Key();
      if (!posteriors_reader.HasKey(key)) {
        num_no_posterior++;
      } else {
        
        const Matrix<BaseFloat> &mat = feature_reader.Value();
        const Posterior &posterior = posteriors_reader.Value(key);
        
        GaussPost gpost(posterior.size());

        if (posterior.size() != mat.NumRows()) {
          KALDI_WARN << "Posterior vector has wrong size "<< (posterior.size()) << " vs. "<< (mat.NumRows());
          num_other_error++;
          continue;
        }

        num_done++;
        BaseFloat tot_like_this_file = 0.0, tot_weight = 0.0;

        Posterior pdf_posterior;
        // 从tid,prob 转化为pdf-id,prb
        ConvertPosteriorToPdfs(trans_model, posterior, &pdf_posterior);

        // foreach frame
        for (size_t i = 0; i < posterior.size(); i++) {
          
          gpost[i].reserve(pdf_posterior[i].size());
          // foreach probable pdf-id
          for (size_t j = 0; j < pdf_posterior[i].size(); j++) {

            int32 pdf_id = pdf_posterior[i][j].first;
            BaseFloat weight = pdf_posterior[i][j].second;
            
            const DiagGmm &gmm = am_gmm.GetPdf(pdf_id);
            Vector<BaseFloat> this_post_vec;

            // 计算各高斯分量的后验概率
            BaseFloat like =
                gmm.ComponentPosteriors(mat.Row(i), &this_post_vec);
            
            this_post_vec.Scale(weight);
            
            if (rand_prune > 0.0)
              for (int32 k = 0; k < this_post_vec.Dim(); k++)
                this_post_vec(k) = RandPrune(this_post_vec(k),
                                             rand_prune);
            // gpost[i] 保存<pdf-id,对应的高斯分量后验概率list>
            if (!this_post_vec.IsZero())
              gpost[i].push_back(std::make_pair(pdf_id, this_post_vec));
            
            tot_like_this_file += like * weight;
            tot_weight += weight;
          }
        }
        

        tot_like += tot_like_this_file;
        tot_t += tot_weight;
        gpost_writer.Write(key, gpost);
      }
    }
}


int gmm_est_fmllr_gpost(int argc, char *argv[]) {

    const char *usage =
        "Estimate global fMLLR transforms, either per utterance or for the supplied\n"
        "set of speakers (spk2utt option).  Reads Gaussian-level posteriors.  Writes\n"
        "to a table of matrices.\n"
        "Usage: gmm-est-fmllr-gpost [options] <model-in> "
        "<feature-rspecifier> <gpost-rspecifier> <transform-wspecifier>\n";

    ParseOptions po(usage);
    FmllrOptions fmllr_opts;
    string spk2utt_rspecifier;
    po.Register("spk2utt", &spk2utt_rspecifier, "rspecifier for speaker to "
                "utterance-list map");

    string
        model_rxfilename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        gpost_rspecifier = po.GetArg(3),
        trans_wspecifier = po.GetArg(4);

    TransitionModel trans_model;
    AmDiagGmm am_gmm;
    {
      bool binary;
      Input ki(model_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_gmm.Read(ki.Stream(), binary);
    }

    // gpost --- 保存的是 <utt <frame [pdf-id, list<高斯分量后验概率] > > 
    RandomAccessGaussPostReader gpost_reader(gpost_rspecifier);

    double tot_impr = 0.0, tot_t = 0.0;

    BaseFloatMatrixWriter transform_writer(trans_wspecifier);

    int32 num_done = 0, num_no_gpost = 0, num_other_error = 0;
    // 特征先经过说话人自适应
    if (spk2utt_rspecifier != "") {  // per-speaker adaptation
      SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
      RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);

      // foreach spker.
      for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {

        // spker fmllr 统计量
        FmllrDiagGmmAccs spk_stats(am_gmm.Dim());
        
        string spk = spk2utt_reader.Key();
        const vector<string> &uttlist = spk2utt_reader.Value();
        // foreach spker's utt
        for (size_t i = 0; i < uttlist.size(); i++) {
          std::string utt = uttlist[i];

          const Matrix<BaseFloat> &feats = feature_reader.Value(utt);
          // gpost对齐 --- pdf-id的带高斯分量后验概率 对齐方式.
          const GaussPost &gpost = gpost_reader.Value(utt);
          // 根据特征feat 和 pdf-id内的高斯分量后验概率 累计 spker对应的 fmllr统计量
          AccumulateForUtterance(feats, gpost, trans_model, am_gmm, &spk_stats);

          num_done++;
        }

        // 对每个spker 计算新的 fmllr 变换矩阵.
        BaseFloat impr, spk_tot_t;
        {
          // Compute the transform and write it out.
          Matrix<BaseFloat> transform(am_gmm.Dim(), am_gmm.Dim()+1);
          transform.SetUnit();
          // 更新 fmllr  变换矩阵  ---> transform.MAT
          spk_stats.Update(fmllr_opts, &transform, &impr, &spk_tot_t);
          // 写入spk
          transform_writer.Write(spk, transform);
        }
        tot_impr += impr;
        tot_t += spk_tot_t;
      }  // end looping over speakers
    } else {  // per-utterance adaptation
      
      SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
      for (; !feature_reader.Done(); feature_reader.Next()) {
        string utt = feature_reader.Key();
        if (!gpost_reader.HasKey(utt)) {
          KALDI_WARN << "Did not find gposts for utterance "
                     << utt;
          num_no_gpost++;
          continue;
        }
        const Matrix<BaseFloat> &feats = feature_reader.Value();
        const GaussPost &gpost = gpost_reader.Value(utt);

        if (static_cast<int32>(gpost.size()) != feats.NumRows()) {
          KALDI_WARN << "GaussPost has wrong size " << (gpost.size())
              << " vs. " << (feats.NumRows());
          num_other_error++;
          continue;
        }
        num_done++;

        FmllrDiagGmmAccs spk_stats(am_gmm.Dim());
        // 
        AccumulateForUtterance(feats, gpost, trans_model, am_gmm,
                               &spk_stats);

        BaseFloat impr, utt_tot_t;
        {  // Compute the transform and write it out.
          Matrix<BaseFloat> transform(am_gmm.Dim(), am_gmm.Dim()+1);
          transform.SetUnit();
          spk_stats.Update(fmllr_opts, &transform, &impr, &utt_tot_t);
          transform_writer.Write(utt, transform);
        }
        tot_impr += impr;
        tot_t += utt_tot_t;
      }
    }

    
}


void AccumulateForUtterance(const Matrix<BaseFloat> &feats,
                            const GaussPost &gpost,
                            const TransitionModel &trans_model,
                            const AmDiagGmm &am_gmm,
                            FmllrDiagGmmAccs *spk_stats) {
  // foreach frame  多个可能的<pdf-id, list高斯分量后验概率>
  for (size_t i = 0; i < gpost.size(); i++) {
    // foreach <pdf-id, list高斯分量后验概率>
    for (size_t j = 0; j < gpost[i].size(); j++) {
      int32 pdf_id = gpost[i][j].first;
      // list 高斯分量后验概率
      const Vector<BaseFloat> & posterior(gpost[i][j].second);
      // 根据高斯分量后验概率, 修改fmllr统计量.
      spk_stats->AccumulateFromPosteriors(am_gmm.GetPdf(pdf_id),
                                          feats.Row(i), posterior);
    }
  }
}

void FmllrDiagGmmAccs:: AccumulateFromPosteriors(
    const DiagGmm &pdf,
    const VectorBase<BaseFloat> &data,
    const VectorBase<BaseFloat> &posterior) {

  // 修改 fmllr 的统计量.
  SingleFrameStats &stats = this->single_frame_stats_;
  
  stats.count += posterior.Sum();
  stats.a.AddMatVec(1.0, pdf.means_invvars(), kTrans, posterior, 1.0);
  stats.b.AddMatVec(1.0, pdf.inv_vars(), kTrans, posterior, 1.0);
}
























// gmmbin/gmm-est-fmllr.cc

void AccumulateForUtterance(const Matrix<BaseFloat> &feats,
                            const Posterior &post,
                            const TransitionModel &trans_model,
                            const AmDiagGmm &am_gmm,
                            FmllrDiagGmmAccs *spk_stats) {
  Posterior pdf_post;
  ConvertPosteriorToPdfs(trans_model, post, &pdf_post);
  for (size_t i = 0; i < post.size(); i++) {
    for (size_t j = 0; j < pdf_post[i].size(); j++) {
      int32 pdf_id = pdf_post[i][j].first;
      spk_stats->AccumulateForGmm(am_gmm.GetPdf(pdf_id),
                                  feats.Row(i),
                                  pdf_post[i][j].second);
    }
  }
}

int gmm_est_fmllr(int argc, char *argv[]) {
    typedef kaldi::int32 int32;
    using namespace kaldi;
    const char *usage =
        "Estimate global fMLLR transforms, either per utterance or for the supplied\n"
        "set of speakers (spk2utt option).  Reads posteriors (on transition-ids).  Writes\n"
        "to a table of matrices.\n"
        "Usage: gmm-est-fmllr [options] <model-in> "
        "<feature-rspecifier> <post-rspecifier> <transform-wspecifier>\n";

    ParseOptions po(usage);
    FmllrOptions fmllr_opts;
    string spk2utt_rspecifier;
    po.Register("spk2utt", &spk2utt_rspecifier, "rspecifier for speaker to "
                "utterance-list map");
    fmllr_opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

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

      for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
        FmllrDiagGmmAccs spk_stats(am_gmm.Dim(), fmllr_opts);
        string spk = spk2utt_reader.Key();
        const vector<string> &uttlist = spk2utt_reader.Value();
        for (size_t i = 0; i < uttlist.size(); i++) {
          std::string utt = uttlist[i];
          if (!feature_reader.HasKey(utt)) {
            KALDI_WARN << "Did not find features for utterance " << utt;
            num_other_error++;
            continue;
          }
          if (!post_reader.HasKey(utt)) {
            KALDI_WARN << "Did not find posteriors for utterance " << utt;
            num_no_post++;
            continue;
          }
          const Matrix<BaseFloat> &feats = feature_reader.Value(utt);
          const Posterior &post = post_reader.Value(utt);
          if (static_cast<int32>(post.size()) != feats.NumRows()) {
            KALDI_WARN << "Posterior vector has wrong size " << (post.size())
                       << " vs. " << (feats.NumRows());
            num_other_error++;
            continue;
          }

          // 和 gpost 的区别就是没有使用gpost 这个概率而已.
          AccumulateForUtterance(feats, post, trans_model, am_gmm, &spk_stats);

          num_done++;
        }  // end looping over all utterances of the current speaker

        BaseFloat impr, spk_tot_t;
        {  // Compute the transform and write it out.
          Matrix<BaseFloat> transform(am_gmm.Dim(), am_gmm.Dim()+1);
          transform.SetUnit();
          spk_stats.Update(fmllr_opts, &transform, &impr, &spk_tot_t);
          transform_writer.Write(spk, transform);
        }
        KALDI_LOG << "For speaker " << spk << ", auxf-impr from fMLLR is "
                  << (impr/spk_tot_t) << ", over " << spk_tot_t << " frames.";
        tot_impr += impr;
        tot_t += spk_tot_t;
      }  // end looping over speakers
    }
}

