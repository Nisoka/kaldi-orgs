// steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" 2000 10000 data/mfcc/train data/lang exp/mono_ali exp/tri1

// StateId next_state = ofst->AddState();
// //以word标注为 inlabel, outlabel 权重=1 目标状态，构建转移弧
// Arc arc(labels[i], labels[i], Weight::One(), next_state);
// //增加转移弧, 
// ofst->AddArc(cur_state, arc);
// cur_state = next_state;
// 因为转移弧是有源状态 目标状态的， 所有只是简单的将弧放入到一个集合(Vector)
// 就可以描述一个图. 因此就是用这种数据结构表示 图FST。


// # To be run from ..
// # Flat start and monophone training, with delta-delta features.
// # This script applies cepstral mean normalization (per speaker).

void configuration(){
  // # Begin configuration section.
  // nj=4
  // cmd=run.pl
  // scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"

  // num_iters=40    # Number of iterations of training
  // max_iter_inc=30 # Last iter to increase #Gauss on.
  // totgauss=1000 # Target #Gaussians.

  // careful=false
  
  // boost_silence=1.0 # Factor by which to boost silence likelihoods in alignment

  // realign_iters="1 2 3 4 5 6 7 8 9 10 12 14 16 18 20 23 26 29 32 35 38";
  // config= # name of config file.
  // stage=-4
  // power=0.25 # exponent to determine number of gaussians from occurrence counts
  // norm_vars=false # deprecated, prefer --cmvn-opts "--norm-vars=false"
  // cmvn_opts=  # can be used to add extra options to cmvn.
  // # End configuration section.
  
}

// if [ $# != 3 ]; then
//   echo "Usage: steps/train_mono.sh [options] <data-dir> <lang-dir> <exp-dir>"
//   echo " e.g.: steps/train_mono.sh data/train.1k data/lang exp/mono"
//   echo "main options (for others, see top of script file)"
//   echo "  --config <config-file>                           # config containing options"
//   echo "  --nj <nj>                                        # number of parallel jobs"
//   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
//   exit 1;
// fi


// data=$1   data/mfcc/train  mfcc输入特征scp
// lang=$2   data/lang        发音词典模型等信息
// dir=$3    exp/mono_ali     输出trans-id对齐信息


// 应用了cmvn 特定说话人特征 倒谱均值归一化的feat特征
// feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |"
// example_feats="`echo $feats | sed s/JOB/1/g`";

// echo "$0: Initializing monophone system."

// 共享因素列表. data/lang/phones/sets.int
// shared_phones_opt="--shared-phones=$lang/phones/sets.int"

// 获得特征的维度
// # Note: JOB=1 just uses the 1st part of the features-- we only need a subset anyway.
// if ! feat_dim=`feat-to-dim "$example_feats" - 2>/dev/null` || [ -z $feat_dim ]; then
//   feat-to-dim "$example_feats" -

// 初始化训练mono 各种参数.
// gmm-init-mono $shared_phones_opt "--train-feats=$feats subset-feats --n=10 ark:- ark:-|" $lang/topo $feat_dim \
//   $dir/0.mdl $dir/tree || exit 1;
    // in:
    // 共享音素列表  sets.int
    // feats 特征, 并且只是一部分特征
    // topo结构
    // dim 特征维度
    // out:
    // 0.mdl transmodel & gmm-paramters
    // tree 基本决树 无用
    // 相当简陋的 转移模型 以及 GMM参数(都是完全一样的)
int gmm_init_mono(int argc, char *argv[]) {
    using namespace kaldi;
    using kaldi::int32;

    const char *usage =
        "Initialize monophone GMM.\n"
        "Usage:  gmm-init-mono <topology-in> <dim> <model-out> <tree-out> \n"
        "e.g.: \n"
        " gmm-init-mono topo 39 mono.mdl mono.tree\n";

    bool binary = true;
    std::string train_feats;
    std::string shared_phones_rxfilename;
    BaseFloat perturb_factor = 0.0;
    ParseOptions po(usage);
    po.Register("train-feats", &train_feats,
                "rspecifier for training features [used to set mean and variance]");
    po.Register("shared-phones", &shared_phones_rxfilename,
                "rxfilename containing, on each line, a list of phones whose pdfs should be shared.");

    std::string topo_filename = po.GetArg(1);
    int dim = atoi(po.GetArg(2).c_str());
    KALDI_ASSERT(dim> 0 && dim < 10000);
    std::string model_filename = po.GetArg(3);
    std::string tree_filename = po.GetArg(4);

    // 全局均值、方差数据
    Vector<BaseFloat> glob_inv_var(dim);
    glob_inv_var.Set(1.0);
    Vector<BaseFloat> glob_mean(dim);
    glob_mean.Set(1.0);

    if (train_feats != "") {
      double count = 0.0;
      Vector<double> var_stats(dim);
      Vector<double> mean_stats(dim);
      SequentialDoubleMatrixReader feat_reader(train_feats);
      // foreach utt 
      for (; !feat_reader.Done(); feat_reader.Next()) {
        const Matrix<double> &mat = feat_reader.Value();
        // foreach frame 
        for (int32 i = 0; i < mat.NumRows(); i++) {
          count += 1.0;
          var_stats.AddVec2(1.0, mat.Row(i));
          mean_stats.AddVec(1.0, mat.Row(i));
        }
      }
      
      
      var_stats.Scale(1.0/count);
      mean_stats.Scale(1.0/count);
      var_stats.AddVec2(-1.0, mean_stats);

      
      var_stats.InvertElements();
      glob_inv_var.CopyFromVec(var_stats);
      glob_mean.CopyFromVec(mean_stats);
    }

    HmmTopology topo;
    bool binary_in;
    Input ki(topo_filename, &binary_in);
    topo.Read(ki.Stream(), binary_in);

    // 获得所有音素
    const std::vector<int32> &phones = topo.GetPhones();

    std::vector<int32> phone2num_pdf_classes (1+phones.back());
    // 根据topo 获得每个音素内部的pdf-class 现在无绑定 一般就是 3
    for (size_t i = 0; i < phones.size(); i++)
      phone2num_pdf_classes[phones[i]] = topo.NumPdfClasses(phones[i]);

    // Now the tree [not really a tree at this point]:
    // 直接根据 topo结构 和音素总数 构建一个简陋树， 为了得到<phoen,pdf-class>对应的pdf-id.
    ContextDependency *ctx_dep = NULL;
    if (shared_phones_rxfilename == "") {  // No sharing of phones: standard approach.
      ctx_dep = MonophoneContextDependency(phones, phone2num_pdf_classes);
    } else {
      std::vector<std::vector<int32> > shared_phones;
      ReadSharedPhonesList(shared_phones_rxfilename, &shared_phones);
      // ReadSharedPhonesList crashes on error.
      ctx_dep = MonophoneContextDependencyShared(shared_phones, phone2num_pdf_classes);
    }

    // 所有的pdf-id
    int32 num_pdfs = ctx_dep->NumPdfs();

    AmDiagGmm am_gmm;
    DiagGmm gmm;
    // 使用全局特征统计量 计算一个GMM的Gconst(Gconst 是为了进行估计GMM参数用的部分统计量)
    gmm.Resize(1, dim);
    {  // Initialize the gmm.
      Matrix<BaseFloat> inv_var(1, dim);
      inv_var.Row(0).CopyFromVec(glob_inv_var);
      Matrix<BaseFloat> mu(1, dim);
      mu.Row(0).CopyFromVec(glob_mean);
      Vector<BaseFloat> weights(1);
      weights.Set(1.0);
      gmm.SetInvVarsAndMeans(inv_var, mu);
      gmm.SetWeights(weights);
      gmm.ComputeGconsts();
    }

    // 此时认为所有Pdf-id (所有的状态生成GMM函数都是相同的) 
    for (int i = 0; i < num_pdfs; i++)
      am_gmm.AddPdf(gmm);

    if (perturb_factor != 0.0) {
      for (int i = 0; i < num_pdfs; i++)
        am_gmm.GetPdf(i).Perturb(perturb_factor);
    }

    // Now the transition model:
    TransitionModel trans_model(*ctx_dep, topo);

    // 写入文件保存
    {
      Output ko(model_filename, binary);
      trans_model.Write(ko.Stream(), binary);
      am_gmm.Write(ko.Stream(), binary);
    }

    // Now write the tree.
    ctx_dep->Write(Output(tree_filename, binary).Stream(),
                   binary);

    delete ctx_dep;
    return 0;
}
    

// // 根据 保存的GMM信息 获得当前的guass 总数
// numgauss=`gmm-info --print-args=false $dir/0.mdl | grep gaussians | awk '{print $NF}'`
// // 计算每次迭代要增加的高斯总数
// incgauss=$[($totgauss-$numgauss)/$max_iter_inc] # per-iter increment for #Gauss



// compile-train-graphs --read-disambig-syms=$lang/phones/disambig.int $dir/tree $dir/0.mdl  $lang/L.fst \
//     "ark:sym2int.pl --map-oov $oov_sym -f 2- $lang/words.txt < $sdata/JOB/text|" \
//     "ark:|gzip -c >$dir/fsts.JOB.gz" || exit 1;

// disambig.int (219 - 275)
// tree (easy way)
// L.fst Lexicon.fst
// word id(0 -- nnnn)  +  sdata/JOB/text (data/mfcc/train/split/n/text) 是每句的word标记
// ---> fst.n.gz


// in:
// tree
// model
// L.fst  这里是不带销岐音素的???
// 每utt标注信息
// out:
// 输出图 graphs-wspecifier  每个utt对应的线性HCLG.FST
 
int compile_train_graphs(int argc, char *argv[]) {

    const char *usage =
        "Creates training graphs (without transition-probabilities, by default)\n"
        "\n"
        "Usage:   compile-train-graphs [options] <tree-in> <model-in> "
        "<lexicon-fst-in> <transcriptions-rspecifier> <graphs-wspecifier>\n"
        "e.g.: \n"
        " compile-train-graphs tree 1.mdl lex.fst "
        "'ark:sym2int.pl -f 2- words.txt text|' ark:graphs.fsts\n";
    ParseOptions po(usage);

    TrainingGraphCompilerOptions gopts;
    int32 batch_size = 250;
    gopts.transition_scale = 0.0;  // Change the default to 0.0 since we will generally add the
    // transition probs in the alignment phase (since they change eacm time)
    gopts.self_loop_scale = 0.0;  // Ditto for self-loop probs.
    std::string disambig_rxfilename;
    gopts.Register(&po);

    po.Register("read-disambig-syms", &disambig_rxfilename, "File containing "
                "list of disambiguation symbols in phone symbol table");
    
    std::string tree_rxfilename = po.GetArg(1);
    std::string model_rxfilename = po.GetArg(2);
    std::string lex_rxfilename = po.GetArg(3);
    std::string transcript_rspecifier = po.GetArg(4);
    
    std::string fsts_wspecifier = po.GetArg(5);


    ContextDependency ctx_dep;  // the tree.
    ReadKaldiObject(tree_rxfilename, &ctx_dep);

    TransitionModel trans_model;
    ReadKaldiObject(model_rxfilename, &trans_model);

    // FST的结构是什么样的 只是一个StdArc的集合.
    // need VectorFst because we will change it by adding subseq symbol.
    VectorFst<StdArc> *lex_fst = fst::ReadFstKaldi(lex_rxfilename);

    // read all the disambig symbals
    std::vector<int32> disambig_syms;  
    if (disambig_rxfilename != "")
      if (!ReadIntegerVectorSimple(disambig_rxfilename, &disambig_syms))
        KALDI_ERR << "fstcomposecontext: Could not read disambiguation symbols from "
                  << disambig_rxfilename;

    // 训练用FST图(每个utt一个简单图) 编译器
    // 保存了 trans-model, tree, lex-fst, disambig_symbals 
    TrainingGraphCompiler gc(trans_model, ctx_dep, lex_fst, disambig_syms, gopts);

    lex_fst = NULL;  // we gave ownership to gc.

    // word 标注信息 是int形式的 word序列.
    SequentialInt32VectorReader transcript_reader(transcript_rspecifier);

    // 保存Table形式的数据 内部保存的是VectorFst.
    TableWriter<fst::VectorFstHolder> fst_writer(fsts_wspecifier);

    int num_succeed = 0, num_fail = 0;

    if (batch_size == 1) {  // We treat batch_size of 1 as a special case in order
      // to test more parts of the code.
    } else {
      std::vector<std::string> keys;
      std::vector<std::vector<int32> > transcripts;

      // forall the utt word标注
      while (!transcript_reader.Done()) {
        keys.clear();
        transcripts.clear();
        
        // foreach utt word 标注 in a batch
        for (; !transcript_reader.Done() &&
                static_cast<int32>(transcripts.size()) < batch_size;
            transcript_reader.Next()) {
          
          keys.push_back(transcript_reader.Key());
          transcripts.push_back(transcript_reader.Value());
        }

        // 将当前batch的transcripts编译生成batch fst图 （对每个utt 编译生成一个fst）
        // 这样得到了 trans-id --- wordID的 HCLG图 每个utt都对应一个线性的HCLG图
        // 在后面进行对齐时, 只需要判断 一个frame 属于那个trans-id， 就实现了对齐
        // <tid1, tid1, tid2,tid2,tid2,tid2, tid3,tid3,tid3>
        std::vector<fst::VectorFst<fst::StdArc>* > fsts;
        if (!gc.CompileGraphsFromText(transcripts, &fsts)) {
          KALDI_ERR << "Not expecting CompileGraphs to fail.";
        }


        // 保存编译结果图.
        KALDI_ASSERT(fsts.size() == keys.size());
        // foreach the fst in the batch
        for (size_t i = 0; i < fsts.size(); i++) {
          
          if (fsts[i]->Start() != fst::kNoStateId) {
            // 判断是个正常state后, 将fst写入文件.
            num_succeed++;
            fst_writer.Write(keys[i], *(fsts[i]));
          }
        }
        DeletePointers(&fsts);
      }
    }
    return (num_succeed != 0 ? 0 : 1);
}


// 训练用 图编译器
// in:
// trans_model
// ctx_dep  tree
// lex_fst
// disambig_syms   销岐符号(也是音素符号，但是非正常音素) 构建图时需要所有音素符号.
// opts 编译选项
// out:
// 得到一个编译器,
// 简单处理了一下L.fst 为了能够和C的右上下文相关性进行组合？？
TrainingGraphCompiler::TrainingGraphCompiler(const TransitionModel &trans_model,
                                             const ContextDependency &ctx_dep,  // Does not maintain reference to this.
                                             fst::VectorFst<fst::StdArc> *lex_fst,
                                             const std::vector<int32> &disambig_syms,
                                             const TrainingGraphCompilerOptions &opts):
    trans_model_(trans_model), ctx_dep_(ctx_dep), lex_fst_(lex_fst),
    disambig_syms_(disambig_syms), opts_(opts) {
  
  using namespace fst;
  // 所有正常音素
  const std::vector<int32> &phone_syms = trans_model_.GetPhones();  // needed to create context fst.
  // 排序销岐音素
  SortAndUniq(&disambig_syms_);
  // check 销岐音素是否是个正常音素.
  for (int32 i = 0; i < disambig_syms_.size(); i++)
    if (std::binary_search(phone_syms.begin(), phone_syms.end(),
                           disambig_syms_[i]))
      KALDI_ERR << "Disambiguation symbol " << disambig_syms_[i]
                << " is also a phone.";

  // subseq_symbol 是所有音素(正常音素+销岐音素)总数
  int32 subseq_symbol = 1 + phone_syms.back();
  if (!disambig_syms_.empty() && subseq_symbol <= disambig_syms_.back())
    subseq_symbol = 1 + disambig_syms_.back();

  {
    int32 N = ctx_dep.ContextWidth(),
        P = ctx_dep.CentralPosition();
    
    // This is needed for systems with right-context or we will not successfully compose with C.
    // 这是右上文需要的，否则不能和C进行组合.
    if (P != N-1)
      AddSubsequentialLoop(subseq_symbol, lex_fst_);

  }

  // 将lex_fst_按照 输出标签排序,因为fst 实际上是一个Arc的组合, 所以能够实现排序.
  {
    // make sure lexicon is olabel sorted.
    fst::OLabelCompare<fst::StdArc> olabel_comp;
    fst::ArcSort(lex_fst_, olabel_comp);
  }
}

// in:
// 所有音素（正常音素 + 销岐音素）
// fst --- lexicconfst
// out:
// 向fst中增加 超级终止状态. 并且保持原本终止状态的终止权重， 不知道具体作用
// 说是为了保证 右上下文需要能够得到满足.
template<class Arc>
void AddSubsequentialLoop(typename Arc::Label subseq_symbol,
                          MutableFst<Arc> *fst) {

  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;

  vector<StateId> final_states;
  // 状态迭代器 遍历Lexicon FST的所有状态?
  // 保存所有终止状态
  for (StateIterator<MutableFst<Arc> > siter(*fst); !siter.Done(); siter.Next()) {
    StateId s = siter.Value();
    if (fst->Final(s) != Weight::Zero())  final_states.push_back(s);
  }

  // 但是这个超级终止状态是从哪里过去的？？？
  // 哦 下面写了 所有的终止状态都增加一个到达超级终止状态的Arc
  
  // 增加超级终止状态
  StateId superfinal = fst->AddState();
  // Arc(ilabel, olabel, weight, state)
  Arc arc(subseq_symbol, 0, Weight::One(), superfinal);
  // loop at superfinal.
  fst->AddArc(superfinal, arc);  
  fst->SetFinal(superfinal, Weight::One());

  // foreach final state.
  for (size_t i = 0; i < final_states.size(); i++) {
    StateId s = final_states[i];
    fst->AddArc(s, Arc(subseq_symbol, 0, fst->Final(s), superfinal));
    // 不要移除原本终止状态的 终止权重, 这样能够增加并发循环... 不知道啥意思.
    // No, don't remove the final-weights of the original states..
    // this is so we can add the subsequential loop in cases where
    // there is no context, and it won't hurt.
    // fst->SetFinal(s, Weight::Zero());
    arc.nextstate = final_states[i];
  }
}




// in:
// transcripts -- batchsize 个 utt 标注word
// out:
// out_fst  -- batchsize 个 utt word图

// use：
// 批量生成标注的 线性fst图(路径只有一条的一条直线)
// fst  是 从trans-id  --> word 的简图.
bool TrainingGraphCompiler::CompileGraphsFromText(
    const std::vector<std::vector<int32> > &transcripts,
    std::vector<fst::VectorFst<fst::StdArc>*> *out_fsts) {
  
  using namespace fst;
  // 根据标注, 构建batchsize 个fst,  输出结果是word
  std::vector<const VectorFst<StdArc>* > word_fsts(transcripts.size());

  // 对每utt 构建线性 FST标记图 ilabel olabel 都是wordID.
  // foreach utt word FST.
  for (size_t i = 0; i < transcripts.size(); i++) {
    VectorFst<StdArc> *word_fst = new VectorFst<StdArc>();
    MakeLinearAcceptor(transcripts[i], word_fst);
    word_fsts[i] = word_fst;
  }

  // 构建一个完整图  HCLG.Fst
  bool ans = CompileGraphs(word_fsts, out_fsts);
  for (size_t i = 0; i < transcripts.size(); i++)
    delete word_fsts[i];
  return ans;
}

// in:
// labels utt标注信息
// out:
// ofst 生成的fst

// 根据wordID utt标注 生成一个fst图
// 因为utt的标注存在 是确定的一条路径 , 因此使用生成线性简图 fst
template<class Arc, class I>
void MakeLinearAcceptor(const vector<I> &labels, MutableFst<Arc> *ofst) {

  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;

  ofst->DeleteStates();
  // 构建初始状态
  StateId cur_state = ofst->AddState();
  ofst->SetStart(cur_state);

  // 根据label循环构建一个线性 FST图。
  // foreach label
  for (size_t i = 0; i < labels.size(); i++) {
    StateId next_state = ofst->AddState();
    // 以word标注为 inlabel, outlabel 权重=1 目标状态，构建转移弧
    Arc arc(labels[i], labels[i], Weight::One(), next_state);
    // 增加转移弧, 
    ofst->AddArc(cur_state, arc);
    cur_state = next_state;
  }
  // 将最后一个状态设置为终止状态
  ofst->SetFinal(cur_state, Weight::One());
}



// in:
// word_fst 所有utt的线性简图, 以wordID为基本label单元的基本线性简图
// out:
// out_fsts 生成对应的 以trans-id为基本label的线性图.
bool TrainingGraphCompiler::CompileGraphs(
    const std::vector<const fst::VectorFst<fst::StdArc>* > &word_fsts,
    std::vector<fst::VectorFst<fst::StdArc>* > *out_fsts) {

  out_fsts->resize(word_fsts.size(), NULL);
  
  ContextFst<StdArc> *cfst = NULL;

  {
    // make cfst [ it's expanded on the fly ]
    // all phones
    const std::vector<int32> &phone_syms = trans_model_.GetPhones();  // needed to create context fst.
    // phone_cnt 得到正常音素总数
    int32 subseq_symbol = phone_syms.back() + 1;
    // 获得所有音素总数
    if (!disambig_syms_.empty() && subseq_symbol <= disambig_syms_.back())
      subseq_symbol = 1 + disambig_syms_.back();

    // 上下文相关的音素图, 构建三音素上下文相关图.
    // important but not read!!!!!!!!!!!!!!!!!!!!111
    cfst = new ContextFst<StdArc>(subseq_symbol,               // 音素总数
                                  phone_syms,                  // 所有实际因素
                                  disambig_syms_,              // 销岐符号
                                  ctx_dep_.ContextWidth(),     // 音素窗
                                  ctx_dep_.CentralPosition());  // 音素窗中位置
  }

  // 根据每句的wordID FST + LEX.FST 构建 phone2word FST(LG.FST)
  // 根据上下文相关的cFst + phone2word Fst 构建一个 Context_phone2word.Fst(CLG.FST)
  // foreach utt word sequeitial
  for (size_t i = 0; i < word_fsts.size(); i++) {
    VectorFst<StdArc> phone2word_fst;
    
    // TableCompose more efficient than compose.
    // lex_fst 和 word_fst, 得到phone2word_fst的一个图. --  LG.fst(不过G比较简单是个线性图)
    TableCompose(*lex_fst_, *(word_fsts[i]), &phone2word_fst, &lex_cache_);

    // 通过讲Cfst + LG.FST 构建三音素相关的 CLG.FST， 以三音素id为基本单元.
    VectorFst<StdArc> ctx2word_fst;
    ComposeContextFst(*cfst, phone2word_fst, &ctx2word_fst);
    // ComposeContextFst is like Compose but faster for this particular Fst type.
    // [and doesn't expand too many arcs in the ContextFst.]

    // 结果得到一个 上下文相关的 ctx2word的fst图.
    (*out_fsts)[i] = ctx2word_fst.Copy();
    // For now this contains the FST with symbols
    // representing phones-in-context.
  }

  
  HTransducerConfig h_cfg;
  h_cfg.transition_scale = opts_.transition_scale;

  std::vector<int32> disambig_syms_h;
  //  !!! 得到三音素声学模型  H.FST
  VectorFst<StdArc> *H = GetHTransducer(cfst->ILabelInfo(),
                                        ctx_dep_,
                                        trans_model_,
                                        h_cfg,
                                        &disambig_syms_h);
  

  // foreach utt的 ctx2wordID FST
  for (size_t i = 0; i < out_fsts->size(); i++) {
    
    VectorFst<StdArc> &ctx2word_fst = *((*out_fsts)[i]);
    VectorFst<StdArc> trans2word_fst;
    // 组合 H.FST + ctx2word_fst ---> HCLG.FST
    TableCompose(*H, ctx2word_fst, &trans2word_fst);



    // 确定化
    DeterminizeStarInLog(&trans2word_fst);

    // 去掉销岐音素, 具体怎么实现的呢？
    if (!disambig_syms_h.empty()) {
      RemoveSomeInputSymbols(disambig_syms_h, &trans2word_fst);
      if (opts_.rm_eps)
        RemoveEpsLocal(&trans2word_fst);
    }
    
    // Encoded minimization.
    MinimizeEncoded(&trans2word_fst);

    std::vector<int32> disambig;
    AddSelfLoops(trans_model_,
                 disambig,
                 opts_.self_loop_scale,
                 opts_.reorder,
                 &trans2word_fst);

    // out_fst 就是 -- 优化后的HCLG.fst
    *((*out_fsts)[i]) = trans2word_fst;
  }

  delete H;
  delete cfst;
  return true;
}

// 主要就是声学模型相关的.

// in:
// ilabel_info  C.FST 中构建的 所有三音素 triphoneID 对应的 {L, central, R} 的ilabel_info_entry
// ctx_dep      当前决策树
// tran-model   转移模型
// config       配置, 主要是声学拉伸
// out: 
// disambig_syms_left  剩余的销岐音素

// 最终的到 所有三音素的fst结构 汇总到一起的 H.FST

fst::VectorFst<fst::StdArc> *GetHTransducer (const std::vector<std::vector<int32> > &ilabel_info,
                                             const ContextDependencyInterface &ctx_dep,
                                             const TransitionModel &trans_model,
                                             const HTransducerConfig &config,
                                             std::vector<int32> *disambig_syms_left) {

  HmmCacheType cache;
  // "cache" is an optimization that prevents GetHmmAsFst repeating work
  // unnecessarily.

  // 对所有三音素结构都构建一个fst???
  std::vector<const ExpandedFst<Arc>* > fsts(ilabel_info.size(), NULL);
  // 所有正常音素
  std::vector<int32> phones = trans_model.GetPhones();

  disambig_syms_left->clear();

  int32 first_disambig_sym = trans_model.NumTransitionIds() + 1;  // First disambig symbol we can have on the input side.
  int32 next_disambig_sym = first_disambig_sym;

  // make sure epsilon is epsilon...
  // 因为epsilon 在ilabel_info[0] 并且epsilon要求是{} 内为空 没有三音素.
  if (ilabel_info.size() > 0)
    KALDI_ASSERT(ilabel_info[0].size() == 0);  

  // foreach triphone， 为每个音素都构建一个 fst
  for (int32 j = 1; j < static_cast<int32>(ilabel_info.size()); j++) {  // zero is eps.
    KALDI_ASSERT(!ilabel_info[j].empty());

    // 判断是销岐音素
    if (ilabel_info[j].size() == 1 &&
       ilabel_info[j][0] <= 0) {  // disambig symbol

      // disambiguation symbol.
      int32 disambig_sym_left = next_disambig_sym++;
      disambig_syms_left->push_back(disambig_sym_left);
      // get acceptor with one path with "disambig_sym" on it.
      VectorFst<Arc> *fst = new VectorFst<Arc>;
      fst->AddState();
      fst->AddState();
      fst->SetStart(0);
      fst->SetFinal(1, Weight::One());
      fst->AddArc(0, Arc(disambig_sym_left, disambig_sym_left, Weight::One(), 1));
      fsts[j] = fst;
    } else {  // Real phone-in-context.
      std::vector<int32> phone_window = ilabel_info[j];

      // 对每一个三音素窗 构建一个 三音素的FST图 
      vectorfst<Arc> *fst = GetHmmAsFst(phone_window,
                                        ctx_dep,
                                        trans_model,
                                        config,
                                        &cache);
      fsts[j] = fst;
    }
  }

  // 将所有三音素的fst 汇总到一个 H.FST中.
  VectorFst<Arc> *ans = MakeLoopFst(fsts);
  // remove duplicate pointers, which we will have in general, since we used the cache.
  SortAndUniq(&fsts);
  DeletePointers(&fsts);
  return ans;
}


// in:
// phone_window   输入三音素
// ctx_dep       
// trans_model
// config
// cache???

// out:
// fst            输出一个上下文三音素的fst图
// 是从 trans-id 为ilabel,olabel, 的Fst.
// 当多个三音素FST组合一起时,就形成了接收器acceptor, 当trans-id都满足就能够输出某个三音素.

fst::VectorFst<fst::StdArc> *GetHmmAsFst(
    std::vector<int32> phone_window,
    const ContextDependencyInterface &ctx_dep,
    const TransitionModel &trans_model,
    const HTransducerConfig &config,
    HmmCacheType *cache) {

  // 获得该音素
  int P = ctx_dep.CentralPosition();
  int32 phone = phone_window[P];
  if (phone == 0)
    KALDI_ERR << "phone == 0.  Some mismatch happened, or there is "
          "a code error.";

  // 获得该音素的 hmmstate结构
  const HmmTopology &topo = trans_model.GetTopo();
  const HmmTopology::TopologyEntry &entry  = topo.TopologyForPhone(phone);

  // vector of the pdfs, indexed by pdf-class (pdf-classes must start from zero
  // and be contiguous).
  std::vector<int32> pdfs(topo.NumPdfClasses(phone));
  // foreach 所有pdf-class 对应的pdf-id?
  for (int32 pdf_class = 0;
       pdf_class < static_cast<int32>(pdfs.size());
       pdf_class++) {
    // 获得pdf-class的pdf-id
    if (! ctx_dep.Compute(phone_window, pdf_class, &(pdfs[pdf_class])) ) {
      std::ostringstream ctx_ss;
      for (size_t i = 0; i < phone_window.size(); i++)
        ctx_ss << phone_window[i] << ' ';
      KALDI_ERR << "GetHmmAsFst: context-dependency object could not produce "
                << "an answer: pdf-class = " << pdf_class << " ctx-window = "
                << ctx_ss.str() << ".  This probably points "
          "to either a coding error in some graph-building process, "
          "a mismatch of topology with context-dependency object, the "
          "wrong FST being passed on a command-line, or something of "
          " that general nature.";
    }
  }
  
  std::pair<int32, std::vector<int32> > cache_index(phone, pdfs);
  if (cache != NULL) {
    HmmCacheType::iterator iter = cache->find(cache_index);
    if (iter != cache->end())
      return iter->second;
  }

  // 目标fst
  VectorFst<StdArc> *ans = new VectorFst<StdArc>;

  // state_ids 索引该因素的状态结构内状态
  // 将所有状态state 加入到state_ids中
  std::vector<StateId> state_ids;
  for (size_t i = 0; i < entry.size(); i++)
    state_ids.push_back(ans->AddState());

  // ans 以第一个状态为初始状态， 最后状态为终止状态（topo结构就是这么定义的）
  ans->SetStart(state_ids[0]);
  StateId final = state_ids.back();
  ans->SetFinal(final, Weight::One());

  // foreach hmm_state
  for (int32 hmm_state = 0;
       hmm_state < static_cast<int32>(entry.size());
       hmm_state++) {
    // 获得每个hmm_state的对应的 pdf-class 然后获得 每个hmm-state的pdf-id
    int32 forward_pdf_class = entry[hmm_state].forward_pdf_class, forward_pdf;
    int32 self_loop_pdf_class = entry[hmm_state].self_loop_pdf_class, self_loop_pdf;
    if (forward_pdf_class == kNoPdf) {  // nonemitting state.
      forward_pdf = kNoPdf;
      self_loop_pdf = kNoPdf;
    } else {
      KALDI_ASSERT(forward_pdf_class < static_cast<int32>(pdfs.size()));
      KALDI_ASSERT(self_loop_pdf_class < static_cast<int32>(pdfs.size()));
      forward_pdf = pdfs[forward_pdf_class];
      self_loop_pdf = pdfs[self_loop_pdf_class];
    }
    
    int32 trans_idx;
    // foreach trans-id from a hmm-state
    for (trans_idx = 0;
        trans_idx < static_cast<int32>(entry[hmm_state].transitions.size());
        trans_idx++) {
      BaseFloat log_prob;
      Label label;
      int32 dest_state = entry[hmm_state].transitions[trans_idx].first;
      bool is_self_loop = (dest_state == hmm_state);
      
      if (is_self_loop)
        continue; // We will add self-loops in at a later stage of processing,
      // not in this function.
      if (forward_pdf_class == kNoPdf) {
        // no pdf, hence non-estimated probability.
        // [would not happen with normal topology] .  There is no transition-state
        // involved in this case.
        log_prob = Log(entry[hmm_state].transitions[trans_idx].second);
        label = 0;
      } else {  // normal probability.
        // 对HMM-STATE的每个trans-id构建一个转移弧, 这样就构成成了每个状态的出发弧,
        // 通过构建 三音素中的所有状态 的 所有出发弧, 就构建了一个三音素的 FST
        int32 trans_state =
            trans_model.TupleToTransitionState(phone, hmm_state, forward_pdf, self_loop_pdf);
        int32 trans_id =
            trans_model.PairToTransitionId(trans_state, trans_idx);
        log_prob = trans_model.GetTransitionLogProbIgnoringSelfLoops(trans_id);
        // log_prob is a negative number (or zero)...
        label = trans_id;
      }
      // Will add probability-scale later (we may want to push first).
      // 向三音素fst中增加转移弧.
      ans->AddArc(state_ids[hmm_state],
                  Arc(label, label, Weight(-log_prob), state_ids[dest_state]));
    }
  }

  fst::RemoveEpsLocal(ans);  // this is safe and will not blow up.

  // 应用概率拉伸
  // Now apply probability scale. 
  // We waited till after the possible weight-pushing steps,
  // because weight-pushing needs "real" weights in order to work.
  ApplyProbabilityScale(config.transition_scale, ans);
  if (cache != NULL)
    (*cache)[cache_index] = ans;
  return ans;
}


// 目的应该是将所有的FST构建成一个完整的 自环FST --- 因为实际上 H就应该是这样的
// 对于一个发音序列输入 可以循环的输出 对应的音素.
template<class Arc>
VectorFst<Arc>* MakeLoopFst(const vector<const ExpandedFst<Arc> *> &fsts) {

  // 一个自环的fst？？
  VectorFst<Arc> *ans = new VectorFst<Arc>;
  StateId loop_state = ans->AddState();  // = 0.
  ans->SetStart(loop_state);
  ans->SetFinal(loop_state, Weight::One());

  // "cache" is used as an optimization when some of the pointers in "fsts"
  // may have the same value.
  unordered_map<const ExpandedFst<Arc> *, Arc> cache;

  // foreach triphone fst
  for (Label i = 0; i < static_cast<Label>(fsts.size()); i++) {
    const ExpandedFst<Arc> *fst = fsts[i];
    

    // 通过cache 进行triphone转移结构一样的优化实现过程，
    // 只是增加一个转移弧,在原本的fst上增加一个新的转移弧,输出标签代表当前triphone
    // 加快了实现过程.
    { // optimization with cache: helpful if some members of "fsts" may
      // contain the same pointer value (e.g. in GetHTransducer).
      typename unordered_map<const ExpandedFst<Arc> *, Arc>::iterator
          iter = cache.find(fst);
      if (iter != cache.end()) {
        Arc arc = iter->second;
        arc.olabel = i;
        ans->AddArc(0, arc);
        continue;
      }
    }

    // 判断要求是一个 Acceptor
    KALDI_ASSERT(fst->Properties(kAcceptor, true) == kAcceptor);  // expect acceptor.

    StateId fst_num_states = fst->NumStates();
    StateId fst_start_state = fst->Start();

    if (fst_start_state == kNoStateId)
      continue;  // empty fst.

    // 如果FST 是一个kInitialAcyclic类型的FST？？
    // 并且从初始状态出发的弧只有一条
    // 并且初始状态不是终止状态
    bool share_start_state =
        fst->Properties(kInitialAcyclic, true) == kInitialAcyclic
        && fst->NumArcs(fst_start_state) == 1
        && fst->Final(fst_start_state) == Weight::Zero();

    // fst的所有状态,添加到完整FST ans 中.
    vector<StateId> state_map(fst_num_states);  // fst state -> ans state
    for (StateId s = 0; s < fst_num_states; s++) {
      if (s == fst_start_state && share_start_state)
        state_map[s] = loop_state;
      // 对于fst中的所有状态, 在ans中创建一个对应状态.
      else
        state_map[s] = ans->AddState();
    }

    // 如果不是 share_start_state
    // 创建一个弧 ilabel 0, olabel i, weight=1, state--- 对应fst中初始状态的在ans中的状态
    // 该弧的源状态为0, 目标状态为 对应fst的起始状态
    if (!share_start_state) {
      Arc arc(0, i, Weight::One(), state_map[fst_start_state]);
      cache[fst] = arc;
      ans->AddArc(0, arc);
    }
    // foreach fst的所有状态 s
    for (StateId s = 0; s < fst_num_states; s++) {
      // Add arcs out of state s.
      // fst中s出发的所有arc
      for (ArcIterator<ExpandedFst<Arc> > aiter(*fst, s); !aiter.Done(); aiter.Next()) {
        const Arc &arc = aiter.Value();
        Label olabel = (s == fst_start_state && share_start_state ? i : 0);
        // 实现将 输入进行匹配trans-id 时， 直接就能够输出triphone.
        // 生成对应fst中弧的弧,
        // 输入标签 -----  原本的弧输入标签 --- trans-id.
        // 输出标签 -----  如果是初始状态， 则将triphonefst ID作为输出标签.
        Arc newarc(arc.ilabel, olabel, arc.weight, state_map[arc.nextstate]);
        ans->AddArc(state_map[s], newarc);
        
        if (s == fst_start_state && share_start_state)
          cache[fst] = newarc;
      }
      
      if (fst->Final(s) != Weight::Zero()) {
        KALDI_ASSERT(!(s == fst_start_state && share_start_state));
        ans->AddArc(state_map[s], Arc(0, 0, fst->Final(s), loop_state));
      }
    }
  }
  return ans;
}




// echo "$0: Aligning data equally (pass 0)"
// align-equal-compiled "ark:gunzip -c $dir/fsts.JOB.gz|" "$feats" ark,t:-  \| \
//   gmm-acc-stats-ali --binary=true $dir/0.mdl "$feats" ark:- \
//   $dir/0.JOB.acc || exit 1;

// align-equal-compiled 编译生成对应的 trans-id 对齐序列
// 其中将输出对齐信息 直接管道给 gmm-acc-stats-ali 计算估计gmm参数用到的 统计量

// in:
// graphs-rspecifier 输入utt 线性fst图
// features-rspecifier 输入 mfcc 特征维度
// out:
// aligenment-wspecifier 输出对齐结果

int align_equal_compiled(int argc, char *argv[]) {

    const char *usage =  "Write an equally spaced alignment (for getting training started)"
        "Usage:  align-equal-compiled <graphs-rspecifier> <features-rspecifier> <alignments-wspecifier>\n"
        "e.g.: \n"
        " align-equal-compiled 1.fsts scp:train.scp ark:equal.ali\n";

    std::string
        fst_rspecifier = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        alignment_wspecifier = po.GetArg(3);


    SequentialTableReader<fst::VectorFstHolder> fst_reader(fst_rspecifier);
    RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);
    Int32VectorWriter alignment_writer(alignment_wspecifier);

    int32 done = 0, no_feat = 0, error = 0;

    // foreach utt fst
    for (; !fst_reader.Done(); fst_reader.Next()) {
      std::string key = fst_reader.Key();
      if (!feature_reader.HasKey(key)) {
        KALDI_WARN << "No features for utterance " << key;
        no_feat++;
      } else {

        // utt feature & fst
        const Matrix<BaseFloat> &features = feature_reader.Value(key);
        VectorFst<StdArc> decode_fst(fst_reader.Value());
        fst_reader.FreeCurrent();  // this stops copy-on-write of the fst
        // by deleting the fst inside the reader, since we're about to mutate
        // the fst by adding transition probs.

        VectorFst<StdArc> path;
        int32 rand_seed = StringHasher()(key); // StringHasher() produces new anonymous

        // object of type StringHasher; we then call operator () on it, with "key".
        // 执行 简单平均对齐 将特征数平均分配给每个word，然后平均分配给每个phone的state
        if (EqualAlign(decode_fst, features.NumRows(), rand_seed, &path) ) {
          std::vector<int32> aligned_seq, words;
          StdArc::Weight w;
          GetLinearSymbolSequence(path, &aligned_seq, &words, &w);
          
          alignment_writer.Write(key, aligned_seq);
          done++;
        } 
      }
    }
}




// do
//     while(随意找到一条路径)
// 确保路径满足条件, 循环找路径path
// while()

// for path
//    向该路径增加自环转移, 要求在 feat长度(T个时间帧) 上都增加自环转移
// 这样保证fst图的平衡性, 从ifst 无自环的简图 得到 ofst具有自环的简图
template<class Arc>
bool EqualAlign(const Fst<Arc> &ifst,
                typename Arc::StateId length,
                int rand_seed,
                MutableFst<Arc> *ofst,
                int num_retries) {
  srand(rand_seed);

  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;


  // First select path through ifst.
  vector<StateId> path;
  vector<size_t> arc_offsets;  // arc taken out of each state.
  vector<int> nof_ilabels;

  StateId num_ilabels = 0;
  int retry_no = 0;

  // Under normal circumstances, this will be one-pass-only process
  // Multiple tries might be needed in special cases, typically when
  // the number of frames is close to number of transitions from
  // the start node to the final node. It usually happens for really
  // short utterances
  
  do {
    num_ilabels = 0;
    arc_offsets.clear();
    path.clear();
    // 获得初始 word
    path.push_back(ifst.Start());

    while (1) {
      // Select either an arc or final-prob.
      // 从扩展状态队列中取出当前状态，进行路径扩展
      // 选择当前需要进行路径扩展的源状态
      StateId s = path.back();
      // 该源状态的多有输出弧
      size_t num_arcs = ifst.NumArcs(s);
      size_t num_arcs_tot = num_arcs;

      // 如果是个终止状态
      if (ifst.Final(s) != Weight::Zero()) num_arcs_tot++;
      
      // kaldi::RandInt is a bit like Rand(), but gets around situations
      // where RAND_MAX is very small.
      // Change this to Rand() % num_arcs_tot if compile issues arise
      // 随便选个弧偏移, 这样后面用这个偏移随意得到一个弧, 来认为是实际弧(因为这个算法本身就是一个快速为了生成简单对齐的算法)
      size_t arc_offset = static_cast<size_t>(kaldi::RandInt(0, num_arcs_tot-1));

      // 是个正常弧
      if (arc_offset < num_arcs) {  // an actual arc.
        // 弧遍历迭代器 以s为源状态 遍历所有发出弧
        ArcIterator<Fst<Arc> > aiter(ifst, s);
        aiter.Seek(arc_offset);
        //获得当前弧
        const Arc &arc = aiter.Value();
        if (arc.nextstate == s) {
          continue;  // don't take this self-loop arc
        } else {
          // 如果不是self-loop arc 将弧 输出到路径中, 并讲该弧的目标状态加入到path队列中, 等待继续进行路径的扩展.
          arc_offsets.push_back(arc_offset);
          path.push_back(arc.nextstate);
          if (arc.ilabel != 0) num_ilabels++;
        }
      }
    }
    
    // 保存路径节点总数
    nof_ilabels.push_back(num_ilabels);  
    // 一定要找到一个 标签<length的路径, 这样才符合实际.
  } while (( ++retry_no < num_retries) && (num_ilabels > length));

  // path中保存一条随意正常路径
  // nof_ilabels 保存多个可能路径的 节点数.

  StateId num_self_loops = 0;
  vector<ssize_t> self_loop_offsets(path.size());

  // foreach node in path
  for (size_t i = 0; i < path.size(); i++)
    // 统计ifst在该路径上的所有 self_loop 并保存自环arc 在该节点出发的多个arc中的index
    if ( (self_loop_offsets[i] = FindSelfLoopWithILabel(ifst, path[i]))
         != static_cast<ssize_t>(-1) )
      num_self_loops++;

// template<class Arc>
// ssize_t FindSelfLoopWithILabel(const Fst<Arc> &fst, typename Arc::StateId s) {
//   for (ArcIterator<Fst<Arc> > aiter(fst, s); !aiter.Done(); aiter.Next())
//     if (aiter.Value().nextstate == s
//        && aiter.Value().ilabel != 0) return static_cast<ssize_t>(aiter.Position());
//   return static_cast<ssize_t>(-1);
// }


  StateId num_extra = length - num_ilabels;  // Number of self-loops we need.
  StateId min_num_loops = 0;                 // 每个节点平均增加自环数
  if (num_extra != 0) min_num_loops = num_extra / num_self_loops;  // prevent div by zero.
  // 剩余需要外增加的自环转移数
  StateId num_with_one_more_loop = num_extra - (min_num_loops*num_self_loops);

  // 输出图 增加初始节点.
  ofst->AddState();
  ofst->SetStart(0);
  StateId cur_state = 0;
  StateId counter = 0;  // tell us when we should stop adding one more loop.
  // foreach 路径节点
  for (size_t i = 0; i < path.size(); i++) {
    // First, add any self-loops that are necessary.
    StateId num_loops = 0;
    // 如果路径中对应节点具有 可进行自环转移
    // 判断如果counter < num_with_one_more_loop 说明还需要增加，除了平均值外的额外自环转移.
    if (self_loop_offsets[i] != static_cast<ssize_t>(-1)) {
      num_loops = min_num_loops + (counter < num_with_one_more_loop ? 1 : 0);
      counter++;
    }
    // 对路径节点i 增加自环转移
    for (StateId j = 0; j < num_loops; j++) {
      ArcIterator<Fst<Arc> > aiter(ifst, path[i]);
      aiter.Seek(self_loop_offsets[i]);
      Arc arc = aiter.Value();
      
      KALDI_ASSERT(arc.nextstate == path[i]
             && arc.ilabel != 0);  // make sure self-loop with ilabel.

      // 输出图中增加状态 next_state. 增加的自环 实际上两个state保持不变.
      StateId next_state = ofst->AddState();
      // 向输出弧中增加转移
      ofst->AddArc(cur_state, Arc(arc.ilabel, arc.olabel, arc.weight, next_state));
      cur_state = next_state;
    }


    // 按照路径 增加路径上的 前向转移Arc 
    if (i+1 < path.size()) {  // add forward transition.
      ArcIterator<Fst<Arc> > aiter(ifst, path[i]);
      aiter.Seek(arc_offsets[i]);
      Arc arc = aiter.Value();
      KALDI_ASSERT(arc.nextstate == path[i+1]);
      StateId next_state = ofst->AddState();
      ofst->AddArc(cur_state, Arc(arc.ilabel, arc.olabel, arc.weight, next_state));
      cur_state = next_state;
    } else {  // add final-prob.
      Weight weight = ifst.Final(path[i]);
      KALDI_ASSERT(weight != Weight::Zero());
      ofst->SetFinal(cur_state, weight);
    }
  }
  return true;
}



// in:
// fst 上面得到的 ofst具有自环的简图
// out:
// 输出的 对齐序列
// 
// std::vector<int32> aligned_seq, words;
// StdArc::Weight w;
// GetLinearSymbolSequence(path, &aligned_seq, &words, &w);

template<class Arc, class I>
bool GetLinearSymbolSequence(const Fst<Arc> &fst,
                             vector<I> *isymbols_out,
                             vector<I> *osymbols_out,
                             typename Arc::Weight *tot_weight_out) {
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;

  Weight tot_weight = Weight::One();
  vector<I> ilabel_seq;
  vector<I> olabel_seq;

  StateId cur_state = fst.Start();
  if (cur_state == kNoStateId) {  // empty sequence.
    if (isymbols_out != NULL) isymbols_out->clear();
    if (osymbols_out != NULL) osymbols_out->clear();
    if (tot_weight_out != NULL) *tot_weight_out = Weight::Zero();
    return true;
  }
  
  while (1) {
    Weight w = fst.Final(cur_state);
    
    if (w != Weight::Zero()) {  // is final..
      tot_weight = Times(w, tot_weight);
      if (fst.NumArcs(cur_state) != 0) return false;
      if (isymbols_out != NULL) *isymbols_out = ilabel_seq;
      if (osymbols_out != NULL) *osymbols_out = olabel_seq;
      if (tot_weight_out != NULL) *tot_weight_out = tot_weight;
      return true;
    } else {
      if (fst.NumArcs(cur_state) != 1) return false;

      ArcIterator<Fst<Arc> > iter(fst, cur_state);  // get the only arc.
      const Arc &arc = iter.Value();
      tot_weight = Times(arc.weight, tot_weight);
      if (arc.ilabel != 0) ilabel_seq.push_back(arc.ilabel);
      if (arc.olabel != 0) olabel_seq.push_back(arc.olabel);
      cur_state = arc.nextstate;
    }
  }
}











int gmm_acc_stats_ali(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
    const char *usage =
        "Accumulate stats for GMM training.\n"
        "Usage:  gmm-acc-stats-ali [options] <model-in> <feature-rspecifier> "
        "<alignments-rspecifier> <stats-out>\n"
        "e.g.:\n gmm-acc-stats-ali 1.mdl scp:train.scp ark:1.ali 1.acc\n";

    ParseOptions po(usage);
    bool binary = true;

    std::string
        model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        alignments_rspecifier = po.GetArg(3),
        accs_wxfilename = po.GetArg(4);

    AmDiagGmm am_gmm;
    TransitionModel trans_model;
    {
      bool binary;
      Input ki(model_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_gmm.Read(ki.Stream(), binary);
    }

    Vector<double> transition_accs;
    trans_model.InitStats(&transition_accs);
    AccumAmDiagGmm gmm_accs;
    gmm_accs.Init(am_gmm, kGmmAll);

    double tot_like = 0.0;
    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessInt32VectorReader alignments_reader(alignments_rspecifier);

    int32 num_done = 0, num_err = 0;
    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string key = feature_reader.Key();
      if (!alignments_reader.HasKey(key)) {
        KALDI_WARN << "No alignment for utterance " << key;
        num_err++;
      } else {
        const Matrix<BaseFloat> &mat = feature_reader.Value();
        const std::vector<int32> &alignment = alignments_reader.Value(key);

        if (alignment.size() != mat.NumRows()) {
          KALDI_WARN << "Alignments has wrong size " << (alignment.size())
                     << " vs. " << (mat.NumRows());
          num_err++;
          continue;
        }

        num_done++;
        BaseFloat tot_like_this_file = 0.0;

        
        for (size_t i = 0; i < alignment.size(); i++) {
          int32 tid = alignment[i],  // transition identifier.
              pdf_id = trans_model.TransitionIdToPdf(tid);
          
          trans_model.Accumulate(1.0, tid, &transition_accs);
          tot_like_this_file += gmm_accs.AccumulateForGmm(am_gmm, mat.Row(i),
                                                          pdf_id, 1.0);
        }
        tot_like += tot_like_this_file;
        tot_t += alignment.size();
        if (num_done % 50 == 0) {
          KALDI_LOG << "Processed " << num_done << " utterances; for utterance "
                    << key << " avg. like is "
                    << (tot_like_this_file/alignment.size())
                    << " over " << alignment.size() <<" frames.";
        }
      }
    }
    {
      Output ko(accs_wxfilename, binary);
      transition_accs.Write(ko.Stream(), binary);
      gmm_accs.Write(ko.Stream(), binary);
    }
}



// # In the following steps, the --min-gaussian-occupancy=3 option is important, otherwise
// # we fail to est "rare" phones and later on, they never align properly.

// if [ $stage -le 0 ]; then
//   gmm-est --min-gaussian-occupancy=3  --mix-up=$numgauss --power=$power \
//     $dir/0.mdl "gmm-sum-accs - $dir/0.*.acc|" $dir/1.mdl
// fi


int gmm_est(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;

  const char *usage =
      "Do Maximum Likelihood re-estimation of GMM-based acoustic model\n"
      "Usage:  gmm-est [options] <model-in> <stats-in> <model-out>\n"
      "e.g.: gmm-est 1.mdl 1.acc 2.mdl\n";

  bool binary_write = true;
  MleTransitionUpdateConfig tcfg;
  MleDiagGmmOptions gmm_opts;
  int32 mixup = 0;
  int32 mixdown = 0;
  BaseFloat perturb_factor = 0.01;
  BaseFloat power = 0.2;
  BaseFloat min_count = 20.0;
  std::string update_flags_str = "mvwt";
  std::string occs_out_filename;

  ParseOptions po(usage);
  po.Register("mix-up", &mixup, "Increase number of mixture components to "
              "this overall target.");
  po.Register("min-count", &min_count,
              "Minimum per-Gaussian count enforced while mixing up and down.");
  po.Register("power", &power, "If mixing up, power to allocate Gaussians to"
              " states.");


  kaldi::GmmFlagsType update_flags =
      StringToGmmFlags(update_flags_str);

  std::string model_in_filename = po.GetArg(1),
      stats_filename = po.GetArg(2),
      model_out_filename = po.GetArg(3);

  AmDiagGmm am_gmm;
  TransitionModel trans_model;
  {
    bool binary_read;
    Input ki(model_in_filename, &binary_read);
    trans_model.Read(ki.Stream(), binary_read);
    am_gmm.Read(ki.Stream(), binary_read);
  }

  Vector<double> transition_accs;
  AccumAmDiagGmm gmm_accs;
  {
    bool binary;
    Input ki(stats_filename, &binary);
    transition_accs.Read(ki.Stream(), binary);
    gmm_accs.Read(ki.Stream(), binary, true);  // true == add; doesn't matter here.
  }


  {  // Update GMMs.
    BaseFloat objf_impr, count;
    BaseFloat tot_like = gmm_accs.TotLogLike(),
        tot_t = gmm_accs.TotCount();
      
    MleAmDiagGmmUpdate(gmm_opts, gmm_accs, update_flags, &am_gmm,
                       &objf_impr, &count);
      
  }

  if (mixup != 0 || mixdown != 0 || !occs_out_filename.empty()) {
    // get pdf occupation counts
    Vector<BaseFloat> pdf_occs;
    pdf_occs.Resize(gmm_accs.NumAccs());
      
    for (int i = 0; i < gmm_accs.NumAccs(); i++)
      pdf_occs(i) = gmm_accs.GetAcc(i).occupancy().Sum();

    if (mixdown != 0)
      am_gmm.MergeByCount(pdf_occs, mixdown, power, min_count);

    // 提升高斯总数
    if (mixup != 0)
      am_gmm.SplitByCount(pdf_occs, mixup, perturb_factor,
                          power, min_count);

    // 将pdf-occs pdf-id统计数写入文件中. 一般不写入.
    if (!occs_out_filename.empty()) {
      bool binary = false;
      WriteKaldiObject(pdf_occs, occs_out_filename, binary);
    }
  }

  {
    Output ko(model_out_filename, binary_write);
    trans_model.Write(ko.Stream(), binary_write);
    am_gmm.Write(ko.Stream(), binary_write);
  }

  KALDI_LOG << "Written model to " << model_out_filename;
  return 0;
}





// am_gmm.SplitByCount(pdf_occs, mixup, perturb_factor, power, min_count);
void AmDiagGmm::SplitByCount(const Vector<BaseFloat> &state_occs,
                             int32 target_components,
                             float perturb_factor, BaseFloat power,
                             BaseFloat min_count) {

  int32 gauss_at_start = NumGauss();
  std::vector<int32> targets;
  // 将所有pdf-id 按照目标的高斯总数 进行平均拆分， 拆分结果 -> targets<pdf-id <num_gauss>>
  GetSplitTargets(state_occs, target_components, power,
                  min_count, &targets);

  // mix-up 实际将DiagGmm进行划分 DiagGmm->Split(targetCnt)
  for (int32 i = 0; i < NumPdfs(); i++) {
    if (densities_[i]->NumGauss() < targets[i])
      densities_[i]->Split(targets[i], perturb_factor);
  }
}



// in:
// state_occs  所有DiagGMM(pdf-id)出现次数
// target_components 目标高斯总数
// power 计算用.
// min_count 20
// out:
// targets <pdf-id<guass-cnt>>


void GetSplitTargets(const Vector<BaseFloat> &state_occs,
                     int32 target_components,
                     BaseFloat power,
                     BaseFloat min_count,
                     std::vector<int32> *targets) {
  // 使用优先队列方式进行.
  std::priority_queue<CountStats> split_queue;
  // pdf-id 总数
  int32 num_pdfs = state_occs.Dim();

  // 为(pdf-index, 1, occ) 构建一个结构体, 初始化每个pdf-id都只有一个高斯分量.
  for (int32 pdf_index = 0; pdf_index < num_pdfs; pdf_index++) {
    BaseFloat occ = pow(state_occs(pdf_index), power);
    // initialize with one Gaussian per PDF, to put a floor
    // of 1 on the #Gauss
    split_queue.push(CountStats(pdf_index, 1, occ));
  }

  // 当前高斯分量总数 -- 增长 直到 到达目标高斯分量总数
  for (int32 num_gauss = num_pdfs; num_gauss < target_components;) {
    CountStats state_to_split = split_queue.top();
    split_queue.pop();

    // 当前pdf-id 的统计总数
    BaseFloat orig_occ = state_occs(state_to_split.pdf_index);
    
    if ((state_to_split.num_components+1) * min_count >= orig_occ) {
      state_to_split.occupancy = 0; // min-count active -> disallow splitting
      // this state any more by setting occupancy = 0.
    } else {
      // 将当前pdf-id的高斯组成+1
      state_to_split.num_components++;
      num_gauss++;
    }
    // 在放入到优先队列 等待继续划分.
    split_queue.push(state_to_split);
  }
  
  targets->resize(num_pdfs);  
  while (!split_queue.empty()) {
    int32 pdf_index = split_queue.top().pdf_index;
    int32 pdf_tgt_comp = split_queue.top().num_components;
    // 将每个pdf-id 的高斯数 放入到target中。
    (*targets)[pdf_index] = pdf_tgt_comp;
    split_queue.pop();
  }
}


void DiagGmm::Split(int32 target_components, float perturb_factor,
                    std::vector<int32> *history) {


  int32 current_components = NumGauss(), dim = Dim();
  DiagGmm *tmp = new DiagGmm;
  tmp->CopyFromDiagGmm(*this);  // so we have copies of matrices
  // First do the resize:
  weights_.Resize(target_components);
  weights_.Range(0, current_components).CopyFromVec(tmp->weights_);
  means_invvars_.Resize(target_components, dim);
  means_invvars_.Range(0, current_components, 0, dim).CopyFromMat(
      tmp->means_invvars_);
  inv_vars_.Resize(target_components, dim);
  inv_vars_.Range(0, current_components, 0, dim).CopyFromMat(tmp->inv_vars_);
  gconsts_.Resize(target_components);

  delete tmp;

  // future work(arnab): Use a priority queue instead?
  while (current_components < target_components) {
    BaseFloat max_weight = weights_(0);
    int32 max_idx = 0;
    for (int32 i = 1; i < current_components; i++) {
      if (weights_(i) > max_weight) {
        max_weight = weights_(i);
        max_idx = i;
      }
    }

    // remember what component was split
    if (history != NULL)
      history->push_back(max_idx);

    weights_(max_idx) /= 2;
    weights_(current_components) = weights_(max_idx);
    Vector<BaseFloat> rand_vec(dim);
    for (int32 i = 0; i < dim; i++) {
      rand_vec(i) = RandGauss() * std::sqrt(inv_vars_(max_idx, i));
      // note, this looks wrong but is really right because it's the
      // means_invvars we're multiplying and they have the dimension
      // of an inverse standard variance. [dan]
    }
    inv_vars_.Row(current_components).CopyFromVec(inv_vars_.Row(max_idx));
    means_invvars_.Row(current_components).CopyFromVec(means_invvars_.Row(
        max_idx));
    means_invvars_.Row(current_components).AddVec(perturb_factor, rand_vec);
    means_invvars_.Row(max_idx).AddVec(-perturb_factor, rand_vec);
    current_components++;
  }
  
  ComputeGconsts();
}


  

// x=1
// while [ $x -lt $num_iters ]; do
//   echo "$0: Pass $x"
//   if [ $stage -le $x ]; then
//     if echo $realign_iters | grep -w $x >/dev/null; then
//       echo "$0: Aligning data"
// ============= 执行对齐 utt -- <tid1, tid2, tid2, tid3>
//       mdl="gmm-boost-silence --boost=$boost_silence `cat $lang/phones/optional_silence.csl` $dir/$x.mdl - |"
//       $cmd JOB=1:$nj $dir/log/align.$x.JOB.log \
//         gmm-align-compiled $scale_opts --beam=$beam --retry-beam=$[$beam*4] --careful=$careful "$mdl" \
//         "ark:gunzip -c $dir/fsts.JOB.gz|" "$feats" "ark,t:|gzip -c >$dir/ali.JOB.gz" \
//         || exit 1;
//     fi
// ============= 统计pdf-id 出现次数 MFCC统计量等
//     $cmd JOB=1:$nj $dir/log/acc.$x.JOB.log \
//       gmm-acc-stats-ali  $dir/$x.mdl "$feats" "ark:gunzip -c $dir/ali.JOB.gz|" \
//       $dir/$x.JOB.acc || exit 1;
// ============ 估计gmm 参数
//     $cmd $dir/log/update.$x.log \
//       gmm-est --write-occs=$dir/$[$x+1].occs --mix-up=$numgauss --power=$power $dir/$x.mdl \
//       "gmm-sum-accs - $dir/$x.*.acc|" $dir/$[$x+1].mdl || exit 1;
//     rm $dir/$x.mdl $dir/$x.*.acc $dir/$x.occs 2>/dev/null
//   fi
//   if [ $x -le $max_iter_inc ]; then
//      numgauss=$[$numgauss+$incgauss];
//   fi
//   beam=10
//   x=$[$x+1]
// done

// 生成final.mdl
// ( cd $dir; rm final.{mdl,occs} 2>/dev/null; ln -s $x.mdl final.mdl; ln -s $x.occs final.occs )



// steps/diagnostic/analyze_alignments.sh --cmd "$cmd" $lang $dir
// utils/summarize_warnings.pl $dir/log

// steps/info/gmm_dir_info.pl $dir

// echo "$0: Done training monophone system in $dir"

// exit 0

// # example of showing the alignments:
// # show-alignments data/lang/phones.txt $dir/30.mdl "ark:gunzip -c $dir/ali.0.gz|" | head -4

