// steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" 2000 10000 data/mfcc/train data/lang exp/mono_ali exp/tri1

// StateId next_state = ofst->AddState();
// //以word标注为 inlabel, outlabel 权重=1 目标状态，构建转移弧
// Arc arc(labels[i], labels[i], Weight::One(), next_state);
// //增加转移弧, 
// ofst->AddArc(cur_state, arc);
// cur_state = next_state;
// 因为转移弧是有源状态 目标状态的， 所有只是简单的将弧放入到一个集合(Vector)
// 就可以描述一个图. 因此就是用这种数据结构表示 图FST。


# To be run from ..
# Flat start and monophone training, with delta-delta features.
# This script applies cepstral mean normalization (per speaker).

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
shared_phones_opt="--shared-phones=$lang/phones/sets.int"

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
// in:
// tree
// model
// L.fst
// 每utt标注信息
// out:
// 输出图 graphs-wspecifier
  
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

    std::vector<int32> disambig_syms;
    if (disambig_rxfilename != "")
      if (!ReadIntegerVectorSimple(disambig_rxfilename, &disambig_syms))
        KALDI_ERR << "fstcomposecontext: Could not read disambiguation symbols from "
                  << disambig_rxfilename;

    // 训练用FST图(每个utt一个简单图) 编译器
    // 保存了 trans-model, tree, lex-fst, disambig_symbals 
    TrainingGraphCompiler gc(trans_model, ctx_dep, lex_fst, disambig_syms, gopts);

    lex_fst = NULL;  // we gave ownership to gc.

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
        std::vector<fst::VectorFst<fst::StdArc>* > fsts;
        if (!gc.CompileGraphsFromText(transcripts, &fsts)) {
          KALDI_ERR << "Not expecting CompileGraphs to fail.";
        }
        
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
  // 根据标注, 构建batchsize 个fst, 每个状态是word.
  std::vector<const VectorFst<StdArc>* > word_fsts(transcripts.size());

  // foreach utt
  for (size_t i = 0; i < transcripts.size(); i++) {
    VectorFst<StdArc> *word_fst = new VectorFst<StdArc>();
    MakeLinearAcceptor(transcripts[i], word_fst);
    word_fsts[i] = word_fst;
  }    
  bool ans = CompileGraphs(word_fsts, out_fsts);
  for (size_t i = 0; i < transcripts.size(); i++)
    delete word_fsts[i];
  return ans;
}


// 根据word utt标注 生成一个fst图
// 因为utt的标注存在 是确定的一条路径 , 因此使用生成线性简图 fst
template<class Arc, class I>
void MakeLinearAcceptor(const vector<I> &labels, MutableFst<Arc> *ofst) {
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;

  ofst->DeleteStates();
  StateId cur_state = ofst->AddState();
  ofst->SetStart(cur_state);
  // foreach utt中的每个word label
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
// word_fst 以word为基本单元的基本简图
// out:
// out_fsts 生成对应的 以trans-id为基本的线性图.
bool TrainingGraphCompiler::CompileGraphs(
    const std::vector<const fst::VectorFst<fst::StdArc>* > &word_fsts,
    std::vector<fst::VectorFst<fst::StdArc>* > *out_fsts) {

  out_fsts->resize(word_fsts.size(), NULL);
  
  ContextFst<StdArc> *cfst = NULL;

  {  // make cfst [ it's expanded on the fly ]
    // all phones
    const std::vector<int32> &phone_syms = trans_model_.GetPhones();  // needed to create context fst.
    // phone_cnt + 销岐音素 得到总音素总数
    int32 subseq_symbol = phone_syms.back() + 1;
    if (!disambig_syms_.empty() && subseq_symbol <= disambig_syms_.back())
      subseq_symbol = 1 + disambig_syms_.back();

    // 上下文相关的音素图, 目的是为了和 phone2word图组合 构建三音素上下文相关图.
    // important but not read!!!!!!!!!!!!!!!!!!!!111
    cfst = new ContextFst<StdArc>(subseq_symbol,               // 音素总数
                                  phone_syms,                  // 所有实际因素
                                  disambig_syms_,              // 销岐符号
                                  ctx_dep_.ContextWidth(),     // 音素窗
                                  ctx_dep_.CentralPosition());  // 音素窗中位置
  }

  // 根据lex FST  以及每句的word FST 构建 phone2word FST
  // 根据上下文相关的cFst + phone2word Fst 构建一个 Context_phone2word.Fst
  // foreach utt word sequeitial
  for (size_t i = 0; i < word_fsts.size(); i++) {
    VectorFst<StdArc> phone2word_fst;
    
    // TableCompose more efficient than compose.
    // lex_fst 和 word_fst, 得到 phone2word_fst的一个图.
    TableCompose(*lex_fst_, *(word_fsts[i]), &phone2word_fst, &lex_cache_);

   
    VectorFst<StdArc> ctx2word_fst;
    ComposeContextFst(*cfst, phone2word_fst, &ctx2word_fst);
    // ComposeContextFst is like Compose but faster for this particular Fst type.
    // [and doesn't expand too many arcs in the ContextFst.]

    // 结果得到一个 上下文相关的 phone2word的fst图.
    (*out_fsts)[i] = ctx2word_fst.Copy();
    // For now this contains the FST with symbols
    // representing phones-in-context.
  }

  HTransducerConfig h_cfg;
  h_cfg.transition_scale = opts_.transition_scale;

  std::vector<int32> disambig_syms_h;
  // ?????????????????????????????????????????????????
  VectorFst<StdArc> *H = GetHTransducer(cfst->ILabelInfo(),
                                        ctx_dep_,
                                        trans_model_,
                                        h_cfg,
                                        &disambig_syms_h);

  for (size_t i = 0; i < out_fsts->size(); i++) {
    VectorFst<StdArc> &ctx2word_fst = *((*out_fsts)[i]);
    VectorFst<StdArc> trans2word_fst;
    TableCompose(*H, ctx2word_fst, &trans2word_fst);

    DeterminizeStarInLog(&trans2word_fst);

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

    KALDI_ASSERT(trans2word_fst.Start() != kNoStateId);

    *((*out_fsts)[i]) = trans2word_fst;
  }

  delete H;
  delete cfst;
  return true;
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

if [ $stage -le 0 ]; then
  gmm-est --min-gaussian-occupancy=3  --mix-up=$numgauss --power=$power \
    $dir/0.mdl "gmm-sum-accs - $dir/0.*.acc|" $dir/1.mdl 2> $dir/log/update.0.log || exit 1;
  rm $dir/0.*.acc
fi


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

    tcfg.Register(&po);
    gmm_opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

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

      if (mixup != 0)
        am_gmm.SplitByCount(pdf_occs, mixup, perturb_factor,
                            power, min_count);

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
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}




  

beam=6 # will change to 10 below after 1st pass
# note: using slightly wider beams for WSJ vs. RM.
x=1
while [ $x -lt $num_iters ]; do
  echo "$0: Pass $x"
  if [ $stage -le $x ]; then
    if echo $realign_iters | grep -w $x >/dev/null; then
      echo "$0: Aligning data"
      mdl="gmm-boost-silence --boost=$boost_silence `cat $lang/phones/optional_silence.csl` $dir/$x.mdl - |"
      $cmd JOB=1:$nj $dir/log/align.$x.JOB.log \
        gmm-align-compiled $scale_opts --beam=$beam --retry-beam=$[$beam*4] --careful=$careful "$mdl" \
        "ark:gunzip -c $dir/fsts.JOB.gz|" "$feats" "ark,t:|gzip -c >$dir/ali.JOB.gz" \
        || exit 1;
    fi
    $cmd JOB=1:$nj $dir/log/acc.$x.JOB.log \
      gmm-acc-stats-ali  $dir/$x.mdl "$feats" "ark:gunzip -c $dir/ali.JOB.gz|" \
      $dir/$x.JOB.acc || exit 1;

    $cmd $dir/log/update.$x.log \
      gmm-est --write-occs=$dir/$[$x+1].occs --mix-up=$numgauss --power=$power $dir/$x.mdl \
      "gmm-sum-accs - $dir/$x.*.acc|" $dir/$[$x+1].mdl || exit 1;
    rm $dir/$x.mdl $dir/$x.*.acc $dir/$x.occs 2>/dev/null
  fi
  if [ $x -le $max_iter_inc ]; then
     numgauss=$[$numgauss+$incgauss];
  fi
  beam=10
  x=$[$x+1]
done

( cd $dir; rm final.{mdl,occs} 2>/dev/null; ln -s $x.mdl final.mdl; ln -s $x.occs final.occs )


steps/diagnostic/analyze_alignments.sh --cmd "$cmd" $lang $dir
utils/summarize_warnings.pl $dir/log

steps/info/gmm_dir_info.pl $dir

echo "$0: Done training monophone system in $dir"

exit 0

# example of showing the alignments:
# show-alignments data/lang/phones.txt $dir/30.mdl "ark:gunzip -c $dir/ali.0.gz|" | head -4

