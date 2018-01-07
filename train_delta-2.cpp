// gmm-init-model  --write-occs=$dir/1.occs  $dir/tree $dir/treeacc $lang/topo $dir/1.mdl

void gmm_init_model(){
  using namespace kaldi;
  using namespace kaldi;
  typedef kaldi::int32 int32;

  const char *usage =
      "Initialize GMM from decision tree and tree stats\n"
      "Usage:  gmm-init-model [options] <tree-in> <tree-stats-in> <topo-file> <model-out> [<old-tree> <old-model>]\n"
      "e.g.: \n"
      "  gmm-init-model tree treeacc topo 1.mdl\n"
      "or (initializing GMMs with old model):\n"
      "  gmm-init-model tree treeacc topo 1.mdl prev/tree prev/30.mdl\n";

  bool binary = true;
  double var_floor = 0.01;
  std::string occs_out_filename;


  ParseOptions po(usage);
  // 占用率, 计算 GMM的 ai ui var 需要的.
  po.Register("write-occs", &occs_out_filename, "File to write state occupancies to.");
    
  std::string
      tree_filename = po.GetArg(1),
      stats_filename = po.GetArg(2),
      topo_filename = po.GetArg(3),
      model_out_filename = po.GetArg(4),
      old_tree_filename = po.GetOptArg(5),
      old_model_filename = po.GetOptArg(6);

  ContextDependency ctx_dep;
  ReadKaldiObject(tree_filename, &ctx_dep);

  BuildTreeStatsType stats;
  {
    bool binary_in;
    GaussClusterable gc;  // dummy needed to provide type.
    Input ki(stats_filename, &binary_in);
    ReadBuildTreeStats(ki.Stream(), binary_in, gc, &stats);
  }
  
  KALDI_LOG << "Number of separate statistics is " << stats.size();

  HmmTopology topo;
  ReadKaldiObject(topo_filename, &topo);

  // 根据ctx_dep 对象得到 EventMap 对象指针.
  const EventMap &to_pdf = ctx_dep.ToPdfMap();  // not owned here.

  TransitionModel trans_model(ctx_dep, topo);
  
  // Now, the summed_stats will be used to initialize the GMM.
  AmDiagGmm am_gmm;
  if (old_tree_filename.empty())
    InitAmGmm(stats, to_pdf, &am_gmm, trans_model, var_floor);  // Normal case: initialize 1 Gauss/model from tree stats.
  else {
  }

  if (!occs_out_filename.empty()) {  // write state occs
    Vector<BaseFloat> occs;
    GetOccs(stats, to_pdf, &occs);
    Output ko(occs_out_filename, binary);
    occs.Write(ko.Stream(), binary);
  }

  // 将初始化的 trans-model am_gmm写入mdl文件.
  {
    Output ko(model_out_filename, binary);
    trans_model.Write(ko.Stream(), binary);
    am_gmm.Write(ko.Stream(), binary);
  }
}




// 根据状态绑定决策树 topo 构建TransModel

TransitionModel::TransitionModel(const ContextDependencyInterface &ctx_dep,
                                 const HmmTopology &hmm_topo): topo_(hmm_topo) {
  // First thing is to get all possible tuples.
  // trans-state --- tuple --- <phone, HMM-state, pdf-id>
  ComputeTuples(ctx_dep);

  // 生成 trans-state trans-id pdf-id互相之间的链接关系
  ComputeDerived();

  // 初始化 转移概率 trans-id probs
  InitializeProbs();
  
  Check();
}



// Tuples <phone, state, pdf-id>
// 1 state, 不同的state可能本身就绑定在同一个pdf-class
// 2 状态绑定决策树 中的EventType <-1,pdf-class> 绑定的是 pdf-class 多以不同的pdf-class 会绑定到同一个 pdf-id上.
// 3 Tuples<phone, state, pdf-id> 就能够完全描述一个使用某个pdf-id 的HMM-state
void TransitionModel::ComputeTuplesIsHmm(const ContextDependencyInterface &ctx_dep) {
  const std::vector<int32> &phones = topo_.GetPhones();

  // this is the case for normal models. but not fot chain models
  std::vector<std::vector<std::pair<int32, int32> > > pdf_info;
  // 1+ *std::max_element(phoens.begin(), phones.end()) --- 音素总数phone_cnt
  // vector<int32> num_pdf 构造函数,Create a vector  with phone_cnt elements of value -1;
  std::vector<int32> num_pdf_classes( 1 + *std::max_element(phones.begin(), phones.end()), -1);

  // 每个音素的 pdf-class总数(就是state数 一般 = 3, 可能存在绑定到相同pdf-class的状态会有不同)
  for (size_t i = 0; i < phones.size(); i++)
    num_pdf_classes[phones[i]] = topo_.NumPdfClasses(phones[i]);

  //获得 pdf-id 对应的 多个<phone, pdf-class> 
  ctx_dep.GetPdfInfo(phones, num_pdf_classes, &pdf_info);
  // pdf_info is list indexed by pdf of which (phone, pdf_class) it
  // can correspond to.

  // <phone, pdf-class>  map to  hmm-states. , 因为不同hmm-state 可能绑定到同一个 pdf-class
  std::map<std::pair<int32, int32>, std::vector<int32> > to_hmm_state_list;
  // to_hmm_state_list is a map from (phone, pdf_class) to the list of hmm-states in the HMM
  // for that phone that (phone, pdf-class) can correspond to.
  for (size_t i = 0; i < phones.size(); i++) {  // setting up to_hmm_state_list.
    int32 phone = phones[i];
    const HmmTopology::TopologyEntry &entry = topo_.TopologyForPhone(phone);
    for (int32 j = 0; j < static_cast<int32>(entry.size()); j++) {  // for each state...
      int32 pdf_class = entry[j].forward_pdf_class;
      if (pdf_class != kNoPdf) {
        to_hmm_state_list[std::make_pair(phone, pdf_class)].push_back(j);
      }
    }
  }

  // foreach pdf-id
  for (int32 pdf = 0; pdf < static_cast<int32>(pdf_info.size()); pdf++) {
    // foreach pdf-id's <phone,pdf-class>
    for (size_t j = 0; j < pdf_info[pdf].size(); j++) {
      int32 phone = pdf_info[pdf][j].first,
            pdf_class = pdf_info[pdf][j].second;
      // <phone,pdf-class> 对应的多个hmm-states.
      const std::vector<int32> &state_vec = to_hmm_state_list[std::make_pair(phone, pdf_class)];
      KALDI_ASSERT(!state_vec.empty());
      // state_vec is a list of the possible HMM-states that emit this pdf_class.
      // tuples 保存的是 <phone, hmm-state, pdf-id>
      for (size_t k = 0; k < state_vec.size(); k++) {
        int32 hmm_state = state_vec[k];
        tuples_.push_back(Tuple(phone, hmm_state, pdf, pdf));
      }
    }
  }
}


// in:
// all phones
// 每个phone 包含的可能的pdf-class
// :
// 通过ctx状态绑定决策树, 获得 对应到一个pdf-id 的多个 <phone, pdf-class>
// out:
// 每个决策树pdf-id 可能对应的<phone, pdf-class>  (因为没考虑 上下文音素, 可能不同得到pdf-id 对应了 相同的<phone, pdf-clas>(因为他们的上下文不同))
void ContextDependency::GetPdfInfo(
    const std::vector<int32> &phones,
    const std::vector<int32> &num_pdf_classes,  // indexed by phone,
    std::vector<std::vector<std::pair<int32, int32> > > *pdf_info) const {

  EventType vec;
  // 多有可能的pdf-id
  pdf_info->resize(NumPdfs());
  // 从每个音素 进行统计 每个可能pdf-id会映射到 那个 <phone, pdf-class>
  for (size_t i = 0 ; i < phones.size(); i++) {
    int32 phone = phones[i];
    vec.clear();
    vec.push_back(std::make_pair(static_cast<EventKeyType>(P_),
                                 static_cast<EventValueType>(phone)));
    // Now get length -- pdf-class_cnt
    KALDI_ASSERT(static_cast<size_t>(phone) < num_pdf_classes.size());
    EventAnswerType len = num_pdf_classes[phone];

    // 音素内每个状态, 这里通过二元定位<-1, 1> 只包含中心音素 以及状态pdf-class, 找到多个可能pdf-id(根据不同上下文的多个不同pdf-id)
    for (int32 pos = 0; pos < len; pos++) {
      vec.resize(2);
      vec[0] = std::make_pair(static_cast<EventKeyType>(P_),
                              static_cast<EventValueType>(phone));
      vec[1] = std::make_pair(kPdfClass, static_cast<EventValueType>(pos));
      
      std::sort(vec.begin(), vec.end());
      std::vector<EventAnswerType> pdfs;  // pdfs that can be at this pos as this phone.
      // 不考虑 0, 2代表的left right 得到多个可能的pdf-id.
      to_pdf_->MultiMap(vec, &pdfs);
      SortAndUniq(&pdfs);
      //<所有可能的pdf-id  <所有可能映射到的 <phone, pos(pdf-class)>>>
      for (size_t j = 0; j < pdfs.size(); j++) {
        KALDI_ASSERT(static_cast<size_t>(pdfs[j]) < pdf_info->size());
        (*pdf_info)[pdfs[j]].push_back(std::make_pair(phone, pos));
      }
    }
  }
  // 对所有 pdf-id 可能的<phone,pdf-class> 进行排序.
  for (size_t i = 0; i < pdf_info->size(); i++) {
    std::sort( ((*pdf_info)[i]).begin(),  ((*pdf_info)[i]).end());
    KALDI_ASSERT(IsSortedAndUniq( ((*pdf_info)[i])));  // should have no dups.
  }
}



// 每个tuple <phone, HMM-state, pdf-id>被认为是个 tstate --- transition-state,
// 每个transition-state实际上是 HMM-State的扩展, 每个确定的Hmm-state.
// trans-state 实际上某个音素的某个状态. 因为状态没有全编号,只是phone内编号, trans-state相当于所有phone-state全编号的state.
// 可能不同的trans-state HMM-state不同但是 pdf-id相同.

// trans-id = HMM-state + trans-index, 一个HMM-state可能包含多个trans-index.
// state2id_ 就是每个trans-state 对应的多个trans-id的起始id.
// id2state_ 每个trans-id 对应的trans-satete
// id2pdf_id_ 每个trans-id 对应的pdf-id (对应tuple中保存这 )
void TransitionModel::ComputeDerived() {
  state2id_.resize(tuples_.size()+2);  // indexed by transition-state, which
  // is one based, but also an entry for one past end of list.

  int32 cur_transition_id = 1;
  num_pdfs_ = 0;
  // foreach tuple<phone, state, pdf-class>
  for (int32 tstate = 1;
      tstate <= static_cast<int32>(tuples_.size()+1);  // not a typo.
      tstate++) {

    // state2id_[1] = 1;???
    state2id_[tstate] = cur_transition_id;

    if (static_cast<size_t>(tstate) <= tuples_.size()) {
      
      int32
          phone = tuples_[tstate-1].phone,
          hmm_state = tuples_[tstate-1].hmm_state,
          forward_pdf = tuples_[tstate-1].forward_pdf,
          self_loop_pdf = tuples_[tstate-1].self_loop_pdf;
      
      num_pdfs_ = std::max(num_pdfs_, 1 + forward_pdf);
      num_pdfs_ = std::max(num_pdfs_, 1 + self_loop_pdf);

      // 某个HmmState 可能的转移 Transition 数(一般两三个)
      const HmmTopology::HmmState &state = topo_.TopologyForPhone(phone)[hmm_state];
      int32 my_num_ids = static_cast<int32>(state.transitions.size());
      cur_transition_id += my_num_ids;  // # trans out of this state.
    }
  }

  // cur_transition_id 就是trans-id 总数
  
  id2state_.resize(cur_transition_id);   // cur_transition_id is #transition-ids+1.
  id2pdf_id_.resize(cur_transition_id);
  // foreach trans-state
  for (int32 tstate = 1; tstate <= static_cast<int32>(tuples_.size()); tstate++)
    // foreach trans-state's trans-id
    for (int32 tid = state2id_[tstate]; tid < state2id_[tstate+1]; tid++) {
      // trans-id --- trans-state(tuple-id)的映射
      id2state_[tid] = tstate;

      // 实际上self-loop-pdf 和 forward-pdf都保存相等的id了.
      // trans-id --- pdf-id
      if (IsSelfLoop(tid))
        id2pdf_id_[tid] = tuples_[tstate-1].self_loop_pdf;
      else
        id2pdf_id_[tid] = tuples_[tstate-1].forward_pdf;
    }
}

// 根据topo 计算所有trans-id的转移概率log
// 并计算所有的非自环转移概率和的log
void TransitionModel::InitializeProbs() {
  // all the trans-id, 
  log_probs_.Resize(NumTransitionIds()+1);  // one-based array, zeroth element empty.
  
  for (int32 trans_id = 1; trans_id <= NumTransitionIds(); trans_id++) {
    int32 trans_state = id2state_[trans_id];
    int32 trans_index = trans_id - state2id_[trans_state];
    const Tuple &tuple = tuples_[trans_state-1];
    
    const HmmTopology::TopologyEntry &entry = topo_.TopologyForPhone(tuple.phone);
    BaseFloat prob = entry[tuple.hmm_state].transitions[trans_index].second;
    if (prob <= 0.0)
      KALDI_ERR << "TransitionModel::InitializeProbs, zero "
          "probability [should remove that entry in the topology]";
    if (prob > 1.0)
      KALDI_WARN << "TransitionModel::InitializeProbs, prob greater than one.";

    // 保存每个trans-id的trans_prob.
    log_probs_(trans_id) = Log(prob);
  }
  //计算所有trans-state的非自环转移概率和 的log
  ComputeDerivedOfProbs();
}

// 返回某个trans-state 的多个trans-index中self-loop的对应trans-id
// returns the self-loop transition-id,
int32 TransitionModel::SelfLoopOf(int32 trans_state) const {  
  const Tuple &tuple = tuples_[trans_state-1];
  // or zero if does not exist.
  int32
      phone = tuple.phone,
      hmm_state = tuple.hmm_state;
  const HmmTopology::TopologyEntry &entry = topo_.TopologyForPhone(phone);
  // foreach trans-index
  for (int32 trans_index = 0;
      trans_index < static_cast<int32>(entry[hmm_state].transitions.size());
      trans_index++)
    // check if a self-loop
    if (entry[hmm_state].transitions[trans_index].first == hmm_state)
      // 返回对应的 trans-id
      return PairToTransitionId(trans_state, trans_index);
  return 0;  // invalid transition id.
}
int32 TransitionModel::PairToTransitionId(int32 trans_state, int32 trans_index) const {
  // return trans-id
  return state2id_[trans_state] + trans_index;
}
// 计算每个trans-state <phone, HMM-state, pdf-id> 对应的多个转移中 非自环转移的概率的和的log值.
// non_self_loop_log_probs_ 保存每个trans-state 的非自环转移概率的和.
void TransitionModel::ComputeDerivedOfProbs() {
  non_self_loop_log_probs_.Resize(NumTransitionStates()+1);  // this array indexed
  // because transition-state with nothing in zeroth element.
  // foreach trans-state
  for (int32 tstate = 1; tstate <= NumTransitionStates(); tstate++) {
    // 返回state self-loop对应的 trans-id
    int32 tid = SelfLoopOf(tstate);
    if (tid == 0) {  // no self-loop
      non_self_loop_log_probs_(tstate) = 0.0;  // log(1.0)
    } else {
      BaseFloat
          // 某个trans-state 对应的self-loop概率 自环转移概率
          self_loop_prob = Exp(GetTransitionLogProb(tid)),
          // 那么non-self-loop概率为 1-self-loop. 非自环转移概率
          non_self_loop_prob = 1.0 - self_loop_prob;
      
      if (non_self_loop_prob <= 0.0) {
        KALDI_WARN << "ComputeDerivedOfProbs(): non-self-loop prob is " << non_self_loop_prob;
        non_self_loop_prob = 1.0e-10;  // just so we can continue...
      }
      // non_self_loop_log_probs_ 保存每个trans-state 对应的  非自环转移概率的log
      non_self_loop_log_probs_(tstate) = Log(non_self_loop_prob);  // will be negative.
    }
  }
}

void TransitionModel::Check() const {
  KALDI_ASSERT(NumTransitionIds() != 0 && NumTransitionStates() != 0);
  {
    int32 sum = 0;
    for (int32 ts = 1; ts <= NumTransitionStates(); ts++) sum += NumTransitionIndices(ts);
    KALDI_ASSERT(sum == NumTransitionIds());
  }
  for (int32 tid = 1; tid <= NumTransitionIds(); tid++) {
    int32 tstate = TransitionIdToTransitionState(tid),
        index = TransitionIdToTransitionIndex(tid);
    KALDI_ASSERT(tstate > 0 && tstate <=NumTransitionStates() && index >= 0);
    KALDI_ASSERT(tid == PairToTransitionId(tstate, index));
    int32 phone = TransitionStateToPhone(tstate),
        hmm_state = TransitionStateToHmmState(tstate),
        forward_pdf = TransitionStateToForwardPdf(tstate),
        self_loop_pdf = TransitionStateToSelfLoopPdf(tstate);
    KALDI_ASSERT(tstate == TupleToTransitionState(phone, hmm_state, forward_pdf, self_loop_pdf));
    KALDI_ASSERT(log_probs_(tid) <= 0.0 && log_probs_(tid) - log_probs_(tid) == 0.0);
    // checking finite and non-positive (and not out-of-bounds).
  }
}













/// InitAmGmm initializes the GMM with one Gaussian per state.
void InitAmGmm(const BuildTreeStatsType &stats,
               const EventMap &to_pdf_map,
               AmDiagGmm *am_gmm,
               const TransitionModel &trans_model,
               BaseFloat var_floor) {
  
  // Get stats split by tree-leaf ( == pdf):
  std::vector<BuildTreeStatsType> split_stats;
  SplitStatsByMap(stats, to_pdf_map, &split_stats);

  split_stats.resize(to_pdf_map.MaxResult() + 1); // ensure that
  // if the last leaf had no stats, this vector still has the right size.
  
  // Make sure each leaf has stats.
  for (size_t i = 0; i < split_stats.size(); i++) {
    if (split_stats[i].empty()) {
      // bad_pdfs -- (pdf-id) 这个pdf-id 没有统计量
      std::vector<int32> bad_pdfs(1, i), bad_phones;
      // 获取 这个没有MFCC统计量pdf-id 的 对应phone
      GetPhonesForPdfs(trans_model, bad_pdfs, &bad_phones);
      std::ostringstream ss;
      for (int32 idx = 0; idx < bad_phones.size(); idx ++)
        ss << bad_phones[idx] << ' ';
      KALDI_WARN << "Tree has pdf-id " << i 
          << " with no stats; corresponding phone list: " << ss.str();
      // 表示可能有 训练中未见的音素
      /*
        This probably means you have phones that were unseen in training 
        and were not shared with other phones in the roots file. 
        You should modify your roots file as necessary to fix this.
        (i.e. share that phone with a similar but seen phone on one line 
        of the roots file). Be sure to regenerate roots.int from roots.txt, 
        if using s5 scripts. To work out the phone, search for 
        pdf-id  i  in the output of show-transitions (for this model). */
    }
  }
    // 按照每个节点累加起来统计量
  std::vector<Clusterable*> summed_stats;
  SumStatsVec(split_stats, &summed_stats);
    // 累加起所有统计量
  Clusterable *avg_stats = SumClusterable(summed_stats);
  KALDI_ASSERT(avg_stats != NULL && "No stats available in gmm-init-model.");
    // foreach pdf-id stats
  for (size_t i = 0; i < summed_stats.size(); i++) {
    GaussClusterable *c =
        static_cast<GaussClusterable*>(summed_stats[i] != NULL ? summed_stats[i] : avg_stats);

    // 将MFCC统计量都加入到 am_gmm中.
    DiagGmm gmm(*c, var_floor);
    am_gmm->AddPdf(gmm);
    
    BaseFloat count = c->count();
    if (count < 100) {
      std::vector<int32> bad_pdfs(1, i), bad_phones;
      GetPhonesForPdfs(trans_model, bad_pdfs, &bad_phones);
      std::ostringstream ss;
      for (int32 idx = 0; idx < bad_phones.size(); idx ++)
        ss << bad_phones[idx] << ' ';
      KALDI_WARN << "Very small count for state " << i << ": " 
                 << count << "; corresponding phone list: " << ss.str();
    }
  }
  DeletePointers(&summed_stats);
  delete avg_stats;

DiagGmm::DiagGmm(const GaussClusterable &gc,
                 BaseFloat var_floor): valid_gconsts_(false) {
  Vector<BaseFloat> x (gc.x_stats());
  Vector<BaseFloat> x2 (gc.x2_stats());
  BaseFloat count =  gc.count();
  KALDI_ASSERT(count > 0.0);
  this->Resize(1, x.Dim());
  x.Scale(1.0/count);
  x2.Scale(1.0/count);
  x2.AddVec2(-1.0, x);  // subtract mean^2.
  x2.ApplyFloor(var_floor);
  x2.InvertElements();  // get inv-var.
  
  KALDI_ASSERT(x2.Min() > 0);
  Matrix<BaseFloat> mean(1, x.Dim());
  mean.Row(0).CopyFromVec(x);
  Matrix<BaseFloat> inv_var(1, x.Dim());
  inv_var.Row(0).CopyFromVec(x2);
  this->SetInvVarsAndMeans(inv_var, mean);
  Vector<BaseFloat> weights(1);
  weights(0) = 1.0;
  this->SetWeights(weights);
  this->ComputeGconsts();
}
}

/// Get state occupation counts. 占存率
void GetOccs(const BuildTreeStatsType &stats,
             const EventMap &to_pdf_map,
             Vector<BaseFloat> *occs) {

    // Get stats split by tree-leaf ( == pdf):
  std::vector<BuildTreeStatsType> split_stats;
  SplitStatsByMap(stats, to_pdf_map, &split_stats);
  
  if (split_stats.size() != to_pdf_map.MaxResult()+1) {
    KALDI_ASSERT(split_stats.size() < to_pdf_map.MaxResult()+1);
    split_stats.resize(to_pdf_map.MaxResult()+1);
  }
  
  // occs -- vector<>  pdf-id的对应的 MFCC统计量的出现次数.
  occs->Resize(split_stats.size());
  // 每个pdf-id 对应的所有 MFCC出现总数.
  for (int32 pdf = 0; pdf < occs->Dim(); pdf++)
    (*occs)(pdf) = SumNormalizer(split_stats[pdf]);
}

// 对某个统计量集合 进行汇总统计 统计量的Normalizer --- 实际就是统计计数 -- MFCC出现次数.
BaseFloat SumNormalizer(const BuildTreeStatsType &stats_in) {
  BaseFloat ans = 0.0;
  BuildTreeStatsType::const_iterator iter = stats_in.begin(), end = stats_in.end();
  for (; iter != end; ++iter) {
    Clusterable *cl = iter->second;
    if (cl != NULL) ans += cl->Normalizer();
  }
  return ans;
}

virtual BaseFloat Normalizer() const { return count_; }

