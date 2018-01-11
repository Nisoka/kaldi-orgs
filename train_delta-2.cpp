// gmm-init-model  --write-occs=$dir/1.occs  $dir/tree      $dir/treeacc  $lang/topo      $dir/1.mdl
//                    w pdf-id占存率         ctx_dep决策树  pdf-id统计量  基本topo结构    w 目标转移模型

//1  根据ctx_dep决策树(pdf-id) topo结构(phone,Hmm-State,pdf-class) 构建初始化的转移模型-trans-model（没有需要统计量?）
//2  根据统计量初始化AmDiagGmm (通过统计量 对每个pdf-id构建一个DiagGmm，然后综合得到 AmDiagGmm)
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
  // 占用率, 后面需要.
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
    // 将统计量 按照 ctx_dep的叶子节点进行再统计, 得到每个pdf-id的占存率。
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













// InitAmGmm initializes the GMM with one Gaussian per state.
// 1 将统计量stats 按照ctx_dep tree的pdf叶子节点
// 2 为每个Pdf叶子节点 构建一个DiagGmm(*c, var_floor)
// 3 将所有DiagGmm加入到AmDiagGmm中.
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

  // =======================
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
}

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






/// Get state occupation counts. 占存率
void GetOccs(const BuildTreeStatsType &stats,
             const EventMap &to_pdf_map,
             Vector<BaseFloat> *occs) {

  // Get stats split by tree-leaf ( == pdf-id):
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













// gmm-mixup --mix-up=$numgauss $dir/1.mdl $dir/1.occs    $dir/1.mdl 
//            目标高斯总数      转移模型   pdf-id占存率   新转移模型
// 从每个初始化的DiagGmm(pdf-id)的原本只有一个高斯分量，进行高斯分量扩充, pdf-id总数不变，但是为pdf-id增加高斯分量
// 实现增加高斯数 提高拟合能力.
int gmm_mixup(int argc, char *argv[]) {

    const char *usage =
        "Does GMM mixing up (and Gaussian merging)\n"
        "Usage:  gmm-mixup [options] <model-in> <state-occs-in> <model-out>\n"
        "e.g. of mixing up:\n"
        " gmm-mixup --mix-up=4000 1.mdl 1.occs 2.mdl\n"
        "e.g. of merging:\n"
        " gmm-mixup --merge=2000 1.mdl 1.occs 2.mdl\n";
        
    bool binary_write = true;
    int32 mixup = 0;
    int32 mixdown = 0;
    BaseFloat perturb_factor = 0.01;
    BaseFloat power = 0.2;
    BaseFloat min_count = 20.0;
    
    ParseOptions po(usage);
    po.Register("mix-up", &mixup, "Increase number of mixture components to "
                "this overall target.");

    std::string
        model_in_filename = po.GetArg(1),
        occs_in_filename = po.GetArg(2),
        model_out_filename = po.GetArg(3);

    AmDiagGmm am_gmm;
    TransitionModel trans_model;
    {
      bool binary_read;
      Input ki(model_in_filename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_gmm.Read(ki.Stream(), binary_read);
    }

    if (mixup != 0 || mixdown != 0) {

      Vector<BaseFloat> occs;
      ReadKaldiObject(occs_in_filename, &occs);
      if (occs.Dim() != am_gmm.NumPdfs())
        KALDI_ERR << "Dimension of state occupancies " << occs.Dim()
                   << " does not match num-pdfs " << am_gmm.NumPdfs();

      if (mixdown != 0)
        am_gmm.MergeByCount(occs, mixdown, power, min_count);

      // 进行高斯数目扩充 见train_mono.cpp
      if (mixup != 0)
        am_gmm.SplitByCount(occs, mixup, perturb_factor, power, min_count);
    }

    {
      Output ko(model_out_filename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
      am_gmm.Write(ko.Stream(), binary_write);
    }
}





//   将对齐信息从 原本trans-id的对齐结果 转化为 tree绑定后的 trans-id。
//   echo "$0: converting alignments from $alidir to use current tree"
//   convert-ali $alidir/final.mdl $dir/1.mdl $dir/tree   \
//      "ark:gunzip -c $alidir/ali.JOB.gz|" "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1;





// gmm-align-compiled $scale_opts --beam=$beam --retry-beam=$retry_beam --careful=$careful "$mdl" \
//                               beam                                                       转移模型
//                       "ark:gunzip -c $dir/fsts.JOB.gz|"    "$feats"        \
//                           输入 utt-fst线性图                 MFCC feature特征
//                       "ark:|gzip -c >$dir/ali.JOB.gz"
//                           输出 utt-trans-id 对齐信息

// gmm-align-compiled 按照编译图 进行对齐
int gmm_align_compiled(int argc, char *argv[]) {
    const char *usage =
        "Align features given [GMM-based] models.\n"
        "Usage:   gmm-align-compiled [options] <model-in> <graphs-rspecifier> "
        "<feature-rspecifier> <alignments-wspecifier> [scores-wspecifier]\n"
        "e.g.: \n"
        " gmm-align-compiled 1.mdl ark:graphs.fsts scp:train.scp ark:1.ali\n"
        "or:\n"
        " compile-train-graphs tree 1.mdl lex.fst 'ark:sym2int.pl -f 2- words.txt text|' \\\n"
        "   ark:- | gmm-align-compiled 1.mdl ark:- scp:train.scp t, ark:1.ali\n";

    ParseOptions po(usage);
    AlignConfig align_config;
    BaseFloat acoustic_scale = 1.0;
    BaseFloat transition_scale = 1.0;
    BaseFloat self_loop_scale = 1.0;
    std::string per_frame_acwt_wspecifier;

    std::string
        model_in_filename = po.GetArg(1),
        fst_rspecifier = po.GetArg(2),
        feature_rspecifier = po.GetArg(3),
        alignment_wspecifier = po.GetArg(4),
        scores_wspecifier = po.GetOptArg(5);

    TransitionModel trans_model;
    AmDiagGmm am_gmm;
    {
      bool binary;
      Input ki(model_in_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_gmm.Read(ki.Stream(), binary);
    }

    SequentialTableReader<fst::VectorFstHolder> fst_reader(fst_rspecifier);
    RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);
    Int32VectorWriter alignment_writer(alignment_wspecifier);
    BaseFloatWriter scores_writer(scores_wspecifier);
    BaseFloatVectorWriter per_frame_acwt_writer(per_frame_acwt_wspecifier);

    int num_done = 0, num_err = 0, num_retry = 0;
    double tot_like = 0.0;
    kaldi::int64 frame_count = 0;

    // foreach utt 
    for (; !fst_reader.Done(); fst_reader.Next()) {
      
      std::string utt = fst_reader.Key();
      if (!feature_reader.HasKey(utt)) {
        num_err++;
        KALDI_WARN << "No features for utterance " << utt;
      } else {
        // utt-MFCC & utt-fst
        const Matrix<BaseFloat> &features = feature_reader.Value(utt);
        VectorFst<StdArc> decode_fst(fst_reader.Value());
        fst_reader.FreeCurrent();  // this stops copy-on-write of the fst
        // by deleting the fst inside the reader, since we're about to mutate
        // the fst by adding transition probs.


        // Add transition-probs to the FST.
        // 在FST上增加转移概率
        {  
          std::vector<int32> disambig_syms;  // empty.
          AddTransitionProbs(trans_model, disambig_syms,
                             transition_scale, self_loop_scale,
                             &decode_fst);
        }

        // 解码用 统计量: amm_gmm, trans_model, MFCC-features, 声学拉伸.
        DecodableAmDiagGmmScaled gmm_decodable(am_gmm, trans_model, features,
                                               acoustic_scale);

        AlignUtteranceWrapper(align_config, utt,
                              acoustic_scale, &decode_fst, &gmm_decodable,
                              &alignment_writer, &scores_writer,
                              &num_done, &num_err, &num_retry,
                              &tot_like, &frame_count, &per_frame_acwt_writer);
      }
    }
}





// 对齐utt的包装 --   utt --- trans-id
void AlignUtteranceWrapper(
    const AlignConfig &config,
    const std::string &utt,
    BaseFloat acoustic_scale,  // affects scores written to scores_writer, if
                               // present
    fst::VectorFst<fst::StdArc> *fst,  // non-const in case config.careful ==
                                       // true.
    DecodableInterface *decodable,  // not const but is really an input.
    Int32VectorWriter *alignment_writer,
    BaseFloatWriter *scores_writer,
    int32 *num_done,
    int32 *num_error,
    int32 *num_retried,
    double *tot_like,
    int64 *frame_count,
    BaseFloatVectorWriter *per_frame_acwt_writer) {


  fst::StdArc::Label special_symbol = 0;
  if (config.careful)
    ModifyGraphForCarefulAlignment(fst);

  FasterDecoderOptions decode_opts;
  decode_opts.beam = config.beam;

  //1 根据图fst，构建解码器decoder
  //2 利用decoder 对解码统计量(utt的特征等信息) 进行解码
  FasterDecoder decoder(*fst, decode_opts);
  decoder.Decode(decodable);

  // 获得解码是否正常达到终止状态.
  bool ans = decoder.ReachedFinal();  // consider only final states.

  // 如果没有正常到达终止状态, 重新解码
  if (!ans && config.retry_beam != 0.0) {
    if (num_retried != NULL) (*num_retried)++;
    KALDI_WARN << "Retrying utterance " << utt << " with beam "
               << config.retry_beam;
    decode_opts.beam = config.retry_beam;
    decoder.SetOptions(decode_opts);
    // 重新解码
    decoder.Decode(decodable);
    ans = decoder.ReachedFinal();
  }


  // 获得最佳路径, 没有什么lattice.
  fst::VectorFst<LatticeArc> decoded;  // linear FST.
  decoder.GetBestPath(&decoded);
  
  if (decoded.NumStates() == 0) {
    KALDI_WARN << "Error getting best path from decoder (likely a bug)";
    if (num_error != NULL) (*num_error)++;
    return;
  }

  // alignment -- trans-id, 是 一个FST的 ilabel
  // words                       是解码的 olabel -- 解码结果
  std::vector<int32> alignment;
  std::vector<int32> words;
  LatticeWeight weight;
  // 获得解码结果
  GetLinearSymbolSequence(decoded, &alignment, &words, &weight);
  BaseFloat like = -(weight.Value1()+weight.Value2()) / acoustic_scale;

  if (num_done != NULL) (*num_done)++;
  if (tot_like != NULL) (*tot_like) += like;
  if (frame_count != NULL) (*frame_count) += decodable->NumFramesReady();

  if (alignment_writer != NULL && alignment_writer->IsOpen())
    alignment_writer->Write(utt, alignment);

  if (scores_writer != NULL && scores_writer->IsOpen())
    scores_writer->Write(utt, -(weight.Value1()+weight.Value2()));

  Vector<BaseFloat> per_frame_loglikes;
  if (per_frame_acwt_writer != NULL && per_frame_acwt_writer->IsOpen()) {
    GetPerFrameAcousticCosts(decoded, &per_frame_loglikes);
    per_frame_loglikes.Scale(-1 / acoustic_scale);
    per_frame_acwt_writer->Write(utt, per_frame_loglikes);
  }
}

// 解码过程！
void FasterDecoder::Decode(DecodableInterface *decodable) {
  InitDecoding();
  // 遍历解码统计量直到最后一帧
  while (!decodable->IsLastFrame(num_frames_decoded_ - 1)) {
    // 按帧进行解码, 解码当前帧可能的 ilabel!=0 的转移弧
    double weight_cutoff = ProcessEmitting(decodable);
    // 解码ilabel==0 的转移弧.
    ProcessNonemitting(weight_cutoff);
  }
}

void FasterDecoder::InitDecoding() {
  // clean up from last time:
  // 清除上次解码结果
  ClearToks(toks_.Clear());
  
  StateId start_state = fst_.Start();
  // 构建一个目标状体为fst_其实状态节点的 转移弧
  Arc dummy_arc(0, 0, Weight::One(), start_state);
  // 从起始节点开始进行 扩展式解码 -- token解码， 解码起点 就是 fst_图的起始状态节点
  // (FasterDecoder 是用于训练用的, 而训练时的解码fst图 是线性图.)
  toks_.Insert(start_state, new Token(dummy_arc, NULL));
  // 处理无激发状态 ilabel==0 需要看构图部分 ilabel 是 trans-id
  ProcessNonemitting(std::numeric_limits<float>::max());
  num_frames_decoded_ = 0;
}

// HashList<StateId, Token*> toks_;  toks_ 是一个hash列表 <stateid, token>
//          key      Token
// 每次 toks_.Inser<new_stateID, Token> key 都是 Token内的目标状态.
// ProcessNonemitting 是处理的 arc.ialbel==0 的转移, 是chapter04最后描述的算法, epsilon 认为是子词word的边界。
// 但是 ProcessNonemitting ProcessEmitting 都会向toks_中增加token 是怎么个意思?

// TODO: first time we go through this, could avoid using the queue.
void FasterDecoder::ProcessNonemitting(double cutoff) {
  // Processes nonemitting arcs for one frame. 
  KALDI_ASSERT(queue_.empty());
  // 从toks_ 解码扩展 中取出当前需要进行扩展解码的节点Elem 加入queue 进行扩展解码.
  for (const Elem *e = toks_.GetList(); e != NULL;  e = e->tail)
    queue_.push_back(e->key);
  
  while (!queue_.empty()) {
    StateId state = queue_.back();
    queue_.pop_back();
    Token *tok = toks_.Find(state)->val;  // would segfault if state not
    // in toks_ but this can't happen.
    if (tok->cost_ > cutoff) { // Don't bother processing successors.
      continue;
    }
   
    // fst中从状态节点state出发的所有弧
    for (fst::ArcIterator<fst::Fst<Arc> > aiter(fst_, state);
         !aiter.Done();
         aiter.Next()) {
      const Arc &arc = aiter.Value();

      //  ===============================================
      // 传播 无激发 -- ilabel 是trans-id.
      // 那么什么时候trans-id==0？这都是经过了fst的各种优化操作之后进行的,
      // 所以当一个arc.ilabel == 0时, 代表的是word边界, 所以arc.ilabel=0 ,就是解码一个word.
      // 而对于一个epsilon转移, 可直接发生转移, 而不用考虑生成概率！！！。
      // propagate nonemitting only...
      if (arc.ilabel == 0) {  
        // 根据可能的 弧，源token节点
        // 构建新token节点, 以源token节点中fst状态ID为源状态 并arc弧的目标状态作为新token节点的状态.
        Token *new_tok = new Token(arc, tok);

        // 剪枝 prune, 根据前面计算的激发状态扩展解码得到的一个 剪枝权重.
        if (new_tok->cost_ > cutoff) {  // prune
          Token::TokenDelete(new_tok);
          
        } else {
          // 在toks 中查找是否存在了新建token的终止状态
          Elem *e_found = toks_.Find(arc.nextstate);
          // 不存在就插入新的token。 并认为可以继续进行扩展解码.
          if (e_found == NULL) {
            toks_.Insert(arc.nextstate, new_tok);
            queue_.push_back(arc.nextstate);
          } else {
            // 操作符重载
            // inline bool operator < (const Token &other) {
            //   return cost_ > other.cost_;
            //    cost_ 保存的是路径下的前向累计权重, 权重越大说明解码正确性越大.
            // }
            if ( *(e_found->val) < *new_tok ) {
              Token::TokenDelete(e_found->val);
              e_found->val = new_tok;
              queue_.push_back(arc.nextstate);
            } else {
              Token::TokenDelete(new_tok);
            }
          }
        }
      }
    }
  }
}

// ProcessEmitting returns the likelihood cutoff used.
double FasterDecoder::ProcessEmitting(DecodableInterface *decodable) {
  int32 frame = num_frames_decoded_;
  // HashList, 只不过内部的HashList转出给last_toks. HashList内部进行一下清理, 因为可能需要修改Hash大小.
  Elem *last_toks = toks_.Clear();
  size_t tok_cnt;
  BaseFloat adaptive_beam;
  Elem *best_elem = NULL;

  
  double weight_cutoff = GetCutoff(last_toks, &tok_cnt,
                                   &adaptive_beam, &best_elem);
  KALDI_VLOG(3) << tok_cnt << " tokens active.";
  PossiblyResizeHash(tok_cnt);  // This makes sure the hash is always big enough.
    
  // This is the cutoff we use after adding in the log-likes (i.e.
  // for the next frame).  This is a bound on the cutoff we will use
  // on the next frame.
  double next_weight_cutoff = std::numeric_limits<double>::infinity();
  
  // First process the best token to get a hopefully
  // reasonably tight bound on the next cutoff.
  if (best_elem) {
    StateId state = best_elem->key;
    Token *tok = best_elem->val;
    for (fst::ArcIterator<fst::Fst<Arc> > aiter(fst_, state);
         !aiter.Done();
         aiter.Next()) {
      const Arc &arc = aiter.Value();
      if (arc.ilabel != 0) {  // we'd propagate..
        BaseFloat ac_cost = - decodable->LogLikelihood(frame, arc.ilabel);
        double new_weight = arc.weight.Value() + tok->cost_ + ac_cost;
        if (new_weight + adaptive_beam < next_weight_cutoff)
          next_weight_cutoff = new_weight + adaptive_beam;
      }
    }
  }

  // int32 n = 0, np = 0;

  
  // 遍历toks_ 保存的上一时间保存的状态节点, 在本时间帧对 特征进行解码, 判断最佳可能转移.
  // 上一时间保存的状态节点有多个,每个都是上一个时间的可能状态, 此时要对每一个状态点 再进行扩展解码
  // 生成更多可能，然后将上一个时间的点(本转移的源状态)从toks_中去掉. 一个token 实际上是一个转移, 所有有了
  // 最后最优的一个token 就能够通过每次保存的Prevtoken 找到最佳路径.
  for (Elem *e = last_toks, *e_tail; e != NULL; e = e_tail) { 
    StateId state = e->key;
    Token *tok = e->val;
   if (tok->cost_ < weight_cutoff) {
      
      // foreach fst中以state 为源状态的 arc
      for (fst::ArcIterator<fst::Fst<Arc> > aiter(fst_, state);
           !aiter.Done();
           aiter.Next()) {
        Arc arc = aiter.Value();
        // 进行扩展解码 说明不是epsilon??? 是个词内转移?
        // 具体需要看chapter04 的具体描述, 实际上应该是 WFST状态内的HMM-state转移, 经过优化
        // WFST状态内部, 没有了epsilon（arc.ilabel != 0），因此可以通过arc.ilabel 是否为0，
        // 决定是否是一个HMM-state转移.  只有正常的HMM-state转移才会需要计算转移概率, 而对于WFST状态转移, 则只需要保存token
        if (arc.ilabel != 0) {  
          // 计算当前帧 是 该转移弧输入标签(HMM-state?)的概率 -- ac_cost().
          BaseFloat ac_cost =  - decodable->LogLikelihood(frame, arc.ilabel);
          // 当前Viterbi权重
          double new_weight = arc.weight.Value() + tok->cost_ + ac_cost;

          // 依旧不发生剪枝
          if (new_weight < next_weight_cutoff) {  // not pruned..
            // 从当前token 构建转移弧, 并保存 该路径权重.
            Token *new_tok = new Token(arc, ac_cost, tok);
            Elem *e_found = toks_.Find(arc.nextstate);
            // 重新计算剪枝权重线
            if (new_weight + adaptive_beam < next_weight_cutoff)
              next_weight_cutoff = new_weight + adaptive_beam;
            
            if (e_found == NULL) {
              toks_.Insert(arc.nextstate, new_tok);
            } else {
              // 判断两条路径的权重, 留下更好的.
              if ( *(e_found->val) < *new_tok ) {
                Token::TokenDelete(e_found->val);
                e_found->val = new_tok;
              } else {
                Token::TokenDelete(new_tok);
              }
            }
          }
        }
      }
    }

    
   // 遍历下一个上时间状态节点, 并将已经扩展传播了的上时间节点 从toks_中删除.
   // 因为已经扩展传播了, toks_中保存的是当前时间需要进行扩展传播的节点.
   e_tail = e->tail;
   Token::TokenDelete(e->val);
   toks_.Delete(e);
  }
  
  num_frames_decoded_++;
  return next_weight_cutoff;
}


// 根据FST图 扩展解码 构建下一个token
// in:
// FST arc
// 上一个token节点, 对应是在FST图 arc弧的源状态
inline Token(const Arc &arc, Token *prev):
    arc_(arc), prev_(prev), ref_count_(1) {
  if (prev) {
    prev->ref_count_++;
    weight_ = Times(prev->weight_, arc.weight);
  } else {
    weight_ = arc.weight;
  }
}


// 解码器判断是否解码正常.
// 存在一条解码路径达到终止状态
bool FasterDecoder::ReachedFinal() {
  // toks_保存的是 上时间 解码路径的结果状态节点, 可以进行继续传播扩展解码, 或者直接获得当前最佳路径.
  // 遍历所有当前解码结果.
  for (const Elem *e = toks_.GetList(); e != NULL; e = e->tail) {
    // cost_ 代价正常 并且 是个FST终止状态.
    if (e->val->cost_ != std::numeric_limits<double>::infinity() &&
        fst_.Final(e->key) != Weight::Zero())
      return true;
  }
  return false;
}



bool FasterDecoder::GetBestPath(fst::MutableFst<LatticeArc> *fst_out,
                                bool use_final_probs) {
  // GetBestPath gets the decoding output.  If "use_final_probs" is true
  // AND we reached a final state, it limits itself to final states;
  // otherwise it gets the most likely token not taking into
  // account final-probs.  fst_out will be empty (Start() == kNoStateId) if
  // nothing was available.  It returns true if it got output (thus, fst_out
  // will be nonempty).
  
  fst_out->DeleteStates();
  Token *best_tok = NULL;
  bool is_final = ReachedFinal();
  if (!is_final) {
    for (const Elem *e = toks_.GetList(); e != NULL; e = e->tail)
      if (best_tok == NULL || *best_tok < *(e->val) )
        best_tok = e->val;
  } else {
    double infinity =  std::numeric_limits<double>::infinity(),
        best_cost = infinity;
    
    for (const Elem *e = toks_.GetList(); e != NULL; e = e->tail) {
      double this_cost = e->val->cost_ + fst_.Final(e->key).Value();
      // 找到最佳的 cost_
      if (this_cost < best_cost && this_cost != infinity) {
        best_cost = this_cost;
        best_tok = e->val;
      }
    }
  }
  
  if (best_tok == NULL) return false;  // No output.

  std::vector<LatticeArc> arcs_reverse;  // arcs in reverse order.

  // 回溯 最优的Token， 找到最优路径的reverse（翻转）
  for (Token *tok = best_tok; tok != NULL; tok = tok->prev_) {
    BaseFloat
        tot_cost = tok->cost_ -  (tok->prev_ ? tok->prev_->cost_ : 0.0),
        graph_cost = tok->arc_.weight.Value(),
        ac_cost = tot_cost - graph_cost;
    
    LatticeArc l_arc(tok->arc_.ilabel,
                     tok->arc_.olabel,
                     LatticeWeight(graph_cost, ac_cost),
                     tok->arc_.nextstate);
    arcs_reverse.push_back(l_arc);
  }
  
  KALDI_ASSERT(arcs_reverse.back().nextstate == fst_.Start());
  arcs_reverse.pop_back();  // that was a "fake" token... gives no info.

  StateId cur_state = fst_out->AddState();
  fst_out->SetStart(cur_state);
  
  for (ssize_t i = static_cast<ssize_t>(arcs_reverse.size())-1; i >= 0; i--) {
    LatticeArc arc = arcs_reverse[i];
    arc.nextstate = fst_out->AddState();
    fst_out->AddArc(cur_state, arc);
    cur_state = arc.nextstate;
  }
  
  if (is_final && use_final_probs) {
    Weight final_weight = fst_.Final(best_tok->arc_.nextstate);
    fst_out->SetFinal(cur_state, LatticeWeight(final_weight.Value(), 0.0));
  } else {
    fst_out->SetFinal(cur_state, LatticeWeight::One());
  }
  RemoveEpsLocal(fst_out);
  return true;
}




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
