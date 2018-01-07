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
  ComputeTuples(ctx_dep);
  
  ComputeDerived();
  
  InitializeProbs();
  
  Check();
}

void TransitionModel::ComputeTuplesIsHmm(const ContextDependencyInterface &ctx_dep) {
  const std::vector<int32> &phones = topo_.GetPhones();
  KALDI_ASSERT(!phones.empty());

  // this is the case for normal models. but not fot chain models
  std::vector<std::vector<std::pair<int32, int32> > > pdf_info;
  std::vector<int32> num_pdf_classes( 1 + *std::max_element(phones.begin(), phones.end()), -1);
  for (size_t i = 0; i < phones.size(); i++)
    num_pdf_classes[phones[i]] = topo_.NumPdfClasses(phones[i]);
  ctx_dep.GetPdfInfo(phones, num_pdf_classes, &pdf_info);
  // pdf_info is list indexed by pdf of which (phone, pdf_class) it
  // can correspond to.

  std::map<std::pair<int32, int32>, std::vector<int32> > to_hmm_state_list;
  // to_hmm_state_list is a map from (phone, pdf_class) to the list
  // of hmm-states in the HMM for that phone that that (phone, pdf-class)
  // can correspond to.
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

  for (int32 pdf = 0; pdf < static_cast<int32>(pdf_info.size()); pdf++) {
    for (size_t j = 0; j < pdf_info[pdf].size(); j++) {
      int32 phone = pdf_info[pdf][j].first,
            pdf_class = pdf_info[pdf][j].second;
      const std::vector<int32> &state_vec = to_hmm_state_list[std::make_pair(phone, pdf_class)];
      KALDI_ASSERT(!state_vec.empty());
      // state_vec is a list of the possible HMM-states that emit this
      // pdf_class.
      for (size_t k = 0; k < state_vec.size(); k++) {
        int32 hmm_state = state_vec[k];
        tuples_.push_back(Tuple(phone, hmm_state, pdf, pdf));
      }
    }
  }
}

void TransitionModel::ComputeDerived() {
  state2id_.resize(tuples_.size()+2);  // indexed by transition-state, which
  // is one based, but also an entry for one past end of list.

  int32 cur_transition_id = 1;
  num_pdfs_ = 0;
  for (int32 tstate = 1;
      tstate <= static_cast<int32>(tuples_.size()+1);  // not a typo.
      tstate++) {
    state2id_[tstate] = cur_transition_id;
    if (static_cast<size_t>(tstate) <= tuples_.size()) {
      int32 phone = tuples_[tstate-1].phone,
          hmm_state = tuples_[tstate-1].hmm_state,
          forward_pdf = tuples_[tstate-1].forward_pdf,
          self_loop_pdf = tuples_[tstate-1].self_loop_pdf;
      num_pdfs_ = std::max(num_pdfs_, 1 + forward_pdf);
      num_pdfs_ = std::max(num_pdfs_, 1 + self_loop_pdf);
      const HmmTopology::HmmState &state = topo_.TopologyForPhone(phone)[hmm_state];
      int32 my_num_ids = static_cast<int32>(state.transitions.size());
      cur_transition_id += my_num_ids;  // # trans out of this state.
    }
  }

  id2state_.resize(cur_transition_id);   // cur_transition_id is #transition-ids+1.
  id2pdf_id_.resize(cur_transition_id);
  for (int32 tstate = 1; tstate <= static_cast<int32>(tuples_.size()); tstate++)
    for (int32 tid = state2id_[tstate]; tid < state2id_[tstate+1]; tid++) {
      id2state_[tid] = tstate;
      if (IsSelfLoop(tid))
        id2pdf_id_[tid] = tuples_[tstate-1].self_loop_pdf;
      else
        id2pdf_id_[tid] = tuples_[tstate-1].forward_pdf;
    }
}

void TransitionModel::InitializeProbs() {
  log_probs_.Resize(NumTransitionIds()+1);  // one-based array, zeroth element empty.
  for (int32 trans_id = 1; trans_id <= NumTransitionIds(); trans_id++) {
    int32 trans_state = id2state_[trans_id];
    int32 trans_index = trans_id - state2id_[trans_state];
    const Tuple &tuple = tuples_[trans_state-1];
    const HmmTopology::TopologyEntry &entry = topo_.TopologyForPhone(tuple.phone);
    KALDI_ASSERT(static_cast<size_t>(tuple.hmm_state) < entry.size());
    BaseFloat prob = entry[tuple.hmm_state].transitions[trans_index].second;
    if (prob <= 0.0)
      KALDI_ERR << "TransitionModel::InitializeProbs, zero "
          "probability [should remove that entry in the topology]";
    if (prob > 1.0)
      KALDI_WARN << "TransitionModel::InitializeProbs, prob greater than one.";
    log_probs_(trans_id) = Log(prob);
  }
  ComputeDerivedOfProbs();
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
