struct ComputationGraph {

  // cindexes[cindex_id] 就可获得 对应 cindex_id 的 Cindex对象.
  /// The mapping of cindex_id to Cindex.
  std::vector<Cindex> cindexes;

  /// For each Cindex this tells us whether it was provided as an input to the
  /// network.  This is necessary for a couple of reasons: firstly, the
  /// framework allows users to provide values for nodes of type kComponent
  /// (e.g. for RNN context).  Also, Cindexes for input nodes that were not
  /// provided by the user may be created during computation graph creation
  /// (although they will not be computable), and we need to distinguish these
  /// from the provided Cindexes.
  std::vector<bool> is_input;

  /// dependencies[cindex_id] gives you the list of other cindex_ids that this
  /// particular cindex_id directly depends on to compute it.  No repeats will
  /// be present.  Note, some of these dependencies may be optional
  /// dependencies; in early stages of compilation this will contain all
  /// "desired" inputs and later we will prune the dependencies contain just
  /// those that are used (which will vary depending on availability).
  std::vector<std::vector<int32> > dependencies;





  // 这个变量 是为了特定的 multi-segment 计算, 只在当为online操作创建计算时使用.
  /// This variable is only of particular interest in a 'multi-segment'
  /// computation, which is used while creating computations for 'online'
  /// operation (for the kind of situation where you provide some input; run the
  /// computation; get some output, provide some more input for larger 't'
  /// values, etc.).
  // 当前说的 segment 是一个连续的 cindex_ids 的范围.
  // 一个 segment_end 是一个通过segment的id, 和下一个segemnt的开始cindex_id 相等.
  // 在一个完整创建的 只有一个segment的 计算图中, 这里只会保存一个值, 等于cindex_ids的数量.
  // In this context, a 'segment' is a continuous range of
  /// cindex_ids, and a segment_end is one past the end of each segment, which
  /// is the same as the beginning of the next segment, if there is one.  In the
  /// case of a fully-created computation graph with only one segment, this will
  /// contain just one value which equals the number of cindex_ids.
  // 这个成员在正确排序计算时需要使用, 因为计算图自己并不会包含编码了segment顺序的依赖关系,
  // (及时确实包含了这样的依赖关系, 也会能真的用我们的scc方法在nnet网络构图时使用)
  /// This information is needed to correctly order the computation, because
  /// the computation graph itself does not contain dependencies that encode the
  /// ordering of segments (and even if it did contain those dependencies, it's
  /// not really compatible with the way we use the scc's in the graph structure
  /// of the network to order the computation).
  std::vector<int32> segment_ends;

  // 映射Cindex -> cindex_id, 如果不存在, 就增加,并设置对应的is_input标记
  // 并设置 is_new = true 返回给调用者.
  /// Maps a Cindex to an integer cindex_id.  If not present, then add it (with
  /// the corresponding "is_input" flag set to the value "input") and set
  /// *is_new to true.  If present, set is_new to false and return the existing
  /// cindex_id.
  int32 GetCindexId(const Cindex &cindex, bool is_input, bool *is_new);

  
  /// Const version of GetCindexId that does not add CindexIds.  It will return
  /// -1 if the Cindex is not present, and the user should check for this.
  int32 GetCindexId(const Cindex &cindex) const;

  /// This function renumbers the cindex-ids (but only those with index c >= start_cindex_id,
  // keeping only for which keep[c - start_cindex_id] is true.
  // The "keep" array must be the same size as this->cindexes.size() - start_cindex_id.
  void Renumber(int32 start_cindex_id, const std::vector<bool> &keep);


  /// This function, useful for debugging/visualization purposes,
  /// prints out a summary of the computation graph (which will take up
  /// multiple lines).
  /// Format is: [ cindex1 -> dep-cindex1 dep-cindex2 ] [ cindex2 -> dep-cindex3 dep-cindex4 ]
  /// showing each Cindex and the Cindexes it depends on.  cindexes from different network
  /// nodes are shown on different lines.
  void Print(std::ostream &os, const std::vector<std::string> &node_names);



  
 private:

  //  是对cindexes的一个翻转映射, cindex_to_cindex_id[Cindex] 就可获得 对应Cindex的 cindex_id
  
  /// Maps each Cindex to an integer cindex_id: reverse mapping of "cindexes".
  /// Must be accessed via the GetCindexId() functions.
  unordered_map<Cindex, int32, CindexHasher> cindex_to_cindex_id_;
};




ComputationGraphBuilder::ComputationGraphBuilder(
    const Nnet &nnet,
    ComputationGraph *graph):
    nnet_(nnet), request_(NULL), graph_(graph),
    current_distance_(-1) {
  KALDI_ASSERT(graph_->cindexes.empty() &&
               "ComputationGraphBuilder initialized with nonempty graph.");
}




void ComputationGraphBuilder::AddInputs() {
  int32 num_added = 0;

  // request.inputs 是request中所有的输入 NnetIo
  // foreach input NnetIo
  for (int32 i = 0; i < request_->inputs.size(); i++) {
    // 获得对应的 nnet中node-index( 一般都是 input-node 的 index)
    int32 n = nnet_.GetNodeIndex(request_->inputs[i].name);
    if (n == -1)
      KALDI_ERR << "Network has no input with name "
                << request_->inputs[i].name;

    // 获得input NnetIo 对应的 input-node
    // 判断类型.
    NodeType t = nnet_.GetNode(n).node_type;
    KALDI_ASSERT((t == kInput || t == kComponent) &&
                 "Inputs to graph only allowed for Input and Component nodes.");

    // 每个 input NnetIo 都具有 很多indexes 是多个 Matrix 样本,
    // 每个 index 都是一个frame样本
    // foreach index 
    for (int32 j = 0; j < request_->inputs[i].indexes.size(); j++) {
      Cindex cindex(n, request_->inputs[i].indexes[j]);
      bool is_input = true, is_new;
      int32 cindex_id = graph_->GetCindexId(cindex, is_input, &is_new);
      KALDI_ASSERT(is_new && "Input index seems to be listed more than once");
      AddCindexId(cindex_id, true, false);
      num_added++;
    }
  }
  KALDI_ASSERT(num_added > 0 && "AddInputToGraph: nothing to add.");
}

void ComputationGraphBuilder::AddOutputs() {
  int32 num_added = 0;
  for (int32 i = 0; i < request_->outputs.size(); i++) {
    int32 n = nnet_.GetNodeIndex(request_->outputs[i].name);
    if (n == -1)
      KALDI_ERR << "Network has no output with name "
                << request_->outputs[i].name;
    for (int32 j = 0; j < request_->outputs[i].indexes.size(); j++) {
      Cindex cindex(n, request_->outputs[i].indexes[j]);
      bool is_input = false, is_new;
      int32 cindex_id = graph_->GetCindexId(cindex, is_input, &is_new);
      KALDI_ASSERT(is_new && "Output index seems to be listed more than once");
      AddCindexId(cindex_id, false, true);
      num_added++;
    }
  }
  if (num_added == 0) {
    KALDI_ERR << "Cannot process computation request with no outputs";
  }
  current_distance_ = 0;
  // the calls to AddCindexId in this function will have added to next_queue_.
  KALDI_ASSERT(current_queue_.empty());
  current_queue_.swap(next_queue_);
}

void ComputationGraphBuilder::Compute(const ComputationRequest &request) {

  // graph_->segment_ends 是不是empty()?
  // 在这个函数调用之前 只有默认构造, 没有任何赋值, 应该是empty的啊.
  // !!!! 靠 前面的条件 是 request_ 不是入参 request!!! 是其原本成员
  // 这样的话 request_ 也是NULL.
  if (request_ != NULL && graph_->segment_ends.empty()) {
    // this check is relevant to multi-segment (i.e. online) computations.
    KALDI_ERR << "You are calling things in the wrong order: should be "
              << "Compute(), Prune(), Compute, Prune(), ...";
  }

  // 0
  int32 cur_segment_start = graph_->cindexes.size();
  // 获得request
  request_ = &request;
  
  AddInputs();
  AddOutputs();  // sets current_distance_ to 0.
  
  // max_distance for debugging, to detect infinite recursion.
  int32 max_distance = 10000;
  while (current_distance_ < max_distance) {
    BuildGraphOneIter();
    // only check rarely if we're running at low verbose level.
    if (GetVerboseLevel() >= 3 || RandInt(1,  (current_distance_ + 1)) == 1)
      Check(cur_segment_start);
    // TODO: come up with a scheme to delay when we call
    // UpdateAllComputableInfo().
    UpdateAllComputableInfo();
    if (current_queue_.empty()) // we're done.
      break;
  }
  if (current_distance_ == max_distance)
    KALDI_ERR << "Loop detected while building computation graph (bad "
              << "network topology?)";

  if (RandInt(1, 2 * (graph_->segment_ends.size() + 1)) == 1)
    Check(cur_segment_start);
}


void ComputationGraphBuilder::Check(int32 start_cindex_id) const {
  int32 num_cindex_ids = graph_->cindexes.size();
  for (int32 cindex_id = start_cindex_id; cindex_id < num_cindex_ids;
       cindex_id += 1 + RandInt(0, num_cindex_ids / 100)) {
    { // check depend_on_this.
      std::vector<int32> depend_on_this = depend_on_this_[cindex_id];
      int32 size = depend_on_this.size();
      std::sort(depend_on_this.begin(), depend_on_this.end());
      KALDI_ASSERT(IsSortedAndUniq(depend_on_this));
      for (size_t j = 0; j < size; j++) {
        int32 other_cindex_id = depend_on_this[j];
        // make sure appears in appropriate dependencies array.
        const std::vector<int32> &dep = graph_->dependencies[other_cindex_id];
        KALDI_ASSERT(std::count(dep.begin(), dep.end(), cindex_id) == 1);
      }
    }
    { // check dependencies.
      std::vector<int32> dependencies = graph_->dependencies[cindex_id];
      int32 size = dependencies.size();
      std::sort(dependencies.begin(), dependencies.end());
      KALDI_ASSERT(IsSortedAndUniq(dependencies));
      for (size_t j = 0; j < size; j++) {
        int32 dep_cindex_id = dependencies[j];
        if (dep_cindex_id >= start_cindex_id) {
          // make sure appears in appropriate depend_on_this_ array.
          const std::vector<int32> &dep = depend_on_this_[dep_cindex_id];
          KALDI_ASSERT(std::count(dep.begin(), dep.end(), cindex_id) == 1);
        }
      }
    }

    {
      // check usable_count_
      int32 node_index = graph_->cindexes[cindex_id].first;
      int32 usable_count = usable_count_[cindex_id],
          usable_count_recomputed = nnet_.IsOutputNode(node_index) ? 1 : 0;
      std::vector<int32> depend_on_this = depend_on_this_[cindex_id];
      int32 size = depend_on_this.size();
      for (size_t j = 0; j < size; j++) {
        int32 other_cindex_id = depend_on_this[j];
        if (usable_count_[other_cindex_id] != 0 &&
            computable_info_[other_cindex_id] != kNotComputable)
          usable_count_recomputed++;
      }
      KALDI_ASSERT(usable_count == usable_count_recomputed);
    }
    // check computable_info_.  note: this will not be accurate
    // if the cindex_id is still queued to have dependencies added
    // (in cur_queue_ or next_queue_).
    if (computable_queue_.empty()) {
      ComputationGraphBuilder::ComputableInfo c =
          ComputeComputableInfo(cindex_id);
      // the status doesn't have to be correct if it's kWillNotCompute,
      // because these are cindex-ids that we chose not to compute
      // because we determined they would not be useful, and
      // ComputeComputableInfo() will never return this value.
      if (c != computable_info_[cindex_id] &&
          computable_info_[cindex_id] != kWillNotCompute) {
        int32 count_cur = std::count(current_queue_.begin(),
                                     current_queue_.end(), cindex_id),
            count_next = std::count(next_queue_.begin(),
                                    next_queue_.end(), cindex_id);
        // if it wasn't queued, then something is wrong.
        if (count_cur + count_next == 0)
          KALDI_ERR << "Mismatch in computable status";
      }
    }
    // check computable_queued_.
    // note, the following checks might be a bit slow.
    if (computable_queued_[cindex_id]) {
      KALDI_ASSERT(std::count(computable_queue_.begin(),
                              computable_queue_.end(),
                              cindex_id) == 1);
    } else {
      KALDI_ASSERT(std::count(computable_queue_.begin(),
                              computable_queue_.end(),
                              cindex_id) == 0);
    }
  }
}





Compiler::Compiler(
    const ComputationRequest &request,
    const Nnet &nnet): nnet_(nnet) {
  requests_.push_back(&request);
}

// 根据 requests_ 构建 Computation.
void Compiler::CreateComputation(const CompilerOptions &opts,
                                 NnetComputation *computation) {
  computation->Clear();
  // ------------ ComputationGraphBuilder ------------------
  ComputationGraphBuilder builder(nnet_, &graph_);

  // ------------ 构建ComputationGraph ---------------
  for (size_t segment = 0; segment < requests_.size(); segment++) {
    builder.Compute(*(requests_[segment]));
    if (!builder.AllOutputsAreComputable()) {
      builder.ExplainWhyAllOutputsNotComputable();  // prints logging info
      KALDI_ERR << "Not all outputs were computable, cannot create computation.";
    }
    builder.Prune();
  }



  // -------------- 将ComputationGraph  分解为 phase -------------
  // 一个phase 会被分解为 一个或多个steps.
  // 对每个segment phase_per_segment 是phases的list, 每个phase 都是cindex_ids的list.
  // see function declaration's comment for more on the meaning of "phases" (a
  // phase will later be decomposed into one or more steps).  for each segment
  // s, phases_per_segment[s] is a list of phases; each phase is a list of
  // cindex_ids.
  std::vector<std::vector<std::vector<int32> > > phases_per_segment;
  ComputeComputationPhases(nnet_, graph_, &phases_per_segment);
  std::vector<std::vector<int32> > steps;
  steps.reserve(1000);

  // maps each step to the segment in which it appears.  in the normal case
  // (non-looped computation), a vector of all zeros.
  std::vector<int32> step_to_segment;


  {
    // note: this class will output to 'steps' and to 'cindex_id_to_location_'.
    // it may incidentally change 'graph_' by adding a few cindexes.
    ComputationStepsComputer steps_computer(nnet_, &graph_, &steps,
                                            &cindex_id_to_location_);

    for (size_t segment = 0; segment < requests_.size(); segment++) {
      steps_computer.ComputeForSegment(*(requests_[segment]),
                                       phases_per_segment[segment]);
      while (step_to_segment.size() < steps.size())
        step_to_segment.push_back(segment);

      // save memory, by deleting the phases we just consumed.  the
      // following two lines just exist to save memory.
      std::vector<std::vector<int32> > temp;
      phases_per_segment[segment].swap(temp);
    }
    steps_computer.Check();
  }
  std::vector<bool> deriv_needed;
  ComputeDerivNeeded(steps, step_to_segment, &deriv_needed);
  CreateStepInfo(deriv_needed, step_to_segment, &steps, computation);
  AddCommands(deriv_needed, step_to_segment, computation);
  // the following command reorders commands so kAcceptInput and kProvideOutput
  // appear in the desired places.
  ConsolidateIoOperations(nnet_, computation);
  if (opts.output_debug_info)
    OutputDebugInfo(computation);
}
