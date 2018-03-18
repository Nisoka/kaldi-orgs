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

  // dependencies 描述 某个 cindex_id 需要的依赖 vector<cindex_id>
  
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

  // ================== unordered_map ================
  // 可以通过.insert()函数, 向unordered_map 中增加元素, 增加元素时,会返回<maptype::iterator, bool>
  // bool 表示是否是新加入数据
  // maptype::iterator 表示插入的位置?
  //  是对cindexes的一个翻转映射, cindex_to_cindex_id[Cindex] 就可获得 对应Cindex的 cindex_id
  unordered_map<Cindex, int32, CindexHasher> cindex_to_cindex_id_;
};



// 向ComputationGraph中增加 Cindex
// cindex_id = cindexes.size()(位置就是id)
// cindexes.push_back
// is_input.push_back
// dependences.resize(size() + 1) --- 为新的cindex_id 的依赖 构建空间.
int32 ComputationGraph::GetCindexId(const Cindex &cindex,
                                    bool input, bool *is_new) {
  typedef unordered_map<Cindex, int32, CindexHasher> map_type;
  int32 new_index = cindexes.size();  // we'll add this if we don't find it.

  //unorder_map< 
  std::pair<map_type::iterator, bool> p;
  p = cindex_to_cindex_id_.insert(std::pair<Cindex, int32>(cindex, new_index));
  
  if (p.second == true) {  // We added something to the hash.
    *is_new = true;
    KALDI_ASSERT(is_input.size() == cindexes.size());
    cindexes.push_back(cindex);
    is_input.push_back(input);
    // make room for this "dependencies" entry.
    dependencies.resize(new_index + 1);
    return new_index;
  } else { // We did not add anything.
    *is_new = false;
    return p.first->second;
  }
}




/// An abstract representation of a set of Cindexes.
/// See \ref dnn3_compile_graph_building.
class ComputationGraphBuilder {
 public:
  ComputationGraphBuilder(const Nnet &nnet,
                          ComputationGraph *graph);

  // Does the initial computation (populating the graph and computing whether
  // each required cindex_id is computable), without the pruning.  In the normal
  // case you call this just once with one 'request', but in the 'online' case
  // you call Compute() [then maybe check AllOutputsAreComputable()] then
  // Prune() multiple times, with a sequence of different requests for
  // increasing time values.
  // Note: it sets the class member request_ to the address of 'request', so
  // you should not let 'request' go out of scope while this class might
  // still use it (e.g. until you call Compute() with a different
  void Compute(const ComputationRequest &request);

  
  // This enum says for each cindex_id, whether we can compute it from the given
  // inputs or not.  Note that there may be situations where before adding
  // dependencies of a particular cindex_id we realize that we won't be able to
  // use this cindex_id (i.e. it may be computable but it's not used) because
  // its usable_count is zero, and in those cases we change the status to
  // kWillNotCompute even though the cindex-id may be computable- for most
  // purposes this status is treated the same as kNotComputable.
  enum ComputableInfo {
    kUnknown = 0,
    kComputable = 1,
    kNotComputable = 2,
    kWillNotCompute = 3
  };

  const Nnet &nnet_;
  const ComputationRequest *request_;
  ComputationGraph *graph_;
  // this is the transpose of graph_->dependencies; it tells us
  // for each cindex_id, which other cindex_ids depend on it.
  std::vector<std::vector<int32> > depend_on_this_;
  // this vector, indexed by cindex_id, contains our information about whether
  // each cindex_id is computable; it's ComputableInfo, cast to char.
  std::vector<char> computable_info_;
  // this is a queue of cindex_ids that we need to re-compute whether they are
  // computable or not (because either they are new and haven't had dependencies
  // added, or their dependencies' computable status has changed since we last
  // computed their computable_ value).
  std::deque<int32> computable_queue_;
  // this vector tells us whether a cindex_id is in computable_queued_; it
  // stops us from adding things twice.
  std::vector<bool> computable_queued_;

  // 一个usable_count[cindex_id] > 0 cindex_id  被认为是可用, 意味着它 可能参与输出output的计算
  // usable_count_[cindex_id] = 1; if cindex_id 是一个 request_.output
  // usable_count_[cindex_id] = ?? 
  // output, and otherwise as the number of other cindex_ids j such that
  // computable_info_[j] is not kNotComputable AND usable_count_[j] > 0 AND i is
  // a member of graph->dependencies[j].
  // 这个变量被设计用来简单能够保持更新 随着我们增加cindex_ids???
  // This quantity is designed to be easy to keep updated as we add cindex_ids.
  std::vector<int32> usable_count_;

  // ?????
  // current_distance_ >= 0 is the distance to the output, of the cindex_ids in
  // current_queue_.
  int32 current_distance_;
  
  // 没有计算依赖的cindex_ids. (那些需要依赖, 但是还没有计算其依赖的cindex_ids)
  // the cindex_ids in current_queue_ are at distance "current_distance" to the
  // output and have not yet had their dependencies processed.
  std::vector<int32> current_queue_;

  // 当前距离+1的位置上的cindex_ids，
  // 并且还没有计算依赖的cindex_ids. (那些需要依赖, 但是还没有计算其依赖的cindex_ids)
  // the cindex_ids in next_queue_ are at distance current_distance + 1 to the
  // output and have not yet had their dependencies processed.
  std::vector<int32> next_queue_;
};






ComputationGraphBuilder::ComputationGraphBuilder(
    const Nnet &nnet,
    ComputationGraph *graph):
    nnet_(nnet), request_(NULL), graph_(graph),
    current_distance_(-1) {
  KALDI_ASSERT(graph_->cindexes.empty() &&
               "ComputationGraphBuilder initialized with nonempty graph.");
}





// input的cindex-id
//   computable_info_.push(kComputable)
//   computable_queued_.push(false)
// output的cindex-id
//   computable_info.push_back(kUnknown)
//   computable_queued.push_back(false)
//   next_queue_.push_back(cindex_id)

// depend_on_this_.push(?) --- 表示依赖于此cindex_id 的其他cindex的id
// usable_count_ 是否??
void ComputationGraphBuilder::AddCindexId(int32 cindex_id,
                                          bool is_input,
                                          bool is_output) {
  // If this cindex_id has just now been added to the graph, the following
  // assert should succeed.
  KALDI_PARANOID_ASSERT(cindex_id == computable_queued_.size() &&
                        cindex_id == computable_info_.size() &&
                        cindex_id == depend_on_this_.size() &&
                        cindex_id == usable_count_.size());

  // input_node (input-MFCC input-Ivector)
  if (is_input) {
    computable_info_.push_back(kComputable);
    computable_queued_.push_back(false);
    // other -- 一般就是 output
  } else {
    // 
    computable_info_.push_back(kUnknown);
    // cindex_ids 是否加入了 computable_queue_ 的标记.
    computable_queued_.push_back(false);
    // next_queue_ 保存的是 output 输出cindex-id
    next_queue_.push_back(cindex_id);
  }
  depend_on_this_.push_back(std::vector<int32>());

  // output:1    others:0(一般就是input-MFCC 和 input-Ivector)
  usable_count_.push_back(is_output ? 1 : 0);
}

// 将request_ 中的input中的数据描述NnetIo-indexes
// 转化为
//  graph_ 中的Cindexes描述
void ComputationGraphBuilder::AddInputs() {
  int32 num_added = 0;

  // request.inputs 是request中所有的输入 NnetIo

  //  ----------- ivector 在init.config 中是 input_node的, 并且在通过 nnet3-init 中构建的节点是
  //  ------- kInput 类型的.
  // foreach input & ivector NnetIo
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

    // ================= 每个  index 对应 一个 Cindex =============
    // Request
    // 1 inputs
    //     input   - NnetIo(merged) --- indexes(merged)
    //     ivector - NnetIo(merged) --- indexes(merged)
    // 2 outputs
    //     output - NnetIo(merged) --- indexes
    // 每个 NnetIo 都具有 很多indexes 都代表多个 Matrix 样本,
    // 每个 index 都是一个frame样本
    // 实际上 每个Request 就对应了 一个minibatch的训练数据(包含输入,输出)
    // foreach index 
    for (int32 j = 0; j < request_->inputs[i].indexes.size(); j++) {
      Cindex cindex(n, request_->inputs[i].indexes[j]);
      bool is_input = true, is_new;
      // 向ComputeGraph 中增加Cindex, 实际上就是将原本的indexes 转化为 Cindex描述 数据.
      int32 cindex_id = graph_->GetCindexId(cindex, is_input, &is_new);
      
      KALDI_ASSERT(is_new && "Input index seems to be listed more than once");
      AddCindexId(cindex_id, true, false);
      num_added++;
    }
  }
  KALDI_ASSERT(num_added > 0 && "AddInputToGraph: nothing to add.");
}
// 为label 构建 cindexes 描述数据.
void ComputationGraphBuilder::AddOutputs() {
  int32 num_added = 0;
  // foreach output NnetIo
  for (int32 i = 0; i < request_->outputs.size(); i++) {
    // 找到对应该的NnetIo 的 nnet网络节点 NnetworkNode.
    int32 n = nnet_.GetNodeIndex(request_->outputs[i].name);
    
    if (n == -1)
      KALDI_ERR << "Network has no output with name "
                << request_->outputs[i].name;

    // =================== 每个index 对应一个 cindex ============
    // 为该NnetIo中的所有 frames数据的label 构建cindex
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




  // 后面 buildGraph过程需要用到如下消息.
  // ??????
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

  // ===================== 将request.input[i].indexes request.output[i].indexes 转化为Cindex =====================
  //        都转化为Cindex 加入到 ComputationGraph  graph_的cindexes成员中, 对应修改dependences等代表性成员
  //        
  AddInputs();
  AddOutputs();  // sets current_distance_ to 0.!!!!!!!!!!!!!!!!!!


  // 为了检测 无限循环.
  // max_distance for debugging, to detect infinite recursion.
  int32 max_distance = 10000;
  while (current_distance_ < max_distance) {

    BuildGraphOneIter();

    void ComputationGraphBuilder::BuildGraphOneIter() {
      while (!current_queue_.empty()) {
        int32 cindex_id = current_queue_.back();
        current_queue_.pop_back();
        
        KALDI_ASSERT(computable_info_[cindex_id] == kUnknown);


        
        // input Cindexes MFCC IVECTOR.
        if (usable_count_[cindex_id] == 0){

          // 将输入的cindex_id GraphBuilder->computable_info_[cindex_id] = kWillNotCompute 不需要计算.
          // 将其depend_on_this_ 的其他cindex_id
          // 1 都加入  需要计算队列-computable_queue_ 
          // 2 并设置  需要计算信息-computable_info_ 为未知(需要看是否是output的依赖)
          //  ----- BuildGraph算法中的 正向方向处理 ------------
          SetAsWillNotCompute(cindex_id);
          // 对cindex_id 设置其对应标记 为 不需要进行计算.
          // 然后对 依赖于cindex_id的所有其他 other_cindex_id 设置为
          // 需要重新计算他们是否需要计算 Computable. (加入到需要计算队列 computable_queue_)
          void ComputationGraphBuilder::SetAsWillNotCompute(int32 cindex_id) {
            KALDI_ASSERT(usable_count_[cindex_id] == 0);

            // Computabel_info_ 描述 cindex_id  是否可以从input数据计算得到, 一开始只有
            // Input数据是kComputable, 其他都是kUnknown
            
            // depend_on_this_ 描述 依赖 cindex_id 的其他所有Cindex.

            // computable_queue_  保存那些需要重新计算Computable的cindex_id
            // computable_queued_ 保存cindex_id是否已经加入到computable_queue_ 的标记.
            computable_info_[cindex_id] = kWillNotCompute;
            std::vector<int32>::const_iterator
                iter = depend_on_this_[cindex_id].begin(),
                end = depend_on_this_[cindex_id].end();
            
            for (; iter != end; ++iter) {
              int32 other_cindex_id = *iter;
              // 未知是否需要计算, && 还没加入到 需要计算队列 -- computable_queue_中. 
              if (computable_info_[other_cindex_id] == kUnknown && !computable_queued_[other_cindex_id]) {
                computable_queue_.push_back(other_cindex_id);
                computable_queued_[other_cindex_id] = true;
              }
            }
          }
        }
        // output Cindex
        else{
          AddDependencies(cindex_id);
          
          // Add cindex_ids that this cindex_id depends on.
          void ComputationGraphBuilder::AddDependencies(int32 cindex_id) {

            // graph_dependencies.size() 是 Cindex总数
            // 目的是类似哈希表 保存长度足够.
            if (static_cast<int32>(graph_->dependencies.size()) <= cindex_id) {
              graph_->dependencies.resize(2 * cindex_id + 1);
            }

            // 获得对应的Cindex
            Cindex cindex = graph_->cindexes[cindex_id];

            // ====================== 查找 cindex的依赖. =================
            // Cindex -- <node_index, Index>

            // find the dependencies of this cindex.
            int32 node_index = cindex.first;
            const Index &index = cindex.second;
            const NetworkNode &node = nnet_.GetNode(node_index);
           
            // 将当前cindex_id 的依赖找到
            std::vector<Cindex> input_cindexes;

            // ===================== 计算某个 Cindex 的依赖 --- input_cindexes =============
            // 根据node.node_type : kDescriptor, kComponent, kInput,等通过node查找对应的依赖 cindex.
            // the following switch statement sets up "input_cindexes".
            switch (node.node_type) {
              case kDescriptor: {
                // desc describes how this node obtains its input from other nodes.
                const Descriptor &desc = node.descriptor;
                desc.GetDependencies(index, &input_cindexes);

                // 内部是通过 parts_ 中对每个子Descriptor都调用 GetDependencies
                // 对简单的子Descriptor 直接 构建上一个<src_node_, index> ==> Cindex(src_node_, index);
                // 这样相当于 某个<node_i, (n, t, x)> 依赖于<node_i-1, (n, t, x)>
                // 对比较复杂的Descriptor 就会 引用多个<src_node_, index> 会形成依赖列表.

                
                break;
              }
              case kComponent: {
                int32 c = node.u.component_index;
                const Component *component = nnet_.GetComponent(c);
                std::vector<Index> input_indexes;
                // 直接返回自身????? 
                component->GetInputIndexes(request_->misc_info, index, &input_indexes);
                // 然后对自身构建一个 Cindex<node_index-1, index> 实际上直接使用上一个节点的cindex.---- 对应的kDescriptor节点.
                // 这就很正常了, 因为实际上这列依赖关系 复杂性都应该放入到kDescriptor中的component-node中实现.
                input_cindexes.resize(input_indexes.size());
                for (size_t i = 0; i < input_indexes.size(); i++) {
                  input_cindexes[i].first = node_index  - 1;  // preceding node
                  input_cindexes[i].second = input_indexes[i];
                }
                break;
              }
              case kDimRange: {
                input_cindexes.resize(1);
                input_cindexes[0] = Cindex(node.u.node_index, index);
                break;
              }
              case kInput:
                break;  // There will be no dependencies.
              default:
                KALDI_ERR << "Invalid node type";
            }

            int32 num_dependencies = input_cindexes.size();

            // reserve 语句目的是保证在下面循环中 如下的引用不会变得无效
            // graph_GetCindexId()调用会增加 num_dependences个元素到 graph_->dependencies, 并且这么做能够避免申请空间
            // (the call to graph_->GetCindexId() could add up to
            // num_dependencies elements to the graph_->dependencies array and we want to avoid allocation).
            // RoundUpToNearestPowerOfTwo 是为了高效, 避免太频繁的重新设置大小
            // RoundUpToNearestPowerOfTwo 直接将数字扩大好多倍.
            graph_->dependencies.reserve(RoundUpToNearestPowerOfTwo(
                graph_->dependencies.size() +  num_dependencies));


            // ================ 向 graph_.dependences[cindex_id] 中加入依赖cindex. ===============
            // 引用graph_ 中的cindex_id 的 dependences
            std::vector<int32> &this_dep = graph_->dependencies[cindex_id];
            this_dep.resize(num_dependencies);
            for (size_t i = 0; i < num_dependencies; i++) {
              bool is_input = false, is_new;
              int32 dep_cindex_id = graph_->GetCindexId(input_cindexes[i],
                                                        is_input, &is_new);
              this_dep[i] = dep_cindex_id;
              if (is_new)
                AddCindexId(dep_cindex_id, false, false);
              // we will keep dependent's usable_count_ up to date below
            }

            // remove duplicates of dependencies.
            SortAndUniq(&this_dep);

            // ==================== 设置 graph_.dependences[cindex_id] 的反向 -- depend_on_this[cindex_id] ================
            // set up the "depend_on_this_" array.
            std::vector<int32>::const_iterator
                iter = this_dep.begin(),
                end = this_dep.end();

            // Populate the "depend_on_this_" array, and
            // append the usable_count_ of things we depend on
            // (see the definition of this quantity next to where it is declared).

            // 下面这些条件 确保了这样情况下增加 该cindex_id的 usable_count_是合理的.?
            // Note: before calling AddDependencies() we verified the following:
            //  computable_info_[cindex_id] == kUnknown
            // and
            //  usable_count_[cindex_id] != 0
            
            // 向被依赖的dep_cindex_id 的 depend_on_this_ 中加入自己.
            for (; iter != end; ++iter) {
              int32 dep_cindex_id = *iter;
              depend_on_this_[dep_cindex_id].push_back(cindex_id);
              IncrementUsableCount(dep_cindex_id);

              void ComputationGraphBuilder::IncrementUsableCount(int32 cindex_id) {
                KALDI_PARANOID_ASSERT(static_cast<size_t>(cindex_id)<usable_count_.size());
                
                // the next line post-increments the reachable count.
                // 1 取值判断 一开始是=0的
                // 2 增加 引用计数. 每次有需要它的就增加计数. 但是更低级的需要不需要计数,
                // --------- 因为从它出发到更低级的没增加新的需要计算的.
                if (usable_count_[cindex_id]++ == 0 &&  computable_info_[cindex_id] != kNotComputable) {
                  // cindex_id 的依赖list.
                  std::vector<int32>::const_iterator
                      iter = graph_->dependencies[cindex_id].begin(),
                      end = graph_->dependencies[cindex_id].end();
                  for (; iter != end; ++iter) {
                    int32 dep_cindex_id = *iter;
                    IncrementUsableCount(dep_cindex_id);
                  }
                }
              }
              
            }

            // Now that we've added the dependencies, we can put this into
            // the computable_queue_ to assess whether it's computable
            KALDI_ASSERT(computable_info_[cindex_id] == kUnknown &&
                         !computable_queued_[cindex_id]);
            // we think it'll be faster in the next line to do push_front instead of
            // push_back; either one would be correct.
            computable_queue_.push_front(cindex_id);
            computable_queued_[cindex_id] = true;
          }

        }
      }
      current_queue_.swap(next_queue_);  // now next_queue_ will be empty.
      current_distance_++;
    }
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
