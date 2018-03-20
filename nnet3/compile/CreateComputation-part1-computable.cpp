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


  // 这个enum 是说对于一个给定的cindex_id, 是否我们可以从给定的input计算它.
  // Note 很多情况 例如对一个cindex_id 增加依赖关系之前, 我们意识到, 我们不会用到这个cindex_id
  // 即, 这个cindex_id 可能是可计算的, 但是没有作用, 当usable_count = 0, 就表明cindex无用.
  // 这样 对 output无用的cindex_id 我们就设置对应的 status 为 kWillNotCompute.
  // 这种情况下, 我们就会设置其标记为kNotComputable.
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
  
  KALDI_ASSERT(graph_->cindexes.empty() && "ComputationGraphBuilder initialized with nonempty graph.");
}







// input的cindex-id
//   computable_info_.push(kComputable),
//      表示可以计算, 对于input,稍后会被设置为 willNotCompute, 表示不需要计算.
//   computable_queued_.push(false)
//      表示没加入到computable_queue_, computable_queue_ -- 是需要计算依赖队列
// output的cindex-id
//   computable_info.push_back(kUnknown)
//   computable_queued.push_back(false)
//   next_queue_.push_back(cindex_id)
//      加入next_queue_ 表示cindex_id 在下一轮 需要计算其依赖, 当到达下一轮之前,会被加入到current_queue_

// depend_on_this_.push(?) --- 表示依赖于此cindex_id 的其他cindex的id
// usable_count_ 是否是计算output的依赖.
//   =1 表示 就是output本身
//   >1 表示 是距离output的距离.
//   =0 表示 input.
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
    // next_queue_
    // output 输出的cindex-id
    // output 向前依赖的cindex-1, 等待计算
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
  current_distance_ = 0;
  // the calls to AddCindexId in this function will have added to next_queue_.
  KALDI_ASSERT(current_queue_.empty());
  // 从output向前需要计算的队列.
  current_queue_.swap(next_queue_);
}




// ================= 官网的计算流程 ================
// 1 从output 开始 output-cindexes -> next_queue_发现计算依赖,
// 2 next_queue_ -> current_queue_  计算依赖cindex队列
// 3 foreach cindex in current_queue_ 计算依赖的 dep_cindex_id
//     1 加入computable_queue_ 更新计算性队列
//     2 修改usabel_count_ 更新需要计数
//     3 加入next_queue_ 等待下次循环 根据计算性情况, 计算依赖性
// 4 更新 computable_queue_ 更新cindex_id 的计算性
// 5 回到2 继续循环.

//  计算过程
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

    // output向前 构建编译逻辑Cindexes.
    BuildGraphOneIter();
    void ComputationGraphBuilder::BuildGraphOneIter() {
      // current_queue_ ------- 扩展依赖性队列
      while (!current_queue_.empty()) {
        int32 cindex_id = current_queue_.back();
        current_queue_.pop_back();
        
        KALDI_ASSERT(computable_info_[cindex_id] == kUnknown);


        // 从output向前计算依赖得到的依赖cindex  =====================
        // usable_count = 0表示无用, 设置其标记为 kWillNotComputable, 表示不需要进行计算(可能可计算,但是无用, 就不计算了)
        // 但是其可能的后续, 可能需要进行计算.
        if (usable_count_[cindex_id] == 0){
          // 将输入的cindex_id GraphBuilder->computable_info_[cindex_id] = kWillNotCompute 不需要计算.
          // 将其depend_on_this_ 的其他cindex_id
          // 1 都加入  需要计算 可计算性 队列 - computable_queue_ 
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

            // 设置为kWillNotCompute 这样是什么作用?????!!!!!
            computable_info_[cindex_id] = kWillNotCompute;

            // computable_queue_  保存那些需要重新计算Computable的cindex_id
            // computable_queued_ 保存cindex_id是否已经加入到computable_queue_ 的标记.
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
        // output cindexes ----------
        // =========== Compile 流程中 从output 向前计算需要计算的依赖 ============
        else{
          AddDependencies(cindex_id);
          
          // Add cindex_ids that this cindex_id depends on.
          void ComputationGraphBuilder::AddDependencies(int32 cindex_id) {

            // graph_dependencies.size() 是 Cindex总数
            // 目的是类似哈希表 保存长度足够, 后面需要增加cindex.
            if (static_cast<int32>(graph_->dependencies.size()) <= cindex_id) {
              graph_->dependencies.resize(2 * cindex_id + 1);
            }

            // 获得对应的Cindex
            Cindex cindex = graph_->cindexes[cindex_id];

            // ====================== 查找 cindex的依赖. =================
            // Cindex -- <node_index, Index>
            int32 node_index = cindex.first;
            const Index &index = cindex.second;
            const NetworkNode &node = nnet_.GetNode(node_index);
           
            // 将当前cindex_id 的依赖找到 ->
            std::vector<Cindex> input_cindexes;

            // ===================== 根据 NnetGraph的结构, 从output向前计算依赖cindex(node_id - 1, index) ==============
            // ===================== 计算某个 Cindex 的依赖 --- input_cindexes =============
            // 根据node.node_type : kDescriptor, kComponent, kInput,等通过node查找对应的依赖 cindex.
            switch (node.node_type) {
              case kDescriptor: {
                // desc describes how this node obtains its input from other nodes.
                const Descriptor &desc = node.descriptor;
                desc.GetDependencies(index, &input_cindexes);

                // 内部是通过 parts_ 中对每个子Descriptor都调用 GetDependencies
                // 对简单的子Descriptor 直接 构建<src_node_-1, index> ==> Cindex(src_node_-1, index);
                // 这样相当于 某个<node_i, (n, t, x)> 依赖于<node_i-1, (n, t, x)>
                // 对比较复杂的Descriptor 就会 引用多个<src_node_, index> 会形成依赖列表.
                
                break;
              }
              case kComponent: {
                int32 c = node.u.component_index;
                const Component *component = nnet_.GetComponent(c);
                std::vector<Index> input_indexes;
                // 直接返回自身, 经过修改, 就是上一个component-node---kDescriptor.
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

            // -------- 当前 cindex_ids 的依赖 cindexes
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
              // 向graph_->cindexes 中增加 input_cindexes[i];
              int32 dep_cindex_id = graph_->GetCindexId(input_cindexes[i],
                                                        is_input, &is_new);
              // cindex_ids的denpendences中增加 依赖的 dep_cindex_id.
              this_dep[i] = dep_cindex_id;

              // =======================  向 next_queue_ ========================
              // 如果是新的cindex -- 并不是input 也不是 output
              // 1 加入到 next_queue_ 队列中  ,等待 下次迭代 计算依赖关系
              // 2 设置对应的usable_count_ = 0,等待下次迭代, 进入上面的current_queu_ & usable_count=0 计算后续的可计算性.
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

            // 下面这些条件 确保了这样情况下增加 该cindex_id的 usable_count_是合理的.?
            // Note: before calling AddDependencies() we verified the following:
            //  computable_info_[cindex_id] == kUnknown
            // and
            //  usable_count_[cindex_id] != 0
            // 向被依赖的dep_cindex_id 的 depend_on_this_ 中加入自己.
            for (; iter != end; ++iter) {
              int32 dep_cindex_id = *iter;
              // 依赖于dep_cindex_id 的队列中 加入 cindex_id.
              depend_on_this_[dep_cindex_id].push_back(cindex_id);

              // 增加依赖dep_cindex_id 的usable_count_
              IncrementUsableCount(dep_cindex_id);
              void ComputationGraphBuilder::IncrementUsableCount(int32 cindex_id) {
                KALDI_PARANOID_ASSERT(static_cast<size_t>(cindex_id)<usable_count_.size());
                
                // the next line post-increments the reachable count.
                // 1 取值判断  =0, 是新加入的dep,  否则是已经存在的dep 只增加其==被需要计数== 即可.
                // 2 判断其可计算性, 如果不是 kNotComputable , 并且是新加入的cindex, === 需要增加更低级依赖的依赖计数 ===
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

            // 增加了依赖cindexes, 将当前cindex_id  加入到 可计算性队列中, 等待计算其可计算性.
            KALDI_ASSERT(computable_info_[cindex_id] == kUnknown && !computable_queued_[cindex_id]);
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

    // ================= 更新 computable_queue_ 中所有cindex 的computable_info_ ===========
    // ==========1 判断cindex node_index 的节点类型 kDescriptor, kComponent kInput等
    //                 计算computable 为kComputable, kNotComputable, kUnknown.
    // ==========2 处理 确定不是 kUnknown的cindex
    //                 1 cindex的后续 other_cindex_id 都需要重新计算 可计算性 加入到 computable_queue_
    //                 2 如果是 kNotComputable, 并且 需要计数 usable_count_ != 0 需要更新其依赖的各个usable_count -- 并判断.
    //                   
    // TODO: come up with a scheme to delay when we call
    // UpdateAllComputableInfo().
    UpdateAllComputableInfo();
    void ComputationGraphBuilder::UpdateAllComputableInfo() {

      // 处理 需要更新计算性队列. 内部还会追加.
      while (!computable_queue_.empty()) {
        int32 cindex_id = computable_queue_.front();
        computable_queue_.pop_front();
        computable_queued_[cindex_id] = false;
        UpdateComputableInfo(cindex_id);

        void ComputationGraphBuilder::UpdateComputableInfo(int32 cindex_id) { 
          // 如果 computable_info_ 不是kUnknown, 就不应该加入到computable_info_ 信息.
          KALDI_ASSERT(static_cast<size_t>(cindex_id) < computable_info_.size());
          
          char &output = computable_info_[cindex_id];
          KALDI_ASSERT(output == kUnknown);

          // 计算 cindex_id 可计算性.
          output = static_cast<char>(ComputeComputableInfo(cindex_id));

          ComputationGraphBuilder::ComputeComputableInfo(int32 cindex_id) const {
            // 获得 cindex对象.
            const Cindex &cindex = graph_->cindexes[cindex_id];
            int32 node_id = cindex.first;
            const Index &index = cindex.second;
            const NetworkNode &node = nnet_.GetNode(node_id);
            
            switch (node.node_type) {
              case kDescriptor: {

                // bool SimpleSumDescriptor::IsComputable( const Index &ind,
                //                                         const CindexSet &cindex_set,
                //                                         std::vector<Cindex> *used_inputs) const {
                //   Cindex c = src_->MapToInput(ind);
                //   // 1 graph_ 中判断cindex 是否存在, 不存在 false
                //   // 2 存在则判断, computable_info_[cindex_id] 是否是kComputable, 是 true.
                //   bool src_present  = cindex_set(c);
                //   if (src_present && used_inputs != NULL)
                //     used_inputs->push_back(c);
                //   return src_present;
                // }

                
                
                const Descriptor &desc = node.descriptor;

                // 1 treat_unknown_as_computable = false
                {
                  CindexSet cindex_set(*graph_, computable_info_, false);

                  if (desc.IsComputable(index, cindex_set, NULL)) {
                    // it's computable even without counting kUnknown inputs as computable
                    // [treat_unknown_as_computable = false] -> definitely computable.
                    return kComputable;
                  }
                }
                // 2 treat_unknown_as_computable = true
                {
                  CindexSet cindex_set2(*graph_, computable_info_, true);
                  if (!desc.IsComputable(index, cindex_set2, NULL)) {
                    // it's not computable even when counting kUnknown inputs as
                    // computable [treat_unknown_as_computable = true] -> definitely not
                    // computable.
                    return kNotComputable;
                  }
                }
                // 1 1 2 两种情况下, 如果Computable 都为false, 则 返回kNotComputable
                // 2 1 ok 则 返回kComputable
                // 3 1 false 2 ok 则返回kUnknown.
                return kUnknown;
              }
                
              case kComponent: {
                // 获得对应的component
                const Component *c = nnet_.GetComponent(node.u.component_index);
                // 获得对应的 kDescriptor component-node.
                const int32 input_node_id = node_id - 1;


                // 对Component的 Computable判断比Descriptor简单.
                // 1 判断 IndexSet中(实际还是判断graph_的cindexes中) 是否存在 output_index,
                // 2 然后 判断 是否是 kComputable. 即可.
                bool Component::IsComputable(const MiscComputationInfo &misc_info,
                                             const Index &output_index,
                                             const IndexSet &input_index_set,
                                             std::vector<Index> *used_inputs) const {
                  // the default Component dependency is for an output index to map directly to
                  // the same input index, which is required to compute the output.
                  if (!input_index_set(output_index))
                    return false;
                  if (used_inputs) {
                    used_inputs->clear();
                    used_inputs->push_back(output_index);
                  }
                  return true;
                }


                {
                  IndexSet index_set(*graph_, computable_info_, input_node_id, false);
                  if (c->IsComputable(request_->misc_info, index, index_set, NULL)) {
                    // it's computable even without counting kUnknown inputs as computable
                    // [treat_unknown_as_computable = false] -> definitely computable.
                    return kComputable;
                  }
                }
                
                IndexSet index_set2(*graph_, computable_info_, input_node_id, true);
                if (!c->IsComputable(request_->misc_info, index, index_set2, NULL)) {
                  // it's not computable even when counting kUnknown inputs as computable
                  // [treat_unknown_as_computable = true] -> definitely not computable.
                  return kNotComputable;
                }

                
                return kUnknown;
              }
              case kDimRange: {
                Cindex input_cindex(node.u.node_index, index);
                int32 input_cindex_id = graph_->GetCindexId(input_cindex);
                if (input_cindex_id != -1)
                  return ComputableInfo(computable_info_[input_cindex_id]);
                else
                  return kUnknown;
              }
              case kInput: {
                // 如果是Request的input, 那么该input对应的indexes的 is_intput[cindex_id] = true.
                return graph_->is_input[cindex_id] ? kComputable : kNotComputable;
              }
              default:
                KALDI_ERR << "Invalid node type.";
                return kUnknown;  // suppress compiler warning.
            }
          }

          

          // ================ cindex_id 计算性确定, 曾更新其后续 other_cindex_id 的计算性 ========================
          // 当计算cindex的输出 不是 kUnknown 的计算性时, 可以更新其后续的other_cindex_id的 可计算性了
          //  将other_cindex_id 加入到 computable_queue_.
          if (output != kUnknown) {

            // 若是kUnknown, 当前cindex 的后续 的computable 计算性, 可能会发生改变, 所以如果他们不在computable_queue_
            // 将他们放入computable_queue_
            std::vector<int32>::const_iterator
                iter = depend_on_this_[cindex_id].begin(),
                end = depend_on_this_[cindex_id].end();
            
            for (; iter != end; ++iter) {
              int32 other_cindex_id = *iter;
              // cindex_id的后续 other_cindex_id 是kUnknown, 并且 不在computable_queue_中, 则放入
              if (computable_info_[other_cindex_id] == kUnknown &&
                  !computable_queued_[other_cindex_id]) {
                computable_queue_.push_back(other_cindex_id);
                computable_queued_[other_cindex_id] = true;
              }
            }
            // 如果 不可计算, 并且 需要计数不为0.
            if (output == kNotComputable && usable_count_[cindex_id] != 0) {
              // 如果我们确定了 原本kUnknown的Cindex的 计算性为kNotComputable
              // 这时, 我们就需要减少其的 计算需要计数, 并减少它依赖的 需要计数.
              std::vector<int32>::const_iterator
                  iter = graph_->dependencies[cindex_id].begin(),
                  end = graph_->dependencies[cindex_id].end();
              for (; iter != end; ++iter) {
                int32 dep_cindex_id = *iter;
                DecrementUsableCount(dep_cindex_id);

                // ================= 递归减少计数 ===============
                void ComputationGraphBuilder::DecrementUsableCount(int32 cindex_id) {
                  KALDI_PARANOID_ASSERT(usable_count_[cindex_id] > 0);
                  // --后, 自身=0, 说明只有一个后续other_cindex_id,并且刚刚那个后续被判定为 kNotComputable.
                  // 并且, 其计算性不是 kNotComputable
                  // 那么递归其所有依赖 减少计数.
                  if (--usable_count_[cindex_id] == 0 &&
                      computable_info_[cindex_id] != kNotComputable) {
                    std::vector<int32>::const_iterator
                        iter = graph_->dependencies[cindex_id].begin(),
                        end = graph_->dependencies[cindex_id].end();
                    for (; iter != end; ++iter) {
                      int32 dep_cindex_id = *iter;
                      DecrementUsableCount(dep_cindex_id);
                    }
                  }
                }

              }
            }
          }
        }
      }
    }

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


    // 重新计算一个cindex的 usable_count_, 只需要计算其后续的computable_info_[other_cindex_id]不为 kNotComputable即可.
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













// 
// 所有 !kComputable的 cindex 对应的node 不是 outputNode 时候 认为 output可计算.
bool ComputationGraphBuilder::AllOutputsAreComputable() const {
  char is_computable_char = static_cast<char>(kComputable);
  std::vector<char>::const_iterator iter = computable_info_.begin(),
      end = computable_info_.end();
  for (int32 cindex_id = 0; iter != end; ++iter, ++cindex_id) {
    if (*iter != is_computable_char) {  // is not computable.
      int32 network_node = graph_->cindexes[cindex_id].first;
      if (nnet_.IsOutputNode(network_node))
        return false;
    }
  }
  return true;
}

// 1 剪枝依赖, 剩下真实用到的依赖
// 2 清理那些不可计算的依赖
void ComputationGraphBuilder::PruneDependencies(int32 cindex_id) {
  ComputableInfo c = static_cast<ComputableInfo>(computable_info_[cindex_id]);
  // 此时不应该再有 kUnknown的cindex_id了.
  // by the time this is called, there should be no cindexes with unknown state.
  KALDI_ASSERT(c != kUnknown);

  // kNotComputable 不可计算的
  // kWillNotCompute 不需要计算的(output不需要的)
  if (c == kNotComputable || c == kWillNotCompute) {
    // if something is not computable, there is no point
    // keeping around its dependencies.
    graph_->dependencies[cindex_id].clear();
    return;
  }

  
  KALDI_ASSERT(c == kComputable);

  // ========== 获得 Cindex
  const Cindex &cindex = graph_->cindexes[cindex_id];
  int32 node_id = cindex.first;
  const Index &index = cindex.second;
  const NetworkNode &node = nnet_.GetNode(node_id);
  // ========== 获得 dependencies 依赖
  std::vector<int32> &dependencies = graph_->dependencies[cindex_id];
  
  std::sort(dependencies.begin(), dependencies.end());
  // 保存必要依赖
  std::vector<int32> used_cindex_ids;

  switch (node.node_type) {
    case kDescriptor: {
      const Descriptor &desc = node.descriptor;

      bool dont_care = false;  // there should be no kUnknown, and we check this
      CindexSet cindex_set(*graph_, computable_info_, dont_care);
      // 必要依赖 used_cindexes
      std::vector<Cindex> used_cindexes;
      bool ans = desc.IsComputable(index, cindex_set, &used_cindexes);
      // If the next assert fails it could be a failure in the assumption that
      // making more inputs available will never change something from not being
      // computable to being computable; or it could be a bug elsewhere.
      KALDI_ASSERT(ans);
      // 必要依赖 size
      size_t size = used_cindexes.size();

      // 必要依赖 copy
      used_cindex_ids.resize(size);
      for (size_t i = 0; i < size; i++) {
        int32 dep_cindex_id = graph_->GetCindexId(used_cindexes[i]);
        used_cindex_ids[i] = dep_cindex_id;
      }
      break;
    }
    case kComponent: {
      const Component *c = nnet_.GetComponent(node.u.component_index);
      bool dont_care = false;  // there should be no kUnknown, and we check this
      // In the line below, node_id - 1 is the index of the component-input
      // node-- the descriptor at the input to the component.  We are interested
      // in the set of inputs to the component that are computable.
      IndexSet index_set(*graph_, computable_info_, node_id - 1, dont_care);
      std::vector<Index> used_indexes;
      bool ans = c->IsComputable(request_->misc_info, index, index_set,
                                 &used_indexes);
      // If the next assert fails it could be a failure in the assumption that
      // making more inputs available will never change something from not being
      // computable to being computable; or it could be a bug elsewhere.
      KALDI_ASSERT(ans);
      size_t size = used_indexes.size();
      used_cindex_ids.resize(size);
      for (size_t i = 0; i < size; i++) {
        Cindex dep_cindex(node_id - 1, used_indexes[i]);
        int32 dep_cindex_id = graph_->GetCindexId(dep_cindex);
        used_cindex_ids[i] = dep_cindex_id;
      }
      break;
    }
    case kDimRange:
      KALDI_ASSERT(dependencies.size() == 1);
      // there should be exactly one dependency and it is required, not
      // optional, so there is nothing to do here.  Return.
      return;
    case kInput:
      KALDI_ASSERT(dependencies.empty());
      // there is nothing to do; return.
      return;
    default:
      KALDI_ERR << "Invalid node type";
  }
  SortAndUniq(&used_cindex_ids);

  // 必要依赖===> graph_->dependencies[cindex_id].
  dependencies.swap(used_cindex_ids);
}

void ComputationGraphBuilder::Prune() {
  // 剪枝, 对每个request都进行剪枝, 所以只处理当前处理的reqeust.
  // 每个request 都对应自己的 cindexes ,
  // 多个request都加入一个graph_, 所以区分需要 使用 graph_中的segment_ends 标记.
  int32 start_cindex_id = (graph_->segment_ends.empty() ? 0 :
                           graph_->segment_ends.back());

  int32 num_cindex_ids = graph_->cindexes.size();

  // foreach cindex_id 剪枝依赖 只留下真正需要用的 --
  // 必要依赖 ===> graph_->dependencies[cindex_id]
  for (int32 cindex_id = start_cindex_id; cindex_id < num_cindex_ids; cindex_id++)
    PruneDependencies(cindex_id);

  // 如下的代码, 清理 cindex_id 的后续. 不需要获得所有的数据?
  // 先删除掉 以前的依赖
  // 然后扩展这些依赖.  (设计也可以循环设置也可以)
  depend_on_this_.resize(start_cindex_id);
  depend_on_this_.resize(num_cindex_ids);

  // ==================== cindex 中所有计算output的必要cindex
  // 设置对应required[cindex_id] = true ==========================
  std::vector<bool> required;
  ComputeRequiredArray(start_cindex_id, &required);
  void ComputationGraphBuilder::ComputeRequiredArray(int32 start_cindex_id, std::vector<bool> *required) const {
    int32 num_cindex_ids = graph_->cindexes.size();

    // 设置为 当前request 加入的cindex的总数
    required->clear();
    required->resize(num_cindex_ids - start_cindex_id, false);

    // output? 1 : 0
    std::vector<char> is_output_node(nnet_.NumNodes());
    for (int32 n = 0; n < nnet_.NumNodes(); n++)
      is_output_node[n] = (char)(nnet_.IsOutputNode(n) ? 1 : 0);


    // ================== queue 依赖关系查找队列 ================
    // queue  从output开始 反向 查找依赖, 将待查找依赖的 cindex 加入queue.
    // require output的依赖 dep_cindex_id 的require[dep_cindex_id] = true;
    // require 就是 输出依赖性, true, 就是依赖会需要这些的计算.
    std::vector<int32> queue;
    
    // foreach cindex_id 是个output就node 加入到 required[cindex_offset] = true
    // cindex_id 是个output-node 的数据 就加入queue
    for (int32 c = start_cindex_id; c < num_cindex_ids; c++) {
      // First put the output cindex_ids into the queue.
      int32 node_id = graph_->cindexes[c].first;
      if (is_output_node[node_id]) {
        (*required)[c - start_cindex_id] = true;
        queue.push_back(c);
      }
    }

    // 从queue -- output-node的cindex开始
    while (!queue.empty()) {
      int32 c = queue.back();
      queue.pop_back();
      // 计算依赖-- 此时已经是 必要依赖
      const std::vector<int32> &dependencies = graph_->dependencies[c];
      
      std::vector<int32>::const_iterator
          iter = dependencies.begin(),
          end = dependencies.end();
      
      for (; iter != end; ++iter) {
        int32 d = *iter;
        // 依赖在 本segment(request中添加) && 没加入过required
        // 将依赖加入 queue ---- 作用和 current_queue_ 差不多 对应依赖的必要性标志 保存在required[dep_cindex_id]
        if (d >= start_cindex_id && !(*required)[d - start_cindex_id]){
          (*required)[d - start_cindex_id] = true;
          queue.push_back(d);
        }
      }
    }
    // just check that we don't have any cindex_ids which are required but have
    // usable_count_ == 0; this would indicate a bug somewhere.
    for (int32 c = start_cindex_id; c < num_cindex_ids; c++)
      KALDI_ASSERT(!((*required)[c - start_cindex_id] &&
                     (usable_count_[c] == 0)));

  }



  // 计算 必须保持的 cindex.
  std::vector<bool> keep(num_cindex_ids - start_cindex_id, false);
  for (int32 c = start_cindex_id; c < num_cindex_ids; c++) {
    // true, 必要依赖 || 是个输入
    if (required[c - start_cindex_id] || graph_->is_input[c]) {
      // 必然是 kComputable的
      KALDI_ASSERT(computable_info_[c] == kComputable &&
                   "You are calling Prune when not everything is computable.");
      keep[c - start_cindex_id] = true;
    }
  }


  // keep 的长度 与 start_cindex_id : cindexes.end() 的长度一样
  // 因为每次都是一个Request 一个Request 的处理, 所以每次都是 start_cindex_id : cindexes.end().
  // 因此处理的是最后面的一部分, 可以随意重新编号.
  graph_->Renumber(start_cindex_id, keep);
  // 我们也需要重新对 computable_info_ 和 usable_count_ 重新编号
  // graph_.Renumber() 并没有处理这些, 但是我们可以做一些简化处理,

  // 1 设置所有的computable_info_ 为kComputable
  // 因为前面已经计算了 可计算性, 并且是 可计算之后才能够进入剪枝, 所以必然是都kComputable.
  // 2 设置所有的usable_count_ = 1 , 这个和我们对usable_count_的定义不是很准确.
  // 但是它防止了额外的计算量 因为 当usable_count_ > 0 就可以了>

  int32 new_num_cindex_ids = graph_->cindexes.size();
  computable_info_.resize(start_cindex_id);
  computable_info_.resize(new_num_cindex_ids, (char)kComputable);
  usable_count_.resize(start_cindex_id);
  usable_count_.resize(new_num_cindex_ids, 1);

  // denpend_on_this 应该是不在需要了.
  // depend_on_this_ is a vector of vectors-- keeping track of the reverse of
  // the dependencies-- and I believe we won't be needing this information any
  // more past this point.
  depend_on_this_.resize(start_cindex_id);
  depend_on_this_.resize(new_num_cindex_ids);

  // computable_queued_ 也不在需要了,computable_queue_ 老早之前就处理完了.
  computable_queued_.resize(new_num_cindex_ids);
  // 断言! 可计算性队列  必然为空.
  KALDI_ASSERT(computable_queue_.empty());

  // 一个新的终止
  graph_->segment_ends.push_back(new_num_cindex_ids);
}
