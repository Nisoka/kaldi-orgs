void ComputeComputationPhases(
    const Nnet &nnet,
    const ComputationGraph &graph,
    std::vector<std::vector<std::vector<int32> > > *phases_per_segment) {
  
  using namespace computation_graph;
  int32 num_cindex_ids = graph.cindexes.size();

  
  std::vector<int32> cindex_id_to_segment_and_epoch;
  std::vector<std::vector<std::vector<int32 > > > epochs_per_segment;
  std::vector<bool> epoch_is_trivial;

  
  // 计算整体数据信息? epoch 在这里描述的是类似阶段的意思.
  ComputeEpochInfo(nnet, graph, &cindex_id_to_segment_and_epoch, &epochs_per_segment, &epoch_is_trivial);

  // 这个函数 计算确定的信息关于 cindex_ids epoch
  // ComputeNnetComputationEpochs() 提供给我们一个映射 从NetworkNode index 到一个 epoch index:
  // 基本上, 首先计算的nodes 有较低的epoch index,
  // 具有较强关系的components 部分的所有的nodes 都具有相同的epoch index
  //  在一个非循环的nnet graph中 每个Component 一般都具有自己的epoch index.
  // 但是在LSTM这样的结构中, 每个LSTM layer层(具有多个component), 会有自己的epoch index.
  
  //  完整的计算顺序 会将顺序安排成为 epoch
  //  (除了kComponent nodes的输出 实际上作为input被提供给network, 不会被limitations支配, 会在最快的顺序)
  //  我们会忽略这个函数的输出, 因为它 认为cindex_ids 被作为network的输入.

  ///  \param nnet [in] The neural net
  ///  \param graph [in] The computation graph
  ///  \param cindex_id_to_segment_and_epoch [out]
  //    一个vector 将 cindex_id 映射到一个 num, 
  //  　如果　两个cindex_ids 在相同的segment 和 相同的epoch. 其他情况 num 会相等.
  //    这个num 是组合了 segment index 和 epoch index. 实现细节不重要.

  ///  \param epochs_per_segment [out]
  //    是一个列表展示, graph 中的所有 cindex_ids 首先被segment 分割, 然后被epoch分割.

  ///  \param epoch_is_trivial [out]
  //    是一个vector, epoch index 为索引, 和 epochs_per_segment中的 第二索引index相同.
  //    true, 如果epoch index 对应一个 NetworkNode,
  //          (对应到 nnet input的 epoch indexes 也会true, 这些会是每个segment中的最开始epoch)
  //    只依赖于 nnet的结构.

  //  epoch 在这里的意思是 计算批次 或者说是计算次序, 一个完整的计算 需要多批进行计算???

  // cindex_id_to_segment_and_epoch ***** 每个 cindex_id 对应的 segment+epoch 次序
  // ------------- request1== 0 1 2 3 4 .. 30, request2= 31 32 ... 60.
  
  // epochs_per_segment *********************************
  // ------------- < request1-epochs< epoch-id-1 <cindex_id, ..>
  //                                  epoch-id-2 <cindex-id, ..>
  //                                  ..
  //                                  epoch-id-n <cindex-id, ..> >
  
  //                 request2-epochs< epoch-id-1 <cindex_id, ..>
  //                                  epoch-id-2 <cindex-id, ..>
  //                                  ..
  //                                  epoch-id-n <cindex-id, ..> > 
  //                 >

  // epoch_is_trivial
  // ------------- 表示每个 每个epoch内含有的cindex-id数量 true 表示数量 <=1

  
  static void ComputeEpochInfo(
      const Nnet &nnet,
      const ComputationGraph &graph,
      std::vector<int32> *cindex_id_to_segment_and_epoch,
      std::vector<std::vector<std::vector<int32 > > > *epochs_per_segment,
      std::vector<bool> *epoch_is_trivial) {

    // ====================== 1 处理nnet 中node 的epoch-c-order ==============
    // --------------------------------------------------------------
    // 注意 这里是映射的 node 而不是cindex, 是node的epoch 次序.
    // 每个request 都具有相同数量的node. 也就是有相同的 node_to_epoch.
    // --------------------------------------------------------------    
    // node_to_epoch 映射每个 nnet node 到 一个index, 粗略的告诉我们 计算他们的顺序.
    // 但是我们可能需要计算 更好的cindex_id级别的顺序, 例如在RNN结构下.
    // 经过 ComputeNnetComputationEpochs() 处理 将nnet的信息转化为 node_to_epoch 结果如下:
    // 在简单的tdnn结构下, 为每个node 安排 epoch 就是简单的顺序向后计算.
    // 但是在复杂的RNN LSTM等 结构 可能就会比较复杂的情况
    // LOG (nnet3-compute-prob[5.3]:ComputeNnetComputationEpochs():nnet-graph.cc:318) Result node to epoch is:
    // 0 -> (1,); 1 -> (0,); 2 -> (2,); 3 -> (3,); 4 -> (4,); 5 -> (5,); 6 -> (6,); 7 -> (7,); 8 -> (8,); 9 -> (9,); 10 -> (10,); 11 -> (11,); 12 -> (12,); 13 -> (13,); 14 -> (14,); 15 -> (15,); 16 -> (16,); 17 -> (17,); 18 -> (18,); 19 -> (19,); 20 -> (20,); 21 -> (21,); 22 -> (22,); 23 -> (23,); 24 -> (24,); 25 -> (25,); 26 -> (26,); 27 -> (27,); 28 -> (28,); 29 -> (29,); 30 -> (30,)
    std::vector<int32> node_to_epoch;
    ComputeNnetComputationEpochs(nnet, &node_to_epoch);
    
    void ComputeNnetComputationEpochs(const Nnet &nnet, std::vector<int32> *node_to_epoch) {
      KALDI_ASSERT(node_to_epoch != NULL);

      
      // 将nnet 转化为直接有向图
      // graph 描述的就是 denpend_on_this 的node-id
      // <node<后续node> node<后续node> node<后续node>  ...   >
      // 是node-id 构成的 二维list
      std::vector<std::vector<int32> > graph;
      NnetToDirectedGraph(nnet, &graph);
      void NnetToDirectedGraph(const Nnet &nnet,
                               std::vector<std::vector<int32> > *graph) {
        graph->clear();
        int32 num_nodes = nnet.NumNodes();
        graph->resize(num_nodes);
        // ----- 对每个Nnet node
        for (int32 n = 0; n < num_nodes; n++) {
          const NetworkNode &node = nnet.GetNode(n);
          // handle dependencies of this node.
          std::vector<int32> node_dependencies;
          switch (node.node_type) {
            case kInput:
              break;  // no node dependencies.
            case kDescriptor:
              node.descriptor.GetNodeDependencies(&node_dependencies);
              break;
            case kComponent:
              node_dependencies.push_back(n - 1);
              break;
            case kDimRange:
              node_dependencies.push_back(node.u.node_index);
              break;
            default:
              KALDI_ERR << "Invalid node type";
          }
          SortAndUniq(&node_dependencies);

          // cur node 的所有直接依赖
          // graph -- dep_on_this_node --- vector<后续node>
          for (size_t i = 0; i < node_dependencies.size(); i++) {
            int32 dep_n = node_dependencies[i];
            KALDI_ASSERT(dep_n >= 0 && dep_n < num_nodes);
            (*graph)[dep_n].push_back(n);
          }
        }
      }
      KALDI_VLOG(6) << "graph is: " << PrintGraphToString(graph);

//LOG (nnet3-compute-prob[5.3]:ComputeNnetComputationEpochs():nnet-graph.cc:272) graph is:
// 0 -> (2); 1 -> (2); 2 -> (3); 3 -> (4); 4 -> (5); 5 -> (6); 6 -> (7); 7 -> (8); 8 -> (9); 9 -> (10); 10 -> (11); 11 -> (12); 12 -> (13); 13 -> (14); 14 -> (15); 15 -> (16); 16 -> (17); 17 -> (18); 18 -> (19); 19 -> (20); 20 -> (21); 21 -> (22); 22 -> (23); 23 -> (24); 24 -> (25); 25 -> (26); 26 -> (27); 27 -> (28); 28 -> (29); 29 -> (30); 30 -> ()
      // 0 的后续是2
      // 1 的后续是2
      // 2 的后续是3

      // sccs ???
      std::vector<std::vector<int32> > sccs;
      FindSccs(graph, &sccs);
      // void FindSccsTarjan(std::vector<std::vector<int32> > &graph,
      //                     std::vector<std::vector<int32> > *sccs) {
      //   KALDI_ASSERT(sccs != NULL);

      //   // Initialization.
      //   std::vector<TarjanNode> tarjan_nodes(graph.size());
      //   std::vector<int32> tarjan_stack;
      //   int32 global_index = 0;

      //   // 对每个 graph 中的 node.
      //   // Calls the recursive function.
      //   for (int32 n = 0; n < graph.size(); ++n) {
      //     if (tarjan_nodes[n].index == -1) {
      //       // global_index 就是输出计数, 每次增加.
      //       TarjanSccRecursive(n, graph, &global_index, &tarjan_nodes, &tarjan_stack, sccs);
            
      //       void TarjanSccRecursive(int32 node,
      //                               const std::vector<std::vector<int32> > &graph,
      //                               int32 *global_index,
      //                               std::vector<TarjanNode> *tarjan_nodes,
      //                               std::vector<int32> *tarjan_stack,
      //                               std::vector<std::vector<int32> > *sccs) {

      //         // node 的 lowlink 永远指向 后续中 最小的index. 并且lowlink 
      //         // Initializes the current Tarjan node.
      //         (*tarjan_nodes)[node].index = *global_index;
      //         (*tarjan_nodes)[node].lowlink = *global_index;
      //         *global_index += 1;
      //         (*tarjan_nodes)[node].on_stack = true;
      //         tarjan_stack->push_back(node);

      //         // DFS from the current node.
      //         // 从当前节点开始 DFS 该几点的后续(depend_on_this 的节点).
      //         for (int32 i = 0; i < graph[node].size(); ++i) {
      //           int32 next = graph[node][i];

      //           // 判断 是否安排了 global_index 的 node 就递归安排global_index.
      //           if ((*tarjan_nodes)[next].index == -1) {
      //             TarjanSccRecursive(next, graph,
      //                                global_index, tarjan_nodes, tarjan_stack, sccs);
      //             // 当前的 低方向link  设置为 递归的 最小lowlink.
      //             (*tarjan_nodes)[node].lowlink = std::min((*tarjan_nodes)[node].lowlink,
      //                                                      (*tarjan_nodes)[next].lowlink);
                  
      //           }
      //           // 安排了 global_index , 判断是在 stack上.
      //           // 则继续安排 lowlink.
      //           else if ((*tarjan_nodes)[next].on_stack) {
      //             // 在stack上-- 回边, 我们不能使用 此next节点的lowlink, 因为可能会指向root的index
      //             // 因为当前的node 可能不是root.
                  
      //             (*tarjan_nodes)[node].lowlink = std::min((*tarjan_nodes)[node].lowlink,
      //                                                      (*tarjan_nodes)[next].index);
      //           }
      //         }

      //         // DFS 当前节点的 后续节点 结束
      //         // 如果形成环.
      //         if ((*tarjan_nodes)[node].index == (*tarjan_nodes)[node].lowlink) {
      //           std::vector<int32> scc;
      //           int32 pop_node;
      //           do {
      //             pop_node = tarjan_stack->back();
      //             tarjan_stack->pop_back();
      //             (*tarjan_nodes)[pop_node].on_stack = false;
      //             scc.push_back(pop_node);
      //           } while (pop_node != node);
      //           KALDI_ASSERT(pop_node == node);
      //           sccs->push_back(scc);
      //         }
      //       }
      //     }
      //   }
      // }
      
      // KALDI_LOG << "sccs is: " << PrintGraphToString(sccs);
      //LOG (nnet3-compute-prob[5.3]:ComputeNnetComputationEpochs():nnet-graph.cc:276) sccs  is:
      //0 -> (30); 1 -> (29); 2 -> (28); 3 -> (27); 4 -> (26); 5 -> (25); 6 -> (24); 7 -> (23); 8 -> (22); 9 -> (21); 10 -> (20); 11 -> (19); 12 -> (18); 13 -> (17); 14 -> (16); 15-> (15); 16 -> (14); 17 -> (13); 18 -> (12); 19 -> (11); 20 -> (10); 21 -> (9); 22 -> (8); 23 -> (7); 24 -> (6); 25 -> (5); 26 -> (4); 27 -> (3); 28 -> (2); 29 -> (0); 30 -> (1)
      // 0 的sccs 是 30
      // 1 的sccs 是 29
      // 
      
      // ---------------- 创建 scc_graph
      // 这个scc_graph 目的是什么?? 在做什么东西呢??? 结果如下:
      std::vector<std::vector<int32> > scc_graph;
      MakeSccGraph(graph, sccs, &scc_graph);
      KALDI_VLOG(6) << "scc graph is: " << PrintGraphToString(scc_graph);

      //LOG (nnet3-compute-prob[5.3]:ComputeNnetComputationEpochs():nnet-graph.cc:281) scc graph is:
      //0 -> (); 1 -> (0); 2 -> (1); 3 -> (2); 4 -> (3); 5 -> (4); 6 -> (5); 7 -> (6); 8 -> (7); 9 -> (8); 10 -> (9); 11 -> (10); 12 -> (11); 13 -> (12); 14 -> (13); 15 -> (14);16 -> (15); 17 -> (16); 18 -> (17); 19 -> (18); 20 -> (19); 21 -> (20); 22 -> (21); 23 -> (22); 24 -> (23); 25 -> (24); 26 -> (25); 27 -> (26); 28 -> (27); 29 -> (28); 30 -> (28)



      // ---------------- 根据scc_graph 安排scc_node -> epochs
      // 对应每个node 属于某个 epoch
      std::vector<int32> scc_node_to_epoch;
      ComputeTopSortOrder(scc_graph, &scc_node_to_epoch);
      // KALDI_VLOG(6) << "scc graph is: " << PrintVectorToString(scc_node_to_epoch);
      
      // LOG (nnet3-compute-prob[5.3]:ComputeNnetComputationEpochs():nnet-graph.cc:299) scc node to epoch is:
      // 0 -> (30,); 1 -> (29,); 2 -> (28,); 3 -> (27,); 4 -> (26,); 5 -> (25,); 6 -> (24,); 7 -> (23,); 8 -> (22,); 9 -> (21,); 10 -> (20,); 11 -> (19,); 12 -> (18,); 13 -> (17,); 14 -> (16,); 15 -> (15,); 16 -> (14,); 17 -> (13,); 18 -> (12,); 19 -> (11,); 20 -> (10,); 21 -> (9,); 22 -> (8,); 23 -> (7,); 24 -> (6,); 25 -> (5,); 26 -> (4,); 27 -> (3,); 28 -> (2,); 29 -> (1,); 30 -> (0,)
      

                                                     
      // ---------------- 根据scc scc_node_to_epoch 将node 映射到 epoch
      node_to_epoch->clear();
      node_to_epoch->resize(graph.size());
      
      for (int32 i = 0; i < sccs.size(); ++i) {
        for (int32 j = 0; j < sccs[i].size(); ++j) {
          int32 node = sccs[i][j];
          KALDI_ASSERT(node >= 0 && node < graph.size());
          (*node_to_epoch)[node] = scc_node_to_epoch[i];
        }
      }
      // KALDI_VLOG(6) << "scc graph is: " << PrintVectorToString(node_to_epoch);
      
      // LOG (nnet3-compute-prob[5.3]:ComputeNnetComputationEpochs():nnet-graph.cc:318) Result node to epoch is:
      // 0 -> (1,); 1 -> (0,); 2 -> (2,); 3 -> (3,); 4 -> (4,); 5 -> (5,); 6 -> (6,); 7 -> (7,); 8 -> (8,); 9 -> (9,); 10 -> (10,); 11 -> (11,); 12 -> (12,); 13 -> (13,); 14 -> (14,); 15 -> (15,); 16 -> (16,); 17 -> (17,); 18 -> (18,); 19 -> (19,); 20 -> (20,); 21 -> (21,); 22 -> (22,); 23 -> (23,); 24 -> (24,); 25 -> (25,); 26 -> (26,); 27 -> (27,); 28 -> (28,); 29 -> (29,); 30 -> (30,)
        

    }


    // ------------- 每个request 都共享相同的 nnet node 结构, 因此每个request的 epoch-c-order 当前是一样的
    // ------------- 都是 node_to_epoch

    // ====================== 2 计算 每个request内 cindex 根据对应node 的epoch-c-order(node_to_epoch),
    // ====================== 计算每个cindex 的 epoch-c-order 
    //            
    // ------------------ node_to_epoch 保存的是每个node 的 epoch-c-order ----------------
    
    // 向node_to_epoch[i] + 1, 因为我们要保留0 给输入network的input.
    // 并且我们不想证明 epoch-id = 0 只对应于input.????
    for (int32 i = 0; i < node_to_epoch.size(); i++)
      node_to_epoch[i]++;
    
    int32
        num_nodes = nnet.NumNodes(),
        num_cindex_ids = graph.cindexes.size(),
        num_segments = graph.segment_ends.size(),

    
        // 对于所有request都是具有相同的 epoch-c-order.
        // 所以num_epoch_indexes 保存的是 epoch总数   ---- 应该是 epoch_last + 1
        num_epoch_indexes = 1 + *std::max_element(node_to_epoch.begin(),
                                                  node_to_epoch.end());
    
    KALDI_ASSERT(node_to_epoch.size() == num_nodes);





    // ======================== 目标 是每个 Request-segment 的cindex 的 epoch-c-order ============
    // vector size = 段总数--Request总数
    epochs_per_segment->clear();
    epochs_per_segment->resize(num_segments);


    
    // epoch_to_num_nodes 只是为了让我们知道是否每个epoch-index 对应多个nodes.
    // 如果只对应一个node, 那么我们就确定该计算就很简单, 我们可以做一个优化
    std::vector<int32> epoch_to_num_nodes(num_epoch_indexes, 0);
    // 对每个node 对应的 epoch-index 的node数目++.
    // epoch_to_num_nodes --- epoch-id 对应的多个node数量.
    for (int32 n = 0; n < num_nodes; n++)
      epoch_to_num_nodes[node_to_epoch[n]]++;

    // vector size =  node level 下 epoch 总数
    epoch_is_trivial->resize(num_epoch_indexes);
    // 对每个epoch-id, 设置epoch_is_trival = true 表示琐碎的 只有对应一个node.
    for (int32 o = 0; o < num_epoch_indexes; o++) {
      KALDI_ASSERT(o == 0 || epoch_to_num_nodes[o] > 0);
      (*epoch_is_trivial)[o] = (epoch_to_num_nodes[o] <= 1);
    }

    
    // vector size = cindex 总数
    cindex_id_to_segment_and_epoch->resize(num_cindex_ids);
    KALDI_ASSERT(graph.segment_ends.back() == num_cindex_ids);

    // foreach request的 cindex区域.
    // 对每个request 的cindex push 进入 node对应的epoch中.
    // 实际就是将每个request内的 node的cindex_id 加入到对应request的node对应的epoch中.
    // 这样就将 cindex 安排进入了 epochs.
    int32 cur_segment_start = 0, cur_segment_end;
    for (int32 segment = 0; segment < num_segments; segment++) {
      cur_segment_end = graph.segment_ends[segment];

      // 引用每个request的epochs, 等待进行设置.
      std::vector<std::vector<int32> > &epochs = (*epochs_per_segment)[segment];
      // 设置该request的 epoch大小, 来保存上面计算得到的epoch-id
      epochs.resize(num_epoch_indexes);
      // foreach cindex
      for (int32 cindex_id = cur_segment_start;
           cindex_id < cur_segment_end; cindex_id++) {

        // input 的epoch = 0
        // 其他的epoch 直接设置为 上面的计算结果.
        int32
            node_index = graph.cindexes[cindex_id].first,
            epoch_index = (graph.is_input[cindex_id] ? 0 :
                           node_to_epoch[node_index]);

        // ====================== 如下就得到 foreach request, foreach cindex 的 epoch-c-order. ================
        // 每个cindex_id 设置为 对应的epoch_index + request-id * num_epoch_indexes.
        //  ------------- request1== 0 1 2 3 4 .. 30, request2= 31 32 ... 60.
        (*cindex_id_to_segment_and_epoch)[cindex_id] = epoch_index + segment * num_epoch_indexes;

        // ====================== epochs_per_segment 就是每个request的每个cindex的计算顺序,
        // ====================== epochs_per_segment 保存的每个request下的每个epoch顺序下的,cindex list
        
        // 对每个epoch-id 向对应的epochs[epoch-id] push 对应的 cindex_id.
        // 实际就是将每个 node的cindex_id 加入到node对应的epoch中.
        // 这样就将 cindex 安排进入了 epochs.
        epochs[epoch_index].push_back(cindex_id);
      }
      cur_segment_start = cur_segment_end;
    }
  }


  // 总数每个所有request-segment的 epochs中所有的epoch对应的cindex总数.
  KALDI_ASSERT(SumVectorSizes(epochs_per_segment) == num_cindex_ids);









  // ================= 计算每个 cindex 关于 epoch 计算次序的依赖子集 ===============
  // =======================(为了计算后续cindex的phase计算次序使用)=================
  // dependencies_subset 包含每个cindex_id的依赖的subset子集
  // 是指cindex-id的那些特殊的依赖dep_cindex_id, dep_cindex_id的 epoch_index 和 cindex_id的 epoch_index相同的依赖
  // 这个子集 是用来 在一个确定的epoch中修正排序cindexes的. 像LSTM中需要.
  // 说明这个 dependencies_subset子集 是
  // ------------ cindex依赖dep 与cindex 的phase-index 相等的那些依赖构成的子集.
  std::vector<std::vector<int32> > dependencies_subset;
  ComputeDependenciesSubset(graph, cindex_id_to_segment_and_epoch, &dependencies_subset);

  // 这个函数 输出 dependencies_subset[c], 对每个cindex_id c,
  // graph.dependencies[c]的元素d 的子集 例如两个cindex_id c d
  // 的 cindex_id_to_segment_and_epoch[d]相等.
  // 就是说, graph.dependencies[d] 是 完整计算的依赖图.
  // 但是去掉了 从一个segment/epoch到另一个segment/epoch的链接.???
  // 拓扑结构上, dependencies_subset因此会包含一个不连续的graph的分支.
  static void ComputeDependenciesSubset(
      const ComputationGraph &graph,
      const std::vector<int32> &cindex_id_to_segment_and_epoch,
      std::vector<std::vector<int32> > *dependencies_subset) {
    
    int32 num_cindex_ids = graph.cindexes.size();
    KALDI_ASSERT(cindex_id_to_segment_and_epoch.size() == num_cindex_ids);
    // 对每个cindex_id.
    dependencies_subset->resize(num_cindex_ids);
    for (int32 cindex_id = 0; cindex_id < num_cindex_ids; cindex_id++) {
      // segement-id*epoch_cnt_in_segment + node_index.
      int32 phase_index = cindex_id_to_segment_and_epoch[cindex_id];
      // cindex_id的依赖
      const std::vector<int32> &dependencies = graph.dependencies[cindex_id];
      // denpendencies_subset的引用 准备设置元素数据
      std::vector<int32> &dep_subset = (*dependencies_subset)[cindex_id];
      // 依赖总数 foreach 依赖
      int32 num_dep = dependencies.size();
      for (int32 i = 0; i < num_dep; i++) {
        int32 d = dependencies[i];
        // 如果cindex的依赖d的 phase_index == cindex的phase_index
        // 将将这样的特殊依赖加入到 dep_subset 依赖子集.
        // 说明这个 dependencies_subset子集 是
        // ------------ cindex依赖dep 与cindex 的phase-index 相等的那些依赖构成的子集.
        if (cindex_id_to_segment_and_epoch[d] == phase_index)
          dep_subset.push_back(d);
      }
    }
  }





  // cindex_id_to_segment_and_epoch 已经生成了 dependencies_subset, 不在需要, destroy.
  // destroy cindex_id_to_segment_and_epoch, it's no longer needed.
  {
    std::vector<int32> temp;
    temp.swap(cindex_id_to_segment_and_epoch);
  }






  // ================= 计算每个 cindex 关于 epoch 计算次序的 依赖子集的 反向后继子集 ===============
  // =======================(为了计算后续cindex的phase计算次序使用)=================
 
  // depned_on_subset 是 正常depend_on的子集,
  // 依赖于cindex_id 的后续的所有cindex_ids的list.
  // depend_on_subset 限制为只有那些 具有相同epoch index 的 cindex_ids.
  //  ------- 根据 依赖子集 denpendencies_subset 生成 反向依赖子集depend_on_subset.
  std::vector<std::vector<int32> > depend_on_subset;
  ComputeGraphTranspose(dependencies_subset, &depend_on_subset);
  
  void ComputeGraphTranspose(const std::vector<std::vector<int32> > &graph,
                             std::vector<std::vector<int32> > *graph_transpose) {
    int32 size = graph.size();
    graph_transpose->clear();
    graph_transpose->resize(size);
    // 对每个cindex_id
    for (int32 n = 0; n < size; n++) {
      // 对该cindex_id 的依赖子集. 
      const std::vector<int32> &nodes = graph[n];
      // 对每个 cindex_id 的依赖 dest.
      // 写入为 depend_on_this[dest].push(cindex_id)
      std::vector<int32>::const_iterator iter = nodes.begin(), end = nodes.end();
      for (; iter != end; ++iter) {
        int32 dest = *iter;
        (*graph_transpose)[dest].push_back(n);
      }
    }
  }

  



  int32
      // epoch总数 per segment
      num_epoch_indexes = epoch_is_trivial.size(),
      // segment-request总数
      num_segments = graph.segment_ends.size();

  // phases_indexes vector size = cindex_cnt
  // "phase_indexes" is used inside ComputeComputationPhasesForEpoch.
  std::vector<int32> phase_indexes(num_cindex_ids, -1);

  // 设置 每个request segment的phase vector设置大小 request大小.
  phases_per_segment->clear();
  phases_per_segment->resize(num_segments);

  // ==================================================================================================
  // ================= 计算  每个reqeust-segment 的每个epoch计算次序 的Cindex的 phase 计算次序=========
  // ==================================================================================================
  // foreach reqeust-segment.
  for (int32 segment = 0; segment < num_segments; segment++) {

    phases_per_segment->reserve(50);  // minimize unnecessary copies.  50 is
                                      // very arbitrarily chosen.
    // 对每个 epoch-index.
    // 根据
    // 1 每个 segment 下 epoch-index 下 对应的 cindex list
    // 2 cindex 的依赖子集 (cindex_id, cindex_id_dep 具有相同的 segment*epoch_cnt + epoch-index)
    // 3 cindex 的后续子集
    // 4 每个 epoch-index 的 cindex个数是否琐碎
    // out:
    // 1 phases_indexes 是每个 cindex的 phase-index
    // 2 phases_per_segment 是该 segment 下的 ???
    for (int32 epoch = 0; epoch < num_epoch_indexes; epoch++)
      ComputeComputationPhasesForEpoch(nnet, graph,
                                       epochs_per_segment[segment][epoch],
                                       dependencies_subset,
                                       depend_on_subset,
                                       epoch_is_trivial[epoch],
                                       &phase_indexes,
                                       &((*phases_per_segment)[segment]));
  }

  // =========================================================
  // ==================== 至此 ===============================
  // 将 每个request-segment 的 Cindex
  // 1 首先计算了 epoch计算次序 是一个粗略的计算次序  -- 结果是 epochs_per_segemnt
  // 2 然后通过epoch 计算了 phase计算次序 实现了 phase计算次序.
  // =========================================================


  // make sure everything was computable.  If the next assert fails it's likely
  // a bug in this function or in PruneComputataionGraph.
  KALDI_ASSERT(SumVectorSizes(*phases_per_segment) == num_cindex_ids);
}


/*

  这个函数 从ComputeComputationPhases调用
  这个函数 处理 从一个 epoch的计算. 这段代码是为了防止 调用者太长了.
  注意, phases 是一个 有id的组, 包含cindexes, 就是说 在这样num-id的顺序下, 我们计算东西,
  即 我们首先计算所有的 phase-0 的cindexes, 然后是 phase-1的 cindexes ...
  
  this function is called from ComputeComputationPhases; it handles the part of
  the computation from one epoch (this code was broken out to avoid that
  function being super-long).  Note: the phases are a numbered grouping of
  cindexes that say in what order we compute things, i.e. we first compute
  all the cindexes for phase 0, then for phase 1, and so on.

   @param [in] nnet       The neural net this computation is for
   @param [in] graph      The computation graph we're computing the phases for.

   当前epoch 的 排好序的 cindex_ids list
   Note cindex_ids 是 graph.cindexes的索引
   简单的说, 是一个cindex_ids的list 对应为 LSTM 的 nnet的一个layer
   或者是 TDNN 这样的 nnet的一个layer的一部分
   @param [in] this_epoch The sorted list of the cindex_ids for this epoch; note,
                          cindex_ids are indexes into the array graph.cindexes.
                          Roughly speaking, this is a list of the cindex_ids that
                          correspond to one "layer" of the neural network, in
                          things like LSTMs, or for one part of one layer (the
                          affine component, the nonlinearity, or the splicing),
                          in things like TDNNs.

  dependencies_subset graph.dependencies 的一个子集
  一般上, 对一个cindex_id c 的依赖dependencies 是一个用来计算c的d1 d2的 list
  因此需要首先计算 d1 d2 和 d1+d2等.
  @param [in] dependencies_subset  A subset of 'graph.dependencies' corresponding
                          just to dependencies within the same epoch (not specifically
                          this epoch; for all epochs).  In general, for a cindex_id c
                          dependencies[c] is a list of other cindex_ids d1, d2,
                          such that in order to compute c we must first compute
                          d1, d2 and so on (plus d1, d2, etc. must be from the
                          same epoch as c).
  depends_on_subset 是 dependencies_subset的图转换 对一个 cindex_id c depends_on_subset[c]
  是一个cindex_ids 的list 直接依赖于cindex_id c 的后续. 所以 c 一定在计算他们之前先计算.
  @param [in] depends_on_subset  The graph-transpose of dependencies_subset;
                          for cindex_id c, depends_on_subset[c] is the list
                          of cindex_ids that directly depend on cindex_id c,
                          so c must be computed before them.
  epoch_is_trivial 是true false的list, 一个epoch 只具有一个component时 true.
  这个list能够使用快速code路径 在一般情况下.
  @param [in] epoch_is_trivial  A bool that's true if this epoch is trivial
                          (meaning it consists of just one component)... this
                          enables a faster code path in this common case.
  phase_indexes 是个vector, 其中的一些元素 这个函数每次调用都写一次.
  映射 cindex_id to phase index. 一个 phase index 是计算的一个phases的编号的标示(像一个简陋的步骤)
  0 是第一个phase     1 是第二phase.
  我们通过phase->size() 计算出 之前的epochs 已经用了多少 phase indexes.
  实际上, phase_indexes 这个函数用的 一个简单的时间变量, 为了高校我们在外面申请.
  初始化为-1, 这个函数的不同的调用 与不同的非重叠的元素 一起工作.
  @param [in,out] phase_indexes  This vector, to some elements of which this
                          function writes each time it is called, maps from
                          cindex_id to the 'phase index'.  A phase index is a
                          number identifying the phases [like coarse steps] of
                          the computation, with zero for the first phase, one
                          for the second, etc.  We work out how many phase
                          indexes have been used already by previous epochs,
                          from phases->size().  Actually, phase_indexes is
                          really just a temporary variable used by this
                          function, that we allocate outside this function for
                          efficiency.  It is initialized to -1 outside this
                          function; different invocations of this function work
                          with different non-overlapping elements of the vector.


  这是本函数的输出, 每次我们增加一个phase, 我们向*phase增加一个vector.
  eg (*phases)[0] 是在计算中的第一个phase中的 被排序的cindexes 的list

  Note, 这个函数会被调用多次, 每次我们都向这个vector 增加一个或多个 phases,
  
  1 当epoch琐碎的时候, 增加一个phase
  2 当epoch不琐碎情况, 在内部就通过
    while(!this_phases.empty())
      phases.resize(phases.size() + 1)
      next_phases_candicates -> this_phases,
    循环的增长.
      
  这个phases每次都增长一下size.
  @param [in,out] phases This is the output of this
                          function.  Each time we add a new phase, we append a
                          vector to *phases.  E.g. (*phases)[0] is the sorted
                          list of cindexes in the first phase of the
                          computation... and so on.  Note, this function is
                          called multiple times, and each time we add one or
                          more phases to this vector, so its size grows.
*/
static inline void ComputeComputationPhasesForEpoch(
    const Nnet &nnet,
    const ComputationGraph &graph,
    
    const std::vector<int32> &this_epoch,
    const std::vector<std::vector<int32> > &dependencies_subset,
    const std::vector<std::vector<int32> > &depend_on_subset,
    bool epoch_is_trivial,
    std::vector<int32> *phase_indexes,
    std::vector<std::vector<int32> > *phases) {



  
  std::vector<int32> this_phase,
      next_phase_candidates;

  if (this_epoch.empty())
    return;

  // 如果是个trivial 琐碎的(只具有一个cindex) 直接使用epoch 作为phase.
  // 即 当前epoch-计算次序 的所有cindex 都直接是相同的phase计算次序
  if (epoch_is_trivial) { // an optimization
    this_phase = this_epoch;
  } else {
    // 这个epoch计算次序的所有cindex_id
    // cindex_id的依赖子集为空, 则将cindex_id 加入到当前 phase计算次序

    std::vector<int32>::const_iterator
        iter = this_epoch.begin(),
        end = this_epoch.end();
    
    for (; iter != end; ++iter) {
      int32 cindex_id = *iter;
      // cindex_id的依赖子集 为空, 将cindex_id 加入phase 先计算. 
      if (dependencies_subset[cindex_id].empty())
        this_phase.push_back(cindex_id);
    }
  }

  

  // 如果如下的 assert 失败, 说明这个图 不是无环的.
  // if the next assert fails, the graph at the level of cindex_ids is not acyclic.
  KALDI_ASSERT(!this_phase.empty() &&
               "Trying to process computation with cycles");

  // 根据 依赖关系 循环向后 计算后续cindex 的次序.
  while (!this_phase.empty()) {
    // phases->push_back(this_phase);
    phases->resize(phases->size() + 1);
    phases->back().swap(this_phase);
    // The next if-statement is an optimization: if for this epoch index
    // there is just one node, we can skip the rest of this loop.  Note: if
    // epoch == 0, even if there is just one node, cindex_ids from
    // multiple nodes may be put here because of the rule that cindex_ids which
    // are inputs always get epoch 0.  But it's still true that they
    // will have no dependencies, so we can still skip the code below.
    if (epoch_is_trivial)
      return;

    // 当前计算次序.
    int32 cur_phase_index = phases->size() - 1;

    // next_phases_candidates 下一个计算次序候选 是一个cindexes list, 我们应该检查他们是否可计算.
    // 他们中的依赖 刚刚变成可计算的.
    next_phase_candidates.clear();

    // =============== 设置当前计算次序的cindex 的 计算次序数组保存计算次序 = cur_phase_index =============
    std::vector<int32>::const_iterator
        this_phase_iter = phases->back().begin(),
        this_phase_end = phases->back().end();
    for (; this_phase_iter != this_phase_end; ++this_phase_iter) {
      // 每个 当前phase计算次序的 cindex
      int32 c = *this_phase_iter;  // c is a cindex_id with phase cur_phase_index.
      // 设置 cindex c 的计算次序 为 cur_phase_index.
      (*phase_indexes)[c] = cur_phase_index;

      // ========== 根据依赖关系 计算cindex c的后续 cindex d 加入下个次序候选 =====
      // 所有依赖于当前计算次序 cur_phase_index cindex c 的后续
      // cindex n 加入到 下个计算次序候选 next_phase_candidates 
      std::vector<int32>::const_iterator
          iter = depend_on_subset[c].begin(),
          end = depend_on_subset[c].end();
      for (; iter != end; ++iter) {
        int32 d = *iter;  // cindex_id that depends on c.
        next_phase_candidates.push_back(d);
      }
    }



    
    SortAndUniq(&next_phase_candidates);

    // 根据依赖关系 继续向后计算 当前计算次序下 cindex 的后续的 cindex 的计算次序.
    // 上面已经将this_phase 清空, 下面用来处理 下一个计算次序.
    this_phase.reserve(next_phase_candidates.size());

    // 当前计算次序 cindex 的后续 cindex.
    std::vector<int32>::const_iterator iter = next_phase_candidates.begin(),
        end = next_phase_candidates.end();
    for (; iter != end; ++iter) {
      // 后续cindex 
      int32 c = *iter;
      std::vector<int32>::const_iterator
          dep_iter = dependencies_subset[c].begin(),
          dep_end = dependencies_subset[c].end();
      // 后续cindex 的依赖 Cindex 
      for (; dep_iter != dep_end; ++dep_iter) {
        int32 d = *dep_iter;  // d is cindex_id that c depends on.
        // 如果还有还有 依赖的cindex 没被安排phase计算次序, 则这个cindex 现在还不能具有计算次序.
        if ((*phase_indexes)[d] < 0)  // we can't compute c yet because something we depend
          break;                      // on has not yet been computed.
      }
      // 如下判断是否 迭代到最后一个依赖,
      // true 表示所有依赖都有计算次序了, 那么可以计算该 后续cindex 的 计算次序. =====> this_phase
      if (dep_iter == dep_end) {
        // we reached the end and did not break -> all dependencies satisfied
        this_phase.push_back(c);
      }
    }

    // 经过上面 计算 后续cindex 的计算次序, next_phases_candidates不为空&& this_phase为空,
    // 说明 next_phases_candidates 中全部都具有向后t+1的依赖, 这样就导致后面t+1 需要依赖t ,形成循环.
    if (!next_phase_candidates.empty() && this_phase.empty())  {
      // this should have been caught earlier so likely a code error rather than
      // a problem with user input.
      KALDI_ERR << "Your model has a type of recurrence that cannot be computed. "
                << "E.g. if x[t] depends on both x[t+1] and x[t-1]... no order "
                << "of computation will work.";
    }
  }
}
