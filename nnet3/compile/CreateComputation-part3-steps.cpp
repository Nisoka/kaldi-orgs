


// ============================== steps_ , locations_ ========================
// steps_ 顺序保存 每个segment的 每个phase的 每个sub_phase的 所有cindexes.
// steps_ 的顺序 等同于locations .
// locations_ 就是最终的计算步骤, locations_[i] 对应每个 cindex, 描述每个cindex的计算次序.

ComputationStepsComputer::ComputationStepsComputer(
    const Nnet &nnet,
    ComputationGraph *graph,
    std::vector<std::vector<int32> > *steps,
    std::vector<std::pair<int32, int32> > *locations):
    nnet_(nnet), graph_(graph), steps_(steps), locations_(locations){
  
  steps_->clear();
  locations_->clear();
  int32 num_cindexes = graph_->cindexes.size();
  // leave a little space in case a few cindexes are added (unlikely
  // but could happen with dim-range nodes).
  locations_->reserve(num_cindexes + num_cindexes / 10);
  locations_->resize(num_cindexes, std::pair<int32,int32>(-1, -1));
  
}


// 为某个request 将对应的 每个phase计算次序的cindex_ids 进行更细致划分

//in: 对每个phase
// <cindex-id1, cindex-id2, ... cindex-id100>

//实际 sub_phase 就是step了.
//out: 对每个phase -- <sub_phase, sub_pahse, ... sub_phase>
//      实际就是将原本phase中的所有cindex 按照 node-index 又进行了划分, 所以相同的sub_phase 都是同一个node的.
// ---- < <cindex-id1, cindex-id2>, <cindex-id3, cindex-id4> ... <cindex-id98, cindex-id99, cindex-id100>>

//结果: 向locations 中根据每个 sub_phase -- step 生成对应的locations

void ComputationStepsComputer::ComputeForSegment(
    const ComputationRequest &request,
    const std::vector<std::vector<int32> > &phases) {
  
  int32 this_num_phases = phases.size();
  // 每个计算顺序
  // ( 顺序1<cindex-id1, cindex-id2, ... >    顺序2<cindex-id32, cindex-id33>   顺序3<cindex-id54, cindex-id55> )
  for (int32 i = 0; i < this_num_phases; i++) {
    std::vector<std::vector<Cindex> > sub_phases;
    SplitIntoSubPhases(phases[i], &sub_phases);
    // 对当前的更低级的划分
    for (size_t j = 0; j < sub_phases.size(); j++) {
      ProcessSubPhase(request, sub_phases[j]);
    }
  }
}


void ComputationStepsComputer::SplitIntoSubPhases(
    const std::vector<int32> &phase,
    std::vector<std::vector<Cindex> > *sub_phases) const {
  
  std::vector<Cindex> phase_cindexes;
  // 1 先简单将 phase 中保存的 cindex-id 转化为 cindex对象.
  ConvertToCindexes(phase, &phase_cindexes);

 


  KALDI_ASSERT(!phase_cindexes.empty());
  std::sort(phase_cindexes.begin(), phase_cindexes.end());
  


  // segment_begins 是 phase_cindex 的划分, node-index 相同的划分进一个segment中
  // 不相同的 划分进后续的segment中.
 
  std::vector<size_t> segment_begins;
  int32 cur_node_index = -1;
  size_t size = phase_cindexes.size();
  
  for (size_t i = 0; i < size; i++) {
    if (phase_cindexes[i].first != cur_node_index) {
      cur_node_index = phase_cindexes[i].first;
      segment_begins.push_back(i);
    }
  }


  size_t num_sub_phases = segment_begins.size();
  segment_begins.push_back(size);

  // 总共划分 数量 --- 实际应该就是node 数量.
  sub_phases->clear();
  sub_phases->resize(num_sub_phases);

  // 对每个划分segment.
  // sub_phases 又对原本的 phase按照node 顺序进行了一下 划分.
  for (size_t i = 0; i < num_sub_phases; i++) {
    size_t this_begin = segment_begins[i],
        this_end = segment_begins[i+1];
    
    (*sub_phases)[i].insert((*sub_phases)[i].end(), phase_cindexes.begin() + this_begin, phase_cindexes.begin() + this_end);
    
  }
}




void ComputationStepsComputer::ProcessSubPhase(
    const ComputationRequest &request,
    const std::vector<Cindex> &sub_phase) {
  
  KALDI_ASSERT(!sub_phase.empty());


  int32 node_index = sub_phase[0].first;
  // 断言, sub_phase 中的 cindex 都 属于 相同 node
  KALDI_ASSERT(sub_phase.back().first == node_index);
  
  if (nnet_.IsComponentNode(node_index)) {
    ProcessComponentStep(sub_phase);
    
  } else if (nnet_.IsInputNode(node_index)) {
    ProcessInputOrOutputStep(request, false, sub_phase);
    
  } else if (nnet_.IsOutputNode(node_index)) {
    ProcessInputOrOutputStep(request, true, sub_phase);
    
  } else if (nnet_.IsDimRangeNode(node_index)) {
    // this might turn out to be multiple steps, see the code.
    ProcessDimRangeSubPhase(sub_phase);
    
  } else if (nnet_.IsComponentInputNode(node_index)) {
    // We actually do nothing with these sub-phases, because they are processed
    // when we process the associated component's sub-phase/step.  Doing it this
    // way resolves certain problems.
    return;
  } else {
    KALDI_ERR << "Unknown node type.";
  }
}












void ComputationStepsComputer::ProcessInputOrOutputStep(
    const ComputationRequest &request,
    bool is_output,
    const std::vector<Cindex> &sub_phase) {

  
  int32 io_node = sub_phase[0].first;
  if (is_output){
    KALDI_ASSERT(nnet_.IsOutputNode(io_node));
  } else {
    KALDI_ASSERT(nnet_.IsInputNode(io_node));
  }

  // input or output
  std::string node_name = nnet_.GetNodeName(io_node);

  // 直接获得对应的 request input的NnetIo.
  const std::vector<IoSpecification>
      &inputs_or_outputs = (is_output ? request.outputs : request.inputs);


  // 对于 input 可能有 1 input_mfcc, 2 input_ivector, 就是看看具体是哪个.
  int32 io_index = -1;
  for (size_t i = 0; i < inputs_or_outputs.size(); i++)
    if (inputs_or_outputs[i].name == node_name)
      io_index = i;

  
  KALDI_ASSERT(io_index >= 0);

  // 获得所有 input_mfcc的 Indexes.
  const std::vector<Index> &io_indexes = inputs_or_outputs[io_index].indexes;

  // 对应构建 cindexes
  std::vector<Cindex> io_cindexes(io_indexes.size());
  for (size_t i = 0, size = io_cindexes.size(); i < size; i++) {
    io_cindexes[i].first = io_node;
    io_cindexes[i].second = io_indexes[i];
  }

  // 必然要和 sub_phases 相等
  KALDI_ASSERT(io_cindexes.size() == sub_phase.size());
  // we expect the list of cindexes in 'io_cindexes' to be identical to
  // that in 'sub_phase' (but they don't have to be in the same order)... for now we check the size, we'll spot-check
  // that they are the same later.

  

  // 每个cindex 对应 一个 location的 pair <step_index, row_index>
  //                                      <sub_phase,  row_index>
  //                                      <node_index, row_index>

  // The actual output in 'steps' must be in the same order as
  int32 step_index = AddStep(io_cindexes);

  // 这里按每个点进行详细检查, sub_phase中的cindexes 都会安排到具体的 locations_中.

  
  // Now spot-check that the cindexes in 'sub_phase' are the same as those
  // we just added.  [note: they don't have to be in the same order, but
  // they should be the same set.]
  for (size_t i = 0; i < sub_phase.size(); i += 10) {
    const Cindex &cindex = sub_phase[i];
    int32 cindex_id = graph_->GetCindexId(cindex);
    KALDI_ASSERT(cindex_id >= 0 && (*locations_)[cindex_id].first == step_index);
  }
  
}






int32 AddStep(const std::vector<Cindex> &cindexes, bool add_if_absent = false);
int32 ComputationStepsComputer::AddStep(const std::vector<Cindex> &cindexes,
                                        bool add_if_absent) {
  // note: we can't assert that cindexes is nonempty, because it's possible for
  // input steps for GeneralComponents to be empty if they require no input
  // indexes; and because the compiler code expects component steps to be
  // preceded by component-input steps, we can't just omit these empty steps.
  // [note: a component-input step is about preparing the input for a component's
  // propagation.]

  // 增加一个 sub_phase 的计算步骤 -- 实际上这个sub_phase 就是step了.
  // 一个sub_phase 是一个 cindexes vector ---- 即输入 cindexes.
  int32 step_index = steps_->size();
  steps_->push_back(std::vector<int32>());
  
  std::vector<int32> &step = steps_->back();  // vector of cindex_id.
  step.resize(cindexes.size());

  
  size_t row_index = 0;
  std::vector<Cindex>::const_iterator
      iter = cindexes.begin(),
      end = cindexes.end();
  
  std::vector<int32>::iterator out_iter = step.begin();

  
  std::pair<int32, int32> *locations = &((*locations_)[0]);
  if (!add_if_absent) {
    // 这个版本的GetCindexId 与以前碰到的不一样, 这个就是简单的在graph中 查找相同的 cindex
    // 如果没找到, 返回-1, 后果严重, 后面的操作可能直接会导致崩溃!!!
    // 但是一定会找到的!!!!!
    for (; iter != end; ++iter, ++out_iter, ++row_index) {
      int32 cindex_id = graph_->GetCindexId(*iter);
      *out_iter = cindex_id;
      // locations ======================
      // 保存的就是 step 计算次序 更加的详细 1 sub_phase--step_index  2 row_index.
      locations[cindex_id].first = step_index;
      locations[cindex_id].second = row_index;
    }
  } else {
    for (; iter != end; ++iter, ++out_iter, ++row_index) {
      bool is_input = false;  // only relevant if we have to add the cindex to
                              // the computation graph, which we won't for
                              // inputs (we only might for dim-range nodes
                              // and for the component-input and component
                              // steps of non-simple Components.
      bool added;
      int32 cindex_id = graph_->GetCindexId(*iter, is_input, &added);
      *out_iter = cindex_id;
      if (added) {
        KALDI_ASSERT(cindex_id == static_cast<int32>(locations_->size()));
        locations_->resize(cindex_id + 1);
        locations_->back().first = step_index;
        locations_->back().second = row_index;
        locations = &((*locations_)[0]);  // in case it was reallocated
      } else {
        locations[cindex_id].first = step_index;
        locations[cindex_id].second = row_index;
      }
    }
  }
  return step_index;
}






int32 ComputationStepsComputer::AddStep(std::vector<int32> *cindex_ids) {
  int32 step_index = steps_->size();
  steps_->push_back(std::vector<int32>());
  
  // 向steps_ 中保存当前 sub_phase 的所有cindexes .
  steps_->back().swap(*cindex_ids);
  
  std::vector<int32>::const_iterator iter = steps_->back().begin(),
      end = steps_->back().end();
  int32 row_index = 0;
  std::pair<int32,int32> *locations = &((*locations_)[0]);
  size_t num_cindexes = graph_->cindexes.size();
  for (; iter != end; ++iter, ++row_index) {
    int32 cindex_id = *iter;
    KALDI_ASSERT(static_cast<size_t>(cindex_id) < num_cindexes);
    locations[cindex_id].first = step_index;
    locations[cindex_id].second = row_index;
  }
  return step_index;
}




void ComputationStepsComputer::ProcessComponentStep(
    const std::vector<Cindex> &step) {
  
  KALDI_ASSERT(!step.empty());

  // 对应的component_node_index
  int32 component_node_index = step.front().first;
  // 对应component_node 的 kDescriptor 类型 node
  int32 component_input_index = component_node_index - 1;

  
  KALDI_ASSERT(nnet_.IsComponentNode(component_node_index));
  
  const NetworkNode &node = nnet_.GetNode(component_node_index);

  // 获得对应的 Component.
  int32 c = node.u.component_index;
  const Component *component = nnet_.GetComponent(c);


  // 对 简单Component 
  if (component->Properties() & kSimpleComponent) {
    // 对 简单component, 对应的输入cindexes 就是 component的kDescriptor,
    // 这两者的cindex 除了node-index不同, 剩下都一样, 快速处理.
    
    std::vector<Cindex> input_step(step.size());
    input_step.resize(step.size());
    
    std::vector<Cindex>::iterator
        iter = input_step.begin(),
        end = input_step.end();
    // 对 kDescriptor的 cindexes 都修改下node-index 即可.
    std::vector<Cindex>::const_iterator src = step.begin();
    for (; iter != end; ++iter,++src) {
      iter->first = component_input_index;
      iter->second = src->second;
    }
    
    AddStep(input_step);
    AddStep(step);
  }


  
  // 对 复杂Component , 这个就有点复杂了.
  else {
    std::vector<int32> step_cindex_ids;
    ConvertToCindexIds(step, &step_cindex_ids);
    // to get the input cindexes we need to follow dependencies back.
    unordered_set<int32> input_cindex_ids;
    std::vector<int32>::iterator iter = step_cindex_ids.begin(),
        end = step_cindex_ids.end();
    for (; iter != end; ++iter) {
      int32 c = *iter;
      const std::vector<int32> &dependencies = graph_->dependencies[c];
      std::vector<int32>::const_iterator dep_iter = dependencies.begin(),
          dep_end = dependencies.end();
      for (; dep_iter != dep_end; ++dep_iter) {
        int32 d = *dep_iter;
        input_cindex_ids.insert(d);
      }
    }
    // Convert to Cindexes so we can sort them as Cindexes.
    std::vector<Cindex> input_step;
    input_step.reserve(input_cindex_ids.size());
    unordered_set<int32>::iterator set_iter = input_cindex_ids.begin(),
        set_end = input_cindex_ids.end();
    for (; set_iter != set_end; ++set_iter) {
      int32 c = *set_iter;
      input_step.push_back(graph_->cindexes[c]);
    }

    // sort the input cindexes.
    std::sort(input_step.begin(), input_step.end());

    if (component->Properties() & kReordersIndexes) {
      std::vector<Index> indexes, input_indexes;
      ConvertToIndexes(input_step, &input_indexes);
      ConvertToIndexes(step, &indexes);


      size_t orig_size = indexes.size() + input_indexes.size();

      // the component wants to have the opportunity to change the
      // order of these indexes from their default.
      component->ReorderIndexes(&input_indexes, &indexes);

      bool added_padding = (orig_size != indexes.size() + input_indexes.size());

      // Now convert back from indexes to cindexes (we know the
      // node-index in each case)
      std::vector<Cindex> reordered_step;
      ConvertToCindexes(indexes, component_node_index, &reordered_step);
      ConvertToCindexes(input_indexes, component_input_index, &input_step);
      // the 'added_padding' argument becomes the 'add_if_absent' arg of
      // AddStep, so it knows to expect that it might have to add new CindexIds.
      AddStep(input_step, added_padding);
      AddStep(reordered_step, added_padding);
    }




    else {
      AddStep(input_step);
      // it's more efficient to add the step with cindex_ids; and we have these
      // available, so we do it that way.  (in the other branch where
      // the flag kReordersIndexes was present, we couldn't do this because
      // of the reordering).
      AddStep(&step_cindex_ids);
    }
  }
}





void ComputationStepsComputer::Check() const {
  
  int32 num_cindexes = graph_->cindexes.size();
  KALDI_ASSERT(locations_->size() == num_cindexes);
  
  for (int32 c = 0; c < num_cindexes; c++) {
    int32 step = (*locations_)[c].first,
        row = (*locations_)[c].second;
    if (!(step >= 0 && row >= 0 && (*steps_)[step][row] == c)) {
      // normally the 'locations' of cindexes should be unique, so we should
      // never normally reach this point; but it's not an error to have
      // duplicates of the cindexes used for 'padding' by the ReorderIndexes()
      // function of non-simple Components.  So we check whether that's the case
      // before we die.
      if (graph_->cindexes[c].second.t != kNoTime) {
        // if this happens it will likely require some debugging by Dan.
        KALDI_ERR << "Error in computing computation steps (likely code error)";
      }
    }

  }
}
