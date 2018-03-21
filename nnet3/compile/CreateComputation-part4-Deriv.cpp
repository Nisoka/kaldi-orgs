

void Compiler::ComputeDerivNeeded(
    const std::vector<std::vector<int32> > &steps,
    const std::vector<int32> &step_to_segment,
    std::vector<bool> *deriv_needed) {

  
  KALDI_ASSERT(steps.size() == step_to_segment.size() &&
               step_to_segment[0] == 0 &&
               step_to_segment.back() + 1 == requests_.size());


  // 所有steps
  deriv_needed->clear();
  int32 num_steps = steps.size();
  deriv_needed->resize(num_steps, false);

  for (int32 step = 0; step < num_steps; step++) {
    const std::vector<int32> &this_step = steps[step];
    if (this_step.empty())  // empty steps are theoretically possible, e.g.
      continue;             // if a non-simple Component requires no input.
    int32 cindex_id = this_step[0];
    int32 node_index = graph_.cindexes[cindex_id].first;
    bool is_input = graph_.is_input[cindex_id];

    std::string node_name = nnet_.GetNodeNames()[node_index];

    // 一个step的依赖input_steps 是step内所有cindex 的依赖dep_cindexes的 step-index的集合.
    // 计算 step的依赖dep_step_index.
    unordered_set<int32> input_steps;
    ComputeStepDependencies(this_step, step, &input_steps);
    
    void Compiler::ComputeStepDependencies(
        const std::vector<int32> &this_step,
        int32 step_index,
        unordered_set<int32> *dep_steps) {
  
      dep_steps->clear();
      if (this_step.empty())
        return;

  
      // steps always have a single node index, we can pick the first.
      int32 node_index = graph_.cindexes[this_step[0]].first;
      if (nnet_.IsComponentNode(node_index)) {
        // there is only one step that a component step depends on, and it's the
        // immediately preceding step (the component-input step).
        KALDI_ASSERT(step_index > 0);
        dep_steps->insert(step_index - 1);
        return;
      }

  
      std::vector<int32>::const_iterator
          step_iter = this_step.begin(),
          step_end = this_step.end();

      // 为了速度的优化.
      int32 prev_input_step = -1;  // this is an optimization for speed.



      // =============== 一个step 的依赖 dep_steps, 是step内所有cindex的依赖dep_cindexes 的step-index的set ==================
      // step内的每个cindex
      for (; step_iter != step_end; ++step_iter) {
    
        int32 cindex_id = *step_iter;
        // cindex的依赖 cindexes
        const std::vector<int32> &dep = graph_.dependencies[cindex_id];
    
        std::vector<int32>::const_iterator iter = dep.begin(), end = dep.end();
        for (; iter != end; ++iter) {
          int32
              dep_cindex_id = *iter,
              // cindex_id_to_location_ 就是 ComputationStepsComputer的 locations_
              // 每个cindex的 location[cindex-id].first 就是 step_index.
              input_step = cindex_id_to_location_[dep_cindex_id].first;

          if (input_step != prev_input_step) {  // optimization.
            prev_input_step = input_step;
            dep_steps->insert(input_step);
          }
        }
      }
    }


    // 如果step 有依赖的dep_step需要导数, 那么这个step 需要求导.
    unordered_set<int32>::iterator iter = input_steps.begin(),
        end = input_steps.end();
    // if some step that we depend on needs a derivative, we need the derivative.
    for (; iter != end; ++iter) {
      int32 dep_step = *iter;
      KALDI_ASSERT(dep_step < step);
      if ((*deriv_needed)[dep_step])
        (*deriv_needed)[step] = true;
    }


    // 获得该step 对应的 segment- request_index. 获得reqeust.
    // 如果step是输入, 用户要求需要对该输入的导数, 我们就需要计算导数

    const ComputationRequest &request = *(requests_[step_to_segment[step]]);

    if (is_input) {
      int32 input_index = request.IndexForInput(node_name);
      KALDI_ASSERT(input_index != -1);
      if (request.inputs[input_index].has_deriv)
        (*deriv_needed)[step] = true;
    }

    // 如果step 是输出, 并且用户提供了导数, 对该输出我们需要一个位置 保存导数, 这样也会设置 deriv_needed=true.

    if (nnet_.IsOutputNode(node_index)) {
      int32 output_index = request.IndexForOutput(node_name);
      KALDI_ASSERT(output_index != -1);
      if (request.outputs[output_index].has_deriv)
        (*deriv_needed)[step] = true;
    }

    // 如果是一个 updatable Component node 具有 非0学习率, 并且用户需要模型导数(如训练时)
    // 我们就需要计算导数

    if (nnet_.IsComponentNode(node_index) && request.need_model_derivative) {
      const NetworkNode &node = nnet_.GetNode(node_index);
      const Component *c = nnet_.GetComponent(node.u.component_index);
      if (c->Properties() & kUpdatableComponent) {
        const UpdatableComponent *u = dynamic_cast<const UpdatableComponent*>(c);
        KALDI_ASSERT(u != NULL);
        if (u->LearningRate() != 0)
          (*deriv_needed)[step] = true;
      }
    }
  }

}













// ===========================================================================================
// 计算 某个 类型是kDescriptor node 的StepInfo的所有cindexes 的所有依赖dep_cindexes 的计算位置
// ===========================================================================================

// 实际上拷贝对应依赖dep_cindexes 的 cindex_id_to_locations_(前面计算好的) .

// 输出向量 locations 是
// 1 先按照 row-index 索引 - 即output_indexes索引
// 2 然后是对那个row-index 的 一系列的输入位置 进行索引
// 语法是 输出的第i行变成 第i个list 的行的总和.
// locations 结果是 <step-index, row-index> 形式的pair

void Compiler::ComputeInputLocationsList(
    int32 step,
    int32 part_index,
    std::vector<std::vector<std::pair<int32, int32> > > *submat_locations_list) const {

  KALDI_ASSERT(static_cast<size_t>(step) < steps_.size());

  const StepInfo &step_info = steps_[step];
  const std::vector<Index> &output_indexes = step_info.output_indexes;
  const NetworkNode &node = nnet_.GetNode(step_info.node_index);

  const SumDescriptor &descriptor = node.descriptor.Part(part_index);

  // 该 StepInfo的输出indexes 总数.
  // submat_locations_list 保存每个indexes的 计算位置信息.
  int32 num_indexes = output_indexes.size();
  submat_locations_list->clear();
  submat_locations_list->resize(num_indexes);

  
  for (int32 i = 0; i < num_indexes; i++) {
    // 引用对应的index, 以及submat_locations_list, 对submat_locations_list[index-id]进行配置.
    const Index &index = output_indexes[i];
    std::vector<std::pair<int32, int32> > &this_locations_list =
        (*submat_locations_list)[i];
    
    if (index.t != kNoTime) {
      // a real Index, not a 'blank' one
      // ('blank' indexes are inserted by some non-simple Components to
      // satisfy internal constraints.
      std::vector<int32> input_cindex_ids;
      std::vector<Cindex> input_cindexes;
      CindexSet cindex_set(graph_);
      
      // ==============  计算当前index的依赖 --> input_cindexes.=============
      // ==============  计算当前index的依赖 --> input_cindexes.=============
      // ==============  计算当前index的依赖 --> input_cindexes.=============
      bool ans = descriptor.IsComputable(index, cindex_set, &input_cindexes);
      
      // earlier compilation stages should have checked that it is computable,
      // and the graph should still contain required inputs.
      KALDI_ASSERT(ans);

      
      std::sort(input_cindexes.begin(), input_cindexes.end());

      // index的所有依赖cindexes 的 cindex_id
      int32 size = input_cindexes.size();
      input_cindex_ids.resize(size);
      for (int32 j = 0; j < size; j++) {
        int32 c = graph_.GetCindexId(input_cindexes[j]);
        KALDI_ASSERT(c != -1);
        input_cindex_ids[j] = c;
      }

      // 将以前计算的 cindex_id_to_location_ 的位置 copy过来即可.
      this_locations_list.resize(size);
      for (int32 j = 0; j < size; j++)
        this_locations_list[j] = cindex_id_to_location_[input_cindex_ids[j]];

      
    } else {
      this_locations_list.clear();
    }
  }
}

//  ======================  为每个step 构建StepInfo 总和计算步骤信息 ============
// 1 为每个step 构建成为 StepInfo对象,

// 2 为每个StepInfo.value    申请数据 计算空间 computation->NewMatrix()
//                                    计算空间都申请在computation中.
// 3 根据每个step的 deriv_need 需求 构建StepInfo.deriv  申请导数计算空间

// 4 对kDescriptor 类型的step 构建的StepInfo保存对应 step的所有part的所有 dep_cindexes 的locations
//      1 保存在StepInfo.input_locations_list[part]  -- vector<    vector <      pair<int32, int32> > > 
//      ----------------------------------------------- indexes    dep_cindexes  location
//      2 每个part 都是value的一部分 用cur_dim_offset 区分.
//         this_info.value_parts[p] = computation->NewSubMatrix(this_info.value, 0, -1, cur_dim_offset, this_dim);

// by_step 就是计算好的steps, 这里将计算步骤加入到computation之后, 被destroy.
void Compiler::CreateStepInfo(
    const std::vector<bool> &deriv_needed,
    const std::vector<int32> &step_to_segment,
    std::vector<std::vector<int32> > *by_step,
    NnetComputation *computation) {
  
  KALDI_ASSERT(!by_step->empty());

  // Compiler 保存的step 是steps_ --  vector<StepInfo>
  // 如下将上面计算的step 构建成 Compiler->steps_.
  int32 num_steps = by_step->size();
  steps_.resize(num_steps);
  
  for (int32 step = 0; step < num_steps; step++) {

    StepInfo &this_info = steps_[step];
    // steps_[step] - StepInfo.output_cindex_ids 是个list
    // 保存对应by_step[step]下的所有cindexes.
    this_info.output_cindex_ids.swap((*by_step)[step]);
    // 保存对应的request
    this_info.segment = step_to_segment[step];

    // 当前step下的所有cindex 获得对应的indexes. num_ids是cindexes总数
    // 一个step下的所有cindex 都是相同node-index的 所以直接获得对应的indexes即可.
    // node-index 通过任意一个cindex.first获得.
    int32 num_ids = this_info.output_cindex_ids.size();
    this_info.output_indexes.resize(num_ids);
    
    for (int32 row_index = 0; row_index < num_ids; row_index++)
      this_info.output_indexes[row_index] =
          graph_.cindexes[this_info.output_cindex_ids[row_index]].second;
    
    if (num_ids > 0) {
      // node id's of all Cindexes are the same, so just use first one.
      this_info.node_index =
          graph_.cindexes[this_info.output_cindex_ids.front()].first;
    } else {
      // it's possible to have an empty step if it's the component-input step of
      // a GeneralComponent that does not always have dependencies, such as the
      // ConstantFunctionComponent.  This is just a kind of placeholder; it will
      // generate no commands.  The next command works because the next
      // step will be the propagate for that Component, whose node-index is one
      // more than the component-input node.
      KALDI_ASSERT((step+1) < by_step->size() && !(*by_step)[step+1].empty());
      this_info.node_index =
          graph_.cindexes[(*by_step)[step+1][0]].first - 1;
      KALDI_ASSERT(this_info.node_index >= 0);
      continue;  // we don't need to do anything else for this step.
    }

    // =================== 为每个StepInfo 构建计算矩阵 this_info.value =============
    //      如果需要求导, 在生成一个相同情况的导数矩阵 this_info.deriv
    // 获得对应node
    const NetworkNode &node = nnet_.GetNode(this_info.node_index);
    // 对应index 总数
    int32 num_rows = num_ids, num_cols = node.Dim(nnet_);
    if (node.node_type != kDimRange) {
      MatrixStrideType stride_type = GetStrideType(this_info.node_index);
      this_info.value = computation->NewMatrix(num_rows, num_cols, stride_type);
      if (deriv_needed[step])
        this_info.deriv = computation->NewMatrix(num_rows, num_cols, stride_type);
    } else {
      // kDimRange.  Will just be a sub-matrix of a Component or Input node.
      int32 cindex_id = this_info.output_cindex_ids.front(),
          input_cindex_id = graph_.dependencies[cindex_id][0],
          input_step = cindex_id_to_location_[input_cindex_id].first;
      KALDI_ASSERT(input_step != -1 && input_step < step);
      KALDI_PARANOID_ASSERT(this_info.output_indexes ==
                            steps_[input_step].output_indexes);
      this_info.value = computation->NewSubMatrix(steps_[input_step].value,
                                                  0, -1,
                                                  node.dim_offset, node.dim);
      if (deriv_needed[step])
        this_info.deriv = computation->NewSubMatrix(steps_[input_step].deriv,
                                                    0, -1,
                                                    node.dim_offset, node.dim);
    }


    
    // ========================= 如果是kDescriptor 还需要处理 依赖的 locations ======================
    if (node.node_type == kDescriptor) {
      // 1 设置input_locations_list --- 对于kDescriptor是简单计算, 一般就是保存从哪里copy数据
      // 2 设置value_parts 和 可能的deriv_parts

      const Descriptor &desc = node.descriptor;
      int32 num_parts = desc.NumParts();
      
      KALDI_ASSERT(num_parts > 0);

      
      // ===================================计算依赖的locations=====================================
      // 计算kDescriptor的每个部分(kDescriptor 的输入)的 所有indexes的依赖dep_cindexes 的locations
      // ------------------------ 通过cindex_id_to_locations_
      this_info.input_locations_list.resize(num_parts);
      for (int32 part = 0; part < num_parts; part++){
        ComputeInputLocationsList(step, part, &(this_info.input_locations_list[part]));
      }


      
      // 如果是一个part 直接保存一个 矩阵.
      if (num_parts == 1) {
        this_info.value_parts.push_back(this_info.value);
        if (deriv_needed[step])
          this_info.deriv_parts.push_back(this_info.deriv);
      }

      
      // 如果是多个part部分, 则没部分都是 this_info.value 的一部分, 通过cur_dim_offset 进行划分
      else { // num_parts > 1.
        int32 cur_dim_offset = 0;
        // Have multiple parts, so need to set up sub-matrices.
        this_info.value_parts.resize(num_parts);
        if (deriv_needed[step])
          this_info.deriv_parts.resize(num_parts);
        
        for (int32 p = 0; p < num_parts; p++) {
          const SumDescriptor &this_part = desc.Part(p);
          int32 this_dim = this_part.Dim(nnet_);
          this_info.value_parts[p] =
              computation->NewSubMatrix(this_info.value, 0, -1, cur_dim_offset, this_dim);

          if (deriv_needed[step])
            this_info.deriv_parts[p] =
                computation->NewSubMatrix(this_info.deriv, 0, -1, cur_dim_offset, this_dim);
          cur_dim_offset += this_dim;
        }
        
        KALDI_ASSERT(cur_dim_offset == desc.Dim(nnet_));
      }
    }
  }
}


























// 某个submatrices 是否是一个完整的matrix, 获得对应的matrix-index
// 如果是, 设置whole_submatrix[matrix-index] = s, 表示某个matrix具有一个全部域的submatrix.
void NnetComputation::GetWholeSubmatrices(
    std::vector<int32> *whole_submatrices) const {
  int32
      num_matrices = matrices.size(),
      num_submatrices = submatrices.size();
  whole_submatrices->clear();
  whole_submatrices->resize(num_matrices, 0);

  for (int32 s = 1; s < num_submatrices; s++) {
    if (IsWholeMatrix(s)) {
      int32 m = submatrices[s].matrix_index;
      (*whole_submatrices)[m] = s;
    }
  }

  
  for (int32 m = 1; m < num_matrices; m++) {
    KALDI_ASSERT((*whole_submatrices)[m] != 0 &&
                 "Matrix exists with no submatrix that is "
                 "the whole of it.");
  }
}

// 根据每个 computation matrix 是否具有全域的 submatrix 增加matrix.
void Compiler::AllocateMatrices(const std::vector<int32> &whole_submatrices,
                                NnetComputation *computation) const {

  // 当前computation 还没有任何命令cmd
  KALDI_ASSERT(computation->commands.empty());

  // 计算出哪些matrix是Computation 的 input. 或者用来作为Computation的输入的一些 输出导数,

  unordered_set<int32> input_and_oderiv_matrices;
  int32 num_steps = steps_.size();
  
  for (int32 step = 0; step < num_steps; step++) {
    const StepInfo &this_info = steps_[step];
    // this_info的输出目标cindexes.
    if (this_info.output_cindex_ids.empty())
      continue;
    
    int32 first_cindex_id = this_info.output_cindex_ids.front(),
        node_index = this_info.node_index;
    bool is_input = graph_.is_input[first_cindex_id],
        is_output = nnet_.IsOutputNode(node_index);

    // 如果当前StepInfo是个input-node类型的, 将对应的Matrix-index 加入到 input_and_oderiv_matrices.
    if (is_input) {
      int32 value_submatrix_index = this_info.value,
          value_matrix_index = computation->submatrices[value_submatrix_index].matrix_index;
      input_and_oderiv_matrices.insert(value_matrix_index);
    }
    // 如果是输出, 并且有导数计算空间, 也将对应的Matrix-index加入到 input_and_oderiv_matrices
    if (is_output && this_info.deriv != 0) {
      int32 deriv_submatrix_index = this_info.deriv,
          deriv_matrix_index = computation->submatrices[deriv_submatrix_index].matrix_index;
      input_and_oderiv_matrices.insert(deriv_matrix_index);
    }
  }

  // 全部Matrix总数
  int32 num_matrices = computation->matrices.size();
  for (int32 m = 1; m < num_matrices; m++) {

    // 如果不是 input 或者 output_deriv 的matrices 那么就需要真实申请空间.
    // 向computation->commands 中增加kAllocMatrix cmd
    if (input_and_oderiv_matrices.count(m) == 0) {
      // get a submatrix index that refers to the entire matrix.
      int32 submatrix_index = whole_submatrices[m];
      computation->commands.push_back( NnetComputation::Command(kAllocMatrix, submatrix_index));

      // 稍后在优化阶段, 发现 zero 对某些matrix是不必要的, 稍后会删除冗余的 kSetConst命令.
      computation->commands.push_back( NnetComputation::Command(0.0, kSetConst, submatrix_index));
    }
  }
}



void Compiler::SetUpPrecomputedIndexes(
    const std::vector<int32> &step_to_segment,
    NnetComputation *computation) {
  // Compiler 保存着 StepInfos -- steps_
  int32 num_steps = steps_.size();
  // 当前 computation->component_precomputed_indexes 某些component需要提前计算的 还没设置.
  KALDI_ASSERT(computation->component_precomputed_indexes.empty());

  // the zeroth commponent is special, contains a NULL pointer.
  computation->component_precomputed_indexes.resize(1);

  // foreach StepInfo
  for (int32 step = 0; step < num_steps; step++) {
    StepInfo &step_info = steps_[step];
    int32 node_index = step_info.node_index;
    const NetworkNode &node = nnet_.GetNode(node_index);

    // 这里只考虑 kComponent node 类型的StepInfo
    // There is only something to do for nodes of type Component.
    if (node.node_type != kComponent)
      continue;

    // 获得对应的kDescriptor StepInfo
    // 对kComponent node类型的StepInfo 具有直接的 kDescriptor node的StepInfo .
    const StepInfo &input_step_info = steps_[step - 1];
    // component-index--- 注意是nnet中的component index
    int32 component_index = node.u.component_index;
    // descriptor-node-index
    int32 input_node_index = input_step_info.node_index;
    
    KALDI_ASSERT(input_node_index == node_index - 1);

    // kDescriptor StepInfo的输出--- 即当前kComponent StepInfo 的输入.
    const std::vector<Index> &input_indexes = input_step_info.output_indexes;
    // kComponent StepInfo 的输出
    const std::vector<Index> &output_indexes = step_info.output_indexes;
    // 获得对应的Component
    const Component *component = nnet_.GetComponent(component_index);
    // 获得对应的Request
    const ComputationRequest &request = *(requests_[step_to_segment[step]]);

    // XXXXX ???? 计算需要先进性处理的 Indexes.
    bool need_derivs = request.NeedDerivatives();
    ComponentPrecomputedIndexes *precomputed_indexes =
        component->PrecomputeIndexes(request.misc_info,
                                     input_indexes, output_indexes,
                                     need_derivs);

    // 简单一般情况 都没有需要先计算的indexes.
    if (precomputed_indexes == NULL) {
      // e.g. simple Components, and some other Components, will return NULL for
      // precomputed_indexes.
      step_info.precomputed_indexes_index = 0;
    } else {
      step_info.precomputed_indexes_index =
          computation->component_precomputed_indexes.size();

      NnetComputation::PrecomputedIndexesInfo info;
      info.data = precomputed_indexes;

      if (!input_indexes.empty() && input_indexes.back().n == 1 &&
          !output_indexes.empty() && output_indexes.back().n == 1) {
        // If these conditions are true, it's *possible* that we are doing
        // 'shortcut' compilation.  So just in case that's what's going on, we
        // store 'input_indexes' and 'output_indexes, which are needed by
        // the ExpandComputation() function that is used in that process.
        info.input_indexes = input_indexes;
        info.output_indexes = output_indexes;
      }
      
      computation->component_precomputed_indexes.push_back(info);
    }

    
  }
}











void Compiler::CompileForward(int32 step,
                              NnetComputation *computation) const {
  KALDI_ASSERT(step < static_cast<int32>(steps_.size()));
  
  const StepInfo &step_info = steps_[step];
  const NetworkNode &node = nnet_.GetNode(step_info.node_index);

  switch (node.node_type) {
    case kInput:  // Note: input nodes appear before other node types.
      AddForwardStepInput(step, computation);
      // 简单增加一个kAcceptInput命令, 指导计算机从user手中获得input数据
      // 因为input类型的StepInfo永远是在steps_最前面, 这个命令会在其他真实有效计算命令之前出现.
      void Compiler::AddForwardStepInput(int32 step,
                                         NnetComputation *computation) const {
        KALDI_ASSERT(static_cast<size_t>(step) < steps_.size());
        
        const StepInfo &step_info = steps_[step];
        int32 node_index = step_info.node_index,
            submatrix_index = step_info.value;
        KALDI_ASSERT(computation->IsWholeMatrix(submatrix_index));

        const NetworkNode &node = nnet_.GetNode(node_index);
        // actually currently the node type would always be kInput.
        KALDI_ASSERT(node.node_type == kInput || node.node_type == kComponent);

        NnetComputation::Command c(kAcceptInput, submatrix_index, node_index);
        computation->commands.push_back(c);
      }

      // 判断下一个StepInfo 是否是 input类型的StepInfo, 如果不是,那么就需要增加一个空命令
      // 如果还是一个input类型的StepInfo, 则下次循环再判断,
      // 最终会形成, Input类型的StepInfo的kAcceptInput命令, 与 真实其他命令之间形成一个 空命令间隔.
      
      if (!IsInputStep(step + 1))  // Make sure forward computation is nonempty.
        computation->commands.push_back( NnetComputation::Command(kNoOperationPermanent));
      
      break;




      
    case kDimRange: break;  // Nothing to do.
    case kComponent:

      AddForwardStepComponent(step, computation);
      void Compiler::AddForwardStepComponent(int32 step,
                                             NnetComputation *computation) const {
        KALDI_ASSERT(static_cast<size_t>(step) < steps_.size());
        // 当前kComponent StepInfo
        const StepInfo &step_info = steps_[step];
        // 配套kDescriptor StepInfo
        int32 input_step = step - 1;
        const StepInfo &input_step_info = steps_[input_step];
        // 当前StepInfo - kComponent node
        int32 node_index = step_info.node_index;
        const NetworkNode &node = nnet_.GetNode(node_index);
        
        KALDI_ASSERT(node.node_type == kComponent);
        // 对应的Component 计算操作组件.
        int32 component_index = node.u.component_index;
        const Component *component = nnet_.GetComponent(component_index);

        // 我们会在optimization阶段为了避免空隙 对memo_index 重新编号
        // 
        // note RE memo_index: we'll renumber them in optimization to get rid of gaps.
        // The use of 'step' as the memo index is OK because step > 0 if we're doing
        // forward propagation, there must be preceding steps for inputs or for
        // component-input nodes).
        int32
            properties = component->Properties(),
            input_submatrix_index = input_step_info.value,
            output_submatrix_index = step_info.value,
            // 如果当前StepInfo需要求导, 并且是具有 kUsesMemo属性, memo_index具有index.
            memo_index = (step_info.deriv > 0 && (properties & kUsesMemo) ? step : 0),
            // 如果request允许保存component统计量 并当前kComponent具有 kStoresStats属性 设置1
            store_stats = (requests_[0]->store_component_stats && (properties & kStoresStats) ?  1 : 0);

        // 增加 前向传播命令, 
        NnetComputation::Command c(kPropagate,
                                   component_index,
                                   step_info.precomputed_indexes_index,
                                   input_submatrix_index,
                                   output_submatrix_index,
                                   memo_index,
                                   store_stats);
        computation->commands.push_back(c);
      }
      break;





    case kDescriptor:

      CompileForwardDescriptor(step, computation);
      void Compiler::CompileForwardDescriptor(int32 step, NnetComputation *computation) const {

        // 对Descriptor的每个part 进行按类型添加不同的command命令.
        int32 num_parts = steps_[step].value_parts.size();
        for (int32 part = 0; part < num_parts; part++){
          CompileForwardSumDescriptor(step, part, computation);

          
          void Compiler::CompileForwardSumDescriptor(
              int32 step, int32 part_index, NnetComputation *computation) const {
            
            const StepInfo &step_info = steps_[step];
            int32 value_submatrix_index = step_info.value_parts[part_index];

            // 当前part的 SumDescriptor.
            const SumDescriptor &descriptor =
                nnet_.GetNode(step_info.node_index).descriptor.Part(part_index);

            // 获得该节点的Scale      对-1 返回 0
            BaseFloat offset_term = descriptor.GetScaleForNode(-1);
            if (offset_term != 0.0) {
              computation->commands.push_back(
                  NnetComputation::Command(offset_term, kSetConst,
                                           value_submatrix_index));
            }


            // input_locations_list 是 kDescriptor的StepInfo的当前part的所有cindexes的 所有依赖dep_cindexes-location的列表.
            // ------- <
            //           cindex1<dep_cindex1-location, dep_cindex2-location, ... >,
            //           cindex2<dep_cindex1-location, dep_cindex2-location, ... >,
            //           ....
            //           cindexn<dep_cindex1-location, dep_cindex2-location, ... >
            //          >
            
            // ---------------- 元素是 pair<step,row_index>的list, 代表加权求和的 公式
            // step_info.input_locations_list[part_index]
            //  --------------- 保存的是 当前的StepInfo的所有cindex的依赖dep_cindexes的位置.
            const std::vector<std::vector<std::pair<int32,int32> > >
                &input_locations_list = step_info.input_locations_list[part_index];

            // split_locations_lists 是 pair<alpha, locations_list>的向量
            // 其中alpha 是拉伸值 其中 pair中元素locations_list是刚刚的 input_locations_list.
            std::vector<std::pair<BaseFloat,
                                  std::vector<std::vector<std::pair<int32,int32> > > > > split_locations_lists;

            // descriptor 构造的 SimpleSumDescriptor
            // input_locations_list 是
            BaseFloat shared_alpha = SplitByScale(descriptor, input_locations_list, &split_locations_lists);
            
            if (shared_alpha - shared_alpha == 0.0) {
              // If the returned value 'shared_alpha' is finite, this indicates that there was no
              // need to split up 'input_locations_list' because all the alpha values
              // (scales) were the same.  We treat this case specially for efficiency
              // reasons; this branch will be the most common branch.
              std::vector<std::vector<std::pair<int32, int32> > > submat_locations_list;
              ComputeValueSubmatLocationsList(input_locations_list,
                                              &submat_locations_list);
              
              CompileForwardFromSubmatLocationsList(
                  value_submatrix_index,
                  shared_alpha,
                  submat_locations_list,
                  computation);
            }

            else {
              for (size_t i = 0; i < split_locations_lists.size(); i++) {
                BaseFloat this_alpha = split_locations_lists[i].first;
                KALDI_ASSERT(this_alpha - this_alpha == 0.0);
                std::vector<std::vector<std::pair<int32, int32> > > submat_locations_list;
                ComputeValueSubmatLocationsList(split_locations_lists[i].second,
                                                &submat_locations_list);
                CompileForwardFromSubmatLocationsList(
                    value_submatrix_index,
                    this_alpha,
                    submat_locations_list,
                    computation);
              }
            }
          }

          
        }



        
        const StepInfo &step_info = steps_[step];
        if (nnet_.IsOutputNode(step_info.node_index)) {
          // 如果该StepInfo是个output-node的StepInfo, 我们需要增加命令, 将输出结果提供给user.
          // 并可能会从user哪里获得导数.

          int32 node_index = step_info.node_index,
              submatrix_index = step_info.value;
          KALDI_ASSERT(computation->IsWholeMatrix(submatrix_index));
          NnetComputation::Command c(kProvideOutput, submatrix_index, node_index);
          computation->commands.push_back(c);
        }
      }

      break;
    default:
      KALDI_ERR << "Invalid node type";
  }

}





void Compiler::CompileBackward(int32 step,
                                     NnetComputation *computation) {
  KALDI_ASSERT(step < static_cast<int32>(steps_.size()));
  const StepInfo &step_info = steps_[step];
  int32 node_index = step_info.node_index;
  const NetworkNode &node = nnet_.GetNode(node_index);

  switch (node.node_type) {
    case kInput:
      AddBackwardStepInput(step, computation);

      void Compiler::AddBackwardStepInput(int32 step,
                                          NnetComputation *computation) const {
        KALDI_ASSERT(static_cast<size_t>(step) < steps_.size());
        const StepInfo &step_info = steps_[step];
        int32 node_index = step_info.node_index,
            deriv_submatrix_index = step_info.deriv;
        
        if (deriv_submatrix_index == 0)
          return;  // Nothing to do.
        // 断言 是个 完整Matrix的 submatrix
        KALDI_ASSERT(computation->IsWholeMatrix(deriv_submatrix_index));
        const NetworkNode &node = nnet_.GetNode(node_index);
        // actually, currently the node type would always be kInput.
        KALDI_ASSERT(node.node_type == kInput || node.node_type == kComponent);
        // 提供输出导数?
        NnetComputation::Command c(kProvideOutput, deriv_submatrix_index, node_index);
        computation->commands.push_back(c);
      }
      
      if (!IsInputStep(step + 1))  // Make sure backward computation is nonempty.
        computation->commands.push_back(
            NnetComputation::Command(kNoOperationPermanent));
      break;



      
    case kDimRange:
      break;  // Nothing to do.

      
    case kComponent:
      AddBackwardStepComponent(step, computation);

      void Compiler::AddBackwardStepComponent(int32 step,
                                              NnetComputation *computation) const {
        KALDI_ASSERT(static_cast<size_t>(step) < steps_.size());

        const StepInfo &step_info = steps_[step];

        int32 input_step = step - 1;
        const StepInfo &input_step_info = steps_[input_step];

        int32 node_index = step_info.node_index;
        const NetworkNode &node = nnet_.GetNode(node_index);
        KALDI_ASSERT(node.node_type == kComponent);

        int32 component_index = node.u.component_index;
        const Component *component = nnet_.GetComponent(component_index);
        int32 properties = component->Properties();

        int32
            input_submatrix_index = input_step_info.value,
            output_submatrix_index = step_info.value,
            
            input_deriv_submatrix_index = input_step_info.deriv,
            output_deriv_submatrix_index = step_info.deriv,
            
            memo_index = (properties & kUsesMemo ? step : 0);
        
        KALDI_ASSERT(output_deriv_submatrix_index > 0 &&
                     (input_deriv_submatrix_index > 0 ||
                      properties & kUpdatableComponent));

        // ?????????????
        // 属性 不包含 kBackprobNeedsInput 反向传播需要Input
        if (! (properties & kBackpropNeedsInput))
          input_submatrix_index = 0;
        // 属性 不包含 kBackprobNeedsOutputs 反向传播需要Output????
        if (! (properties & kBackpropNeedsOutput))
          output_submatrix_index = 0;

        NnetComputation::Command c(kBackprop,
                                   component_index,
                                   step_info.precomputed_indexes_index,
                                   input_submatrix_index,
                                   output_submatrix_index,
                                   output_deriv_submatrix_index,
                                   input_deriv_submatrix_index,
                                   memo_index);
        computation->commands.push_back(c);
      }

      break;


      
    case kDescriptor:
      CompileBackwardDescriptor(step, computation);
      void Compiler::CompileBackwardDescriptor(
          int32 step, NnetComputation *computation) {

        StepInfo &step_info = steps_[step];
        // 如果StepInfo 是个kDescriptor的 Output 并且需要求导 导数矩阵
        if (nnet_.IsOutputNode(step_info.node_index) && step_info.deriv > 0) {
          int32 deriv_submatrix_index = step_info.deriv;

          KALDI_ASSERT(computation->IsWholeMatrix(deriv_submatrix_index));
          // 增加 从user 接收导数命令.
          NnetComputation::Command c(kAcceptInput, deriv_submatrix_index, step_info.node_index);
          computation->commands.push_back(c);
        }


        // kDescriptor StepInfo 的 每个part 计算对应的 BackwardSumDescriptor
        int32 num_parts = step_info.value_parts.size();
        for (int32 part = 0; part < num_parts; part++){
          CompileBackwardSumDescriptor(step, part, computation);
          
          void Compiler::CompileBackwardSumDescriptor(
              int32 step, int32 part_index, NnetComputation *computation) const {
            const StepInfo &step_info = steps_[step];
            int32 deriv_submatrix_index = step_info.deriv_parts[part_index];
            
            KALDI_ASSERT(deriv_submatrix_index > 0);  // or should not have called this.
            
            const SumDescriptor &descriptor =
                nnet_.GetNode(step_info.node_index).descriptor.Part(part_index);
            
            // Note:
            // `offset_term` appeared in the forward computation here but does not
            // come into the backward computation.

            // `input_locations_list` is a vector indexed by row-index, with each element
            // being a list of pairs (step, row_index) representing terms in a weighted
            // sum.
            
            const std::vector<std::vector<std::pair<int32,int32> > >
                &input_locations_list = step_info.input_locations_list[part_index];

            // `split_locations_lists` is a vector of pairs `(alpha, locations_list)`
            // where alpha is the scale in which these items appear in the
            // summation and `locations_list` is the same format as `input_locations_list`
            std::vector<std::pair<BaseFloat,
                                  std::vector<std::vector<std::pair<int32,int32> > > > > split_locations_lists;
            BaseFloat shared_alpha = SplitByScale(descriptor, input_locations_list,
                                                  &split_locations_lists);
            
            if (shared_alpha - shared_alpha == 0.0) {
              // If the returned value 'shared_alpha' is finite, this indicates that there
              // was no need to split up 'input_locations_list' because all the alpha
              // values (scales) were the same.  We treat this case specially for
              // efficiency reasons; this branch will be the most common branch.
              std::vector<std::vector<std::pair<int32, int32> > > submat_locations_list;
              ComputeDerivSubmatLocationsList(input_locations_list,
                                              &submat_locations_list);
              CompileBackwardFromSubmatLocationsList(deriv_submatrix_index,
                                                     shared_alpha,
                                                     submat_locations_list,
                                                     computation);
            } else {
              for (size_t i = 0; i < split_locations_lists.size(); i++) {
                BaseFloat this_alpha = split_locations_lists[i].first;
                KALDI_ASSERT(this_alpha - this_alpha == 0.0);
                std::vector<std::vector<std::pair<int32, int32> > > submat_locations_list;
                ComputeValueSubmatLocationsList(split_locations_lists[i].second,
                                                &submat_locations_list);
                CompileBackwardFromSubmatLocationsList(deriv_submatrix_index,
                                                       this_alpha,
                                                       submat_locations_list,
                                                       computation);
              }
            }
          }
        }
        
      }

      break;
    default:
      KALDI_ERR << "Invalid node type";
  }
}




void Compiler::DeallocateMatrices(const std::vector<int32> &whole_submatrices,
                                  const std::vector<int32> &step_to_segment,
                                  NnetComputation *computation) {
  // 这个函数增加 destroy全部matrices的命令, 除了作为计算的输出的 matrices.
  // 那些matrices 是从销毁操作中分离出来的,
  // 1 对应为计算的输出, 2 那些对应为用户需要的input导数

  int32 num_matrices = computation->matrices.size();
  std::vector<bool> will_destroy(num_matrices, true);

  int32 num_steps = steps_.size();
  for (int32 step = 0; step < num_steps; step++) {
    const StepInfo &step_info = steps_[step];
    const ComputationRequest &request = *(requests_[step_to_segment[step]]);

    // 判断是output StepInfo, 这样的matrix 需要被保留 设置will_destroy[matrix-index] = false
    if (nnet_.IsOutputNode(step_info.node_index)) {
      
      // steps corresponding to output nodes need to have their "value" kept.
      int32 value_matrix_index =
          computation->submatrices[step_info.value].matrix_index;
      will_destroy[value_matrix_index] = false;
      
    }

    // 判断是 input StepInfo
    else if (nnet_.IsInputNode(step_info.node_index)) {
      // 对应为input-node 的StepInfo 需要他们的deriv被保持.
      // 但是只有当 对应input的 derivative 被设置为需要的 时候.
      // 我们不需要担心是否输出是否是需要的, 因为如果output是不需要的 都不会计算.

      std::string input_name = nnet_.GetNodeNames()[step_info.node_index];
      // 对该request的每个 input
      int32 i = 0, num_inputs = request.inputs.size();
      bool has_deriv = false;
      // 当确定该input需要 deriv时候, 则设置 will_destroy = false, 需要保持给 user.
      for (; i < num_inputs; i++) {
        if (input_name == request.inputs[i].name) {
          has_deriv = request.inputs[i].has_deriv;
          break;
        }
      }
      KALDI_ASSERT(i != num_inputs); // assert we found an input-request with
                                     // this name
      if (has_deriv) {
        int32 deriv_matrix_index =
          computation->submatrices[step_info.deriv].matrix_index;
        will_destroy[deriv_matrix_index] = false;
      }
    }
  }


  // 顺序增加每个submatrix-index 的 kDeallocMatrix命令 计算完成后 销毁这些matrices.
  // note: matrix-index 0 is the empty matrix.
  for (int32 m = 1; m < num_matrices; m++) {
    if (will_destroy[m]) {
      int32 submatrix_index = whole_submatrices[m];
      computation->commands.push_back(
          NnetComputation::Command(kDeallocMatrix, submatrix_index));
    }
  }
}

// 1 计算computation的matrix的全域submatrix的 submaxtrix-id 保存到 whold_submatrices
//   向computation->commands 中 对 非input 非output_deriv 的StepInfo 增加 kAllocMatrix cmd 命令,
// 2 SetUpProcomputedIndexes 是否有需要先计算的 Indexes 加入到 Computation->component_precompute_indexes

void Compiler::AddCommands(const std::vector<bool> &deriv_needed,
                           const std::vector<int32> &step_to_segment,
                           NnetComputation *computation) {
  
  computation->need_model_derivative = requests_[0]->need_model_derivative;
  
  int32 arbitrary_factor = 8;
  computation->commands.reserve(computation->matrices.size() * arbitrary_factor);

  std::vector<int32> whole_submatrices;
  computation->GetWholeSubmatrices(&whole_submatrices);
  AllocateMatrices(whole_submatrices, computation);
  
  SetUpPrecomputedIndexes(step_to_segment, computation);

  // foreach StepInfo
  int32 num_steps = steps_.size();
  for (int32 step = 0; step < num_steps; step++) {
    // ============ 编译StepInfo =========
    CompileForward(step, computation);

    // 1 如果当前StepInfo 不是最后一个reqeust的最后一个StepInfo
    // 2 且  当前 StepInfo 是一个request的最后一个 StepInfo
    // 增加一个kNoOperationMarker,标记区分两个request.
    //  ----- 
    if (step + 1 < static_cast<int32>(step_to_segment.size()) &&  step_to_segment[step + 1] != step_to_segment[step]) {
      // insert a marker that separates segments of the computation.
      computation->commands.push_back( NnetComputation::Command(kNoOperationMarker));
    }
  }

  // mark the end of the forward phase.
  computation->commands.push_back(
      NnetComputation::Command(kNoOperationMarker));

  // 每个需要求导数的StepInfo 增加反向传播 求导命令.
  for (int32 step = num_steps - 1; step >= 0; step--)
    if (deriv_needed[step])
      CompileBackward(step, computation);

  
  DeallocateMatrices(whole_submatrices, step_to_segment, computation);
}
