// xxxxxxxxxxxxxxxxxxxxx part-Step xxxxxxxxxxxxxxxxxxxxxx

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

      // 如果step 是输出, 并且用户提供了导数, 对该输出我们需要一个位置
      // 保存导数, 这样也会设置 deriv_needed=true.
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

