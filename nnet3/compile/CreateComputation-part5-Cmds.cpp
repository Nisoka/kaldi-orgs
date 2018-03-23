


// xxxxxxxxxxxxxxxx  part-Cmds  xxxxxxxxxxxxxxxxxxxxxxxx
{
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
              // ---------------- 元素是 pair<step,row_index>的list, 代表加权求和的 公式???

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
                // 这里是正常情况
                // 结果sub_locations_list 主要从input_locations_list 中获得位置信息
                // 但是将对应每个 cindex的依赖dep_cindex_location中的 step-index,
                // 将step-index域 设置为对应的 计算矩阵index.
                std::vector<std::vector<std::pair<int32, int32> > > submat_locations_list;
                ComputeValueSubmatLocationsList(input_locations_list,
                                                &submat_locations_list);

                void Compiler::ComputeValueSubmatLocationsList(
                    std::vector<std::vector<std::pair<int32, int32> > > &input_locations_list,
                    std::vector<std::vector<std::pair<int32, int32> > >*submat_locations_list)
                    const {
                  submat_locations_list->clear();
                  submat_locations_list->resize(input_locations_list.size());
                  int32 size = submat_locations_list->size();
                  
                  for (int32 i = 0; i < size; i++) {
                    const std::vector<std::pair<int32, int32> > &this_list = input_locations_list[i];
                    std::vector<std::pair<int32, int32> > &this_submat_list = (*submat_locations_list)[i];
                    this_submat_list.resize(this_list.size());
                    std::vector<std::pair<int32, int32> >::const_iterator
                        input_iter = this_list.begin(), input_end = this_list.end();
                    std::vector<std::pair<int32, int32> >::iterator iter =
                        this_submat_list.begin();
                    for (; input_iter != input_end; ++input_iter, ++iter) {
                      int32 step = input_iter->first,
                          value_submat_index = steps_[step].value,
                          row = input_iter->second;
                      iter->first = value_submat_index;
                      iter->second = row;
                    }
                  }
                }

              
                CompileForwardFromSubmatLocationsList(
                    value_submatrix_index,
                    shared_alpha,
                    submat_locations_list,
                    computation);

                void Compiler::CompileForwardFromSubmatLocationsList(
                    int32 value_submatrix_index,
                    BaseFloat alpha,
                    const std::vector<std::vector<std::pair<int32, int32> > > &submat_lists,
                    NnetComputation *computation) const {


                  std::vector<std::vector<std::pair<int32, int32> > > split_lists;
                  SplitLocations(submat_lists, &split_lists);
                  
                  int32 size = split_lists.size();
                  // note: `size` may be empty in unusual cases so don't assert that it's
                  // nonzero.
                  for (int32 i = 0; i < size; i++)
                    CompileForwardFromSubmatLocations(
                        value_submatrix_index,
                        alpha,
                        split_lists[i],
                        computation);
                }

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


}  // done part-Cmds
