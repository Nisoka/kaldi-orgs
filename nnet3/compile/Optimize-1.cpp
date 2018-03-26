void ConsolidateModelUpdate(const Nnet &nnet,
                            NnetComputation *computation) {
  // This following if-statement is an optimization: if the computation
  // request(s) had need_model_derivative == false, there would be nothing to
  // optimize, so don't bother trying.
  if (!computation->need_model_derivative)
    return;
  ModelUpdateConsolidator consolidator(nnet, computation);

  ModelUpdateConsolidator::ModelUpdateConsolidator(
    const Nnet &nnet,
    NnetComputation *computation):
    nnet_(nnet), computation_(computation),
    extra_commands_(computation->commands.size()) { }
  
  consolidator.ConsolidateModelUpdate();
}


void ModelUpdateConsolidator::ConsolidateModelUpdate() {

  // 1 nnet component 总数
  // 2 Compute后的 command总数
  int32
      num_components = nnet_.NumComponents(),
      num_commands = computation_->commands.size();

  
  // backprob_commands 二级vector
  // <
  //   component1 <command1(kBackprob), command2(kBackprob) ... >,
  //   component2 <command1(kBackprob), command2(kBackprob) ... >,
  //   component3 <command1(kBackprob), command2(kBackprob) ... >
  // >
  
  // 保存所有updatable的component 的backprob command-index 的list.
  // 取每个 kBackprob类型的command 加入到 对应component 的 backprob_commands[component-index]中.
  std::vector<std::vector<int32> > backprop_commands(num_components);
  for (int32 command_index = 0; command_index < num_commands; command_index++) {
    // only the kBackprob 
    const NnetComputation::Command &c = computation_->commands[command_index];
    if (c.command_type == kBackprop) {
      
      int32 component_index = c.arg1;
      const Component *component = nnet_.GetComponent(component_index);
      int32 properties = component->Properties();
      if ((properties & kUpdatableComponent) &&
          (properties & kSimpleComponent) &&
          !(properties & kUsesMemo))
        backprop_commands[component_index].push_back(command_index);
    }
  }



  // 某个component 具有多个 kBackprob command 进行处理.
  bool consolidated = false;
  for (int32 component = 0; component < num_components; component++) {
    if (backprop_commands[component].size() > 1) {
      ConsolidateUpdateForComponent(component,
                                    backprop_commands[component]);
      consolidated = true;
    }
  }

  // 避免冗余计算的优化.
  if (!consolidated)  // This is an optimization to avoid redundant computation
    return;           // if there is nothing to do.

  // 这个函数可以 将保存到成员变量 ???  中的commands 加入到 computation_->commands.
  AddCommandsToComputation();
}


// 这个函数从ConsolidataModelUpdate调用, 传入一个commands list, 都是同一个component的backprobs命令
// 这个函数 将这几个command统一到一个 单一的 mode-update 命令.
/** This function, called from ConsolidateModelUpdate, is passed a list of
    commands that are all backprops for the same component, and it consolidates
    them into a single model-update command. */
void ModelUpdateConsolidator::ConsolidateUpdateForComponent(
    int32 component_index,
    const std::vector<int32> &backprop_commands) {
  
  const Component *component = nnet_.GetComponent(component_index);
  int32 num_backprop_commands = backprop_commands.size();

  bool need_input = (component->Properties() & kBackpropNeedsInput) != 0,
      need_output = (component->Properties() & kBackpropNeedsOutput) != 0;

  std::vector<int32>  input_submatrices(num_backprop_commands),
      output_submatrices(num_backprop_commands),
      output_deriv_submatrices(num_backprop_commands);

  for (int32 i = 0; i < num_backprop_commands; i++) {
    int32 command_index = backprop_commands[i];
    NnetComputation::Command &command =
        computation_->commands[command_index];
    // arg2 must be 0 because simple components don't use precomputed indexes.
    KALDI_ASSERT(command.command_type == kBackprop && command.arg2 == 0);
    command.command_type = kBackpropNoModelUpdate;
    int32 input_submatrix = command.arg3,
        output_submatrix = command.arg4,
        output_deriv_submatrix = command.arg5;
    KALDI_ASSERT((input_submatrix != 0) == need_input &&
                 (output_submatrix != 0) == need_output);
    input_submatrices[i] = input_submatrix;
    output_submatrices[i] = output_submatrix;
    output_deriv_submatrices[i] = output_deriv_submatrix;
  }
  // 获得sub-matrix 索引, 每个sub-matrix 是我们需要的统一的matrix
  int32 input_submatrix = (need_input ?
                           ConsolidateSubmatrices(backprop_commands,
                                                  input_submatrices) : 0),
      output_submatrix = (need_output ?
                         ConsolidateSubmatrices(backprop_commands,
                                                output_submatrices) : 0),
      output_deriv_submatrix = ConsolidateSubmatrices(backprop_commands,
                                                      output_deriv_submatrices);
  int32 precomputed_indexes_index = 0,  // unused since simple component
      input_deriv_submatrix = 0,  // we don't need the input-deriv.
      memo_index = 0;  // we checked that no memos were used.
  NnetComputation::Command c(kBackprop, component_index, precomputed_indexes_index,
                             input_submatrix, output_submatrix,
                             output_deriv_submatrix, input_deriv_submatrix,
                             memo_index);
  final_commands_.push_back(c);
}




int32 ModelUpdateConsolidator::ConsolidateSubmatrices(
    const std::vector<int32> &commands,
    const std::vector<int32> &submatrices) {
  int32 num_submatrices = submatrices.size();
  KALDI_ASSERT(num_submatrices > 1 && commands.size() == submatrices.size());
  int32 first_submatrix = submatrices[0];
  int32 num_cols = computation_->submatrices[first_submatrix].num_cols,
      num_rows = 0;
  MatrixStrideType stride_type = kDefaultStride;
  NnetComputation::MatrixDebugInfo debug_info;
  for (int32 i = 0; i < num_submatrices; i++) {
    int32 submatrix = submatrices[i];
    num_rows += computation_->submatrices[submatrix].num_rows;
    KALDI_ASSERT(computation_->submatrices[submatrix].num_cols == num_cols);
    if (!computation_->matrix_debug_info.empty())
      AppendDebugInfoForSubmatrix(submatrix, &debug_info);
    if (computation_->IsWholeMatrix(submatrix)) {
      int32 matrix = computation_->submatrices[submatrix].matrix_index;
      if (computation_->matrices[matrix].stride_type == kStrideEqualNumCols)
        stride_type = kStrideEqualNumCols;
    }
  }

  // 创建一个新的 whole_submatrix, 将多个command对应的matrix 合并到这个新的matrix中.
  int32 new_whole_submatrix = computation_->NewMatrix(num_rows, num_cols,
                                                      stride_type);
  // Add commands at the very start, to initialize and then zero this new
  // matrix.  we can later on remove the zeroing if it is not necessary.
  extra_commands_[0].push_back(
      NnetComputation::Command(kAllocMatrix, new_whole_submatrix));
  extra_commands_[0].push_back(
      NnetComputation::Command(0.0, kSetConst, new_whole_submatrix));

  final_deallocate_commands_.push_back(
      NnetComputation::Command(kDeallocMatrix, new_whole_submatrix));
  int32 new_matrix_index =
      computation_->submatrices[new_whole_submatrix].matrix_index;
  if (!computation_->matrix_debug_info.empty())
    computation_->matrix_debug_info[new_matrix_index].Swap(&debug_info);

  int32 row_offset = 0;
  for (int32 i = 0; i < num_submatrices; i++) {
    int32 submatrix_index = submatrices[i];
    int32 this_num_rows = computation_->submatrices[submatrix_index].num_rows;
    // 将原本命令中的每个matrix 都映射到新的 new_whole_submatrix中.
    int32 new_submatrix = computation_->NewSubMatrix(new_whole_submatrix,
                                                     row_offset, this_num_rows,
                                                     0, num_cols);
    // extra_commands_
    // <
    //   command0 <额外增加在command0 之前需要进行的一些commands>
    //   command1 <额外增加在command1 之前需要进行的一些commands>
    //   ...
    // >
    
    NnetComputation::Command c(kMatrixCopy, new_submatrix, submatrices[i]);
    extra_commands_[commands[i]].push_back(c);
    row_offset += this_num_rows;
  }
  KALDI_ASSERT(row_offset == num_rows);
  return new_whole_submatrix;
}



void ModelUpdateConsolidator::AddCommandsToComputation() {
  KALDI_ASSERT(computation_->commands.size() == extra_commands_.size());

  int32
      old_num_commands = computation_->commands.size(),
      new_num_commands = old_num_commands +
                         static_cast<int32>(final_commands_.size() + final_deallocate_commands_.size());

  for (size_t i = 0; i < extra_commands_.size(); i++)
    new_num_commands += static_cast<int32>(extra_commands_[i].size());

  // 将extra_commands_[c]中的额外处理命令, 加入到原本每个command之前. 然后一起加入到新命令队列
  std::vector<NnetComputation::Command> new_commands;
  new_commands.reserve(new_num_commands);
  for (int32 c = 0; c < old_num_commands; c++) {
    new_commands.insert(new_commands.end(),
                        extra_commands_[c].begin(), extra_commands_[c].end());
    new_commands.push_back(computation_->commands[c]);
  }

  // 将final_commands中的命令加入到 新命令队列
  new_commands.insert(new_commands.end(),
                      final_commands_.begin(), final_commands_.end());

  new_commands.insert(new_commands.end(),
                      final_deallocate_commands_.begin(),
                      final_deallocate_commands_.end());

  // 将新命令队列 设置给 commands.
  computation_->commands.swap(new_commands);
}



















// ========================== second =============================
// ConvertAdditionToAssignment
{
  void ConvertAdditionToAssignment(const Nnet &nnet,
                                   NnetComputation *computation) {

    Analyzer analyzer;
    // ========= 构建分析器 ======= down
    // 主要就是 根据 CommandAttributes 完成各个variable小块的 访问序列.
    analyzer.Init(nnet, *computation);

    // 构建分析器
    ComputationAnalysis analysis(*computation, analyzer);
    // 所有命令commands 将一些命令类型 修改为简单命令 .
    int32 num_commands = computation->commands.size();
    for (int32 command = 0; command < num_commands; command++) {
      NnetComputation::Command &c = computation->commands[command];
      switch (c.command_type) {
        case kMatrixAdd: case kAddRows: case kAddRowsMulti:
        case kAddToRowsMulti: {
          const std::vector<int32> &submatrices_written =
              analyzer.command_attributes[command].submatrices_written;
          KALDI_ASSERT(!submatrices_written.empty());
          std::vector<int32>::const_iterator iter = submatrices_written.begin(),
              end = submatrices_written.end();
          bool can_convert = true;
          for (; iter != end; ++iter) {
            int32 submatrix_written = *iter;
            int32 first_access_command = analysis.FirstNontrivialAccess(submatrix_written);
            // first_access_command is first command other than zeroing and
            // allocation that accesses this submatrix.  It can be assumed to be a
            // write command, since it makes no sense to read a variable before
            // it's written to.  If it's before this command then we need to add
            // rather than copy; we can't do the conversion to a copy command.
            if (first_access_command != command) {
              can_convert = false;
              break;
            }
          }
          if (can_convert) {  // convert to a copy command.
            switch (c.command_type) {
              case kMatrixAdd: c.command_type = kMatrixCopy;
                break;
              case kAddRows: c.command_type = kCopyRows;
                break;
              case kAddRowsMulti: c.command_type = kCopyRowsMulti;
                break;
                // note: kCopyToRowsMulti does not currently support alpha != 1.0.
              case kAddToRowsMulti: if (c.alpha == 1.0) c.command_type = kCopyToRowsMulti;
                break;
              default: KALDI_ERR << "Unexpected command type.";
            }
          }
          break;
        }
        default:
          break;
      }
    }
  }



  // 这个结构体是为了设置分析的各个部分, 帮助在顺序过程中计算一些东西时候, 避免代码重复, 
  struct Analyzer {
    ComputationVariables variables;
  
    std::vector<CommandAttributes> command_attributes;
    std::vector<std::vector<Access> > variable_accesses;
    std::vector<MatrixAccesses> matrix_accesses;
  
    void Init(const Nnet &nnet, const NnetComputation &computation);
  };

  void Analyzer::Init(const Nnet &nnet, const NnetComputation &computation) {

    // -------- 用computation中的信息 初始化 ComputationVariables ---------
    // variables 中最终要的 variables概念, 对一个 matrix
    // 会存在多个映射其part的 submatrix, 都是matrix 的 row and col 域
    // 这些row col 将matrix分割成不同的 小块 -- variable, 每个submatrix 都是由这些小块variable构成的
    // 所以对一个 submatrix 只需要保存 variable-index-list 就可以确定对应的数据.
    variables.Init(computation);
  
    void ComputationVariables::Init(const NnetComputation &computation) {
      // don't call this twice on the same object..
      KALDI_ASSERT(row_split_points_.empty());
      // 计算computation中所有 matrix的划分点, 
      ComputeSplitPoints(computation);
      // 计算每个submatrix 包含的 所有 根据划分点生成的 variable-index
      ComputeVariablesForSubmatrix(computation);
      // 将每个variable 属于的matrix 构建映射 variable_to_matrix_.
      ComputeVariableToMatrix();
    }




  
    // -------------- 计算 每个命令的 CommandAttributes --------------
    // CommanAttributes 主要是保存命令用到的 matrix_read matrix_write 以及 variable_read variable_wirte. 
    ComputeCommandAttributes(nnet, computation, variables, &command_attributes);

    void ComputeCommandAttributes(
        const Nnet &nnet,
        const NnetComputation &computation,
        const ComputationVariables &vars,
        std::vector<CommandAttributes> *attributes) {

      // attributes 是每个 command 一个 CommandAttributes.
      // commands-size
      int32 num_commands = computation.commands.size();
      attributes->clear();
      attributes->resize(num_commands);

      // 为每个command 构建一个 CommandAttributes对象
      // 内部保存 每个command 必要的submatrix的 variable-list.
      // variable-list 会被加入到 对应的attr 的 read wirte readandwrite 等队列
      for (int32 command_index = 0; command_index < num_commands; command_index++) {
        const NnetComputation::Command &c = computation.commands[command_index];
        CommandAttributes &attr = (*attributes)[command_index];
        switch (c.command_type) {
          case kPropagate:
            // 记录对submatrix的访问 --- 是某个command的操作.
            vars.RecordAccessForSubmatrix(c.arg3, kReadAccess, &attr);
            if (nnet.GetComponent(c.arg1)->Properties() & kPropagateAdds)
              vars.RecordAccessForSubmatrix(c.arg4, kReadWriteAccess, &attr);
            else
              vars.RecordAccessForSubmatrix(c.arg4, kWriteAccess, &attr);
            break;

          default:
            KALDI_ERR << "Unknown command type.";
        }
        SortAndUniq(&attr.variables_read);
        SortAndUniq(&attr.variables_written);
        SortAndUniq(&attr.submatrices_read);
        SortAndUniq(&attr.submatrices_written);
        SortAndUniq(&attr.matrices_read);
        SortAndUniq(&attr.matrices_written);
      }
    }


    // 计算每个 variable 的访问流程 read write readAndWrite list
    // 从每个 command的 attributes 中获得 variables_read variables_write variable列表
    // 这样对每个 variable的access访问列表 增加 read write readAndWrite等访问操作.
    ComputeVariableAccesses(variables, command_attributes, &variable_accesses);


    void ComputeVariableAccesses(
        const ComputationVariables &variables,
        const std::vector<CommandAttributes> &command_attributes,
        std::vector<std::vector<Access> > *variable_accesses) {

      int32 num_variables = variables.NumVariables(),
          num_commands = command_attributes.size();

      // 每个variable 都有一个access访问权限列表list.
      variable_accesses->clear();
      variable_accesses->resize(num_variables);

      // 对每个command, 每个command 都具有对应的attr, 即对应的 某个 matrix的variable小块,
      // 然后每个 variable 小块 会被不同的command 使用, 所以对每个variable 的访问 是一个序列的list.
      for (int32 c = 0; c < num_commands; c++) {
        const CommandAttributes &attr = command_attributes[c];
        KALDI_ASSERT(IsSortedAndUniq(attr.variables_read));
        KALDI_ASSERT(IsSortedAndUniq(attr.variables_written));
      
        std::vector<int32> all_variables;
        all_variables.reserve(attr.variables_read.size() +  attr.variables_written.size());
      
        all_variables.insert(all_variables.end(), attr.variables_read.begin(), attr.variables_read.end());
        all_variables.insert(all_variables.end(), attr.variables_written.begin(), attr.variables_written.end());
        SortAndUniq(&all_variables);

        std::vector<int32>::const_iterator iter = all_variables.begin(),
            end = all_variables.end();
        for (; iter != end; ++iter) {
          int32 variable_index = *iter;
          bool is_read = std::binary_search(attr.variables_read.begin(), attr.variables_read.end(), variable_index),
              is_written = (!is_read ? true :
                            std::binary_search(attr.variables_written.begin(),
                                               attr.variables_written.end(),
                                               variable_index));
          if (is_read && is_written) {
            (*variable_accesses)[variable_index].push_back(
                Access(c, kReadWriteAccess));
          } else if (is_read) {
            (*variable_accesses)[variable_index].push_back(
                Access(c, kReadAccess));
          } else {
            (*variable_accesses)[variable_index].push_back(
                Access(c, kWriteAccess));
          }
        }
      }
    }


    // 类似上面的 variable_accesses
    // 这里确定每个matrix的 访问流程read write readAndWrite list,
    // 并且向 allocate deallocate等命令的 matrix 增加上对应的command-index.
    ComputeMatrixAccesses(nnet, computation, variables, command_attributes,
                          &matrix_accesses);

    void ComputeMatrixAccesses(
        const Nnet &nnet,
        const NnetComputation &computation,
        const ComputationVariables &variables,
        const std::vector<CommandAttributes> &command_attributes,
        std::vector<MatrixAccesses> *matrix_accesses) {

      //每个 matrix都具有 访问情况 list
      int32 num_matrices = computation.matrices.size(),
          num_commands = command_attributes.size();
      matrix_accesses->clear();
      matrix_accesses->resize(num_matrices);

      // 对每个command, 取CommandAttributes中的 read write matrices.
      for (int32 c = 0; c < num_commands; c++) {
        const CommandAttributes &attr = command_attributes[c];
        KALDI_ASSERT(IsSortedAndUniq(attr.matrices_read));
        KALDI_ASSERT(IsSortedAndUniq(attr.matrices_written));
        std::vector<int32> all_matrices;
        all_matrices.reserve(attr.matrices_read.size() +
                             attr.matrices_written.size());
        all_matrices.insert(all_matrices.end(), attr.matrices_read.begin(),
                            attr.matrices_read.end());
        all_matrices.insert(all_matrices.end(), attr.matrices_written.begin(),
                            attr.matrices_written.end());
        SortAndUniq(&all_matrices);

        // 对每个command 内的 所有matrix. 将对应的matrix的访问方式 加入对应的matrix的 访问列表中
        std::vector<int32>::const_iterator iter = all_matrices.begin(),
            end = all_matrices.end();
        for (; iter != end; ++iter) {
          int32 matrix_index = *iter;
          bool is_read = std::binary_search(attr.matrices_read.begin(),
                                            attr.matrices_read.end(),
                                            matrix_index),
              is_written = (!is_read ? true :
                            std::binary_search(attr.matrices_written.begin(),
                                               attr.matrices_written.end(),
                                               matrix_index));
          if (is_read && is_written) {
            (*matrix_accesses)[matrix_index].accesses.push_back(
                Access(c, kReadWriteAccess));
          } else if (is_read) {
            (*matrix_accesses)[matrix_index].accesses.push_back(
                Access(c, kReadAccess));
          } else {
            (*matrix_accesses)[matrix_index].accesses.push_back(
                Access(c, kWriteAccess));
          }
        }


        // 这里设置 对应matrix的allocate deallocate等命令的 command-index c,
        // 这样每个matrix 都知道自己的 allocate 等操作的 执行command确定是哪个.
        const NnetComputation::Command &command = computation.commands[c];
        int32 matrix_index1, matrix_index2;

        switch (command.command_type) {
          case kAllocMatrix:
            if (!computation.IsWholeMatrix(command.arg1))
              KALDI_ERR << "Command does not operate on whole matrix";
            matrix_index1 = computation.submatrices[command.arg1].matrix_index;
            if ((*matrix_accesses)[matrix_index1].allocate_command != -1)
              KALDI_ERR << "Matrix " << matrix_index1 << " initialized twice.";
            (*matrix_accesses)[matrix_index1].allocate_command = c;
            break;
          case kSwapMatrix:
            if (!computation.IsWholeMatrix(command.arg1))
              KALDI_ERR << "Command does not operate on whole matrix";
            matrix_index1 = computation.submatrices[command.arg1].matrix_index;
            KALDI_ASSERT(computation.IsWholeMatrix(command.arg2));
            matrix_index2 = computation.submatrices[command.arg2].matrix_index;
            if ((*matrix_accesses)[matrix_index1].allocate_command != -1)
              KALDI_ERR << "Matrix " << matrix_index1 << " initialized twice.";
            (*matrix_accesses)[matrix_index1].allocate_command = c;
            if ((*matrix_accesses)[matrix_index2].deallocate_command != -1)
              KALDI_ERR << "Matrix " << matrix_index2 << " destroyed twice.";
            (*matrix_accesses)[matrix_index2].deallocate_command = c;
            break;
          case kDeallocMatrix:
            if (!computation.IsWholeMatrix(command.arg1))
              KALDI_ERR << "Command does not operate on whole matrix";
            matrix_index1 = computation.submatrices[command.arg1].matrix_index;
            if ((*matrix_accesses)[matrix_index1].deallocate_command != -1)
              KALDI_ERR << "Matrix " << matrix_index1 << " destroyed twice.";
            (*matrix_accesses)[matrix_index1].deallocate_command = c;
            break;
          case kAcceptInput:
            if (!computation.IsWholeMatrix(command.arg1))
              KALDI_ERR << "Command does not operate on whole matrix";
            matrix_index1 = computation.submatrices[command.arg1].matrix_index;
            (*matrix_accesses)[matrix_index1].is_input = true;
            // If a certain matrix is accepted as input multiple times, we
            // count the first one as allocating it (the second will just
            // allocate it again, which is harmless).
            if ((*matrix_accesses)[matrix_index1].allocate_command == -1)
              (*matrix_accesses)[matrix_index1].allocate_command = c;
            break;
          case kProvideOutput:
            if (!computation.IsWholeMatrix(command.arg1))
              KALDI_ERR << "Command does not operate on whole matrix";
            matrix_index1 = computation.submatrices[command.arg1].matrix_index;
            (*matrix_accesses)[matrix_index1].is_output = true;
            break;
          default:
            ;
        }
      }
    }

  }



  void ComputationVariables::RecordAccessForSubmatrix(
      int32 submatrix_index,
      AccessType access_type,
      CommandAttributes *ca) const {
    if (submatrix_index == 0)
      return;
    KALDI_ASSERT(static_cast<size_t>(submatrix_index) <
                 submatrix_to_matrix_.size());
    int32 matrix_index = submatrix_to_matrix_[submatrix_index];
    bool is_whole_matrix = submatrix_is_whole_matrix_[submatrix_index];
    switch (access_type) {
      case kReadAccess:
        // 将submatrix_index 中的variable小块,
        // 加入到 对应CommandAttributes 的 variables_read队列中.
        AppendVariablesForSubmatrix(submatrix_index,
                                    &(ca->variables_read));
      
        ca->matrices_read.push_back(matrix_index);
        ca->submatrices_read.push_back(submatrix_index);
        break;
      case kWriteAccess:
        AppendVariablesForSubmatrix(submatrix_index,
                                    &(ca->variables_written));
        ca->submatrices_written.push_back(submatrix_index);
        ca->matrices_written.push_back(matrix_index);
        // if submatrix does not span the full row range of the matrix,
        // a write operation has to be considered a read/write operation
        // on the underlying matrix
        if (!is_whole_matrix)
          ca->matrices_read.push_back(matrix_index);
        break;
      case kReadWriteAccess:
        AppendVariablesForSubmatrix(submatrix_index,
                                    &(ca->variables_written));
        AppendVariablesForSubmatrix(submatrix_index,
                                    &(ca->variables_read));
        ca->submatrices_written.push_back(submatrix_index);
        ca->submatrices_read.push_back(submatrix_index);
        ca->matrices_written.push_back(matrix_index);
        ca->matrices_read.push_back(matrix_index);
    }
  }





















  // 
  /**

     这个类 将computation中的 matrices 和 sub-matrices 和 想象中的"variables" 联系起来
     这样我们可以认为 这些操作是作用在各个独立的variables上的.
     我们可以做一些分析 让我进行优化.
     原理上 将这些变量对应上各个matrices的元素是有意义的, 但是这样很低效,
     另一方便我们可以做一些粗糙的分析 将variables对应上对应的matrices, 但是这样会导致结果分析不准确.

     因此我们做一下修改, 结果会让我们希望的足够准确.
     通过将variables 对应上我们访问的matrices中最确定的行和列区域.

     我们这样做
     对每个computation中的matrix我们获得一个 所有 分割点-split_points, 其中每个row col
     对应上每个start end 并定义一个split_point_index 作为结果中的索引.
     variable 可以被定义为三元组(matrix, row_split_point_index, column_split_point_index)
     然后我们将这个三元组映射到一个int 索引 variable_index. 是一个0-based的索引
     通过列出所有存在的variables(从matrix的索引index第一个开始迭代)
     然后是行的 split-point-index, 然后是col的 split-point-index
     最后,如果我们知道matrix-index, row-split-point-index column-split-point-index, 我们可以计算出
     variable-index . 计算公式为 vari
     variable-index = matrix_to_variable_index_[matrix-index] +
     row-split-point-index * num-column-variables-for-this-matrix +
     column-split-point-index
     在代码中 num-column-variables-for-this-matrix equals column_split_points_[matrix-index].size()-1.

     数组中 matrix_to_variable_index_ 是一个预计算数组 告知我们对一个matrix开始 具体是那个variable?
     每个sub-matrix 都不会对应到一系列的varibales, 因为这些list 是邻接的range范围 我们可以只保存row col split-points
     对应到每个 submatrix的 start end.
     除此之外 对每个submatrix 不论是否它跨越整个基本矩阵, 我们需要知道的原因是
     这是一个写入操作 对matrix的确定部分 必须是一个rw操作的基本
     因为最终的内容经过操作后, 会依赖与原本的内容.???
   
     Each sub-matrix in the computation will now correspond to a list of
     variables, and because these lists are always a contiguous range we can just
     store the row and column split-points corresponding to the start and end of
     the submatrix.  In addition we note, for each submatrix, whether it spans
     the entirety of the underlying matrix.  The reason we need to know this is
     that a write operation to just part of a matrix would have to be classed as
     a read-write operation on the underlying matrix because the final contents
     after the operation would in that case depend on the original contents.
  */
  class ComputationVariables {
   public:
    // This function must only be called once per object.
    void Init(const NnetComputation &computation);

    // 这个函数 更新commandAttributes对象, 为了记录一个在对应sub-matrixd的varibales read write 或者 rw操作
    // 并且更新matrix被访问变量 通过增加基本matrix的数量. 它做的不明显的是如果访问类型是write 并且sub-matrix
    // 并没有跨越其属于的matrix的所有行(因此并没有跨越我们定义对应的variables的所有内容)
    // 这个方位就会被记录为 read and write, 因为操作的结果 需要一个预先存在的内容, 所以它就不可能就是一个简单的write操作
    void RecordAccessForSubmatrix(
        int32 submatrix_index,
        AccessType access_type,
        CommandAttributes *ca) const;

    // 向variables_indexes 中增加排好序的 matrix index对应的 varibles list, 
    void AppendVariablesForMatrix(
        int32 matrix_index,
        std::vector<int32> *variable_indexes) const;


    // Appends to variable_indexes the sorted list of variables corresponding to a
    // submatrix index.
    void AppendVariablesForSubmatrix(
        int32 submatrix_index,
        std::vector<int32> *variable_indexes) const;

    // note: variables are zero-indexed.
    int32 NumVariables() const { return num_variables_; }

    int32 GetMatrixForVariable(int32 variable) const;

    // returns a string that describes a variable in Matlab-like format (but with
    // zero indexing): something like "m1" or "m1(0:99,:)" or "m1(0:19,10:49)"
    std::string DescribeVariable(int32 variable) const;

   private:
    // 设置 split_points_ matrix_to_variable_index num_variables_
    // sets up split_points_, matrix_to_variable_index_, and num_variables_.
    // called from constructor.
    void ComputeSplitPoints(const NnetComputation &computation);
    // sets up variables_for_submatrix_, is_full_matrix_, and submatrix_to_matrix_.  called
    // from constructor.
    void ComputeVariablesForSubmatrix(const NnetComputation &computation);
    // sets up variable_to_matrix_.  called from constructor.
    void ComputeVariableToMatrix();

    // This function assumes that 'sorted_vec' is sorted and unique, and that
    // 'i' is an element of 'sorted_vec'; it returns the index of 'i' in vec,
    // i.e. the k such that sorted_vec[k] == i.
    static int32 FindIndexOf(const std::vector<int32> &sorted_vec, int32 i);

    // 用matrix-index索引 得到一个list  保存我们所有的split points 其中保存col的start和 end.
    // 例如 如果第三个matrix 具有20 col 然后将其分割为0:9 和10:19
    // split_points_[3] == [0, 10, 20]
    // column_split_points_[0] 永远是空, 因为0 matrix-index 是为了 空matrix 保留的.
    std::vector<std::vector<int32> > column_split_points_;
    // This is as column_split_points_, except for row indexes instead of column
    // indexes.
    std::vector<std::vector<int32> > row_split_points_;

    // 将matrix-index 映射到 variable-index 作为它的第一个分割点.
    // 为了写代码方便 这里保留一个额外的元素在最后, 等于所有variables的总数.
    // 对每个matrix m, m 矩阵会具有总共 num-row-variables*num-column-variables 的variables.
    // num-row-variables 等于 row_split_points_[m].size-1, 并且
    // num-column-variables 等于 column_split_points_[m].size -1 .
    std::vector<int32> matrix_to_variable_index_;

    std::vector<int32> submatrix_to_matrix_;
    // 以submatrix-index 为索引, 如果对应的submatrix跨越了 对应matrix 的完整row 和 col 区域, 保存为true
    // 无论write操作被分类为 wirte 或者 read-write 都会被影响.
    std::vector<bool> submatrix_is_whole_matrix_;


    // 记录每个variable 的 matrix-index
    // records the matrix index underlying each variable.
    std::vector<int32> variable_to_matrix_;

    int32 num_variables_;

    // 每个submatrix 具有的潜在的 variables
    // For each submatrix, a list of the variables underlying it.
    std::vector<std::vector<int32> > variables_for_submatrix_;
  };


}
