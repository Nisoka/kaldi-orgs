void VariableMergingOptimization(const NnetOptimizeOptions &config,
                                 const Nnet &nnet,
                                 NnetComputation *computation) {

  bool changed = true;
  while (changed) {
    changed = false;
    // variable合并优化器 
    VariableMergingOptimizer opt(config, nnet, computation);
    // 
    if (opt.MergeVariables())
      changed = true;
  }
}


VariableMergingOptimizer::VariableMergingOptimizer(
    const NnetOptimizeOptions &config,
    const Nnet &nnet,
    NnetComputation *computation):
    config_(config), nnet_(nnet),
    computation_(computation),
    already_called_merge_variables_(false) {
  // 重新生成了一个 analyzer_, 以前不是已经生成了一个么
  analyzer_.Init(nnet, *computation);
  ComputeMatrixToSubmatrix(*computation_, &matrix_to_submatrix_);
  void ComputeMatrixToSubmatrix(
      const NnetComputation &computation,
      std::vector<std::vector<int32> > *mat_to_submat) {
    int32 num_matrices = computation.matrices.size(),
        num_submatrices = computation.submatrices.size();
    mat_to_submat->clear();
    mat_to_submat->resize(num_matrices);
    
    for (int32 submatrix_index = 1;
         submatrix_index < num_submatrices;
         submatrix_index++) {
      int32 matrix_index = computation.submatrices[submatrix_index].matrix_index;
      KALDI_ASSERT(matrix_index > 0 && matrix_index < num_matrices);
      (*mat_to_submat)[matrix_index].push_back(submatrix_index);
    }
  }
  
  
  
  variable_dirty_.resize(analyzer_.variables.NumVariables(), false);
}





bool VariableMergingOptimizer::MergeVariables() {
  KALDI_ASSERT(!already_called_merge_variables_);
  already_called_merge_variables_ = true;
  if (!config_.optimize)
    return false;
  bool merged = false;
  int32 num_commands = computation_->commands.size();
  
  for (int32 command_index = 0; command_index < num_commands; command_index++) {

    // 这个循环目的是查找 pair <s1, s2> , 可能被merge到一个variable中的 submatrix对.
    const NnetComputation::Command &c = computation_->commands[command_index];
    int32 s1 = -1, s2 = -1;
    
    // TODO: add kScale command and remove the check for 1.0
    if (c.command_type == kMatrixCopy &&
        //        c.alpha == 1.0 &&
        config_.remove_assignments) {
      s2 = c.arg1;  // s2 is the written-to matrix.
      s1 = c.arg2;
    }


    else if (c.command_type == kPropagate &&  config_.propagate_in_place) {
      const Component *component = nnet_.GetComponent(c.arg1);
      if (component->Properties() & kPropagateInPlace) {
        s1 = c.arg3;
        s2 = c.arg4;  // s2 is the written-to matrix.
      }
    }


    else if ((c.command_type == kBackprop || c.command_type == kBackpropNoModelUpdate) &&  config_.backprop_in_place) {
      const Component *component = nnet_.GetComponent(c.arg1);
      if (component->Properties() & kBackpropInPlace) {
        s1 = c.arg5;
        s2 = c.arg6;  // s2 is the written-to matrix.
      }
    }

    
    // 
    if (s1 > 0 && s2 > 0) {
      // MayBeMerged 内部判断所有 7 个condition 如果都通过, 则认为可以进行merge合并.
      std::pair<bool,bool> p = MayBeMerged(command_index, s1, s2);
      if (p.first) {
        DoMerge(command_index, s1, s2);
        merged = true;
      } else if (p.second) {
        DoMerge(command_index, s2, s1);
        merged = true;
      }
    }
  }
  if (merged) {
    RenumberComputation(computation_);
    RemoveNoOps(computation_);
  }
  return merged;
}



std::pair<bool,bool> VariableMergingOptimizer::MayBeMerged(
    int32 command_index, int32 s1, int32 s2) const {
  KALDI_ASSERT(s1 > 0 && s2 > 0 && static_cast<size_t>(command_index) <  computation_->commands.size());
  
  if (!config_.allow_left_merge && !config_.allow_right_merge)
    return std::pair<bool,bool>(false,false);
  
  int32
      m1 = computation_->submatrices[s1].matrix_index,
      m2 = computation_->submatrices[s2].matrix_index;
  // we can't merge two different submatrices of the same matrix.
  if (m1 == m2) return std::pair<bool,bool>(false,false);
  
  std::vector<int32> variable_indexes;
  // 将s1 s2 submatrix 中的variable 都加入到 variable_indexes中
  analyzer_.variables.AppendVariablesForSubmatrix(s1, &variable_indexes);
  analyzer_.variables.AppendVariablesForSubmatrix(s2, &variable_indexes);

  std::vector<int32>::iterator iter = variable_indexes.begin(),
      end = variable_indexes.end();
  // condition c5:
  for (; iter != end; ++iter)
    if (variable_dirty_[*iter])
      return std::pair<bool,bool>(false,false);

  
  const std::vector<MatrixAccesses> &matrix_accesses = analyzer_.matrix_accesses;
  const MatrixAccesses &m1_access = matrix_accesses[m1],
      &m2_access = matrix_accesses[m2];
  // condition c1:
  if ((m1_access.is_input && m2_access.is_input) ||
      (m1_access.is_output && m2_access.is_output))
    return std::pair<bool,bool>(false,false);

  // condition c2:
  if ((m1_access.is_input || m1_access.is_output ||
       m2_access.is_input || m2_access.is_output) &&
      (!computation_->IsWholeMatrix(s1) ||
       !computation_->IsWholeMatrix(s2)))
    return std::pair<bool,bool>(false,false);

  
  bool left = config_.allow_left_merge,
      right = config_.allow_right_merge;

  // 肯定至少一个是WholeMatrix
  // condition c3:
  if (!computation_->IsWholeMatrix(s2)) left = false;

  // condition c4:
  if (!computation_->IsWholeMatrix(s1)) right = false;

  // condition c6:
  if (computation_->matrices[m2].stride_type == kStrideEqualNumCols &&
      !computation_->IsWholeMatrix(s1)) left = false;

  // condition c7:
  if (computation_->matrices[m1].stride_type == kStrideEqualNumCols &&
      !computation_->IsWholeMatrix(s2)) right = false;


  if (!left && !right)  // save some time.
    return std::pair<bool,bool>(false,false);


  bool is_assignment = (computation_->commands[command_index].command_type == kMatrixCopy);
  ComputationAnalysis analysis(*computation_, analyzer_);
  if (is_assignment) {
    if (analysis.FirstNontrivialAccess(s2) == command_index &&
        analysis.LastWriteAccess(s1) < command_index &&
        analysis.LastAccess(s1) <
        analysis.DataInvalidatedCommand(command_index, s2)) {
      return std::pair<bool,bool>(left, right);  // possible success.
    }
  } else {
    if (analysis.FirstNontrivialAccess(s2) == command_index &&
        analysis.LastAccess(s1) == command_index) {
      return std::pair<bool,bool>(left, right);  // possible success.
    }
  }
  // failure.
  return std::pair<bool,bool>(false,false);
}






/**
   这个函数返回一个 submatrix-info 负责替换在a的matrix_index域保存的matrix-index  本质将保存sub-matrix b的matrix-index
   当然 matrix-index 将会是 b的 matrix_index,    但是我们可能必须修改row 和 col offset.
   这个想法是 submatrix submat_b 应该具有相同的维度 与 submat_a的潜在matrix矩阵.
 */

// submat_a --- one_s_to_discard
// submat_b --- s_to_keep
static NnetComputation::SubMatrixInfo GetSubMatrixOfSubMatrix(
    const NnetComputation &computation, int32 submat_a, int32 submat_b) {
  
  const NnetComputation::SubMatrixInfo &a = computation.submatrices[submat_a],
                                       &b = computation.submatrices[submat_b];
  
  const NnetComputation::MatrixInfo &a_mat =   computation.matrices[a.matrix_index];

  // 哪里确定的条件?？？？ 我曹！
  // 这是前面必须满足的条件么?
  KALDI_ASSERT(a_mat.num_rows == b.num_rows && a_mat.num_cols == b.num_cols);
  
  NnetComputation::SubMatrixInfo ans;
  ans.matrix_index = b.matrix_index;
  ans.row_offset = a.row_offset + b.row_offset;
  ans.num_rows = a.num_rows;
  ans.col_offset = a.col_offset + b.col_offset;
  ans.num_cols = a.num_cols;
  return ans;
}


// 注意 可以合并的两个submatrix 被discard的那个一定是个完整matrix
// 并且还会和m_to_keep 的大小完全相等??????

// 将两个 submatrix 进行合并 s_to_keep, 是要保留的submatrix.
void VariableMergingOptimizer::DoMerge(int32 command_index,
                                       int32 s_to_keep,
                                       int32 s_to_discard) {
  // 防止未来的优化操作 优化这些submatrix.
  // Prevent further optimizations touching either submatrix (we can try again
  // in a later round of optimization, with a new instance of this class).
  MarkAsDirty(s_to_keep);
  MarkAsDirty(s_to_discard);

  int32
      m_to_keep = computation_->submatrices[s_to_keep].matrix_index,
      m_to_discard = computation_->submatrices[s_to_discard].matrix_index;
  KALDI_ASSERT(m_to_keep != m_to_discard && m_to_keep > 0 && m_to_discard > 0);

  // m_to_discard 矩阵的 所有子矩阵submatrix 
  {
    std::vector<int32>::const_iterator
        iter = matrix_to_submatrix_[m_to_discard].begin(),
        end = matrix_to_submatrix_[m_to_discard].end();

    // 对m_to_discard的所有submatrix 都根据 s_to_keep 生成一个 submatrix.
    // 保存会原地, 实际上就是将原本 以m_to_discard matrix为数据的submatrix
    // 都改变为 以 m_to_keep的matrix为数据。
    // 这样就将m_to_discard的matrix 删掉了??????
    for (; iter != end; ++iter) {
      int32 submatrix_index = *iter;
      computation_->submatrices[submatrix_index] = GetSubMatrixOfSubMatrix(*computation_, submatrix_index, s_to_keep);
    }
  }

  
  ComputationAnalysis analysis(*computation_, analyzer_);
  NnetComputation::Command &c = computation_->commands[command_index];
  // 矩阵访问过程.
  const std::vector<MatrixAccesses> &matrix_accesses = analyzer_.matrix_accesses;

  // 如果是个 matrixCopy命令, 可以将 matrixCopy命令直接变为 no-op命令
  // 如果是个 matrixCopy+sacle命令, 保留命令,
  // 它会具有 加权矩阵的作用(会被映射 arg1=arg2??)
  if (c.command_type == kMatrixCopy && c.alpha == 1.0) {
    // remove the command.
    c.command_type = kNoOperation;
    c.arg1 = -1;
    c.arg2 = -1;
  }

  // 我们需要确保 只有一个 deallocate 命令
  // 如果 两个matrix都不是output 这里就会有两个deallocate命令, 我们会保存 m_to_keep的那个命令
  // 如果大小不同, 会保存两者中较大的按个, 所以它是两个submatrix中引用了 完整matrix的那个.

  // 如果两者中一个是output, 删除其中不是 output的 deallocation命令
  // 作为上面逻辑的一个简化, 如果discard的matrix 具有deallocation comand, 即
  // 即如果这个矩阵不是一个output, 就移除该command, 否则移除keep矩阵的deallocation command.

  int32 dealloc_keep = matrix_accesses[m_to_keep].deallocate_command,
      dealloc_discard = matrix_accesses[m_to_discard].deallocate_command;
  if (dealloc_discard != -1) {
    computation_->commands[dealloc_discard].command_type = kNoOperation;
  } else {
    KALDI_ASSERT(dealloc_keep != -1);
    computation_->commands[dealloc_keep].command_type = kNoOperation;
  }

  {
    // m_to_keep 和 m_to_discard都会具有 allocate command
    // 因为每个matrix都会进行申请空间, 如果其中一个matrix是 kAacceptINput, 就删除另外一个
    // 因为kAcceptInput命令的位置很重要? 否则删除discard的那个命令.
    // 一个简化：
    // 如果discard的allocate命令是 kAcceptInput， 删除kepp的allocate命令, 否则删除discard的allocate命令
    // 注意, 后面我们会对submatrix重新编号, 他们都会应用相同的matrix， 但是我们需要使用submatrix 引用完整matrix
    // 我们保留的那个 会一直引用完整matrix， 在这样情况下, 他们中一个是输入， 两个submatrix 都确保引用完整matrix
    // 这是通过我们用来决定那个matrix可以被merge的逻辑确定了的。
    int32
        alloc_keep = matrix_accesses[m_to_keep].allocate_command,
        alloc_discard = matrix_accesses[m_to_discard].allocate_command;

    KALDI_ASSERT(alloc_keep != -1 && alloc_discard != -1);
    KALDI_ASSERT(analysis.FirstNontrivialMatrixAccess(m_to_discard) >
                 alloc_keep);

    NnetComputation::Command
        &keep_alloc_command = computation_->commands[alloc_keep],
        &discard_alloc_command = computation_->commands[alloc_discard];
    int32 matrix_whose_zeroing_to_discard;
    if (discard_alloc_command.command_type == kAcceptInput) {
      keep_alloc_command.command_type = kNoOperation;
      matrix_whose_zeroing_to_discard = m_to_keep;
    } else {
      discard_alloc_command.command_type = kNoOperation;
      matrix_whose_zeroing_to_discard = m_to_discard;
    }


    // 现在移除命令将其中一个矩阵（我们刚刚丢弃了allocate命令的矩阵）归零的命令， 
    // Now remove the command that zeroed one of the matrices
    // (the one whose allocation command we just discarded).
    int32 zeroing_command_to_discard =
     matrix_accesses[matrix_whose_zeroing_to_discard].accesses[0].command_index;
    NnetComputation::Command &zeroing_command =
        computation_->commands[zeroing_command_to_discard];
    if (zeroing_command.command_type == kSetConst &&
        zeroing_command.alpha == 0.0) {
      // if 'zeroing_command' actually *was* a zeroing command, then remove it.
      zeroing_command.command_type = kNoOperation;
    }
  }

  //  If the matrix to discard had stride_type == kStrideEqualNumCols, set the
  //  stride type of the matrix we're keeping to kStrideEqualNumCols.
  if (computation_->matrices[m_to_discard].stride_type == kStrideEqualNumCols) {
    computation_->matrices[m_to_keep].stride_type = kStrideEqualNumCols;
    // ... and perform an additional check.
    KALDI_ASSERT(computation_->matrices[m_to_discard].num_rows ==
                 computation_->matrices[m_to_keep].num_rows &&
                 computation_->matrices[m_to_discard].num_cols ==
                 computation_->matrices[m_to_keep].num_cols);
  }
}

