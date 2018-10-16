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



  // 获得matrix to submatrix的映射  matrix -- submatrix1 submatrix2 submatrix....  
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



  // 对所有variable设置 非dirty标记, 表示可以进行合并.
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


// 对每个命令 找到可能可以被合并的 两个submatrices
// (实际其中至少一个是wholeMatrices, 因此可以合并掉m_to_discard的matrix)
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
  // 并且wholeMatrix的会被合并掉discard.
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

  // 经过前面确定, a_mat的一个完全submatrix 与 submat_b 完全相同, 所以
  // a_mat直接与 submat_b 完全相同.
  KALDI_ASSERT(a_mat.num_rows == b.num_rows && a_mat.num_cols == b.num_cols);

  // 因此 原本 a_mat上的各个submat_a 都应该移植以 submat_b 为数据
  // 所以更新后的各个submat_a 都是submat_b的子矩阵, 那么相对位置就通过如下设置.
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

  // m_to_discard 矩阵的s_to_discard子矩阵 是wholeMatrix的
  // s_to_discard 子矩阵 和 s_to_keep子矩阵是相同规格的 需要合并, 只保存一个s_to_keep.
  // 这样m_to_discard 会被删除, 因此m_to_discard的所有子矩阵 都应该以s_to_keep为父矩阵.
  // 如下就是安排 m_to_discard的各个子矩阵的新父矩阵
  {
    std::vector<int32>::const_iterator
        iter = matrix_to_submatrix_[m_to_discard].begin(),
        end = matrix_to_submatrix_[m_to_discard].end();

    // 安排submatrix_index的父矩阵为 s_to_keep.
    for (; iter != end; ++iter) {
      int32 submatrix_index = *iter;
      computation_->submatrices[submatrix_index] = GetSubMatrixOfSubMatrix(*computation_, submatrix_index, s_to_keep);
    }
  }



  // =================== 删除当前矩阵操作命令 ================
  ComputationAnalysis analysis(*computation_, analyzer_);
  NnetComputation::Command &c = computation_->commands[command_index];
  // 矩阵访问过程.
  const std::vector<MatrixAccesses> &matrix_accesses = analyzer_.matrix_accesses;

  // 如果是kMatrixCopy 命令, 则按如下方式删除 kMatrixCopy命令
  if (c.command_type == kMatrixCopy && c.alpha == 1.0) {
    // remove the command.
    c.command_type = kNoOperation;
    c.arg1 = -1;
    c.arg2 = -1;
  }

  // 对 Matrix矩阵的 allocate_command命令 进行删除, 但是删除的不一定是m_to_discard的命令
  // 我们需要确保 只有一个 deallocate 命令
  // 如果 两个matrix都不是output 这里就会有两个deallocate命令, 我们会保存 m_to_keep的那个命令
  // 如果大小不同, 会保存两者中较大的按个, 所以它是两个submatrix中引用了 完整matrix的那个.
  // 如果两者中一个是output, 删除其中不是 output的 deallocation命令

  // 作为上面逻辑的一个简化, 如果discard的matrix 具有deallocation comand,
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
    // 因为丢弃了 allocate 那么 也必然不能留着 归零命令.
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

  // 如果待丢弃的matrix 有 stride type == kStrideEqualNumCols, 设置其keep的matrix的stride type  为kStrideEqualNumCols.
  if (computation_->matrices[m_to_discard].stride_type == kStrideEqualNumCols) {
    computation_->matrices[m_to_keep].stride_type = kStrideEqualNumCols;
    // ... and perform an additional check.
    KALDI_ASSERT(computation_->matrices[m_to_discard].num_rows ==
                 computation_->matrices[m_to_keep].num_rows &&
                 computation_->matrices[m_to_discard].num_cols ==
                 computation_->matrices[m_to_keep].num_cols);
  }
}






// 对 computation 重新编号
void RenumberComputation(NnetComputation *computation) {
  ComputationRenumberer renumberer(computation);
  renumberer.Renumber();
}

// 对 matrices submatrices indexes indexes_multi
// 1 重新编号
// 2 去除重复
// 3 清理掉不必要的
void ComputationRenumberer::Renumber() {
  // 移除无用indexes_multi?
  RemoveUnusedIndexesMulti();
  void ComputationRenumberer::RemoveUnusedIndexesMulti() {
    // 所有的indexes_multi总数
    int32 num_indexes_multi = computation_->indexes_multi.size();
    if (num_indexes_multi == 0)
      return;  // Nothing to do.  An optimization.
    // 对所有indexes_multi 分配一个标记
    std::vector<bool> indexes_multi_used(num_indexes_multi, false);

    // 将所有 kMulti类型命令的参数 matrix-index  copy => indexes_multi_args
    std::vector<int32*> indexes_multi_args;
    IdentifyIndexesMultiArgs(&(computation_->commands), &indexes_multi_args);
    
    // 每个kMulti命令的 matrix-index
    // 将每个matrix-index 是否被用到indexes_multi_used = 1
    std::vector<int32*>::iterator iter = indexes_multi_args.begin(),
        end = indexes_multi_args.end();
    for (; iter != end; ++iter) {
      int32 indexes_multi_index = **iter;
      // 每个kMulti命令的 multi matrix-index = 1
      indexes_multi_used[indexes_multi_index] = 1;
    }

    // 将indexes_multi数组 映射为 新的编号.
    std::vector<int32> old_to_new(num_indexes_multi, -1);
    int32 new_num_indexes_multi = CreateRenumbering(indexes_multi_used, &old_to_new);

    int32 ComputationRenumberer::CreateRenumbering(
        const std::vector<bool> &used,
        std::vector<int32> *renumbering) {
      renumbering->clear();
      renumbering->reserve(used.size());

      // 如果对应的kMulti的 matrix-index是被用到的,
      // 对其进行编号为 cur_index++.
      std::vector<bool>::const_iterator iter = used.begin(), end = used.end();
      int32 cur_index = 0;
      for (; iter != end; ++iter) {
        if (*iter)
          renumbering->push_back(cur_index++);
        else
          renumbering->push_back(-1);
      }
      return cur_index;
    }
    

    // 新安排的 和 原本的indexes 总数没变, 直接不进行操作.
    if (new_num_indexes_multi == num_indexes_multi)
      return;  // Nothing to do.  An optimization.

    // 重新分配一个 indexes_multi 按新编号将原本的 indexes_multi保存下.
    std::vector<std::vector<std::pair<int32, int32> > >  new_indexes_multi(new_num_indexes_multi);
    for (int32 i = 0; i < num_indexes_multi; i++) {
      if (old_to_new[i] != -1)
        new_indexes_multi[old_to_new[i]].swap(computation_->indexes_multi[i]);
    }
    computation_->indexes_multi.swap(new_indexes_multi);
    // renumber within the commands.
    for (iter = indexes_multi_args.begin(); iter != end; ++iter)
      **iter = old_to_new[**iter];
  }

  // 计算所有必要的submatrix
  ComputeSubmatrixIsUsed();
  void ComputationRenumberer::ComputeSubmatrixIsUsed() {
    int32 num_submatrices = computation_->submatrices.size();
    submatrix_is_used_.clear();
    submatrix_is_used_.resize(num_submatrices, false);
    submatrix_is_used_[0] = true;
    
    // the zeroth element of the array is 'special', it refers to the
    // zero submatrix, and we don't want to renumber it.
    std::vector<int32*> submatrix_args;
    IdentifySubmatrixArgsInComputation(computation_, &submatrix_args);
    void IdentifySubmatrixArgsInComputation(NnetComputation *computation,
                                            std::vector<int32*> *submatrix_args) {

      // 返回所有命令 必要的所有 submatrix-index
      // command数组
      // submatrix_args vector<int32*>
      IdentifySubmatrixArgs(&(computation->commands), submatrix_args);
      void IdentifySubmatrixArgs(std::vector<NnetComputation::Command> *commands,
                                 std::vector<int32*> *submatrix_args) {
        submatrix_args->clear();
        std::vector<NnetComputation::Command>::iterator iter = commands->begin(),
            end = commands->end();
        std::vector<int32*> this_submatrix_args;

        // IdentifySubmatrixArgs()
        // 对每个command  根据命令 增加作为参数的 submatrix-index.
        // eg
        // allocate -  push(c.arg1)
        // propagate - push(c.arg3, c.arg4)
        //
        // 向submatrix_args中增加 识别出来的各个必然用到的submatrix-index
        // 而像之前优化操作中 将很多m_to_discard的matrix的命令都修改为了 noOperate
        // 因此就不需要考虑这些submatrix, 也就被删除了.
        for (; iter != end; ++iter) {
          IdentifySubmatrixArgs(&(*iter), &this_submatrix_args);
          
          submatrix_args->insert(submatrix_args->end(),
                                 this_submatrix_args.begin(),
                                 this_submatrix_args.end());
        }
      }

      // 增加考虑 computation中的 indexes_multi的 submatrix ???
      size_t extra_size = 0;
      for (size_t i = 0; i < computation->indexes_multi.size(); i++)
        extra_size += computation->indexes_multi[i].size();
      submatrix_args->reserve(submatrix_args->size() + extra_size);

      // 对 computation->indexes_multi(这是保存的indexes, 是矩阵中的rows)
      // 将这些rows 所属的 submatrix 加入到submatrix_args.
      for (size_t i = 0; i < computation->indexes_multi.size(); i++) {
        std::vector<std::pair<int32, int32> > &indexes_multi = computation->indexes_multi[i];
        std::vector<std::pair<int32, int32> >::iterator
            iter = indexes_multi.begin(),
            end = indexes_multi.end();
        
        for (; iter != end; ++iter)
          if (iter->first != -1)
            submatrix_args->push_back(&(iter->first));
      }
    }



    
    std::vector<int32*>::iterator
        iter = submatrix_args.begin(),
        end = submatrix_args.end();


    int32 cur_submatrix_index = -1;
    // 对每个 submatrix-index
    for (; iter != end; ++iter) {
      int32 submatrix_index = **iter;
      
      if (submatrix_index > 0 && submatrix_index != cur_submatrix_index) {
        // 设置该submatrix 是被使用的.
        cur_submatrix_index = submatrix_index;
        submatrix_is_used_[submatrix_index] = true;
      }
    }
  }

  
  // 计算所有必要的Matrix
  ComputeMatrixIsUsed();
  void ComputationRenumberer::ComputeMatrixIsUsed() {
    matrix_is_used_.clear();
    // 对每个matrix 都分配是否使用标记
    matrix_is_used_.resize(computation_->matrices.size(), false);
    matrix_is_used_[0] = true;
    // 我们也需要 当matrix直接通过submatrix引用时 将这种情况纳入考虑
    // 这也是我门最常见的情况.
    int32 num_submatrices = computation_->submatrices.size();
    for (int32 s = 1; s < num_submatrices; s++) {
      int32 matrix_index = computation_->submatrices[s].matrix_index;
      if (submatrix_is_used_[s])
        matrix_is_used_[matrix_index] = true;
    }
  }

  // 通过映射 submatrix-count
  // 将每个submatrix 是否是个重复,
  // 将重复的 submatrix 都标记为不必要的, 
  SetUpMappings();
  void ComputationRenumberer::SetUpMappings() {
    // 根据每个matrix是否使用标记, 对他们重新编号
    num_matrices_new_ = CreateRenumbering(matrix_is_used_, &old_to_new_matrix_);
    // 映射 SubMatrixInfo -> SubMatrixHasher???
    unordered_map<NnetComputation::SubMatrixInfo, int32, SubMatrixHasher> submat_map;

    // cur_index 新编号的起始
    // num_submatrices_orig 原本的 submatrices总数
    int32 cur_index = 1, num_submatrices_orig = computation_->submatrices.size();

    

    // 每个submatrix 是否是被用到的.(true, false)
    // 当一个submatrix是前面一个submatrix的重复, 那么将该submatrix设置为false, 表示不用再保留了.
    submatrix_is_kept_ = submatrix_is_used_;
    
    // old_to_new_submatrix_ map 会删除重复
    // 当一个submatrix不再需要时, 设置标记为-1
    old_to_new_submatrix_.resize(num_submatrices_orig, -1);
    old_to_new_submatrix_[0] = 0;
    // 对每个submatrix
    // 1 如果已经被设置 要被用到
    // 判断submat_map 是否有对这个submatrix的重复
    // 如果有, 将该submatrix的old_to_new编号设置为 map中保存的编号
    // 将submatrix_is_kept_[s] 设置为false, 认为并不是
    for (int32 s = 1; s < num_submatrices_orig; s++) {
      if (submatrix_is_used_[s]) {
        const NnetComputation::SubMatrixInfo &info = computation_->submatrices[s];
        if (submat_map.count(info) > 0) {  // a duplicate...
          old_to_new_submatrix_[s] = submat_map[info];
          // 设置本submatrix 不在被用到.???
          submatrix_is_kept_[s] = false;
        } else {
          old_to_new_submatrix_[s] = (submat_map[info] = cur_index++);
        }
      }
    }

    // 当前 必要submatrix的重新编号后 总数.
    num_submatrices_new_ = cur_index;
  }


  // 根据submatrices 的新编号 old_to_new_submtrix_
  // 更新所有命令中的submatrices-index 为新编号
  // 清理重复的submatrices.
  RenumberSubmatrices();
  void ComputationRenumberer::RenumberSubmatrices() {

    // 所有submatrix的 submatrix-index
    std::vector<int32*> submatrix_args;
    IdentifySubmatrixArgsInComputation(computation_, &submatrix_args);

    // 每个submatrix-index
    // 获取新的编号 old_to_new_submatrix_[s]
    // 将原本 submatrix_args中保存的submatrix-index 保存为 新编号.
    // (这时候怎么映射找到对应的submatrix)
    std::vector<int32*>::iterator iter = submatrix_args.begin(),
        end = submatrix_args.end();
    for (; iter != end; ++iter) {
      if (**iter > 0) {
        int32 new_submatrix_index = old_to_new_submatrix_[**iter];
        // old_to_new_submatrix_[s] for s > 0 is only <= 0 (actually, -1) for
        // submatrices that are never accessed, and these should never appear
        // in this list.
        KALDI_ASSERT(new_submatrix_index > 0);
        **iter = new_submatrix_index;
      }
    }

    // 所有原本的SubMatrixInfo
    // 将确定需要保留的submatrix加入到 new_submatrices,
    // 最终computation 将原本的submatrix swap 保留这个new_submatrices.
    std::vector<NnetComputation::SubMatrixInfo> new_submatrices;
    int32 num_submatrices_old = computation_->submatrices.size();
    new_submatrices.reserve(num_submatrices_old);
    for (int32 s = 0; s < num_submatrices_old; s++)
      if (submatrix_is_kept_[s])
        new_submatrices.push_back(computation_->submatrices[s]);
    // 重新编号, 去掉重复后的submatrices
    computation_->submatrices.swap(new_submatrices);
    // We'll map the matrix indexes inside computation_->submatrices
    // when we call RenumberMatrices().
  }



  // 根据新编号的Matrices - old_to_new_matrix 
  // 将所有的submatrices的matrix-index域更新为 new-index
  // 将不必要的matrices 都swap清理掉.
  RenumberMatrices();
  void ComputationRenumberer::RenumberMatrices() {
    std::vector<int32*> matrix_args;
    // 所有submatrix(经过上面处理已经是去除重复等之后的)对应的matrix
    // 将matrix-index映射为重新编号的 new_matrix_index
    // 回写到对应submatrix,让submatrix映射到对应的new_matrix_index, 这样就不用原本的matrix编号了.
    int32 num_submatrices = computation_->submatrices.size();
    for (int32 s = 1; s < num_submatrices; s++) {
      int32 *matrix_index = &(computation_->submatrices[s].matrix_index);
      // old_to_new_matrix_[s] for s > 0 is only <= 0 (actually, -1) for
      // submatrices that are never accessed, and these should never appear
      // in this list.  (presumably because we renumber the submatrices first).
      int32 new_matrix_index = old_to_new_matrix_[*matrix_index];
      KALDI_ASSERT(new_matrix_index > 0);
      *matrix_index = new_matrix_index;
    }

    // 按照上面结算得到的 matrix_is_used_ 得到必要的matrix.
    // 去掉computation中matrix的重复, 保留必要matrix
    std::vector<NnetComputation::MatrixInfo> new_matrices;
    int32 num_matrices_old = computation_->matrices.size();
    new_matrices.reserve(num_matrices_old);
    for (int32 m = 0; m < num_matrices_old; m++)
      if (matrix_is_used_[m])
        new_matrices.push_back(computation_->matrices[m]);
    computation_->matrices.swap(new_matrices);

    // 同上, 保留必要的matrix_debug_info.
    std::vector<NnetComputation::MatrixDebugInfo> new_debug_info;
    int32 debug_info_size = computation_->matrix_debug_info.size();
    // 断言 debug_info_size == num_matrices_old
    KALDI_ASSERT(debug_info_size == 0 || debug_info_size == num_matrices_old);
    new_debug_info.reserve(debug_info_size);
    for (int32 m = 0; m < debug_info_size; m++) {
      if (matrix_is_used_[m]) {
        new_debug_info.push_back(NnetComputation::MatrixDebugInfo());
        new_debug_info.back().Swap(&(computation_->matrix_debug_info[m]));
      }
    }
    computation_->matrix_debug_info.swap(new_debug_info);
  }

  
  RemoveIndexesMultiDuplicates();
  // 上面是删除了indexes_multi中不必要的unused
  // 这里删除重复的.
  void ComputationRenumberer::RemoveIndexesMultiDuplicates() {
    int32 cur_index = 0,
        old_indexes_multi_size = computation_->indexes_multi.size();
    if (old_indexes_multi_size == 0)
      return;
    // 相同的步骤,
    // 创建 old_to_new的映射,每个indexes_multi 都映射到new编号
    // 这里使用的思想是 我们可以只依靠vector的size, 做很多比较工作,
    // 甚至能够避免访问数据内容.
    std::vector<int32> indexes_multi_old_to_new(old_indexes_multi_size);
    typedef std::vector<std::pair<int32,int32> > PairVectorType;
    typedef std::map<const PairVectorType*, int32,
                     PointerCompare<std::pair<int32,int32> > > MapType;
    
    MapType indexes_multi_map;
    for (int32 i = 0; i < computation_->indexes_multi.size(); i++) {
      // indexes_multi_map.insert() 返回一个iter, 以及一个bool标记表示是否是新加入的.
      // insert(pair<indexes_multi, cur_index>)
      std::pair<MapType::iterator, bool> p =
          indexes_multi_map.insert(
              std::pair<const PairVectorType*, int32>(&(computation_->indexes_multi[i]), cur_index)
                                   );
      // 新加入, 则指定该indexes_multi的new编号索引为 cur_index.
      if (p.second) {  // was inserted-- was not there already.
        indexes_multi_old_to_new[i] = cur_index++;
      } else {
        // 否则直接设置新索引为 相同的indexes_multi的编号.
        int32 index_from_map = p.first->second;
        indexes_multi_old_to_new[i] = index_from_map;
      }
    }
    // 如果安排的 new编号总数 == old编号总数
    if (cur_index == old_indexes_multi_size)
      return;  // An optimization.  No duplicates were found.

    // 将每个 indexes_multi  swap到 new_indexes_multi[new_index],
    // 将indexes_multi 放入到新数组中 重新安排位置
    std::vector<PairVectorType> new_indexes_multi(cur_index);
    for (int32 i = 0; i < old_indexes_multi_size; i++) {
      int32 new_index = indexes_multi_old_to_new[i];
      computation_->indexes_multi[i].swap(new_indexes_multi[new_index]);
    }
    computation_->indexes_multi.swap(new_indexes_multi);

    // 将所有的 kMulti类型 commands的 indexes_multi参数数据 加入到indexes_multi_args -- 必要的indexes_multi-index
    std::vector<int32*> indexes_multi_args;
    IdentifyIndexesMultiArgs(&(computation_->commands), &indexes_multi_args);
    // 将所有必要的 indexes_multi-index 安排为新编号 new-index.
    // 完成将 indexes_multi数据的清理(去掉重复)
    std::vector<int32*>::const_iterator iter = indexes_multi_args.begin(),
        end = indexes_multi_args.end();
    for (; iter != end; ++iter)
      **iter = indexes_multi_old_to_new[**iter];
  }

  // 重新编号 kRows命令的 indexes
  // 清理不必要的indexes.
  // 安排命令的所有必要indexes 的索引位置 为new-index
  RenumberIndexes();
  void ComputationRenumberer::RenumberIndexes() {
    // 所有的indexes.----- ?? 
    int32 old_num_indexes = computation_->indexes.size();
    if (old_num_indexes == 0)
      return;
    std::vector<int32*> indexes_args;
    // 所有kRows类命令的 indexes--某个矩阵的某个行 加入到indexes_args--- 必要的indexes 行.
    IdentifyIndexesArgs(&(computation_->commands), &indexes_args);

    // 所有indexes都标记为可见
    std::vector<bool> indexes_seen(old_num_indexes, false);
    std::vector<int32*>::const_iterator iter = indexes_args.begin(),
        end = indexes_args.end();
    for (; iter != end; ++iter)
      indexes_seen[**iter] = true;

    // 对indexes的重新编号
    std::vector<int32> old_to_new_index(old_num_indexes);
    typedef std::map<const std::vector<int32>*, int32, PointerCompare<int32> > MapType;
    MapType indexes_map;
    // 对每个index row行.
    // 如果不可将直接设置 安排新编号为-1
    // 否则通过insert的返回值, 设置index row行 的新编号.
    int32 cur_index = 0;
    for (int32 i = 0; i < old_num_indexes; i++) {
      if (!indexes_seen[i]) {
        old_to_new_index[i] = -1;
      } else {
        std::pair<MapType::iterator, bool> p =
            indexes_map.insert(std::pair<const std::vector<int32>*, int32>(
                &(computation_->indexes[i]), cur_index));
        if (p.second) {  // was inserted-- was not there already.
          old_to_new_index[i] = cur_index++;
        } else {
          int32 index_from_map = p.first->second;
          old_to_new_index[i] = index_from_map;
        }
      }
    }

    // 如果新编号总数 和 旧的总数相同, 说明不需要优化.
    if (cur_index == old_num_indexes)
      return;  // An optimization.  No changes to the numbering are made.

    // 将indexes的数据 swap保存到 new_indexes[new-index]中, 用新索引保存
    std::vector<std::vector<int32> > new_indexes(cur_index);
    for (int32 i = 0; i < old_num_indexes; i++) {
      int32 new_index = old_to_new_index[i];
      if (new_index != -1)
        computation_->indexes[i].swap(new_indexes[new_index]);
    }
    computation_->indexes.swap(new_indexes);


    
    // 重新编号 命令中必要的 indexes 的 编号位置 old-index => new-index
    // renumber the indexes inside the commmands.
    for (iter = indexes_args.begin(); iter != end; ++iter) {
      int32 old_index = **iter;
      KALDI_ASSERT(old_index >= 0 && old_index < old_num_indexes);
      int32 new_index = old_to_new_index[old_index];
      KALDI_ASSERT(new_index >= 0);
      **iter = new_index;
    }
  }

  // 流程和上面的相同, 不过是AddRowsRange类命令的 Indexes.
  RenumberIndexesRanges();

  // 重新编号 Memos.
  RenumberMemos();
  void ComputationRenumberer::RenumberMemos() {
    // 用memo-index索引, 映射到 propagate backprop命令, 需要memo-index的命令.
    // this is indexed by memo-index, and maps to the
    // (propagate, backprop) commands that use that memo-index, or
    // (-1, -1) if there are no such commands.
    
    std::vector<std::pair<int32, int32> > memo_to_commands;
    std::vector<int32> memo_indexes_used;
    std::pair<int32, int32> blank(-1, -1);
    int32 num_commands = computation_->commands.size();
    for (int32 c = 0; c < num_commands; c++) {
      NnetComputation::Command &command = computation_->commands[c];
      if (command.command_type == kPropagate) {
        int32 memo_index = command.arg5;
        if (memo_index > 0) {
          if (memo_to_commands.size() <= static_cast<size_t>(memo_index))
            memo_to_commands.resize(memo_index + 1, blank);
          KALDI_ASSERT(memo_to_commands[memo_index].first == -1);
          memo_to_commands[memo_index].first = c;
          memo_indexes_used.push_back(memo_index);
        }
      } else if (command.command_type == kBackprop) {
        int32 memo_index = command.arg7;
        if (memo_index > 0) {
          if (memo_to_commands.size() <= static_cast<size_t>(memo_index))
            memo_to_commands.resize(memo_index + 1, blank);
          KALDI_ASSERT(memo_to_commands[memo_index].first > 0 &&
                       memo_to_commands[memo_index].second == -1);
          memo_to_commands[memo_index].second = c;
        }
      }
    }
    int32 new_memo_index = 1;
    for (std::vector<int32>::iterator iter = memo_indexes_used.begin();
         iter != memo_indexes_used.end(); ++iter) {
      int32 memo_index = *iter;
      int32 propagate_command = memo_to_commands[memo_index].first,
          backprop_command = memo_to_commands[memo_index].second;
      KALDI_ASSERT(backprop_command > 0 &&
                   "Propagate generates memo but backprop doesn't use it.");
      computation_->commands[propagate_command].arg5 = new_memo_index;
      computation_->commands[backprop_command].arg7 = new_memo_index;
      new_memo_index++;
    }
  }

}

// 所有命令, 如果命令为kNoOperation
void RemoveNoOps(NnetComputation *computation) {
  
  std::vector<NnetComputation::Command>::iterator
      input_iter = computation->commands.begin(),
      input_end = computation->commands.end(),
      output_iter = computation->commands.begin();
  // 遍历 commands, 保留 非kNoOpteraction 命令.
  // 最终 output_iter之前保留了 必要命令.
  for (; input_iter != input_end; ++input_iter) {
    if (input_iter->command_type != kNoOperation) {
      *output_iter = *input_iter;
      ++output_iter;
    }
  }
  computation->commands.resize(output_iter - computation->commands.begin());
}
