
// 削减Row Ops
bool SnipRowOps(NnetComputation *computation) {
  bool ans = false;
  int32 num_commands = computation->commands.size();
  for (int32 command_index = 0; command_index < num_commands;
       command_index++) {
    // non-const because we'll be changing it.
    NnetComputation::Command &c = computation->commands[command_index];

    // 我们不能削减 kCopyRows kCopyRowsMulti 等命令
    // 因为-1 不是一个干净的 no-op
    // 他们具有设置目标值为0的含义, 所以我们不能将他们处理掉

    switch (c.command_type) {
      case kAddRows: {
        // 重新安排命令中的 arg1 arg3,
        // arg1 是一个submatrix
        // arg3 是submatrix的多个行row - indexes
        // 去掉arg3中开头结尾的 负值, 重新得到一个 vec indexes,
        // 并重新生成一个submatrix
        if (SnipSingleRowOp(computation, command_index))
          ans = true;
        break;
      }
      case kAddRowsMulti: case kAddToRowsMulti:
      case kCopyToRowsMulti: {
        if (SnipMultiRowOp(computation, command_index))
          ans = true;
        break;
      }
      case kAddRowRanges: {
        if (SnipRangesRowOp(computation, command_index))
          ans = true;
        break;
      }
      default:
        break;
    }
  }
  return ans;
}


// 这个函数在遇到 kAddRows命令时被调用, 它会修改这样的命令, 当indexes行
// 使他们操作在一个更小的submatrix上,
// 返回true 如果它做了一些修改.
static bool SnipSingleRowOp(NnetComputation *computation,
                            int32 command_index) {
  NnetComputation::Command &c = computation->commands[command_index];
  // 断言 arg3 必然在indexes编号范围内.
  KALDI_ASSERT(static_cast<size_t>(c.arg3) < computation->indexes.size());

  // 返回c.arg3指定的indexes 多行.
  const std::vector<int32> &indexes = computation->indexes[c.arg3];
  
  int32 num_leading_negatives, num_trailing_negatives;
  FindNumLeadingAndTrailingNegatives(indexes,
                                    &num_leading_negatives,
                                    &num_trailing_negatives);
  
  // 前面负值总数=0, 后面负值总数=0
  if (num_leading_negatives == 0 && num_trailing_negatives == 0)
    return false;

  // 去掉前后负值
  int32 new_num_rows = static_cast<int32>(indexes.size()) - num_leading_negatives - num_trailing_negatives;
  KALDI_ASSERT(new_num_rows > 0);
  // 保存去掉负值之后的indexes
  std::vector<int32> new_indexes(indexes.begin() + num_leading_negatives,
                                 indexes.begin() + num_leading_negatives +
                                 new_num_rows);
  // 重新安排命令的 arg3
  c.arg3 = computation->indexes.size();
  computation->indexes.push_back(std::vector<int32>());
  computation->indexes.back().swap(new_indexes);
  // 重新构建命令的 arg1 -- indexes 所属的submatrix.
  c.arg1 = computation->NewSubMatrix(c.arg1,
                                     num_leading_negatives, new_num_rows,
                                     0, -1);
  return true;  // made a change.
}

/*
  This function, used in SnipSingleRowOp(),
  找到 leading trailing 的负值编号数量 在一个int vector中 例如如果vec是
  [ -1 -1 2 3 -1 4 5 -1 ]
  这时 从头开始负值总数 =2 -1 -1
  从尾开始 负值总数=1 -1.
  如果所有的数字都是负值, 或者 vec为空, 会出错.

*/
static void FindNumLeadingAndTrailingNegatives(const std::vector<int32> &vec,
                                               int32 *num_leading_negatives,
                                               int32 *num_trailing_negatives) {
  KALDI_ASSERT(!vec.empty());
  
  const int32 *begin = &(vec[0]), *ptr = begin, *end = ptr + vec.size();
  // 找到第一个 非负值
  while (ptr != end && *ptr < 0)
    ptr++;

  // 考虑错误信息, 我们将所有负值设置为-1, 根据我们调用的方式, 这只会影响我们如何描述错误.??
  KALDI_ASSERT(ptr != end && "Vector consists entirely of -1's.");

  // 正向  开头负值总数
  *num_leading_negatives = ptr - begin;

  
  const int32 *ptr2 = end - 1;
  // 反向查找第一个非负值, 没有检查到开头, 是因为我们确定 vec中至少包含一个非负值.
  while (*ptr2 < 0)
    ptr2--;
  
  KALDI_ASSERT(ptr2 >= begin);  // or would be code error.
  // 反向 结尾负值总数
  *num_trailing_negatives = end - 1 - ptr2;
}







// 使用Matrix操作 替换row操作
bool ReplaceRowWithMatrixOps(NnetComputation *computation) {
  bool ans = false;
  int32 num_commands = computation->commands.size(),
      num_indexes = computation->indexes.size();


  
  for (int32 command_index = 0; command_index < num_commands;
       command_index++) {
    // non-const because we'll be changing it.
    NnetComputation::Command &c = computation->commands[command_index];

    int32 first_nonnegative_pos,
        first_nonnegative_value,
        num_nonnegative_indexes;
    switch (c.command_type) {
      case kCopyRows: case kAddRows: {
        int32 indexes_index = c.arg3;
        KALDI_ASSERT(indexes_index < num_indexes);
        const std::vector<int32> &indexes = computation->indexes[indexes_index];

        // 判断indexes是否具有特殊结构
        if (IndexesHaveSpecialStructure(indexes,
                                        &first_nonnegative_pos,
                                        &first_nonnegative_value,
                                        &num_nonnegative_indexes)) {

          // 如果具有特殊结构, 将命令修改为 Matrix操作
          // 并根据indexes的特殊结构修改构建 submatrix参数.
          ans = true;
          c.arg1 = computation->NewSubMatrix(c.arg1, first_nonnegative_pos,
                                             num_nonnegative_indexes,
                                             0, -1);
          c.arg2 = computation->NewSubMatrix(c.arg2, first_nonnegative_value,
                                             num_nonnegative_indexes,
                                             0, -1);

          
          // 替换kCopyRows kAddRows操作为 kMatrixCopy kMatrixAdd命令.
          c.command_type = (c.command_type == kCopyRows ? kMatrixCopy :  kMatrixAdd);
        }
        break;
      }
      default:
        break;
    }
  }
  return ans;
}


/*

  这个函数 检测出 当vecotor indexes 具有一个特殊结构时.
  一个0 或多个-1 然后是一个 非负的多个数 然后是 一个0或多个-1.的结构
  This helper function, used in ReplaceRowWithMatrixOps, detects
  when the vector 'indexes' has a 'special structure'.  The special structure
  is:
    zero or more -1's, then
    a consecutive nonempty sequence of nonnegative numbers, e.g. 6 7 8 9 10, then
    zero or more -1's.

  这个函数假设 所有index的负值元素都是-1, 如果有<-1的元素 就认为是错误,
  但是这个函数并不会检查这样的情况, indexes 要求是一个非空的vector

  如果indexes 具有特殊结构 函数就返回true, 并设置为如下值
  用如下的例子解释 'indexes = [ -1, -1, 5 6 7 8, -1 ]'.
     - '*first_nonnegative_pos' is set to the number of initial -1's (and also
       the location of the first nonnegative element): 2 in this case.
     - '*first_nonnegative_value' is set to the value of the first nonnegative
       element (5 in this case)
     - '*num_nonnegative_values' is set to the number of nonnegative values in
       the sequence (4 in this case).

  如果indexes 没有这样的特殊结构, 返回false, '*first_nonnegative_pos',
  '*first_nonnegative_value' and '*num_nonnegative_indexes' 变量的值都是未定义, 不关心.
*/
static bool IndexesHaveSpecialStructure(const std::vector<int32> &indexes,
                                        int32 *first_nonnegative_pos,
                                        int32 *first_nonnegative_value,
                                        int32 *num_nonnegative_indexes) {
  KALDI_ASSERT(!indexes.empty());
  const int32 *indexes_ptr = &(indexes[0]);
  size_t pos = 0, size = indexes.size();

  // Find the first nonnegative element of 'indexes'.
  for (; pos < size; ++pos)
    if (indexes_ptr[pos] >= 0)
      break;
  if (pos == size)
    return false;  // all -1's... should not happen, but not our problem.
  *first_nonnegative_pos = static_cast<int32>(pos);
  int32 n = indexes_ptr[pos];
  *first_nonnegative_value = n;
  // Find the first element after '*first_nonnegative_index' that isn't
  // consecutive.
  for (; pos < size; ++pos,++n)
    if (indexes_ptr[pos] != n)
      break;

  *num_nonnegative_indexes = n - *first_nonnegative_value;

  // Check that the remaining values are all <0 (assumed equal to -1, but
  // checking <0 may be faster as just one instruction).
  for (; pos < size; ++pos)
    if (indexes_ptr[pos] >= 0)
      return false;  // does not have the special structure.

  return true;
}









// ====================== part 2 ============================
// 删除 alpha=0.0的kSetConst操作
void RemoveUnnecessaryZeroing(const Nnet &nnet,
                              NnetComputation *computation) {

  // 构建分析器
  Analyzer a;
  // 分析每个 matrix submatrix variables的访问(r w rw)流程.
  a.Init(nnet, *computation);

  // 现在计算出哪些matrix
  // 他们的pieces-小块variables (例如属于那个matrix的所有variables)
  // 跟着第一个指令 和初始化0操作分开的指令 写入??.
  // OK, now we'll work out which matrices have all their pieces written to as the first instruction
  // apart from the initial zeroing.
  // 这些matrices 可以具有初始化0操作 通过用一个 sizing操作, 保留数据为未定义

  // 分析器分析出所有所有matrix的访问(r w rw)流程
  int32 num_matrices = a.matrix_accesses.size();
  // 每个matirx
  for (int32 matrix_index = 0; matrix_index < num_matrices; matrix_index++) {
    const MatrixAccesses &accesses = a.matrix_accesses[matrix_index];
    if (accesses.accesses.empty())
      continue;
    // 该matrix的第一个访问命令如果是 不是kSetConst(alpha=0), 则continue下一个matrix.
    // 只处理kSetConst,且apha=0.0的命令.
    int32 zeroing_command_index = accesses.accesses[0].command_index;
    NnetComputation::Command *command = &(computation->commands[zeroing_command_index]);
    if (!(command->command_type == kSetConst &&
          command->alpha == 0.0)) {
      continue;  // First command is not a zeroing command
    }

    // 如果是一个 kSetConst alpha=0.0 命令. 我们需要计算出matrix-index的所有variables小块
    std::vector<int32> variables_for_matrix;
    a.variables.AppendVariablesForMatrix(matrix_index, &variables_for_matrix);
    void ComputationVariables::AppendVariablesForMatrix(
        int32 matrix_index,
        std::vector<int32> *variable_indexes) const {

      // 获取属于matrix-index 的所有 variables.
      int32 start = matrix_to_variable_index_[matrix_index],
          end = matrix_to_variable_index_[matrix_index + 1];
      // 取出所有属于matrix-index 的 variables小块
      variable_indexes->reserve(variable_indexes->size() + end - start);
      for (int32 variable_index = start; variable_index < end; variable_index++)
        variable_indexes->push_back(variable_index);
    }


    // 如果经过如下操作,保持为true, 则不需要初始化zero操作.
    bool all_variables_ok = true;
    // matrix的所有variables
    // 该variables的访问(r w rw)流程
    for (size_t i = 0; i < variables_for_matrix.size(); i++) {
      int32 variable_index = variables_for_matrix[i];
      const std::vector<Access> &v_accesses =
          a.variable_accesses[variable_index];
      // 必须初始化0,之后直接接一个write操作, 这样初始化0操作,才可以省略,
      // 如果是读操作, 那么0是有用的不能被省略.
      if (v_accesses.size() > 1 && v_accesses[1].access_type != kWriteAccess) {
        all_variables_ok = false;  // first access after zeroing was not a write
        break;
      }
      
      if (v_accesses.size() == 1 && accesses.is_output) {
        // the only command that touches this variable is the allocation, and it
        // is an output variable.  (this is unusual, but can happen e.g. if it's
        // a derivative, but due to min_deriv_time and max_deriv_time it ends up
        // always being zero.
        all_variables_ok = false;
        break;
      }
    }

    // 直接将该kSetConst 操作设置为NoOperation.
    if (all_variables_ok) {
      // Here is where the change actually happens.
      // Remove the zeroing command.
      command->command_type = kNoOperation;
    }
  }
}














// ====================== part 3 ============================
// 移除 尽可能多的 resize zero matrix的命令
// 但是需要保持 input output命令, 因为这样的命令如果我们删除了 会创建令人头痛的东西.???
void MoveSizingCommands(const Nnet &nnet, NnetComputation *computation) {

  // 实际也是实现了 Analyzer的功能, 分析每个matrix variables的访问流程
  ComputationVariables variables;
  variables.Init(*computation);
  
  std::vector<CommandAttributes> attributes;
  ComputeCommandAttributes(nnet, *computation, variables, &attributes);
  std::vector<std::vector<Access> > variable_accesses;
  ComputeVariableAccesses(variables, attributes, &variable_accesses);
  std::vector<MatrixAccesses> matrix_accesses;
  ComputeMatrixAccesses(nnet, *computation, variables, attributes,
                        &matrix_accesses);


  // 我们对command重新编号的方法, 我们会首先用pair<command-index, pointer-to-command> 设置vector,
  // 然后我们会修改command-indexes 为了按照我们希望的顺序重新编号, 然后排序
  // 使用command-index*3的原因是这样我们可以编号命令 按照 
  // The reason for the * 3 is so that we can number commands "just-after"
  // existing indexes (by adding 1) and "just-before" (by subtracting 1).
  int32 num_commands = computation->commands.size(),
      num_matrices = matrix_accesses.size();

  // Matrix allocate命令希望 后面接一个zero该matrix的命令
  // 我们希望将两个命令看做为一个 重新排序的单元,
  // 如果命令c 是这样一个pair的第一个元素 is_command_pair[c]=true
  std::vector<bool> is_command_pair(num_commands, false);
  for (int32 c = 0; c + 1 < num_commands; c++) {
    if (computation->commands[c].command_type == kAllocMatrix &&
        computation->commands[c+1].command_type == kSetConst &&
        computation->commands[c].arg1 == computation->commands[c+1].arg1 &&
        computation->commands[c+1].alpha == 0.0) {
      is_command_pair[c] = true;
    }
  }

  // 每个命令 一个 command_reordering.
  // command_reordering 包含命令的 new-number, old-number
  // new-number 是乘以3之后的值
  std::vector<std::pair<int32,int32> >   command_reordering(num_commands);

  // 为每个命令都安排 index*3 给命令一个排序的index索引.
  // 现在我们引入pairs的第二个元素(即 allocate命令之后的 zero命令)
  // command_reordering[c] - pair中保存的 是 first: 排序后命令index, second: 原始命令index.
  for (int32 c = 0; c < num_commands; c++) {
    command_reordering[c].first = c * 3;
    command_reordering[c].second = c;
  }

  // 对每个matrix
  for (int32 m = 1; m < num_matrices; m++) {
    
    const MatrixAccesses &ma = matrix_accesses[m];
    // 如下的if-block 涉及 allocate命令的重排序(隐性包含了 zero)
    if (ma.allocate_command != -1 && computation->commands[ma.allocate_command].command_type == kAllocMatrix) {
      // first_access_command 回事 first access的索引 除了zero命令 隐含跟着初始化命令.
      // first_access_command will be index of first access, except for the
      // zeroing command that immediately follows the initialization command.
      
      int32 first_access_command = -1;
      // 判断第一个访问命令 是否是个kSetConst, 能够成为一个 pair, 因此能够进行重新排序 allocate?
      if (!ma.accesses.empty()) {
        first_access_command = ma.accesses[0].command_index;

        // allocate命令之后紧接着的命令 是一个访问命令(r w 等操作)
        // 并且 是个command_pair(说明第二个命令是个 kSetConst alpha=0.0)
        if (first_access_command == ma.allocate_command + 1 && is_command_pair[ma.allocate_command]) {
          if (ma.accesses.size() > 1)
            first_access_command = ma.accesses[1].command_index;
          else
            first_access_command = -1;
        }
      }

      // =====================================================
      // 将allocate命令 安排到 去掉第一个kSetConst命令之后第一个命令之前.
      // ** 这样才能保证命令顺序不会出错, 对其他不相关的命令 相对顺序随意即可 不用关心 **
      if (first_access_command != -1) {
        KALDI_ASSERT(first_access_command > ma.allocate_command);
        // 重新安排allocate命令的 command_reordering.first = first_access_command*3-1.
        command_reordering[ma.allocate_command].first = first_access_command * 3 - 1;
      }
    }

    // 如下的代码 设计重新排序 deallocate命令.
    // =====================================================    
    // 将deallocate命令 安排到 最后该matrix 最后访问命令 之后.
    // ** 保证命令不会出错 ** 
    if (ma.deallocate_command != -1 && !ma.accesses.empty() &&
        computation->commands[ma.deallocate_command].command_type ==
        kDeallocMatrix) {
      int32 last_access_command = ma.accesses.back().command_index;
      // move the deallocation command to just after the last access.
      command_reordering[ma.deallocate_command].first =
          last_access_command * 3 + 1;
    }
  }



  
  std::sort(command_reordering.begin(), command_reordering.end());

  // 保存排序后命令 list
  std::vector<NnetComputation::Command> reordered_commands;
  reordered_commands.reserve(num_commands);

  // 每个command
  for (int32 c = 0; c < num_commands; c++) {
    // 命令原始index
    int32 old_index = command_reordering[c].second;
    // 获得命令
    NnetComputation::Command &old_command = computation->commands[old_index];
    
    // the following assert is because this optimization is not allowed
    // after looped optimization.
    KALDI_ASSERT(old_command.command_type != kGotoLabel);

    // 如果旧的命令 是一个在allocate之后的zero命令 忽略它.
    // 因为它将会被重新排序到 其配对的allocate命令之后,
    // 我们在处理 那个allocate命令时候处理这个zero命令.
    if (old_index > 0 && is_command_pair[old_index - 1]) {
      // If the old command-index was a zeroing command that follows
      // an allocation command, ignore it; it will be reordered to
      // right after wherever the allocation command went, and we'll
      // deal with it when we deal with the first element of the pair.
      continue;
    }
    // 按顺序将原本的命令加入到重排顺序之后.
    else {
      reordered_commands.push_back(computation->commands[old_index]);
      // 如果是个 pair 的allocate 命令(pair <allocate + zero>), 直接将zero命令安排在之后.
      if (is_command_pair[old_index]) {
        // if this command is the first member of an (allocation, zeroing)
        // pair then we need to deal with the zeroing command as well.
        reordered_commands.push_back(computation->commands[old_index + 1]);
      }
    }
  }

  // 保存排序后命令.
  computation->commands = reordered_commands;
}



