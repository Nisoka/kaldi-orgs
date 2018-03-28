
void RemoveUnnecessaryAllocation(const Nnet &nnet,
                                 NnetComputation *computation) {
  // 为每个matrix的size 和 stride-type 描述为一个 pair<matrix-rows, matrix-cols*(stride-type == kDefalutStride ? 1:-1)
  // 我们 为该matrix-size计算
  // 一个 deallocate命令的 indexes索引列表list
  // 一个 allocate命令的索引列表
  // 对每个不同的matrix size, 调用ComputeCommandPairs 在那两个list上
  // 获得一个 deallocation, allocation的命令索引的pair 我们将他们优化为一个单一的命令.

  // 这个map 从一个(num-rows, num-cols)(实际matrix-size) 映射到两个list. -- deallocate命令列表+ allocate命令列表
  // 顺序可能反向的, 但是这就是我们想要的pairs的顺序???

  typedef unordered_map<std::pair<int32,int32>,  std::pair<std::vector<int32>,std::vector<int32> >,  PairHasher<int32> > MapType;
  //                    pair-size<num-rows, num-cols>    pair<deallocate command list, allocate command list>
  
  MapType pair_map;
  int32 num_commands = computation->commands.size();
  // 对每个命令
  for (int32 command_index = 0; command_index < num_commands; command_index++) {
    // 判断是 allocate deallocate 命令
    NnetComputation::Command &command = computation->commands[command_index];
    if (command.command_type == kAllocMatrix ||
        command.command_type == kDeallocMatrix) {
      // 获得对应的 subMatrix 和 matrix, 以及matrix的 rows cols cols-mod
      int32 s = command.arg1, m = computation->submatrices[s].matrix_index,
          num_rows = computation->matrices[m].num_rows,
          num_cols = computation->matrices[m].num_cols,
          num_cols_mod = num_cols * (computation->matrices[m].stride_type == kDefaultStride ? 1 : -1);

      // 构建一个描述size的 pair
      std::pair<int32,int32> p(num_rows, num_cols_mod);

      // 引用该size的allocate deallocate命令列表, 将对应命令加入到列表中.
      std::pair<std::vector<int32>,std::vector<int32> > &lists = pair_map[p];
      if (command.command_type == kDeallocMatrix)
        lists.first.push_back(command_index);
      else
        lists.second.push_back(command_index);
    }
  }


  // 所有allocate deallocate命令都加入到对应的 size的列表中了
  // 如下对每个matrix size进行遍历计算commands_pairs.
  MapType::const_iterator iter = pair_map.begin(), end = pair_map.end();
  std::vector<std::pair<int32,int32> > command_pairs;
  for (; iter != end; ++iter)
    ComputeCommandPairs(iter->second, &command_pairs);

  // 对所有size 的 所有命令对
  for (size_t i = 0; i < command_pairs.size(); i++) {
    
    int32
        dealloc_index = command_pairs[i].first,
        alloc_index = command_pairs[i].second;
    
    NnetComputation::Command
        &dealloc_command = computation->commands[dealloc_index],
        &alloc_command = computation->commands[alloc_index];
    
    KALDI_ASSERT(dealloc_command.command_type ==
                 kDeallocMatrix);
    KALDI_ASSERT(alloc_command.command_type ==
                 kAllocMatrix);

    // 将相同size的成对的 deallocate allocate
    // 转化为一个 kNoOperation + kSwapMatrix命令 省的 deallocate 然后还 allocate.
    dealloc_command.command_type =  kNoOperation;
    alloc_command.arg2 = dealloc_command.arg1;
    alloc_command.command_type = kSwapMatrix;
  }

  // 移除 kNoOperation命令
  RemoveNoOps(computation);
  
  FixGotoLabel(computation);
}




/*
  这个函数 input输入是两个排序的unique的list 保存的是 of (deallocation-commands, allocation-commands)
  
  e.g. (d1, d2, d3.. ), (a1, a2, a3..);

  输出是一个 累计列表  pairs (d, a).
  每个输出pair 必须满足 d < a , 并且input lists中的元素 都不会出现在output pairs两次
  (although it's OK for input a and d values not to appear in any output pairs).

  // 实现的目标是 输出尽可能多的pairs, 并且尽可能的两个pair接近,节省空间
*/
// lists 是一个matrix-size的 所有allocate deallocate的命令列表
// pairs 是输出目标
static void ComputeCommandPairs(
    const std::pair<std::vector<int32>, std::vector<int32> > &lists,
    std::vector<std::pair<int32,int32> > *pairs) {

  // 获得deallocate命令列表
  std::vector<int32> d_list = lists.first;

  // 一个集合, 将allocate命令列表转化到 集合形式(sorted and unique)
  std::set<int32> a_set;
  CopyVectorToSet(lists.second, &a_set);


  // from the latest to the earliest deallocation command...
  // 反向索引 从后向前
  std::vector<int32>::reverse_iterator
      iter = d_list.rbegin(),
      end = d_list.rend();

  for (; iter != end; ++iter) {
    int32 d = *iter;
    std::set<int32>::iterator a_iter = a_set.upper_bound(d);
    // a_iter 是一个迭代器, 获得第一个 大于 d的元素 a.
    // (最晚能够插入d的位置, 实际得到a就是 刚刚比d大的最小id的allocate命令)

    // 这样的 deallocate命令 没有配对的 allocate命令. 不构成pair.
    if (a_iter == a_set.end())
      continue;  // we will output no pair for this d.
    
    int32 a = *a_iter;
    KALDI_PARANOID_ASSERT(a > d);  // or code error
    a_set.erase(a_iter);  // remove this a from 'a_set' so it doesn't get used twice

    
    // 构成deallocate - allocate对??
    // 这样可以重用 一个相同大小的matrix, 而不用deallocate 然后allocate了.
    pairs->push_back(std::pair<int32,int32>(d, a));
  }
}









// ================== part2 =============
// 重新排序Commands
// 将 kAcceptInput 安排在每命令段的最开始
//  middle
// 将 kProvideOutput 安排在每命令段的最末尾
void ConsolidateIoOperations(const Nnet &nnet,
                             NnetComputation *computation) {

  // 按照segments(start-index, end-index) 划分计算.
  // 实际已经可以通过kNoOpearationMarker得到划分
  // 内部保存 <seg-index, kNoOperationMarker-cmd-index>
  std::vector<std::pair<int32, int32> > segments;
  SplitComputationIntoSegments(*computation, &segments);

  
  int32 num_commands = computation->commands.size();
  
  std::vector<NnetComputation::Command> reordered_commands(num_commands);
  
  // put kNoOperationMarker between all segments in the reordered commands.
  for (size_t s = 0; s + 1 < segments.size(); s++)
    reordered_commands[segments[s].second].command_type = kNoOperationMarker;

  
  // for each segment we'll divide the commands up into those that must appear
  // at the left of the segment (kAcceptInput for inputs and output-derivs), those
  // that must appear in the middle (most commands), those that must appear
  // on the right (kProvideOutput for output nodes and input derivatives).
  std::vector<int32> left_commands, middle_commands, right_commands;

  for (size_t s = 0; s < segments.size(); s++) {
    int32 segment_start = segments[s].first,
        segment_end = segments[s].second;
    left_commands.clear();
    middle_commands.clear();
    right_commands.clear();
    for (int32 c = segment_start; c < segment_end; c++) {
      if (computation->commands[c].command_type == kProvideOutput) {
        right_commands.push_back(c);
      } else if (computation->commands[c].command_type == kAcceptInput) {
        left_commands.push_back(c);
      } else {
        middle_commands.push_back(c);
      }
    }
    std::vector<int32>::const_iterator iter = left_commands.begin(),
        end = left_commands.end();
    int32 c = segment_start;
    for (; iter != end; ++iter, ++c)
      reordered_commands[c] = computation->commands[*iter];
    iter = middle_commands.begin();
    end = middle_commands.end();
    for (; iter != end; ++iter, ++c)
      reordered_commands[c] = computation->commands[*iter];
    iter = right_commands.begin();
    end = right_commands.end();
    for (; iter != end; ++iter, ++c)
      reordered_commands[c] = computation->commands[*iter];
    KALDI_ASSERT(c == segment_end);
  }
  computation->commands.swap(reordered_commands);
}
