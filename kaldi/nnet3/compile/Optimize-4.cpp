ComputationLoopedOptimizer(const Nnet &nnet,
                           NnetComputation *computation):
    nnet_(nnet), computation_(computation) { }


void OptimizeLoopedComputation(const Nnet &nnet,
                               NnetComputation *computation) {
  ComputationLoopedOptimizer optimizer(nnet, computation);
  optimizer.Optimize();
}




bool ComputationLoopedOptimizer::Optimize() {
  analyzer_.Init(nnet_, *computation_);
  
  KALDI_ASSERT(!computation_->matrix_debug_info.empty() &&
               "You must request matrix debug info when compiling "
               "looped computations.");

  // 获得 潜在分割点的indexes, 每个Computation一个segment
  // 我们定位kNoOperationPermanent为分割点, 这能够保证在input接收之后, 并在段中计算之前
  // 并且当然在我们output之前.
  // 通过选择这个为分割点我们能够避免一些问题的发生, 例如 如果我们选择segment边界 kNoOperationMarker.
  std::vector<int32> splice_points;
  GetCommandsOfType(*computation_, kNoOperationPermanent,
                    &splice_points);

  
  // 每个段的 时间shift.
  int32 time_shift_per_segment = FindTimeShift(*computation_);

  
  // 找到  在每个分割点上的actived matrix list.
  std::vector<std::vector<int32> > active_matrices;
  FindActiveMatrices(*computation_, analyzer_, splice_points, &active_matrices);

  // unique-id 是通过map 安排的每个matrix的cindexes的id
  // t_offset 是每个matrix的cindexes 中的left_context 在utt中的时间偏移
  // 找到一个计算的matrix的表示方法, 用pair(unique_id, time_offset)
  // 更加容易能够找到matrices 能够表示 只在时间offset不同的cindexes的list
  std::vector<std::pair<int32, int32> > matrix_to_pair;
  CreateMatrixPairs(*computation_, &matrix_to_pair);

  // 一个pair to matrix的反向map. 后续会用到.
  unordered_map<std::pair<int32, int32>, int32, PairHasher<int32> > pair_to_matrix;
  GetPairToMatrixMap(matrix_to_pair, &pair_to_matrix);



  
  // 获得在pair描述下的 每个分割点的 matrix-pair list
  std::vector<std::vector<std::pair<int32, int32> > > pair_lists;
  ConvertListsToPairLists(active_matrices, matrix_to_pair, &pair_lists);

  // 找到 第一个重复分割点, 只在时间偏移上有不同的 两个相似的段.
  // 段中所有的matrix都 只在内部的cindexes的 Index(n,t,) t域有不同
  // 不同也仅仅是时间偏移.
  int32 seg1, seg2;
  if (!FindFirstRepeat(pair_lists,
                       time_shift_per_segment,
                       &seg1, &seg2)) {
    KALDI_VLOG(2) << "Could not find repeats of variables.";
    return false;
  }

  // 找到上面得到的两个结构相同的分割点的 所有matrices.
  std::vector<int32> seg1_matrices, seg2_matrices;
  GetIdentifiedMatrices(pair_lists[seg1], pair_lists[seg2],
                        pair_to_matrix,
                        &seg1_matrices, &seg2_matrices);

  // 时间区别(两个eg seg1 seg2之间只具有不同的 time_offset)
  // time_shift_per_segment 是两个seg的time移动 一般就是8.
  int32 time_difference = time_shift_per_segment * (seg2 - seg1);


  // 通过matrix的cindex的内的数据判断 seg的选择正确性.
  // in:
  // seg1_matrices seg1中所有matrices
  // seg2_matrices seg2中所有matrices
  // time_difference seg1 seg2之间的时间区别
  CheckIdentifiedMatrices(*computation_, seg1_matrices, seg2_matrices, time_difference);
  
  // 两个 切分点. 构建一个循环的命令结构
  FormInfiniteLoop(splice_points[seg1], splice_points[seg2], computation_);

  // 添加 SubMatirxSwap命令. 将两个seg的所有matrix的 whole subMatrices 增加swapMatrix操作.
  AddMatrixSwapCommands(seg1_matrices, seg2_matrices, computation_);

  // 重新排序number ..... 
  RenumberComputation(computation_);

  FixGotoLabel(computation_);

  return true;
}



int32 ComputationLoopedOptimizer::FindTimeShift(
    const NnetComputation &computation) {
  std::vector<int32> segment_ends;

  // 计算每个kNoOperationMarker的命令位置.
  GetCommandsOfType(computation, kNoOperationMarker, &segment_ends);
  KALDI_ASSERT(segment_ends.size() >= 3);

  // 忽略第一个段, 因为第一个段是一个特殊情况?
  // Ignore the first segment as it tends to be a special case
  // (it has more left context),
  int32 second_segment_begin = segment_ends[0],
      third_segment_begin = segment_ends[1],
      fourth_segment_begin = segment_ends[2];
  
  int32 first_output_command_seg2 = -1,
      first_output_command_seg3 = -1;

  // 选取第二个段中kProvideOutput命令位置.
  // ==> first_output_command_seg2
  for (int32 c = second_segment_begin; c < third_segment_begin; c++)
    if (computation.commands[c].command_type == kProvideOutput &&  first_output_command_seg2 < 0)
      first_output_command_seg2 = c;

  // 选取第三个段中的
  for (int32 c = third_segment_begin; c < fourth_segment_begin; c++)
    if (computation.commands[c].command_type == kProvideOutput &&
        first_output_command_seg3 < 0)
      first_output_command_seg3 = c;

  
  if (first_output_command_seg2 < 0 || first_output_command_seg3 < 0)
    KALDI_ERR << "Could not locate output commands for segments 2 and 3.";

  
  const NnetComputation::Command
      &command2 = computation.commands[first_output_command_seg2],
      &command3 = computation.commands[first_output_command_seg3];
  // 断言 kProvideOutput命令对应的节点相同
  int32 seg2_node = command2.arg2, seg3_node = command3.arg2;
  KALDI_ASSERT(seg2_node == seg3_node);

  // 断言ProvideOutput命令的输出都是完整matrix
  int32 seg2_submatrix = command2.arg1,
      seg3_submatrix = command3.arg1;
  
  KALDI_ASSERT(computation.IsWholeMatrix(seg2_submatrix) &&
               computation.IsWholeMatrix(seg3_submatrix));
  
  int32 seg2_matrix = computation.submatrices[seg2_submatrix].matrix_index,
      seg3_matrix = computation.submatrices[seg3_submatrix].matrix_index;
  KALDI_ASSERT(computation.matrices[seg2_matrix].num_rows ==
               computation.matrices[seg3_matrix].num_rows);

  
  KALDI_ASSERT(!computation.matrix_debug_info.empty());

  const NnetComputation::MatrixDebugInfo
      &debug_info2 = computation.matrix_debug_info[seg2_matrix],
      &debug_info3 = computation.matrix_debug_info[seg3_matrix];

  // 两个段的时间偏移????
  // 因为在miniRequest中 实际上是至少两个 eg,
  // 即每个eg是一个段, 两个eg之间是具有相同 n 不同t的frames.
  int32 t_offset = debug_info3.cindexes[0].second.t - debug_info2.cindexes[0].second.t;
  
  int32 num_rows = debug_info2.cindexes.size();
  for (int32 r = 0; r < num_rows; r++) {
    KALDI_ASSERT(debug_info3.cindexes[r].second.t ==
                 debug_info2.cindexes[r].second.t + t_offset);
  }
  return t_offset;
}


// 输出 一个 command indexes的 list
// 对每个 分割点 command 输出一个 matrix list, 在该命令上是active
// active表示 matrix已经在分割点之前被写入, 在分割点之后会被read, 说明在该分割点 matrix是actived.
// 是个二级vector
void ComputationLoopedOptimizer::FindActiveMatrices(
    const NnetComputation &computation,
    const Analyzer &analyzer,
    const std::vector<int32> &splice_point_commands,
    std::vector<std::vector<int32> > *active_matrices) {
  
  int32 num_matrices = computation.matrices.size();
  int32 num_splice_points = splice_point_commands.size();
  
  active_matrices->clear();
  active_matrices->resize(num_splice_points);

  
  // this object just makes available some extra functions, vs. the Analyzer
  // object.
  ComputationAnalysis analysis(computation, analyzer);
  KALDI_ASSERT(IsSortedAndUniq(splice_point_commands));

  // 每个matrix的 submatrix(对matrix的全部引用的submatrix)
  std::vector<int32> whole_submatrices;
  computation.GetWholeSubmatrices(&whole_submatrices);

  // 所有matrix
  for (int32 m = 1; m < num_matrices; m++) {
    // 如下是一些command indexes, 可与 splice_point_commands的indexes 比较的?

    int32 s = whole_submatrices[m],  // submatrix consisting of the whole of
                                     // 'm'.
        first_access = analysis.FirstNontrivialAccess(s),
        last_access = analysis.LastAccess(s);
    // 每个kNoOpearationPerment 分割点
    for (int32 i = 0; i < num_splice_points; i++) {
      
      int32 splice_point = splice_point_commands[i];
      // 如果submatrix的第一个访问在访问之前, 最后一个访问在分割之后
      // 这个matrix在当前分割点 认为是actived的.
      if (first_access < splice_point && last_access > splice_point) {
        (*active_matrices)[i].push_back(m);
      }
    }
  }
}



// 这里通过 matrix_debug_info 来获得 matrix的 cindexes
// 然后通过cindexes的t_offset 来重新排序 matrix.
// 具体每个matrix的 matrix_debug_info是什么, 是对matrix的调试信息输出
// 保存的cindexes 和 matrix 的一样.
void ComputationLoopedOptimizer::CreateMatrixPairs(
    const NnetComputation &computation,
    std::vector<std::pair<int32, int32> > *matrix_to_pair) {
  
  typedef unordered_map<std::vector<Cindex>, int32, CindexVectorHasher>  MapType;
  
  int32 cur_vector_id = 1;
  // cindex_map 将vector<Cindex> 映射为一个 value,然后我们可以算出一个能够加入到 is_deriv值计算的id
  MapType cindex_map;


  // 为每个matrix分配一个pair
  int32 num_matrices = computation.matrices.size();
  matrix_to_pair->resize(num_matrices);

  
  KALDI_ASSERT(computation.matrix_debug_info.size() == num_matrices);


  for (int32 m = 1; m < num_matrices; m++) {
    KALDI_ASSERT(!computation.matrix_debug_info[m].cindexes.empty());
   
    std::vector<Cindex> cindexes = computation.matrix_debug_info[m].cindexes;
    // 让cindexes的t域 从0开始计数, 并返回原本的left_context所在时间帧time.
    // (即left_context相对utt的时间偏移)
    int32 t_offset = NormalizeCindexes(&cindexes);
    int32 ComputationLoopedOptimizer::NormalizeCindexes(std::vector<Cindex> *cindexes) {
      std::vector<Cindex>::iterator
          iter = cindexes->begin(),
          end = cindexes->end();

      // 第一个正常时间.
      int32 ans;
      for (; iter != end; iter++) {
        if (iter->second.t != kNoTime) {
          ans = iter->second.t;
          break;
        }
      }
      
      if (iter == end) {
        // this should not happen.
        KALDI_ERR << "All t values are kNoTime in matrix.";
      }

      // 格式化cindexes(node-index, Index(n,t,x))的 t域 保证都是从0 开始
      iter = cindexes->begin();
      for (; iter != end; iter++)
        if (iter->second.t != kNoTime)
          iter->second.t -= ans;

      // 返回时间偏移---- 这个时间偏移应该是 本个样本eg 在其对应utt的 leftcontext的时间帧 的偏移.
      return ans;
    }

    // 判断cindexes是否已经加入cindex_map  
    // 存在 直接返回cindexes的已知id
    // 否则 创建新的 map键值对, 获得id
    MapType::const_iterator iter = cindex_map.find(cindexes);
    int32 vector_id;
    if (iter != cindex_map.end()) {
      vector_id = iter->second;
    } else {
      vector_id = cur_vector_id++;
      cindex_map[cindexes] = vector_id;
    }

    // matrix_debug_info[m] 是否是deriv的.
    bool is_deriv = computation.matrix_debug_info[m].is_deriv;
    int32 unique_id = 2 * vector_id + (is_deriv ? 1 : 0);
    (*matrix_to_pair)[m].first = unique_id;
    (*matrix_to_pair)[m].second = t_offset;
  }
}


void ComputationLoopedOptimizer::GetPairToMatrixMap(
      std::vector<std::pair<int32, int32> > &matrix_to_pair,
      unordered_map<std::pair<int32, int32>, int32, PairHasher<int32> > *pair_to_matrix) {
  int32 num_matrices = matrix_to_pair.size();
  // actually there are one fewer matrices than num_matrices.
  pair_to_matrix->clear();
  for (int32 m = 1; m < num_matrices; m++)
    (*pair_to_matrix)[matrix_to_pair[m]] = m;
}



// static
void ComputationLoopedOptimizer::ConvertListsToPairLists(
      const std::vector<std::vector<int32> > &active_matrices,
      const std::vector<std::pair<int32, int32> > &matrix_to_pair,
      std::vector<std::vector<std::pair<int32, int32> > > *active_pairs) {

  // active_pairs 设置大小为 分割点数量
  // active_matrices 是vect 每个分割点 vector matrices.
  active_pairs->clear();
  active_pairs->resize(active_matrices.size());

  // matrices总数
  int32 num_matrices = matrix_to_pair.size();

  // 每个分割点, 每个seg
  for (size_t seg = 0; seg < active_matrices.size(); seg++) {
    // 该分割点的 active_matrices
    const std::vector<int32> &this_active_matrix_list = active_matrices[seg];

    // 安排该分割点的 active_pairs, 保存大小为 该分割点的 active_matrices.
    std::vector<std::pair<int32, int32> > &this_active_pair_list = (*active_pairs)[seg];
    this_active_pair_list.resize(this_active_matrix_list.size());
    
    std::vector<int32>::const_iterator
        iter = this_active_matrix_list.begin(),
        end  = this_active_matrix_list.end();
    
    std::vector<std::pair<int32, int32> >::iterator
        out_iter = this_active_pair_list.begin();
    // 将matrix_to_pair 输出到每个 active_pairs.
    for (; iter != end; ++iter, ++out_iter) {
      KALDI_ASSERT(*iter > 0 && *iter < num_matrices);
      *out_iter = matrix_to_pair[*iter];
    }
  }
}



// 找到第一次两个eg 具有相同的pair结构
bool ComputationLoopedOptimizer::FindFirstRepeat(
    const std::vector<std::vector<std::pair<int32, int32> > > &active_pairs,
    int32 time_shift_per_segment,
    int32 *seg1, int32 *seg2) {

  // 分割点总数 -- segment总数
  int32 num_segments = active_pairs.size();

  // 这个算法是将会很慢, 但是段的数量将会相当小, 并且active_pairs中元素的比较将会很块.
  KALDI_ASSERT(num_segments >= 2);

  // 相邻两个分割点
  // 若相邻两个分割点(两个eg) 只在时间偏移上不同时返回true
  // 返回两个只在 time_offset有不同的段
  // 其他都相同(matrix-unique-id相同, 且t_offset 相差相同)
  for (int32 s = 0; s < num_segments; s++) {
    for (int32 t = s + 1; t < num_segments; t++) {
      if (ListsAreEqualExceptForPossibleShift(active_pairs[s],
                                              active_pairs[t],
                                              (t - s) * time_shift_per_segment)) {
        *seg1 = s;
        *seg2 = t;
        return true;
      }
    }
  }
  return false;
}



// 这个函数告诉我们 除了time-shift之外 是相同的
// a b 中的每个元素是pair(matrix-unique-index, time-offset)
// 假设我们有两个pair p1=(m1, o1) p2=(m2, o2)
// p1 和 p2 是等价的条件是 require m2 == m1 and either o2 == o1 + 'shift' or o2 == o1.
// 这个返回 a b 中的每两个pair pa pb 都等价时 返回true.

bool ComputationLoopedOptimizer::ListsAreEqualExceptForPossibleShift(
    const std::vector<std::pair<int32, int32> > &a,
    const std::vector<std::pair<int32, int32> > &b,
    int32 shift) {
  size_t size = a.size();
  if (b.size() != size)
    return false;
  for (size_t i = 0; i < size; i++) {
    const std::pair<int32, int32> &p1 = a[i],
        &p2 = b[i];
    if (p1.first != p2.first)
      return false;
    if (p2.second != p1.second + shift && p2.second != p1.second)
      return false;
  }
  return true;
}




// pair_list1 保存的是 1分割点下的active_matrix的 pairs.
void ComputationLoopedOptimizer::GetIdentifiedMatrices(
    const std::vector<std::pair<int32, int32> > &pair_list1,
    const std::vector<std::pair<int32, int32> > &pair_list2,
    const unordered_map<std::pair<int32, int32>, int32, PairHasher<int32> > &pair_to_matrix,
    std::vector<int32> *matrix_list1,
    std::vector<int32> *matrix_list2) {

  
  size_t size = pair_list1.size();
  KALDI_ASSERT(pair_list2.size() == size);
  
  matrix_list1->clear();
  matrix_list2->clear();
  matrix_list1->reserve(size);
  matrix_list2->reserve(size);

  // 对两个eg 分割点的 
  std::vector<std::pair<int32, int32> >::const_iterator
      iter1 = pair_list1.begin(),
      end1 = pair_list1.end(),
      iter2 = pair_list2.begin();
  for (; iter1 != end1; ++iter1, ++iter2) {
    // 如果时间偏移相同, 跳过不处理.
    if (iter1->second == iter2->second)
      continue;
    // skip those that have no time shift, we won't have to do any swapping for
    // those.

    // 反向映射为matrix.
    unordered_map<std::pair<int32, int32>, int32, PairHasher<int32> >::const_iterator
        map_iter1 = pair_to_matrix.find(*iter1),
        map_iter2 = pair_to_matrix.find(*iter2);

    // 如果没查到该pair对应的matrix 出错
    if (map_iter1 == pair_to_matrix.end() ||  map_iter2 == pair_to_matrix.end())
      KALDI_ERR << "Could not find pair in map (code error)";

    // 保存该相同结构的分割点的 matrices
    matrix_list1->push_back(map_iter1->second);
    matrix_list2->push_back(map_iter2->second);
  }
}



// seg1 seg2 具有一个特性 是内部的cindexes的所有信息除了t时间域 剩下都相同.
// list1 seg1中的所有matrices
void ComputationLoopedOptimizer::CheckIdentifiedMatrices(
    const NnetComputation &computation,
    const std::vector<int32> &list1,
    const std::vector<int32> &list2,
    int32 time_difference) {

  
  KALDI_ASSERT(time_difference > 0);
  KALDI_ASSERT(list1.size() == list2.size());
  KALDI_ASSERT(!computation.matrix_debug_info.empty());

  // 每个matrix
  for (size_t i = 0; i < list1.size(); i++) {
    int32 m1 = list1[i], m2 = list2[i];
    
    const NnetComputation::MatrixInfo
        &matrix_info1 = computation.matrices[m1],
        &matrix_info2 = computation.matrices[m2];
    KALDI_ASSERT(matrix_info1.num_rows == matrix_info2.num_rows &&
                 matrix_info1.num_cols == matrix_info2.num_cols &&
                 matrix_info1.stride_type == matrix_info2.stride_type);
    
    const NnetComputation::MatrixDebugInfo
        &debug_info1 = computation.matrix_debug_info[m1],
        &debug_info2 = computation.matrix_debug_info[m2];
    KALDI_ASSERT(debug_info1.is_deriv == debug_info2.is_deriv);
    KALDI_ASSERT(debug_info1.cindexes.size() == debug_info2.cindexes.size());


    // seg1下该matrix的 cindexes的每个cindex
    // cindex.first == node-index
    // cindex.second == Index(n, t, x)
    std::vector<Cindex>::const_iterator
        iter1 = debug_info1.cindexes.begin(),
        end1 = debug_info1.cindexes.end(),
        iter2 = debug_info2.cindexes.begin();

    // 检查 seg1 seg2 的matrices来判断 选择的正确性.
    for (; iter1 != end1; iter1++,iter2++) {
      KALDI_ASSERT(iter2->first == iter1->first &&  
                   iter2->second.n == iter1->second.n &&
                   ((iter1->second.t == kNoTime && iter2->second.t == kNoTime) ||
                    iter2->second.t == iter1->second.t + time_difference) &&        // 只在cindexes的t域有区别
                   iter2->second.x == iter1->second.x);
    }
  }
}



// 构建无限循环的命令结构
void ComputationLoopedOptimizer::FormInfiniteLoop(
    int32 command1, int32 command2,
    NnetComputation *computation) {


  // 不太理解 到底这两个命令 kNoOperationPermanent 的作用是什么?
  // kNoOperationPermanent 分割两个seg的所有命令?
  // 这些命令是怎么个结构的?
  
  KALDI_ASSERT(static_cast<int32>(computation->commands.size()) >= command2 + 1 && command1 < command2);
  KALDI_ASSERT(
      computation->commands[command1].command_type == kNoOperationPermanent &&
      computation->commands[command2].command_type == kNoOperationPermanent);

  // 删除所有 第二个分割点之后的command.
  computation->commands.resize(command2 + 1);
  // 将分割点命令 设置为 kGotoLabel 指向上一个命令分割点, 形成环.
  computation->commands[command2].command_type = kGotoLabel;
  computation->commands[command2].arg1 = command1;

  // 向第一个分割点之后 加入kNoOperationLabel 命令
  // 现在kNoOperationLabel 命令在 command1命令位置了.
  // kNoOperationLabel 是 kGotoLabel 命令的目标.
  NnetComputation::Command c(kNoOperationLabel);
  computation->commands.insert(computation->commands.begin() + command1, c);
  
  // Now the kNoOperationLabel command is at position 'command1'.
}




void ComputationLoopedOptimizer::AddMatrixSwapCommands(
    const std::vector<int32> &matrices1,
    const std::vector<int32> &matrices2,
    NnetComputation *computation) {

  
  std::vector<std::pair<int32, int32> > swaps;
  // 在easy情况下 matrices matrices2是不相交的.
  // swap操作 会形成 这样的结构 vector { (matrices1[0],matrices2[0]), (matrices1[1],matrices2[1]), ... },
  // 在复杂情况下 可能需要重排序
  GetMatrixSwapOrder(matrices1, matrices2, &swaps);

  // 获得刚刚 FormInfiniteLoop()形成的 循环结构的kGotoLabel 命令.
  NnetComputation::Command goto_label_command = computation->commands.back();
  KALDI_ASSERT(goto_label_command.command_type == kGotoLabel);
  
  computation->commands.pop_back();

  // 获得所有matrices 的 whole submatrices.
  std::vector<int32> whole_submatrices;
  computation->GetWholeSubmatrices(&whole_submatrices);
  
  size_t num_matrices = whole_submatrices.size();

  
  for (size_t i = 0; i < swaps.size(); i++) {
    int32 m1 = swaps[i].first, m2 = swaps[i].second;
    KALDI_ASSERT(static_cast<size_t>(m1) < num_matrices &&
                 static_cast<size_t>(m2) < num_matrices);
    int32 s1 = whole_submatrices[m1], s2 = whole_submatrices[m2];

    // 添加swap交叉命令. 形成 submatrices 的价差结构么
    computation->commands.push_back(
        NnetComputation::Command(kSwapMatrix, s1, s2));
  }
  computation->commands.push_back(goto_label_command);
}




void FixGotoLabel(NnetComputation *computation) {
  int32 num_commands = computation->commands.size();
  if (num_commands == 0)
    return;
  
  for (int32 c = num_commands - 1; c >= 0; c--) {
    if (computation->commands[c].command_type == kGotoLabel) {
      int32 dest_command = computation->commands[c].arg1;
      
      if (static_cast<size_t>(dest_command) <  computation->commands.size() &&
          computation->commands[dest_command].command_type == kNoOperationLabel)
        return;  // nothing to fix.
      
      for (int32 d = 0; d + 1 < num_commands; d++) {
        if (computation->commands[d].command_type == kNoOperationLabel) {
          computation->commands[c].arg1 = d;
          return;
        }
      }
      KALDI_ERR << "Label not found.";
    } else if (computation->commands[c].command_type == kProvideOutput) {
      // sometimes kProvideOutput commands are temporarily ordered after
      // the kGotoLabel command, and we need to work in that case.
      continue;
    } else {
      // it loks like there is no 'goto' command in this computation-
      // if there were, it would be right at the end, possibly followed by
      // kProvideOutput commands.
      break;
    }
  }
}
