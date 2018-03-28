//src/nnet3/nnet-optimize-utils.cc

// 实现ExpandComputation的操作, 
class ComputationExpander {
 public:
  ComputationExpander(const Nnet &nnet,
                      const MiscComputationInfo &misc_info,
                      const NnetComputation &computation,
                      bool need_debug_info,
                      int32 num_n_values,
                      NnetComputation *expanded_computation):
      nnet_(nnet), misc_info_(misc_info),
      computation_(computation),
      need_debug_info_(need_debug_info),
      num_n_values_(num_n_values),
      expanded_computation_(expanded_computation) {
    KALDI_ASSERT(num_n_values > 2);
  }

  
  // This function call implements the functionality of the class,
  // expanding the computation.
  void Expand();
}
  






/**
 // 这个函数用在 shortcut 编译生成之后, 扩展 用两个Example构建的mini_request生成的 computation
 // 扩展为 正常多Example的reqeust的computation

     @param [in] nnet         The neural network for which this computation
                              is being built.

     //  用来生成PrecomputeIndex用的.
     @param [in] misc_info    The same MiscComputationInfo object that was
                              present in the ComputationRequests that were
                              originally used to generate the computation
                              (required to generated the PrecomputedIndexes)

                              
     @param [in] computation  The computation that was compiled for exactly
                              2 'n' values (n=0 and n=1)

                              
     @param [in] need_debug_info True if we want to retain the 'debug_info'
                              in the output 'expanded_computation'.  In any
                              case, the 'debug_info' is required in the
                              input computation.


                              
     @param [in] num_n_values The number of 'n' values we want in the output
                              computation

                              
     @param [out] expanded_computation  The expanded computation.

 */
void ExpandComputation(const Nnet &nnet,
                       const MiscComputationInfo &misc_info,
                       const NnetComputation &computation,
                       bool need_debug_info,
                       int32 num_n_values,
                       NnetComputation *expanded_computation) {
  ComputationExpander expander(nnet, misc_info, computation,
                               need_debug_info, num_n_values,
                               expanded_computation);
  expander.Expand();
}






void ComputationExpander::Expand() {

  InitStrideInfo();
  
  ComputeMatrixInfo();
  
  if (need_debug_info_)
    ComputeDebugInfo();
  else
    expanded_computation_->matrix_debug_info.clear();

  
  ComputeSubmatrixInfo();


  // ----------- not read -----------
  ComputePrecomputedIndexes();

  
  ComputeCommands();

  
  expanded_computation_->need_model_derivative =  computation_.need_model_derivative;
  
}



// n_stride 是每个Example的 frame总数  left_context=18 cur_frames=8  right_context=12
// 当old_row_index 是第一个样本的cindex时, 返回 原本的cindex位置
// 当old_row_index 是第二个样本的cindex时, 返回 扩展为最后一个样本的相对位置.
int32 ComputationExpander::GetNewMatrixLocationInfo(
    int32 matrix_index, int32 old_row_index) const {

  // to understand 'block_size', read the comment for FindNStride().
  int32
      n_stride = n_stride_[matrix_index],
      old_num_n_values = 2,
      new_num_n_values = num_n_values_,

      // matrix的cindexes数目?
      old_block_size = old_num_n_values * n_stride,
      
      // 新的matrix的 cindexes总数
      new_block_size = new_num_n_values * n_stride,
      
      block_index = old_row_index / old_block_size,
      // 在对应block中的index,
      offset_within_block = old_row_index % old_block_size;

  // 在每个block内, 我们可以显示, 给定一个假设,
  // 我们有
  // 一个子块的数据是n==0的
  // 一个子块的数据 n==1,
  // ...
  // 一个子块的数据 n==num_examples-1,
  
  // 对于 input computation 这里没有后续n>1的情况. 因为input computation 只具有两个Example
  int32
      // 第几个样本 
      old_n_value = offset_within_block / n_stride,
      // cindex在 n_stride中的位置.
      index_within_subblock = offset_within_block % n_stride;

  // mini computation 中matrix-index 的cindexes
  const std::vector<Cindex> &cindexes = computation_.matrix_debug_info[matrix_index].cindexes;

  // 原本的第old_row_index行的cindex的n域 与 old_n_value 相等
  // old_n_value 必然是0 或者 1.
  KALDI_ASSERT(old_n_value == cindexes[old_row_index].second.n && (old_n_value == 0 || old_n_value == 1));


  
  // Search for CAVEAT in the comment for this function to see what this is
  // about.  Mapping old_n_value == 1 -> new_n_value == new_num_n_values - 1
  // just happens to be useful for the way we use this function... it maps the
  // end of an old submatrix to the end of a new submatrix.
  int32 new_n_value = (old_n_value == 0 ? 0 : new_num_n_values - 1);

  return block_index * new_block_size + index_within_subblock +
      new_n_value * n_stride;
}


// 这个里面调用了FindNStride(cindexes, full_check)
// 这里面具体得到的是什么呢? 现在姑且认为是 frames_per_eg = 8.
void ComputationExpander::InitStrideInfo() {
  // note: the zeroth matrix is not a real matrix, it's the empty matrix.
  int32 num_matrices = computation_.matrices.size();
  n_stride_.resize(num_matrices);
  n_stride_[0] = 0;

  // the input computation to class ComputationExpander is required to
  // have its debug info set up.
  KALDI_ASSERT(!computation_.matrix_debug_info.empty());
  for (int32 m = 1; m < num_matrices; m++) {
    int32 num_rows = computation_.matrices[m].num_rows;
    const NnetComputation::MatrixDebugInfo &debug_info = computation_.matrix_debug_info[m];
    KALDI_ASSERT(debug_info.cindexes.size() == num_rows);
    bool full_check = true;  // TODO: eventually change this to false.
    int32 n_stride = FindNStride(debug_info.cindexes, full_check);
    if (n_stride == 0) {
      KALDI_ERR << "Problem encountered in 'shortcut' compilation: the computation "
                << "does not have the expected structure.  Try compiling with "
                << "--use-shortcut=false.";
    }
    n_stride_[m] = n_stride;
  }
}

// 将原本的2个example的 每个matrix /2 *n
void ComputationExpander::ComputeMatrixInfo() {
  int32 num_matrices = computation_.matrices.size();
  expanded_computation_->matrices.resize(num_matrices);
  
  // Matrix zero is a special case; it's the empty matrix.
  expanded_computation_->matrices[0] = computation_.matrices[0];
  
  int32 old_num_n_values = 2,
      new_num_n_values = num_n_values_;

  // 扩充matrix的方式 很简单 就是原本的matrix都是对2个Example的, 所以rows 是 2个Examples的
  // 这里直接 matrix.rows/2 * n 就得到了目标的matrices
  // matrices总数不变, 行数变了.
  // 说明原本computation中的matrices 都是对所有Examples的.???
  for (int32 m = 1; m < num_matrices; m++) {
    expanded_computation_->matrices[m] = computation_.matrices[m];
    expanded_computation_->matrices[m].num_rows = (computation_.matrices[m].num_rows / old_num_n_values) *  new_num_n_values;
  }
}

// 类似上面, 不过将cindexes 的n 域 设置为 0 - num_Examples (类似merge操作)
void ComputationExpander::ComputeDebugInfo() {
  int32 num_matrices = computation_.matrices.size();
  
  KALDI_ASSERT(computation_.matrix_debug_info.size() == num_matrices);
  
  expanded_computation_->matrix_debug_info.resize(num_matrices);
  
  // Matrix zero is a special case; it's the empty matrix.
  expanded_computation_->matrix_debug_info[0] =  computation_.matrix_debug_info[0];



  // 输出computation应该包含的Examples数量
  int32 num_n_values = num_n_values_;
  
  // 对每个matrix
  for (int32 m = 1; m < num_matrices; m++) {
    // mini_computation 的 每个对应的matrix_debug_info
    const NnetComputation::MatrixDebugInfo &info_in =  computation_.matrix_debug_info[m];
    // out_computation 的 每个matrix的 对应的matrix_debug_info的引用
    NnetComputation::MatrixDebugInfo &info_out = expanded_computation_->matrix_debug_info[m];
    
    info_out.is_deriv = info_in.is_deriv;
    int32 num_rows_in = computation_.matrices[m].num_rows,
        num_rows_out = expanded_computation_->matrices[m].num_rows;
    
    KALDI_ASSERT(num_rows_in == info_in.cindexes.size());
    
    info_out.cindexes.resize(num_rows_out);
    
    const Cindex *cindexes_in = &(info_in.cindexes[0]);
    Cindex *cindexes_out = &(info_out.cindexes[0]);

    // 多个Examples
    for (int32 r = 0; r < num_rows_in; r++) {
      // 如果mini_computation的matrix的 cindex 是第一个Example
      if (info_in.cindexes[r].second.n == 0) {

        // 获得对应的新Matrix位置的行位置 new_r
        int32
            new_r = GetNewMatrixLocationInfo(m, r),
            n_stride = n_stride_[m];

        // 新的debug_info的每行数据 cindexes
        // 直接拷贝第一个Examples的数据, 只修改对应的n域(Example-id) 为 0 - num_n_values
        for (int32 n = 0; n < num_n_values; n++) {
          int32 r_out = new_r + n * n_stride;
          cindexes_out[r_out] = cindexes_in[r];
          cindexes_out[r_out].second.n = n;
        }
      }
    }
  }
}

// 将submatrix中 对matrix的数据引用位置 修改为 N Example情况下的索引.
void ComputationExpander::ComputeSubmatrixInfo() {
  int32 num_submatrices = computation_.submatrices.size();
  expanded_computation_->submatrices.resize(num_submatrices);

  // Sub-matrix zero is a special case; it's the empty submatrix.
  expanded_computation_->submatrices[0] = computation_.submatrices[0];

  for (int32 s = 1; s < num_submatrices; s++) {

    // matrix总数不变, matrix内部的cindexes增多了.
    const NnetComputation::SubMatrixInfo &info_in = computation_.submatrices[s];
    int32 m = info_in.matrix_index;
    
    const NnetComputation::MatrixDebugInfo &debug_info_in =
        computation_.matrix_debug_info[m];

    // we may need to change the row_offset and num_rows.
    int32 first_row_in = info_in.row_offset,
        last_row_in = first_row_in + info_in.num_rows - 1;

    // 如果不是跨越两个样本的 submatrix.???
    if (!(debug_info_in.cindexes[first_row_in].second.n == 0 &&
          debug_info_in.cindexes[last_row_in].second.n == 1)) {
      std::ostringstream computation_ss;
      std::vector<std::string> submat_strings;
      computation_.GetSubmatrixStrings(nnet_, &submat_strings);
      computation_.Print(computation_ss, nnet_);
      KALDI_ERR << "Submatrix s" << s << " = " << submat_strings[s]
                << " has strange dimensions.  Computation is: "
                << computation_ss.str();
    }


    //  submatrix 映射后的 的 起始终止行位置
    int32
        first_row_out = GetNewMatrixLocationInfo(m, first_row_in),
        last_row_out = GetNewMatrixLocationInfo(m, last_row_in),
        // 总共行数
        new_num_rows = (last_row_out + 1 - first_row_out);

    
    NnetComputation::SubMatrixInfo &info_out =  expanded_computation_->submatrices[s];
    // matrix_index 保存相同index
    info_out.matrix_index = m;
    // 映射后的起始位置
    info_out.row_offset = first_row_out;
    // 映射后的总行数
    info_out.num_rows = new_num_rows;

    // col相对位置保持不变
    info_out.col_offset = info_in.col_offset;
    info_out.num_cols = info_in.num_cols;
  }
}




// 计算 输出的扩展computation的commands
// 实际就是直接拷贝即可, 因为matrix等都没变, 只不过是容量扩充
// 这里命令也就保持不变
// 但是这里也增加了扩充行的cindexes

// 这个函数也增加一些必要的indexes, indexes_multi index_ranges,主要工作也在这里.
// (因为原本的computation是为了两个Example的, 这里的computation是为了正常数据量的)
// 稍后我们可以调用 RenumberComputation() 来移除这里增加的重复的东西.

void ComputationExpander::ComputeCommands() {
  int32 num_commands = computation_.commands.size();
  expanded_computation_->commands.resize(num_commands);
  
  for (int32 command_index = 0; command_index < num_commands;
       command_index++) {
    const NnetComputation::Command &c = computation_.commands[command_index];
    NnetComputation::Command &c_out =
        expanded_computation_->commands[command_index];

    c_out = c;

    // 只操作在submatrices  component precomputed-indexes 上的 命令
    // 并不必要改变,因为我们会处理这个扩展, 通过恰当的对matrices 和 submatrices的重新定义
    // 以及重新创建 precomputed-indexes.

    // 然后 处理 indexes indexes_multi indexes_ranges的命令 一定需要被修改
    switch (c.command_type) {
      case kAllocMatrix:
      case kDeallocMatrix:
      case kSetConst:
      case kSwapMatrix:
      case kPropagate: case kBackprop:
      case kBackpropNoModelUpdate: case kMatrixCopy: case kMatrixAdd:
        break;
        
      case kCopyRows: case kAddRows:
        ExpandRowsCommand(c, &c_out);
        break;
      case kCopyRowsMulti: case kAddRowsMulti:
      case kCopyToRowsMulti: case kAddToRowsMulti:
        ExpandRowsMultiCommand(c, &c_out);
        break;
      case kAddRowRanges:
        ExpandRowRangesCommand(c, &c_out);
        break;
        
      case kAcceptInput: case kProvideOutput: case kNoOperation:
      case kNoOperationPermanent: case kNoOperationMarker:
      case kNoOperationLabel: case kGotoLabel:
        break;
      default:
        KALDI_ERR << "Un-handled command type";
    }
  }
}





// =============== Note computation_.indexes 保存的东西
// computation 中很多都是矩阵运算, 这样使用的数据就是 subMatrix
// 但是也有 AddRow这样的命令, 这样的命令的数据 是 cindexes 是一个cindex的vector
// eg AddRow(arg1, arg2, arg3)
// arg1 是 目标subMatrix
// arg2 是 源 subMatrix
// arg3 是 源中的各行索引 是个 vector<index>的索引
//      索引的就是 computation_.indexes中的位置
//      computation_.indexes 格式为  vector<               vector<index>     >
//                                   每个Row类命令的数据,  某个subMatrix的多个行

// // used in kAddRows, kAddToRows, kCopyRows, kCopyToRows.  contains row-indexes.
// std::vector<std::vector<int32> > indexes;


void ComputationExpander::ExpandRowsCommand(
    const NnetComputation::Command &c_in,
    NnetComputation::Command *c_out) {

  
  // 需要扩充 c_in.arg3中的行索引vector<row-index>, 然后将结果复制给 c_out.arg3
  
  int32 s1 = c_in.arg1, s2 = c_in.arg2;

  // 调用这个函数的命令 应该是  submat1.AddRows(submat2, indexes)
  // 如果submat1 是 s1的submatrx submat2 是 s2的submatrix
  // indexes vector<row-index>行索引 应该具有和submat1相同的行数目, 并且vector中的值是 s2中的行索引.

  // 原本的 Row命令的 多行数据索引 index in computation.indexes.
  int32 old_arg3 = c_out->arg3;

  // 向expand_computation_ 的indexes 中加入新的 命令数据 vector<int32> 某个subMatrx的多个行索引.
  c_out->arg3 = expanded_computation_->indexes.size();
  expanded_computation_->indexes.push_back(std::vector<int32>());


  // 新数据保存的位置
  std::vector<int32> &new_indexes = expanded_computation_->indexes.back();
  // 原本数据保存的位置
  const std::vector<int32> &old_indexes = computation_.indexes[old_arg3];

  
  int32
      // 原本数据总行数
      old_size = old_indexes.size(),
      // 新的Example总数
      num_n_values = num_n_values_,

      // 新的subMatrix的总行数
      new_s1_size = expanded_computation_->submatrices[s1].num_rows,
      new_s2_size = expanded_computation_->submatrices[s2].num_rows;

  KALDI_ASSERT(old_size == computation_.submatrices[s1].num_rows);

  new_indexes.resize(new_s1_size, -1);


  // i1 是目标submatrix中的行索引
  // i2 是源 submatrix中的行索引  the CopyRows or AddRows command.
  // n0 表示数据行中Index的n域 = 0.
  // 没有new的变量表示 old computation的引用
  // 有 new的 变量表示 新的computation的数据的引用.
  
  for (int32 i1 = 0; i1 < old_size; i1++) {
    
    int32 new_i1_n0, n_stride1;

    // 获得原本i1 在新的compuation中subMatrix s1 的位置
    // (新旧computation的submatrix索引相同, 只改变了内部的行总数)

    // 只考虑 n=0的情况,其他情况直接copy.
    if (GetNewSubmatLocationInfo(s1, i1, &new_i1_n0, &n_stride1)) {
      // GetNewSubmatLocationInfo() returns true if this corresponds to a Cindex with n == 0.

      // old computation中 Row命令的源submatrix的 行索引.
      int32 i2 = old_indexes[i1];  // note: i2 is the row index into submatrix s2.
      // 映射到new_i2_n0--- 经过扩充后,对应的行索引号
      int32 new_i2_n0, n_stride2;
      if (i2 < 0) {  // if i2 is -1, we'll just leave any relevant positions in
                     // 'new_indexes' with -1's in them.
        continue;
      } else {
        // 映射到新的 new_i2_n0 位置
        bool ans = GetNewSubmatLocationInfo(s2, i2, &new_i2_n0, &n_stride2);

        // 将 n=0的所有 索引位置, 直接通过计算获得 扩充的多个Example的索引位置, 完成computation.indexes的扩充
        int32 new_i1 = new_i1_n0, new_i2 = new_i2_n0;
        for (int32 n = 0; n < num_n_values; ++n, new_i1 += n_stride1, new_i2 += n_stride2) {
          KALDI_ASSERT(new_i1 < new_s1_size && new_i2 < new_s2_size);
          
          new_indexes[new_i1] = new_i2;
        }
      }
    }
  }
}





// 这个函数 扩展indexes集合,  这些indexes应该都具有正常的规律.
void ComputationExpander::ExpandIndexes(
    const std::vector<Index> &indexes,
    std::vector<Index> *indexes_expanded) const {

  
  bool full_check = false;
  // 返回indexes 中 每个样本的frames帧总数 规律.
  int32 n_stride = FindNStride(indexes, full_check);
  
  KALDI_ASSERT(n_stride > 0);
  
  ConvertNumNValues(n_stride, 2, num_n_values_, indexes, indexes_expanded);
}




/*

  将
  n变化范围为(0, 1, .. old_N-1)的Indexes 转化为
  n 变化范围为(0, 1, ... new_N - 1).的Indexes.

  输入的Indexes 应该具有 n_stride > 0
  输出的Indexes 具有相同的n_stride.
  
 */
static void ConvertNumNValues(int32 n_stride, int32 old_N, int32 new_N,
                              const std::vector<Index> &indexes_in,
                              std::vector<Index> *indexes_out) {

  int32 size_in = indexes_in.size();
  
  KALDI_ASSERT(size_in > 0 && indexes_in[size_in - 1].n == old_N - 1);
  
  int32
      block_size_in = n_stride * old_N,
      block_size_out = n_stride * new_N;

  // 输出Indexes的总数.
  indexes_out->resize((size_in / old_N) * new_N);

  // 只利用 输入Indexes中 第一个样本Example的Indexes 来生成 输出Indexes.
  for (int32 i_in = 0; i_in < size_in; i_in++) {
    if (indexes_in[i_in].n != 0)
      continue;
    
    Index index(indexes_in[i_in]);

    // 获得相对位置
    int32
        block_index = i_in / block_size_in,
        offset_within_block = i_in % block_size_in;


    // 根据输入Index的位置 计算输出Index的位置
    int32 i_out = block_index * block_size_out + offset_within_block;

    // 将Index 拷贝 new_N份 (即拷贝new_N个样本数量, 说明每个样本具有相同的数据结构)
    for (int32 n = 0; n < new_N; n++, i_out += n_stride) {
      index.n = n;
      (*indexes_out)[i_out] = index;
    }
    
  }
}


