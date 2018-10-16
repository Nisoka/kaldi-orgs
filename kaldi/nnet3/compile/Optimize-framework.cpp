// src/nnet3/nnet-optimize.cc

void Optimize(const NnetOptimizeOptions &config,
              const Nnet &nnet,
              int32 max_output_time_in_request,
              NnetComputation *computation) {
  if (GetVerboseLevel() >= 3) {
    CheckComputation(nnet, *computation, true);
    KALDI_LOG << "Before optimization, max memory use (bytes) = "
              << GetMaxMemoryUse(*computation);
  }

  { // Call LimitDerivativeTimes(); it's important that this
    // should come before other optimizations (search for "insist" in
    // nnet-optimize-utils.cc for the reasons).
    // this will do nothing unless --min-deriv-time or --max-deriv-time
    // or --max-deriv-time-relative was set.
    int32 max_deriv_time = config.max_deriv_time;
    if (config.max_deriv_time_relative != std::numeric_limits<int32>::max())
      max_deriv_time = config.max_deriv_time_relative +
          max_output_time_in_request;
    if (config.min_deriv_time != std::numeric_limits<int32>::min() ||
        max_deriv_time != std::numeric_limits<int32>::max())
      LimitDerivativeTimes(nnet, config.min_deriv_time,
                           max_deriv_time, computation);
  }



  // -------------- Optimize-1.cpp ------------------------

  if (config.optimize && config.consolidate_model_update) {
    ConsolidateModelUpdate(nnet, computation);

  }

  if (config.optimize && config.convert_addition) {
    ConvertAdditionToAssignment(nnet, computation);
  }



  
  // -------------  Optimize-2.cpp ---------------
  if (config.optimize && (config.remove_assignments || config.backprop_in_place ||  config.propagate_in_place)) {
    VariableMergingOptimization(config, nnet, computation);
  }



  
  // -------------- Optimize-3.cpp ----------------
  if (config.optimize && (config.snip_row_ops || config.optimize_row_ops)) {
    bool must_renumber = false;
    // 削减不必要的Row操作
    if (config.snip_row_ops && SnipRowOps(computation))
      must_renumber = true;
    // 将某些特殊的Rows操作转化为 Matrix操作
    if (config.optimize_row_ops && ReplaceRowWithMatrixOps(computation))
      must_renumber = true;

    // 如果经过上面的处理确实有操作被修改, 需要重新更新 matrix submatrix 等的编号.
    if (must_renumber) {
      RenumberComputation(computation);
    }
  }

  // ----------------- Optimize-3.cpp part2 -----------------
  if (config.optimize && config.initialize_undefined) {
    RemoveUnnecessaryZeroing(nnet, computation);
  }

  // ----------------- Optimize-3.cpp part3 -----------------  
  if (config.optimize && config.move_sizing_commands) {
    MoveSizingCommands(nnet, computation);
  }



  
  // ------------------ Optimize-4.cpp ----------------
  // false.
  // 循环的计算优化 必须在 RemoveUnnecessaryAllocation之前
  // 这个循环的意思是 让computation的 command 构成一个循环, 只保留两个 seg -- NnetExample.
  // 然后形成循环结构.
  if (config.optimize_looped_computation){
    OptimizeLoopedComputation(nnet, computation);
  }


  // ------------------ Optimize-5.cpp ----------------
  if (config.optimize && config.allocate_from_other &&
      !config.optimize_looped_computation) {
    // 如果构成循环结构, 这个就不要调用, 因为我们不确定是否在循环命令结构下 是正确的.
    // 无论如何 优化效率都很小.
    RemoveUnnecessaryAllocation(nnet, computation);
  }


  // ------------------ Optimize-5.cpp part2 ----------------
  // 这个是不能配置的必须调用的
  // 为了让computation 运算正确, 我们经过编译时也要调用一下
  // 但是经过其他优化操作之后 已经可能已经被淘汰了.
  ConsolidateIoOperations(nnet, computation);


  // 如果经过了 循环命令结构优化, 需要在进行一下修正GotoLabel命令的目标地点.
  if (config.optimize_looped_computation)
    FixGotoLabel(computation);

  if (GetVerboseLevel() >= 3) {
    CheckComputation(nnet, *computation, false);
    KALDI_LOG << "After optimization, max memory use (bytes) = "
              << GetMaxMemoryUse(*computation);
  }
}



