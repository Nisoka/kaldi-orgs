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
    if (GetVerboseLevel() >= 3)
      CheckComputation(nnet, *computation, false);
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
      if (GetVerboseLevel() >= 3)
        CheckComputation(nnet, *computation, false);
    }
  }


  if (config.optimize && config.initialize_undefined) {
    RemoveUnnecessaryZeroing(nnet, computation);
    if (GetVerboseLevel() >= 3)
      CheckComputation(nnet, *computation, false);
  }

  
  if (config.optimize && config.move_sizing_commands) {
    MoveSizingCommands(nnet, computation);
    if (GetVerboseLevel() >= 3)
      CheckComputation(nnet, *computation, false);
  }



  // ------------------ Optimize-4.cpp ----------------
  // the looped computation optimization has to go before
  // 'RemoveUnnecessaryAllocation()'.  We don't gate this by 'config.optimize'
  // because it's necessary for looped computation to run.
  if (config.optimize_looped_computation){
    OptimizeLoopedComputation(nnet, computation);
    if (GetVerboseLevel() >= 3)
      CheckComputation(nnet, *computation, false);
  }

  if (config.optimize && config.allocate_from_other &&
      !config.optimize_looped_computation) {
    // Don't do this if it's an looped computation because we're not sure if it
    // would be correct in that case, as written.  In any case the performance
    // benefit is tiny.
    RemoveUnnecessaryAllocation(nnet, computation);
    if (GetVerboseLevel() >= 3)
      CheckComputation(nnet, *computation, false);
  }

  // The following is not configurable because it is necessary for
  // the computation to run correctly (we do it after compilation too,
  // but the operations may have been put out of order by
  // other optimizations.)
  ConsolidateIoOperations(nnet, computation);

  if (config.optimize_looped_computation)
    FixGotoLabel(computation);

  if (GetVerboseLevel() >= 3) {
    CheckComputation(nnet, *computation, false);
    KALDI_LOG << "After optimization, max memory use (bytes) = "
              << GetMaxMemoryUse(*computation);
  }
}



