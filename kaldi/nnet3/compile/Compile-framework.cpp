
Compiler::Compiler(
    const ComputationRequest &request,
    const Nnet &nnet): nnet_(nnet) {
  // 构造Compiler 时, 将request 加入了 内部的 request_ 队列中, 但是一般实际上就只有一个request.
  requests_.push_back(&request);
}
// ============================== Main ====================================
// ============================== Main ====================================
// ============================== Main ====================================
// 根据 requests_ 构建 Computation.
void Compiler::CreateComputation(const CompilerOptions &opts,
                                 NnetComputation *computation) {




  // ***** Target computation *****
  computation->Clear();



  


  // ================ part1 构建ComputationGraph Compute Cindex computable  ================
  // -------- use the ComputationGraphBuilder build the graph_ ---------

  
  ComputationGraphBuilder builder(nnet_, &graph_);
  
  // 为每个request_ 构建 ComputationGraph -- cindexes 依赖以及计算性.
  for (size_t segment = 0; segment < requests_.size(); segment++) {

    // 从output cindexes  开始计算依赖cindexes , 然后计算所有Cindexes 的 可计算性.
    builder.Compute(*(requests_[segment]));

    
    // 根据可计算性, 判断output节点是否都可计算, output节点都可计算则 可计算.
    if (!builder.AllOutputsAreComputable()) {
      builder.ExplainWhyAllOutputsNotComputable();  // prints logging info
      KALDI_ERR << "Not all outputs were computable, cannot create computation.";
    }
    
    // 剪枝掉 无用cindexes.
    builder.Prune();
  }






  // ================  part2 为ComputationGraph 中的 cindex 计算 phase 计算次序  ===============
  // 一个phase 会被分解为 一个或多个steps.
  // 对每个segment phase_per_segment 是phase计算次序 list,
  //               每个phase计算次序 都保存了该次序应该进行计算的cindex_ids
  std::vector<std::vector<std::vector<int32> > > phases_per_segment;
  ComputeComputationPhases(nnet_, graph_, &phases_per_segment);







  // =============== part3  将获得的phases_per_segment 转化为 steps.
  // steps_ 顺序保存 每个segment的 每个phase的 每个sub_phase的 所有cindexes.
  std::vector<std::vector<int32> > steps;
  steps.reserve(1000);
  // 将每个 step 映射划分为segment
  // <0,0,0,0, 1,1,1,1,1,1,1, 2,2,2,2,2 .... >
  std::vector<int32> step_to_segment;

  {
    // ComputationStepsComputer 会输出steps cindex_id_to_location_
    // 可能会增加一些cindexes 改变graph_
    // cindex_id_to_location_ 就是 ComputationStepsComputer->locations_
    ComputationStepsComputer steps_computer(nnet_, &graph_, &steps, &cindex_id_to_location_);
    
    // foreach request_, phases_per_segment
    for (size_t segment = 0; segment < requests_.size(); segment++) {
      // ===================== 根据reqeust 和 phase计算次序的 cindex_ids 计算steps ===============
      steps_computer.ComputeForSegment(*(requests_[segment]), phases_per_segment[segment]);
      // 
      while (step_to_segment.size() < steps.size())
        step_to_segment.push_back(segment);
      // 节省空间.
      std::vector<std::vector<int32> > temp;
      phases_per_segment[segment].swap(temp);
    }
    steps_computer.Check();
  }





  // ============ part4   生成StepInfo的过程 ==============
  // 1 计算step的依赖关系dep_steps
  // 2 依赖需要导数, 那么step 也需要导数, 结果得到每个 step是否需要导数.
  // - deriv_needed - 每个step 是否需要导数
  std::vector<bool> deriv_needed;
  ComputeDerivNeeded(steps, step_to_segment, &deriv_needed);



  // ============ 创建StepInfo,
  // in:
  // deriv_needed     每个step 是否需要计算导数
  // step_to_segment  每个step属于那个request <0,0,0,0,0, 1,1,1,1,1, 2,2,2,2, request-cnt,request-cnt,>
  // steps            每个phase的每个子sub_phase(phase中相同node-index) 的vector
  // computation      目标计算computation
  CreateStepInfo(deriv_needed, step_to_segment, &steps, computation);




  
  // =========== part5 向computation中增加 命令Commands ===========
  AddCommands(deriv_needed, step_to_segment, computation);


  // =========== 如下函数 从新安排添加的command的顺序。
  // 所以kAcceptInput 和 kProvideOutput命令 会出现在正确位置上.
  ConsolidateIoOperations(nnet_, computation);

  
  if (opts.output_debug_info)
    OutputDebugInfo(computation);
  
}
