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
