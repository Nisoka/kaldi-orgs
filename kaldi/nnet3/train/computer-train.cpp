{
  NnetComputer computer(config_.compute_config, *computation, nnet_, deriv_nnet_);
  
  // give the inputs to the computer object.
  computer.AcceptInputs(nnet_, eg.io);
  
  computer.Run();
  this->ProcessOutputs(eg, &computer);
  
  if (config_.compute_deriv)
    computer.Run();
}

