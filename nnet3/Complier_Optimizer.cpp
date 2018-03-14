// Options class for optimizing a NnetComputation.  The main projected use for
// this is in debugging the optimization code itself, so that if an error is
// detected, we can work out which optimization was responsible for the error.
struct NnetOptimizeOptions {
  bool optimize;  // setting this false disallow all optimization.
  bool consolidate_model_update;
  bool propagate_in_place;
  bool backprop_in_place;
  bool optimize_row_ops;
  bool convert_addition;
  bool remove_assignments;
  bool allow_left_merge;
  bool allow_right_merge;
  bool initialize_undefined;
  bool move_sizing_commands;
  bool allocate_from_other;
  int32 min_deriv_time;
  int32 max_deriv_time;
  int32 max_deriv_time_relative;
  bool snip_row_ops;
  // optimize_looped_computation is a 'hidden config' not available from
  // the command line; it's set to true to enable the optimization for
  // looped computation that turns a linear computation into a loop.
  bool optimize_looped_computation;

  NnetOptimizeOptions():
      optimize(true),
      consolidate_model_update(true),
      propagate_in_place(true),
      backprop_in_place(true),
      optimize_row_ops(true),
      convert_addition(true),
      remove_assignments(true),
      allow_left_merge(true),
      allow_right_merge(true),
      initialize_undefined(true),
      move_sizing_commands(true),
      allocate_from_other(true),
      min_deriv_time(std::numeric_limits<int32>::min()),
      max_deriv_time(std::numeric_limits<int32>::max()),
      max_deriv_time_relative(std::numeric_limits<int32>::max()),
      snip_row_ops(true),
      optimize_looped_computation(false) { }

  void Register(OptionsItf *opts) {
    opts->Register("optimize", &optimize, "Set this to false to turn off all "
                 "optimizations");
    opts->Register("consolidate-model-update", &consolidate_model_update,
                   "Set to false to disable optimization that consolidates "
                   "the model-update phase of backprop (e.g. for recurrent "
                   "architectures");
    opts->Register("propagate-in-place", &propagate_in_place, "Set to false to "
                   "disable optimization that allows in-place propagation");
    opts->Register("backprop-in-place", &backprop_in_place, "Set to false to "
                   "disable optimization that allows in-place backprop");
    opts->Register("optimize-row-ops", &optimize_row_ops, "Set to false to "
                   "disable certain optimizations that act on operations of "
                   "type *Row*.");
    opts->Register("convert-addition", &convert_addition, "Set to false to "
                   "disable the optimization that converts Add commands into "
                   "Copy commands wherever possible.");
    opts->Register("remove-assignments", &remove_assignments, "Set to false to "
                   "disable optimization that removes redundant assignments");
    opts->Register("allow-left-merge", &allow_left_merge, "Set to false to "
                   "disable left-merging of variables in remove-assignments "
                   "(obscure option)");
    opts->Register("allow-right-merge", &allow_right_merge, "Set to false to "
                   "disable right-merging of variables in remove-assignments "
                   "(obscure option)");
    opts->Register("initialize-undefined", &initialize_undefined, "Set to false "
                   "to disable optimization that avoids redundant zeroing");
    opts->Register("move-sizing-commands", &move_sizing_commands, "Set to false "
                   "to disable optimization that moves matrix allocation and "
                   "deallocation commands to conserve memory.");
    opts->Register("allocate-from-other", &allocate_from_other, "Instead of "
                   "deleting a matrix of a given size and then allocating "
                   "a matrix of the same size, allow re-use of that memory");
    opts->Register("min-deriv-time", &min_deriv_time, "You can set this to "
                   "the minimum t value that you want derivatives to be computed "
                   "at when updating the model.  This is an optimization that "
                   "saves time in the backprop phase for recurrent frameworks");
    opts->Register("max-deriv-time", &max_deriv_time, "You can set this to "
                   "the maximum t value that you want derivatives to be computed "
                   "at when updating the model.  This is an optimization that "
                   "saves time in the backprop phase for recurrent frameworks");
    opts->Register("max-deriv-time-relative", &max_deriv_time_relative,
                   "An alternative mechanism for setting the --max-deriv-time, "
                   "suitable for situations where the length of the egs is "
                   "variable.  If set, it is equivalent to setting the "
                   "--max-deriv-time to this value plus the largest 't' value "
                   "in any 'output' node of the computation request.");
    opts->Register("snip-row-ops", &snip_row_ops, "Set this to false to "
                   "disable an optimization that reduces the size of certain "
                   "per-row operations");
  }
  void Read(std::istream &is, bool binary);
  void Write(std::ostream &os, bool binary) const;
  bool operator == (const NnetOptimizeOptions &other) const;
};

struct NnetComputeOptions {
  bool debug;
  NnetComputeOptions(): debug(false) { }
  void Register(OptionsItf *opts) {
    opts->Register("debug", &debug, "If true, turn on "
                   "debug for the neural net computation (very verbose!) "
                   "Will be turned on regardless if --verbose >= 5");
  }

};

struct CachingOptimizingCompilerOptions {
  bool use_shortcut;
  int32 cache_capacity;

  CachingOptimizingCompilerOptions():
      use_shortcut(true),
      cache_capacity(64) { }

  void Register(OptionsItf *opts) {
    opts->Register("use-shortcut", &use_shortcut,
                   "If true, use the 'shortcut' in compilation whereby "
                   "computation requests with regular structure are identified "
                   "as such, a computation with a smaller number of distinct "
                   "values of 'n' is compiled (e.g. 2), and the compiled "
                   "computation is expanded to match the size of the real "
                   "computation request.");
    opts->Register("cache-capacity", &cache_capacity,
                   "Determines how many computations the computation-cache will "
                   "store (most-recently-used).");
  }
};

struct NnetComputeProbOptions {
  bool debug_computation;
  bool compute_deriv;
  bool compute_accuracy;
  // note: the component stats, if stored, will be stored in the derivative nnet
  // (c.f. GetDeriv()) if compute_deriv is true; otherwise, you should use the
  // constructor of NnetComputeProb that takes a pointer to the nnet, and the
  // stats will be stored there.
  bool store_component_stats;
  
  bool compute_per_dim_accuracy;


  // 见上面
  NnetOptimizeOptions optimize_config;
  NnetComputeOptions compute_config;
  CachingOptimizingCompilerOptions compiler_config;

  NnetComputeProbOptions():
      debug_computation(false),
      compute_deriv(false),
      compute_accuracy(true),
      store_component_stats(false),
      compute_per_dim_accuracy(false) { }

  
  void Register(OptionsItf *opts) {
    // compute_deriv is not included in the command line options
    // because it's not relevant for nnet3-compute-prob.
    // store_component_stats is not included in the command line
    // options because it's not relevant for nnet3-compute-prob.
    opts->Register("debug-computation", &debug_computation, "If true, turn on "
                   "debug for the actual computation (very verbose!)");
    opts->Register("compute-accuracy", &compute_accuracy, "If true, compute "
                   "accuracy values as well as objective functions");
    opts->Register("compute-per-dim-accuracy", &compute_per_dim_accuracy,
                   "If true, compute accuracy values per-dim");

    // register the optimization options with the prefix "optimization".
    ParseOptions optimization_opts("optimization", opts);
    optimize_config.Register(&optimization_opts);
    // register the compiler options with the prefix "compiler".
    ParseOptions compiler_opts("compiler", opts);
    compiler_config.Register(&compiler_opts);
    // register the compute options with the prefix "computation".
    ParseOptions compute_opts("computation", opts);
    compute_config.Register(&compute_opts);
  }
};




/** This class is for computing cross-entropy and accuracy values in a neural
    network, for diagnostics.
    Note: because we put a "logsoftmax" component in the nnet, the actual
    objective function becomes linear at the output, but the printed messages
    reflect the fact that it's the cross-entropy objective.
 */
class NnetComputeProb {
 public:
  // does not store a reference to 'config' but does store one to 'nnet'.
  NnetComputeProb(const NnetComputeProbOptions &config,
                  const Nnet &nnet);

  NnetComputeProb::NnetComputeProb(const NnetComputeProbOptions &config,
                                   const Nnet &nnet):
      config_(config),
      nnet_(nnet),
      deriv_nnet_owned_(true),
      deriv_nnet_(NULL),
      
      // ====================  构造 CachingOptimizingCompiler compiler_ =============== ;
      //      config  -- 是 NnetComputeProbOptions类型. 而optimize_config compiler_config 都是成员, 分别是 优化 编译配置.
      // 1 nnet
      // 2 优化配置选项
      // NnetOptimizeOptions  optimize_config
      // 3 编译配置选项
      // CachingOptimizingCompilerOptions compiler_config 都是默认构造的.
      compiler_(nnet, config_.optimize_config, config_.compiler_config),
      num_minibatches_processed_(0)
      
  {
    if (config_.compute_deriv) {
      deriv_nnet_ = new Nnet(nnet_);
      ScaleNnet(0.0, deriv_nnet_);
      SetNnetAsGradient(deriv_nnet_); // force simple update
    } else if (config_.store_component_stats) {
      KALDI_ERR << "If you set store_component_stats == true and "
                << "compute_deriv == false, use the other constructor.";
    }
  }
  

  // This version of the constructor may only be called if
  // config.store_component_stats == true and config.compute_deriv == false;
  // it means it will store the component stats in 'nnet'.  In this
  // case you should call ZeroComponentStats(nnet) first if you want
  // the stats to be zeroed first.
  NnetComputeProb(const NnetComputeProbOptions &config,
                  Nnet *nnet);


  // Reset the likelihood stats, and the derivative stats (if computed).
  void Reset();

  // compute objective on one minibatch.
  void Compute(const NnetExample &eg);

  // Prints out the final stats, and return true if there was a nonzero count.
  bool PrintTotalStats() const;

  // returns the objective-function info for this output name (e.g. "output"),
  // or NULL if there is no such info.
  const SimpleObjectiveInfo *GetObjective(const std::string &output_name) const;

  // This function returns the total objective over all output nodes recorded here, and
  // outputs to 'tot_weight' the total weight (typically the number of frames)
  // corresponding to it.
  double GetTotalObjective(double *tot_weight) const;

  // if config.compute_deriv == true, returns a reference to the
  // computed derivative.  Otherwise crashes.
  const Nnet &GetDeriv() const;

  ~NnetComputeProb();
 private:
  void ProcessOutputs(const NnetExample &eg,
                      NnetComputer *computer);

  NnetComputeProbOptions config_;
  const Nnet &nnet_;

  bool deriv_nnet_owned_;
  Nnet *deriv_nnet_;
  CachingOptimizingCompiler compiler_;

  // this is only for diagnostics.
  int32 num_minibatches_processed_;

  unordered_map<std::string, SimpleObjectiveInfo, StringHasher> objf_info_;

  unordered_map<std::string, PerDimObjectiveInfo, StringHasher> accuracy_info_;
};




void NnetComputeProb::Compute(const NnetExample &eg) {
  bool
      need_model_derivative = config_.compute_deriv,  // false
      store_component_stats = config_.store_component_stats;  // false

  // 计算目标request
  ComputationRequest request;
  
  GetComputationRequest(nnet_, eg, need_model_derivative,
                        store_component_stats,
                        &request);
  const NnetComputation *computation = compiler_.Compile(request);
  NnetComputer computer(config_.compute_config, *computation,
                        nnet_, deriv_nnet_);
  // give the inputs to the computer object.
  computer.AcceptInputs(nnet_, eg.io);
  computer.Run();
  this->ProcessOutputs(eg, &computer);
  if (config_.compute_deriv)
    computer.Run();
}

void NnetComputeProb::ProcessOutputs(const NnetExample &eg,
                                     NnetComputer *computer) {
  std::vector<NnetIo>::const_iterator iter = eg.io.begin(),
      end = eg.io.end();
  for (; iter != end; ++iter) {
    const NnetIo &io = *iter;
    int32 node_index = nnet_.GetNodeIndex(io.name);
    if (node_index < 0)
      KALDI_ERR << "Network has no output named " << io.name;
    ObjectiveType obj_type = nnet_.GetNode(node_index).u.objective_type;
    if (nnet_.IsOutputNode(node_index)) {
      const CuMatrixBase<BaseFloat> &output = computer->GetOutput(io.name);
      if (output.NumCols() != io.features.NumCols()) {
        KALDI_ERR << "Nnet versus example output dimension (num-classes) "
                  << "mismatch for '" << io.name << "': " << output.NumCols()
                  << " (nnet) vs. " << io.features.NumCols() << " (egs)\n";
      }
      {
        BaseFloat tot_weight, tot_objf;
        bool supply_deriv = config_.compute_deriv;
        ComputeObjectiveFunction(io.features, obj_type, io.name,
                                 supply_deriv, computer,
                                 &tot_weight, &tot_objf);
        SimpleObjectiveInfo &totals = objf_info_[io.name];
        totals.tot_weight += tot_weight;
        totals.tot_objective += tot_objf;
      }
      // May not be meaningful in non-classification tasks
      if (config_.compute_accuracy) {  
        BaseFloat tot_weight, tot_accuracy;
        PerDimObjectiveInfo &acc_totals = accuracy_info_[io.name];

        if (config_.compute_per_dim_accuracy && 
            acc_totals.tot_objective_vec.Dim() == 0) {
          acc_totals.tot_objective_vec.Resize(output.NumCols());
          acc_totals.tot_weight_vec.Resize(output.NumCols());
        }

        ComputeAccuracy(io.features, output,
                        &tot_weight, &tot_accuracy,
                        config_.compute_per_dim_accuracy ? 
                          &acc_totals.tot_weight_vec : NULL,
                        config_.compute_per_dim_accuracy ? 
                          &acc_totals.tot_objective_vec : NULL);
        acc_totals.tot_weight += tot_weight;
        acc_totals.tot_objective += tot_accuracy;
      }
    }
  }
  num_minibatches_processed_++;
}






//  这个类, 能够在一个调用中执行 Compilation 和 Optimization操作
//  并确保当 ComputationRequest没变时, 不会重复进行编译过程.
/// This class enables you to do the compilation and optimization in one call,
/// and also ensures that if the ComputationRequest is identical to the previous
/// one, the compilation process is not repeated.
class CachingOptimizingCompiler {
 public:
  CachingOptimizingCompiler(const Nnet &nnet,
                            const CachingOptimizingCompilerOptions config =
                            CachingOptimizingCompilerOptions());

  // 注意 nnet 通过 const引用传递. opt_config 会被cp下来???
  /// Note: nnet is retained as a const reference but opt_config is copied.
  CachingOptimizingCompiler(const Nnet &nnet,
                            const NnetOptimizeOptions &opt_config,
                            const CachingOptimizingCompilerOptions config =
                            CachingOptimizingCompilerOptions());

  CachingOptimizingCompiler::CachingOptimizingCompiler(
    const Nnet &nnet,
    const NnetOptimizeOptions &opt_config,
    const CachingOptimizingCompilerOptions config):
    nnet_(nnet),              //  nnet- 文件 (这时候已经是final.config定义的raw了??????)
    config_(config),          //  CompilerOptions
    opt_config_(opt_config),  //  OptimizeOptions
    seconds_taken_total_(0.0), seconds_taken_compile_(0.0),
    seconds_taken_optimize_(0.0), seconds_taken_expand_(0.0),
    seconds_taken_check_(0.0), seconds_taken_indexes_(0.0) { }
  

  ~CachingOptimizingCompiler();
  /// Does the compilation and returns a const pointer to
  /// the result, which is owned by this class, not the caller.
  /// It calls ComputeCudaIndexes() for you, because you wouldn't
  /// be able to do this on a const object.
  const NnetComputation* Compile(const ComputationRequest &request);
  void ReadCache(std::istream &is, bool binary);
  void WriteCache(std::ostream &os, bool binary) const;
 private:

  // This function just implements the work of Compile(); it's made a separate
  // function for the convenience of the timer code, to avoid it being called
  // twice (we also call this function directly from inside the class).
  const NnetComputation* CompileInternal(const ComputationRequest &request);

  // This function, called from CompileInternal(), is called when a
  // ComputationRequest has been determined not to have already been cached.  It
  // otherwise has the same interface as CompileInternal(), but assumes that
  // there is nothing cached for this computation as yet.  It compiles the
  // computation and takes care of caching it.
  const NnetComputation* CompileAndCache(const ComputationRequest &request);


  // This function, called from CompileAndCache(), tries to compile the
  // ComputationRequest 'request' via 'shortcut' compilation; if this is
  // possible, it returns a pointer to a newly allocated computation that it has
  // compiled this way (note: this computation will not yet have been placed in
  // the computation cache).  If this is not possible for some reason
  // (e.g. shortcut compilation is disabled in the config; or the computation
  // request was not decomposable because of too few n values or irregular or
  // unexpected structure), this function returns NULL and you should compile
  // via CompileNoShortcut.
  const NnetComputation* CompileViaShortcut(const ComputationRequest &request);

  // This function, called from CompileAndCache(), tries to compile the
  // ComputationRequest 'request' via the regular (not shortcut) compilation
  // process; it returns a pointer to a newly allocated computation that it has
  // compiled this way (note: this computation will not yet have been placed in
  // the computation cache).
  const NnetComputation* CompileNoShortcut(const ComputationRequest &request);

  const Nnet &nnet_;
  CachingOptimizingCompilerOptions config_;
  NnetOptimizeOptions opt_config_;

  // The access queue for keeping track of the freshness of computation.
  // Most-recently-accessed computation is at the end, and
  // least-recently-accessed computaiton is at the beginning.
  // Together with computation_cache_, this forms a most-recently-used (MRU)
  // cache for Computations, indexed by ComputationRequest. Pointers
  // are owned in computation_cache_.
  typedef std::list<const ComputationRequest*> AqType;
  AqType access_queue_;

  // Map from computation-request to pair of (computation, and position in
  // access_queue_). Used for fast lookup of previously compiled computations.
  // All pointers are owned here.
  typedef unordered_map<const ComputationRequest*,
                        std::pair<const NnetComputation*, AqType::iterator>,
                        ComputationRequestHasher,
                        ComputationRequestPtrEqual> CacheType;
  CacheType computation_cache_;

  // 在编译的不同阶段 耗费的时间.
  // 为了诊断输出信息
  // seconds spent in various phases of compilation-- for diagnostic messages
  double seconds_taken_total_;
  double seconds_taken_compile_;
  double seconds_taken_optimize_;
  double seconds_taken_expand_;
  double seconds_taken_check_;
  double seconds_taken_indexes_;

  // This function updates the computation cache. It is called within
  // CompileInternal().  It takes ownership of the pointers.  It inserts the
  // request at the end of the queue, and purges the least-recently-accessed
  // request from the queue and the cache if the capacity is reached.
  void UpdateCache(const ComputationRequest *request,
                   const NnetComputation *computation);
  // This function updates the recently accessed queue.
  void UpdateAccessQueue(CacheType::iterator &cit);
};

