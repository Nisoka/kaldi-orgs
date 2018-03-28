

ParseOptions::ParseOptions(const std::string &prefix,
                           OptionsItf *other):
    print_args_(false), help_(false), usage_(""), argc_(0), argv_(NULL) {
  ParseOptions *po = dynamic_cast<ParseOptions*>(other);
  if (po != NULL && po->other_parser_ != NULL) {
    // we get here if this constructor is used twice, recursively.
    other_parser_ = po->other_parser_;
  } else {
    other_parser_ = other;
  }
  if (po != NULL && po->prefix_ != "") {
    prefix_ = po->prefix_ + std::string(".") + prefix;
  } else {
    prefix_ = prefix;
  }
}



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




//  这个类, 能够在一个调用中执行 Compilation 和 Optimization操作
//  并确保当 ComputationRequest没变时, 不会重复进行编译过程 (Cache机制).
class CachingOptimizingCompiler {
 public:
  CachingOptimizingCompiler(const Nnet &nnet,
                            const CachingOptimizingCompilerOptions config =
                            CachingOptimizingCompilerOptions());

  // 注意 nnet 通过 const引用传递. opt_config 会被cp下来???
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
