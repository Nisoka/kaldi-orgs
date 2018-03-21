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

// 这个累用来计算交叉熵 和 准确率, 为了诊断.
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



// 根据当前eg, 构建一个 ComputationRequest
// input output 都是 NnetIo, 都是来自eg中的Vector<NnetIo> 根据内部name判断所属nnet node 是否是 input output来构建.
void GetComputationRequest(const Nnet &nnet,
                           const NnetExample &eg,
                           bool need_model_derivative,
                           bool store_component_stats,
                           ComputationRequest *request) {

  // reserve是容器预留空间，但并不真正创建元素对象，在创建对象之前，不能引用容器内的元素，因此当加入新的元素时，需要用push_back()/insert()函数。
  // resize是改变容器的大小，并且创建对象，因此，调用这个函数之后，就可以引用容器内的对象了，因此当加入新的元素时，用operator[]操作符，或者用迭代器来引用元素对象。
  request->inputs.clear();
  // request目标计算 是以eg为输入,
  // 输出结果也保持eg.io.size()??
  request->inputs.reserve(eg.io.size());
  request->outputs.clear();
  request->outputs.reserve(eg.io.size());

  // false
  request->need_model_derivative = need_model_derivative;
  // false
  request->store_component_stats = store_component_stats;

  // foreach NnetIo (经过汇总的内部 indexes n t都变化)
  for (size_t i = 0; i < eg.io.size(); i++) {
    const NnetIo &io = eg.io[i];
    const std::string &name = io.name;
    // 获得对应的node_index, 根据name判断, 返回对应node的index.
    int32 node_index = nnet.GetNodeIndex(name);

    // 等于-1 且 不是input 且 不是 output节点时
    if (node_index == -1 && !nnet.IsInputNode(node_index) && !nnet.IsOutputNode(node_index))
      KALDI_ERR << "Nnet example has input or output named '" << name
                << "', but no such input or output node is in the network.";

    // inputs 是一个NnetIo向量, 内部包含多个NnetIo NnetIo保存的是输入样本.
    // 判断是否是input节点, 是的话dest = request->inputs.
    std::vector<IoSpecification> &dest =
        nnet.IsInputNode(node_index) ? request->inputs : request->outputs;
    
    dest.resize(dest.size() + 1);
    IoSpecification &io_spec = dest.back();
    io_spec.name = name;
    io_spec.indexes = io.indexes;
    io_spec.has_deriv = nnet.IsOutputNode(node_index) && need_model_derivative;
  }
  
  // check to see if something went wrong.
  if (request->inputs.empty())
    KALDI_ERR << "No inputs in computation request.";
  if (request->outputs.empty())
    KALDI_ERR << "No outputs in computation request.";
}




/*
   本函数发现并返回Indexes向量的n_stride规则.
   或者返回0,如果并没有什么好的规则结构存在.
   This function finds and returns the 'n-stride' of the vector of Indexes, or
   returns 0 if it does not exist because the Indexes lack the required regular
   structure.
   这个函数会影响shortcut编译, 被用在IoSpecificationIsDecomposable中.
   This function relates to 'shortcut' compilation and is used in
   class IoSpecificationIsDecomposable().  There is an overloaded version of
   this function that works with Cindex input, that has almost exactly
   the same code.

   本函数用于发现 indexes向量中的规则结构.
   我们对indexes中的n域最感兴趣, 所以想要求出 n 的规则.
   我们希望indexes具有n具有 0,1 2... N-1的形式??
   It is used to discover regular structure in vectors of indexes.  We are
   interested in the structure on the 'n' index; in particular, the stride on
   the 'n' index.  We expect the vector 'indexes' to contain 'n' values of the
   form 0, 1, ... N-1 (where the value of N can be obtained easily by looking at
   the .n value of the last element of 'indexes').
   // 并且我们希望Indexes的n值 有一个固定分隔的规则.????
   And we expect the 'n' values of Indexes that are otherwise the same to be separated by a fixed stride,
   which we will return.

   如果stride 是不一致的 或者我们其他需求中的一个不满足, 就直接返回0
   如果是一直一致的并且要求都满足, 我们才返回该stride.
   If the stride is inconsistent or one of our other requirements (see below) is
   not fulfilled, we will return 0.  If it's always consistent and our
   requirements are fulfilled we'll return the stride.

   // 如果full_check is true, 我们执行详细的check 来检验一致性, 否则我们就随机check检查一下.
   If 'full_check' is true we do an exhaustive check for consistency; otherwise we do a randomized
   check.

   // 一致性的完整定义如下:
   The full definition of 'consistency' is as follows:

   // 正常的规则结构 n_stride>=1 并且N 是n 值总数
   For some n_stride >= 1 (which we'll return), and with N as the number of
   'n' values (which should be numbered 0, 1, ... N-1):

   对于一个Index 其 n < N-1, 位置在i. 那么另一个Index带有大于刚刚的n值, 而其他的t x域都相同的话,
   该Index的位置 必然是 i+ n_stride.
     - For any Index with n < N-1 located at position i, an Index with one
       greater 'n' but otherwise the same must exist at position i + n_stride

     - For any Index with n > 0 located at position i, an Index with one
       smaller 'n' but otherwise the same must exist at position i - n_stride.

       输入必须按照blocks方式被安排, block 具有 block_size=n_stride*N, 其中这些strides从来不会交叉???.
       block_size 会根据 n 不同而发生变化.
     - The input must be arranged in blocks of size block_size = n_stride * N,
       which these strides never cross.

       "Strides never cross" is an informal definition:
       we can formalize this by saying that for an Index with n == 0
       at position i, we must have (i / block_size) == ((i + n_stride*(N-1)) /
       block_size), with integer division.

   // 上面的条件暗示, 输入的size必须是 正比与n-stride的.
   The above conditions imply that the size of the input must be a multiple
   of the n-stride.

   Reminder: we return 0 if the regular structure is not found, and the n-stride
   if the regular structure is found.
*/
//  ===== 返回 frames_per_eg.
static int32 FindNStride(const std::vector<Index> &indexes,
                         bool full_check) {
  // 首先找到候选stride, 然后检查一致性
  // First find candidate stride.  Later we'll check for consistency.
  // size 所有输入样本数量N*frames_per_eg.
  int32 size = indexes.size();
  KALDI_ASSERT(size > 0);
  // indexes[i].n 表示来自第几个eg. N表示来总体有多个eg, 一般是minibatch.
  // indexes[i].t 表示某个eg中的第t帧. 那么每个indexes 还是表示的是一帧数据.
  int32 N = indexes[size-1].n + 1,
      n_stride;
  if (N <= 1) {
    // we wouldn't be able to determine the stride if N <= 1.
    return 0;
  }
  
  Index index(indexes[0]);
  if (index.n != 0 || size % N != 0) {
    // for the n stride to be positive, we must start with an index with n == 0.
    // if indexes.size() is not divisible by N, we have no hope of finding the
    // regular structure.
    return 0;
  }
  
  index.n = 1;
  
  // 这个stride 貌似就是描述的是 frames_per_eg.  merged_eg.indexes中来自某个eg的indexes中的t变化.
  // 一般就是返回的是 size/N. N是来自汇总的eg的总数
  // merged_eg.indexes--size 表示结果的帧总数.
  // 那么size/N 就是 frames_per_eg.
  // First check the two most common strides, which are 1 and size / N.
  if (indexes[1] == index) {
    n_stride = 1;
  } else if (indexes[size / N] == index) {
    n_stride = size / N;
  } else {
    int32 stride;
    // try the other possible strides one by one (for subsampling layers of convnets, we might see strides of 2, for instance).
    for (stride = 2; stride < size / N; stride++) {
      if (size % stride == 0 && indexes[stride] == index) {
        n_stride = stride;
        break;
      }
    }
    if (stride == size / N) {
      // if we fell off the loop then we found no candidates, which is strange
      // and means we did not find the expected structure; we'll return 0 as we
      // failed.
      return 0;
    }
  }


  
  // Now is the checking phase.
  // to understand block_size, see the comment above this functcion.
  int32 block_size = n_stride * N;

  // 如果full_check 对merged_eg.indexes的每帧都进行checck, 那么indexes_to_check 就需要设置为 indexes.size()大小.
  std::vector<int32> indexes_to_check;
  if (full_check) {
    indexes_to_check.resize(size);
    for (int32 i = 0; i < size; i++)
      indexes_to_check[i] = i;
  } else {
    int32 num_to_check = std::min<int32>(5, size);
    indexes_to_check.resize(num_to_check);
    for (int32 j = 0; j < num_to_check; j++)
      indexes_to_check[j] = RandInt(0, size - 1);
    SortAndUniq(&indexes_to_check);
  }


  // foreach frame-to-check
  for (std::vector<int32>::iterator iter = indexes_to_check.begin();
       iter != indexes_to_check.end(); ++iter) {
    int32 i = *iter;
    Index index = indexes[i];
    // 来自某个eg
    int32 n = index.n;
    // n_stride 就是frames_per_eg, 这样规则情况就是 每个eg占用8个空间,所以下面的check都正确
    if (n < N - 1) {
      index.n = n + 1;
      if (i + n_stride >= size || indexes[i + n_stride] != index)
        return 0;
    }
    
    if (n == 0) {
      if (i / block_size != (i + n_stride * (N-1)) / block_size) {
        // this is a check that the input divides into blocks of size n_stride *
        // N and the N different versions of the same Index are always within a
        // block (i.e. that the n stride never crosses over the block; having
        // the same Index repeated within different blocks actually would not
        // matter).
        return 0;
      }
    } else { // n > 0
      index.n = n - 1;
      if (i - n_stride < 0 || indexes[i - n_stride] != index)
        return 0;
    }
  }
  return n_stride;
}




// 就是增加 new_N - old_N 个indexes,
// 注意 index 是描述 数据位置的描述性对象, 不是数据, 又因为indexes都是有规则的,
// 所以下面 直接使用 一个n_stride 中的indexes 就生成了新的 Indexes.
/*
  
  // 转化 Indexes(其n值具有 0- old_N-1 范围)向量为 新的Indexes向量
  // t x 域都不变, 只在n域发生变化.
  This function, used in shortcut compilation, converts a vector of Indexes
  having a range of 'n' values (0, 1, ... old_N - 1), to a vector of
  Indexes that's otherwise the same, but has a different range of 'n' values
  (0, 1, ... new_N - 1).

  // 输入Indexes 希望具有 n_stride>0, 输出会保持这个n_stride.???
  The input vector is expected to have a stride 'n_stride > 0', as
  would be returned by FindNStride, and the output vector will be given the
  same n-stride.
 */
static void ConvertNumNValues(int32 n_stride, int32 old_N, int32 new_N,
                              const std::vector<Index> &indexes_in,
                              std::vector<Index> *indexes_out) {
  int32 size_in = indexes_in.size();
  KALDI_ASSERT(size_in > 0 && indexes_in[size_in - 1].n == old_N - 1);
  int32 block_size_in = n_stride * old_N,
      block_size_out = n_stride * new_N;

  // 大小扩大.
  indexes_out->resize((size_in / old_N) * new_N);
  for (int32 i_in = 0; i_in < size_in; i_in++) {
    if (indexes_in[i_in].n != 0)
      continue;
    Index index(indexes_in[i_in]);
    int32 block_index = i_in / block_size_in,
        offset_within_block = i_in % block_size_in;


    int32 i_out = block_index * block_size_out +
        offset_within_block;
    for (int32 n = 0; n < new_N; n++, i_out += n_stride) {
      index.n = n;
      (*indexes_out)[i_out] = index;
    }
  }
}

// 判断Io是否可计算
// out:
// 1 将汇总的NnetIo内 eg数量减少到2, 返回新的NnetIo
// 2 返回原本 汇总的eg总数.
// 这个函数 和 RequestIsDecomposable() 几乎一样, 不过不同于RequestIsDecomposable() 是为了求解ComputationRequest
// 这个函数是为了求解 IoSpecificationIsDecomposable().
// This helper function is used in RequestIsDecomposable(); you can work out
// what it does, and why, from the documentation of RequestIsDecomposable() in
// the header.  This function does basically the same thing, except
// at a lower level, for an IoSpecification rather than a ComputationRequest.
static bool IoSpecificationIsDecomposable(const IoSpecification &io_spec,
                                          IoSpecification *mini_io_spec,
                                          int32 *num_n_values_out) {
  mini_io_spec->name = io_spec.name;
  mini_io_spec->has_deriv = io_spec.has_deriv;
  // 对应的NnetIo内部的特征描述 indexes.
  const std::vector<Index> &indexes = io_spec.indexes;
  KALDI_ASSERT(!indexes.empty() && "Empty Indexes in computation request");

  bool full_check = true;  // We might eventually change this to false, for efficiency.
  
  // merged_eg.vector<NnetIo>.indexes.n 以0为起始.
  // indexes.back().n 表示merged_eg中最后一个样本+1.
  int32 num_n_values = indexes.back().n + 1;
  if (num_n_values <= 2) {
    // ???? 当Computations 只有2 或更少的值时, 不值得使用shortcut编译, 速度提升太小.
    // Computations with 2 or fewer 'n' values are not decomposable, as there
    // would be no speed benefit in shortcut compilation (which relies on
    // compiling an otherwise similar computation with n == 2).
    return false;
  }
  // ------------- 输出 io 汇总的eg的总数 ------------
  *num_n_values_out = num_n_values;

  // ------------- 返回的实际上n_stride 就是 frames_per_eg -----------
  int32 n_stride = FindNStride(indexes, full_check);

  if (n_stride == 0)
    return false;

  // ------------- 将原本indexes 描述的样本eg数量 减少到2个, 而每个eg的frames保持不变 -----------
  ConvertNumNValues(n_stride, num_n_values, 2,
                    indexes, &(mini_io_spec->indexes));

  return true;
}



void NnetComputeProb::Compute(const NnetExample &eg) {

  bool
      // 是否需要计算导数.
      need_model_derivative = config_.compute_deriv,  // false
      store_component_stats = config_.store_component_stats;  // false

  // 计算目标request
  ComputationRequest request;

  // ==================== 获得计算目标request ==================
  GetComputationRequest(nnet_, eg, need_model_derivative, store_component_stats, &request);


  // ====================== 根据 request 利用编译器 Compile 生成computation =================
  // 使用Compiler 编译request 生成一个computation.
  const NnetComputation *computation = compiler_.Compile(request);

  const NnetComputation* CachingOptimizingCompiler::Compile(const ComputationRequest  &in_request) {
    Timer timer;
    const NnetComputation *ans = CompileInternal(in_request);
    seconds_taken_total_ += timer.Elapsed();
    return ans;
  }

  const NnetComputation* CachingOptimizingCompiler::CompileInternal( const ComputationRequest  &in_request) {
    const NnetComputation *ans;
    // find computation in the cache
    CacheType::iterator cit = computation_cache_.find(&in_request);

    // 还没有缓存过该计算
    if (cit == computation_cache_.end()) {
      // ================================= 编译request获得计算Computation 并缓存起来 =================
      ans = CompileAndCache(in_request);

      // #########################################
      const NnetComputation* CachingOptimizingCompiler::CompileAndCache( const ComputationRequest  &in_request) {
        //需要构建一个ComputationRequest的copy, 因为需要加入到cache中做key使用.
        // we need to make a copy of ComputationRequest, because it's stored
        // as the key in the cache, and we need to own the pointer.
        ComputationRequest *request = new ComputationRequest(in_request);

        // 最短路径编译request
        const NnetComputation *computation = CompileViaShortcut(*request);
        const NnetComputation* CachingOptimizingCompiler::CompileViaShortcut( const ComputationRequest &request) {
          // !True
          if (!config_.use_shortcut)
            return NULL;


          // ============= 精简reqeust
          int32 num_n_values;
          ComputationRequest mini_request;

          if (!RequestIsDecomposable(request, &mini_request, &num_n_values))
            return NULL;
          // 将request精简了一下, 将汇总的多个eg的很多数据, 减少使用量, 只描述使用2个eg.
          bool RequestIsDecomposable(const ComputationRequest &request,
                                     ComputationRequest *mini_request,
                                     int32 *num_n_values) {
            size_t
                num_inputs = request.inputs.size(),
                num_outputs = request.outputs.size();
            // mini_request 的输入输出都与 原本的大小一样.
            mini_request->inputs.resize(num_inputs);
            mini_request->outputs.resize(num_outputs);
            mini_request->need_model_derivative = request.need_model_derivative;
            mini_request->store_component_stats = request.store_component_stats;
            mini_request->misc_info = request.misc_info;

            KALDI_ASSERT(num_inputs != 0 && num_outputs != 0);

            // 先判断每个 input 是否都可以获得
            for (size_t i = 0; i < num_inputs; i++) {
              int32 this_num_n_values = 0;

              // 利用IoSpecificationIsDecomposable 将Request 的每个Input都判断是否能够计算
              // 这个能够计算实际上也就是 减小了 input[i] 的数据数量 为2.
              // mini_request->input[i] 将原本 汇总的多个eg 减少到2个
              // this_num_n_values 返回原本汇总的eg 数量.
              if (!IoSpecificationIsDecomposable(request.inputs[i],
                                                 &(mini_request->inputs[i]),
                                                 &this_num_n_values))
                return false;
  
              if (i == 0) {
                // 原本汇总的eg数量
                *num_n_values = this_num_n_values;
              } else {
                if (this_num_n_values != *num_n_values)
                  return false;  // .. which would be odd.
              }
              
            }

            // 
            for (size_t i = 0; i < num_outputs; i++) {
              int32 this_num_n_values = 0;
              if (!IoSpecificationIsDecomposable(request.outputs[i],
                                                 &(mini_request->outputs[i]),
                                                 &this_num_n_values))
                return false;
              
              if (this_num_n_values != *num_n_values)
                return false;  // .. which would be odd.
            }
            return true;
          }

          // 通过对 mini request 调用CompileInternel()
          // 精简过的 mini_request 在递归的CompileInternel中 进入CompileViaShortcut.后会返回NULL
          // 转而去调用 CompileNoShortcut 进行编译过程.
          // 然后通过编译好的mini_computation 进行扩展得到原本 汇总较多eg的 标准request的Computation.
          // By invoking CompileInternal() on the mini request, we go through the same
          // caching process as for any externally requested computation.  [the only
          // difference from Compile() is that it doesn't call the timer code; this
          // avoids double-counting the time taken.]

          // this pointer will not have to be deleted by this function; it's owned by the class, in the cache.
          const NnetComputation *mini_computation = CompileInternal(mini_request);

          // note: by default we always create debug_info, even in regular compilation.
          // (e.g. it defaults to true in CompilerOptions).  If it really seems to be a
          // significant overhead, we can revisit this at some point in future.
          bool need_debug_info = true;


          NnetComputation *ans = new NnetComputation();

          {
            Timer timer;
            ExpandComputation(nnet_, request.misc_info, *mini_computation,
                              need_debug_info, num_n_values, ans);
            seconds_taken_expand_ += timer.Elapsed();
          }
          if (GetVerboseLevel() >= 3) {
            CheckComputation(nnet_, *ans, false);
          }

          {
            Timer timer;
            ans->ComputeCudaIndexes();
            seconds_taken_indexes_ += timer.Elapsed();
          }
          return ans;
        }




        // 进入这个if 一般是 原本调用进入了上面的 CompileViaShortcut,
        // 然后构建了一个mini_request, 递归回来Compile()-> CompilerNoShortcut来编译mini_request 得到 mini_computation
        // 然后返回到上面的CompileViaShortcut 在对 mini_computation进行扩展得到 原本想要的Computation.
        if (computation == NULL){

          computation = CompileNoShortcut(*request);
          const NnetComputation* CachingOptimizingCompiler::CompileNoShortcut( const ComputationRequest &request) {

            // ============================ 编译 过程 ============================
            // ------------------ 构建 Compiler, 编译目标为 request ---------------
            Compiler compiler(request, nnet_);
            // note: 'opts' only contains 'output_debug_info', which is true by default.
            // There may be situations where we'd prefer not to keep it, for speed.
            CompilerOptions opts;
            NnetComputation *computation = new NnetComputation;

            // ------------------ 编译 request 构建一个 computation 见下方代码 --------------
            {
              Timer timer;
              compiler.CreateComputation(opts, computation);
              seconds_taken_compile_ += timer.Elapsed();
            }

            

            // --------------- 优化Optimize -----------------
            {
              Timer timer;
              Optimize(opt_config_, nnet_,
                       MaxOutputTimeInRequest(request),
                       computation);
              seconds_taken_optimize_ += timer.Elapsed();
            }


            // ------------------- computation 变为 cudaIndexes ----- 
            {
              Timer timer;
              computation->ComputeCudaIndexes();
              seconds_taken_indexes_ += timer.Elapsed();
            }
            return computation;
          }
        }

        // 更新 Cache 保存到Cache数组中.
        UpdateCache(request, computation);
        return computation;
      }


      
    }  // done 还没缓存过相同的计算

    // 缓存过相同计算 直接通过缓存获得 Computation, 省去Compile过程.
    else {
      // if found, update access queue
      const NnetComputation *computation = cit->second.first;
      UpdateAccessQueue(cit);
      ans = computation;
    }
    return ans;
  }

  
  NnetComputer computer(config_.compute_config, *computation, nnet_, deriv_nnet_);
  
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






Compiler::Compiler(
    const ComputationRequest &request,
    const Nnet &nnet): nnet_(nnet) {
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







  // =============== part3 
  // 将获得的phases_per_segment 转化为 steps.
  // steps_ 顺序保存 每个segment的 每个phase的 每个sub_phase的 所有cindexes.
  std::vector<std::vector<int32> > steps;
  steps.reserve(1000);

  // 将每个 step 映射划分为segment
  // 正常情况下(无环计算), 是全0向量.
  // <0,0,0,0, 1,1,1,1,1,1,1, 2,2,2,2,2 .... >
  std::vector<int32> step_to_segment;

  {
    // ComputationStepsComputer 会输出steps cindex_id_to_location_
    // 可能会增加一些cindexes 改变graph_
    // cindex_id_to_location_ 就是 ComputationStepsComputer->locations_
    ComputationStepsComputer steps_computer(nnet_, &graph_, &steps, &cindex_id_to_location_);

    
    // foreach request_, phases_per_segment
    for (size_t segment = 0; segment < requests_.size(); segment++) {
      // ===================== 根据reqeust, 和 phase计算次序的 cindex_ids 计算steps ===============
      steps_computer.ComputeForSegment(*(requests_[segment]), phases_per_segment[segment]);

      // 
      while (step_to_segment.size() < steps.size())
        step_to_segment.push_back(segment);


      // 节省空间.
      // save memory, by deleting the phases we just consumed.  the
      // following two lines just exist to save memory.
      std::vector<std::vector<int32> > temp;
      phases_per_segment[segment].swap(temp);
      
    }
    
    steps_computer.Check();
  }





  // ============ part4   计算每个step 是否需要求解导数 ==============
  // 1 计算step的依赖关系dep_steps
  // 2 依赖需要导数, 那么step 也需要导数, 结果得到每个 step是否需要导数.
  std::vector<bool> deriv_needed;
  ComputeDerivNeeded(steps, step_to_segment, &deriv_needed);



  // ============ 创建StepInfo,
  // in:
  // deriv_needed     每个step 是否需要计算导数
  // step_to_segment  每个step属于那个request <0,0,0,0,0, 1,1,1,1,1, 2,2,2,2, request-cnt,request-cnt,>
  // steps            每个phase的每个子sub_phase(phase中相同node-index) 的vector
  // computation      目标计算computation
  CreateStepInfo(deriv_needed, step_to_segment, &steps, computation);


  // =========== 向computation中增加 命令Commands ===========
  AddCommands(deriv_needed, step_to_segment, computation);
  // the following command reorders commands so kAcceptInput and kProvideOutput
  // appear in the desired places.
  ConsolidateIoOperations(nnet_, computation);
  if (opts.output_debug_info)
    OutputDebugInfo(computation);
}




