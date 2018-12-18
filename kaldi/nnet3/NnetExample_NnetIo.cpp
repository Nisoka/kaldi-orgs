// NnetExample 是输入数据, 对应一个帧或多个帧的label或labels
// 用于训练nnet3, 正常情况会只有1帧1label
/// NnetExample is the input data and corresponding label (or labels) for one or
/// more frames of input, used for standard cross-entropy training of neural
/// nets (and possibly for other objective functions).
struct NnetExample {

  /// "io" contains the input and output.  In principle there can be multiple
  /// types of both input and output, with different names.  The order is
  /// irrelevant.
  std::vector<NnetIo> io;

  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &is, bool binary);

  NnetExample() { }

  NnetExample(const NnetExample &other): io(other.io) { }

  void Swap(NnetExample *other) { io.swap(other->io); }

  /// Compresses any (input) features that are not sparse.
  void Compress();

  /// Caution: this operator == is not very efficient.  It's only used in
  /// testing code.
  bool operator == (const NnetExample &other) const { return io == other.io; }
};



struct NnetIo {
  /// the name of the input in the neural net; in simple setups it
  /// will just be "input".
  std::string name;

  // indexes 是一个vector向量长度 和 特征行数相同, 解释了 特征矩阵中每行的意义.
  /// "indexes" is a vector the same length as features.NumRows(), explaining
  /// the meaning of each row of the "features" matrix.

  // indexes 中的n值会一直是0 在一个单独examples-eg中(因为每个eg都是一句话中的多个帧?),
  // 但是当我们将多个egs合并到一个minibatch时 indexes中的n 会变为非0, 因为这时候会有多个样本混入minibatch中.
  // Note: the "n" values in the indexes will always be zero in individual examples, but in general
  /// nonzero after we aggregate the examples into the minibatch level.
  std::vector<Index> indexes;

  
  /// The features or labels.  GeneralMatrix may contain either a CompressedMatrix,
  /// a Matrix, or SparseMatrix (a SparseMatrix would be the natural format for posteriors).
  GeneralMatrix features;

  // 这个构造函数 使用name 构建一个NnetIo indexes中n=0,x=0,
  // 而t值进行递增(t_begin -- t_begin+t_stride*feats.NumRows-1), t_begin 应该是feats.Row(0) 表示的帧.
  
  /// This constructor creates NnetIo with name "name", indexes with n=0, x=0,
  /// and t values ranging from t_begin to 
  /// (t_begin + t_stride * feats.NumRows() - 1) with a stride t_stride, and
  /// the provided features.  t_begin should be the frame that feats.Row(0)
  /// represents.
  NnetIo(const std::string &name,
         int32 t_begin, const MatrixBase<BaseFloat> &feats,
         int32 t_stride = 1);

  // features成员 就是保存的frame数据或者frame label.
  // indexes 成员 保存了frame的 n:样本id, t:样本中帧时间t, x=0;
  //              而在一个eg中 样本id无用=0, 在minibatch中样本id才被分配值,区分样本.
  //              一个eg中 会有一个utt的多帧, 所以id无用.
  //              而一个minibatch中 会有多个eg, 会涉及多个utt, 这时候需要utt-id

  // 一个NnetIo 实际上就是代表的一个可输入的数据.
  // 可以代表一个eg
  // 可以代表一个minibatch
  // 数据用 features成员保存
  // features的每行数据 用 indexes 表示具体代表 utt-id, frame-time,.
  NnetIo::NnetIo(const std::string &name,
                 int32 t_begin, const MatrixBase<BaseFloat> &feats,
                 int32 t_stride):
      name(name), features(feats) {
    
    int32 num_rows = feats.NumRows();
    KALDI_ASSERT(num_rows > 0);
    indexes.resize(num_rows);  // sets all n,t,x to zeros.
    for (int32 i = 0; i < num_rows; i++)
      indexes[i].t = t_begin + i * t_stride;
  }
  

  /// This constructor creates NnetIo with name "name", indexes with n=0, x=0,
  /// and t values ranging from t_begin to 
  /// (t_begin + t_stride * feats.NumRows() - 1) with a stride t_stride, and
  /// the provided features.  t_begin should be the frame that the first row
  /// of 'feats' represents.
  NnetIo(const std::string &name,
         int32 t_begin, const GeneralMatrix &feats,
         int32 t_stride = 1);

  /// This constructor sets "name" to the provided string, sets "indexes" with
  /// n=0, x=0, and t from t_begin to (t_begin + t_stride * labels.size() - 1)
  /// with a stride t_stride, and the labels
  /// as provided.  t_begin should be the frame to which labels[0] corresponds.
  NnetIo(const std::string &name,
         int32 dim,
         int32 t_begin,
         const Posterior &labels,
         int32 t_stride = 1);

  void Swap(NnetIo *other);

  NnetIo() { }

  // Use default copy constructor and assignment operators.
  void Write(std::ostream &os, bool binary) const;

  void Read(std::istream &is, bool binary);

  // this comparison is not very efficient, especially for sparse supervision.
  // It's only used in testing code.
  bool operator == (const NnetIo &other) const;
};

// 获得多个NnetExample的所有NnetIo名字, sorted and unique.
// 一般情况下, 可能就 input output ivector几种, 所以本来的NnetIo数量可能很多
// 但是结果得到的 names_vec 维度不会很长.
// 这样 好像IoNames 就是input-node output-node 等几个表示.

// get a sorted list of all NnetIo names in all examples in the list (will
// normally be just the strings "input" and "output", but maybe also "ivector").
static void GetIoNames(const std::vector<NnetExample> &src,
                            std::vector<std::string> *names_vec) {
  // 注意是个set, 会去除重复
  std::set<std::string> names;
  std::vector<NnetExample>::const_iterator iter = src.begin(), end = src.end();
  for (; iter != end; ++iter) {
    std::vector<NnetIo>::const_iterator iter2 = iter->io.begin(),
                                         end2 = iter->io.end();
    for (; iter2 != end2; ++iter2)
      names.insert(iter2->name);
  }
  CopySetToVector(names, names_vec);
}



// Get feature "sizes" for each NnetIo name, which are the total number of
// Indexes for that NnetIo (needed to correctly size the output matrix).  Also
// make sure the dimensions are consistent for each name.
static void GetIoSizes(const std::vector<NnetExample> &src,
                       const std::vector<std::string> &names,
                       std::vector<int32> *sizes) {
  // names --  Minibatch中所有NnetIo.(多个NnetExample-eg下的多个NnetIo)
  //  目的是为每个NnetIo 都保存其特征维度
  std::vector<int32> dims(names.size(), -1);  // just for consistency checking.
  sizes->clear();
  // NNetIo 数量
  sizes->resize(names.size(), 0);
  
  // foreach NnetIo
  std::vector<std::string>::const_iterator names_begin = names.begin(),
                                             names_end = names.end();

  // foreach NnetExample-eg
  std::vector<NnetExample>::const_iterator iter = src.begin(), end = src.end();
  for (; iter != end; ++iter) {

    
    // foreach NnetIo in a eg
    std::vector<NnetIo>::const_iterator iter2 = iter->io.begin(),
                                         end2 = iter->io.end();
    for (; iter2 != end2; ++iter2) {
      
      const NnetIo &io = *iter2;
      // 查找当前 NnetIo 在 所有Minibathch NnetIo中的位置
      std::vector<std::string>::const_iterator names_iter =
          std::lower_bound(names_begin, names_end, io.name);
      KALDI_ASSERT(*names_iter == io.name);
      // 获得位置pos
      int32 i = names_iter - names_begin;
      // 当前Io的特征维度
      int32 this_dim = io.features.NumCols();
      // 将对应的Io的特征维度写入 dims[i]
      if (dims[i] == -1) {
        dims[i] = this_dim;
      } else if (dims[i] != this_dim) {
        KALDI_ERR << "Merging examples with inconsistent feature dims: "
                  << dims[i] << " vs. " << this_dim << " for '"
                  << io.name << "'.";
      }

      // NnetIo内的 特征rows, 一定与 indexes相等.
      KALDI_ASSERT(io.features.NumRows() == io.indexes.size());

      // 每个NnetIo的 特征行数量++.
      int32 this_size = io.indexes.size();
      // 保存NnetIo 特征总行数(应该是不同的eg 可能具有相同的NnetIo names?)
      (*sizes)[i] += this_size;
    }
  }
}




// Do the final merging of NnetIo, once we have obtained the names, dims and
// sizes for each feature/supervision type.
static void MergeIo(const std::vector<NnetExample> &src,
                    const std::vector<std::string> &names,
                    const std::vector<int32> &sizes,
                    bool compress,
                    NnetExample *merged_eg) {

  // total NnetIo.
  // The total number of Indexes we have across all examples.
  int32 num_feats = names.size();
  // sorted uniqued NnetIo的统计大小
  std::vector<int32> cur_size(num_feats, 0);


  // 不同的NnetIo 中的 不同的Indexes 的所有Matrx.?
  // 为什么一个 Index 也表示一个 Matrix了? (或者说是 Index中的n 表示一个Matrix)
  // The features in the different NnetIo in the Indexes across all examples
  std::vector<std::vector<GeneralMatrix const*> > output_lists(num_feats);


  // 按照上面的 GetIoNames  GetIoSizes 来初始化 merged_eg.
  // Initialize the merged_eg
  merged_eg->io.clear();
  merged_eg->io.resize(num_feats);
  
  for (int32 f = 0; f < num_feats; f++) {
    NnetIo &io = merged_eg->io[f];
    int32 size = sizes[f];
    KALDI_ASSERT(size > 0);
    io.name = names[f];
    io.indexes.resize(size);
  }



  // each NnetIo-name
  std::vector<std::string>::const_iterator names_begin = names.begin(),
                                             names_end = names.end();
  // each NnetExample-eg
  std::vector<NnetExample>::const_iterator eg_iter = src.begin(),
                                           eg_end = src.end();
  // foreach eg
  for (int32 n = 0; eg_iter != eg_end; ++eg_iter, ++n) {
    // foreach NnetIo in cur eg.
    std::vector<NnetIo>::const_iterator io_iter = eg_iter->io.begin(),
                                        io_end = eg_iter->io.end();
    for (; io_iter != io_end; ++io_iter) {
      const NnetIo &io = *io_iter;
      // 获得在 经过set sort and unique之后的names
      std::vector<std::string>::const_iterator names_iter =
          std::lower_bound(names_begin, names_end, io.name);
      KALDI_ASSERT(*names_iter == io.name);
      // 获得在 set中的 offset
      int32 f = names_iter - names_begin;
      // 对应的NnetIo 特征行数
      int32 this_size = io.indexes.size();
      // 获得第f'th的当前已经汇总的总数偏移, 新加入的NnetIo 的 feat 和 indexes都向后添加到 merged_eg
      int32 &this_offset = cur_size[f];
      // sizes[f] 是已经统计过的.
      KALDI_ASSERT(this_size + this_offset <= sizes[f]);

      // ============= 汇总属于f'th的NnetIo的的所有特征 =============
      // Add f'th Io's features
      // 向 ouput_lists[f] 加入同名NnetIo 原本不同实例的 特征, 汇总一下.
      output_lists[f].push_back(&(io.features));

      // ============= 汇总在merged_eg中 的综合NnetIo的indexes =============
      // 将原来NnetIo 的indexes copy到 merged的对应汇总NnetIo的indexes中
      // Work on the Indexes for the f^th Io in merged_eg
      NnetIo &output_io = merged_eg->io[f];
      std::copy(io.indexes.begin(), io.indexes.end(),
                output_io.indexes.begin() + this_offset);

      // ============= 汇总不同的eg 到merged-eg中,
      // 原本eg是单一eg,表示一个utt的多帧组成的样本, 所以indexes中n=0,
      // -----------------------------------------
      // 而汇总后 属于f'th的NnetIo 汇总起来, 需要为其增加n. n 表示来自原本的第n个单独eg(实际应该是具体的NnetIo, 但是一个eg内NnetIo是不会重复引用同一个地点的(name))
      // -----------------------------------------
      std::vector<Index>::iterator output_iter = output_io.indexes.begin();
      // Set the n index to be different for each of the original examples.
      for (int32 i = this_offset; i < this_offset + this_size; i++) {
        // we could easily support merging already-merged egs, but I don't see a
        // need for it right now.
        KALDI_ASSERT(output_iter[i].n == 0 &&
                     "Merging already-merged egs?  Not currentlysupported.");
        output_iter[i].n = n;
      }
      this_offset += this_size;  // note: this_offset is a reference.
    }
  }

  
  KALDI_ASSERT(cur_size == sizes);
  // 将对应的 汇总的 output_list 特征 合并成一个对应的 features.
  for (int32 f = 0; f < num_feats; f++) {
    AppendGeneralMatrixRows(output_lists[f],
                            &(merged_eg->io[f].features));
    if (compress) {
      // the following won't do anything if the features were sparse.
      merged_eg->io[f].features.Compress();
    }
  }
}


void ExampleMerger::Finish() {
  if (finished_) return;  // already finished.
  finished_ = true;

  // we'll convert the map eg_to_egs_ to a vector of vectors to avoid
  // iterator invalidation problems.
  // 简单的将map 转为 vector.
  std::vector<std::vector<NnetExample*> > all_egs;
  all_egs.reserve(eg_to_egs_.size());

  MapType::iterator iter = eg_to_egs_.begin(), end = eg_to_egs_.end();
  for (; iter != end; ++iter)
    all_egs.push_back(iter->second);
  eg_to_egs_.clear();

  for (size_t i = 0; i < all_egs.size(); i++) {
    int32 minibatch_size;
    std::vector<NnetExample*> &vec = all_egs[i];
    KALDI_ASSERT(!vec.empty());
    int32 eg_size = GetNnetExampleSize(*(vec[0]));
    bool input_ended = true;
    while (!vec.empty() &&
           (minibatch_size = config_.MinibatchSize(eg_size, vec.size(),
                                                   input_ended)) != 0) {
      // MergeExamples() expects a vector of NnetExample, not of pointers,
      // so use swap to create that without doing any real work.
      std::vector<NnetExample> egs_to_merge(minibatch_size);
      for (int32 i = 0; i < minibatch_size; i++) {
        egs_to_merge[i].Swap(vec[i]);
        delete vec[i];  // we owned those pointers.
      }
      vec.erase(vec.begin(), vec.begin() + minibatch_size);
      WriteMinibatch(egs_to_merge);
    }

    // 正常应该不会进入下面
    if (!vec.empty()) {
      int32 eg_size = GetNnetExampleSize(*(vec[0]));
      NnetExampleStructureHasher eg_hasher;
      size_t structure_hash = eg_hasher(*(vec[0]));
      int32 num_discarded = vec.size();
      stats_.DiscardedExamples(eg_size, structure_hash, num_discarded);
      for (int32 i = 0; i < num_discarded; i++)
        delete vec[i];
      vec.clear();
    }
  }

  
  stats_.PrintStats();
  
}
