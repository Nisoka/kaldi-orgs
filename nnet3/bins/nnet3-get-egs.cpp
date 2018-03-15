


// nnet3-get-egs
//     --num-pdfs=2943
//     --frame-subsampling-factor=1
//     --online-ivectors=scp:exp/nnet3/ivectors_train_sp/ivector_online.scp
//     --online-ivector-period=10 --left-context=16 --right-context=12 --compress=true --num-frames=8

    
//     'ark,s,cs:utils/filter_scp.pl exp/nnet3/tdnn_sp/egs/valid_uttlist data/train_sp_hires/feats.scp | apply-cmvn --norm-means=false --norm-vars=false --utt2spk=ark:data/train_sp_hires/utt2spk scp:data/train_sp_hires/cmvn.scp scp:- ark:- |'

//     ark,s,cs:- ark:exp/nnet3/tdnn_sp/egs/valid_all.egs


void nnet3_get_egs()
{
  // egs_opts="--left-context=$left_context --right-context=$right_context --compress=$compress --num-frames=$frames_per_eg"
  //                                                                                                     8
  // ivector_opts="--online-ivectors=scp:$online_ivector_dir/ivector_online.scp --online-ivector-period=$ivector_period"
  //                                                                                                    10
  // valid_feats="ark,s,cs:utils/filter_scp.pl $dir/valid_uttlist $data/feats.scp               | apply-cmvn $cmvn_opts --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp scp:- ark:- |"
  //                                                       data/train_sp_hires/feats.scp :>valid_filter> feats.scp.
  
  // utils/filter_scp.pl $dir/valid_uttlist $dir/ali_special.scp \| \
  // ali-to-pdf $alidir/final.mdl scp:- ark:- \|                    \
  // ali-to-post ark:- ark:- \|                                         \                    ---------- arg2  ali-pdf-post
  // nnet3-get-egs --num-pdfs=$num_pdfs --frame-subsampling-factor=$frame_subsampling_factor \   ---- factor = 1
  //   $ivector_opts                                                              -------- addtion -ivector-feature
  //   $egs_opts
  //   "$valid_feats"                                                   \                    ---------- arg1   MFCC-features
  //   ark,s,cs:- "ark:$dir/valid_all.egs" || touch $dir/.error &                            ---------- arg3   output.
  
  
  // "Usage:  nnet3-get-egs [options] <features-rspecifier> "
  // "<pdf-post-rspecifier> <egs-out>\n"
  // examples :
  //     "nnet3-get-egs --num-pdfs=2658 --left-context=12 --right-context=9 --num-frames=8 \"$feats\"\\\n"
  //     "\"ark:gunzip -c exp/nnet/ali.1.gz | ali-to-pdf exp/nnet/1.nnet ark:- ark:- | ali-to-post ark:- ark:- |\" \\\n"
  //     "   ark:- \n"

  

  int nnet3_get_egs_______(int argc, char *argv[]) {

    bool compress = true;
    int32 num_pdfs = -1,
        length_tolerance = 100,
        targets_length_tolerance = 2,  
        online_ivector_period = 1;

    ExampleGenerationConfig eg_config;  // controls num-frames,
                                        // left/right-context, etc.

    std::string online_ivector_rspecifier;

    ParseOptions po(usage);

    po.Register("compress", &compress, "If true, write egs with input features "
                "in compressed format (recommended).  This is "
                "only relevant if the features being read are un-compressed; "
                "if already compressed, we keep we same compressed format when "
                "dumping egs.");
    po.Register("num-pdfs", &num_pdfs, "Number of pdfs in the acoustic "
                "model");
    po.Register("ivectors", &online_ivector_rspecifier, "Alias for "
                "--online-ivectors option, for back compatibility");
    po.Register("online-ivectors", &online_ivector_rspecifier, "Rspecifier of "
                "ivector features, as a matrix.");
    po.Register("online-ivector-period", &online_ivector_period, "Number of "
                "frames between iVectors in matrices supplied to the "
                "--online-ivectors option");

    eg_config.Register(&po);
    po.Read(argc, argv);

    if (num_pdfs <= 0)
      KALDI_ERR << "--num-pdfs options is required.";


    // ==================== 获得UtteranceSpliter的配置文件 =========
    // ------------- 在这里 num_frames_str = 8
    // ------------- 结果 num_frames 就是一个元素的数组 8
    // ------------- 并且后面 获得对utterance 的划分**最主要**的数字就是 primary_length = 8---
    eg_config.ComputeDerived();

    
    void ExampleGenerationConfig::ComputeDerived() {
      // split the num_frames_str to array.
      if (!SplitStringToIntegers(num_frames_str, ",", false, &num_frames) ||
          num_frames.empty()) {
        KALDI_ERR << "Invalid option (expected comma-separated list of integers): "
                  << "--num-frames=" << num_frames_str;
      }

      // frame_subsampling_factor = 1
      int32 m = frame_subsampling_factor;
      if (m < 1) {
        KALDI_ERR << "Invalid value --frame-subsampling-factor=" << m;
      }
      
      bool changed = false;
      
      for (size_t i = 0; i < num_frames.size(); i++) {
        int32 value = num_frames[i];
        if (value <= 0) {
          KALDI_ERR << "Invalid option --num-frames=" << num_frames_str;
        }
        if (value % m != 0) {
          value = m * ((value / m) + 1);
          changed = true;
        }
        num_frames[i] = value;
      }
      
      if (changed) {
        std::ostringstream rounded_num_frames_str;
        for (size_t i = 0; i < num_frames.size(); i++) {
          if (i > 0)
            rounded_num_frames_str << ',';
          rounded_num_frames_str << num_frames[i];
        }
        KALDI_LOG << "Rounding up --num-frames=" << num_frames_str
                  << " to multiples of --frame-subsampling-factor=" << m
                  << ", to: " << rounded_num_frames_str.str();
      }
    }

    // ============================ 划分工具 UtteranceSplitter ================
    // 利用这个对utterance 进行划分.
    UtteranceSplitter utt_splitter(eg_config);

    
    UtteranceSplitter::UtteranceSplitter(const ExampleGenerationConfig &config):
        config_(config),
        total_num_utterances_(0), total_input_frames_(0),
        total_frames_overlap_(0), total_num_chunks_(0),
        total_frames_in_chunks_(0) {
      if (config.num_frames.empty()) {
        KALDI_ERR << "You need to call ComputeDerived() on the "
            "ExampleGenerationConfig().";
      }
      
      InitSplitForLength();
      void UtteranceSplitter::InitSplitForLength() {
        // 计算 得到 24  干嘛用的???
        int32 max_utterance_length = MaxUtteranceLength();

        // The 'splits' vector is a list of possible splits
        // (a split being a sorted vector of chunk-sizes).
        // The vector 'splits' is itself sorted.

        // splits 中的每个split 都是一个排序好的 chunk-size list. 
        std::vector<std::vector<int32> > splits;
        
        InitSplits(&splits);

        void UtteranceSplitter::InitSplits(std::vector<std::vector<int32> > *splits) const {
          // we consider splits whose default duration (as returned by
          // DefaultDurationOfSplit()) is up to MaxUtteranceLength() + primary_length.
          // We can be confident without doing a lot of math, that splits above this
          // length will never be chosen for any utterance-length up to
          // MaxUtteranceLength() (which is the maximum we use).
          int32 primary_length = config_.num_frames[0],
              default_duration_ceiling = MaxUtteranceLength() + primary_length;

          typedef unordered_set<std::vector<int32>, VectorHasher<int32> > SetType;

          SetType splits_set;

          int32 num_lengths = config_.num_frames.size();

          // The splits we are allow are: zero to two 'alternate' lengths, plus
          // an arbitrary number of repeats of the 'primary' length.  The repeats
          // of the 'primary' length are handled by the inner loop over n.
          // The zero to two 'alternate' lengths are handled by the loops over
          // i and j.  i == 0 and j == 0 are special cases; they mean, no
          // alternate is chosen.
          for (int32 i = 0; i < num_lengths; i++) {
            for (int32 j = 0; j < num_lengths; j++) {
              std::vector<int32> vec;
              if (i > 0)
                vec.push_back(config_.num_frames[i]);
              if (j > 0)
                vec.push_back(config_.num_frames[j]);
              int32 n = 0;
              while (DefaultDurationOfSplit(vec) <= default_duration_ceiling) {
                if (!vec.empty()) // Don't allow the empty vector as a split.
                  splits_set.insert(vec);
                n++;
                vec.push_back(primary_length);
                std::sort(vec.begin(), vec.end());
              }
            }
          }
          for (SetType::const_iterator iter = splits_set.begin();
               iter != splits_set.end(); ++iter)
            splits->push_back(*iter);
          std::sort(splits->begin(), splits->end());  // make the order deterministic,
          // for consistency of output
          // between runs and C libraries.
        }



        // Define a split-index 0 <= s < splits.size() as index into the 'splits'
        // vector, and let a cost c >= 0 represent the mismatch between an
        // utterance length and the total length of the chunk sizes in a split:

        //  c(default_duration, utt_length) = (default_duration > utt_length ?
        //                                    default_duration - utt_length :
        //                                    2.0 * (utt_length - default_duration))
        // [but as a special case, set c to infinity if the largest chunk size in the
        //  split is longer than the utterance length; we couldn't, in that case, use
        //  this split for this utterance].

        // 'costs_for_length[u][s]', indexed by utterance-length u and then split,
        // contains the cost for utterance-length u and split s.

        std::vector<std::vector<float> > costs_for_length(max_utterance_length + 1);
        int32 num_splits = splits.size();

        for (int32 u = 0; u <= max_utterance_length; u++)
          costs_for_length[u].reserve(num_splits);

        for (int32 s = 0; s < num_splits; s++) {
          const std::vector<int32> &split = splits[s];
          float default_duration = DefaultDurationOfSplit(split);
          int32 max_chunk_size = *std::max_element(split.begin(), split.end());
          for (int32 u = 0; u <= max_utterance_length; u++) {
            // c is the cost for this utterance length and this split.  We penalize
            // gaps twice as strongly as overlaps, based on the intuition that
            // completely throwing out frames of data is worse than counting them
            // twice.
            float c = (default_duration > float(u) ? default_duration - float(u) :
                       2.0 * (u - default_duration));
            if (u < max_chunk_size)  // can't fit the largest of the chunks in this
              // utterance
              c = std::numeric_limits<float>::max();
            KALDI_ASSERT(c >= 0);
            costs_for_length[u].push_back(c);
          }
        }


        splits_for_length_.resize(max_utterance_length + 1);

        for (int32 u = 0; u <= max_utterance_length; u++) {
          const std::vector<float> &costs = costs_for_length[u];
          float min_cost = *std::min_element(costs.begin(), costs.end());
          if (min_cost == std::numeric_limits<float>::max()) {
            // All costs were infinity, becaues this utterance-length u is shorter
            // than the smallest chunk-size.  Leave splits_for_length_[u] as empty
            // for this utterance-length, meaning we will not be able to choose any
            // split, and such utterances will be discarded.
            continue;
          }
          float cost_threshold = 1.9999; // We will choose pseudo-randomly from splits
          // that are within this distance from the
          // best cost.  Make the threshold just
          // slightly less than 2...  this will
          // hopefully make the behavior more
          // deterministic for ties.
          std::vector<int32> possible_splits;
          std::vector<float>::const_iterator iter = costs.begin(), end = costs.end();
          int32 s = 0;
          for (; iter != end; ++iter,++s)
            if (*iter < min_cost + cost_threshold)
              splits_for_length_[u].push_back(splits[s]);
        }
      }
    }


  

    

    std::string
        feature_rspecifier = po.GetArg(1),              // online-ivectors - exp/nnet3/ivectors_train_sp
        pdf_post_rspecifier = po.GetArg(2),             // ali-pdf-post    - exp/nnet3/tdnn_sp/egs
        examples_wspecifier = po.GetArg(3);             // .....           - exp/nnet3/tdnn_sp/egs/

    // SequentialGeneralMatrixReader can read either a Matrix or
    // CompressedMatrix (or SparseMatrix, but not as relevant here),
    // and it retains the type.  This way, we can generate parts of
    // the feature matrices without uncompressing and re-compressing.
    SequentialGeneralMatrixReader feat_reader(feature_rspecifier);
    RandomAccessPosteriorReader pdf_post_reader(pdf_post_rspecifier);
    NnetExampleWriter example_writer(examples_wspecifier);
    
    RandomAccessBaseFloatMatrixReader online_ivector_reader(
        online_ivector_rspecifier);

    int32 num_err = 0;

    // foreach UTT - MFCC - IVECTOR - ALI-PDF-POST
    for (; !feat_reader.Done(); feat_reader.Next()) {
      std::string key = feat_reader.Key();

      // UTT-MFCC
      const GeneralMatrix &feats = feat_reader.Value();
      if (!pdf_post_reader.HasKey(key)) {
      } else {
        // UTT-PDF-POST
        const Posterior &pdf_post = pdf_post_reader.Value(key);
        
        const Matrix<BaseFloat> *online_ivector_feats = NULL;
        if (!online_ivector_rspecifier.empty()) {
          if (!online_ivector_reader.HasKey(key)) {
            KALDI_WARN << "No iVectors for utterance " << key;
            num_err++;
            continue;
          } else {
            // this address will be valid until we call HasKey() or Value()
            // again.
            online_ivector_feats = &(online_ivector_reader.Value(key));
          }
        }

        // Check if the length of  utt-mfcc is different from the utt-ivector
        if (online_ivector_feats != NULL &&
            (abs(feats.NumRows() - (online_ivector_feats->NumRows() *
                                    online_ivector_period)) > length_tolerance
             || online_ivector_feats->NumRows() == 0)) {
          KALDI_WARN << "Length difference between feats " << feats.NumRows()
                     << " and iVectors " << online_ivector_feats->NumRows()
                     << "exceeds tolerance " << length_tolerance;
          num_err++;
          continue;
        }


        //                       并且还加入ivector特征.(注意 ivector特征 并不是每帧不同的,因为一句话实际上是一个人)
        
        // prcocess frames of current utt.

        
        // 1 ================== utt_splitter(内部初始化等操作得到Split_for_length), 
        // 2 ================== 利用utt_splitter 对utterance进行划分得到 vector<chunk> 对utterance的划分,
        //    ----------------- 每个chunk就代表一个TDNN的输入样本eg
        // (vector<chunk> 是对utterance 按照<primary, primary, ... primary, other> 的划分,内部可能还会包含 gap空白 overlap重叠)
        // 3 ================== 每个chunk 本身对应的是utterance的MFCC特征+ivector特征, 内部利用了Index描述数据信息

        if (!ProcessFile(feats,
                         online_ivector_feats,    online_ivector_period,
                         pdf_post,
                         key, compress, num_pdfs, 
                         targets_length_tolerance,
                         &utt_splitter,
                         &example_writer))



          
          num_err++;
      }
    }
    if (num_err > 0)
      KALDI_WARN << num_err << " utterances had errors and could "
          "not be processed.";
    // utt_splitter prints stats in its destructor.
    return utt_splitter.ExitStatus();
}


  static bool ProcessFile(const GeneralMatrix &feats,
                        const MatrixBase<BaseFloat> *ivector_feats,
                        int32 ivector_period,
                        const Posterior &pdf_post,
                        const std::string &utt_id,
                        bool compress,
                        int32 num_pdfs,
                        int32 length_tolerance,
                        UtteranceSplitter *utt_splitter,
                        NnetExampleWriter *example_writer) {

    // utt frames cnt
  int32 num_input_frames = feats.NumRows();
  // 判断给定的 num_input_frames 是否与pdf_post 的大小相等 (误差不超过 length_tolerance)
  if (!utt_splitter->LengthsMatch(utt_id, num_input_frames,
                                  static_cast<int32>(pdf_post.size()),
                                  length_tolerance))
    return false;  // LengthsMatch() will have printed a warning.


  



  // =======================  划分一个utterence 为多个 chunk ========================
  // 每个chunk 是一个n frames的数据块.
  //    1 primary chunksize x n
  //    2 possibal_split_size  --- split_for_length_  (在 UtteranceSpliter 构造时候创建计算的)
  //    3 gaps 插入到各个chunk 之间.
  std::vector<ChunkTimeInfo> chunks;
  utt_splitter->GetChunksForUtterance(num_input_frames, &chunks);

  
  void UtteranceSplitter::GetChunksForUtterance(
      int32 utterance_length,
      std::vector<ChunkTimeInfo> *chunk_info) {
    
    std::vector<int32> chunk_sizes;

    // ======================= 生成对utterance_length 的划分 --- chunk_size ==============
    // chunk_sizes 是划分结果(primary, primary, ... primary, possiable-split-size)
    GetChunkSizesForUtterance(utterance_length, &chunk_sizes);
    
    void UtteranceSplitter::GetChunkSizesForUtterance(int32 utterance_length,
                                                      std::vector<int32> *chunk_sizes) const {
      KALDI_ASSERT(!splits_for_length_.empty());

      // 'primary_length' is the first-specified num-frames.
      // It's the only chunk that may be repeated an arbitrary number
      // of times.
      int32
          primary_length = config_.num_frames[0],
          num_frames_overlap = config_.num_frames_overlap,
          max_tabulated_length = splits_for_length_.size() - 1,
          num_primary_length_repeats = 0;
      
      KALDI_ASSERT(primary_length - num_frames_overlap > 0);
      KALDI_ASSERT(utterance_length >= 0);

      // ---------------------  primary 划分  -----------------------
      // 先将utterence尽可能的按照 primary 长度. 留下剩余 utterance_length
      //     循环切分utterance_length, 以 primary_length repeat,
      //     直到 最后剩余utterance_length < splits_for_length_.size() 长度.
      while (utterance_length > max_tabulated_length) {
        utterance_length -= (primary_length - num_frames_overlap);
        num_primary_length_repeats++;
      }
      
      KALDI_ASSERT(utterance_length >= 0);

      // ---------------------- 剩余 utterance_length 按照上面获得的 splits_for_length_ 进行一个划分 --------------
      // 可能的 剩余长度划分可能.
      const std::vector<std::vector<int32> > &possible_splits = splits_for_length_[utterance_length];

      // 如果可能划分isempty().
      if (possible_splits.empty()) {
        chunk_sizes->clear();
        return;
      }
     
      int32
          //
          num_possible_splits = possible_splits.size(),
          // 随机选择一个可能划分.
          randomly_chosen_split = RandInt(0, num_possible_splits - 1);

      // --------------------- 随机获得一个可能的划分 chunk_sizes  作为 剩余utterance 的划分
      // --------------------- 将N 个 primary 划分 加入到 chunk_sizes, 完成对 utterance的划分.
      // ???????????????????  但是这里chunk_size换分结果 并不和 utterance长度相等?
      // ??????  因为split_for_length 的求解 并不能覆盖所有长度可能. .... 
      *chunk_sizes = possible_splits[randomly_chosen_split];
      for (int32 i = 0; i < num_primary_length_repeats; i++)
        chunk_sizes->push_back(primary_length);

      std::sort(chunk_sizes->begin(), chunk_sizes->end());
      if (RandInt(0, 1) == 0) {
        std::reverse(chunk_sizes->begin(), chunk_sizes->end());
      }
    }

    // gaps 空白. 将空白 按照chunk内比例 分配给 gaps
    std::vector<int32> gaps(chunk_sizes.size());

    // 计算需要增加的空白frames
    GetGapSizes(utterance_length, true, chunk_sizes, &gaps);


    
    void UtteranceSplitter::GetGapSizes(int32 utterance_length,
                                        bool enforce_subsampling_factor,
                                        const std::vector<int32> &chunk_sizes,
                                        std::vector<int32> *gap_sizes) const {
      if (chunk_sizes.empty()) {
        gap_sizes->clear();
        return;
      }

      // 忽略这个 if
      if (enforce_subsampling_factor && config_.frame_subsampling_factor > 1) {
        int32 sf = config_.frame_subsampling_factor, size = chunk_sizes.size();
        int32 utterance_length_reduced = (utterance_length + (sf - 1)) / sf;
        std::vector<int32> chunk_sizes_reduced(chunk_sizes);
        for (int32 i = 0; i < size; i++) {
          KALDI_ASSERT(chunk_sizes[i] % config_.frame_subsampling_factor == 0);
          chunk_sizes_reduced[i] /= config_.frame_subsampling_factor;
        }

        GetGapSizes(utterance_length_reduced, false,
                    chunk_sizes_reduced, gap_sizes);
        
        KALDI_ASSERT(gap_sizes->size() == static_cast<size_t>(size));
        for (int32 i = 0; i < size; i++)
          (*gap_sizes)[i] *= config_.frame_subsampling_factor;
        return;
      }


      // 计算剩余空白或重叠 total_gap = utterance_length - total_of_chunk_sizes
      int32
          num_chunks = chunk_sizes.size(),
          total_of_chunk_sizes = std::accumulate(chunk_sizes.begin(),
                                                 chunk_sizes.end(),
                                                 int32(0)),
          total_gap = utterance_length - total_of_chunk_sizes;
      gap_sizes->resize(num_chunks);

      // ------------------- total_gap < 0 说明会产生重叠,
      // ------------------- utterance_length 不足以填充分化chunk_sizes. 所以会重叠chunk_sizes
      if (total_gap < 0) {
        // there is an overlap.  Overlaps can only go between chunks, not at the
        // beginning or end of the utterance.  Also, we try to make the length of
        // overlap proportional to the size of the smaller of the two chunks
        // that the overlap is between.
        if (num_chunks == 1) {
          // there needs to be an overlap, but there is only one chunk... this means
          // the chunk-size exceeds the utterance length, which is not allowed.
          KALDI_ERR << "Chunk size is " << chunk_sizes[0]
                    << " but utterance length is only "
                    << utterance_length;
        }

        // note the elements of 'overlaps' will be <= 0.
        std::vector<int32>
            magnitudes(num_chunks - 1),
            overlaps(num_chunks - 1);
        // magnitude向量 会保存相邻两个chunk之间较小的那个,
        // 因为两者之间可能会存在重叠, magnitude会用来作为 overlap的分配比例依据.
        // the 'magnitudes' vector will contain the minimum of the lengths of the
        // two adjacent chunks between which are are going to consider having an
        // overlap.  These will be used to assign the overlap proportional to that
        // size.
        for (int32 i = 0; i + 1 < num_chunks; i++) {
          magnitudes[i] = std::min<int32>(chunk_sizes[i], chunk_sizes[i + 1]);
        }



        // -----------------------  分化空白或者重叠 ------------------------
        // 将空白部分, 按照magnitudes向量各元素比例 分化空白 ==> overlaps.
        DistributeRandomly(total_gap, magnitudes, &overlaps);
        void UtteranceSplitter::DistributeRandomly(int32 n,
                                                   const std::vector<int32> &magnitudes,
                                                   std::vector<int32> *vec) {
          KALDI_ASSERT(!vec->empty() && vec->size() == magnitudes.size());
          int32 size = vec->size();

          // ---------------------- 空白情况 -------------------
          // 
          if (n < 0) {
            DistributeRandomly(-n, magnitudes, vec);
            for (int32 i = 0; i < size; i++)
              (*vec)[i] *= -1;
            return;
          }

          // --------------------- 重叠情况 ---------------------
          float total_magnitude = std::accumulate(magnitudes.begin(), magnitudes.end(),
                                                  int32(0));
          KALDI_ASSERT(total_magnitude > 0);
          // note: 'partial_counts' contains the negative of the partial counts, so
          // when we sort the larger partial counts come first.
          std::vector<std::pair<float, int32> > partial_counts;
          int32 total_count = 0;

          // 按照 magnitude比例分化 重叠, 每个chunk 会有 vec[i] 的空白.
          // 总共空白最后不会等于 n, 后面继续处理, 最终让整体空白total_count==n.
          for (int32 i = 0; i < size; i++) {
            float this_count = n * float(magnitudes[i]) / total_magnitude;
            int32 this_whole_count = static_cast<int32>(this_count),
                this_partial_count = this_count - this_whole_count;
            (*vec)[i] = this_whole_count;
            total_count += this_whole_count;
            partial_counts.push_back(std::pair<float, int32>(-this_partial_count, i));
          }
 
          KALDI_ASSERT(total_count <= n && total_count + size >= n);
          std::sort(partial_counts.begin(), partial_counts.end());
          int32 i = 0;

          // 向排好序的vec中逐个增加1个空白, 直到 total_count = n.
          // 并且 total_count 都按比例的分布在了 vec[] 中.
          // Increment by one the elements of the vector that has the largest partial
          // count, then the next largest partial count, and so on... until we reach the
          // desired total-count 'n'.
          for(; total_count < n; i++,total_count++) {
            (*vec)[partial_counts[i].second]++;
          }
          KALDI_ASSERT(std::accumulate(vec->begin(), vec->end(), int32(0)) == n);
        }



        // 不会出现的情况.... 不理解why?
        for (int32 i = 0; i + 1 < num_chunks; i++) {
          // If the following condition does not hold, it's possible we
          // could get chunk start-times less than zero.  I don't believe
          // it's possible for this condition to fail, but we're checking
          // for it at this level to make debugging easier, just in case.
          KALDI_ASSERT(overlaps[i] <= magnitudes[i]);
        }

        // begin end 都不许有空白.
        (*gap_sizes)[0] = 0;  // no gap before 1st chunk.
        for (int32 i = 1; i < num_chunks; i++)
          (*gap_sizes)[i] = overlaps[i-1];
        
      }

      // ------------------- total_gap > 0 说明会产生空白, utterance_length 太长??????????????
      else {
        // There may be a gap.  Gaps can go at the start or end of the utterance, or
        // between segments.  We try to distribute the gaps evenly.
        std::vector<int32> gaps(num_chunks + 1);
        DistributeRandomlyUniform(total_gap, &gaps);
        // the last element of 'gaps', the one at the end of the utterance, is
        // implicit and doesn't have to be written to the output.
        for (int32 i = 0; i < num_chunks; i++)
          (*gap_sizes)[i] = gaps[i];
      }
    }



    
    int32 num_chunks = chunk_sizes.size();
    chunk_info->resize(num_chunks);

    // ===========================   按照chunk_sizes 划分utterence 获得chunk 时间  =============================

    // 规划 chunk 时间
    // chunk_info Vector<ChunkTimeInfo, ChunkTimeInfo ... >
    // chunk
    //     first_frames: 起始帧id
    //     num_frames:   chunk 帧总数
    //     left_context: 当前chunk的 L上下文帧数(是之前几帧frames特征数据,可能在前几个chunk里面)
    //     right_context: 当前chunk的 R上下文帧数.
    int32 t = 0;
    for (int32 i = 0; i < num_chunks; i++) {
      t += gaps[i];
      ChunkTimeInfo &info = (*chunk_info)[i];
      info.first_frame = t;
      info.num_frames = chunk_sizes[i];
      info.left_context = (i == 0 && config_.left_context_initial >= 0 ?
                           config_.left_context_initial : config_.left_context);
      info.right_context = (i == num_chunks - 1 && config_.right_context_final >= 0 ?
                            config_.right_context_final : config_.right_context);
      t += chunk_sizes[i];
    }



    // ======================= 为chunk_info 内对应frame 设置frame权重, 重叠情况权重降低为1/2
    SetOutputWeights(utterance_length, chunk_info);
    void UtteranceSplitter::SetOutputWeights(int32 utterance_length,
                                             std::vector<ChunkTimeInfo> *chunk_info) const {
      int32 sf = config_.frame_subsampling_factor;
      int32 num_output_frames = (utterance_length + sf - 1) / sf;
      // num_output_frames is the number of frames of supervision.




      // 'count[t]' will be the number of chunks that this output-frame t appears in.
      // Note: the
      // 'first_frame' and 'num_frames' members of ChunkTimeInfo will always be
      // multiples of frame_subsampling_factor.
      // ----------------- count: 每frame 出现在多少个chunk中 (每frame的重叠情况)
      std::vector<int32> count(num_output_frames, 0);
      int32 num_chunks = chunk_info->size();
      for (int32 i = 0; i < num_chunks; i++) {
        ChunkTimeInfo &chunk = (*chunk_info)[i];
        for (int32 t = chunk.first_frame / sf;
             t < (chunk.first_frame + chunk.num_frames) / sf;
             t++)
          count[t]++;
      }

      // -----------------chunk.output_weights: 对应为内部frame 的权重
      // --------- frame权重一般为1, 有重叠 权重为1/2
      for (int32 i = 0; i < num_chunks; i++) {
        ChunkTimeInfo &chunk = (*chunk_info)[i];
        chunk.output_weights.resize(chunk.num_frames / sf);
        int32 t_start = chunk.first_frame / sf;
        for (int32 t = t_start;
             t < (chunk.first_frame + chunk.num_frames) / sf;
             t++)
          chunk.output_weights[t - t_start] = 1.0 / count[t];
      }
    }


    // =============== 计算一些统计计数
    // --------- total_num_utterances_, total_input_frames_, total_frames_overlap_, chunk_size_to_count_(某个size的chunk的总数)
    AccStatsForUtterance(utterance_length, *chunk_info);
    void UtteranceSplitter::AccStatsForUtterance(
        int32 utterance_length,
        const std::vector<ChunkTimeInfo> &chunk_info) {
      
      total_num_utterances_ += 1;
      total_input_frames_ += utterance_length;

      for (size_t c = 0; c < chunk_info.size(); c++) {
        int32 chunk_size = chunk_info[c].num_frames;

        if (c > 0) {
          int32 last_chunk_end = chunk_info[c-1].first_frame +
              chunk_info[c-1].num_frames;
          // --------- 有重叠情况 total_frames_overlap 总体重叠数量统计.
          if (last_chunk_end > chunk_info[c].first_frame)
            total_frames_overlap_ += last_chunk_end - chunk_info[c].first_frame;
        }
        
        std::map<int32, int32>::iterator iter = chunk_size_to_count_.find(chunk_size);

        // ----- 增加 chunks_size_to_count_<chunk_size, count> 的统计计数
        if (iter == chunk_size_to_count_.end())
          chunk_size_to_count_[chunk_size] = 1;
        else
          iter->second++;

        // 增加统计计数
        total_num_chunks_ += 1;
        total_frames_in_chunks_ += chunk_size;
      }
    }

    
    // check that the end of the last chunk doesn't go more than
    // 'config_.frame_subsampling_factor - 1' frames past the end
    // of the utterance.  That amount, we treat as rounding error.
    KALDI_ASSERT(t - utterance_length < config_.frame_subsampling_factor);
  }


  // ------------ check: 是否一个utterance 没有分配成功一个chunk划分,
  // ------------ 对一个utterance的 chunk 划分结果 就是 egs.
  if (chunks.empty()) {
    KALDI_WARN << "Not producing egs for utterance " << utt_id
               << " because it is too short: "
               << num_input_frames << " frames.";
  }


  

  // ---------- comment: frame_subsampling_factor 当前没有作用, 是为了支持使用chain 方法的一些用处保留的.
  // 'frame_subsampling_factor' is not used in any recipes at the time of
  // writing, this is being supported to unify the code with the 'chain' recipes
  // and in case we need it for some reason in future.
  int32 frame_subsampling_factor =
      utt_splitter->Config().frame_subsampling_factor;








  
  // ===================== 一个chunk(+ C L R context) 作为TDNN的输入样本 =================
  // ----------------------   这个chunk 本身是 MFCC, 还会可能附加 ivector.联合做TDNN的输入样本
  // 将utterance 划分为chunk后, chunk就代表一个训练用的输入样本
  for (size_t c = 0; c < chunks.size(); c++) {
    const ChunkTimeInfo &chunk = chunks[c];
    // TDNN输入样本 == 当前chunk的frames + L context frames + R context frames
    int32 tot_input_frames = chunk.left_context + chunk.num_frames + chunk.right_context;
    // TDNN输入起始帧(但是并不是实际代表当前代表的数据, 而是以L 上下文开始, 共同描述当前代表的音素状态)
    int32 start_frame = chunk.first_frame - chunk.left_context;
    // input_frames 抽取 原始特征中 对应当前chunk eg 的L C R 的行.
    GeneralMatrix input_frames;
    ExtractRowRangeWithPadding(feats, start_frame, tot_input_frames, &input_frames);


    
    // 'input_frames' now stores the relevant rows (maybe with padding) from the
    // original Matrix or (more likely) CompressedMatrix.
    // If a CompressedMatrix, it does this without un-compressing and re-compressing, so there is no loss
    // of accuracy.



    
    // ================ 构造eg NNet 训练用输入样本 ======================
    // ================ 构造eg NNet 训练用输入样本 ======================
    // ================ 构造eg NNet 训练用输入样本 ======================
    
    // 通过NnetIo 表示一个NNet的输入, 使用的还是上面的MFCC 特征 ==============
    // ------------- 内部使用 Index 索引描述数据.
    NnetExample eg;
    // call the regular input "input".
    eg.io.push_back(NnetIo("input", -chunk.left_context, input_frames));



    
    NnetIo::NnetIo(const std::string &name,
                   int32 t_begin, const GeneralMatrix &feats,
                   int32 t_stride = 1):
        name(name), features(feats) {
      
      int32 num_rows = feats.NumRows();

      // indexes 代表所有帧, indexes[i], 代表某一帧.
      // indexes vector<Index> ------- 前面提到的 index 描述一个输入帧的索引.
      // struct Index {
      //     int32 n;  // member-index of minibatch, or zero.      ------------------??? minibatch 批次索引, 没用上.
      //     int32 t;  // time-frame.                              ------------------代表当前帧的时间.
      //     int32 x;  // this may come in useful in convoluational approaches.
      // ..........
      // }
      indexes.resize(num_rows);  // sets all n,t,x to zeros.
      for (int32 i = 0; i < num_rows; i++)
        indexes[i].t = t_begin + i * t_stride;
    }
    


    // ================ 上面都是MFCC特征, 这里将 ivector特征 加入作为eg的一部分输入  =================
    if (ivector_feats != NULL) {
      // if applicable, add the iVector feature.
      // choose iVector from a random frame in the chunk
      int32 ivector_frame = RandInt(start_frame,
                                    start_frame + num_input_frames - 1),
          ivector_frame_subsampled = ivector_frame / ivector_period;

      // check:
      if (ivector_frame_subsampled < 0)
        ivector_frame_subsampled = 0;
      if (ivector_frame_subsampled >= ivector_feats->NumRows())
        ivector_frame_subsampled = ivector_feats->NumRows() - 1;
      
      Matrix<BaseFloat> ivector(1, ivector_feats->NumCols());
      ivector.Row(0).CopyFromVec(ivector_feats->Row(ivector_frame_subsampled));
      eg.io.push_back(NnetIo("ivector", 0, ivector));
    }

    // --------- Notice:  这里一直都考虑 frame_subsampling_factor 但是这里 就是1
    // start_frame_subsampled = chunk.first_frame
    // num_frames_subsampled = chunk.num_frames = 8
    int32
        start_frame_subsampled = chunk.first_frame / frame_subsampling_factor,
        num_frames_subsampled = chunk.num_frames / frame_subsampling_factor;


    // ================  完善一个 chunk-eg-NNet训练样本, 加上对应的 labels ==============
    // ==================  构建对应的chunk(eg)的样本标签
    Posterior labels(num_frames_subsampled);
    
    for (int32 i = 0; i < num_frames_subsampled; i++) {
      int32 t = i + start_frame_subsampled;
      if (t < pdf_post.size())
        labels[i] = pdf_post[t];
      // foreach: 每个pdf的可能概率, 都 乘以 当前chunk的 chunk.output_weights[i] 权重( -帧重叠权重-).
      for (std::vector<std::pair<int32, BaseFloat> >::iterator
               iter = labels[i].begin(); iter != labels[i].end(); ++iter)
        iter->second *= chunk.output_weights[i];
    }
    // ----------   注意一个 chunk的num_frames=8 所以 labels 
    // ----------   所以 labels是  8 x num_pdfs 的矩阵.
    eg.io.push_back(NnetIo("output", num_pdfs, 0, labels));


    
    NnetIo::NnetIo(const std::string &name,
                   int32 dim,
                   int32 t_begin,
                   const Posterior &labels,
                   int32 t_stride=1):
        name(name) {
      int32 num_rows = labels.size();
      KALDI_ASSERT(num_rows > 0);
      SparseMatrix<BaseFloat> sparse_feats(dim, labels);
      
      features = sparse_feats;
      // ----------- indexes[i] 代表每一帧,
      // ----------- indexes[i].t 代表当前帧的实际time.
      indexes.resize(num_rows);  // sets all n,t,x to zeros.
      for (int32 i = 0; i < num_rows; i++)
        indexes[i].t = t_begin + i * t_stride;
    }

    if (compress)
      eg.Compress();

    std::ostringstream os;
    os << utt_id << "-" << chunk.first_frame;

    std::string key = os.str(); // key is <utt_id>-<frame_id>

    example_writer->Write(key, eg);
  }
  return true;
}


  
}
