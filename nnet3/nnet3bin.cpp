


void nnet3-get-egs()
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



  int main(int argc, char *argv[]) {

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

    UtteranceSplitter utt_splitter(eg_config);

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

        // prcocess frames in current utt.

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


  
  std::vector<ChunkTimeInfo> chunks;

  utt_splitter->GetChunksForUtterance(num_input_frames, &chunks);


  void UtteranceSplitter::GetChunksForUtterance(
      int32 utterance_length,
      std::vector<ChunkTimeInfo> *chunk_info) {
    
    std::vector<int32> chunk_sizes;

    // 生成一个 对utterance_length 的可能划分, chunk_sizes 是划分结果(primary, primary, ... primary, other)
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

      // 循环切分utterance_length, 以 primary_length repeat,
      // 直到 最后剩余utterance_length < splits_for_length_.size() 长度.
      while (utterance_length > max_tabulated_length) {
        utterance_length -= (primary_length - num_frames_overlap);
        num_primary_length_repeats++;
      }
      
      KALDI_ASSERT(utterance_length >= 0);

      // 可能的划分可能.
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

      // <8>, <8,8>
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
    GetGapSizes(utterance_length, true, chunk_sizes, &gaps);
    void UtteranceSplitter::GetGapSizes(int32 utterance_length,
                                        bool enforce_subsampling_factor,
                                        const std::vector<int32> &chunk_sizes,
                                        std::vector<int32> *gap_sizes) const {
      if (chunk_sizes.empty()) {
        gap_sizes->clear();
        return;
      }
      
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
      
      int32 num_chunks = chunk_sizes.size(),
          total_of_chunk_sizes = std::accumulate(chunk_sizes.begin(),
                                                 chunk_sizes.end(),
                                                 int32(0)),
          total_gap = utterance_length - total_of_chunk_sizes;
      gap_sizes->resize(num_chunks);

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
        std::vector<int32> magnitudes(num_chunks - 1),
            overlaps(num_chunks - 1);
        // the 'magnitudes' vector will contain the minimum of the lengths of the
        // two adjacent chunks between which are are going to consider having an
        // overlap.  These will be used to assign the overlap proportional to that
        // size.
        for (int32 i = 0; i + 1 < num_chunks; i++) {
          magnitudes[i] = std::min<int32>(chunk_sizes[i], chunk_sizes[i + 1]);
        }


        
        DistributeRandomly(total_gap, magnitudes, &overlaps);

        // 将空白部分, 按照magnitudes向量各元素比例 分化空白 ==> overlaps.
        void UtteranceSplitter::DistributeRandomly(int32 n,
                                                   const std::vector<int32> &magnitudes,
                                                   std::vector<int32> *vec) {
          KALDI_ASSERT(!vec->empty() && vec->size() == magnitudes.size());
          int32 size = vec->size();
          if (n < 0) {
            DistributeRandomly(-n, magnitudes, vec);
            for (int32 i = 0; i < size; i++)
              (*vec)[i] *= -1;
            return;
          }
          float total_magnitude = std::accumulate(magnitudes.begin(), magnitudes.end(),
                                                  int32(0));
          KALDI_ASSERT(total_magnitude > 0);
          // note: 'partial_counts' contains the negative of the partial counts, so
          // when we sort the larger partial counts come first.
          std::vector<std::pair<float, int32> > partial_counts;
          int32 total_count = 0;

          // 按照 magnitude分化 空白, 每个chunk 会有 vec[i] 的空白.
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

          // 向排好序的vec中逐个增加1个空白, 知道 total_count = n.
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
      } else {
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

    // 规划 chunk 时间
    // chunk_info Vector<ChunkTimeInfo, ChunkTimeInfo ... >
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



    
    SetOutputWeights(utterance_length, chunk_info);
    AccStatsForUtterance(utterance_length, *chunk_info);
    
    // check that the end of the last chunk doesn't go more than
    // 'config_.frame_subsampling_factor - 1' frames past the end
    // of the utterance.  That amount, we treat as rounding error.
    KALDI_ASSERT(t - utterance_length < config_.frame_subsampling_factor);
  }

  
  if (chunks.empty()) {
    KALDI_WARN << "Not producing egs for utterance " << utt_id
               << " because it is too short: "
               << num_input_frames << " frames.";
  }

  // 'frame_subsampling_factor' is not used in any recipes at the time of
  // writing, this is being supported to unify the code with the 'chain' recipes
  // and in case we need it for some reason in future.
  int32 frame_subsampling_factor =
      utt_splitter->Config().frame_subsampling_factor;

  for (size_t c = 0; c < chunks.size(); c++) {
    const ChunkTimeInfo &chunk = chunks[c];

    int32 tot_input_frames = chunk.left_context + chunk.num_frames +
        chunk.right_context;

    int32 start_frame = chunk.first_frame - chunk.left_context;

    GeneralMatrix input_frames;
    ExtractRowRangeWithPadding(feats, start_frame, tot_input_frames,
                               &input_frames);

    // 'input_frames' now stores the relevant rows (maybe with padding) from the
    // original Matrix or (more likely) CompressedMatrix.  If a CompressedMatrix,
    // it does this without un-compressing and re-compressing, so there is no loss
    // of accuracy.

    NnetExample eg;
    // call the regular input "input".
    eg.io.push_back(NnetIo("input", -chunk.left_context, input_frames));

    if (ivector_feats != NULL) {
      // if applicable, add the iVector feature.
      // choose iVector from a random frame in the chunk
      int32 ivector_frame = RandInt(start_frame,
                                    start_frame + num_input_frames - 1),
          ivector_frame_subsampled = ivector_frame / ivector_period;
      if (ivector_frame_subsampled < 0)
        ivector_frame_subsampled = 0;
      if (ivector_frame_subsampled >= ivector_feats->NumRows())
        ivector_frame_subsampled = ivector_feats->NumRows() - 1;
      Matrix<BaseFloat> ivector(1, ivector_feats->NumCols());
      ivector.Row(0).CopyFromVec(ivector_feats->Row(ivector_frame_subsampled));
      eg.io.push_back(NnetIo("ivector", 0, ivector));
    }

    // Note: chunk.first_frame and chunk.num_frames will both be
    // multiples of frame_subsampling_factor.
    int32 start_frame_subsampled = chunk.first_frame / frame_subsampling_factor,
        num_frames_subsampled = chunk.num_frames / frame_subsampling_factor;

    Posterior labels(num_frames_subsampled);

    // TODO: it may be that using these weights is not actually helpful (with
    // chain training, it was not), and that setting them all to 1 is better.
    // We could add a boolean option to this program to control that; but I
    // don't want to add such an option if experiments show that it is not
    // helpful.
    for (int32 i = 0; i < num_frames_subsampled; i++) {
      int32 t = i + start_frame_subsampled;
      if (t < pdf_post.size())
        labels[i] = pdf_post[t];
      for (std::vector<std::pair<int32, BaseFloat> >::iterator
               iter = labels[i].begin(); iter != labels[i].end(); ++iter)
        iter->second *= chunk.output_weights[i];
    }

    eg.io.push_back(NnetIo("output", num_pdfs, 0, labels));

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
