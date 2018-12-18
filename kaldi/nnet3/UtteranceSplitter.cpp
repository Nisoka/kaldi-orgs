ExampleGenerationConfig::ExampleGenerationConfig()
    :left_context(0), right_context(0),
     left_context_initial(-1), right_context_final(-1),
     num_frames_overlap(0), frame_subsampling_factor(1),
     num_frames_str("1") {
}
ExampleGenerationConfig::Register()
{
     // maybe 8
     po->Register("left-context", &left_context, "Number of frames of left "
                  "context of input features that are added to each "
                  "example");
     // maybe 8
     po->Register("right-context", &right_context, "Number of frames of right "
                  "context of input features that are added to each "
                  "example");
     
     po->Register("left-context-initial", &left_context_initial, "Number of "
                  "frames of left context of input features that are added to "
                  "each example at the start of the utterance (if <0, this "
                  "defaults to the same as --left-context)");
     po->Register("right-context-final", &right_context_final, "Number of "
                  "frames of right context of input features that are added "
                  "to each example at the end of the utterance (if <0, this "
                  "defaults to the same as --right-context)");
     // THIS IS SET TO BE     --- 8
     po->Register("num-frames", &num_frames_str, "Number of frames with labels "
                 "that each example contains (i.e. the left and right context "
                 "are to be added to this).  May just be an integer (e.g. "
                 "--num-frames=8), or a principal value followed by "
                 "alternative values to be used at most once for each utterance "
                 "to deal with odd-sized input, e.g. --num-frames=40,25,50 means "
                 "that most of the time the number of frames will be 40, but to "
                 "deal with odd-sized inputs we may also generate egs with these "
                 "other sizes.  All these values will be rounded up to the "
                 "closest multiple of --frame-subsampling-factor.  As a special case, "
                 "--num-frames=-1 means 'don't do any splitting'.");
     po->Register("num-frames-overlap", &num_frames_overlap, "Number of frames of "
                  "overlap between adjacent eamples (applies to chunks of size "
                  "equal to the primary [first-listed] --num-frames value... "
                  "will be adjusted for different-sized chunks).  Advisory; "
                  "will not be exactly enforced.");
     // THIS IS SET TO BE     --- 1
     po->Register("frame-subsampling-factor", &frame_subsampling_factor, "Used "
                  "if the frame-rate of the output labels in the generated "
                  "examples will be less than the frame-rate at the input");
  }

ExampleGenerationConfig::ComputeDerived()
{
  if (num_frames_str == "-1") {
     return;
   }
   if (!SplitStringToIntegers(num_frames_str, ",", false, &num_frames) ||
       num_frames.empty()) {
     KALDI_ERR << "Invalid option (expected comma-separated list of integers): "
               << "--num-frames=" << num_frames_str;
   }
 
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
}



/*
  This comment describes the idea behind what InitChunkSize() is supposed to do,
  and how it relates to the purpose of class UtteranceSplitter.

  Class UtteranceSplitter is supposed to tell us, for a given utterance length,
  what chunk sizes to use.  The chunk sizes it may choose are:h
    - zero or more chunks of the 'principal' size (the first-listed value in
      --num-frames option)
    - at most two chunks of 'alternative' num-frames (meaning, any but the
      first-listed choice in the --num-frames option).

  (note: an empty list of chunks is not allowed as a split).  A split is
  a list of chunk-sizes in increasing order (we when we actually split the
  utterance into chunks, we may, at random, reverse the order.

  The choice of split to use for a given utterance-length is determined as
  follows.  Firstly, for each split we compute a 'default duration' (see
  DefaultDurationOfSplit()... if --num-frames-overlap is zero, this is just the
  sum of the chunk sizes).  We then use by a cost-function that depends on
  default-duration and the length of the utterance: the idea is that these two
  should be as close as possible, but penalizing the default-duration being
  larger than the utterance-length (which in the normal case of
  --num-frames-overlap=0 would lead to gaps between the segments), twice as much
  as the other sign of difference.

  Specifically:
    cost(default_duration, utt_length) = (default_duration > utt_length ?
                                         default_duration - utt_length :
                                         2.0 * (utt_length - default_duration))
  [but as a special case, set c to infinity if the largest chunk size in the
   split is longer than the utterance length; we couldn't, in that case, use
   this split for this utterance].

  We want to make sure a good variety of combinations of chunk sizes are chosen
  in case there are ties from the cost function.  For each utterance length
  we store the set of splits, whose costs are within 2
  of the best cost available for that utterance length.  When asked to find
  chunks for a particular utterance of that length, we will choose randomly
  from that pool of splits.
 */



// frames_per_eg = 8
// num_frames = 8
// out:
// primary_length = 8
// max_length = 8
// maxUtteranceLength = 24

int32 UtteranceSplitter::MaxUtteranceLength() const {
  // config_.num_frames is [8]
  // so num_lengths == 1
  int32 num_lengths = config_.num_frames.size();
  
  KALDI_ASSERT(num_lengths > 0);
  // 'primary_length' is the first-specified num-frames.
  // It's the only chunk that may be repeated an arbitrary number
  // of times.
  // config_.num_frames = [8]
  int32 primary_length = config_.num_frames[0],
      max_length = primary_length;
  
  for (int32 i = 0; i < num_lengths; i++) {
    KALDI_ASSERT(config_.num_frames[i] > 0);
    max_length = std::max(config_.num_frames[i], max_length);
  }
  // return 2*8 + 8 = 24
  return 2 * max_length + primary_length;
}

void UtteranceSplitter::InitSplits(std::vector<std::vector<int32> > *splits) const {
  // we consider splits whose default duration (as returned by
  // DefaultDurationOfSplit()) is up to MaxUtteranceLength() + primary_length.
  
  // We can be confident without doing a lot of math, that splits above this
  // length will never be chosen for any utterance-length up to
  // MaxUtteranceLength() (which is the maximum we use).
  int32
      // 8
      primary_length = config_.num_frames[0],
      // 32 = 24 + 8
      default_duration_ceiling = MaxUtteranceLength() + primary_length;

  typedef unordered_set<std::vector<int32>, VectorHasher<int32> > SetType;

  SetType splits_set;

  // num_lengths=1 
  int32 num_lengths = config_.num_frames.size();



  // ??????????????????????????????
  // The splits we are allow are: zero to two 'alternate' lengths, plus
  // an arbitrary number of repeats of the 'primary' length.  The repeats
  // of the 'primary' length are handled by the inner loop over n.
  // The zero to two 'alternate' lengths are handled by the loops over
  // i and j.  i == 0 and j == 0 are special cases; they mean, no
  // alternate is chosen.

  // ------------------------ the splits will be:
  // [8] , [8 8], [8 8 8], [8 8 8 8]
  
  for (int32 i = 0; i < num_lengths; i++) {
    for (int32 j = 0; j < num_lengths; j++) {
      std::vector<int32> vec;
      if (i > 0)
        vec.push_back(config_.num_frames[i]);
      if (j > 0)
        vec.push_back(config_.num_frames[j]);

      int32 n = 0;

      // ret     0   0    16         24                32   > 24
      // split   .   8    8,(8,8)    8,(8,8),(8,8,8)
      // vec     8   8,8  8,8,8      8,8,8,8
      while (DefaultDurationOfSplit(vec) <= default_duration_ceiling) {

        // -------------------------return the vec.accumulate()
        float UtteranceSplitter::DefaultDurationOfSplit(const std::vector<int32> &split) const {
          if (split.empty())  // not a valid split, but useful to handle this case.
            return 0.0;
          float
              // 8
              principal_num_frames = config_.num_frames[0],
              // 0  重叠frames数量.
              num_frames_overlap = config_.num_frames_overlap;
          
          KALDI_ASSERT(num_frames_overlap < principal_num_frames &&
                       "--num-frames-overlap value is too high");
          // 重叠比例. 0
          float overlap_proportion = num_frames_overlap / principal_num_frames;
          // 累加  ans = 8, 16, 24
          float ans = std::accumulate(split.begin(), split.end(), int32(0));
          
          for (size_t i = 0; i + 1 < split.size(); i++) {
            float min_adjacent_chunk_length = std::min(split[i], split[i + 1]),
                overlap = overlap_proportion * min_adjacent_chunk_length;
            ans -= overlap;
          }
          KALDI_ASSERT(ans > 0.0);
          return ans;
        }  // end the function
        // -------------------------

        
       
        if (!vec.empty()) // Don't allow the empty vector as a split.
          splits_set.insert(vec);
        n++;
        vec.push_back(primary_length);
        std::sort(vec.begin(), vec.end());
      }
    }
  }

  // Then the splits_set ----- 8, (8,8), (8,8,8)
  for (SetType::const_iterator iter = splits_set.begin();
       iter != splits_set.end(); ++iter)
    splits->push_back(*iter);
  std::sort(splits->begin(), splits->end());

  // make the order deterministic, for consistency of output between runs and C libraries.
}

void UtteranceSplitter::InitSplitForLength() {
  // 24
  int32 max_utterance_length = MaxUtteranceLength();

  // The 'splits' vector is a list of possible splits
  // (a split being a sorted vector of chunk-sizes).
  // The vector 'splits' is itself sorted.
  
  
  std::vector<std::vector<int32> > splits;
  // 初始化spd li
  InitSplits(&splits);


  // 定义一个 split划分可能的 index 用index来选择使用哪种split划分.
  // Define a split-index 0 <= s < splits.size() as index into the 'splits'vector,

  // 定义一个 cost 表示在一个split划分中 utt-length 和 chunksizes 总长度之间的差别. ???
  // and let a cost c >= 0 represent the mismatch between an
  // utterance length and the total length of the chunk sizes in a split:

  //  c(default_duration, utt_length) = (default_duration > utt_length ?
  //                                    default_duration - utt_length :
  //                                    2.0 * (utt_length - default_duration))
  
  // [but as a special case, set c to infinity if the largest chunk size in the
  //  split is longer than the utterance length; we couldn't, in that case, use
  //  this split for this utterance].

  // 'costs_for_length[u][s]', indexed by utterance-length u and then split,
  // contains the cost for utterance-length u and split s.

  
  // splits.size = 4 ([8], [8 8], [8 8 8], [8 8 8 8])
  // max_utt_length = 24+1
  std::vector<std::vector<float> > costs_for_length(max_utterance_length + 1);
  int32 num_splits = splits.size();

  // 对每个costs_for_length[index] 设置为 splits
  
  // max_utterance_length is 24
  for (int32 u = 0; u <= max_utterance_length; u++)
    costs_for_length[u].reserve(num_splits);

  // foreach split可能划分.
  for (int32 s = 0; s < num_splits; s++) {
    const std::vector<int32> &split = splits[s];

    // 8, 16, 24, 32
    float default_duration = DefaultDurationOfSplit(split);

    // alway 8
    int32 max_chunk_size = *std::max_element(split.begin(), split.end());
    // 每种utt长度
    for (int32 u = 0; u <= max_utterance_length; u++) {
      // c 是对这样的长度的utt使用该种划分方式，出现的损失。我们和overlaps一样惩罚两次间隔gaps，
      // 给予这样的原因，完全扔掉的帧数， 比两次计算他们更坏？？
      // c is the cost for this utterance length and this split.  We penalize
      // gaps twice as strongly as overlaps, based on the intuition that
      // completely throwing out frames of data is worse than counting them
      // twice.
      // max_utterance_length is 24, so, usuraly default_duration - u
      // 8 16 24 32
      float c = (default_duration > float(u) ? default_duration - float(u) :  2.0 * (u - default_duration));
      if (u < max_chunk_size)  // can't fit the largest of the chunks in this utterance
        c = std::numeric_limits<float>::max();
      KALDI_ASSERT(c >= 0);
      // 表示 不同长度的utt 使用不同中划分方式，带来的损害
      costs_for_length[u].push_back(c);
    }
  }

  // 用来真正选择某个utt长度，需要使用某种划分方式
  splits_for_length_.resize(max_utterance_length + 1);

  // 0 - 24
  for (int32 u = 0; u <= max_utterance_length; u++) {
    const std::vector<float> &costs = costs_for_length[u];
    // get min cose
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
    // 每个utt长度 都选择几个损耗较小的划分方式.
    for (; iter != end; ++iter,++s)
      if (*iter < min_cost + cost_threshold)
        splits_for_length_[u].push_back(splits[s]);
  }

  if (GetVerboseLevel() >= 3)  {
    std::ostringstream os;
    for (int32 u = 0; u <= max_utterance_length; u++) {
      if (!splits_for_length_[u].empty()) {
        os << u << "=(";
        std::vector<std::vector<int32 > >::const_iterator
            iter1 = splits_for_length_[u].begin(),
            end1 = splits_for_length_[u].end();

        while (iter1 != end1) {
          std::vector<int32>::const_iterator iter2 = iter1->begin(),
              end2 = iter1->end();
          while (iter2 != end2) {
            os << *iter2;
            ++iter2;
            if (iter2 != end2) os << ",";
          }
          ++iter1;
          if (iter1 != end1) os << "/";
        }
        os << ")";
        if (u < max_utterance_length) os << ", ";
      }
    }
    KALDI_VLOG(3) << "Utterance-length-to-splits map is: " << os.str();
  }
}

