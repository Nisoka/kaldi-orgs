

// $decoder --cmd "$decode_cmd" --nj $nj $srcdir/graph_word $datadir/test $srcdir/decode_test_word || exit 1



// graphdir=$1      图路径 -- exp/mono/graph_word  刚刚mkgraph 构建得到的HCLG.FST
// data=$2          数据  --  data/mfcc/test
// dir=$3           输出结果/log --  exp/mono/decode_test_word.

// srcdir=`dirname $dir`;   
// sdata=$data/split$nj;

//  一般结果是 final.mdl 上面得到的转移模型
// if [ -z "$model" ]; then # if --model <mdl> was not specified on the command line...
//   if [ -z $iter ]; then
//      model=$srcdir/final.mdl;
//   else
//      model=$srcdir/$iter.mdl;
//   fi
// fi


// 特征 、 cmvn（应用说话人特征均值倒谱系数）   model？现在无定义  当前图HCLG.FST.  
// for f in $sdata/1/feats.scp $sdata/1/cmvn.scp $model $graphdir/HCLG.fst; do
//   [ ! -f $f ] && echo "decode.sh: no such file $f" && exit 1;
// done

// srcdir  exp/mono/
// splice_opts=`cat $srcdir/splice_opts 2>/dev/null` # frame-splicing options.
// cmvn_opts=`cat $srcdir/cmvn_opts 2>/dev/null`
// delta_opts=`cat $srcdir/delta_opts 2>/dev/null`




//     gmm-latgen-faster$thread_string --max-active=$max_active --beam=$beam --lattice-beam=$lattice_beam \
//     --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=$graphdir/words.txt $decode_extra_opts
//          $model $graphdir/HCLG.fst "$feats" "ark:|gzip -c > $dir/lat.JOB.gz" || exit 1;

int gmm_latgen_faster(int argc, char *argv[]) {
    const char *usage =
        "Generate lattices using GMM-based model.\n"
        "Usage: gmm-latgen-faster [options] model-in (fst-in|fsts-rspecifier) features-rspecifier"
        " lattice-wspecifier [ words-wspecifier [alignments-wspecifier] ]\n";
    ParseOptions po(usage);
    Timer timer;
    bool allow_partial = false;
    BaseFloat acoustic_scale = 0.1;
    LatticeFasterDecoderConfig config;

    std::string word_syms_filename;
    config.Register(&po);
    po.Register("acoustic-scale", &acoustic_scale,
                "Scaling factor for acoustic likelihoods");
    po.Register("word-symbol-table", &word_syms_filename,
                "Symbol table for words [for debug output]");
    po.Register("allow-partial", &allow_partial,
                "If true, produce output even if end state was not reached.");

    std::string
        model_in_filename = po.GetArg(1),
        fst_in_str = po.GetArg(2),
        feature_rspecifier = po.GetArg(3),
        
        lattice_wspecifier = po.GetArg(4),
        words_wspecifier = po.GetOptArg(5),
        alignment_wspecifier = po.GetOptArg(6);

    TransitionModel trans_model;
    AmDiagGmm am_gmm;
    {
      bool binary;
      Input ki(model_in_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_gmm.Read(ki.Stream(), binary);
    }

    bool determinize = config.determinize_lattice;
    CompactLatticeWriter compact_lattice_writer;
    LatticeWriter lattice_writer;
    
    if (! (determinize ? compact_lattice_writer.Open(lattice_wspecifier)
           : lattice_writer.Open(lattice_wspecifier)))
      KALDI_ERR << "Could not open table for writing lattices: "
                 << lattice_wspecifier;

    Int32VectorWriter words_writer(words_wspecifier);
    Int32VectorWriter alignment_writer(alignment_wspecifier);
    fst::SymbolTable *word_syms = NULL;
    if (word_syms_filename != "")
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename)))
        KALDI_ERR << "Could not read symbol table from file "
                   << word_syms_filename;

    double tot_like = 0.0;
    kaldi::int64 frame_count = 0;
    int num_done = 0, num_err = 0;

    if (ClassifyRspecifier(fst_in_str, NULL, NULL) == kNoRspecifier) {
      
      SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);

      // Input FST is just one FST, not a table of FSTs.
      Fst<StdArc> *decode_fst = fst::ReadFstKaldiGeneric(fst_in_str);
      timer.Reset();

      {
        // 定义解码器, HCLG图. 
        LatticeFasterDecoder decoder(*decode_fst, config);

        // foreach utt
        for (; !feature_reader.Done(); feature_reader.Next()) {
          std::string utt = feature_reader.Key();
          
          Matrix<BaseFloat> features (feature_reader.Value());
          
          feature_reader.FreeCurrent();
          // 解码所需统计量  所有gmm参数 转移模型 utt特征， 声学部分权重扩展.
          DecodableAmDiagGmmScaled gmm_decodable(am_gmm, trans_model, features,
                                                 acoustic_scale);

          double like;
          if (DecodeUtteranceLatticeFaster(
                  decoder, gmm_decodable, trans_model, word_syms, utt,
                  acoustic_scale, determinize, allow_partial, &alignment_writer,
                  &words_writer, &compact_lattice_writer, &lattice_writer,
                  &like)) {
            tot_like += like;
            frame_count += features.NumRows();
            num_done++;
          }
          else
            num_err++;
          
        }
        
      }
      
      delete decode_fst; // delete this only after decoder goes out of scope.
      
    }     
}


// in
// decoder 解码器
// decodable 解码统计量
// trans_model 转移模型


bool DecodeUtteranceLatticeFaster(
    LatticeFasterDecoder &decoder, // not const but is really an input.
    DecodableInterface &decodable, // not const but is really an input.
    const TransitionModel &trans_model,
    const fst::SymbolTable *word_syms,
    std::string utt,

    double acoustic_scale,
    bool determinize,
    bool allow_partial,
    
    Int32VectorWriter *alignment_writer,
    Int32VectorWriter *words_writer,
    CompactLatticeWriter *compact_lattice_writer,
    LatticeWriter *lattice_writer,
    double *like_ptr) { // puts utterance's like in like_ptr on success.

  
  using fst::VectorFst;

  // 执行解码
  if (!decoder.Decode(&decodable)) {
    KALDI_WARN << "Failed to decode file " << utt;
    return false;
  }
  // 判断解码结果
  if (!decoder.ReachedFinal()) {
    // 允许部分解码, 一般是不正确的.
    if (allow_partial) {
      KALDI_WARN << "Outputting partial output for utterance " << utt
                 << " since no final-state reached\n";
    } else {
      KALDI_WARN << "Not producing output for utterance " << utt
                 << " since no final-state reached and "
                 << "--allow-partial=false.\n";
      return false;
    }
  }

  double likelihood;
  LatticeWeight weight;
  int32 num_frames;

  // First do some stuff with word-level traceback...
  {
    // 解码结果
    VectorFst<LatticeArc> decoded;
    if (!decoder.GetBestPath(&decoded))
      // Shouldn't really reach this point as already checked success.
      KALDI_ERR << "Failed to get traceback for utterance " << utt;

    std::vector<int32> alignment;
    std::vector<int32> words;
    GetLinearSymbolSequence(decoded, &alignment, &words, &weight);
    num_frames = alignment.size();
    
    if (words_writer->IsOpen())
      words_writer->Write(utt, words);
    
    if (alignment_writer->IsOpen())
      alignment_writer->Write(utt, alignment);
    
    if (word_syms != NULL) {
      std::cerr << utt << ' ';
      for (size_t i = 0; i < words.size(); i++) {
        std::string s = word_syms->Find(words[i]);
        if (s == "")
          KALDI_ERR << "Word-id " << words[i] << " not in symbol table.";
        std::cerr << s << ' ';
      }
      std::cerr << '\n';
    }
    likelihood = -(weight.Value1() + weight.Value2());
  }

  // Get lattice, and do determinization if requested.
  Lattice lat;
  decoder.GetRawLattice(&lat);
  if (lat.NumStates() == 0)
    KALDI_ERR << "Unexpected problem getting lattice for utterance " << utt;
  fst::Connect(&lat);
  if (determinize) {
    CompactLattice clat;
    if (!DeterminizeLatticePhonePrunedWrapper(
            trans_model,
            &lat,
            decoder.GetOptions().lattice_beam,
            &clat,
            decoder.GetOptions().det_opts))
      KALDI_WARN << "Determinization finished earlier than the beam for "
                 << "utterance " << utt;
    // We'll write the lattice without acoustic scaling.
    if (acoustic_scale != 0.0)
      fst::ScaleLattice(fst::AcousticLatticeScale(1.0 / acoustic_scale), &clat);
    compact_lattice_writer->Write(utt, clat);
  } else {
    // We'll write the lattice without acoustic scaling.
    if (acoustic_scale != 0.0)
      fst::ScaleLattice(fst::AcousticLatticeScale(1.0 / acoustic_scale), &lat);
    lattice_writer->Write(utt, lat);
  }
  KALDI_LOG << "Log-like per frame for utterance " << utt << " is "
            << (likelihood / num_frames) << " over "
            << num_frames << " frames.";
  KALDI_VLOG(2) << "Cost for utterance " << utt << " is "
                << weight.Value1() << " + " << weight.Value2();
  *like_ptr = likelihood;
  return true;
}



// Returns true if any kind of traceback is available
// (not necessarily from a final state).  It should only very rarely return false; this indicates an unusual search error.
bool LatticeFasterDecoder::Decode(DecodableInterface *decodable) {
  InitDecoding();

  while (!decodable->IsLastFrame(NumFramesDecoded() - 1)) {
    if (NumFramesDecoded() % config_.prune_interval == 0)
      PruneActiveTokens(config_.lattice_beam * config_.prune_scale);
    BaseFloat cost_cutoff = ProcessEmitting(decodable);
    ProcessNonemitting(cost_cutoff);
  }
  FinalizeDecoding();

  return !active_toks_.empty() && active_toks_.back().toks != NULL;
}


void LatticeFasterDecoder::InitDecoding() {
  // clean up from last time:
  DeleteElems(toks_.Clear());
  
  cost_offsets_.clear();
  ClearActiveTokens();
  
  warned_ = false;
  num_toks_ = 0;
  decoding_finalized_ = false;
  final_costs_.clear();
  
  // fst 的初始状态节点
  StateId start_state = fst_.Start();

  // 有效tokens std::vector<TokenList> active_toks_,
  // 通过时间frame索引, 所以每个TokenList代表 某个时间的可能token.
  active_toks_.resize(1);
  
  // 构建Token
  Token *start_tok = new Token(0.0, 0.0, NULL, NULL);
  // active_toks_ 有效token
  // 向时间帧0的有效tokenList 中加入初始Token.
  active_toks_[0].toks = start_tok;
  // 最后通过active_toks 保存一个lattice网络。
  
  // 向toks_插入初始Token.
  toks_.Insert(start_state, start_tok);
  num_toks_++;
  ProcessNonemitting(config_.beam);
}


void LatticeFasterDecoder::ProcessNonemitting(BaseFloat cutoff) {
  KALDI_ASSERT(!active_toks_.empty());
  
  int32 frame = static_cast<int32>(active_toks_.size()) - 2;
  // Note: "frame" is the time-index we just processed, or -1 if
  // we are processing the nonemitting transitions before the
  // first frame (called from InitDecoding()).

  // Processes nonemitting arcs for one frame.  Propagates within toks_.
  // Note-- this queue structure is is not very optimal as
  // it may cause us to process states unnecessarily (e.g. more than once),
  // but in the baseline code, turning this vector into a set to fix this
  // problem did not improve overall speed.

  KALDI_ASSERT(queue_.empty());
  
  for (const Elem *e = toks_.GetList(); e != NULL;  e = e->tail)
    queue_.push_back(e->key);
  
  while (!queue_.empty()) {
    StateId state = queue_.back();
    queue_.pop_back();

    // 获得待扩展token
    Token *tok = toks_.Find(state)->val;  // would segfault if state not in toks_ but this can't happen.
    BaseFloat cur_cost = tok->tot_cost;
    
    if (cur_cost > cutoff) // Don't bother processing successors.
      continue;

    // 如果token 存在前向link, 删除它们, 因为会需要重建token的前向link.
    // 这是一种非优化操作，但是因为这个是个简单解码器, 并且状态大多是emitting的
    // 所以这不是大问题.
    // If "tok" has any existing forward links, delete them,
    // because we're about to regenerate them.  This is a kind
    // of non-optimality (remember, this is the simple decoder),
    // but since most states are emitting it's not a huge issue.
    
    tok->DeleteForwardLinks(); // necessary when re-visiting
    tok->links = NULL;

    // Fst 中 state出发的所有可能Arc
    for (fst::ArcIterator<fst::Fst<Arc> > aiter(fst_, state);
         !aiter.Done();
         aiter.Next()) {
      
      const Arc &arc = aiter.Value();
      // 扩展传播非emitting转移 ---- 
      if (arc.ilabel == 0) {  // propagate nonemitting only...
        BaseFloat
            graph_cost = arc.weight.Value(),
            tot_cost = cur_cost + graph_cost;
        // 在剪枝范围内.
        if (tot_cost < cutoff) {
          bool changed;
          // 增加新的 Token.
          Token *new_tok = FindOrAddToken(arc.nextstate, frame + 1, tot_cost, &changed);

          // 构建前向link 目标token 是根据转移Arc新建的Token，权重
          tok->links = new ForwardLink(new_tok, 0, arc.olabel,
                                       graph_cost, 0, tok->links);

          // "changed" tells us whether the new token has a different
          // cost from before, or is new [if so, add into queue].
          if (changed)
            queue_.push_back(arc.nextstate);
        }
      }
    } // for all arcs
  } // while queue not empty
}

// in
// state
// frame_plus_one 当前帧id+1 进行扩展解码
// tot_cost       当前可能转移弧Arc 累计上后得到的权重
// out
// change   是否发生变化？？

inline LatticeFasterDecoder::Token *LatticeFasterDecoder::FindOrAddToken(
    StateId state, int32 frame_plus_one, BaseFloat tot_cost, bool *changed) {
  
  // Returns the Token pointer.  Sets "changed" (if non-NULL) to true
  // if the token was newly created or the cost changed.
  
  KALDI_ASSERT(frame_plus_one < active_toks_.size());
  
  Token *&toks = active_toks_[frame_plus_one].toks;
  // 在toks_ 中查看是否已经存在了 目标状态.
  Elem *e_found = toks_.Find(state);
  if (e_found == NULL) {  // no such token presently.
    const BaseFloat extra_cost = 0.0;
    // tokens on the currently final frame have zero extra_cost
    // as any of them could end up on the winning path.
    Token *new_tok = new Token (tot_cost, extra_cost, NULL, toks);
    // NULL: no forward links yet
    toks = new_tok;
    num_toks_++;
    toks_.Insert(state, new_tok);
    if (changed) *changed = true;
    return new_tok;
  } else {
    Token *tok = e_found->val;  // There is an existing Token for this state.
    if (tok->tot_cost > tot_cost) {  // replace old token
      tok->tot_cost = tot_cost;
      // we don't allocate a new token, the old stays linked in active_toks_
      // we only replace the tot_cost
      // in the current frame, there are no forward links (and no extra_cost)
      // only in ProcessNonemitting we have to delete forward links
      // in case we visit a state for the second time
      // those forward links, that lead to this replaced token before:
      // they remain and will hopefully be pruned later (PruneForwardLinks...)
      if (changed) *changed = true;
    } else {
      if (changed) *changed = false;
    }
    return tok;
  }
}
