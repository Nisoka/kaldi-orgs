// #MPE training

// srcdir=exp/tri4b_dnn
// acwt=0.1


//   # generate lattices and alignments
//  ====================== 生成 对齐结果
//   steps/nnet/align.sh --nj $nj --cmd "$train_cmd"                    \
//     data/fbank/train data/lang                     $srcdir                       ${srcdir}_ali 
//     train_feats      data/lang- 包含的是L.fst      dnn训练结果- exp/tri4b_dnn    exp/tri4b_dnn_ali

//  ====================== 生成 lattice  用作MPE的分母 denominator lattice
// steps/nnet/make_denlats.sh
//   --nj $nj --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt \
//   data/fbank/train        data/lang          $srcdir                ${srcdir}_denlats 
//   fbank 训练数据          发音词典等        srcdir=exp/tri4b_dnn    exp/tri4b_dnn_denlats


//  生成对齐结果, ==========> exp/tri4b_dnn_ali/ali.n.gz
void align_sh(){
  // 将数据 对其为 tid结果,使用基于神经网络的声学模型, 可选生成lattice格式的对其结果,lattice格式很容易得到word对齐结果
  // # Aligns 'data' to sequences of transition-ids using Neural Network based acoustic model.
  // # Optionally produces alignment in lattice format, this is handy to get word alignment.
 
  // # Begin configuration.
  {
    // scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
    // beam=10
    // retry_beam=40

    // nnet_forward_opts="--no-softmax=true --prior-scale=1.0"
  
    // ivector=                    # rx-specifier with i-vectors (ark-with-vectors),
    // text= # (optional) transcipts we align to,
  
    // align_to_lats=false         # optionally produce alignment in lattice format
    //  lats_decode_opts="--acoustic-scale=0.1 --beam=20 --lattice_beam=10"
    //  lats_graph_scales="--transition-scale=1.0 --self-loop-scale=0.1"

    // use_gpu="no" # yes|no|optionaly
    // # End configuration options.

  

    // if [ $# != 4 ]; then
    //    echo "usage: $0 <data-dir> <lang-dir> <src-dir> <align-dir>"
    //    echo "e.g.:  $0 data/train data/lang exp/tri1 exp/tri1_ali"
    //    echo "main options (for others, see top of script file)"
    //    echo "  --config <config-file>                           # config containing options"
    //    echo "  --nj <nj>                                        # number of parallel jobs"
    //    echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
    //    exit 1;
    // fi
  }

  // data=$1       //fbank  特征
  // lang=$2       //lang/---> L_disambig.fst  L.fst  oov.int  oov.txt  phones  phones.txt  topo  words.txt 
  // srcdir=$3     // srcdir=exp/tri4b_dnn

  // dir=$4        // srcdir=exp/tri4b_dnn_ali

  
  // mkdir -p $dir/log
  // echo $nj > $dir/num_jobs
  
  // sdata=$data/split$nj
  // [[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

  // utils/lang/check_phones_compatible.sh $lang/phones.txt $srcdir/phones.txt
  // cp $lang/phones.txt $dir

  // cp $srcdir/{tree,final.mdl} $dir || exit 1;

  
  // ======================= nnet feature_transform model 等模型 ===============
  // # Select default locations to model files
  // nnet=$srcdir/final.nnet;
  // class_frame_counts=$srcdir/ali_train_pdf.counts
  // feature_transform=$srcdir/final.feature_transform
  // model=$dir/final.mdl

  
  // # Check that files exist
  // for f in $sdata/1/feats.scp $lang/L.fst $nnet $model $feature_transform $class_frame_counts; do
  //   [ ! -f $f ] && echo "$0: missing file $f" && exit 1;
  // done
  // [ -z "$text" -a ! -f $sdata/1/text ] && echo "$0: missing file $f" && exit 1


  // # PREPARE FEATURE EXTRACTION PIPELINE
  // # import config,
  // cmvn_opts=
  // delta_opts=
  // D=$srcdir
  // [ -e $D/norm_vars ] && cmvn_opts="--norm-means=true --norm-vars=$(cat $D/norm_vars)" # Bwd-compatibility,
  // [ -e $D/cmvn_opts ] && cmvn_opts=$(cat $D/cmvn_opts)
  // [ -e $D/delta_order ] && delta_opts="--delta-order=$(cat $D/delta_order)" # Bwd-compatibility,
  // [ -e $D/delta_opts ] && delta_opts=$(cat $D/delta_opts)
  // #


  
  //  ==================== feature steam ========================
  // # Create the feature stream,
  // feats="ark,s,cs:copy-feats scp:$sdata/JOB/feats.scp ark:- |"
  
  // # apply-cmvn (optional),
  // [ ! -z "$cmvn_opts" -a ! -f $sdata/1/cmvn.scp ] && echo "$0: Missing $sdata/1/cmvn.scp" && exit 1
  // [ ! -z "$cmvn_opts" ] && feats="$feats apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp ark:- ark:- |"
  
  // # add-deltas (optional),
  // [ ! -z "$delta_opts" ] && feats="$feats add-deltas $delta_opts ark:- ark:- |"

  

  // ==================== 通过 feature_transform 和 nnet 运行一下fbank特征数据  ==> 转化为 matrix<pdf-id>
  // # nnet-forward,
  // feats="$feats nnet-forward $nnet_forward_opts --feature-transform=$feature_transform --class-frame-counts=$class_frame_counts --use-gpu=$use_gpu $nnet ark:- ark:- |"



  
  // 将 输入utt的feat数据 使用nnet模型--nnet等 生成对其结果 --> 写入到 $dir.
  // echo "$0: aligning data '$data' using nnet/model '$srcdir', putting alignments in '$dir'"


  // ====================== 获得 tra 是word标注结果.
  // # Map oovs in reference transcription, 
  // oov=`cat $lang/oov.int` || exit 1;
  // [ -z "$text" ] && text=$sdata/JOB/text
  // tra="ark:utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt $text |";

  // 只使用align-mapped进行映射, 但是会比较低效率, 因为它一个接一个的编译训练图 
  // # We could just use align-mapped in the next line, but it's less efficient
  // as it compiles the training graphs one by one.
  
  // if [ $stage -le 0 ]; then
  //   $cmd JOB=1:$nj $dir/log/align.JOB.log \
  //     编译训练数据简图 与 train_mono  train_delta 没有区别.
  //     compile-train-graphs --read-disambig-syms=$lang/phones/disambig.int
  //                          $dir/tree   $dir/final.mdl   $lang/L.fst     "$tra"     ark:-|        \
  
  //     根据训练数据简图 通过align-map方式进行映射 为 对齐结果 ===> exp/tri4b_dnn_ali/ali.n.gz 
  //     align-compiled-mapped $scale_opts --beam=$beam --retry-beam=$retry_beam $dir/final.mdl ark:- \
  //       "$feats" "ark,t:|gzip -c >$dir/ali.JOB.gz" || exit 1;
  // fi

  
  // 可选的 生成lattice格式的对齐结果, 这里并不生成, 稍后会进行生成lattice结果.
  // # Optionally align to lattice format (handy to get word alignment)
  // if [ "$align_to_lats" == "true" ]; then
  //   echo "$0: aligning also to lattices '$dir/lat.*.gz'"
  //   $cmd JOB=1:$nj $dir/log/align_lat.JOB.log \
  //     compile-train-graphs --read-disambig-syms=$lang/phones/disambig.int $lats_graph_scales $dir/tree $dir/final.mdl  $lang/L.fst "$tra" ark:- \| \
  //     latgen-faster-mapped $lats_decode_opts --word-symbol-table=$lang/words.txt $dir/final.mdl ark:- \
  //       "$feats" "ark:|gzip -c >$dir/lat.JOB.gz" || exit 1;
  // fi

  // echo "$0: done aligning data."

}
// ------------------------------------------------------------------------------
// scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
// beam=10
// retry_beam=40
// -------------------------------------------------------------------------------
//     align-compiled-mapped $scale_opts --beam=$beam --retry-beam=$retry_beam
//     $dir/final.mdl     ark:-          "$feats"             "ark,t:|gzip -c >$dir/ali.JOB.gz" || exit 1;
//     转移模型            简图            utt trian-feats         输出对齐结果
int align_compiled_mapped(int argc, char *argv[]) {
    const char *usage =
        "Generate alignments, reading log-likelihoods as matrices.\n"
        " (model is needed only for the integer mappings in its transition-model)\n"
        "Usage:   align-compiled-mapped [options] trans-model-in graphs-rspecifier feature-rspecifier alignments-wspecifier\n"
        "e.g.: \n"
        " nnet-align-compiled trans.mdl ark:graphs.fsts scp:train.scp ark:nnet.ali\n"
        "or:\n"
        " compile-train-graphs tree trans.mdl lex.fst ark:train.tra b, ark:- | \\\n"
        "   nnet-align-compiled trans.mdl ark:- scp:loglikes.scp t, ark:nnet.ali\n";

    ParseOptions po(usage);
    AlignConfig align_config;
    bool binary = true;
    BaseFloat acoustic_scale = 1.0;
    BaseFloat transition_scale = 1.0;
    BaseFloat self_loop_scale = 1.0;

    align_config.Register(&po);
    po.Register("binary", &binary, "Write output in binary mode");
    
    po.Register("transition-scale", &transition_scale,
                "Transition-probability scale [relative to acoustics]");
    po.Register("acoustic-scale", &acoustic_scale,
                "Scaling factor for acoustic likelihoods");
    po.Register("self-loop-scale", &self_loop_scale,
                "Scale of self-loop versus non-self-loop log probs [relative to acoustics]");
    po.Read(argc, argv);

    // one 4 Args scores_wspecifier is null.
    std::string model_in_filename = po.GetArg(1);
    std::string fst_rspecifier = po.GetArg(2);
    
    std::string feature_rspecifier = po.GetArg(3);
    std::string alignment_wspecifier = po.GetArg(4);
    std::string scores_wspecifier = po.GetOptArg(5);

    TransitionModel trans_model;
    ReadKaldiObject(model_in_filename, &trans_model);

    SequentialBaseFloatMatrixReader loglikes_reader(feature_rspecifier);
    RandomAccessTableReader<fst::VectorFstHolder> fst_reader(fst_rspecifier);
    Int32VectorWriter alignment_writer(alignment_wspecifier);
    BaseFloatWriter scores_writer(scores_wspecifier);

    int num_done = 0, num_err = 0, num_retry = 0;
    double tot_like = 0.0;
    kaldi::int64 frame_count = 0;

    // foreach utt feature
    for (; !loglikes_reader.Done(); loglikes_reader.Next()) {

      std::string utt = loglikes_reader.Key();
      
      const Matrix<BaseFloat> &loglikes = loglikes_reader.Value();
      VectorFst<StdArc> decode_fst(fst_reader.Value(utt));
      
      // fst_reader.FreeCurrent();  // this stops copy-on-write of the fst
      // by deleting the fst inside the reader, since we're about to mutate
      // the fst by adding transition probs.

      
      {  // Add transition-probs to the FST.
        std::vector<int32> disambig_syms;  // empty.
        AddTransitionProbs(trans_model, disambig_syms,
                           transition_scale, self_loop_scale,
                           &decode_fst);
      }

      // 解码统计量
      DecodableMatrixScaledMapped decodable(trans_model, loglikes, acoustic_scale);

      // 解码, 得到utt frame : <tid1, tid2, tid3, tid.....>.
      AlignUtteranceWrapper(align_config, utt,
                            acoustic_scale, &decode_fst, &decodable,
                            &alignment_writer, &scores_writer,
                            &num_done, &num_err, &num_retry,
                            &tot_like, &frame_count);
    }
}



// 生成MPE lattice 过程   =>> exp/tri4b_dnn_denlats/lat......scp
void make_denlats(){

  // # Create denominator lattices for MMI/MPE/sMBR training.
  // # Creates its output in $dir/lat.*.ark,   $dir/lat.scp
  // # The lattices are uncompressed, we need random access for DNN training.

  // # Begin configuration section.
  //       nj=4
  //       cmd=run.pl
  //       sub_split=1
  //       beam=13.0
  //       lattice_beam=7.0
  //       acwt=0.1
  //       max_active=5000
  //       nnet=
  //       nnet_forward_opts="--no-softmax=true --prior-scale=1.0"

  //       max_mem=20000000    // # This will stop the processes getting too large.
  // # This is in bytes, but not "real" bytes-- you have to multiply
  // # by something like 5 or 10 to get real bytes (not sure why so large)
  
  // # End configuration section.
  
  //       use_gpu=no # yes|no|optional
  //       parallel_opts="--num-threads 2"
  //       ivector=         # rx-specifier with i-vectors (ark-with-vectors),


  //       if [ $# != 4 ]; then
  //                           echo "Usage: steps/$0 [options] <data-dir> <lang-dir> <src-dir> <exp-dir>"
  //                           echo "  e.g.: steps/$0 data/train data/lang exp/tri1 exp/tri1_denlats"
  
  //                           echo "Works for plain features (or CMN, delta), forwarded through feature-transform."
  //                           echo ""
  //                           echo "Main options (for others, see top of script file)"
  //                           echo "  --config <config-file>                           # config containing options"
  //                           echo "  --nj <nj>                                        # number of parallel jobs"
  //                           echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  //                           echo "  --sub-split <n-split>                            # e.g. 40; use this for "
  //                           echo "                           # large databases so your jobs will be smaller and"
  //                           echo "                           # will (individually) finish reasonably soon."
  //                           exit 1;
  //   fi

  //       data=$1    //  data/fbank/train/
  //       lang=$2    //  data/lang
  //       srcdir=$3  //  exp/tri4b_dnn
  //       dir=$4     //  exp/tri4b_denlat

  //       oov=`cat $lang/oov.int` || exit 1;
  //       检查因素
  //       utils/lang/check_phones_compatible.sh $lang/phones.txt $srcdir/phones.txt

  //       cp -r $lang $dir/




  //       # Compute grammar FST which corresponds to unigram decoding graph.
  //       在目标路径生成 lang/
  //       new_lang="$dir/"$(basename "$lang")

  //       怎么算 unigram语法FST呢? 如果没有语法关系, 那所有词构成一个环就完了啊.?
  //       # 生成 unigram 的词语法 FST ==> new_lang
  //       echo "Making unigram grammar FST in $new_lang"
  //       将text(标注结果) words.txt 也是标注结果 构建G.fst G.fst 是unigram的
  //       cat $data/text | utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt | \
  //       awk '{for(n=2;n<=NF;n++){ printf("%s ", $n); } printf("\n"); }' | \
  //       utils/make_unigram_grammar.pl | fstcompile | fstarcsort --sort_type=ilabel > $new_lang/G.fst \
  //       || exit 1;

  // mkgraph 会生成一个完整的解码目录,
  // 1  组合 L_disambig.fst G.fst final.mdl  ==> HCLG.fst
  // 2  将一些解码需要的 phone 什么的 放入到 $dir 中
  // mkgraph.sh expects a whole directory "lang", so put everything in one directory...
  // it gets L_disambig.fst and G.fst (among other things) from $dir/lang, and final.mdl from $srcdir;
  // the output HCLG.fst goes in $dir/graph.
  
  //   echo "Compiling decoding graph in $dir/dengraph"
  //       if [ -s $dir/dengraph/HCLG.fst ] && [ $dir/dengraph/HCLG.fst -nt $srcdir/final.mdl ]; then
  //         echo "Graph $dir/dengraph/HCLG.fst already exists: skipping graph creation."
  //       else
  //         utils/mkgraph.sh $new_lang $srcdir $dir/dengraph || exit 1;
  //   fi
  void mkgraph_sh(){
    // # This script creates a fully expanded decoding graph (HCLG) that represents
    // # all the language-model, pronunciation dictionary (lexicon), context-dependency,
    // # and HMM structure in our model.  The output is a Finite State Transducer
    // # that has word-ids on the output, and pdf-ids on the input
    // (these are indexes that resolve to Gaussian Mixture Models)
    { 
      // # See
      // #  http://kaldi-asr.org/doc/graph_recipe_test.html
      // # (this is compiled from this repository using Doxygen,
      // # the source for this part is in src/doc/graph_recipe_test.dox)

    
      // tscale=1.0
      // loopscale=0.1
      // remove_oov=false

      // for x in `seq 4`; do
      //   [ "$1" == "--mono" -o "$1" == "--left-biphone" -o "$1" == "--quinphone" ] && shift && \
      //     echo "WARNING: the --mono, --left-biphone and --quinphone options are now deprecated and ignored."
      //   [ "$1" == "--remove-oov" ] && remove_oov=true && shift;
      //   [ "$1" == "--transition-scale" ] && tscale=$2 && shift 2;
      //   [ "$1" == "--self-loop-scale" ] && loopscale=$2 && shift 2;
      // done

      // if [ $# != 3 ]; then
      //    echo "Usage: utils/mkgraph.sh [options] <lang-dir> <model-dir> <graphdir>"
      //    echo "e.g.: utils/mkgraph.sh data/lang_test exp/tri1/ exp/tri1/graph"
      //    echo " Options:"
      //    echo " --remove-oov       #  If true, any paths containing the OOV symbol (obtained from oov.int"
      //    echo "                    #  in the lang directory) are removed from the G.fst during compilation."
      //    echo " --transition-scale #  Scaling factor on transition probabilities."
      //    echo " --self-loop-scale  #  Please see: http://kaldi-asr.org/doc/hmm.html#hmm_scale."
      //    echo "Note: the --mono, --left-biphone and --quinphone options are now deprecated"
      //    echo "and will be ignored."
      //    exit 1;
      // fi
    }
    
    // lang=$1                       //data/new_lang.
    // tree=$2/tree                  //srcdir --  exp/tri4b_dnn
    // model=$2/final.mdl
    // dir=$3                        // outpudir --- exp/tri4b_dnn_denlat/dengraph

    // mkdir -p $dir

    // # If $lang/tmp/LG.fst does not exist or is older than its sources, make it...

    // required="$lang/L.fst $lang/G.fst $lang/phones.txt $lang/words.txt $lang/phones/silence.csl $lang/phones/disambig.int $model $tree"
    // for f in $required; do
    //   [ ! -f $f ] && echo "mkgraph.sh: expected $f to exist" && exit 1;
    // done

    // N=$(tree-info $tree | grep "context-width" | cut -d' ' -f2) || { echo "Error when getting context-width"; exit 1; }
    // P=$(tree-info $tree | grep "central-position" | cut -d' ' -f2) || { echo "Error when getting central-position"; exit 1; }

    
    // [[ -f $2/frame_subsampling_factor && "$loopscale" == "0.1" ]] && \
    //   echo "$0: WARNING: chain models need '--self-loop-scale 1.0'";

    // mkdir -p $lang/tmp
    // trap "rm -f $lang/tmp/LG.fst.$$" EXIT HUP INT PIPE TERM
    // # Note: [[ ]] is like [ ] but enables certain extra constructs, e.g. || in
    // # place of -o
    
    // 构图 compose L+G  | determinize | minimizen | pushspechial | sort ---> LG.fst.
    // if [[ ! -s $lang/tmp/LG.fst || $lang/tmp/LG.fst -ot $lang/G.fst || \
    //       $lang/tmp/LG.fst -ot $lang/L_disambig.fst ]]; then
    //   fsttablecompose $lang/L_disambig.fst $lang/G.fst | fstdeterminizestar --use-log=true | \
    //     fstminimizeencoded | fstpushspecial | \
    //     fstarcsort --sort_type=ilabel > $lang/tmp/LG.fst.$$ || exit 1;
    //   mv $lang/tmp/LG.fst.$$ $lang/tmp/LG.fst
    //   fstisstochastic $lang/tmp/LG.fst || echo "[info]: LG not stochastic."
    // fi
    

    // clg=$lang/tmp/CLG_${N}_${P}.fst
    // clg_tmp=$clg.$$
    // ilabels=$lang/tmp/ilabels_${N}_${P}
    // ilabels_tmp=$ilabels.$$
    // trap "rm -f $clg_tmp $ilabels_tmp" EXIT HUP INT PIPE TERM
    // if [[ ! -s $clg || $clg -ot $lang/tmp/LG.fst \
    //     || ! -s $ilabels || $ilabels -ot $lang/tmp/LG.fst ]]; then
    //   fstcomposecontext --context-size=$N --central-position=$P \
    //    --read-disambig-syms=$lang/phones/disambig.int \
    //    --write-disambig-syms=$lang/tmp/disambig_ilabels_${N}_${P}.int \
    //     $ilabels_tmp < $lang/tmp/LG.fst |\
    //     fstarcsort --sort_type=ilabel > $clg_tmp
    //   mv $clg_tmp $clg
    //   mv $ilabels_tmp $ilabels
    //   fstisstochastic $clg || echo "[info]: CLG not stochastic."
    // fi

    // trap "rm -f $dir/Ha.fst.$$" EXIT HUP INT PIPE TERM
    // if [[ ! -s $dir/Ha.fst || $dir/Ha.fst -ot $model  \
    //     || $dir/Ha.fst -ot $lang/tmp/ilabels_${N}_${P} ]]; then
    //   make-h-transducer --disambig-syms-out=$dir/disambig_tid.int \
    //     --transition-scale=$tscale $lang/tmp/ilabels_${N}_${P} $tree $model \
    //      > $dir/Ha.fst.$$  || exit 1;
    //   mv $dir/Ha.fst.$$ $dir/Ha.fst
    // fi

    // trap "rm -f $dir/HCLGa.fst.$$" EXIT HUP INT PIPE TERM
    // if [[ ! -s $dir/HCLGa.fst || $dir/HCLGa.fst -ot $dir/Ha.fst || \
    //       $dir/HCLGa.fst -ot $clg ]]; then
    //   if $remove_oov; then
    //     [ ! -f $lang/oov.int ] && \
    //       echo "$0: --remove-oov option: no file $lang/oov.int" && exit 1;
    //     clg="fstrmsymbols --remove-arcs=true --apply-to-output=true $lang/oov.int $clg|"
    //   fi
    //   fsttablecompose $dir/Ha.fst "$clg" | fstdeterminizestar --use-log=true \
    //     | fstrmsymbols $dir/disambig_tid.int | fstrmepslocal | \
    //      fstminimizeencoded > $dir/HCLGa.fst.$$ || exit 1;
    //   mv $dir/HCLGa.fst.$$ $dir/HCLGa.fst
    //   fstisstochastic $dir/HCLGa.fst || echo "HCLGa is not stochastic"
    // fi

    // trap "rm -f $dir/HCLG.fst.$$" EXIT HUP INT PIPE TERM
    // if [[ ! -s $dir/HCLG.fst || $dir/HCLG.fst -ot $dir/HCLGa.fst ]]; then
    //   add-self-loops --self-loop-scale=$loopscale --reorder=true \
    //     $model < $dir/HCLGa.fst | fstconvert --fst_type=const > $dir/HCLG.fst.$$ || exit 1;
    //   mv $dir/HCLG.fst.$$ $dir/HCLG.fst
    
    //   if [ $tscale == 1.0 -a $loopscale == 1.0 ]; then
    //     # No point doing this test if transition-scale not 1, as it is bound to fail.
    //     fstisstochastic $dir/HCLG.fst || echo "[info]: final HCLG is not stochastic."
    //   fi
    // fi

    // # note: the empty FST has 66 bytes.  this check is for whether the final FST
    // # is the empty file or is the empty FST.
    // if ! [ $(head -c 67 $dir/HCLG.fst | wc -c) -eq 67 ]; then
    //   echo "$0: it looks like the result in $dir/HCLG.fst is empty"
    //   exit 1
    // fi

    
    // # save space.
    // rm $dir/HCLGa.fst $dir/Ha.fst 2>/dev/null || true

    // 保存lexicon 的副本, 和一系列的静音因素 这样我们解码就不需要 $lang目录了
    // # keep a copy of the lexicon and a list of silence phones with HCLG...
    // # this means we can decode without reference to the $lang directory.


    // cp $lang/words.txt $dir/ || exit 1;
    // mkdir -p $dir/phones
    // cp $lang/phones/word_boundary.* $dir/phones/ 2>/dev/null # might be needed for ctm scoring,
    // cp $lang/phones/align_lexicon.* $dir/phones/ 2>/dev/null # might be needed for ctm scoring,
    // cp $lang/phones/optional_silence.* $dir/phones/ 2>/dev/null # might be needed for analyzing alignments.
    //   # but ignore the error if it's not there.

    // cp $lang/phones/disambig.{txt,int} $dir/phones/ 2> /dev/null
    // cp $lang/phones/silence.csl $dir/phones/ || exit 1;
    // cp $lang/phones.txt $dir/ 2> /dev/null # ignore the error if it's not there.

    // am-info --print-args=false $model | grep pdfs | awk '{print $NF}' > $dir/num_pdfs

  }


  
  // cp $srcdir/{tree,final.mdl} $dir

  // # Select default locations to model files
  // nnet=$srcdir/final.nnet;
  // class_frame_counts=$srcdir/ali_train_pdf.counts
  // feature_transform=$srcdir/final.feature_transform
  // model=$dir/final.mdl

  // # Check that files exist
  // for f in $sdata/1/feats.scp $nnet $model $feature_transform $class_frame_counts; do
  //   [ ! -f $f ] && echo "$0: missing file $f" && exit 1;
  // done


  // # ==================== PREPARE FEATURE EXTRACTION PIPELINE
  // # import config,
  // cmvn_opts=
  // delta_opts=
  // D=$srcdir
  // [ -e $D/norm_vars ] && cmvn_opts="--norm-means=true --norm-vars=$(cat $D/norm_vars)" # Bwd-compatibility,
  // [ -e $D/cmvn_opts ] && cmvn_opts=$(cat $D/cmvn_opts)
  // [ -e $D/delta_order ] && delta_opts="--delta-order=$(cat $D/delta_order)" # Bwd-compatibility,
  // [ -e $D/delta_opts ] && delta_opts=$(cat $D/delta_opts)
  // #
  // # Create the feature stream,
  // feats="ark,s,cs:copy-feats scp:$sdata/JOB/feats.scp ark:- |"
  // # apply-cmvn (optional),
  // [ ! -z "$cmvn_opts" -a ! -f $sdata/1/cmvn.scp ] && echo "$0: Missing $sdata/1/cmvn.scp" && exit 1
  // [ ! -z "$cmvn_opts" ] && feats="$feats apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp ark:- ark:- |"
  // # add-deltas (optional),
  // [ ! -z "$delta_opts" ] && feats="$feats add-deltas $delta_opts ark:- ark:- |"


  // # nnet-forward,
  // feats="$feats nnet-forward $nnet_forward_opts --feature-transform=$feature_transform --class-frame-counts=$class_frame_counts --use-gpu=$use_gpu $nnet ark:- ark:- |"

  // # if this job is interrupted by the user, we want any background jobs to be
  // # killed too.
  // cleanup() {
  //   local pids=$(jobs -pr)
  //   [ -n "$pids" ] && kill $pids || true
  // }
  // trap "cleanup" INT QUIT TERM EXIT


  // echo "$0: generating denlats from data '$data', putting lattices in '$dir'"
  // MPE 会需要使用 Lattice 做 MPE损失函数的分母, 进行去区分性训练
  // 
  // #1) Generate the denominator lattices
  // if [ $sub_split -eq 1 ]; then

  //   准备SCP 文件 用来分离保存所有的 lattice,
  //   每个lattice都被gizp到一个 exp/tri4b_dnn_denlat/latn/utt.gz文件中.
  //   此时 实际上只是提供了目标地址, 后面 latgen-faster-mapped 是生成lattice过程
  //   # Prepare 'scp' for storing lattices separately and gzipped
  //   for n in `seq $nj`; do
  //     [ ! -d $dir/lat$n ] && mkdir $dir/lat$n;
  //     多任务下 构建lattice??
  //     cat $sdata/$n/feats.scp | \
  //     awk -v dir=$dir -v n=$n '{ utt=$1; utt_noslash=utt; gsub("/","_",utt_noslash);
  //                                printf("%s | gzip -c >%s/lat%d/%s.gz\n", utt, dir, n, utt_noslash); }'
  //   done  >$dir/lat.store_separately_as_gz.scp

  //   生成词图 lattice 过程 latgen-faster-mapped.
  //   按照刚for done 生成的SCP文件, 将所有utt-feats 通过WFST-HCLG.fst
  //   生成lattice  ===>  exp/tri4b_dnn_denlat/latn/utt.gz.
  //   相当耗时间, 已经等了
  //   # Generate the lattices
  //   $cmd $parallel_opts JOB=1:$nj $dir/log/decode_den.JOB.log \
  //     latgen-faster-mapped --beam=$beam --lattice-beam=$lattice_beam --acoustic-scale=$acwt \
  //       --max-mem=$max_mem --max-active=$max_active --word-symbol-table=$lang/words.txt $srcdir/final.mdl  \
  //       $dir/dengraph/HCLG.fst "$feats" "scp:$dir/lat.store_separately_as_gz.scp" || exit 1;
  // fi


  // 
  // #2) Generate 'scp' for reading the lattices
  // 测试 就是更改为绝对路径
  // # make $dir an absolute pathname.
  // [ '/' != ${dir:0:1} ] && dir=$PWD/$dir

  // 将所有 *.gz 经过怎么处理 然后 > $dir/lat.scp.
  // for n in `seq $nj`; do
  //   find $dir/lat${n} -name "*.gz" | perl -ape 's:.*/([^/]+)\.gz$:$1 gunzip -c $& |:; '
  // done | sort >$dir/lat.scp
  // [ -s $dir/lat.scp ] || exit 1

  // echo "$0: done generating denominator lattices."
}


// latgen-faster-mapped --beam=$beam --lattice-beam=$lattice_beam --acoustic-scale=$acwt \
// --max-mem=$max_mem --max-active=$max_active --word-symbol-table=$lang/words.txt
// $srcdir/final.mdl    $dir/dengraph/HCLG.fst   "$feats"                          "scp:$dir/lat.store_separately_as_gz.scp"
// transition-model     HCLG.fst                  pdf-id's log-likelihood          out> lattice write scp 写描述符

void latgen_faster_mapped(){
    const char *usage =
        "Generate lattices, reading log-likelihoods as matrices\n"
        " (model is needed only for the integer mappings in its transition-model)\n"
        "Usage: latgen-faster-mapped [options] trans-model-in (fst-in|fsts-rspecifier) loglikes-rspecifier"
        " lattice-wspecifier [ words-wspecifier [alignments-wspecifier] ]\n";
    ParseOptions po(usage);
    Timer timer;
    bool allow_partial = false;
    BaseFloat acoustic_scale = 0.1;
    LatticeFasterDecoderConfig config;

    std::string word_syms_filename;
    config.Register(&po);
    // 声学拉伸.
    po.Register("acoustic-scale", &acoustic_scale, "Scaling factor for acoustic likelihoods");
    //  每utt的标注
    po.Register("word-symbol-table", &word_syms_filename, "Symbol table for words [for debug output]");
    po.Register("allow-partial", &allow_partial, "If true, produce output even if end state was not reached.");

    po.Read(argc, argv);

    if (po.NumArgs() < 4 || po.NumArgs() > 6) {
      po.PrintUsage();
      exit(1);
    }

    //  Args = 4

    std::string model_in_filename = po.GetArg(1),
        fst_in_str = po.GetArg(2),
        feature_rspecifier = po.GetArg(3),     // 实际上是feat 经过feature-trans nnet 后得到的pdf-id 后验概率
        lattice_wspecifier = po.GetArg(4),
        
        words_wspecifier = po.GetOptArg(5),
        alignment_wspecifier = po.GetOptArg(6);

    TransitionModel trans_model;
    ReadKaldiObject(model_in_filename, &trans_model);

    bool determinize = config.determinize_lattice;
    CompactLatticeWriter compact_lattice_writer;
    LatticeWriter lattice_writer;
    
    
    Int32VectorWriter words_writer(words_wspecifier);

    Int32VectorWriter alignment_writer(alignment_wspecifier);

    //  将utt 标注 >> word_syms
    fst::SymbolTable *word_syms = NULL;
    if (word_syms_filename != "")
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename)))
        KALDI_ERR << "Could not read symbol table from file "
                   << word_syms_filename;

    double tot_like = 0.0;
    kaldi::int64 frame_count = 0;
    int num_success = 0, num_fail = 0;

    // 
    if (ClassifyRspecifier(fst_in_str, NULL, NULL) == kNoRspecifier) {
      
      SequentialBaseFloatMatrixReader loglike_reader(feature_rspecifier);
      // Input FST is just one FST, not a table of FSTs.
      Fst<StdArc> *decode_fst = fst::ReadFstKaldiGeneric(fst_in_str);
      timer.Reset();

      {
        LatticeFasterDecoder decoder(*decode_fst, config);

        // foreach utt
        for (; !loglike_reader.Done(); loglike_reader.Next()) {
          
          std::string utt = loglike_reader.Key();
          // pdf-id's log-likelihoods [frames X num_pdfs]
          Matrix<BaseFloat> loglikes (loglike_reader.Value());
          loglike_reader.FreeCurrent();
          // 解码统计量
          DecodableMatrixScaledMapped decodable(trans_model, loglikes, acoustic_scale);

          double like;
          // 根据pdf-id 结果  +  decode_fst + word_syms 标注Table(内部会识别具体的words) + utt_key
          if (DecodeUtteranceLatticeFaster(
                  decoder, decodable, trans_model, word_syms, utt,
                  acoustic_scale, determinize, allow_partial, &alignment_writer, &words_writer,
                  &compact_lattice_writer, &lattice_writer,
                  &like)) {
            tot_like += like;
            frame_count += loglikes.NumRows();
            num_success++;
          } else num_fail++;
        }
      }
      delete decode_fst; // delete this only after decoder goes out of scope.
    } else { // We have different FSTs for different utterances.

    }

    delete word_syms;
}

  














//  =======================  MPE区分性训练 训练过程

// if [ $stage -le 3 ]; then
//   outdir=exp/tri4b_dnn_mpe

//   通过3 iter MPE 重新训练 DNN.
//   #Re-train the DNN by 3 iteration of MPE
//   steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 3 --acwt $acwt --do-smbr false \
//     data/fbank/train      data/lang    $srcdir     ${srcdir}_ali    ${srcdir}_denlats    $outdir || exit 1

void nnet_train_mpe_sh(){
  // 序列-区分性 MPE/sMBR 的DNN训练
  // # Sequence-discriminative MPE/sMBR training of DNN.
  // # 4 iterations (by default) of Stochastic Gradient Descent with per-utterance updates.
  // # We select between MPE/sMBR optimization by '--do-smbr <bool>' option.

  // # For the numerator we have a fixed alignment rather than a lattice--
  // # this actually follows from the way lattices are defined in Kaldi, which
  // # is to have a single path for each word (output-symbol) sequence.


  // # Begin configuration section.
  {
    // cmd=run.pl
    // num_iters=4
    // acwt=0.1
    // lmwt=1.0
    // learn_rate=0.00001
    // momentum=0.0
    // halving_factor=1.0 #ie. disable halving
    // do_smbr=true
    // one_silence_class=true # if true : all the `silphones' are mapped to a single class in the Forward-backward of sMBR/MPE,
    //                        # (this prevents the sMBR from WER explosion, which was happenning with some data).
    //                        # if false : the silphone-frames are always counted as 'wrong' in the calculation of the approximate accuracies,
    // silphonelist=          # this overrides default silphone-list (for selecting a subset of sil-phones)

    // unkphonelist=          # dummy deprecated option, for backward compatibility,
    // exclude_silphones=     # dummy deprecated option, for backward compatibility,

    
    // verbose=0 # 0 No GPU time-stats, 1 with GPU time-stats (slower),
    // ivector=
    // nnet=  # For non-default location of nnet,

    // seed=777    # seed value used for training data shuffling
    // skip_cuda_check=false
    // # End configuration section
    
    // if [ $# -ne 6 ]; then
    //   echo "Usage: $0 <data> <lang> <srcdir> <ali> <denlats> <exp>"
    //   echo " e.g.: $0 data/train_all data/lang exp/tri3b_dnn exp/tri3b_dnn_ali exp/tri3b_dnn_denlats exp/tri3b_dnn_smbr"
    //   echo "Main options (for others, see top of script file)"
    //   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
    //   echo "  --config <config-file>                           # config containing options"
    //   echo "  --num-iters <N>                                  # number of iterations to run"
    //   echo "  --acwt <float>                                   # acoustic score scaling"
    //   echo "  --lmwt <float>                                   # linguistic score scaling"
    //   echo "  --learn-rate <float>                             # learning rate for NN training"
    //   echo "  --do-smbr <bool>                                 # do sMBR training, otherwise MPE"

    //   exit 1;
    // fi
  }


  //     data/fbank/train      data/lang    $srcdir     ${srcdir}_ali    ${srcdir}_denlats    $outdir || exit 1
  // data=$1
  // lang=$2
  // srcdir=$3               exp/tri4b_dnn
  // alidir=$4               exp/tri4b_dnn_ali
  // denlatdir=$5            exp/tri4b_dnn_denlat
  // dir=$6                  exp/tri4b_dnn_mpe

  // for f in $data/feats.scp $denlatdir/lat.scp \
  //          $alidir/{tree,final.mdl,ali.1.gz} \
  //          $srcdir/{final.nnet,final.feature_transform}; do
  //   [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
  // done



  // =======================  cuda  phones.txt  alidir/final.mdl  alidir/tree => exp/tri4b_dnn_mpe
  // ======================= 获得一些模型   ==> exp/tri4b_dnn_mpe/
  {
    // # check if CUDA compiled in,
    // if ! $skip_cuda_check; then cuda-compiled || { echo "Error, CUDA not compiled-in!"; exit 1; } fi
    // mkdir -p $dir/log
    
    // utils/lang/check_phones_compatible.sh $lang/phones.txt $srcdir/phones.txt
    // utils/lang/check_phones_compatible.sh $lang/phones.txt $alidir/phones.txt
    // cp $lang/phones.txt $dir
    
    // cp $alidir/{final.mdl,tree} $dir
    // [ -z $silphonelist ] && silphonelist=`cat $lang/phones/silence.csl` # Default 'silphonelist',



    // #Get the files we will need
    // [ -z "$nnet" ] && nnet=$srcdir/$(readlink $srcdir/final.nnet || echo final.nnet);
    // [ -z "$nnet" ] && echo "Error nnet '$nnet' does not exist!" && exit 1;
    // cp $nnet $dir/0.nnet; nnet=$dir/0.nnet

    // class_frame_counts=$srcdir/ali_train_pdf.counts ?????  对齐后 每pdf对应的frame 总数
    // [ -z "$class_frame_counts" ] && echo "Error class_frame_counts '$class_frame_counts' does not exist!" && exit 1;
    // cp $srcdir/ali_train_pdf.counts $dir

    // feature_transform=$srcdir/final.feature_transform
    // if [ ! -f $feature_transform ]; then
    //   echo "Missing feature_transform '$feature_transform'"
    //   exit 1
    // fi
    // cp $feature_transform $dir/final.feature_transform

    // model=$dir/final.mdl
    // [ -z "$model" ] && echo "Error transition model '$model' does not exist!" && exit 1;
  }





  // ===================== 清洗 随机化 feats特征
  // # Shuffle the feature list to make the GD stochastic!
  // # By shuffling features, we have to use lattices with random access (indexed by .scp file).
  // cat $data/feats.scp | utils/shuffle_list.pl --srand $seed > $dir/train.scp

  // [ -n "$unkphonelist" ] && echo "WARNING: The option '--unkphonelist' is now deprecated. Please remove it from your recipe..."
  // [ -n "$exclude_silphones" ] && echo "WARNING: The option '--exclude-silphones' is now deprecated. Please remove it from your recipe..."

  
  // ###
  // ### PREPARE FEATURE EXTRACTION PIPELINE
  // ###
  // # import config,
  // cmvn_opts=
  // delta_opts=
  // D=$srcdir
  // [ -e $D/norm_vars ] && cmvn_opts="--norm-means=true --norm-vars=$(cat $D/norm_vars)" # Bwd-compatibility,
  // [ -e $D/cmvn_opts ] && cmvn_opts=$(cat $D/cmvn_opts)
  // [ -e $D/delta_order ] && delta_opts="--delta-order=$(cat $D/delta_order)" # Bwd-compatibility,
  // [ -e $D/delta_opts ] && delta_opts=$(cat $D/delta_opts)


  // # =================== 创建特征 steam  feats + cmvn + delta 
  // # Create the feature stream,
  {
    // feats="ark,o:copy-feats scp:$dir/train.scp ark:- |"
    // # apply-cmvn (optional),
    // [ ! -z "$cmvn_opts" -a ! -f $data/cmvn.scp ] && echo "$0: Missing $data/cmvn.scp" && exit 1
    // [ ! -z "$cmvn_opts" ] && feats="$feats apply-cmvn $cmvn_opts --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp ark:- ark:- |"
    // # add-deltas (optional),
    // [ ! -z "$delta_opts" ] && feats="$feats add-deltas $delta_opts ark:- ark:- |"
    // # add-pytel transform (optional),
    // [ -e $D/pytel_transform.py ] && feats="$feats /bin/env python $D/pytel_transform.py |"

    // # add-ivector (optional),
    // if [ -e $D/ivector_dim ]; then
    //   [ -z $ivector ] && echo "Missing --ivector, they were used in training!" && exit 1
    //   # Get the tool,
    //   ivector_append_tool=append-vector-to-feats # default,
    //   [ -e $D/ivector_append_tool ] && ivector_append_tool=$(cat $D/ivector_append_tool)
    //   # Check dims,
    //   dim_raw=$(feat-to-dim "$feats" -)
    //   dim_raw_and_ivec=$(feat-to-dim "$feats $ivector_append_tool ark:- '$ivector' ark:- |" -)
    //   dim_ivec=$((dim_raw_and_ivec - dim_raw))
    //   [ $dim_ivec != "$(cat $D/ivector_dim)" ] && \
    //     echo "Error, i-vector dim. mismatch (expected $(cat $D/ivector_dim), got $dim_ivec in '$ivector')" && \
    //     exit 1
    //   # Append to feats,
    //   feats="$feats $ivector_append_tool ark:- '$ivector' ark:- |"
    // fi
  }
  
  // ### Record the setup,
  // [ ! -z "$cmvn_opts" ] && echo $cmvn_opts >$dir/cmvn_opts
  // [ ! -z "$delta_opts" ] && echo $delta_opts >$dir/delta_opts
  // [ -e $D/pytel_transform.py ] && cp {$D,$dir}/pytel_transform.py
  // [ -e $D/ivector_dim ] && cp {$D,$dir}/ivector_dim
  // [ -e $D/ivector_append_tool ] && cp $D/ivector_append_tool $dir/ivector_append_tool
  // ###


  
  // ###
  // ### Prepare the alignments
  // ###
  // # Assuming all alignments will fit into memory
  // ali="ark:gunzip -c $alidir/ali.*.gz |"


  // ###
  // ### Prepare the lattices
  // ###
  // # The lattices are indexed by SCP (they are not gziped because of the random access in SGD)
  // lats="scp:$denlatdir/lat.scp"


  // # Run several iterations of the MPE/sMBR training

  // cur_mdl=$nnet
  // x=1
  // while [ $x -le $num_iters ]; do
  //   echo "Pass $x (learnrate $learn_rate)"
  //   if [ -f $dir/$x.nnet ]; then
  //     echo "Skipped, file $dir/$x.nnet exists"
  //   else
  //     #train
  //     $cmd $dir/log/mpe.$x.log \
  //      nnet-train-mpe-sequential \
  //        --feature-transform=$feature_transform \
  //        --class-frame-counts=$class_frame_counts \
  //        --acoustic-scale=$acwt \
  //        --lm-scale=$lmwt \
  //        --learn-rate=$learn_rate \
  //        --momentum=$momentum \
  //        --do-smbr=$do_smbr \
  //        --verbose=$verbose \
  //        --one-silence-class=$one_silence_class \
  //        ${silphonelist:+ --silence-phones=$silphonelist} \
  //        $cur_mdl $alidir/final.mdl "$feats" "$lats" "$ali" $dir/$x.nnet
  //   fi
  //   cur_mdl=$dir/$x.nnet

  //   #report the progress
  //   grep -B 2 "Overall average frame-accuracy" $dir/log/mpe.$x.log | sed -e 's|.*)||'

  //   x=$((x+1))
  //   learn_rate=$(awk "BEGIN{print($learn_rate*$halving_factor)}")

  // done




  // 将最终的 nnet --> finnale.nnet
  // N.nnet --> final.nnet
  // (cd $dir; [ -e final.nnet ] && unlink final.nnet; ln -s $((x-1)).nnet final.nnet)


  // echo "MPE/sMBR training finished"


  // 获得 ? 先验概率?
  // if [ -e $dir/prior_counts ]; then
  //   echo "Priors are already re-estimated, skipping... ($dir/prior_counts)"
  // else
  //   echo "Re-estimating priors by forwarding 10k utterances from training set."
  //   . ./cmd.sh
  //   nj=$(cat $alidir/num_jobs)
  //   steps/nnet/make_priors.sh --cmd "$train_cmd" --nj $nj \
  //     ${ivector:+ --ivector "$ivector"} $data $dir
  // fi

  // echo "$0: Done. '$dir'"
  // exit 0
 
}























//   #Decode (reuse HCLG graph)
//   for ITER in 3 2 1; do
//    (
//     steps/nnet/decode.sh --nj $nj --cmd "$decode_cmd" --nnet $outdir/${ITER}.nnet --config conf/decode_dnn.config --acwt $acwt 
//       $gmmdir/graph_word data/fbank/test $outdir/decode_test_word_it${ITER} || exit 1;
//    )&

//    (
//    steps/nnet/decode.sh --nj $nj --cmd "$decode_cmd" --nnet $outdir/${ITER}.nnet --config conf/decode_dnn.config --acwt $acwt \
//      $gmmdir/graph_phone data/fbank/test_phone $outdir/decode_test_phone_it${ITER} || exit 1;
//    )&
//   done
// fi
