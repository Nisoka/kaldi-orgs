//     steps/nnet/train.sh --copy_feats false --cmvn-opts "--norm-means=true --norm-vars=false" --hid-layers 4 --hid-dim 1024 \
//     --learn-rate 0.008
//             data/fbank/train     data/fbank/dev     data/lang     $alidir    $alidir_cv $outdir


void configuration(){
  // # Begin configuration.
  // config=             # config, also forwarded to 'train_scheduler.sh',

  // # topology, initialization,

  // network_type=dnn    # select type of neural network (dnn,cnn1d,cnn2d,lstm),
  // hid_layers=4        # nr. of hidden layers (before sotfmax or bottleneck),
  // hid_dim=1024        # number of neurons per layer,
  // bn_dim=             # (optional) adds bottleneck and one more hidden layer to the NN,
  // dbn=                # (optional) prepend layers to the initialized NN,

  // proto_opts=         # adds options to 'make_nnet_proto.py',
  // cnn_proto_opts=     # adds options to 'make_cnn_proto.py',

  // nnet_init=          # (optional) use this pre-initialized NN,
  // nnet_proto=         # (optional) use this NN prototype for initialization,

  // # feature processing,
  // splice=5            # (default) splice features both-ways along time axis,
  // cmvn_opts=          # (optional) adds 'apply-cmvn' to input feature pipeline, see opts,
  // delta_opts=         # (optional) adds 'add-deltas' to input feature pipeline, see opts,
  // ivector=            # (optional) adds 'append-vector-to-feats', the option is rx-filename for the 2nd stream,
  // ivector_append_tool=append-vector-to-feats # (optional) the tool for appending ivectors,

  // feat_type=plain
  // traps_dct_basis=11    # (feat_type=traps) nr. of DCT basis, 11 is good with splice=10,
  // transf=               # (feat_type=transf) import this linear tranform,
  // splice_after_transf=5 # (feat_type=transf) splice after the linear transform,

  // feature_transform_proto= # (optional) use this prototype for 'feature_transform',
  // feature_transform=  # (optional) directly use this 'feature_transform',
  // pytel_transform=    # (BUT) use external python transform,

  // # labels,
  // labels=            # (optional) specify non-default training targets,
  //                    # (targets need to be in posterior format, see 'ali-to-post', 'feat-to-post'),
  // num_tgt=           # (optional) specifiy number of NN outputs, to be used with 'labels=',

  // # training scheduler,
  // learn_rate=0.008   # initial learning rate,
  // scheduler_opts=    # options, passed to the training scheduler,
  // train_tool=        # optionally change the training tool,
  // train_tool_opts=   # options for the training tool,
  // frame_weights=     # per-frame weights for gradient weighting,
  // utt_weights=       # per-utterance weights (scalar for --frame-weights),

  // # data processing, misc.
  // copy_feats=true     # resave the train/cv features into /tmp (disabled by default),
  // copy_feats_tmproot=/tmp/kaldi.XXXX # sets tmproot for 'copy-feats',
  // copy_feats_compress=true # compress feats while resaving
  // feats_std=1.0

  // split_feats=        # split the training data into N portions, one portion will be one 'epoch',
  //                     # (empty = no splitting)

  // seed=777            # seed value used for data-shuffling, nn-initialization, and training,
  // skip_cuda_check=false
  // skip_phoneset_check=false

  // # End configuration.





  // if [ $# != 6 ]; then
  //    echo "Usage: $0 <data-train> <data-dev> <lang-dir> <ali-train> <ali-dev> <exp-dir>"
  //    echo " e.g.: $0 data/train data/cv data/lang exp/mono_ali_train exp/mono_ali_cv exp/mono_nnet"
  //    echo ""
  //    echo " Training data : <data-train>,<ali-train> (for optimizing cross-entropy)"
  //    echo " Held-out data : <data-dev>,<ali-dev> (for learn-rate scheduling, model selection)"
  //    echo " note.: <ali-train>,<ali-dev> can point to same directory, or 2 separate directories."
  //    echo ""
  //    echo "main options (for others, see top of script file)"
  //    echo "  --config <config-file>   # config containing options"
  //    echo ""
  //    echo "  --network-type (dnn,cnn1d,cnn2d,lstm)  # type of neural network"
  //    echo "  --nnet-proto <file>      # use this NN prototype"
  //    echo "  --feature-transform <file> # re-use this input feature transform"
  //    echo ""
  //    echo "  --feat-type (plain|traps|transf) # type of input features"
  //    echo "  --cmvn-opts  <string>            # add 'apply-cmvn' to input feature pipeline"
  //    echo "  --delta-opts <string>            # add 'add-deltas' to input feature pipeline"
  //    echo "  --splice <N>                     # splice +/-N frames of input features"
  //    echo
  //    echo "  --learn-rate <float>     # initial leaning-rate"
  //    echo "  --copy-feats <bool>      # copy features to /tmp, lowers storage stress"
  //    echo ""
  //    exit 1;
  // fi

}

  
// data=$1                  data/fbank/train           train fbank 特征
// data_cv=$2               data/fbank/dev             dev fbank 特征
// lang=$3                  data/lang                  语言模型
// alidir=$4                exp/tri4b_ali              上次对齐结果
// alidir_cv=$5             exp/tri4b_cv               上次dev对齐结果
// dir=$6                   outputdir - exp/tri4b_dnn  dnn输出结果目录.

// mkdir -p $dir/{log,nnet}



// ###### PREPARE ALIGNMENTS ######
void PREPARE_ALIGNMENTS(){

  //   将上次对齐结果从 trans-id 转化为 pdf-id. 或者post后验概率形式
  //   对齐结果都是 trans-id的, 所以通过ali-to-pdf 将trans-id -> pdf-id,
  //   再通过ali-to-post 将pdf-id -> post(ali-to-post的转换非常简单, 只不过为pdf-id 增加了权重1.0 实际上每改变什么).
  
  //   将对齐结果转化为
  //   labels_tr="ark:ali-to-pdf $alidir/final.mdl \"ark:gunzip -c $alidir/ali.*.gz |\" ark:- | ali-to-post ark:- ark:- |"
  //   labels_cv="ark:ali-to-pdf $alidir/final.mdl \"ark:gunzip -c $alidir_cv/ali.*.gz |\" ark:- | ali-to-post ark:- ark:- |"

  //   将每帧转化为 pdf-id phones
  //   # training targets for analyze-counts,
  //   labels_tr_pdf="ark:ali-to-pdf $alidir/final.mdl \"ark:gunzip -c $alidir/ali.*.gz |\" ark:- |"
  //   labels_tr_phn="ark:ali-to-phones --per-frame=true $alidir/final.mdl \"ark:gunzip -c $alidir/ali.*.gz |\" ark:- |"


  //   获得pdf总数, 稍后解码对齐时 需要
  //   num_pdf=$(hmm-info $alidir/final.mdl | awk '/pdfs/{print $4}')

  //   统计 pdf 总数
  //   analyze-counts --verbose=1 --binary=false --counts-dim=$num_pdf \
  //     "$labels_tr_pdf" $dir/ali_train_pdf.counts 

  //   拷贝模型 留待使用
  //   # copy the old transition model, will be needed by decoder,
  //   copy-transition-model --binary=false $alidir/final.mdl $dir/final.mdl
  //   # copy the tree
  //   cp $alidir/tree $dir/tree



  //   统计phone 总数.
  //   analyze-counts --verbose=1 --symbol-table=$lang/phones.txt --counts-dim=$num_pdf \
  //                               所有音素(带销岐符号)标签       pdf 总数
  //     "$labels_tr_phn" /dev/null 
  //      phone对齐结果   统计结果
  // fi

  //  汇总 对齐-phone 中的phone总数.
  int analyze_counts(int argc, char *argv[]) {
    const char *usage =
        // 从int vector 表 计算元素总数
        // 例如 获得pdf总数 来为数据分析 估计DNN-输出的先验概率？？？
        "Computes element counts from integer vector table.\n"
        "(e.g. get pdf-counts to estimate DNN-output priors "
        "for data analysis)\n"

        "Usage: analyze-counts [options] <alignments-rspecifier> <counts-wxfilname>\n"
        " analyze-counts ark:1.ali prior.counts\n"
        // 显示phone总数？
        " Show phone counts by:\n"
        " ali-to-phones --per-frame=true ark:1.ali ark:- |"
        " analyze-counts --verbose=1 ark:- - >/dev/null\n"
        "Note: this is deprecated, see post-to-tacc.\n";

    ParseOptions po(usage);

    bool binary = false;
    std::string symbol_table_filename = "";

    // 音素总数
    po.Register("symbol-table", &symbol_table_filename,
                "Read symbol table for display of counts");

    // pdf总数
    int32 counts_dim = 0;
    po.Register("counts-dim", &counts_dim,
                "Output dimension of the counts, "
                "a hint for dimension auto-detection.");
    std::string
        // phone对齐结果
        alignments_rspecifier = po.GetArg(1),
        wxfilename = po.GetArg(2);

    SequentialInt32VectorReader alignment_reader(alignments_rspecifier);

    // counts长度 使用counts_dim, 但是对于统计phone时, counts_dim使用的是pdf-count?? 这个有问题.
    Vector<double> counts(counts_dim, kSetZero);

    int32 num_done = 0, num_other_error = 0;
    // foreach utt ali-phone.
    for (; !alignment_reader.Done(); alignment_reader.Next()) {
      std::string utt = alignment_reader.Key();
      const std::vector<int32> &alignment = alignment_reader.Value();
      BaseFloat utt_w = 1.0;
      Vector<BaseFloat> frame_w;
      // Check if per-frame weights are provided

      // ali-phone 中的每个phone
      // Accumulate the counts
      for (size_t i = 0; i < alignment.size(); i++) {
        KALDI_ASSERT(alignment[i] >= 0);  // 特殊phone不该出现.
      
        // Extend the vector if it is not large enough to hold every pdf-ids
        if (alignment[i] >= counts.Dim()) {
          counts.Resize(alignment[i]+1, kCopyData);
        }

        // counts 中保存对应Index音素的总数
        if (frame_weights != "") {
          counts(alignment[i]) += 1.0 * utt_w * frame_w(i);
        } else {
          counts(alignment[i]) += 1.0 * utt_w;
        }
      }
      num_done++;
    }

    // check
    for (size_t i = 0; i < counts.Dim(); i++) {
      if (0.0 == counts(i)) {
        KALDI_WARN << "Zero count for label " << i << ", this is suspicious.";
      }
    }

    // Add a ``half-frame'' to all the elements to
    // avoid zero-counts which would cause problems in decoding
    // 对所有phone总数 + 0.5 避免总数为0， 写入文件.
    Vector<double> counts_nozero(counts);
    counts_nozero.Add(0.5);
    Output ko(wxfilename, binary);
    counts_nozero.Write(ko.Stream(), binary);

    //
    // THE REST IS FOR ANALYSIS, IT GETS PRINTED TO LOG
    //
    if (symbol_table_filename != "" || (kaldi::g_kaldi_verbose_level >= 1)) {
      // load the symbol table
      fst::SymbolTable *elem_syms = NULL;
      if (symbol_table_filename != "") {
        elem_syms = fst::SymbolTable::ReadText(symbol_table_filename);
        if (!elem_syms)
          KALDI_ERR << "Could not read symbol table from file "
                    << symbol_table_filename;
      }



      
      // 将phones总数count 转化为 <count, index>形式.
      // sort the counts
      std::vector<std::pair<double, int32> > sorted_counts;
      for (int32 i = 0; i < counts.Dim(); i++) {
        sorted_counts.push_back(
            std::make_pair(static_cast<double>(counts(i)), i));
      }
      std::sort(sorted_counts.begin(), sorted_counts.end());

    }
  }

}






// ###### PREPARE FEATURES ######
void PREPARE_FEATURES(){
  //  准备特征

  //   只是拷贝一下, 为了节省磁盘读取时间?
  //   cp $data/feats.scp $dir/train_sorted.scp
  //   cp $data_cv/feats.scp $dir/cv.scp


  //   对数据进行移动固定偏移量. 
  //   utils/shuffle_list.pl --srand ${seed:-777} <$dir/train_sorted.scp >$dir/train.scp

  // copy-feats --compress=$copy_feats_compress scp:$data/feats.scp     ark,scp:$tmpdir/train.ark,$dir/train_sorted.scp
  //                                               源feats路径         目标路径
  int copy_feats(int argc, char *argv[]) {
    using namespace kaldi;
  
    const char *usage =
        "Copy features [and possibly change format]\n"
        // 拷贝特征
        "Usage: copy-feats [options] <feature-rspecifier> <feature-wspecifier>\n"
        "e.g.: copy-feats ark:- ark,scp:foo.ark,foo.scp\n"
        " or: copy-feats ark:foo.ark ark,t:txt.ark\n"
            

        ParseOptions po(usage);
    bool binary = true;
    bool htk_in = false;
    bool sphinx_in = false;
    bool compress = false;
    int32 compression_method_in = 1;
    std::string num_frames_wspecifier;

    po.Register("compress", &compress, "If true, write output in compressed form"
                "(only currently supported for wxfilename, i.e. archive/script,"
                "output)");

    int32 num_done = 0;

    CompressionMethod compression_method = static_cast<CompressionMethod>(compression_method_in);

    if (ClassifyRspecifier(po.GetArg(1), NULL, NULL) != kNoRspecifier) {
      // Copying tables of features.
      std::string rspecifier = po.GetArg(1);
      std::string wspecifier = po.GetArg(2);
    
      Int32Writer num_frames_writer(num_frames_wspecifier);

      if (!compress) {
        BaseFloatMatrixWriter kaldi_writer(wspecifier);
        if (htk_in) {
          SequentialTableReader<HtkMatrixHolder> htk_reader(rspecifier);
          for (; !htk_reader.Done(); htk_reader.Next(), num_done++) {
            kaldi_writer.Write(htk_reader.Key(), htk_reader.Value().first);
            if (!num_frames_wspecifier.empty())
              num_frames_writer.Write(htk_reader.Key(),
                                      htk_reader.Value().first.NumRows());
          }
        } else if (sphinx_in) {
          SequentialTableReader<SphinxMatrixHolder<> > sphinx_reader(rspecifier);
          for (; !sphinx_reader.Done(); sphinx_reader.Next(), num_done++) {
            kaldi_writer.Write(sphinx_reader.Key(), sphinx_reader.Value());
            if (!num_frames_wspecifier.empty())
              num_frames_writer.Write(sphinx_reader.Key(),
                                      sphinx_reader.Value().NumRows());
          }
        } else {
          SequentialBaseFloatMatrixReader kaldi_reader(rspecifier);
          for (; !kaldi_reader.Done(); kaldi_reader.Next(), num_done++) {
            kaldi_writer.Write(kaldi_reader.Key(), kaldi_reader.Value());
            if (!num_frames_wspecifier.empty())
              num_frames_writer.Write(kaldi_reader.Key(),
                                      kaldi_reader.Value().NumRows());
          }
        }
      } else {

        CompressedMatrixWriter kaldi_writer(wspecifier);
        if (htk_in) {
        } else {
        
          SequentialBaseFloatMatrixReader kaldi_reader(rspecifier);
          // foreach utt feats
          for (; !kaldi_reader.Done(); kaldi_reader.Next(), num_done++) {
            // 将utt feat 构成的Matrix 进行压缩存储.
            kaldi_writer.Write(kaldi_reader.Key(),
                               CompressedMatrix(kaldi_reader.Value(),
                                                compression_method));
          
            if (!num_frames_wspecifier.empty())
              num_frames_writer.Write(kaldi_reader.Key(),
                                      kaldi_reader.Value().NumRows());
          }
        }
      }
      KALDI_LOG << "Copied " << num_done << " feature matrices.";
      return (num_done != 0 ? 0 : 1);
    } else {
    }
  }



  //   取训练特征的 前 10k 数据, 准备用来进行 cmvn用.
  //   head -n 10000 $dir/train.scp > $dir/train.scp.10k

}


// 从预训练中引入特征设置？？？？ 
// ###### OPTIONALLY IMPORT FEATURE SETTINGS (from pre-training) ######






// ###### PREPARE FEATURE PIPELINE ######
// 构建两个基本nnet 并且逐个组合起来
// 1 nnet proto原型  <Splice> 升维
// 2 nnet cmvn归一化 <Shift><Scale> 归一化变换
void PREPARE_FEATURES(){
  

  //  =====================  一些 特征准备 =========================
  // 特征句柄
  // feats_tr="ark:copy-feats scp:$dir/train.scp ark:- |"
  // feats_cv="ark:copy-feats scp:$dir/cv.scp ark:- |"

  //   对特征使用 spker到普均值归一化  spker到普均值归一化的特征句柄
  //   feats_tr="$feats_tr apply-cmvn $cmvn_opts --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp ark:- ark:- |"
  //   feats_cv="$feats_cv apply-cmvn $cmvn_opts --utt2spk=ark:$data_cv/utt2spk scp:$data_cv/cmvn.scp ark:- ark:- |"

  // 部分训练数据子集
  // feats_tr_10k="${feats_tr/train.scp/train.scp.10k}"

  // 获得特征维度
  // # get feature dim,
  // feat_dim=$(feat-to-dim "$feats_tr_10k" -)




  //   =======================  特征转换 -- 实际上就是 一个nnet  后面构建nnet的过程就是在其基础上增加component======
  //   现在开始构建特征转换, 之后会进行神经网络训练, 然后会使用GPU进行计算. 在frame帧进行shuffling之前.
  //   GPU 既进行特征转换 还会用来进行NN训练 所以必须使用单线程.这也会减少CPU-GPU 上下行时间.

  //   这里的特征转换原型 实际上 就是最简单的nnet的原型.
  //   特征转换 步骤, 具体进行的是 将低维度特征扩展为高维度特征.
  //   使用splice 进行初始化一个 特征转换原型

  //     特征转换原型
  //     feature_transform_proto=$dir/splice${splice}.proto

  //     生成一个简单基本的特征转换模型是个基础原型proto -- 只执行了 Splice转换 只有一个Splice Component
  //     echo "<Splice> <InputDim> $feat_dim <OutputDim> $(((2*splice+1)*feat_dim)) <BuildVector> -$splice:$splice </BuildVector>"
  //           > $feature_transform_proto


  //   nnet-initialize 利用特征转换模型proto, 生成基本的nnet网络
  
  //   feature_transform是构建目标 只具有最简单的Splice Component.(shell中很多都是先设定目标去实现或者使用,所以有时比较变扭)
  //   feature_transform=$dir/tr_$(basename $feature_transform_proto .proto).nnet     (目标文件名字)
  //   利用 特征转换原型进行初始化 得到一个正常基础的nnet
  //   nnet-initialize --binary=false $feature_transform_proto $feature_transform

  // 根据nnet原型 初始化nnet网络参数.
  int nnet_initialize(int argc, char *argv[]) {

    const char *usage =
        "Initialize Neural Network parameters according to a prototype (nnet1).\n"
        "Usage:  nnet-initialize [options] <nnet-prototype-in> <nnet-out>\n"
        "e.g.: nnet-initialize --binary=false nnet.proto nnet.init\n";


    SetVerboseLevel(1);  // be verbose by default,
    // inline void SetVerboseLevel(int32 i) { g_kaldi_verbose_level = i; }  

    ParseOptions po(usage);
    bool binary_write = false;
    int32 seed = 777;

    std::string
        nnet_config_in_filename = po.GetArg(1),
        nnet_out_filename = po.GetArg(2);

    std::srand(seed);

    // initialize the network
    // 根据转换proto原型 构建基本nnet网络
    Nnet nnet;
    nnet.Init(nnet_config_in_filename);

    // store the network
    // 保存nnet, 实际上此时nnet 和 转换原型的能力一样.
    Output ko(nnet_out_filename, binary_write);
    nnet.Write(ko.Stream(), binary_write);
  }

  // Nnet init
  void Nnet::Init(const std::string &proto_file) {
    Input in(proto_file);
    std::istream &is = in.Stream();
    std::string proto_line, token;

    // 从转移模型初始化NNET, 每一行描述一个component组件.
    while (is >> std::ws, !is.eof()) {
      // get a line from the proto file,
      std::getline(is, proto_line);
      // get the 1st token from the line,
      std::istringstream(proto_line) >> std::ws >> token;
      // ignore these tokens:
      if (token == "<NnetProto>" || token == "</NnetProto>") continue;
      // create new component, append to Nnet,
      this->AppendComponentPointer(Component::Init(proto_line+"\n"));
    }
    // cleanup
    in.Close();
    Check();
  }

  // <Splice> <InputDim> $feat_dim <OutputDim> $(((2*splice+1)*feat_dim)) <BuildVector> -$splice:$splice </BuildVector>
  // 每个component 必须包含的几个Token -- <InputDim> <OutputDim>
  Component* Component::Init(const std::string &conf_line) {

    std::istringstream is(conf_line);
    std::string component_type_string;
    int32 input_dim, output_dim;

    // initialize component w/o internal data
    ReadToken(is, false, &component_type_string);
    // 将<Splice> 映射为一个componentType
    ComponentType component_type = MarkerToType(component_type_string);
    // 读取基本Token
    ExpectToken(is, false, "<InputDim>");
    ReadBasicType(is, false, &input_dim);
    ExpectToken(is, false, "<OutputDim>");
    ReadBasicType(is, false, &output_dim);
    // 根据基本Token构建component
    Component *ans = NewComponentOfType(component_type, input_dim, output_dim);

    // initialize internal data with the remaining part of config line
    // 用剩余 特殊Token初始化Component.
    ans->InitData(is);

    return ans;
  }

  // 所有可能的 nnet 的每一层.
  Component* Component::NewComponentOfType(ComponentType comp_type,
                                           int32 input_dim, int32 output_dim) {
    Component *ans = NULL;
    switch (comp_type) {
      case Component::kAffineTransform :
        ans = new AffineTransform(input_dim, output_dim);
        break;
      case Component::kLinearTransform :
        ans = new LinearTransform(input_dim, output_dim);
        break;
      case Component::kConvolutionalComponent :
        ans = new ConvolutionalComponent(input_dim, output_dim);
        break;
      case Component::kConvolutional2DComponent :
        ans = new Convolutional2DComponent(input_dim, output_dim);
        break;
      case Component::kLstmProjected :
        ans = new LstmProjected(input_dim, output_dim);
        break;
      case Component::kBlstmProjected :
        ans = new BlstmProjected(input_dim, output_dim);
        break;
      case Component::kRecurrentComponent :
        ans = new RecurrentComponent(input_dim, output_dim);
        break;
      case Component::kSoftmax :
        ans = new Softmax(input_dim, output_dim);
        break;
      case Component::kHiddenSoftmax :
        ans = new HiddenSoftmax(input_dim, output_dim);
        break;
      case Component::kBlockSoftmax :
        ans = new BlockSoftmax(input_dim, output_dim);
        break;
      case Component::kSigmoid :
        ans = new Sigmoid(input_dim, output_dim);
        break;
      case Component::kTanh :
        ans = new Tanh(input_dim, output_dim);
        break;
      case Component::kParametricRelu :
        ans = new ParametricRelu(input_dim, output_dim);
        break;
      case Component::kDropout :
        ans = new Dropout(input_dim, output_dim);
        break;
      case Component::kLengthNormComponent :
        ans = new LengthNormComponent(input_dim, output_dim);
        break;
      case Component::kRbm :
        ans = new Rbm(input_dim, output_dim);
        break;
      case Component::kSplice :
        ans = new Splice(input_dim, output_dim);
        break;
      case Component::kCopy :
        ans = new CopyComponent(input_dim, output_dim);
        break;
      case Component::kAddShift :
        ans = new AddShift(input_dim, output_dim);
        break;
      case Component::kRescale :
        ans = new Rescale(input_dim, output_dim);
        break;
      case Component::kKlHmm :
        ans = new KlHmm(input_dim, output_dim);
        break;
      case Component::kSentenceAveragingComponent :
        ans = new SentenceAveragingComponent(input_dim, output_dim);
        break;
      case Component::kSimpleSentenceAveragingComponent :
        ans = new SimpleSentenceAveragingComponent(input_dim, output_dim);
        break;
      case Component::kAveragePoolingComponent :
        ans = new AveragePoolingComponent(input_dim, output_dim);
        break;
      case Component::kAveragePooling2DComponent :
        ans = new AveragePooling2DComponent(input_dim, output_dim);
        break;
      case Component::kMaxPoolingComponent :
        ans = new MaxPoolingComponent(input_dim, output_dim);
        break;
      case Component::kMaxPooling2DComponent :
        ans = new MaxPooling2DComponent(input_dim, output_dim);
        break;
      case Component::kFramePoolingComponent :
        ans = new FramePoolingComponent(input_dim, output_dim);
        break;
      case Component::kParallelComponent :
        ans = new ParallelComponent(input_dim, output_dim);
        break;
      case Component::kMultiBasisComponent :
        ans = new MultiBasisComponent(input_dim, output_dim);
        break;
      case Component::kUnknown :
      default :
        KALDI_ERR << "Missing type: " << TypeToMarker(comp_type);
    }
    return ans;
  }


  //  ========================  为nnet 增加一个归一化层 ==============
  //   # Renormalize the MLP input to zero mean and unit variance,
  //   重新归一化 多层神经网络输入为 zero均值 单位协方差矩阵.

  //   此时:
  //   feature_transform_old 是当前的nnet模型 通过基本proto原型得到的nnet基础模型--Splice-Component.
  //   feature_transform为目标nnet模型(应用特征spker到普均值归一化)
  
  //   feature_transform_old=$feature_transform
  //   feature_transform=${feature_transform%.nnet}_cmvn-g.nnet

  //       > testVari=kkkk.nnet
  //       > echo ${testVari%.nnet}
  //       kkkk

  //   echo "# compute normalization stats from 10k sentences"
  //   对10k数据子集的数据 进行 统计归一化

  //   nnet-forward 跑一边nnet，可以带一些参数 实现不同的nnet执行流程
  //   compute-cmvn-stats 统计cmvn 所需统计量
  //   nnet-forward --print-args=true --use-gpu=yes $feature_transform_old "$feats_tr_10k" ark:- | \
  //     compute-cmvn-stats ark:- $dir/cmvn-g.stats

  //   echo "# + normalization of NN-input at '$feature_transform'"
  //   在 刚刚的nnet网络 上增加一个nnet结构     倒普均值归一化 转换操作-nnet.
  //   nnet-concat --binary=false $feature_transform_old \
  //     "cmvn-to-nnet --std-dev=$feats_std $dir/cmvn-g.stats -|" $feature_transform


  // 执行前向传播 按照nnet结构中的层对应的Component进行执行.
  // in:
  // final.nnet nnet模型
  // feature_r  输入特征
  // out:
  // feature_w  特征写入
  int nnet_forward(int argc, char *argv[]) {
    // 执行NNET的前向传播
    const char *usage =
        "Perform forward pass through Neural Network.\n"
        "Usage: nnet-forward [options] <nnet1-in> <feature-rspecifier> <feature-wspecifier>\n"
        "e.g.: nnet-forward final.nnet ark:input.ark ark:output.ark\n";



    std::string
        model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        feature_wspecifier = po.GetArg(3);

    // Select the GPU ---- CUDA编程
#if HAVE_CUDA == 1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    Nnet nnet_transf;
    if (feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }

    Nnet nnet;
    nnet.Read(model_filename);

    // optionally remove softmax, 可选删除最后的softmax层
    Component::ComponentType last_comp_type = nnet.GetLastComponent().GetType();
    if (no_softmax) {
      if (last_comp_type == Component::kSoftmax ||
          last_comp_type == Component::kBlockSoftmax) {
        KALDI_LOG << "Removing " << Component::TypeToMarker(last_comp_type)
                  << " from the nnet " << model_filename;
        nnet.RemoveLastComponent();
      } else {
        KALDI_WARN << "Last component 'NOT-REMOVED' by --no-softmax=true, "
                   << "the component was " << Component::TypeToMarker(last_comp_type);
      }
    }

    // avoid some bad option combinations,
    if (apply_log && no_softmax) {
      KALDI_ERR << "Cannot use both --apply-log=true --no-softmax=true, "
                << "use only one of the two!";
    }

    // we will subtract log-priors later,
    PdfPrior pdf_prior(prior_opts);

    // disable dropout,
    nnet_transf.SetDropoutRate(0.0);
    nnet.SetDropoutRate(0.0);

    kaldi::int64 tot_t = 0;


    // 因为对于nnet来说每一层的输出 都是下一层的特征, 所以应用nnet_forward时候就相当于 特征变换
    // 所以可以称nnet模型为 特征变换feature_transform 并且每层输出都能叫做特征.
    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    BaseFloatMatrixWriter feature_writer(feature_wspecifier);

    CuMatrix<BaseFloat> feats, feats_transf, nnet_out;
    Matrix<BaseFloat> nnet_out_host;

    Timer time;
    double time_now = 0;
    int32 num_done = 0;

    // foreach utt feature
    for (; !feature_reader.Done(); feature_reader.Next()) {
      // read
      Matrix<BaseFloat> mat = feature_reader.Value();
      std::string utt = feature_reader.Key();
      
      // push it to gpu,
      feats = mat;

      // fwd-pass, feature transform,
      // 此时nnet_transf 中没有Component ,
      // Feedforward前向传播 不做任何变换  feats_transf == feats
      nnet_transf.Feedforward(feats, &feats_transf);
      if (!KALDI_ISFINITE(feats_transf.Sum())) {  // check there's no nan/inf,
        KALDI_ERR << "NaN or inf found in transformed-features for " << utt;
      }

      // fwd-pass, nnet,
      // nnet 中存在Splice转换层
      nnet.Feedforward(feats_transf, &nnet_out);
      if (!KALDI_ISFINITE(nnet_out.Sum())) {  // check there's no nan/inf,
        KALDI_ERR << "NaN or inf found in nn-output for " << utt;
      }

      // convert posteriors to log-posteriors,
      // 将后验概率 转化为 log后验概率
      if (apply_log) {
        if (!(nnet_out.Min() >= 0.0 && nnet_out.Max() <= 1.0)) {
          KALDI_WARN << "Applying 'log()' to data which don't seem to be "
                     << "probabilities," << utt;
        }
        nnet_out.Add(1e-20);  // avoid log(0),
        nnet_out.ApplyLog();
      }

      // subtract log-priors from log-posteriors or pre-softmax,
      // 从log后验概率中减去先验概率
      if (prior_opts.class_frame_counts != "") {
        pdf_prior.SubtractOnLogpost(&nnet_out);
      }

      // download from GPU,
      // 从GPU中取出计算结果数据 ---> nnet_out_host
      nnet_out_host = Matrix<BaseFloat>(nnet_out);

      // write,
      if (!KALDI_ISFINITE(nnet_out_host.Sum())) {  // check there's no nan/inf,
        KALDI_ERR << "NaN or inf found in final output nn-output for " << utt;
      }
      feature_writer.Write(feature_reader.Key(), nnet_out_host);

      num_done++;
      tot_t += mat.NumRows();
    }

    // 关闭GPU
#if HAVE_CUDA == 1
    if (GetVerboseLevel() >= 1) {
      CuDevice::Instantiate().PrintProfile();
    }
#endif
  }

  // 链接多个nnet  ---> nnet_out
  int nnet_concat(int argc, char *argv[]) {
    const char *usage =
        "Concatenate Neural Networks (and possibly change binary/text format)\n"
        "Usage: nnet-concat [options] <nnet-in1> <...> <nnet-inN> <nnet-out>\n"
        "e.g.:\n"
        " nnet-concat --binary=false nnet.1 nnet.2 nnet.1.2\n";

    ParseOptions po(usage);

    bool binary_write = true;
    po.Register("binary", &binary_write, "Write output in binary mode");
    binary_write = false;


    std::string model_in_filename = po.GetArg(1);
    std::string model_in_filename_next;
    std::string model_out_filename = po.GetArg(po.NumArgs());

    // read the first nnet,
    Nnet nnet;
    {
      bool binary_read;
      Input ki(model_in_filename, &binary_read);
      nnet.Read(ki.Stream(), binary_read);
    }

    // read all the other nnets,
    for (int32 i = 2; i < po.NumArgs(); i++) {
      // read the nnet,
      model_in_filename_next = po.GetArg(i);
      KALDI_LOG << "Concatenating " << model_in_filename_next;
      Nnet nnet_next;
      {
        bool binary_read;
        Input ki(model_in_filename_next, &binary_read);
        nnet_next.Read(ki.Stream(), binary_read);
      }

      // 直接调用AppendNnet() 将nnet_next 中的所有Component 添加到 nnet中.
      // append nnet_next to the network nnet,
      nnet.AppendNnet(nnet_next);
    }

    // finally write the nnet to disk,
    {
      Output ko(model_out_filename, binary_write);
      nnet.Write(ko.Stream(), binary_write);
    }

  }

  // 构建一个 归一化的nnet 里面包含了两个Component
  // 1 shift设置均值为0， 2 拉伸变换scale --归一化
  int cmv_to_nnet(int argc, char *argv[]) {
    const char *usage =
        // 将cmvn-stats 转化为Component.
        "Convert cmvn-stats into <AddShift> and <Rescale> components.\n"
        "Usage:  cmvn-to-nnet [options] <transf-in> <nnet-out>\n"
        "e.g.:\n"
        " cmvn-to-nnet --binary=false transf.mat nnet.mdl\n";


    bool binary_write = false;
    float std_dev = 1.0;
    float var_floor = 1e-10;
    float learn_rate_coef = 0.0;

    ParseOptions po(usage);
    po.Register("std-dev", &std_dev, "Standard deviation of the output.");

    std::string
        cmvn_stats_rxfilename = po.GetArg(1),
        model_out_filename = po.GetArg(2);

    // read the matrix
    Matrix<double> cmvn_stats;
    {
      bool binary_read;
      Input ki(cmvn_stats_rxfilename, &binary_read);
      cmvn_stats.Read(ki.Stream(), binary_read);
    }
    // [2 X N]
    // [ dim0-mean, dim1-mean, dim2-mean, frames-cnt
    //   dim0-var,  dim1-var,  dim2-var,  NULL ]
    int32 num_dims = cmvn_stats.NumCols() - 1;
    // (0, col-1) 保存的是帧总数. 具体要看cmvn_stats里面的数据结构.
    double frame_count = cmvn_stats(0, cmvn_stats.NumCols() - 1);

    // buffers for shift and scale
    Vector<BaseFloat> shift(num_dims);
    Vector<BaseFloat> scale(num_dims);

    // 计算每个维度的 平移 和 拉伸
    // compute the shift and scale per each dimension
    for (int32 d = 0; d < num_dims; d++) {
      BaseFloat mean = cmvn_stats(0, d) / frame_count;
      BaseFloat var = cmvn_stats(1, d) / frame_count - mean * mean;
      if (var <= var_floor) {
        KALDI_WARN << "Very small variance " << var
                   << " flooring to " << var_floor;
        var = var_floor;
      }
      shift(d) = -mean;
      scale(d) = std_dev / sqrt(var);
    }

    // create empty nnet,
    Nnet nnet;

    // append shift component to nnet,
    {
      AddShift shift_component(shift.Dim(), shift.Dim());
      shift_component.SetParams(shift);
      shift_component.SetLearnRateCoef(learn_rate_coef);
      nnet.AppendComponent(shift_component);
    }

    // append scale component to nnet,
    {
      Rescale scale_component(scale.Dim(), scale.Dim());
      scale_component.SetParams(scale);
      scale_component.SetLearnRateCoef(learn_rate_coef);
      nnet.AppendComponent(scale_component);
    }

    // write the nnet,
    {
      Output ko(model_out_filename, binary_write);
      nnet.Write(ko.Stream(), binary_write);
      KALDI_LOG << "Written cmvn in 'nnet1' model to: " << model_out_filename;
    }
  }

}






// ###### Show the final 'feature_transform' in the log,
// 显示最终的 特征转换 nnet.
// echo
// echo "### Showing the final 'feature_transform':"
// nnet-info $feature_transform
// echo "###"


// ###### MAKE LINK TO THE FINAL feature_transform, so the other scripts will find it ######
// [ -f $dir/final.feature_transform ] && unlink $dir/final.feature_transform
// (cd $dir; ln -s $(basename $feature_transform) final.feature_transform )
// feature_transform=$dir/final.feature_transform


// ###### INITIALIZE THE NNET ######
// echo
// 判断是否已经存在了nnet_init/nnet_proto可能的基本nnet模型,
// 如果存在就合并刚刚得到的nnet 和 nnet_init/nnet_proto
// echo "# NN-INITIALIZATION"
// if [ ! -z $nnet_init ]; then                               NULL
// elif [ ! -z $nnet_proto ]; then                            NULL
// else                                                       THIS
//   echo "# getting input/output dims :"
//   # input-dim,
//   get_dim_from=$feature_transform
//   =============================================== $DBN is NULL
//   [ ! -z "$dbn" ] && get_dim_from="nnet-concat $feature_transform '$dbn' -|"

//   经过nnet后得到的 feats dim
//   num_fea=$(feat-to-dim "$feats_tr_10k nnet-forward \"$get_dim_from\" ark:- ark:- |" -)

//   # output-dim,  pdf-总数, 实际就是nnet目标维度.
//   [ -z $num_tgt ] &&                    ===============    YES  pdf总数
//       num_tgt=$(hmm-info --print-args=false $alidir/final.mdl | grep pdfs | awk '{ print $NF }')

//   # make network prototype, ============== 生成一个nnet网络原型 (实际上刚刚的已经是nnet网络原形了).
//   nnet_proto=$dir/nnet.proto
//   echo "# genrating network prototype $nnet_proto"
//   case "$network_type" in
//     dnn)
//       utils/nnet/make_nnet_proto.py   $proto_opts  \
//         ${bn_dim:+ --bottleneck-dim=$bn_dim} \
//         $num_fea   $num_tgt    $hid_layers    $hid_dim
//         > $nnet_proto

//   根据 inputdim, outputdim, hidden layers, hid_dims 构建整体的nnet网络结构 === >nnet.proto.
//   实际上是一个写 proto文件的过程.
void make_nnet_proto_py(){

  // # Generated Nnet prototype, to be initialized by 'nnet-initialize'.

  // ================ 使用方法
  // from optparse import OptionParser
  // usage="%prog [options] <feat-dim> <num-leaves> <num-hid-layers> <num-hid-neurons> >nnet-proto-file"
  // parser = OptionParser(usage)

  
  // Softmax设置相关
  // # Softmax related, 

  // 拓扑结果相关
  // # Topology related,
  // parser.add_option('--bottleneck-dim', dest='bottleneck_dim',
  //                    help='Make bottleneck network with desired bn-dim (0 = no bottleneck) [default: %default]',
  //                    default=0, type='int');

  // o 是程序的配置选项, 主要用来控制nnet中各种Component的参数.
  // ？？？？？？？？？？？？
  // # A HACK TO PASS MULTI-WORD OPTIONS, WORDS ARE CONNECTED BY UNDERSCORES '_',
  // o.activation_opts = o.activation_opts.replace("_"," ")
  // o.affine_opts = o.affine_opts.replace("_"," ")
  // o.dropout_opts = o.dropout_opts.replace("_"," ")


  
  // 获得 输入dim  输出dim  隐藏层数目  隐藏层节点数
  // (feat_dim, num_leaves, num_hid_layers, num_hid_neurons) = map(int,args);



  // 如果使用softmax 必须要求softmax的输出dim == num_leaves?
  // if o.block_softmax_dims:
  //   assert(  sum(map(int, re.split("[,:]", o.block_softmax_dims))) == num_leaves ) 


  
  // # Optionaly scale
  // def Glorot(dim1, dim2):
  //   if o.with_glorot:
  //     # 35.0 = magic number, gives ~1.0 in inner layers for hid-dim 1024dim,
  //     return 35.0 * math.sqrt(2.0/(dim1+dim2));
  //   else:
  //     return 1.0


  // # NO HIDDEN LAYER, ADDING BOTTLENECK!
  // # No hidden layer while adding bottleneck means:
  // # - add bottleneck layer + hidden layer + output layer


  // # THE USUAL DNN PROTOTYPE STARTS HERE!
  // # Assuming we have >0 hidden layers,
  // assert(num_hid_layers > 0)




  // 构建proto过程
  // # Begin the prototype,
  // # First AffineTranform
  // print "<AffineTransform> <InputDim> %d <OutputDim> %d <BiasMean> %f <BiasRange> %f <ParamStddev> %f <MaxNorm> %f %s" % \
  //       (feat_dim, num_hid_neurons, o.hid_bias_mean, o.hid_bias_range, \
  //        (o.param_stddev_factor * Glorot(feat_dim, num_hid_neurons) * \
  //         (math.sqrt(1.0/12.0) if o.smaller_input_weights else 1.0)), o.max_norm, o.affine_opts)

  // print "%s <InputDim> %d <OutputDim> %d %s" % (o.activation_type, num_hid_neurons, num_hid_neurons, o.activation_opts)
  // if o.with_dropout:
  //   print "<Dropout> <InputDim> %d <OutputDim> %d %s" % (num_hid_neurons, num_hid_neurons, o.dropout_opts)


  // # Internal AffineTransforms,
  // for i in range(num_hid_layers-1):
  //   print "<AffineTransform> <InputDim> %d <OutputDim> %d <BiasMean> %f <BiasRange> %f <ParamStddev> %f <MaxNorm> %f %s" % \
  //         (num_hid_neurons, num_hid_neurons, o.hid_bias_mean, o.hid_bias_range, \
  //          (o.param_stddev_factor * Glorot(num_hid_neurons, num_hid_neurons)), o.max_norm, o.affine_opts)
  //   print "%s <InputDim> %d <OutputDim> %d %s" % (o.activation_type, num_hid_neurons, num_hid_neurons, o.activation_opts)
  //   if o.with_dropout:
  //     print "<Dropout> <InputDim> %d <OutputDim> %d %s" % (num_hid_neurons, num_hid_neurons, o.dropout_opts)
        

  // 可选 增加瓶颈层, 这里没有添加瓶颈层.
  // # Optionaly add bottleneck,
  // if o.bottleneck_dim != 0:
  //   assert(o.bottleneck_dim > 0)
  //   if o.bottleneck_trick:
  //     # 25% smaller stddev -> small bottleneck range, 10x smaller learning rate
  //     print "<LinearTransform> <InputDim> %d <OutputDim> %d <ParamStddev> %f <LearnRateCoef> %f" % \
  //      (num_hid_neurons, o.bottleneck_dim, \
  //       (o.param_stddev_factor * Glorot(num_hid_neurons, o.bottleneck_dim) * 0.75 ), 0.1)
  //     # 25% smaller stddev -> smaller gradient in prev. layer, 10x smaller learning rate for weigts & biases
  //     print "<AffineTransform> <InputDim> %d <OutputDim> %d <BiasMean> %f <BiasRange> %f <ParamStddev> %f <LearnRateCoef> %f <BiasLearnRateCoef> %f <MaxNorm> %f %s" % \
  //      (o.bottleneck_dim, num_hid_neurons, o.hid_bias_mean, o.hid_bias_range, \
  //       (o.param_stddev_factor * Glorot(o.bottleneck_dim, num_hid_neurons) * 0.75 ), 0.1, 0.1, o.max_norm, o.affine_opts)
  //   else:
  //     # Same learninig-rate and stddev-formula everywhere,
  //     print "<LinearTransform> <InputDim> %d <OutputDim> %d <ParamStddev> %f" % \
  //      (num_hid_neurons, o.bottleneck_dim, \
  //       (o.param_stddev_factor * Glorot(num_hid_neurons, o.bottleneck_dim)))
  //     print "<AffineTransform> <InputDim> %d <OutputDim> %d <BiasMean> %f <BiasRange> %f <ParamStddev> %f <MaxNorm> %f %s" % \
  //      (o.bottleneck_dim, num_hid_neurons, o.hid_bias_mean, o.hid_bias_range, \
  //       (o.param_stddev_factor * Glorot(o.bottleneck_dim, num_hid_neurons)), o.max_norm, o.affine_opts)
  //   print "%s <InputDim> %d <OutputDim> %d %s" % (o.activation_type, num_hid_neurons, num_hid_neurons, o.activation_opts)
  //   if o.with_dropout:
  //     print "<Dropout> <InputDim> %d <OutputDim> %d %s" % (num_hid_neurons, num_hid_neurons, o.dropout_opts)

  
  //   最终在softmax之前, 网络节点总数应该是与最终节点数相同, 然后经过softmax？ 
  //   # Last AffineTransform (10x smaller learning rate on bias)
  //   print "<AffineTransform> <InputDim> %d <OutputDim> %d <BiasMean> %f <BiasRange> %f <ParamStddev> %f <LearnRateCoef> %f <BiasLearnRateCoef> %f" % \
  //       (num_hid_neurons, num_leaves, 0.0, 0.0, \
  //        (o.param_stddev_factor * Glorot(num_hid_neurons, num_leaves)), 1.0, 0.1)

  //   可选增加 softmax层
  //   # Optionaly append softmax
  //   if o.with_softmax:
  //     if o.block_softmax_dims == "":
  //       print "<Softmax> <InputDim> %d <OutputDim> %d" % (num_leaves, num_leaves)
  //     else:
  //       print "<BlockSoftmax> <InputDim> %d <OutputDim> %d <BlockDims> %s" % (num_leaves, num_leaves, o.block_softmax_dims)

}


//   nnet_initialize 根据 nnet.proto 生成 ====> nnet.init
//   # initialize,
//   nnet_init=$dir/nnet.init
//   echo "# initializing the NN '$nnet_proto' -> '$nnet_init'"
//   nnet-initialize --seed=$seed $nnet_proto $nnet_init

//   可选使用dbn(受限玻尔兹曼机)应用到 得到的nnet.
//   # optionally prepend dbn to the initialization,  
//   if [ ! -z "$dbn" ]; then
//     nnet_init_old=$nnet_init; nnet_init=$dir/nnet_dbn_dnn.init
//     nnet-concat "$dbn" $nnet_init_old $nnet_init
//   fi

// fi


// ###### TRAIN ######
// 核心代码 train_scheduler.sh 使用nnet_init模型， feat_train, feat_cv  label_train, label_cv 进行nnet的训练 最终将结果输出到$dir

// echo
// echo "# RUNNING THE NN-TRAINING SCHEDULER"
// steps/nnet/train_scheduler.sh \
//   ${feature_transform:+ --feature-transform $feature_transform} \
//   --learn-rate $learn_rate \
//   $nnet_init "$feats_tr" "$feats_cv" "$labels_tr" "$labels_cv" $dir

// echo "$0: Successfuly finished. '$dir'"

void train_scheduler(){

  // # Schedules epochs and controls learning rate during the neural network training
  // 神经网络训练期间的 完整循环调度 以及 学习率控制

  // # Begin configuration.
  {
    // # training options,
    // learn_rate=0.008
    // momentum=0
    // l1_penalty=0
    // l2_penalty=0

    // # data processing,
    // train_tool="nnet-train-frmshuff"
    // train_tool_opts="--minibatch-size=256 --randomizer-size=32768 --randomizer-seed=777"
    // feature_transform=

    // split_feats= # int -> number of splits 'feats.scp -> feats.${i}.scp', starting from feats.1.scp,
    //              # (data are alredy shuffled and split to N parts),
    //              # empty -> no splitting,

  
    // # learn rate scheduling,
    // max_iters=20
    // min_iters=0 # keep training, disable weight rejection, start learn-rate halving as usual,
    // keep_lr_iters=0 # fix learning rate for N initial epochs, disable weight rejection,
    // dropout_schedule= # dropout-rates for N initial epochs, for example: 0.1,0.1,0.1,0.1,0.1,0.0
    // start_halving_impr=0.01
    // end_halving_impr=0.001
    // halving_factor=0.5

    // # misc,
    // verbose=0 # 0 No GPU time-stats, 1 with GPU time-stats (slower),
    // frame_weights=
    // utt_weights=

    // # End configuration.
  }

  // # USE
  {
    // if [ $# != 6 ]; then
    //    echo "Usage: $0 <mlp-init> <feats-tr> <feats-cv> <labels-tr> <labels-cv> <exp-dir>"
    //    echo " e.g.: $0 0.nnet scp:train.scp scp:cv.scp ark:labels_tr.ark ark:labels_cv.ark exp/dnn1"
    //    echo "main options (for others, see top of script file)"
    //    echo "  --config <config-file>  # config containing options"
    //    exit 1;
    // fi

    // mlp_init=$1
    // feats_tr=$2
    // feats_cv=$3
    // labels_tr=$4
    // labels_cv=$5
    // dir=$6

    // [ ! -d $dir ] && mkdir $dir
    // [ ! -d $dir/log ] && mkdir $dir/log
    // [ ! -d $dir/nnet ] && mkdir $dir/nnet

    
    // dropout_array=($(echo ${dropout_schedule} | tr ',' ' '))

    // # Skip training  //跳过训练, 如果存在了 final.nnet
    // [ -e $dir/final.nnet ] && echo "'$dir/final.nnet' exists, skipping training" && exit 0
  }

  // ##############################
  // # start training

  // # choose mlp to start with,
  // mlp_best=$mlp_init
  // mlp_base=${mlp_init##*/};
  // mlp_base=${mlp_base%.*}  去掉nnet网络结构文件的路径以及后缀 得到名字.

  
  // 可以进行从最佳时期 恢复训练, 通过保存学习率 以及学习结果nnet 结构文件.
  // # optionally resume training from the best epoch, using saved learning-rate,
  // [ -e $dir/.mlp_best ] && mlp_best=$(cat $dir/.mlp_best)
  // [ -e $dir/.learn_rate ] && learn_rate=$(cat $dir/.learn_rate)


  // ------------------------------------------------------------------------------------
  // train_tool="nnet-train-frmshuff"
  // train_tool_opts="--minibatch-size=256 --randomizer-size=32768 --randomizer-seed=777"
  // ------------------------------------------------------------------------------------
  // ==================================  原始网络的交叉验证集合的设置？
  // # cross-validation on original network,
  // log=$dir/log/iter00.initial.log; hostname>$log

  // =============================  nnet-train-frmshuff  按帧进行nnet训练
  // 这里 实际上并没有训练，只是进行了一次前向传播, 然后通过xent交叉熵测试了下当前的nnet网络.
  // $train_tool --cross-validate=true --randomize=false --verbose=$verbose $train_tool_opts \
  //   ${feature_transform:+ --feature-transform=$feature_transform} \
  //   "$feats_cv" "$labels_cv" $mlp_best \

  // 执行NNET训练的一次完整循环 训练目标是 pdf后验概率. 通过ali-to-post提供.
  void nnet_train_frmshuff___liu_easyGo(){
    const char *usage =
        "Perform one iteration (epoch) of Neural Network training with\n"
        "mini-batch Stochastic Gradient Descent. The training targets\n"
        "are usually pdf-posteriors, prepared by ali-to-post.\n"
        "Usage:  nnet-train-frmshuff [options] <feature-rspecifier> <targets-rspecifier> <model-in> [<model-out>]\n"
        "e.g.: nnet-train-frmshuff scp:feats.scp ark:posterior.ark nnet.init nnet.iter1\n";

    ParseOptions po(usage);

    NnetTrainOptions trn_opts;
    trn_opts.Register(&po);
    NnetDataRandomizerOptions rnd_opts;
    rnd_opts.Register(&po);
    LossOptions loss_opts;
    loss_opts.Register(&po);

    // TRUE
    bool crossvalidate = false;
    po.Register("cross-validate", &crossvalidate,
                "Perform cross-validation (don't back-propagate)");

    // FALSE
    bool randomize = true;
    po.Register("randomize", &randomize,
                "Perform the frame-level shuffling within the Cache::");

    // TRANSFORM
    std::string feature_transform;
    po.Register("feature-transform", &feature_transform,
                "Feature transform in Nnet format");

    // 目标函数(损失函数)  xent 交叉熵损失函数.
    std::string objective_function = "xent";
    po.Register("objective-function", &objective_function,
                "Objective function : xent|mse|multitask");

    // 
    int32 max_frames = 360000;
    po.Register("max-frames", &max_frames,
                "Maximum number of frames an utterance can have (skipped if longer)");

    std::string use_gpu="yes";
    po.Register("use-gpu", &use_gpu,
                "yes|no|optional, only has effect if compiled with CUDA");

    std::string
        feature_rspecifier = po.GetArg(1),
        targets_rspecifier = po.GetArg(2),
        model_filename = po.GetArg(3);

    // 如果不是交叉验证集 那就是在进行训练, 必须有输出NNET.
    // 如果是交叉验证, 那就是直接输出结果即可.
    std::string target_model_filename;
    if (!crossvalidate) {
      target_model_filename = po.GetArg(4);
    }

#if HAVE_CUDA == 1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    Nnet nnet_transf;
    if (feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }

    Nnet nnet;
    nnet.Read(model_filename);
    nnet.SetTrainOptions(trn_opts);

    if (crossvalidate) {
      nnet_transf.SetDropoutRate(0.0);
      nnet.SetDropoutRate(0.0);
    }

    kaldi::int64 total_frames = 0;
    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessPosteriorReader targets_reader(targets_rspecifier);
    RandomAccessBaseFloatVectorReader weights_reader;
    
    RandomizerMask randomizer_mask(rnd_opts);
    MatrixRandomizer feature_randomizer(rnd_opts);
    PosteriorRandomizer targets_randomizer(rnd_opts);
    VectorRandomizer weights_randomizer(rnd_opts);

    // 交叉熵
    Xent xent(loss_opts);
    // 均方误差
    Mse mse(loss_opts);

    // 判断是否是多损失函数的训练.
    MultiTaskLoss multitask(loss_opts);
    if (0 == objective_function.compare(0, 9, "multitask")) {
      // objective_function contains something like :
      // 'multitask,xent,2456,1.0,mse,440,0.001'
      //
      // the meaning is following:
      // 'multitask,<type1>,<dim1>,<weight1>,...,<typeN>,<dimN>,<weightN>'
      multitask.InitFromString(objective_function);
    }

    
    CuMatrix<BaseFloat> feats_transf, nnet_out, obj_diff;

    Timer time;
    KALDI_LOG << (crossvalidate ? "CROSS-VALIDATION" : "TRAINING")
              << " STARTED";

    int32
        num_done = 0,
        num_no_tgt_mat = 0,
        num_other_error = 0;

    // main loop,
    while (!feature_reader.Done()) {
#if HAVE_CUDA == 1
      // check that GPU computes accurately,
      CuDevice::Instantiate().CheckGpuHealth();
#endif
      // fill the randomizer,
      for ( ; !feature_reader.Done(); feature_reader.Next()) {
        
        std::string utt = feature_reader.Key();
        KALDI_VLOG(3) << "Reading " << utt;
        // check that we have targets,

        // get feature / target pair,
        Matrix<BaseFloat> mat = feature_reader.Value();
        Posterior targets = targets_reader.Value(utt);
        
        // skip too long utterances (or we run out of memory),
        if (mat.NumRows() > max_frames) {
          KALDI_WARN << "Utterance too long, skipping! " << utt
                     << " (length " << mat.NumRows() << ", max_frames "
                     << max_frames << ")";
          num_other_error++;
          continue;
        }

        // 修正有些(特征长度 与 对齐长度)匹配不上的句子, 或者都截断到最短, 或者直接舍弃.
        // correct small length mismatch or drop sentence,
        {
          // add lengths to vector,
          std::vector<int32> length;
          length.push_back(mat.NumRows());
          length.push_back(targets.size());
          length.push_back(weights.Dim());
          // find min, max,
          int32 min = *std::min_element(length.begin(), length.end());
          int32 max = *std::max_element(length.begin(), length.end());
          // fix or drop ?
          if (max - min < length_tolerance) {
            // we truncate to shortest,
            if (mat.NumRows() != min) mat.Resize(min, mat.NumCols(), kCopyData);
            if (targets.size() != min) targets.resize(min);
            if (weights.Dim() != min) weights.Resize(min, kCopyData);
          } else {
            KALDI_WARN << "Length mismatch! Targets " << targets.size()
                       << ", features " << mat.NumRows() << ", " << utt;
            num_other_error++;
            continue;
          }
        }

        // 应用 特征转换, 将 utt-feature 进行特征变换--> feats_transf.
        // apply feature transform (if empty, input is copied),
        nnet_transf.Feedforward(CuMatrix<BaseFloat>(mat), &feats_transf);

        // 去掉 权重为0的帧 因为认为这些帧不重要.
        // remove frames with '0' weight from training,
        {
          // are there any frames to be removed? (frames with zero weight),
          BaseFloat weight_min = weights.Min();
          KALDI_ASSERT(weight_min >= 0.0);
          if (weight_min == 0.0) {
            // create vector with frame-indices to keep,
            std::vector<MatrixIndexT> keep_frames;
            for (int32 i = 0; i < weights.Dim(); i++) {
              if (weights(i) > 0.0) {
                keep_frames.push_back(i);
              }
            }

            // when all frames are removed, we skip the sentence,
            if (keep_frames.size() == 0) continue;

            // filter feature-frames,
            CuMatrix<BaseFloat> tmp_feats(keep_frames.size(), feats_transf.NumCols());
            tmp_feats.CopyRows(feats_transf, CuArray<MatrixIndexT>(keep_frames));
            tmp_feats.Swap(&feats_transf);

            // filter targets,
            Posterior tmp_targets;
            for (int32 i = 0; i < keep_frames.size(); i++) {
              tmp_targets.push_back(targets[keep_frames[i]]);
            }
            tmp_targets.swap(targets);

            // filter weights,
            Vector<BaseFloat> tmp_weights(keep_frames.size());
            for (int32 i = 0; i < keep_frames.size(); i++) {
              tmp_weights(i) = weights(keep_frames[i]);
            }
            tmp_weights.Swap(&weights);
          }
        }

        // utt-feats_transf 长度 一定会等于 对齐结果长度
        // pass data to randomizers,
        KALDI_ASSERT(feats_transf.NumRows() == targets.size());
        feature_randomizer.AddData(feats_transf);
        targets_randomizer.AddData(targets);
        weights_randomizer.AddData(weights);
        
        num_done++;

        // report the speed,  打印进度.
        if (num_done % 5000 == 0) {
          double time_now = time.Elapsed();
          KALDI_VLOG(1) << "After " << num_done << " utterances: "
                        << "time elapsed = " << time_now / 60 << " min; "
                        << "processed " << total_frames / time_now << " frames per sec.";
        }
      }

      // randomize,
      if (!crossvalidate && randomize) {
        const std::vector<int32>& mask =
            randomizer_mask.Generate(feature_randomizer.NumFrames());
        feature_randomizer.Randomize(mask);
        targets_randomizer.Randomize(mask);
        weights_randomizer.Randomize(mask);
      }

      // 从randomizers随机化数据中 进行训练, 使用Mini-batches 批量训练.
      // train with data from randomizers (using mini-batches),
      for ( ; !feature_randomizer.Done(); feature_randomizer.Next(),
                targets_randomizer.Next(), weights_randomizer.Next()) {
        // 获得一批 特征-对齐 用来训练
        // get block of feature/target pairs,
        const CuMatrixBase<BaseFloat>& nnet_in = feature_randomizer.Value();
        const Posterior& nnet_tgt = targets_randomizer.Value();
        const Vector<BaseFloat>& frm_weights = weights_randomizer.Value();

        // 前向传播
        // forward pass,
        nnet.Propagate(nnet_in, &nnet_out);

        // 求值  损失函数
        // evaluate objective function we've chosen,
        if (objective_function == "xent") {
          // 根据计算中的帧权重 梯度重新计算拉伸
          // gradients re-scaled by weights in Eval,
          xent.Eval(frm_weights, nnet_out, nnet_tgt, &obj_diff);
        } 

        // 如果不是交叉验证 进行后向残差传播
        if (!crossvalidate) {
          // back-propagate, and do the update,
          nnet.Backpropagate(obj_diff, NULL);
        }
        
        // 1st mini-batch : show what happens in network,
        if (total_frames == 0) {
          KALDI_VLOG(1) << "### After " << total_frames << " frames,";
          KALDI_VLOG(1) << nnet.InfoPropagate();
          if (!crossvalidate) {
            KALDI_VLOG(1) << nnet.InfoBackPropagate();
            KALDI_VLOG(1) << nnet.InfoGradient();
          }
        }

        // 冗余信息 日志
        // VERBOSE LOG
        // monitor the NN training (--verbose=2),
        if (GetVerboseLevel() >= 2) {
          static int32 counter = 0;
          counter += nnet_in.NumRows();
          // print every 25k frames,
          if (counter >= 25000) {
            KALDI_VLOG(2) << "### After " << total_frames << " frames,";
            KALDI_VLOG(2) << nnet.InfoPropagate();
            if (!crossvalidate) {
              KALDI_VLOG(2) << nnet.InfoBackPropagate();
              KALDI_VLOG(2) << nnet.InfoGradient();
            }
            counter = 0;
          }
        }

        total_frames += nnet_in.NumRows();
      }
    }  // main loop,

    // after last mini-batch : show what happens in network,
    {
    KALDI_VLOG(1) << "### After " << total_frames << " frames,";
    KALDI_VLOG(1) << nnet.InfoPropagate();
    if (!crossvalidate) {
      KALDI_VLOG(1) << nnet.InfoBackPropagate();
      KALDI_VLOG(1) << nnet.InfoGradient();
    }
    }

    if (!crossvalidate) {
      nnet.Write(target_model_filename, binary);
    }

    {
    KALDI_LOG << "Done " << num_done << " files, "
              << num_no_tgt_mat << " with no tgt_mats, "
              << num_other_error << " with other errors. "
              << "[" << (crossvalidate ? "CROSS-VALIDATION" : "TRAINING")
              << ", " << (randomize ? "RANDOMIZED" : "NOT-RANDOMIZED")
              << ", " << time.Elapsed() / 60 << " min, processing "
              << total_frames / time.Elapsed() << " frames per sec.]";
    }

    if (objective_function == "xent") {
      KALDI_LOG << xent.ReportPerClass();
      KALDI_LOG << xent.Report();
    }
    
#if HAVE_CUDA == 1
    CuDevice::Instantiate().PrintProfile();
#endif
  }


  //  ============== 获得刚刚 nnet-train-frmshuff 测试交叉验证集合的交叉熵误差 并打印显示.
  // loss=$(cat $dir/log/iter00.initial.log | grep "AvgLoss:" | tail -n 1 | awk '{ print $4; }')
  // loss_type=$(cat $dir/log/iter00.initial.log | grep "AvgLoss:" | tail -n 1 | awk '{ print $5; }')
  // echo "CROSSVAL PRERUN AVG.LOSS $(printf "%.4f" $loss) $loss_type"


  // 恢复      lr-halving 减半变量
  // # resume lr-halving,  
  // halving=0
  // [ -e $dir/.halving ] && halving=$(cat $dir/.halving)

  
  //  ==========================  迭代训练过程 ==========================
  // # training,
  
  // for iter in $(seq -w $max_iters); do
  //   echo -n "ITERATION $iter: "
  //   mlp_next=$dir/nnet/${mlp_base}_iter${iter}

  //   # skip iteration (epoch) if already done,
  //   [ -e $dir/.done_iter$iter ] && echo -n "skipping... " && ls $mlp_next* && continue

  //   ============ 判断当前迭代次数, 修改当前 nnet中的dropout率.
  //   (()) 是进行整数求值
  //   -'' 表示变量清除无用的''?? 是为了确定变量正常
  //   # set dropout-rate from the schedule,
  //   if [ -n ${dropout_array[$((${iter#0}-1))]-''} ]; then
  //     dropout_rate=${dropout_array[$((${iter#0}-1))]}
  //     nnet-copy --dropout-rate=$dropout_rate $mlp_best ${mlp_best}.dropout_rate${dropout_rate}
  //     mlp_best=${mlp_best}.dropout_rate${dropout_rate}
  //   fi

  //   选择划分数据集？ 未进行划分
  //   # select the split,
  //   feats_tr_portion="$feats_tr"      # no split?
  //   if [ -n "$split_feats" ]; then
  //     portion=$((1 + iter % split_feats))
  //     feats_tr_portion="${feats_tr/train.scp/train.${portion}.scp}"
  //   fi


  //  penalty 处罚 verbose=0 冗余? momentum 动量???
  //   ====================== TRAINING ===================
  //   # training,
  //   log=$dir/log/iter${iter}.tr.log; hostname>$log

  // ------------------------------------------------------------------------------------
  // train_tool="nnet-train-frmshuff"
  // train_tool_opts="--minibatch-size=256 --randomizer-size=32768 --randomizer-seed=777"
  // ------------------------------------------------------------------------------------
  //   $train_tool --cross-validate=false --randomize=true --verbose=$verbose $train_tool_opts \
  //     --learn-rate=$learn_rate --momentum=$momentum \
  //     --l1-penalty=$l1_penalty --l2-penalty=$l2_penalty \
  //     ${feature_transform:+ --feature-transform=$feature_transform} \
  //     "$feats_tr_portion" "$labels_tr" $mlp_best $mlp_next \
 
  //   获取训练数据交叉熵残差结果  --- 这种误差叫经验误差？ 在验证集上的是结构误差?
  //   tr_loss=$(cat $dir/log/iter${iter}.tr.log | grep "AvgLoss:" | tail -n 1 | awk '{ print $4; }')
  //   echo -n "TRAIN AVG.LOSS $(printf "%.4f" $tr_loss), (lrate$(printf "%.6g" $learn_rate)), "

  // frm帧 shuff?? 真实训练过程
  int nnet_train_frmshuff(int argc, char *argv[]) {
    const char *usage =
        "Perform one iteration (epoch) of Neural Network training with\n"
        "mini-batch Stochastic Gradient Descent. The training targets\n"
        "are usually pdf-posteriors, prepared by ali-to-post.\n"
        "Usage:  nnet-train-frmshuff [options] <feature-rspecifier> <targets-rspecifier> <model-in> [<model-out>]\n"
        "e.g.: nnet-train-frmshuff scp:feats.scp ark:posterior.ark nnet.init nnet.iter1\n";

    ParseOptions po(usage);

    NnetTrainOptions trn_opts;
    trn_opts.Register(&po);
    NnetDataRandomizerOptions rnd_opts;
    rnd_opts.Register(&po);
    LossOptions loss_opts;
    loss_opts.Register(&po);

    {
      bool binary = true;
      po.Register("binary", &binary, "Write output in binary mode");

      bool crossvalidate = false;
      po.Register("cross-validate", &crossvalidate,
                  "Perform cross-validation (don't back-propagate)");

      bool randomize = true;
      po.Register("randomize", &randomize,
                  "Perform the frame-level shuffling within the Cache::");

      std::string feature_transform;
      po.Register("feature-transform", &feature_transform,
                  "Feature transform in Nnet format");

      std::string objective_function = "xent";
      po.Register("objective-function", &objective_function,
                  "Objective function : xent|mse|multitask");

      int32 max_frames = 360000;
      po.Register("max-frames", &max_frames,
                  "Maximum number of frames an utterance can have (skipped if longer)");

      int32 length_tolerance = 5;
      po.Register("length-tolerance", &length_tolerance,
                  "Allowed length mismatch of features/targets/weights "
                  "(in frames, we truncate to the shortest)");

      std::string frame_weights;
      po.Register("frame-weights", &frame_weights,
                  "Per-frame weights, used to re-scale gradients.");

      std::string utt_weights;
      po.Register("utt-weights", &utt_weights,
                  "Per-utterance weights, used to re-scale frame-weights.");

      std::string use_gpu="yes";
      po.Register("use-gpu", &use_gpu,
                  "yes|no|optional, only has effect if compiled with CUDA");

      po.Read(argc, argv);

      if (po.NumArgs() != 3 + (crossvalidate ? 0 : 1)) {
        po.PrintUsage();
        exit(1);
      }

    }

    std::string
        feature_rspecifier = po.GetArg(1),
        targets_rspecifier = po.GetArg(2),
        model_filename = po.GetArg(3);

    // 非验证集合
    std::string target_model_filename;
    if (!crossvalidate) {
      target_model_filename = po.GetArg(4);
    }

    using namespace kaldi;
    using namespace kaldi::nnet1;
    typedef kaldi::int32 int32;

#if HAVE_CUDA == 1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    // 预处理 nnet 特征转换
    Nnet nnet_transf;
    if (feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }

    // nnet 神经网络模型
    Nnet nnet;
    nnet.Read(model_filename);
    nnet.SetTrainOptions(trn_opts);
    // false
    if (crossvalidate) {
      nnet_transf.SetDropoutRate(0.0);
      nnet.SetDropoutRate(0.0);
    }

    kaldi::int64 total_frames = 0;
    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessPosteriorReader targets_reader(targets_rspecifier);
    
    RandomAccessBaseFloatVectorReader weights_reader;
    if (frame_weights != "") {
      weights_reader.Open(frame_weights);
    }
    RandomAccessBaseFloatReader utt_weights_reader;
    if (utt_weights != "") {
      utt_weights_reader.Open(utt_weights);
    }

    // 随机化容器
    RandomizerMask randomizer_mask(rnd_opts);
    MatrixRandomizer feature_randomizer(rnd_opts);
    PosteriorRandomizer targets_randomizer(rnd_opts);
    VectorRandomizer weights_randomizer(rnd_opts);

    Xent xent(loss_opts);
    Mse mse(loss_opts);

    // 是否是综合损失函数(综合多个损失函数)
    MultiTaskLoss multitask(loss_opts);
    if (0 == objective_function.compare(0, 9, "multitask")) {
      // objective_function contains something like :
      // 'multitask,xent,2456,1.0,mse,440,0.001'
      //
      // the meaning is following:
      // 'multitask,<type1>,<dim1>,<weight1>,...,<typeN>,<dimN>,<weightN>'
      multitask.InitFromString(objective_function);
    }

    // 转换后特征、nnet_out网络输出、 
    CuMatrix<BaseFloat> feats_transf, nnet_out, obj_diff;
    Timer time;
    KALDI_LOG << (crossvalidate ? "CROSS-VALIDATION" : "TRAINING")
              << " STARTED";

    int32
        num_done = 0,
        num_no_tgt_mat = 0,
        num_other_error = 0;

    // main loop, 读取特征
    while (!feature_reader.Done()) {
#if HAVE_CUDA == 1
      // check that GPU computes accurately,
      CuDevice::Instantiate().CheckGpuHealth();
#endif
      // 填充随机化容器, 等待用来进行训练.
      // fill the randomizer,
      for ( ; !feature_reader.Done(); feature_reader.Next()) {
        if (feature_randomizer.IsFull()) {
          // break the loop without calling Next(),
          // we keep the 'utt' for next round,
          break;
        }
        
        std::string utt = feature_reader.Key();
        KALDI_VLOG(3) << "Reading " << utt;

        // check that we have targets,
        if (!targets_reader.HasKey(utt)) {
          KALDI_WARN << utt << ", missing targets";
          num_no_tgt_mat++;
          continue;
        }
        // check we have per-frame weights,
        if (frame_weights != "" && !weights_reader.HasKey(utt)) {
          KALDI_WARN << utt << ", missing per-frame weights";
          num_other_error++;
          continue;
        }
        // check we have per-utterance weights,
        if (utt_weights != "" && !utt_weights_reader.HasKey(utt)) {
          KALDI_WARN << utt << ", missing per-utterance weight";
          num_other_error++;
          continue;
        }


        // 获得feature - 对齐目标
        // get feature / target pair,
        Matrix<BaseFloat> mat = feature_reader.Value();
        Posterior targets = targets_reader.Value(utt);
        
        // get per-frame weights,
        Vector<BaseFloat> weights;
        if (frame_weights != "") {
          weights = weights_reader.Value(utt);
        } else {  // all per-frame weights are 1.0,
          weights.Resize(mat.NumRows());
          weights.Set(1.0);
        }

        // multiply with per-utterance weight,
        if (utt_weights != "") {
          BaseFloat w = utt_weights_reader.Value(utt);
          KALDI_ASSERT(w >= 0.0);
          if (w == 0.0) continue;  // remove sentence from training,
          weights.Scale(w);
        }

        // 如果utt帧过长, skip 否则会内存不足
        // skip too long utterances (or we run out of memory),
        if (mat.NumRows() > max_frames) {
          KALDI_WARN << "Utterance too long, skipping! " << utt
                     << " (length " << mat.NumRows() << ", max_frames "
                     << max_frames << ")";
          num_other_error++;
          continue;
        }

        // 修正某些匹配 有些缺点的utt, 或者直接skip。
        // correct small length mismatch or drop sentence,
        {
          // add lengths to vector,
          std::vector<int32> length;
          length.push_back(mat.NumRows());
          length.push_back(targets.size());
          length.push_back(weights.Dim());
          // find min, max,
          int32 min = *std::min_element(length.begin(), length.end());
          int32 max = *std::max_element(length.begin(), length.end());
          
          // fix or drop ?
          if (max - min < length_tolerance) {
            // we truncate to shortest,
            if (mat.NumRows() != min) mat.Resize(min, mat.NumCols(), kCopyData);
            if (targets.size() != min) targets.resize(min);
            if (weights.Dim() != min) weights.Resize(min, kCopyData);
          } else {
            KALDI_WARN << "Length mismatch! Targets " << targets.size()
                       << ", features " << mat.NumRows() << ", " << utt;
            num_other_error++;
            continue;
          }
        }
        
        // 进行feature transf 特征转换, 只需要使用nnete_transf 网络前向传播即可.
        // apply feature transform (if empty, input is copied),
        nnet_transf.Feedforward(CuMatrix<BaseFloat>(mat), &feats_transf);

        
        // remove frames with '0' weight from training,
        {
          // are there any frames to be removed? (frames with zero weight),
          BaseFloat weight_min = weights.Min();
          KALDI_ASSERT(weight_min >= 0.0);
          if (weight_min == 0.0) {
            // create vector with frame-indices to keep,
            std::vector<MatrixIndexT> keep_frames;
            for (int32 i = 0; i < weights.Dim(); i++) {
              if (weights(i) > 0.0) {
                keep_frames.push_back(i);
              }
            }

            // when all frames are removed, we skip the sentence,
            if (keep_frames.size() == 0) continue;

            // filter feature-frames,
            CuMatrix<BaseFloat> tmp_feats(keep_frames.size(), feats_transf.NumCols());
            tmp_feats.CopyRows(feats_transf, CuArray<MatrixIndexT>(keep_frames));
            tmp_feats.Swap(&feats_transf);

            // filter targets,
            Posterior tmp_targets;
            for (int32 i = 0; i < keep_frames.size(); i++) {
              tmp_targets.push_back(targets[keep_frames[i]]);
            }
            tmp_targets.swap(targets);

            // filter weights,
            Vector<BaseFloat> tmp_weights(keep_frames.size());
            for (int32 i = 0; i < keep_frames.size(); i++) {
              tmp_weights(i) = weights(keep_frames[i]);
            }
            tmp_weights.Swap(&weights);
          }
        }

        // 将数据加入到 randomizer 容器中, 随机化进行训练
        // pass data to randomizers,
        KALDI_ASSERT(feats_transf.NumRows() == targets.size());
        feature_randomizer.AddData(feats_transf);
        targets_randomizer.AddData(targets);
        weights_randomizer.AddData(weights);
        num_done++;

        // report the speed,
        if (num_done % 5000 == 0) {
          double time_now = time.Elapsed();
          KALDI_VLOG(1) << "After " << num_done << " utterances: "
                        << "time elapsed = " << time_now / 60 << " min; "
                        << "processed " << total_frames / time_now << " frames per sec.";
        }
      }

      // 随机化数据集的排列
      // randomize,
      if (!crossvalidate && randomize) {
        const std::vector<int32>& mask =
            randomizer_mask.Generate(feature_randomizer.NumFrames());
        feature_randomizer.Randomize(mask);
        targets_randomizer.Randomize(mask);
        weights_randomizer.Randomize(mask);
      }



      // ==========================================
      // 使用随机化数据进行 mini-batches 训练
      // train with data from randomizers (using mini-batches),
      for ( ; !feature_randomizer.Done(); feature_randomizer.Next(),
                targets_randomizer.Next(),
                weights_randomizer.Next()) {

        // 取一个 mini batch 数据量
        // get block of feature/target pairs,
        const CuMatrixBase<BaseFloat>& nnet_in = feature_randomizer.Value();
        const Posterior& nnet_tgt = targets_randomizer.Value();
        const Vector<BaseFloat>& frm_weights = weights_randomizer.Value();

        // forward pass,前向传播计算
        nnet.Propagate(nnet_in, &nnet_out);
        
        void Nnet::Propagate(const CuMatrixBase<BaseFloat> &in,
                             CuMatrix<BaseFloat> *out) {
          // In case of empty network copy input to output,
          if (NumComponents() == 0) {
            (*out) = in;  // copy,
            return;
          }
          
          // 前向传播中各层神经元节点输出值.
          if (propagate_buf_.size() != NumComponents()+1) {
            propagate_buf_.resize(NumComponents()+1);
          }
          
          // Copy input to first buffer,
          propagate_buf_[0] = in;
          // 经过各层神经网络向后传播, 直到propagate_buf_[NumComponents()]  --- out
          // [0 - Cnt]
          // 0 是输入层, 1 是第一层的输出节点, Cnt 是最后一层的输出节点
          // Propagate through all the components,
          for (int32 i = 0; i < static_cast<int32>(components_.size()); i++) {
            components_[i]->Propagate(propagate_buf_[i], &propagate_buf_[i+1]);
          }
          // Copy the output from the last buffer,
          (*out) = propagate_buf_[NumComponents()];
        }


        // 估计损失--> obj_diff
        // obj_diff 的行是所有frame, 列是one-hot形式的误差(每种可能的pdf都会产生误差)
        // 一个frame可能生成所有可能的pdf，one-hot内部不同的pdf概率不同.
        // evaluate objective function we've chosen,
        if (objective_function == "xent") {
          // gradients re-scaled by weights in Eval,
          //       帧权重      nnet输出    nnet真实值  nnet_out - nnet_tgt?
          xent.Eval(frm_weights, nnet_out, nnet_tgt, &obj_diff);


          // 计算xent损失
          void Xent::Eval(const VectorBase<BaseFloat> &frame_weights,
                          const CuMatrixBase<BaseFloat> &net_out,
                          const Posterior &post,
                          CuMatrix<BaseFloat> *diff) {
            int32
                num_frames = net_out.NumRows(),
                num_pdf = net_out.NumCols();
            KALDI_ASSERT(num_frames == post.size());
            // convert posterior to matrix,
            PosteriorToMatrix(post, num_pdf, &tgt_mat_);
            // call the other eval function,
            Eval(frame_weights, net_out, tgt_mat_, diff);
          }
          

          // post_dim 的维度是 num_pdf, 将frame的真实结果 映射为Matrix
          void PosteriorToMatrix(const Posterior &post,
                                 const int32 post_dim, Matrix<Real> *mat) {
            // Make a host-matrix,
            int32 num_rows = post.size();
            mat->Resize(num_rows, post_dim, kSetZero);  // zero-filled
            
            // Fill from Posterior,
            for (int32 t = 0; t < post.size(); t++) {
              for (int32 i = 0; i < post[t].size(); i++) {
                int32 col = post[t][i].first;
                if (col >= post_dim) {
                  KALDI_ERR << "Out-of-bound Posterior element with index " << col
                            << ", higher than number of columns " << post_dim;
                }
                (*mat)(t, col) = post[t][i].second;
              }
            }
          }



          // in:
          // frame_weights 帧权重
          // net_out   网络输出
          // targets   结果标签
          // out:
          // diff      out-targets (基于softmax的交叉熵的导数, 计算残差用)
          // use:
          // 根据帧权重, 通过net_out, targets 计算误差, 计算熵, 假设交叉熵, diff 实际上是交叉熵的导数, 
          void Xent::Eval(const VectorBase<BaseFloat> &frame_weights,
                          const CuMatrixBase<BaseFloat> &net_out,
                          const CuMatrixBase<BaseFloat> &targets,
                          CuMatrix<BaseFloat> *diff) {
            // frame_weight 每帧权重 == 1.0   [FrameCnt x 1]
            // net_out   nnet 输出结果 [FrameCnt x Num_pdf]
            // targets   nnet 真实标签 [FrameCnt x Num_pdf]
            // check inputs,
            KALDI_ASSERT(net_out.NumCols() == targets.NumCols());
            KALDI_ASSERT(net_out.NumRows() == targets.NumRows());
            KALDI_ASSERT(net_out.NumRows() == frame_weights.Dim());

            KALDI_ASSERT(KALDI_ISFINITE(frame_weights.Sum()));
            KALDI_ASSERT(KALDI_ISFINITE(net_out.Sum()));
            KALDI_ASSERT(KALDI_ISFINITE(targets.Sum()));

            // 每个pdf的主要统计量 (nnet 输出结果[Num_pdf x 1])
            // buffer initialization,
            int32 num_classes = targets.NumCols();
            if (frames_.Dim() == 0) {
              frames_.Resize(num_classes);
              xentropy_.Resize(num_classes);
              entropy_.Resize(num_classes);
              correct_.Resize(num_classes);
            }

            // get frame_weights to GPU,
            frame_weights_ = frame_weights;

            // 有一些帧的 输出结果总和 = 0, 多语言训练时会出现.这里可以不考虑
            {
              // There may be frames for which the sum of targets is zero.
              // This happens in multi-lingual training when the frame
              // has target class in the softmax of another language.
              // We 'switch-off' such frames by masking the 'frame_weights_',
              target_sum_.Resize(targets.NumRows());
              target_sum_.AddColSumMat(1.0, targets, 0.0);
              frame_weights_.MulElements(target_sum_);
            }

            // 计算 上一层神经元激活函数的导数
            // compute derivative wrt. activations of last layer of neurons,
            // （Y-T）
            *diff = net_out;
            diff->AddMat(-1.0, targets);
            diff->MulRowsVec(frame_weights_);  // weighting,

            // 计算pdf的帧总数
            // count frames per class,
            frames_aux_ = targets;
            frames_aux_.MulRowsVec(frame_weights_);
            frames_.AddRowSumMat(1.0, CuMatrix<double>(frames_aux_));

            // 估计帧 级别的分类结果?
            // evaluate the frame-level classification,
            // 找到每行row 的最大概率的pdf-id 
            net_out.FindRowMaxId(&max_id_out_);  // find max in nn-output
            targets.FindRowMaxId(&max_id_tgt_);  // find max in targets

            // 统计所有pdf-id 的正确frame数量(weighted加权重的)  ===> correct_
            CountCorrectFramesWeighted(max_id_out_, max_id_tgt_,
                                       frame_weights_, &correct_);


            // 计算交叉熵
            // calculate cross_entropy (in GPU),
            xentropy_aux_ = net_out;  // y
            xentropy_aux_.Add(1e-20);  // avoid log(0)
            xentropy_aux_.ApplyLog();  // log(y)
            xentropy_aux_.MulElements(targets);  // t*log(y)
            xentropy_aux_.MulRowsVec(frame_weights_);  // w*t*log(y)
            xentropy_.AddRowSumMat(-1.0, CuMatrix<double>(xentropy_aux_));

            // 计算熵
            // caluculate entropy (in GPU),
            entropy_aux_ = targets;  // t
            entropy_aux_.Add(1e-20);  // avoid log(0)
            entropy_aux_.ApplyLog();  // log(t)
            entropy_aux_.MulElements(targets);  // t*log(t)
            entropy_aux_.MulRowsVec(frame_weights_);  // w*t*log(t)
            entropy_.AddRowSumMat(-1.0, CuMatrix<double>(entropy_aux_));

            // progressive loss reporting
            // 过程中损失报告
            if (opts_.loss_report_frames > 0) {
              // 帧权重总损失
              frames_progress_ += frame_weights_.Sum();
              // 交叉熵总体值
              xentropy_progress_ += -xentropy_aux_.Sum();
              // 熵总体值
              entropy_progress_ += -entropy_aux_.Sum();

              KALDI_ASSERT(KALDI_ISFINITE(xentropy_progress_));
              KALDI_ASSERT(KALDI_ISFINITE(entropy_progress_));

              if (frames_progress_ > opts_.loss_report_frames) {
                // loss value,  交叉熵与 真实熵误差率
                double progress_value =
                    (xentropy_progress_ - entropy_progress_) / frames_progress_;

                // time-related info (fps is weighted),
                double time_now = timer_.Elapsed();
                double fps = frames_progress_ / (time_now - elapsed_seconds_);
                double elapsed_hours = time_now / 3600;
                elapsed_seconds_ = time_now; // store,

                // print,
                KALDI_LOG << "ProgressLoss[last "
                    << static_cast<int>(frames_progress_/100/3600) << "h of "
                    << static_cast<int>(frames_.Sum()/100/3600) << "h]: "
                          << progress_value << " (Xent)"
                          << ", fps=" << fps
                          << std::setprecision(3)
                          << ", elapsed " << elapsed_hours << "h";
                // store,
                loss_vec_.push_back(progress_value);
                // reset,
                frames_progress_ = 0;
                xentropy_progress_ = 0.0;
                entropy_progress_ = 0.0;
              }
            }
          }
         
          
        }

        // 后向传播 残差分配,进行参数更新
        if (!crossvalidate) {
          // back-propagate, and do the update,
          nnet.Backpropagate(obj_diff, NULL);

          /**
           * Error back-propagation through the network,
           * (from last component to first).
           */
          // out_diff 是输出节点的误差.
          // in_diff 是反向传导回到输入节点.
          void Nnet::Backpropagate(const CuMatrixBase<BaseFloat> &out_diff,
                                   CuMatrix<BaseFloat> *in_diff) {

            // std::vector<CuMatrix<BaseFloat> > propagate_buf_;  
            // std::vector<CuMatrix<BaseFloat> > backpropagate_buf_;
            
            // Copy the derivative in case of empty network,
            if (NumComponents() == 0) {
              (*in_diff) = out_diff;  // copy,
              return;
            }
            // We need C+1 buffers,
            KALDI_ASSERT(static_cast<int32>(propagate_buf_.size()) == NumComponents()+1);
            
            if (backpropagate_buf_.size() != NumComponents()+1) {
              backpropagate_buf_.resize(NumComponents()+1);
            }

            
            // Copy 'out_diff' to last buffer,
            backpropagate_buf_[NumComponents()] = out_diff;

            // 逐层 进行反向传播.
            // Loop from last Component to the first,
            for (int32 i = NumComponents()-1; i >= 0; i--) {
              // 反向传播 神经元 残差!!
              // Backpropagate through 'Component',
              components_[i]->Backpropagate(propagate_buf_[i],
                                            propagate_buf_[i+1],
                                            backpropagate_buf_[i+1],
                                            &backpropagate_buf_[i]);


              void Softmax::BackpropagateFnc(const CuMatrixBase<BaseFloat> &in,
                                    const CuMatrixBase<BaseFloat> &out,
                                    const CuMatrixBase<BaseFloat> &out_diff,
                                    CuMatrixBase<BaseFloat> *in_diff) {
                // out_diff虽然是简单的(out - target),但是经过推导, 实际上out_diff是
                // 上一层的的残差直接是上一层神经元的输出与输入相等 所以直接是残差. 
                in_diff->CopyFromMat(out_diff);
              }

              void AffineTranform::BackpropagateFnc(const CuMatrixBase<BaseFloat> &in,
                                    const CuMatrixBase<BaseFloat> &out,
                                    const CuMatrixBase<BaseFloat> &out_diff,
                                    CuMatrixBase<BaseFloat> *in_diff) {
                // multiply error derivative by weights
                in_diff->AddMatMat(1.0, out_diff, kNoTrans, linearity_, kNoTrans, 0.0);
              }

              // 利用本层神经元前向输出 与 本层神经元的反向残差 更新参数 (两者在数组中index 差1)
              // Update 'Component' (if applicable),
              if (components_[i]->IsUpdatable()) {
                UpdatableComponent* uc =
                    dynamic_cast<UpdatableComponent*>(components_[i]);
                uc->Update(propagate_buf_[i], backpropagate_buf_[i+1]);

                // 按照公式进行逐层的 W b等参数更新.
                void AffineTranform::Update(const CuMatrixBase<BaseFloat> &input,
                                            const CuMatrixBase<BaseFloat> &diff) {

                  // W b 的学习率
                  // we use following hyperparameters from the option class
                  const BaseFloat lr = opts_.learn_rate * learn_rate_coef_;
                  const BaseFloat lr_bias = opts_.learn_rate * bias_learn_rate_coef_;

                  // 动量 与 惩罚项
                  const BaseFloat mmt = opts_.momentum;
                  const BaseFloat l2 = opts_.l2_penalty;
                  const BaseFloat l1 = opts_.l1_penalty;
                  
                  // we will also need the number of frames in the mini-batch
                  const int32 num_frames = input.NumRows();
                  // 残差对W的梯度
                  // compute gradient (incl. momentum)
                  linearity_corr_.AddMatMat(1.0, diff, kTrans, input, kNoTrans, mmt);
                  // 残差对b的梯度
                  bias_corr_.AddRowSumMat(1.0, diff, mmt);
                  
                  // l2 regularization
                  if (l2 != 0.0) {
                    linearity_.AddMat(-lr*l2*num_frames, linearity_);
                  }
                  // l1 regularization
                  if (l1 != 0.0) {
                    cu::RegularizeL1(&linearity_, &linearity_corr_, lr*l1*num_frames, lr);
                  }

                  // W = W + 学习率*梯度
                  // update
                  linearity_.AddMat(-lr, linearity_corr_);
                  bias_.AddVec(-lr_bias, bias_corr_);
                  
                  // max-norm
                  if (max_norm_ > 0.0) {
                    CuMatrix<BaseFloat> lin_sqr(linearity_);
                    lin_sqr.MulElements(linearity_);
                    CuVector<BaseFloat> l2(OutputDim());
                    l2.AddColSumMat(1.0, lin_sqr, 0.0);
                    l2.ApplyPow(0.5);  // we have per-neuron L2 norms,
                    CuVector<BaseFloat> scl(l2);
                    scl.Scale(1.0/max_norm_);
                    scl.ApplyFloor(1.0);
                    scl.InvertElements();
                    linearity_.MulRowsVec(scl);  // shink to sphere!
                  }
                }
              }
            }

            // Export the derivative (if applicable),
            if (NULL != in_diff) {
              (*in_diff) = backpropagate_buf_[0];
            }
          }  // end Backpropagate

          
        }

        // 1st mini-batch : show what happens in network,
        if (total_frames == 0) {
          KALDI_VLOG(1) << "### After " << total_frames << " frames,";
          KALDI_VLOG(1) << nnet.InfoPropagate();
          if (!crossvalidate) {
            KALDI_VLOG(1) << nnet.InfoBackPropagate();
            KALDI_VLOG(1) << nnet.InfoGradient();
          }
        }
        
        // VERBOSE LOG  冗余日志?? 监视NN训练过程.
        // monitor the NN training (--verbose=2),
        if (GetVerboseLevel() >= 2) {
          static int32 counter = 0;
          counter += nnet_in.NumRows();
          // print every 25k frames,
          if (counter >= 25000) {
            KALDI_VLOG(2) << "### After " << total_frames << " frames,";
            KALDI_VLOG(2) << nnet.InfoPropagate();
            if (!crossvalidate) {
              KALDI_VLOG(2) << nnet.InfoBackPropagate();
              KALDI_VLOG(2) << nnet.InfoGradient();
            }
            counter = 0;
          }
        }

        
        total_frames += nnet_in.NumRows();
      }
    }  // main loop,

    // 显示nnetwork 训练结果
    // after last mini-batch : show what happens in network,
    KALDI_VLOG(1) << "### After " << total_frames << " frames,";
    KALDI_VLOG(1) << nnet.InfoPropagate();
    if (!crossvalidate) {
      KALDI_VLOG(1) << nnet.InfoBackPropagate();
      KALDI_VLOG(1) << nnet.InfoGradient();
    }

    // 将nnet模型 写入nnet模型文件
    if (!crossvalidate) {
      nnet.Write(target_model_filename, binary);
    }

    KALDI_LOG << "Done " << num_done << " files, "
              << num_no_tgt_mat << " with no tgt_mats, "
              << num_other_error << " with other errors. "
              << "[" << (crossvalidate ? "CROSS-VALIDATION" : "TRAINING")
              << ", " << (randomize ? "RANDOMIZED" : "NOT-RANDOMIZED")
              << ", " << time.Elapsed() / 60 << " min, processing "
              << total_frames / time.Elapsed() << " frames per sec.]";

    // 交叉熵的log信息
    if (objective_function == "xent") {
      KALDI_LOG << xent.ReportPerClass();
      KALDI_LOG << xent.Report();
    }

#if HAVE_CUDA == 1
    CuDevice::Instantiate().PrintProfile();
#endif

    return 0;
  
  }


  
  

  




  
  
  //   交叉验证 ============= 通过交叉验证集合的loss 判断nnet性能.
  //   # cross-validation,
  //   log=$dir/log/iter${iter}.cv.log; hostname>$log
  //   $train_tool --cross-validate=true --randomize=false --verbose=$verbose $train_tool_opts \
  //     ${feature_transform:+ --feature-transform=$feature_transform} \
  //     "$feats_cv" "$labels_cv" $mlp_next \

  //   loss_new=$(cat $dir/log/iter${iter}.cv.log | grep "AvgLoss:" | tail -n 1 | awk '{ print $4; }')
  //   echo -n "CROSSVAL AVG.LOSS $(printf "%.4f" $loss_new), "


  
  //   性能满足 则停止训练迭代
  //   # accept or reject?
  //   loss_prev=$loss
  //   误差有所下降 accept
  //                             keep_lr_iters = 0, min_iters=0;
  //   if [ 损失下降 or 迭代次数<keep_lr_iters or 迭代次数<最小迭代次数]; 说明此时还需要继续训练.
    //   if [ 1 == $(awk "BEGIN{print($loss_new < $loss ? 1:0);}") -o $iter -le $keep_lr_iters -o $iter -le $min_iters ]; then
  //     # accepting: the loss was better, or we had fixed learn-rate, or we had fixed epoch-number,
  //     认为当前的nnet结构更好 将其设置为mlp_best.
  //     loss=$loss_new
  //     mlp_best=$dir/nnet/${mlp_base}_iter${iter}_learnrate${learn_rate}_tr$(printf "%.4f" $tr_loss)_cv$(printf "%.4f" $loss_new)
  //     如果是如下两种情况, 则使用如下的名字. 这里不会进入.
  //     [ $iter -le $min_iters ] && mlp_best=${mlp_best}_min-iters-$min_iters
  //     [ $iter -le $keep_lr_iters ] && mlp_best=${mlp_best}_keep-lr-iters-$keep_lr_iters
  //     mv $mlp_next $mlp_best
  //     echo "nnet accepted ($(basename $mlp_best))"
  //     echo $mlp_best > $dir/.mlp_best
  //   否则此时 不需要再进行训练 把当前的nnet结果设置为拒绝的mlp
  //   else
  //     # rejecting,
  //     mlp_reject=$dir/nnet/${mlp_base}_iter${iter}_learnrate${learn_rate}_tr$(printf "%.4f" $tr_loss)_cv$(printf "%.4f" $loss_new)_rejected
  //     mv $mlp_next $mlp_reject
  //     echo "nnet rejected ($(basename $mlp_reject))"
  //   fi


  //   创建.done file， 当前迭代完成
  //   # create .done file, the iteration (epoch) is completed,
  //   touch $dir/.done_iter$iter

  //   按照原始学习率 继续迭代学习?
  //   # continue with original learn-rate,
  //   [ $iter -le $keep_lr_iters ] && continue

  //   停止准则, 损失提升率 (loss-prev - loss) / loss_prev
  //   # stopping criterion,
  //   rel_impr=$(awk "BEGIN{print(($loss_prev-$loss)/$loss_prev);}")
  //   if [ 减半学习率 = 1 &&　提升率较小]
  //   if [ 1 == $halving -a 1 == $(awk "BEGIN{print($rel_impr < $end_halving_impr ? 1:0);}") ]; then
  //     if [ $iter -le $min_iters ]; then
  //       建议停止, 但是应该要至少继续完成了　最少迭代次数.
  //       echo we were supposed to finish, but we continue as min_iters : $min_iters
  //       continue
  //     fi
  // 　　提升率较小.
  //     echo finished, too small rel. improvement $rel_impr
  //     break
  //   fi

  

  //   if 损失提升率较小,　则设置减半变量.
  //   # start learning-rate fade-out when improvement is low,
  //   if [ 1 == $(awk "BEGIN{print($rel_impr < $start_halving_impr ? 1:0);}") ]; then
  //     halving=1
  //     echo $halving >$dir/.halving
  //   fi

  
  //   判断减半变量, 降低学习率
  //   # reduce the learning-rate,
  //   if [ 1 == $halving ]; then
  //     learn_rate=$(awk "BEGIN{print($learn_rate*$halving_factor)}")
  //     echo $learn_rate >$dir/.learn_rate
  //   fi
  // done


  // 选择最佳nnetwork
  // # select the best network,
  // 如果最佳网络 不同于初始化的网络 则认为训练有效
  // if [ $mlp_best != $mlp_init ]; then
  //   mlp_final=${mlp_best}_final_
  //   ( cd $dir/nnet; ln -s $(basename $mlp_best) $(basename $mlp_final); )
  //   ( cd $dir; ln -s nnet/$(basename $mlp_final) final.nnet; )
  //   echo "$0: Succeeded training the Neural Network : '$dir/final.nnet'"
  // else
  //   echo "$0: Error training neural network..."
  //   exit 1
  // fi
}


