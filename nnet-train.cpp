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

  // 构建一个 归一化的nnet 里面包含了两个Component  1 shift设置均值为0， 2 拉伸变换scale --归一化
  int cmv_to_nnet(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet1;
    typedef kaldi::int32 int32;

    const char *usage =
      "Convert cmvn-stats into <AddShift> and <Rescale> components.\n"
      "Usage:  cmvn-to-nnet [options] <transf-in> <nnet-out>\n"
      "e.g.:\n"
      " cmvn-to-nnet --binary=false transf.mat nnet.mdl\n";


    bool binary_write = false;
    float std_dev = 1.0;
    float var_floor = 1e-10;
    float learn_rate_coef = 0.0;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("std-dev", &std_dev, "Standard deviation of the output.");
    po.Register("var-floor", &var_floor,
        "Floor the variance, so the factors in <Rescale> are bounded.");
    po.Register("learn-rate-coef", &learn_rate_coef,
        "Initialize learning-rate coefficient to a value.");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string cmvn_stats_rxfilename = po.GetArg(1),
        model_out_filename = po.GetArg(2);

    // read the matrix,
    Matrix<double> cmvn_stats;
    {
      bool binary_read;
      Input ki(cmvn_stats_rxfilename, &binary_read);
      cmvn_stats.Read(ki.Stream(), binary_read);
    }
    KALDI_ASSERT(cmvn_stats.NumRows() == 2);
    KALDI_ASSERT(cmvn_stats.NumCols() > 1);

    int32 num_dims = cmvn_stats.NumCols() - 1;
    double frame_count = cmvn_stats(0, cmvn_stats.NumCols() - 1);

    // buffers for shift and scale
    Vector<BaseFloat> shift(num_dims);
    Vector<BaseFloat> scale(num_dims);

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
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


  

}










// if [ ! -z $ivector ]; then
//   echo
//   echo "# ADDING IVECTOR FEATURES"
//   # The iVectors are concatenated 'as they are' directly to the input of the neural network,
//   # To do this, we paste the features, and use <ParallelComponent> where the 1st component
//   # contains the transform and 2nd network contains <Copy> component.

//   echo "# getting dims,"
//   dim_raw=$(feat-to-dim "$feats_tr_10k" -)
//   dim_raw_and_ivec=$(feat-to-dim "$feats_tr_10k $ivector_append_tool ark:- '$ivector' ark:- |" -)
//   dim_ivec=$((dim_raw_and_ivec - dim_raw))
//   echo "# dims, feats-raw $dim_raw, ivectors $dim_ivec,"

//   # Should we do something with 'feature_transform'?
//   if [ ! -z $ivector_dim ]; then
//     # No, the 'ivector_dim' comes from dir with 'feature_transform' with iVec forwarding,
//     echo "# assuming we got '$feature_transform' with ivector forwarding,"
//     [ $ivector_dim != $dim_ivec ] && \
//     echo -n "Error, i-vector dimensionality mismatch!" && \
//     echo " (expected $ivector_dim, got $dim_ivec in $ivector)" && exit 1
//   else
//     # Yes, adjust the transform to do ``iVec forwarding'',
//     feature_transform_old=$feature_transform
//     feature_transform=${feature_transform%.nnet}_ivec_copy.nnet
//     echo "# setting up ivector forwarding into '$feature_transform',"
//     dim_transformed=$(feat-to-dim "$feats_tr_10k nnet-forward $feature_transform_old ark:- ark:- |" -)
//     nnet-initialize --print-args=false <(echo "<Copy> <InputDim> $dim_ivec <OutputDim> $dim_ivec <BuildVector> 1:$dim_ivec </BuildVector>") $dir/tr_ivec_copy.nnet
//     nnet-initialize --print-args=false <(echo "<ParallelComponent> <InputDim> $((dim_raw+dim_ivec)) <OutputDim> $((dim_transformed+dim_ivec)) \
//                                                <NestedNnetFilename> $feature_transform_old $dir/tr_ivec_copy.nnet </NestedNnetFilename>") $feature_transform
//   fi
//   echo $dim_ivec >$dir/ivector_dim # mark down the iVec dim!
//   echo $ivector_append_tool >$dir/ivector_append_tool

//   # pasting the iVecs to the feaures,
//   echo "# + ivector input '$ivector'"
//   feats_tr="$feats_tr $ivector_append_tool ark:- '$ivector' ark:- |"
//   feats_cv="$feats_cv $ivector_append_tool ark:- '$ivector' ark:- |"
// fi

// ###### Show the final 'feature_transform' in the log,
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
// echo "# NN-INITIALIZATION"
// if [ ! -z $nnet_init ]; then
//   echo "# using pre-initialized network '$nnet_init'"
// elif [ ! -z $nnet_proto ]; then
//   echo "# initializing NN from prototype '$nnet_proto'";
//   nnet_init=$dir/nnet.init; log=$dir/log/nnet_initialize.log
//   nnet-initialize --seed=$seed $nnet_proto $nnet_init
// else
//   echo "# getting input/output dims :"
//   # input-dim,
//   get_dim_from=$feature_transform
//   [ ! -z "$dbn" ] && get_dim_from="nnet-concat $feature_transform '$dbn' -|"
//   num_fea=$(feat-to-dim "$feats_tr_10k nnet-forward \"$get_dim_from\" ark:- ark:- |" -)

//   # output-dim,
//   [ -z $num_tgt ] && \
//     num_tgt=$(hmm-info --print-args=false $alidir/final.mdl | grep pdfs | awk '{ print $NF }')

//   # make network prototype,
//   nnet_proto=$dir/nnet.proto
//   echo "# genrating network prototype $nnet_proto"
//   case "$network_type" in
//     dnn)
//       utils/nnet/make_nnet_proto.py $proto_opts \
//         ${bn_dim:+ --bottleneck-dim=$bn_dim} \
//         $num_fea $num_tgt $hid_layers $hid_dim >$nnet_proto
//       ;;
//     cnn1d)
//       delta_order=$([ -z $delta_opts ] && echo "0" || { echo $delta_opts | tr ' ' '\n' | grep "delta[-_]order" | sed 's:^.*=::'; })
//       echo "Debug : $delta_opts, delta_order $delta_order"
//       utils/nnet/make_cnn_proto.py $cnn_proto_opts \
//         --splice=$splice --delta-order=$delta_order --dir=$dir \
//         $num_fea >$nnet_proto
//       cnn_fea=$(cat $nnet_proto | grep -v '^$' | tail -n1 | awk '{ print $5; }')
//       utils/nnet/make_nnet_proto.py $proto_opts \
//         --no-smaller-input-weights \
//         ${bn_dim:+ --bottleneck-dim=$bn_dim} \
//         "$cnn_fea" $num_tgt $hid_layers $hid_dim >>$nnet_proto
//       ;;
//     cnn2d)
//       delta_order=$([ -z $delta_opts ] && echo "0" || { echo $delta_opts | tr ' ' '\n' | grep "delta[-_]order" | sed 's:^.*=::'; })
//       echo "Debug : $delta_opts, delta_order $delta_order"
//       utils/nnet/make_cnn2d_proto.py $cnn_proto_opts \
//         --splice=$splice --delta-order=$delta_order --dir=$dir \
//         $num_fea >$nnet_proto
//       cnn_fea=$(cat $nnet_proto | grep -v '^$' | tail -n1 | awk '{ print $5; }')
//       utils/nnet/make_nnet_proto.py $proto_opts \
//         --no-smaller-input-weights \
//         ${bn_dim:+ --bottleneck-dim=$bn_dim} \
//         "$cnn_fea" $num_tgt $hid_layers $hid_dim >>$nnet_proto
//       ;;
//     lstm)
//       utils/nnet/make_lstm_proto.py $proto_opts \
//         $num_fea $num_tgt >$nnet_proto
//       ;;
//     blstm)
//       utils/nnet/make_blstm_proto.py $proto_opts \
//         $num_fea $num_tgt >$nnet_proto
//       ;;
//     *) echo "Unknown : --network-type $network_type" && exit 1;
//   esac

//   # initialize,
//   nnet_init=$dir/nnet.init
//   echo "# initializing the NN '$nnet_proto' -> '$nnet_init'"
//   nnet-initialize --seed=$seed $nnet_proto $nnet_init

//   # optionally prepend dbn to the initialization,
//   if [ ! -z "$dbn" ]; then
//     nnet_init_old=$nnet_init; nnet_init=$dir/nnet_dbn_dnn.init
//     nnet-concat "$dbn" $nnet_init_old $nnet_init
//   fi
// fi


// ###### TRAIN ######
// echo
// echo "# RUNNING THE NN-TRAINING SCHEDULER"
// steps/nnet/train_scheduler.sh \
//   ${scheduler_opts} \
//   ${train_tool:+ --train-tool "$train_tool"} \
//   ${train_tool_opts:+ --train-tool-opts "$train_tool_opts"} \
//   ${feature_transform:+ --feature-transform $feature_transform} \
//   ${split_feats:+ --split-feats $split_feats} \
//   --learn-rate $learn_rate \
//   ${frame_weights:+ --frame-weights "$frame_weights"} \
//   ${utt_weights:+ --utt-weights "$utt_weights"} \
//   ${config:+ --config $config} \
//   $nnet_init "$feats_tr" "$feats_cv" "$labels_tr" "$labels_cv" $dir

// echo "$0: Successfuly finished. '$dir'"

// sleep 3
// exit 0
