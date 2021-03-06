// * overall
// steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
// 2000 10000 data/mfcc/train data/lang exp/mono_ali exp/tri1 || exit 1;

// # 决策树叶节点总数 2000 绑定状态数
// numleaves = $1

// # mfcc/train/
// data=$3

// # lang/ 拓扑结构、发音词典、其他发音、所有词words.txt
// lang=$4

// # mono_ali 已对齐的单音素训练结果.
// alidir=$5
// # tri1 三音素结果 输出目录
// dir=$6


// acc-tree-stats --ci-phones=$ciphonelist $alidir/final.mdl "$feats" "ark:gunzip -c $alidir/ali.JOB.gz|" $dir/JOB.treeacc 
// #                                       单音素训练结果     特征          对齐结果                      ===> 统计量
// sum-tree-stats $dir/treeacc $dir/*.treeacc 
// #               统计量        综合统计量
// cluster-phones $context_opts $dir/treeacc $lang/phones/sets.int  $dir/questions.int 
// #               null (3, 1)   决策数统计量   音素变体            输出问题集合(音素分割树)
// compile-questions $context_opts $lang/topo $dir/questions.int $dir/questions.qst  ？？？？？
// #                               topo结构   音素分割树          输出问题???x
// build-tree --max-leaves=$numleaves  $dir/treeacc $lang/phones/roots.int $dir/questions.qst $lang/topo $dir/tree
// # 最大节点数                     统计量         roots.int            qst   topo     ==> tree


/** acc-tree-stats
  # 统计 训练决策数需要的统计量
  # in HMM-GMM模型   特征  对齐的状态序列   
  # out 计算统计量   map<EventType, GaussClusterable> 所有状态的 统计量
  # Context width 和 central position用来识别上下文环境
  # 转移模型 用来获得pdf-id 和 音素.
   @brief Accumulate tree statistics for decision tree training. The
program reads in a feature archive, and the corresponding alignments,
and generates the sufficient statistics for the decision tree
creation. Context width and central phone position are used to
identify the contexts.Transition model is used as an input to identify
the PDF's and the phones.  */

int acc_tree_stats(int argc, char *argv[]) {
    const char *usage =
        "Accumulate statistics for phonetic-context tree building.\n"
        "Usage:  acc-tree-stats [options] <model-in> <features-rspecifier> <alignments-rspecifier> <tree-accs-out>\n"
        "e.g.: \n"

    // # 输入 HMM-GMM模型   特征  对齐的状态序列   
    // # 输出 计算统计量
    " acc-tree-stats 1.mdl scp:train.scp ark:1.ali 1.tacc\n";

    bool binary = true;
    // # 计算 决策树需要的 统计信息 选项. 
    // # Context-width = 3 central postion = 1, 标准三音素窗。
    // 并且传入进来了 ci-phones 用来将phone 映射到 phone-id??
    AccumulateTreeStatsOptions opts;

    std::string
    model_filename = po.GetArg(1),
    feature_rspecifier = po.GetArg(2),
    // # 对齐状态序列  // 
    alignment_rspecifier = po.GetArg(3),
    accs_out_wxfilename = po.GetOptArg(4);

    // # 统计 决策树统计信息
    // 主要包含了 cetral-phone, context-width, ci-phone-map.
    AccumulateTreeStatsInfo acc_tree_stats_info(opts);

    TransitionModel trans_model;
    {
      bool binary;
      Input ki(model_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
    }

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessInt32VectorReader alignment_reader(alignment_rspecifier);

    // # 绑定树 用的统计量 
    // # EventType -- <三音素+状态>  某个确定状态
    // # GaussClusterable  该状态对应的特征向量个数、特征向量累加、特征向量平方和累加.
    *std::map<EventType, GaussClusterable*> tree_stats;*
                                                           
    int num_done = 0, num_no_alignment = 0, num_other_error = 0;

    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string key = feature_reader.Key();
      if (!alignment_reader.HasKey(key)) {
        num_no_alignment++;
      } else {
        // # 某段语音 的特征                  [MFCC[39] X TIME_LEN]
        const Matrix<BaseFloat> &mat = feature_reader.Value();
        // # vector<trans-ids>  对齐的转移id  [Trans-id X TIME_LEN]
        const std::vector<int32> &alignment = alignment_reader.Value(key);

        // # 根据统计模型、统计用参数、对齐的trans-ids、特征mat, 
        // # 计算统计信息  --- > tree_stats
        AccumulateTreeStats(trans_model,
                            acc_tree_stats_info,
                            alignment,
                            mat,
                            &tree_stats);
        num_done++;
        if (num_done % 1000 == 0)
          KALDI_LOG << "Processed " << num_done << " utterances.";
      }
    }

    



    BuildTreeStatsType stats;  // vectorized form.

    // # 将 <EventType, GaussClusterable> 保存到文件中.
    for (std::map<EventType, GaussClusterable*>::const_iterator iter = tree_stats.begin();
         iter != tree_stats.end();
         ++iter) {
      stats.push_back(std::make_pair(iter->first, iter->second));
    }
    tree_stats.clear();
    
    {
      Output ko(accs_out_wxfilename, binary);
      WriteBuildTreeStats(ko.Stream(), binary, stats);
    }

    DeleteBuildTreeStats(&stats);
}


/** AccumulateTreeStats 计算决策树构建需要的统计量

# # in:
#    trans_model    转移模型
#    计算统计量用的参数 -- 没什么配置
#    alignment      一句utt的对齐trans-id
#    features       一句utt的特征向量矩阵
# # out:
#    stats          输出统计量 (根据决策树决定的 所有pdf-id(多个不同的三音素状态可能对应相同的pdf-id) 的统计量)
*/
void AccumulateTreeStats(const TransitionModel &trans_model,
                         const AccumulateTreeStatsInfo &info,
                         const std::vector<int32> &alignment,
                         const Matrix<BaseFloat> &features,
                         std::map<EventType, GaussClusterable*> *stats) {

  
  // # SplitToPhones 将utt 对齐trans-ids 根据对应的音素 进行split划分, 
  // # 划分得到以音素为top单元的Vector<音素- vector<Trans-id> > 
  // # 将trans-ids 转化为 以phone为分割的 状态序列. 
  std::vector<std::vector<int32> > split_alignment;
  bool ans = SplitToPhones(trans_model, alignment, &split_alignment);

  
  int32 cur_pos = 0;

  // # 每个分割好的句子. context_width = 3 , central_postion = 1 . 根据时序形成三音素上下文环境
  // # 模型从单音素转化为三音素，通过上下文信息认为是三音素内状态，每个状态就是三音素内状态。
  for (int32 i = -info.context_width; i < static_cast<int32>(split_alignment.size()); i++) {
    // # 形成三音素上下文环境   ; 从第一个音素开始 && 音素在该utt总音素长度范围内.
    if (i + info.central_position >= 0 &&
        i + info.central_position < static_cast<int32>(split_alignment.size())) {

      // info.phone_map == 实际只保存了 1. 代表sil音素, 所以phone-id 实际上没有改变
      // # 获得以i为开始的三音素窗的中心音素
      int32 central_phone = MapPhone(
                      // # 音素映射map
                      info.phone_map,
                      // # 对应的中心音素
                      // split_alignment[i + info.central_postion] 表示i（作为L）开始的三音素窗的中心音素
                      trans_model.TransitionIdToPhone(split_alignment[i+info.central_position][0]));

      // # 确定是否ctx_dep音素. ci -- context-independent  上下文无关. 都是false
      //  ===> is_ctx_dep = true;
      bool is_ctx_dep = !std::binary_search(info.ci_phones.begin(),
                                            info.ci_phones.end(),
                                            central_phone);

      // # 构建 EventType map内定位
      EventType evec;
      // # 遍历音素窗内每个音素 构建EventType  (0, L), (1, C), (2, R)
      for (int32 j = 0; j < info.context_width; j++) {
        int32 phone;
        // # 判断界定范围 
        if (i + j >= 0 && i + j < static_cast<int32>(split_alignment.size()))
          // # 获得三音素窗的每个音素
          phone = MapPhone(info.phone_map,
                       trans_model.TransitionIdToPhone(split_alignment[i+j][0]));
        else
          phone = 0;  
        // we also set the phone arbitrarily to 0

        // # 将<contex-width-index, phone> 加入 evec  得到某个状态的 EventTyep
        if (is_ctx_dep || j == info.central_position)
          evec.push_back(std::make_pair(static_cast<EventKeyType>(j), static_cast<EventValueType>(phone)));
      }
      

      // # 对齐中 某个音素内的所有状态-trans-id 
      for (int32 j = 0; j < static_cast<int32>(split_alignment[i+info.central_position].size());j++) {
        // # for central phone of this window...
        EventType evec_more(evec);
        // # 获得该状态当前的pdf-class
        int32 pdf_class = trans_model.TransitionIdToPdfClass(split_alignment[i+info.central_position][j]);

        // =========================  为三音素状态 构建 EventType ================
        // # pdf_class will normally by 0, 1 or 2 for 3-state HMM.
        // # 将<-1, state> 加入evec
        std::pair<EventKeyType, EventValueType> pr(kPdfClass, pdf_class);
        evec_more.push_back(pr);

        std::sort(evec_more.begin(), evec_more.end());  // these must be sorted!

        // # 为三音素的HMM状态 构建统计量.
        if (stats->count(evec_more) == 0)
          (*stats)[evec_more] = new GaussClusterable(dim, info.var_floor);

        // =========================  为三音素状态 增加统计量 ================
        // # 增加统计 features 特征 ------------------- 统计量就是统计feature, 用mfcc来计算gmm参数.
        BaseFloat weight = 1.0;
        *(*stats)[evec_more]->AddStats(features.Row(cur_pos), weight);*
        cur_pos++;
      }
    }
  }
  KALDI_ASSERT(cur_pos == static_cast<int32>(alignment.size()));
}


// SplitToPhones   Internal 转化到音素序列
//  static bool kaldi::SplitToPhonesInternal ( const TransitionModel &  trans_model,
//                                             const std::vector< int32 > &  alignment,
//                                             bool  reordered,
//                                             std::vector< std::vector< int32 > > *  split_output 
//                                             ) 

// 618   std::vector<size_t> end_points;  // points at which phones end [in an
// 619   // stl iterator sense, i.e. actually one past the last transition-id within
// 620   // each phone]..

// 622   bool was_ok = true;
//       # foreach 每帧状态
// 623   for (size_t i = 0; i < alignment.size(); i++) {
// 624     int32 trans_id = alignment[i];
//         # 正常音素分割点 是否是终止state
// 625     if (trans_model.IsFinal(trans_id)) {  // is final-prob
// 626       if (!reordered) end_points.push_back(i+1);
// 627       else {  // reordered.
// 628         while (i+1 < alignment.size() &&
// 629               trans_model.IsSelfLoop(alignment[i+1])) {
// 630           KALDI_ASSERT(trans_model.TransitionIdToTransitionState(alignment[i]) ==
// 631                  trans_model.TransitionIdToTransitionState(alignment[i+1]));
// 632           i++;
// 633         }
// 634         end_points.push_back(i+1);
// 635       }

//         # 错误情况
// 636     } else if (i+1 == alignment.size()) {
// 637       // need to have an end-point at the actual end.
// 638       // but this is an error- should have been detected already.
// 639       was_ok = false;
// 640       end_points.push_back(i+1);

//         # 状态判断
// 641     } else {
// 642       int32 this_state = trans_model.TransitionIdToTransitionState(alignment[i]),
// 643           next_state = trans_model.TransitionIdToTransitionState(alignment[i+1]);
// 644       if (this_state == next_state) continue;  // optimization.
// 645       int32 this_phone = trans_model.TransitionStateToPhone(this_state),
// 646           next_phone = trans_model.TransitionStateToPhone(next_state);
// 647       if (this_phone != next_phone) {
// 650         was_ok = false;
// 651         end_points.push_back(i+1);
// 652       }
// 653     }
// 654   }

//       # 将属于各自音素的状态 划归到音素队列中，形成 <音素 <状态>> 的结构
// 656   size_t cur_point = 0;
// 657   for (size_t i = 0; i < end_points.size(); i++) {
// 658     split_output->push_back(std::vector<int32>());
// 662     int32 trans_state =
// 663       trans_model.TransitionIdToTransitionState(alignment[cur_point]);
// 664     int32 phone = trans_model.TransitionStateToPhone(trans_state);
// 665     int32 forward_pdf_class = trans_model.GetTopo().TopologyForPhone(phone)[0].forward_pdf_class;
// 666     if (forward_pdf_class != kNoPdf)  // initial-state of the current phone is emitting
// 667       if (trans_model.TransitionStateToHmmState(trans_state) != 0)
// 668         was_ok = false;
//         # 划归状态到音素操作
// 669     for (size_t j = cur_point; j < end_points[i]; j++)
// 670       split_output->back().push_back(alignment[j]);
// 671     cur_point = end_points[i];
// 672   }
// 673   return was_ok;
// 674 }


// * sum-tree-stats
//   # in: 状态的MFCC统计量(并行多任务得到的)
//   # out: 汇总后统计量
int sum_tree_stats(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Sum statistics for phonetic-context tree building.\n"
        "Usage:  sum-tree-stats [options] tree-accs-out tree-accs-in1 tree-accs-in2 ...\n"
        "e.g.: \n"
        " sum-tree-stats treeacc 1.treeacc 2.treeacc 3.treeacc\n";

    ParseOptions po(usage);
    bool binary = true;

    *std::map<EventType, Clusterable*> tree_stats;*
    std::string tree_stats_wxfilename = po.GetArg(1);

    // A reminder on what BuildTreeStatsType is:
    // typedef std::vector<std::pair<EventType, Clusterable*> > BuildTreeStatsType;
    // # 多个并行任务的 状态 特征统计量.

    for (int32 arg = 2; arg <= po.NumArgs(); arg++) {
      std::string tree_stats_rxfilename = po.GetArg(arg);
      bool binary_in;
      Input ki(tree_stats_rxfilename, &binary_in);

      // # 统计量<EventType, GaussClusterable>      
      BuildTreeStatsType  stats_array;
      GaussClusterable example; // Lets ReadBuildTreeStats know which type to read..

      // # 读取统计量<EventType, GaussClusterable> 到 stats_array      
      ReadBuildTreeStats(ki.Stream(), binary_in, example, &stats_array);
      
      // # 汇总统计量  foreach pdf-id's GaussClusterable.
      for (BuildTreeStatsType::iterator iter = stats_array.begin();
           iter != stats_array.end(); ++iter) {

        EventType e = iter->first;
        Clusterable *c = iter->second;

        // # 获得EventType进行综合统计
        std::map<EventType, Clusterable*>::iterator map_iter = tree_stats.find(e);

        // # EventType 在 tree_stats 中还不存在, 则先构建一个<EventType, Clusterable> 对象加入到map中 等待总和统计.
        if (map_iter == tree_stats.end()) { // Not already present.
          tree_stats[e] = c;
        } else {
          map_iter->second->Add(*c);
          delete c;
        }
      }
    }



    // # 写入综合统计量
    BuildTreeStatsType stats;  // vectorized form.
    for (std::map<EventType, Clusterable*>::const_iterator iter = tree_stats.begin();  
        iter != tree_stats.end();
         ++iter) {
      stats.push_back(std::make_pair(iter->first, iter->second));
    }
    tree_stats.clear();

    {
      Output ko(tree_stats_wxfilename, binary);
      WriteBuildTreeStats(ko.Stream(), binary, stats);
    }
    KALDI_LOG << "Wrote summed accs ( " << stats.size() << " individual stats)";
    DeleteBuildTreeStats(&stats);
    return (stats.size() != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

  


// * cluster-phones
//   # in:
//   #   音素窗配置， 状态-MFCC统计量， 音素变体集合.
//   # out:
//   #   输出聚类的音素集合
int cluster_phones(int argc, char *argv[]) {

  const char *usage =
      "Cluster phones (or sets of phones) into sets for various purposes\n"
      "Usage:  cluster-phones [options] <tree-stats-in> <phone-sets-in> <clustered-phones-out>\n"
      "e.g.: \n"
      " cluster-phones 1.tacc phonesets.txt questions.txt\n";

  // phone sets.txt的格式 如下 将每行作为一组, 认为是聚类的一个基本单元.
  // Format of phonesets.txt is e.g.
  // 1
  // 2 3 4
  // 5 6
  // ...
  // Format of questions.txt output is similar, but with more lines (and the same phone
  // may appear on multiple lines).

  // bool binary = true;
  int32 P = 1, N = 3; // Note: N does not matter.
  // # central postion.
  std::string pdf_class_list_str = "1";  // 1 is just the central position of 3.
  std::string mode = "questions";
  int32 num_classes = -1;

  std::string 
      // # 统计量
      stats_rxfilename = po.GetArg(1),
      // # 音素集合
      phone_sets_rxfilename = po.GetArg(2),
      phone_sets_wxfilename = po.GetArg(3);

  BuildTreeStatsType stats;*

                           {  // Read tree stats.
                             bool binary_in;
                             GaussClusterable gc;  // dummy needed to provide type.
                             Input ki(stats_rxfilename, &binary_in);
                             // # 统计量 -> stats
                             ReadBuildTreeStats(ki.Stream(), binary_in, gc, &stats);
                           }

  // # 聚类音素使用的状态 的状态index
  std::vector<int32> pdf_class_list;
  // # 按： 分割字符串 pdf-class-list = 1
  if (!SplitStringToIntegers(pdf_class_list_str, ":", false, &pdf_class_list) || pdf_class_list.empty()) {
    KALDI_ERR << "Invalid pdf-class-list string [expecting colon-separated list of integers]: " 
              << pdf_class_list_str;
  }
    
    
  std::vector<std::vector< int32> > phone_sets;
  // # 读取sets.int 获得音素变体结合  <集合<音素变体>>
  if (!ReadIntegerVectorVectorSimple(phone_sets_rxfilename, &phone_sets)) ;

  // # ========
  if (mode == "questions") {

    std::vector<std::vector<int32> > phone_sets_out;
    // # in   统计量、 音素变体集合、1、1（中心状态、中心音素）
    // # out 聚类音素集合输出. phones_sets_out, 
    // # 音素划分树结构, 从顶层全部音素变体组 进行划分到一个节点的音素集合.
    AutomaticallyObtainQuestions(stats,
                                 phone_sets,
                                 pdf_class_list,
                                 P,
                                 &phone_sets_out);
  } else if (mode == "k-means") {
  }

  // # write聚类后音素集合 此时写入的question  是音素集合 phone_sets_out
  if (!WriteIntegerVectorVectorSimple(phone_sets_wxfilename, phone_sets_out))
    KALDI_ERR << "Error writing questions to "
              << PrintableWxfilename(phone_sets_wxfilename);
  else
    KALDI_LOG << "Wrote questions to "<<phone_sets_wxfilename;

  DeleteBuildTreeStats(&stats);
}


// ** AutomaticallyObjtainQuestion
// # # in:
// #    统计量
// #    sets.int 音素变体集合
// #    聚类音素 使用的状态 pdf-class
// #    P = 1 中心音素
// # # out:
// #    输出问题集合 --- 音素集合 -- 树型

void AutomaticallyObtainQuestions(BuildTreeStatsType &stats,
                                  const std::vector<std::vector<int32> > &phone_sets_in,
                                  const std::vector<int32> &all_pdf_classes_in,
                                  int32 P,
                                  std::vector<std::vector<int32> > *questions_out) 
// # stats   phone_set_in  state-pos, phone-pos (根据统计量 以及初始的变体音素集合，用 中心因素的中心状态进行聚类)
// # 输出聚类后问题
{
  std::vector<std::vector<int32> > phone_sets(phone_sets_in);
  std::vector<int32> phones;

  // # 读取所有音素 ==> phones
  for (size_t i = 0; i < phone_sets.size() ;i++) {
    std::sort(phone_sets[i].begin(), phone_sets[i].end());
    for (size_t j = 0; j < phone_sets[i].size(); j++)
      phones.push_back(phone_sets[i][j]);
  }
  std::sort(phones.begin(), phones.end());


  // # 只使用中心状态 all_pdf_classes = <1>
  std::vector<int32> all_pdf_classes(all_pdf_classes_in);

  // # filter 统计量, 只要中心状态的统计量 --> retained_stats
  BuildTreeStatsType retained_stats;
  FilterStatsByKey(stats, kPdfClass, all_pdf_classes,
                   true,  // retain only the listed positions
                   &retained_stats);

  // # 从 retained_stats , 按中心音素 划分三音素
  std::vector<BuildTreeStatsType> split_stats;  // split by phone.
  SplitStatsByKey(retained_stats, P, &split_stats);

  // # 按音素累计所有中心状态
  std::vector<Clusterable*> summed_stats;  // summed up by phone.
  SumStatsVec(split_stats, &summed_stats);

  // # 最大音素
  int32 max_phone = phones.back();
  if (static_cast<int32>(summed_stats.size()) < max_phone+1) {
    // this can happen if the last phone had no data.. if we are using
    // stress-marked, position-marked phones, this can happen.  The later
    // code will assume that a summed_stats entry exists for all phones.
    summed_stats.resize(max_phone+1, NULL);
  }


  EnsureClusterableVectorNotNull(&summed_stats);  // make sure no NULL pointers in summed_stats.
  // will replace them with pointers to empty stats.

  // # 按照phone_sets 中的方式将变体音素进行 综合统计统计量
  std::vector<Clusterable*> summed_stats_per_set(phone_sets.size(), NULL);  // summed up by set.
  for (size_t i = 0; i < phone_sets.size(); i++) {
    const std::vector<int32> &this_set = phone_sets[i];
    summed_stats_per_set[i] = summed_stats[this_set[0]]->Copy();
    for (size_t j = 1; j < this_set.size(); j++)
      summed_stats_per_set[i]->Add(*(summed_stats[this_set[j]]));
  }


  // # 进行音素聚类 =====================================================
  TreeClusterOptions topts;
  topts.kmeans_cfg.num_tries = 10;  // This is a slow-but-accurate setting,
  // # 每个音素 指定属于某个cluster
  std::vector<int32> assignments;  // assignment of phones to clusters. dim == summed_stats.size().
  // # 每个cluster的父节点
  std::vector<int32> clust_assignments;  // Parent of each cluster.  Dim == #clusters.

  int32 num_leaves;  // number of leaf-level clusters.
  // # 执行聚类
  TreeCluster(summed_stats_per_set,
              summed_stats_per_set.size(),  // max-#clust is all of the points.
              NULL,  // don't need the clusters out.
              &assignments,
              &clust_assignments,
              &num_leaves,
              topts);

  // process the information obtained by TreeCluster into the
  // form we want at output.


  // # 根据聚类结果, 进行输出得到, 
  // # questions_out
  // # 通过聚类算法得到的 对音素的划分树结构.
  ObtainSetsOfPhones(phone_sets,
                     assignments,
                     clust_assignments,
                     num_leaves,
                     questions_out);

  // The memory in summed_stats was newly allocated. [the other algorithms
  // used here do not allocate].
  DeletePointers(&summed_stats);
  DeletePointers(&summed_stats_per_set);
}

// *** SplitStatsByKey()
//     # 将状态按照某个音素进行划分 得到每个音素的状态统计量 <音素 <状态统计量>>
void SplitStatsByKey(const BuildTreeStatsType &stats_in, EventKeyType key, std::vector<BuildTreeStatsType> *stats_out) {

  BuildTreeStatsType::const_iterator iter, end = stats_in.end();

  size_t size = 0;
  // This loop works out size of output vector.
  for (iter = stats_in.begin(); iter != end; ++iter) {
    const EventType &evec = iter->first;
    EventValueType val;

    // # 中心状态中 所有属于中心音素的状态大小(全部都是) key = 1
    // # val 保存音素id
    if (! EventMap::Lookup(evec, key, &val)) // no such key.
      KALDI_ERR << "SplitStats: key "<< key << " is not present in event vector " << EventTypeToString(evec);
    // # 最终获得训练中得到的最大的音素id, 做数组大小
    size = std::max(size, (size_t)(val+1));
  }

  stats_out->resize(size);

  // This loop splits up the stats.
  for (iter = stats_in.begin(); iter != end; ++iter) {
    const EventType &evec = iter->first;
    EventValueType val;
    // # 将状态按中心音素 => stats_out.
    EventMap::Lookup(evec, key, &val);  // will not fail.
    // # 某个音素的统计量
    (*stats_out)[val].push_back(*iter);
  }
}




// *** TreeCluster
//      # 音素变体集合 进行聚类
//      TreeCluster(
//      # in  
//      # 按phone_set为集合 统计得到 集合内中心状态统计量
//      # 音素变体集合总数
//      summed_stats_per_set,
//      summed_stats_per_set.size(),  // max-#clust is all of the points.
//      NULL,  // don't need the clusters out.
//      # out
//      # 某个音素变体集合 属于某个cluster
//      &assignments,
//      # 所有节点cluster - id, 
//      # 叶子节点 0 - cnt_leaf, 
//      # 非叶子节点 cnt_leaf ---- clust_assignments.size()
//      # top 节点 == clust_assignments.size()
//      &clust_assignments,
//      &num_leaves,
//      topts);

  TreeClusterer(const std::vector<Clusterable*> &points,
                int32 max_clust,
                TreeClusterOptions cfg):
      points_(points), max_clust_(max_clust), ans_(0.0), cfg_(cfg)
  {
    KALDI_ASSERT(cfg_.branch_factor > 1);
    Init();
  }

  // # ======================
  BaseFloat Cluster(std::vector<Clusterable*> *clusters_out,
                    std::vector<int32> *assignments_out,
                    std::vector<int32> *clust_assignments_out,
                    int32 *num_leaves_out) {
    // # 循环优先队列queue 取出最大划分方法 对对应的Node进行继续划分
    *while (static_cast<int32>(leaf_nodes_.size()) < max_clust_ && !queue_.empty())*
    {
      std::pair<BaseFloat, Node*> pr = queue_.top();
      queue_.pop();
      ans_ += pr.first;
      // # 划分操作
      DoSplit(pr.second);
    }

    CreateOutput(clusters_out, assignments_out, clust_assignments_out,
                 num_leaves_out);
    return ans_;
  }

  // # ======================
  void DoSplit(Node *node) {
    node->children.resize(cfg_.branch_factor);
    for (int32 i = 0;i < cfg_.branch_factor;i++) {
      Node *child = new Node;
      node->children[i] = child;
      child->is_leaf = true;
      child->parent = node;
      // # node_total 统计量
      child->node_total = node->leaf.clusters[i];
      if (i == 0) {
        child->index = node->index;  // assign node's own index in leaf_nodes_ to 1st child.
        leaf_nodes_[child->index] = child;
      } else {
        child->index = leaf_nodes_.size();  // generate new indices for other children.
        leaf_nodes_.push_back(child);
      }
    }

    for (int32 i = 0; i < static_cast<int32>(node->leaf.points.size()); i++) {
      int32 child_index = node->leaf.assignments[i];
      KALDI_ASSERT(child_index < static_cast<int32>(cfg_.branch_factor));
      node->children[child_index]->leaf.points.push_back(node->leaf.points[i]);
      node->children[child_index]->leaf.point_indices.push_back(node->leaf.point_indices[i]);
    }
    node->leaf.points.clear();
    node->leaf.point_indices.clear();
    node->leaf.clusters.clear();  // already assigned pointers to children.
    node->leaf.assignments.clear();
    node->is_leaf = false;
    node->index = nonleaf_nodes_.size();  // new index at end of nonleaf_nodes_.
    nonleaf_nodes_.push_back(node);

    // # 对新节点进行计算可能的划分操作. ×××××× 并将可能的划分加入优先队列等待划分××××××
    for (int32 i = 0;i < static_cast<int32>(cfg_.branch_factor);i++)
      FindBestSplit(node->children[i]);
  }



// *** ObtainSetsOfPhones
// # ObtainSetsOfPhones
// # 根据 assignment clust-assignment 
// # 将所有音素放入到最顶层节点
// # 按问题将音素 分割放入到子节点
// # 继续将音素向下分割, 完成音素的聚类.

static void ObtainSetsOfPhones(const std::vector<std::vector<int32> > &phone_sets,  // the original phone sets, may
                               // just be individual phones.
                               const std::vector<int32> &assignments,  // phone-sets->clusters
                               const std::vector<int32> &clust_assignments,  // clust->parent
                               int32 num_leaves,  // number of clusters present..
                               std::vector<std::vector<int32> > *sets_out) {

  // # 聚类结果 父节点包含了音素集合的 <cluster < phones >>
  std::vector<std::vector<int32> > raw_sets(clust_assignments.size());

  // # 所有音素变体
  for (size_t i = 0; i < assignments.size(); i++) {
    // # 某个音素变体属于的某个叶子cluster
    int32 clust = assignments[i];  // this is an index into phone_sets.
    for (size_t j = 0; j < phone_sets[i].size(); j++) {
      // and not just a hole.
      // # 将对应的音素变体都加入到 cluster中
      raw_sets[clust].push_back(phone_sets[i][j]);
    }
  }

  // for all clusters including the top-level cluster:
  // [note that the top-level cluster contains all phones, but it may actually
  //  be useful because sometimes we cluster just the non-silence phones, so
  //  the list of all phones is a way of asking about silence in such a way
  // that epsilon (end-or-begin-of-utterance) gets lumped with silence.
  // # 每个簇
  for (int32 j = 0; j < static_cast<int32>(clust_assignments.size()); j++) {
    // # 父节点
    int32 parent = clust_assignments[j];
    // # 某个cluster的所有变体phone
    std::sort(raw_sets[j].begin(), raw_sets[j].end());
    // # 按树结构 将音素都安排到节点上, 越高节点安排的音素越多
    if (parent < static_cast<int32>(clust_assignments.size())-1) {  // parent is not out of range [i.e. not the top one]...
      // add all j's phones to its parent.
      raw_sets[parent].insert(raw_sets[parent].end(),
                              raw_sets[j].begin(),
                              raw_sets[j].end());
    }
  }

  // Reverse the 'raw_sets' so the most important things (top-level questions)
  // appear at the front... this will end up mattering because of the
  // --truncate-leftmost-questions option to compile-questions.
  std::reverse(raw_sets.begin(), raw_sets.end());

  // Now add the original sets-of-phones to the raw sets, to make sure all of
  // these are present.  (The main reason they might be absent is if the stats
  // are empty, but we want to ensure they are all there regardless).  
  // note these will be actual singleton sets if the sets-of-phones each contain just one
  // phone, which in some sense is the normal situation.
  for (size_t i = 0; i < phone_sets.size(); i++) {
    raw_sets.push_back(phone_sets[i]);
  }

  // Remove duplicate sets from "raw_sets".
  RemoveDuplicates(&raw_sets);
  sets_out->reserve(raw_sets.size());

  for (size_t i = 0; i < raw_sets.size(); i++)
    if (! raw_sets[i].empty())  // if the empty set is present, remove it...
      sets_out->push_back(raw_sets[i]);
  
}


// * compile-question
//   # in: 构造的txt类型的问题  --> question.qst 类型的问题集合
//   将问题转为qst模式?? 具体没看
int compile_question(int argc, char *argv[]) {

  const char *usage =
      "Compile questions\n"
      "Usage:  compile-questions [options] <topo> <questions-text-file> <questions-out>\n"
      "e.g.: \n"

      " compile-questions questions.txt questions.qst\n";
  bool binary = true;
  int32 P = 1, N = 3;
  int32 num_iters_refine = 0;

  std::string
      topo_filename = po.GetArg(1),

      questions_rxfilename = po.GetArg(2),
      questions_out_filename = po.GetArg(3);

  HmmTopology topo;  // just needed for checking, and to get the
  ReadKaldiObject(topo_filename, &topo);

  // # 多个 音素集合的集合
  // # < <音素集合> <音素集合> <>>
  std::vector<std::vector<int32> > questions;  // sets of phones.

  // # read question     <phones_set<phone-id>>
  if (!ReadIntegerVectorVectorSimple(questions_rxfilename, &questions))
    KALDI_ERR << "Could not read questions from "
              << PrintableRxfilename(questions_rxfilename);

  // # foreach phones_set
  for (size_t i = 0; i < questions.size(); i++) {
    std::sort(questions[i].begin(), questions[i].end());
    if (!IsSortedAndUniq(questions[i]))
      KALDI_ERR << "Questions contain duplicate phones";
  }

  size_t nq = static_cast<int32>(questions.size());
  SortAndUniq(&questions);
  if (questions.size() != nq)
    KALDI_WARN << (nq-questions.size())
               << " duplicate questions present in " << questions_rxfilename;

  // # 检查 topo中的所有音素都在至少一个问题中, 并返回所有音素中最大的pdf-class？？？
  int32 max_num_pdfclasses = ProcessTopo(topo, questions);




  // # 构造 Questions 对象.
  Questions qo;

  // # 对音素提问 可以有 postion  0,1,2， 可能的问题是 是否属于某个音素集合.
  // # phone questions (0, 1, 2)
  QuestionsForKey phone_opts(num_iters_refine);

  // the questions-options corresponding to keys 0, 1, .. N-1 which
  // represent the phonetic context positions (including the central phone).
  // # 音素窗N=3, 问题qo, 对所有位置0,1,2的问题 初始时都是全部音素
  for (int32 n = 0; n < N; n++) {
    KALDI_LOG << "Setting questions for phonetic-context position "<< n;
    // # 所有 音素集合
    // # std::vector<std::vector<int32> > questions;  // sets of phones.
    // # std::vector<std::vector<EventValueType> > initial_questions;  // sets of phones.
    phone_opts.initial_questions = questions;
    // # 对某个key，增加可能的phone_opts 是多个 可能的音素集合, 默认是问题集整体
    qo.SetQuestionsOf(n, phone_opts);
  }




  // # 对状态提问, 只有pos = -1, 并且问题 也只是 pdf-class = 0, pdf-class = 1, |??? 但是和结果不相符合呢.
  // # (-1)
  QuestionsForKey pdfclass_opts(num_iters_refine);
  // # <0<>, 1<>, 2<>>
  std::vector<std::vector<int32> > pdfclass_questions(max_num_pdfclasses-1);
  // # 每个可能状态index 0, 1, 2
  for (int32 i = 0; i < max_num_pdfclasses - 1; i++)
    // # 从0 - 状态index
    for (int32 j = 0; j <= i; j++)
      pdfclass_questions[i].push_back(j);

  // # 什么意思??
  // # 0, <0>
  // # 1, <0, 1>
  // # <<0>, <0, 1>>
  // # E.g. if max_num_pdfclasses == 3,  pdfclass_questions is now.  

      pdfclass_opts.initial_questions = pdfclass_questions;
      qo.SetQuestionsOf(kPdfClass, pdfclass_opts);
      WriteKaldiObject(qo, questions_out_filename, binary);
}


// * build-tree
//   # in:  状态MFCC统计量  roots.txt  问题-音素集合 topo
//   # out: 状态绑定决策树  GetStubMap 基本树----  SplitDecisionTree 准决策树----- ClusterEventMapRestrictedByMap 状态绑定决策树
int build_tree(int argc, char *argv[]) {
  using namespace kaldi;
  const char *usage =
        "Train decision tree\n"
        "Usage:  build-tree [options] <tree-stats-in> <roots-file> <questions-file> <topo-file> <tree-out>\n"
  
  "e.g.: \n"
  " build-tree treeacc roots.txt 1.qst topo tree\n";

    bool binary = true;
    int32 P = 1, N = 3;

    BaseFloat thresh = 300.0;
    BaseFloat cluster_thresh = -1.0;  // negative means use smallest split in splitting phase as thresh.
    int32 max_leaves = 0;
    std::string occs_out_filename;

    std::vector<std::vector<int32> > phone_sets;
    std::vector<bool> is_shared_root;
    std::vector<bool> is_split_root;

    {
      Input ki(roots_filename.c_str());
      ReadRootsFile(ki.Stream(), &phone_sets, &is_shared_root, &is_split_root);
    }

    HmmTopology topo;
    ReadKaldiObject(topo_filename, &topo);

    BuildTreeStatsType stats;
    {
      bool binary_in;
      GaussClusterable gc;  // dummy needed to provide type.
      Input ki(stats_filename, &binary_in);
      ReadBuildTreeStats(ki.Stream(), binary_in, gc, &stats);
    }
    KALDI_LOG << "Number of separate statistics is " << stats.size();

    Questions qo;
    {
      qo.Read(ki.Stream(), binary_in);
    }


    std::vector<int32> phone2num_pdf_classes;
    topo.GetPhoneToNumPdfClasses(&phone2num_pdf_classes);

    EventMap *to_pdf = NULL;


    // # //////// Build the tree. ////////////
    to_pdf = BuildTree(qo,
                       phone_sets,
                       phone2num_pdf_classes,
                       is_shared_root,
                       is_split_root,
                       stats,
                       thresh,
                       max_leaves,
                       cluster_thresh,
                       P);  // 1


    // # 根据构建的树 构建ctx_dep 对象 写文件
    ContextDependency ctx_dep(N, P, to_pdf);  // takes ownership
    WriteKaldiObject(ctx_dep, tree_out_filename, binary);

    KALDI_LOG << "Wrote tree";
    DeleteBuildTreeStats(&stats);
}


// 构建树的实际过程
EventMap *BuildTree(Questions &qopts,                                      // question
                    const std::vector<std::vector<int32> > &phone_sets,    // roots
                    const std::vector<int32> &phone2num_pdf_classes,       // 每个音素的状态
                    const std::vector<bool> &share_roots,                  // roots中是否进行共享
                    const std::vector<bool> &do_split,                     // 是否进行分列
                    const BuildTreeStatsType &stats,                       // 音素决策统计量
                    BaseFloat thresh,
                    int32 max_leaves,
                    BaseFloat cluster_thresh,  // typically == thresh.  If negative, use smallest split.
                    int32 P) {

  // the inputs will be further checked in GetStubMap.
  int32 num_leaves = 0;  // allocator for leaves.

  // # ########################################################
  // # 构建状态绑定树基础
  EventMap *tree_stub = GetStubMap(P,
                                   phone_sets,
                                   phone2num_pdf_classes,
                                   share_roots,
                                   &num_leaves);

  BaseFloat impr;
  BaseFloat smallest_split = 1.0e+10;

  // # 全部插入到 nonsplit_phones
  std::vector<int32> nonsplit_phones;
  for (size_t i = 0; i < phone_sets.size(); i++)
    if (!do_split[i])
      nonsplit_phones.insert(nonsplit_phones.end(), phone_sets[i].begin(), phone_sets[i].end());

  std::sort(nonsplit_phones.begin(), nonsplit_phones.end());



  // # #############################################  
  BuildTreeStatsType filtered_stats;
  // # 过滤统计量.// retain only those not in "nonsplit_phones"
  FilterStatsByKey(stats, P, nonsplit_phones, false,  
                   &filtered_stats);

  // # 在 tree_sub 基础上 根据过滤后统计量, 问题 门限  要求节点数 进行状态绑定 得到 准决策树tree_split
  EventMap *tree_split = SplitDecisionTree(*tree_stub,
                                           filtered_stats,
                                           qopts, thresh, max_leaves,
                                           &num_leaves, &impr, &smallest_split);





  // #    ?????????????????????????????? 
  if (cluster_thresh < 0.0) {
    KALDI_LOG <<  "Setting clustering threshold to smallest split " << smallest_split;
    cluster_thresh = smallest_split;
  }

  BaseFloat 
  // # 归一化
  normalizer = SumNormalizer(stats),
  impr_normalized = impr / normalizer,
  normalizer_filt = SumNormalizer(filtered_stats),
  impr_normalized_filt = impr / normalizer_filt;


  // # 状态绑定 ================
  if (cluster_thresh != 0.0) {   // Cluster the tree.
    BaseFloat objf_before_cluster = ObjfGivenMap(stats, *tree_split);

    // Now do the clustering.
    int32 num_removed = 0;
    EventMap *tree_clustered = ClusterEventMapRestrictedByMap(*tree_split,
                                                              stats,
                                                              cluster_thresh,
                                                              *tree_stub,
                                                              &num_removed);
    KALDI_LOG <<  "BuildTree: removed "<< num_removed << " leaves.";

    int32 num_leaves = 0;
    EventMap *tree_renumbered = RenumberEventMap(*tree_clustered, &num_leaves);

    BaseFloat objf_after_cluster = ObjfGivenMap(stats, *tree_renumbered);
  }
}


// GetStubMap

// # 从roots.int 的音素集合开始 为每一行构建一个叶子节点, 作为状态绑定数的基础
EventMap *GetStubMap(int32 P,
                     const std::vector<std::vector<int32> > &phone_sets,    
                     const std::vector<int32> &phone2num_pdf_classes,
                     const std::vector<bool> &share_roots,
                     int32 *num_leaves_out) 
// # POSTION = 1
// # rooots
// # 音素含有状态数 
// # bool 是否共享
// # 输出叶节点数
{

  // Initially create a single leaf for each phone set.
  // # roots音素集合中 包含最多的音素集合的音素数目
  size_t max_set_size = 0;
  // # 所有音素中的最大音素id??
  int32 highest_numbered_phone = 0;
  for (size_t i = 0; i < phone_sets.size(); i++) {
    max_set_size = std::max(max_set_size, phone_sets[i].size());
    
    highest_numbered_phone =
        std::max(highest_numbered_phone,
                 // # 音素中的最大值
                 * std::max_element(phone_sets[i].begin(), phone_sets[i].end()));
  }

  // # 当分类到达终止时, 只有一个roots的音素集合, 说明到达 状态决策树的树根
  if (phone_sets.size() == 1) {  // there is only one set so the recursion finishes.
    // # 是否共享根 是 用CE 否则 TE
    if (share_roots[0]) {  // if "shared roots" return a single leaf.
      return new ConstantEventMap( (*num_leaves_out)++ );
    } else {  // not sharing roots -> work out the length and return a
             // TableEventMap splitting on length.
      EventAnswerType max_len = 0;
      for (size_t i = 0; i < phone_sets[0].size(); i++) {
        EventAnswerType len;
        EventValueType phone = phone_sets[0][i];
        KALDI_ASSERT(static_cast<size_t>(phone) < phone2num_pdf_classes.size());
        len = phone2num_pdf_classes[phone];
        KALDI_ASSERT(len > 0);
        if (i == 0) max_len = len;
        else {
          if (len != max_len) {
            KALDI_WARN << "Mismatching lengths within a phone set: " << len
                       << " vs. " << max_len << " [unusual, but not necessarily fatal]. ";
            max_len = std::max(len, max_len);
          }
        }
      }
      std::map<EventValueType, EventAnswerType> m;
      for (EventAnswerType p = 0; p < max_len; p++)
        m[p] = (*num_leaves_out)++;
      return new TableEventMap(kPdfClass,  // split on hmm-position
                               m);
    }
  }
  // # 有多个音素集合但所有因素集合中都只有一个音素  直接使用TE 分类
  else if (max_set_size == 1 && static_cast<int32>(phone_sets.size()) <= 2*highest_numbered_phone) {
    // create table map splitting on phone-- more efficient.
    // the part after the && checks that this would not contain a very sparse vector.
    std::map<EventValueType, EventMap*> m;

    for (size_t i = 0; i < phone_sets.size(); i++) {
      std::vector<std::vector<int32> > phone_sets_tmp;
      phone_sets_tmp.push_back(phone_sets[i]);
      // # 某个集合是否共享
      std::vector<bool> share_roots_tmp;
      share_roots_tmp.push_back(share_roots[i]);
      EventMap *this_stub = GetStubMap(P, phone_sets_tmp, phone2num_pdf_classes,
                                       share_roots_tmp,
                                       num_leaves_out);
      KALDI_ASSERT(m.count(phone_sets_tmp[0][0]) == 0);
      m[phone_sets_tmp[0][0]] = this_stub;
    }
    return new TableEventMap(P, m);
  }
  // # 还可继续划分时, 直接进行二分化分  ==== SE
  else {
    // Do a split.  Recurse.
    size_t half_sz = phone_sets.size() / 2;

    // # 取一般 得到 一般的音素集合 以及对应集合是否shared
    std::vector<std::vector<int32> >::const_iterator half_phones =
        phone_sets.begin() + half_sz;  
    std::vector<bool>::const_iterator half_share =
        share_roots.begin() + half_sz;

    std::vector<std::vector<int32> > phone_sets_1, phone_sets_2;
    std::vector<bool> share_roots_1, share_roots_2;

    phone_sets_1.insert(phone_sets_1.end(), phone_sets.begin(), half_phones);
    phone_sets_2.insert(phone_sets_2.end(), half_phones, phone_sets.end());
    share_roots_1.insert(share_roots_1.end(), share_roots.begin(), half_share);
    share_roots_2.insert(share_roots_2.end(), half_share, share_roots.end());
    // # 无理由分半划分
    EventMap *map1 = GetStubMap(P, phone_sets_1, phone2num_pdf_classes, share_roots_1, num_leaves_out);
    EventMap *map2 = GetStubMap(P, phone_sets_2, phone2num_pdf_classes, share_roots_2, num_leaves_out);

    // # EventType <EventKeyType, EventValueType>
    std::vector<EventKeyType> all_in_first_set;

    // # 每个集合每个音素
    for (size_t i = 0; i < half_sz; i++)
      for (size_t j = 0; j < phone_sets[i].size(); j++)
        all_in_first_set.push_back(phone_sets[i][j]);
        
    std::sort(all_in_first_set.begin(), all_in_first_set.end());
    return new SplitEventMap(P, all_in_first_set, map1, map2);
  }
}


// SplitDecisionTree
EventMap *SplitDecisionTree(const EventMap &input_map,
                            const BuildTreeStatsType &stats,
                            Questions &q_opts,
                            BaseFloat thresh,
                            int32 max_leaves,  // max_leaves<=0 -> no maximum.
                            int32 *num_leaves,
                            BaseFloat *obj_impr_out,
                            BaseFloat *smallest_split_change_out) 
{


  int32 num_empty_leaves = 0;
  BaseFloat like_impr = 0.0;
  BaseFloat smallest_split_change = 1.0e+20;

  std::vector<DecisionTreeSplitter*> builders;


  // # =========================
  {
    // # 讲stats 按照 状态绑定基础树 上的 roots每行音素 进行划分 状态绑定统计量. --> split_stats
    std::vector<BuildTreeStatsType> split_stats;
    SplitStatsByMap(stats, input_map, &split_stats);

    KALDI_ASSERT(split_stats.size() != 0);
    builders.resize(split_stats.size());  // size == #leaves.

    // # 对tree_sub的基本树 roots每行音素的 节点、统计量 构建一个DTS
    for (size_t i = 0;i < split_stats.size();i++) {
      // #  EventAnswerType  leaf????
      EventAnswerType leaf = static_cast<EventAnswerType>(i);

      if (split_stats[i].size() == 0) num_empty_leaves++;
      // # 为该叶子节点构建一个 DecisionTreeSplitter， 后面用来构建状态绑定过程树, 基本问题集，就是传入的q_opts.
      *builders[i] = new DecisionTreeSplitter(leaf, split_stats[i], q_opts);*
    }
  }

  // # ========================= Do the splitting.  // 
  {  
    int32 count = 0;
    // # queue < <float, size_t>> <最优化分对似然度的提升,  某个roots行(not leaf-id)>
    std::priority_queue<std::pair<BaseFloat, size_t> > queue;  

    // Initialize queue.
    for (size_t i = 0; i < builders.size(); i++)
        *queue.push(std::make_pair(builders[i]->BestSplit(), i));*



    // # 似然度 > 门限 && 节点数还不够多
    while (queue.top().first > thresh
          && (max_leaves<=0 || *num_leaves < max_leaves)) {

      smallest_split_change = std::min(smallest_split_change, queue.top().first);
      // # 某个roots行
      size_t i = queue.top().second;
      like_impr += queue.top().first;
      // # #######################################
      // # 根据问题等 进行决策, 划分状态  
      // # 决策树 划分操作  按问题划分, 判断划分后结果熵增
      builders[i]->DoSplit(num_leaves);
      queue.pop();
      queue.push(std::make_pair(builders[i]->BestSplit(), i));
      count++;
    }
    KALDI_LOG << "DoDecisionTreeSplit: split "<< count << " times, #leaves now " << (*num_leaves);
  }

  if (smallest_split_change_out)
    *smallest_split_change_out = smallest_split_change;


   // Create the output EventMap  状态绑定树
  EventMap *answer = NULL;
  {  
    // # 多个EventMap   每个roots行 具有一个EventMap
    std::vector<EventMap*> sub_trees(builders.size());
    // # 根据绑定结果 用 EventMap表示.
    for (size_t i = 0; i < sub_trees.size();i++) 
        sub_trees[i] = builders[i]->GetMap();

    // # 将状态决策树的结果追加到 tree_sub基本树上 如此从基本树 得到了完整的 状态绑定树.
    // # 因为sub_trees input_map 实际上都是保存 <EventType, EventAnswer> 的EventMAP对象
    // 直接讲sub_trees中的<EventType, EventAnswer>
    // # 拷贝进入 input_map 就可以了.
    answer = input_map.Copy(sub_trees);
    for (size_t i = 0; i < sub_trees.size();i++) delete sub_trees[i];
  }

  // Free up memory.
  for (size_t i = 0;i < builders.size();i++) delete builders[i];
  if (obj_impr_out != NULL) *obj_impr_out = like_impr;
  return answer;
}

void SplitStatsByKey(const BuildTreeStatsType &stats_in, EventKeyType key, std::vector<BuildTreeStatsType> *stats_out) {

  BuildTreeStatsType::const_iterator iter, end = stats_in.end();
  stats_out->clear();
  size_t size = 0;
  // # This loop works out size of output vector.
  for (iter = stats_in.begin(); iter != end; ++iter) {
    const EventType &evec = iter->first;
    EventValueType val;
    if (! EventMap::Lookup(evec, key, &val)) // no such key.
      KALDI_ERR << "SplitStats: key "<< key << " is not present in event vector " << EventTypeToString(evec);
    size = std::max(size, (size_t)(val+1));
  }
  stats_out->resize(size);
  // This loop splits up the stats.
  for (iter = stats_in.begin(); iter != end; ++iter) {
    const EventType &evec = iter->first;
    EventValueType val;
    EventMap::Lookup(evec, key, &val);  // will not fail.
    (*stats_out)[val].push_back(*iter);
  }
}


// **** FindBestSplitForKey
//     # 按某个key （-1, 0, 1, 2）进行特征划分. 是否属于某个question集合, 是的话划分为两部分 yes_set_out & .
//     # in:
//     #    stats 所有统计量
//     #    问题 音素集合Vector
//     #    key
//     # out:
//     #    yes_set_out   
BaseFloat FindBestSplitForKey(const BuildTreeStatsType &stats,
                              const Questions &q_opts,
                              EventKeyType key,
                              std::vector<EventValueType> *yes_set_out) {

  # 按key位置的 所有×音素× 进行划分stats统计量
　std::vector<Clusterable*> summed_stats;
  {  // compute summed_stats
    std::vector<BuildTreeStatsType> split_stats;
    # 每个音素的统计量的所有状态的统计量
    SplitStatsByKey(stats, key, &split_stats);
    # 总和音素的所有统计量 -> summed_stats
    SumStatsVec(split_stats, &summed_stats);
  }

  # 计算按key 进行划分的提升
  std::vector<EventValueType> yes_set;
  # yes_set 是问题集合q_opts 中的一个问题 --- key可能属于的一个音素集合
  BaseFloat improvement = ComputeInitialSplit(summed_stats,
                                               q_opts, key, &yes_set);


  # find best basic question.
  # 所有音素的 assignment
  std::vector<int32> assignments(summed_stats.size(), 0);  // assigns to "no" (0) by default.

  # yes_set 集合内每个音素, 讲yes_set中的音素的assignment设置为 1.
  for (std::vector<EventValueType>::const_iterator iter = yes_set.begin(); iter != yes_set.end(); ++iter) {
    # 音素 phone-id < assignment.size
    if (*iter < (EventValueType)assignments.size()) {
      # assign to yes.
      assignments[*iter] = 1;  // assign to "yes" (1).
    }
  }

  # 两个clusterable统计量 代表 yes no
  std::vector<Clusterable*> clusters(2, (Clusterable*)NULL);  // no, yes.
  # 根据 assignment 将统计量 分配到 clusters。
  kaldi::AddToClusters(summed_stats, assignments, &clusters);

  // even if improvement == 0 we continue; if we do RefineClusters we may get further improvement.
  // now do the RefineClusters stuff.  Note that this is null-op if
  // q_opts.GetQuestionsOf(key).refine_opts.num_iters == 0.  We could check for this but don't bother;
  // it happens in RefineClusters anyway.

  # refine === 0; this will not do refine
  if (q_opts.GetQuestionsOf(key).refine_opts.num_iters > 0) {
    // If we want to refine the questions... (a bit like k-means w/ 2 classes).
    // Note: the only reason we introduced the if-statement is so the yes_set
    // doesn't get modified (truncated, actually) if we do the refine stuff with
    // zero iters.
    BaseFloat refine_impr = RefineClusters(summed_stats, &clusters, &assignments,
                                           q_opts.GetQuestionsOf(key).refine_opts);
    KALDI_ASSERT(refine_impr > std::min(-1.0, -0.1*fabs(improvement)));
    // refine_impr should always be positive
    improvement += refine_impr;
    yes_set.clear();
    for (size_t i = 0;i < assignments.size();i++) if (assignments[i] == 1) yes_set.push_back(i);
  }
  *yes_set_out = yes_set;
    
  DeletePointers(&clusters);
  DeletePointers(&summed_stats);
  return improvement; // objective-function improvement.
}


// **** ComputeInitialSplit()
//      # 按key 进行最优划分 统计量,
//      # 因为统计量 实际上就代表了 需要进行状态绑定决策树的所有状态

// # # in:
// #     key所有可能取值的 统计量
// #     q_opts 问题集合
// #     key (-1, 0, 1, 2)
// # # out     
// #     key 属于某个问题--音素集时达到 improvement 达到最大.
// #     yes_set 就是该音素集

BaseFloat ComputeInitialSplit(const std::vector<Clusterable*> &summed_stats,
                              const Questions &q_opts, EventKeyType key,
                              std::vector<EventValueType> *yes_set) {
  KALDI_ASSERT(yes_set != NULL);
  yes_set->clear();
  # key可能的集合 (初始时 对于音素EventKeyType 都是全部问题)
  const QuestionsForKey &key_opts = q_opts.GetQuestionsOf(key);

  // "total" needed for optimization in AddToClustersOptimized,
  // and also used to work out total objf.
  Clusterable *total = SumClusterable(summed_stats);
  if (total == NULL) return 0.0;  // because there were no stats or non-NULL stats.
  BaseFloat unsplit_objf = total->Objf();

  # 对某个key 的可能问题 是多个可能的音素集合
  const std::vector<std::vector<EventValueType> > &questions_of_this_key = key_opts.initial_questions;

  int32 best_idx = -1;
  BaseFloat best_objf_change = 0;

  # foreach question
  for (size_t i = 0; i < questions_of_this_key.size(); i++) {
    #  yes_set 一个可能的问题 -- 音素集合。
    const std::vector<EventValueType> &yes_set = questions_of_this_key[i];
    # size -- 所有可能的集合
    std::vector<int32> assignments(summed_stats.size(), 0);  // 0 is index of "no".
    std::vector<Clusterable*> clusters(2);  // no and yes clusters.
    # question 中集合的 每个音素
    for (std::vector<EventValueType>::const_iterator iter = yes_set.begin(); iter != yes_set.end(); ++iter) {
      KALDI_ASSERT(*iter>=0);

      if (*iter < (EventValueType)assignments.size()) assignments[*iter] = 1;
    }
    # 进行熵增计算
    kaldi::AddToClustersOptimized(summed_stats, assignments, *total, &clusters);
    BaseFloat this_objf = SumClusterableObjf(clusters);
    # 选择增益最大的 音素集合.
    BaseFloat this_objf_change = this_objf - unsplit_objf;
    if (this_objf_change > best_objf_change) {
      best_objf_change = this_objf_change;
      # 选择某个问题 --- 某个音素集合   作为最佳划分
      best_idx = i;  
    }

    DeletePointers(&clusters);
  }
  # end for


  delete total;
  # yes_set ===== 最佳问题 -- 某个音素集合
  if (best_idx != -1)
    *yes_set = questions_of_this_key[best_idx];
  return best_objf_change;
}


// *** ClusterEventMapRestrictedByMap

// **** ObjfGivenMap(stats, tree_split)
     
BaseFloat ObjfGivenMap(const BuildTreeStatsType &stats_in, const EventMap &e) {

  std::vector<BuildTreeStatsType> split_stats;
  # 将所有统计信息 按 决策树叶节点 划分统计量 --> split_stats
  SplitStatsByMap(stats_in, e, &split_stats);

  std::vector<Clusterable*> summed_stats;
  # 将split_stats 进行汇总
  SumStatsVec(split_stats, &summed_stats);
  # 计算objf 熵增
  BaseFloat ans = SumClusterableObjf(summed_stats);
  DeletePointers(&summed_stats);
  return ans;
}

// # 将 统计量集合 stats 按EventMap 进行划分 得到stats_out split分割的统计集合
// # EventMap 是SE TE CE构成的结构, Map函数最终映射到CE上 代表某个待聚类的 叶节点id.

void SplitStatsByMap(const BuildTreeStatsType &stats, const EventMap &e, std::vector<BuildTreeStatsType> *stats_out) {
  BuildTreeStatsType::const_iterator iter, end = stats.end();
  KALDI_ASSERT(stats_out != NULL);
  stats_out->clear();
  size_t size = 0;

  for (iter = stats.begin(); iter != end; ++iter) {
    const EventType &evec = iter->first;
    EventAnswerType ans;
    # 找到最大的id
    if (!e.Map(evec, &ans)) // this is an error--could not map it.
      KALDI_ERR << "SplitStatsByMap: could not map event vector " << EventTypeToString(evec)
                << "if error seen during tree-building, check that "
                << "--context-width and --central-position match stats, "
                << "and that phones that are context-independent (CI) during "
                << "stats accumulation do not share roots with non-CI phones.";
    size = std::max(size, (size_t)(ans+1));
  }
  
  stats_out->resize(size);
  for (iter = stats.begin(); iter != end; ++iter) {
    const EventType &evec = iter->first;
    EventAnswerType ans;
    # 将stats 按照Map 叶节点分割统计量
    bool b = e.Map(evec, &ans);
    KALDI_ASSERT(b);
    (*stats_out)[ans].push_back(*iter);
  }
}
}


// **** ClusterEventMapRestrictedByMap()

EventMap *tree_clustered = ClusterEventMapRestrictedByMap(*tree_split,
                                                           stats,
                                                           cluster_thresh,
                                                           *tree_stub,
                                                           &num_removed);

// # # in:
// #     tree_split 划分到待聚类的状态
// #     stats       
// #     cluster_thresh
// #     tree_stub 划分到roots的每行音素变体
// #     num_removed
// # # out:
// #     tree_clustered --- 最终决策树 EventMap

EventMap *ClusterEventMapRestrictedByMap(const EventMap &e_in,
                                         const BuildTreeStatsType &stats,
                                         BaseFloat thresh,
                                         const EventMap &e_restrict,
                                         int32 *num_removed_ptr) {
                                         
  std::vector<EventMap*> leaf_mapping;
  std::vector<BuildTreeStatsType> split_stats;

  int num_removed = 0;
  # 先按照 tree_stub 基本树划分 得到基本树统计量
  SplitStatsByMap(stats, e_restrict, &split_stats);

  # split_stats 对每个基本树的统计量进行划分
  for (size_t i = 0; i < split_stats.size(); i++) {
    if (!split_stats[i].empty())
      # 按照e_in 准决策树进行统计量划分
      num_removed += ClusterEventMapGetMapping(e_in, split_stats[i], thresh,
                                               &leaf_mapping);
  }

  if (num_removed_ptr != NULL) *num_removed_ptr = num_removed;

  # 将leaf 追加到 准决策树上, 完成决策树.
  EventMap *ans = e_in.Copy(leaf_mapping);
  DeletePointers(&leaf_mapping);
  return ans;
}




// # 是对某个 roots 行子树 进行的再聚类
// # # in:
// #     e_in  准决策树  -- 从根(所有音素所有状态)-> roots音素变体 -> 基本决策完成
// #     当前roots某行 基本树叶节点 的统计信息
// #     thresh
    
// # # out
// #     mapping   对该roots行的子树聚类结果

int ClusterEventMapGetMapping(const EventMap &e_in,
                              const BuildTreeStatsType &stats,
                              BaseFloat thresh,
                              std::vector<EventMap*> *mapping) {

  // First map stats

  # vector<多个叶节点 <每个叶节点包含的所有EventType的统计量>>
  # split  使用准决策树 进行将从 roots行音素变体出发的统计量 划分到叶子节点
  std::vector<BuildTreeStatsType> split_stats;
  SplitStatsByMap(stats, e_in, &split_stats);

  # 汇总roots行音素子树 的 每个叶子节点的多个状态的统计量
  std::vector<Clusterable*> summed_stats;
  SumStatsVec(split_stats, &summed_stats);

  
  std::vector<int32> indexes;
  std::vector<Clusterable*> summed_stats_contiguous;
  size_t max_index = 0;

  # summed_stats.size() == 该roots行子树 的所有叶子节点
  for (size_t i = 0;i < summed_stats.size();i++) {
    if (summed_stats[i] != NULL) {
      # indexes保存该准决策树叶子节点的 index
      # summed_stats_contiguous 保存每个叶节点的汇总统计量
      indexes.push_back(i);
      summed_stats_contiguous.push_back(summed_stats[i]);
      if (i > max_index) max_index = i;
    }
  }

  std::vector<int32> assignments;
  BaseFloat 
  # 汇总正则化
  normalizer = SumClusterableNormalizer(summed_stats_contiguous), 
  change;
  # 
  change = ClusterBottomUp(summed_stats_contiguous,
                           thresh,
                           0,  // no min-clust: use threshold for now.
                           NULL,  // don't need clusters out.
                           &assignments);  // this algorithm is quadratic, so might be quite slow.


  KALDI_ASSERT(assignments.size() == summed_stats_contiguous.size() && !assignments.empty());
  size_t num_clust = * std::max_element(assignments.begin(), assignments.end()) + 1;
  int32 num_combined = summed_stats_contiguous.size() - num_clust;
  KALDI_ASSERT(num_combined >= 0);

  KALDI_VLOG(2) <<  "ClusterBottomUp combined "<< num_combined
                << " leaves and gave a likelihood change of " << change
                << ", normalized = " << (change/normalizer)
                << ", normalizer = " << normalizer;
  KALDI_ASSERT(change < 0.0001);  // should be negative or zero.

  KALDI_ASSERT(mapping != NULL);
  if (max_index >= mapping->size()) mapping->resize(max_index+1, NULL);

  for (size_t i = 0;i < summed_stats_contiguous.size();i++) {
    size_t index = indexes[i];
    size_t new_index = indexes[assignments[i]];  // index assigned by clusterig-- map to existing indices in the map,
    // that we clustered from, so we don't conflict with indices in other parts
    // of the tree.
    KALDI_ASSERT((*mapping)[index] == NULL || "Error: Cluster seems to have been "
                 "called for different parts of the tree with overlapping sets of "
                 "indices.");
    (*mapping)[index] = new ConstantEventMap(new_index);
  }
  DeletePointers(&summed_stats);
  return num_combined;
}



// # # in:
// #     points    准基本树叶子节点 汇总统计量
// #     max_merge_thresh  门限, 门限内的两个叶子节点可以绑定.(限制的是两个统计量的距离, 相似度)
// #     min_clust  最少需要绑定的数量
// #     clusters_out 汇总输出, 一般不需要
// # # out:
// #     assignment 节点绑定    绑定结果, 多个叶子节点 如果assignment保存index相同说明被绑定
// #     BaseFloat 输出绑定后增益

BaseFloat ClusterBottomUp(const std::vector<Clusterable*> &points,
                          BaseFloat max_merge_thresh,
                          int32 min_clust,
                          std::vector<Clusterable*> *clusters_out,
                          std::vector<int32> *assignments_out) {
  # 当前roots行准决策子树 的 节点总数
  int32 npoints = points.size();
  // make sure fits in uint_smaller and does not hit the -1 which is reserved.

  # 构造初始化, 统计量 绑定门限 最小绑定数目, 绑定后统计量输出(no need) 绑定结果
  BottomUpClusterer bc(points, max_merge_thresh, min_clust, clusters_out, assignments_out);
  BaseFloat ans = bc.Cluster();
  return ans;
}


BaseFloat BottomUpClusterer::Cluster() {

  KALDI_VLOG(2) << "Initializing cluster assignments.";
  InitializeAssignments();
  # clusters_ 保存统计量
  # assignment 保存绑定信息, 初始化对应每个叶子节点
  # 317   clusters_->resize(npoints_);
  # 318   assignments_->resize(npoints_);
  # 319   for (int32 i = 0; i < npoints_; i++) {  // initialize as 1-1 mapping.
  # 320     (*clusters_)[i] = points_[i]->Copy();
  # 321     (*assignments_)[i] = i;
  # 322   }
  # 323 }


  KALDI_VLOG(2) << "Setting initial distances.";
  SetInitialDistances();
  # 计算两个统计量的区别, 区别较小的 加入到queue_队列中, 等待被绑定
  # 326   for (int32 i = 0; i < npoints_; i++) {
  # 327     for (int32 j = 0; j < i; j++) {
  # 328       BaseFloat dist = (*clusters_)[i]->Distance(*((*clusters_)[j]));
  # 329       dist_vec_[(i * (i - 1)) / 2 + j] = dist;
  # 330       if (dist <= max_merge_thresh_)
  # 331         queue_.push(std::make_pair(dist, std::make_pair(static_cast<uint_smaller>(i),
  # 332             static_cast<uint_smaller>(j))));
  # 333     }
  # 334   }
  # 335 }

  KALDI_VLOG(2) << "Clustering...";
  while (nclusters_ > min_clust_ && !queue_.empty()) {
    std::pair<BaseFloat, std::pair<uint_smaller, uint_smaller> > pr = queue_.top();
    BaseFloat dist = pr.first;
    int32 i = (int32) pr.second.first, j = (int32) pr.second.second;
    queue_.pop();
    # 绑定两个叶子节点
    # 将assignment 设置为靠后的节点index 实现绑定两个叶子节点
    # 并将绑定的统计量增加到 目标节点上, 删除被绑定节点统计量
    # 更新与目标相关的统计量距离
    if (CanMerge(i, j, dist)) MergeClusters(i, j);
  }
  KALDI_VLOG(2) << "Renumbering clusters to contiguous numbers.";
  # 更新assignment, 某个叶节点 绑定到目标节点上.  [1, 2, 3, 4, 5, 6] ---> [1, 2, 6, 6, 5, 6]
  # 完成绑定， 并且具有对应的统计量
  Renumber();
  return ans_;
}

# 判断是否能够合并了两个叶子节点
bool BottomUpClusterer::CanMerge(int32 i, int32 j, BaseFloat dist) {
  KALDI_ASSERT(i != j && i < npoints_ && j < npoints_);
  if ((*clusters_)[i] == NULL || (*clusters_)[j] == NULL)
    return false;
  BaseFloat cached_dist = dist_vec_[(i * (i - 1)) / 2 + j];
  return (std::fabs(cached_dist - dist) <= 1.0e-05 * std::fabs(dist));
}


# MergeCluster(i, j)
# 合并两个叶子节点, 将其中一个叶子节点j的统计量传递给目标节点i， 删除节点j的统计量， 并设置j的assignment = i.
# 更新叶子节点距离.

void BottomUpClusterer::MergeClusters(int32 i, int32 j) {
  KALDI_ASSERT(i != j && i < npoints_ && j < npoints_);
  (*clusters_)[i]->Add(*((*clusters_)[j]));
  delete (*clusters_)[j];
  (*clusters_)[j] = NULL;
  // note that we may have to follow the chain within "assignment_" to get
  // final assignments.
  (*assignments_)[j] = i;
  // subtract negated objective function change, i.e. add objective function
  // change.
  ans_ -= dist_vec_[(i * (i - 1)) / 2 + j];
  nclusters_--;
  // Now update "distances".
  for (int32 k = 0; k < npoints_; k++) {
    if (k != i && (*clusters_)[k] != NULL) {
      if (k < i)
        SetDistance(i, k);  // SetDistance requires k < i.
      else
        SetDistance(k, i);
    }
  }
}


// * classes
  
//   EventMap EventType
//   GaussCluterable
//   BuildTreeStatsType stats;  // vectorized form.
  
//   音素聚类 
//   Node
//   TreeClusterer
//   构造函数 以及 DoSplit  以及 聚类信息.
 
 
