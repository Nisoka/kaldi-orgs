// ivectorbin/ivector-plda-scoring.cc

// Copyright 2013  Daniel Povey

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "ivector/plda.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  typedef std::string string;
  try {
    const char *usage =
        "Computes log-likelihood ratios for trials using PLDA model\n"
        "Note: the 'trials-file' has lines of the form\n"
        "<key1> <key2>\n"
        "and the output will have the form\n"
        "<key1> <key2> [<dot-product>]\n"
        "(if either key could not be found, the dot-product field in the output\n"
        "will be absent, and this program will print a warning)\n"
        "For training examples, the input is the iVectors averaged over speakers;\n"
        "a separate archive containing the number of utterances per speaker may be\n"
        "optionally supplied using the --num-utts option; this affects the PLDA\n"
        "scoring (if not supplied, it defaults to 1 per speaker).\n"
        "\n"
        "Usage: ivector-plda-scoring <plda> <train-ivector-rspecifier> <test-ivector-rspecifier>\n"
        " <trials-rxfilename> <scores-wxfilename>\n"
        "\n"
        "e.g.: ivector-plda-scoring --num-utts=ark:exp/train/num_utts.ark plda "
        "ark:exp/train/spk_ivectors.ark ark:exp/test/ivectors.ark trials scores\n"
        "See also: ivector-compute-dot-products, ivector-compute-plda\n";

    ParseOptions po(usage);

    std::string num_utts_rspecifier;

    PldaConfig plda_config;
    plda_config.Register(&po);
    po.Register("num-utts", &num_utts_rspecifier, "Table to read the number of "
                "utterances per speaker, e.g. ark:num_utts.ark\n");

    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string
        plda_rxfilename = po.GetArg(1),
        train_ivector_rspecifier = po.GetArg(2),
        test_ivector_rspecifier = po.GetArg(3),
        trials_rxfilename = po.GetArg(4),
        scores_wxfilename = po.GetArg(5);

    //  diagnostics:
    double tot_test_renorm_scale = 0.0, tot_train_renorm_scale = 0.0;
    int64 num_train_ivectors = 0, num_train_errs = 0, num_test_ivectors = 0;

    int64 num_trials_done = 0, num_trials_err = 0;

    Plda plda;
    ReadKaldiObject(plda_rxfilename, &plda);

    int32 dim = plda.Dim();

    SequentialBaseFloatVectorReader train_ivector_reader(train_ivector_rspecifier);
    SequentialBaseFloatVectorReader test_ivector_reader(test_ivector_rspecifier);
    RandomAccessInt32Reader num_utts_reader(num_utts_rspecifier);

    typedef unordered_map<string, Vector<BaseFloat>*, StringHasher> HashType;

    // These hashes will contain the iVectors in the PLDA subspace
    // (that makes the within-class variance unit and diagonalizes the
    // between-class covariance).  They will also possibly be length-normalized,
    // depending on the config.
    HashType train_ivectors, test_ivectors;



    // 读取 注册人 mean ivector
    KALDI_LOG << "Reading train iVectors";
    // 实际是注册人 enrollment 的 spker mean ivector , 所以一个人只能读取一次
    for (; !train_ivector_reader.Done(); train_ivector_reader.Next()) {
      std::string spk = train_ivector_reader.Key();
      if (train_ivectors.count(spk) != 0) {
        KALDI_ERR << "Duplicate training iVector found for speaker " << spk;
      }
      const Vector<BaseFloat> &ivector = train_ivector_reader.Value();
      int32 num_examples;
      if (!num_utts_rspecifier.empty()) {
        // 一个spker 只具有一个 mean ivector
        if (!num_utts_reader.HasKey(spk)) {
          KALDI_WARN << "Number of utterances not given for speaker " << spk;
          num_train_errs++;
          continue;
        }
        num_examples = num_utts_reader.Value(spk);
      } else {
        num_examples = 1;
      }
      Vector<BaseFloat> *transformed_ivector = new Vector<BaseFloat>(dim);

      // spker mean ivector 经过 plda 预处理变换
      tot_train_renorm_scale += plda.TransformIvector(plda_config, ivector,
                                                      num_examples,
                                                      transformed_ivector);
      train_ivectors[spk] = transformed_ivector;
      num_train_ivectors++;
    }



    KALDI_LOG << "Read " << num_train_ivectors << " training iVectors, "
              << "errors on " << num_train_errs;
    if (num_train_ivectors == 0)
      KALDI_ERR << "No training iVectors present.";
    KALDI_LOG << "Average renormalization scale on training iVectors was "
              << (tot_train_renorm_scale / num_train_ivectors);

    // test utt ivector
    KALDI_LOG << "Reading test iVectors";

    for (; !test_ivector_reader.Done(); test_ivector_reader.Next()) {
      std::string utt = test_ivector_reader.Key();
      if (test_ivectors.count(utt) != 0) {
        KALDI_ERR << "Duplicate test iVector found for utterance " << utt;
      }
      const Vector<BaseFloat> &ivector = test_ivector_reader.Value();
      int32 num_examples = 1; // this value is always used for test (affects the
                              // length normalization in the TransformIvector
                              // function).
      Vector<BaseFloat> *transformed_ivector = new Vector<BaseFloat>(dim);

      // test utt ivector 经过 plda 预处理变换
      tot_test_renorm_scale += plda.TransformIvector(plda_config, ivector,
                                                     num_examples,
                                                     transformed_ivector);
      test_ivectors[utt] = transformed_ivector;
      num_test_ivectors++;
    }

    KALDI_LOG << "Read " << num_test_ivectors << " test iVectors.";
    if (num_test_ivectors == 0)
      KALDI_ERR << "No test iVectors present.";
    KALDI_LOG << "Average renormalization scale on test iVectors was "
              << (tot_test_renorm_scale / num_test_ivectors);



    // 根据trials文件(label文件)
    // target  testutt  ifTrue
    // -----------------------
    // spker  utt   target
    // nonspker utt nontarget

    // 下面就是计算 utt 和 spk
    // 计算得分, 写入 score文件
    Input ki(trials_rxfilename);
    bool binary = false;
    Output ko(scores_wxfilename, binary);

    double sum = 0.0, sumsq = 0.0;
    std::string line;

    // 测试trials中的entry
    // entry为
    //     targetSpker utt  target
    // 对一个utt 测试是否与spker 的mean ivector 是相近的, 越相近得分越高
    // target 表示 utt 确实为 spker的语音
    //eg
    //    targetSpker     utt1  target
    //    nontargetSpker  utt1  nontarget
    //    targetSpker     utt2  target
    //    nontargetSpker  utt2  nontarget

    while (std::getline(ki.Stream(), line)) {
      std::vector<std::string> fields;
      SplitStringToVector(line, " \t\n\r", true, &fields);
      if (fields.size() != 2) {
        KALDI_ERR << "Bad line " << (num_trials_done + num_trials_err)
                  << "in input (expected two fields: key1 key2): " << line;
      }
      //key1 spker
      //key2 utt-id
      std::string key1 = fields[0], key2 = fields[1];
      if (train_ivectors.count(key1) == 0) {
        KALDI_WARN << "Key " << key1 << " not present in training iVectors.";
        num_trials_err++;
        continue;
      }
      if (test_ivectors.count(key2) == 0) {
        KALDI_WARN << "Key " << key2 << " not present in test iVectors.";
        num_trials_err++;
        continue;
      }
      // spker mean ivector
      // test utt ivector
      const Vector<BaseFloat> *train_ivector = train_ivectors[key1],
          *test_ivector = test_ivectors[key2];

      Vector<double> train_ivector_dbl(*train_ivector),
          test_ivector_dbl(*test_ivector);

      int32 num_train_examples;
      if (!num_utts_rspecifier.empty()) {
        // we already checked that it has this key.
        num_train_examples = num_utts_reader.Value(key1);
      } else {
        num_train_examples = 1;
      }


      // !!!!!　计算似然得分
      BaseFloat score = plda.LogLikelihoodRatio(train_ivector_dbl,
                                                num_train_examples,
                                                test_ivector_dbl);
      sum += score;
      sumsq += score * score;
      num_trials_done++;
      // spker 　　　utt 　　　score
      ko.Stream() << key1 << ' ' << key2 << ' ' << score << std::endl;
      // 可以通过 对比 trials 文件
      // spker      utt      target
      // 计算一个门限, 来得到eer.
    }




    for (HashType::iterator iter = train_ivectors.begin();
         iter != train_ivectors.end(); ++iter)
      delete iter->second;
    for (HashType::iterator iter = test_ivectors.begin();
         iter != test_ivectors.end(); ++iter)
      delete iter->second;


    if (num_trials_done != 0) {
      BaseFloat mean = sum / num_trials_done, scatter = sumsq / num_trials_done,
          variance = scatter - mean * mean, stddev = sqrt(variance);
      KALDI_LOG << "Mean score was " << mean << ", standard deviation was "
                << stddev;
    }
    KALDI_LOG << "Processed " << num_trials_done << " trials, " << num_trials_err
              << " had errors.";
    return (num_trials_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
