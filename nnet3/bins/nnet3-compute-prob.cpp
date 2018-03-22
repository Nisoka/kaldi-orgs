// nnet3bin/nnet3-compute-prob.cc

// Copyright 2015  Johns Hopkins University (author: Daniel Povey)

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
#include "nnet3/nnet-diagnostics.h"
#include "nnet3/nnet-utils.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    // 计算 提供给nnet3 网络的数据的平均 对数概率.
    // 输入是 是 nnet3-merge-egs的输出结果.
    const char *usage =
        "Computes and prints in logging messages the average log-prob per frame of\n"
        "the given data with an nnet3 neural net.  The input of this is the output of\n"
        "e.g. nnet3-get-egs | nnet3-merge-egs.\n"
        "\n"
        
        "Usage:  nnet3-compute-prob [options] <raw-model-in> <training-examples-in>\n"
        "e.g.: nnet3-compute-prob 0.raw ark:valid.egs\n";


    bool
        batchnorm_test_mode = true,
        dropout_test_mode = true,
        collapse_model = true;

    // 这个程序不支持 GPU, 因为这些概率被用于验证 诊断?
    // 你可以使用少量数据计算他们, CPU对这样的计算就足够了.
    // This program doesn't support using a GPU, because these probabilities are
    // used for diagnostics, and you can just compute them with a small enough
    // amount of data that a CPU can do it within reasonable time.

    NnetComputeProbOptions opts;

    ParseOptions po(usage);

    po.Register("batchnorm-test-mode", &batchnorm_test_mode,
                "If true, set test-mode to true on any BatchNormComponents.");
    po.Register("dropout-test-mode", &dropout_test_mode,
                "If true, set test-mode to true on any DropoutComponents and "
                "DropoutMaskComponents.");
    po.Register("collapse-model", &collapse_model,
                "If true, collapse model to the extent possible before "
                "using it (for efficiency).");

    // 命令行参数中没有参数提供给 - NnetComputeProbOptions
    opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string raw_nnet_rxfilename = po.GetArg(1),
        examples_rspecifier = po.GetArg(2);

    Nnet nnet;
    ReadKaldiObject(raw_nnet_rxfilename, &nnet);



    
    if (batchnorm_test_mode)
      SetBatchnormTestMode(true, &nnet);

    if (dropout_test_mode)
      SetDropoutTestMode(true, &nnet);

    if (collapse_model)
      CollapseModel(CollapseModelConfig(), &nnet);







    // ==============================================
    NnetComputeProb prob_computer(opts, nnet);

    SequentialNnetExampleReader example_reader(examples_rspecifier);

    for (; !example_reader.Done(); example_reader.Next())
      prob_computer.Compute(example_reader.Value());

    bool ok = prob_computer.PrintTotalStats();

    return (ok ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}








