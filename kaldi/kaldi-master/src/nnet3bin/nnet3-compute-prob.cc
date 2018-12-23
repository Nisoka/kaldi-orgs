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

// 计算并打印 给定数据的每帧对数概率. 
// 输入数据是 merged-egs
int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Computes and prints in logging messages the average log-prob per frame of\n"
        "the given data with an nnet3 neural net.  The input of this is the output of\n"
        "e.g. nnet3-get-egs | nnet3-merge-egs.\n"
        "\n"
        "Usage:  nnet3-compute-prob [options] <raw-model-in> <training-examples-in>\n"
        "e.g.: nnet3-compute-prob 0.raw ark:valid.egs\n";


    bool batchnorm_test_mode = true, dropout_test_mode = true,
        collapse_model = true;

    // 这个程序不支持实用GPU, 因为是计算诊断概率的, 你可以实用小量数据进行计算,这样CPU 也可以在合理时间完成.
    // This program doesn't support using a GPU, because these probabilities are
    // used for diagnostics, and you can just compute them with a small enough
    // amount of data that a CPU can do it within reasonable time.


    // just constructor
    // 包含有
    // 1 NnetOptimizeOptions,   优化调试选项
    // 2 NnetComputeOptions,    计算选项(简单)
    // 3 CachingOptimizingCompilerOptions   编译优化选项 简单
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

    // 主要设置一些配置
    // 1 编译配置
    // 2 优化配置等
    NnetComputeProb prob_computer(opts, nnet);

    SequentialNnetExampleReader example_reader(examples_rspecifier);

    //exmple_reader 保存的是 minibatch merged NnetExample
    //每个merged NnetExample 是64 个NnetExample(n=0)合并得到的.
    //所以如下 prob_computer.Comput() 计算的是一个minibatch的 NnetExample.
    for (; !example_reader.Done(); example_reader.Next())
      prob_computer.Compute(example_reader.Value());

    bool ok = prob_computer.PrintTotalStats();

    return (ok ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
