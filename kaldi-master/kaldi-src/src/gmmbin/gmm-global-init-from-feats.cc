// gmmbin/gmm-global-init-from-feats.cc

// Copyright 2013   Johns Hopkins University (author: Daniel Povey)

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
#include "gmm/model-common.h"
#include "gmm/full-gmm.h"
#include "gmm/diag-gmm.h"
#include "gmm/mle-full-gmm.h"

namespace kaldi {

// 初始化GMM参数
// 1 方差 设置为全局方差
// 2 means 设置为随机选择的 一帧MFCC.
// We initialize the GMM parameters by setting the variance to the global
// variance of the features, and the means to distinct randomly chosen frames.
void InitGmmFromRandomFrames(const Matrix<BaseFloat> &feats, DiagGmm *gmm) {

  //高斯总数
  //帧总数
  //维度
  int32
      num_gauss = gmm->NumGauss(),
      num_frames = feats.NumRows(),
      dim = feats.NumCols();
  KALDI_ASSERT(num_frames >= 10 * num_gauss && "Too few frames to train on");
  Vector<double> mean(dim), var(dim);

  //计算 mean 和 vars
  for (int32 i = 0; i < num_frames; i++) {
      // += xi
      // += xi^2
    mean.AddVec(1.0 / num_frames, feats.Row(i));
    var.AddVec2(1.0 / num_frames, feats.Row(i));
  }
  // var 1 mean^2 (方差因为是对角,所以不计算协方差)
  var.AddVec2(-1.0, mean);



  if (var.Max() <= 0.0)
    KALDI_ERR << "Features do not have positive variance " << var;
  
  DiagGmmNormal gmm_normal(*gmm);

  std::set<int32> used_frames;
  // 每个高斯分量 设置参数
  for (int32 g = 0; g < num_gauss; g++) {
    //随机帧
    int32 random_frame = RandInt(0, num_frames - 1);
    while (used_frames.count(random_frame) != 0)
      random_frame = RandInt(0, num_frames - 1);

    // cache 随机帧, 以后不在选取同一帧作为某个分量的均值
    used_frames.insert(random_frame);
    //分量权重, 设置为平均
    gmm_normal.weights_(g) = 1.0 / num_gauss;
    // mean 直接拷贝
    gmm_normal.means_.Row(g).CopyFromVec(feats.Row(random_frame));
    // variance 方差(因为是对角的, 所以不计算协方差) 直接cp
    gmm_normal.vars_.Row(g).CopyFromVec(var);
  }

  gmm->CopyFromNormal(gmm_normal);
  // 计算响应度
  gmm->ComputeGconsts();
}

void TrainOneIter(const Matrix<BaseFloat> &feats,
                  const MleDiagGmmOptions &gmm_opts,
                  int32 iter,
                  int32 num_threads,
                  DiagGmm *gmm) {

  // 累计统计量
  AccumDiagGmm gmm_acc(*gmm, kGmmAll);

  Vector<BaseFloat> frame_weights(feats.NumRows(), kUndefined);
  frame_weights.Set(1.0);

  double tot_like;
  // 多线程计算 logGauss(xi)
  tot_like = gmm_acc.AccumulateFromDiagMultiThreaded(*gmm, feats, frame_weights,
                                                     num_threads);

  // 全部帧的 likelihood
  KALDI_LOG << "Likelihood per frame on iteration " << iter
            << " was " << (tot_like / feats.NumRows()) << " over "
            << feats.NumRows() << " frames.";
  
  BaseFloat objf_change, count;
  // 更新公式
  MleDiagGmmUpdate(gmm_opts, gmm_acc, kGmmAll, gmm, &objf_change, &count);

  KALDI_LOG << "Objective-function change on iteration " << iter << " was "
            << (objf_change / count) << " over " << count << " frames.";
}

} // namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "This program initializes a single diagonal GMM and does multiple iterations of\n"
        "training from features stored in memory.\n"
        "Usage:  gmm-global-init-from-feats [options] <feature-rspecifier> <model-out>\n"
        "e.g.: gmm-global-init-from-feats scp:train.scp 1.mdl\n";

    ParseOptions po(usage);
    MleDiagGmmOptions gmm_opts;
    
    bool binary = true;
    int32 num_gauss = 100;
    int32 num_gauss_init = 0;
    int32 num_iters = 50;
    int32 num_frames = 200000;
    int32 srand_seed = 0;
    int32 num_threads = 4;
    
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("num-gauss", &num_gauss, "Number of Gaussians in the model");
    po.Register("num-gauss-init", &num_gauss_init, "Number of Gaussians in "
                "the model initially (if nonzero and less than num_gauss, "
                "we'll do mixture splitting)");
    po.Register("num-iters", &num_iters, "Number of iterations of training");
    po.Register("num-frames", &num_frames, "Number of feature vectors to store in "
                "memory and train on (randomly chosen from the input features)");
    po.Register("srand", &srand_seed, "Seed for random number generator ");
    po.Register("num-threads", &num_threads, "Number of threads used for "
                "statistics accumulation");
                
    gmm_opts.Register(&po);

    po.Read(argc, argv);

    srand(srand_seed);    
    
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string feature_rspecifier = po.GetArg(1),
        model_wxfilename = po.GetArg(2);
    
    // 内存中保存 num_frames(700 000) 帧的数据, 进行训练
    Matrix<BaseFloat> feats;

    // feats 经过 cmvn splice(3 3) transform   [frames X 91]
    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);


    KALDI_ASSERT(num_frames > 0);
    
    int64 num_read = 0, dim = 0;

    KALDI_LOG << "Reading features (will keep " << num_frames << " frames.)";
    
    for (; !feature_reader.Done(); feature_reader.Next()) {
      const Matrix<BaseFloat>  &this_feats = feature_reader.Value();
      for (int32 t = 0; t < this_feats.NumRows(); t++) {
        num_read++;
        if (dim == 0) {
          dim = this_feats.NumCols();
          feats.Resize(num_frames, dim);
        } else if (this_feats.NumCols() != dim) {
          KALDI_ERR << "Features have inconsistent dims "
                    << this_feats.NumCols() << " vs. " << dim
                    << " (current utt is) " << feature_reader.Key();
        }
        if (num_read <= num_frames) {
          feats.Row(num_read - 1).CopyFromVec(this_feats.Row(t));
        } else {
          // 如果数据够多, 随机 num_frames 个即可
          BaseFloat keep_prob = num_frames / static_cast<BaseFloat>(num_read);
          if (WithProb(keep_prob)) { // With probability "keep_prob"
            feats.Row(RandInt(0, num_frames - 1)).CopyFromVec(this_feats.Row(t));
          }
        }
      }
    }

    if (num_read < num_frames) {
      KALDI_WARN << "Number of frames read " << num_read << " was less than "
                 << "target number " << num_frames << ", using all we read.";
      feats.Resize(num_read, dim, kCopyData);
    } else {
        //保留数据(700000) 占总读取frames 百分比
      BaseFloat percent = num_frames * 100.0 / num_read;
      KALDI_LOG << "Kept " << num_frames << " out of " << num_read
                << " input frames = " << percent << "%.";
    }

    // 高斯分量数
    if (num_gauss_init <= 0 || num_gauss_init > num_gauss)
      num_gauss_init = num_gauss;
    
    //构造 diag GMM -- UBM模型, 初始化为num_gauss_init个分量 [num_gauss & feat-dim]
    DiagGmm gmm(num_gauss_init, dim);
    
    KALDI_LOG << "Initializing GMM means from random frames to "
              << num_gauss_init << " Gaussians.";
    //初始化 gmm的 mean variances weight权重
    InitGmmFromRandomFrames(feats, &gmm);

    // we'll increase the #Gaussians by splitting,
    // till halfway through training.
    int32 cur_num_gauss = num_gauss_init,
        gauss_inc = (num_gauss - num_gauss_init) / (num_iters / 2);
        
    // 迭代训练 GMM gconst_  mean_invvars_ inv_vars_
    for (int32 iter = 0; iter < num_iters; iter++) {
      TrainOneIter(feats, gmm_opts, iter, num_threads, &gmm);

      // 逐渐划分混合数
      int32 next_num_gauss = std::min(num_gauss, cur_num_gauss + gauss_inc);
      if (next_num_gauss > gmm.NumGauss()) {
        KALDI_LOG << "Splitting to " << next_num_gauss << " Gaussians.";
        gmm.Split(next_num_gauss, 0.1);
        cur_num_gauss = next_num_gauss;
      }
    }

    WriteKaldiObject(gmm, model_wxfilename, binary);
    KALDI_LOG << "Wrote model to " << model_wxfilename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
