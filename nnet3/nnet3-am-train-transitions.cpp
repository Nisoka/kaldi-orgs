// nnet3bin/nnet3-am-train-transitions.cc

// Copyright 2012-2015  Johns Hopkins University (author:  Daniel Povey)

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
#include "hmm/transition-model.h"
#include "nnet3/am-nnet-simple.h"
#include "tree/context-dep.h"

namespace kaldi {
namespace nnet3 {
void SetPriors(const TransitionModel &tmodel,
               const Vector<double> &transition_accs,
               double prior_floor,
               AmNnetSimple *am_nnet) {
  
  KALDI_ASSERT(tmodel.NumPdfs() == am_nnet->NumPdfs());

  // pdf_cnt 为长度的 vector, 用来保存统计数据
  Vector<BaseFloat> pdf_counts(tmodel.NumPdfs());
  KALDI_ASSERT(transition_accs(0) == 0.0); // transition-id = 0, 没有统计量.

  // tid_acc - 每个tid出现总数
  // 直接转化为 pdf_counts 统计总数
  for (int32 tid = 1; tid < transition_accs.Dim(); tid++) {
    int32 pdf = tmodel.TransitionIdToPdf(tid);
    pdf_counts(pdf) += transition_accs(tid);
  }
  
  BaseFloat sum = pdf_counts.Sum();
  KALDI_ASSERT(sum != 0.0);
  KALDI_ASSERT(prior_floor > 0.0 && prior_floor < 1.0);
  // 正则化
  pdf_counts.Scale(1.0 / sum);
  // 使用floor化
  pdf_counts.ApplyFloor(prior_floor);
  // 重新正则化 这时候 pdf_counts 保存的实际上就是 pdf的先验概率(统计总数得到的统计概率)
  pdf_counts.Scale(1.0 / pdf_counts.Sum()); // normalize again.
  // 设置am-nnet 的先验概率 pdf_counts.
  am_nnet->SetPriors(pdf_counts);
}               


} // namespace nnet3
} // namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;

    const char *usage =
        "Train the transition probabilities of an nnet3 neural network acoustic model\n"
        "\n"
        "Usage:  nnet3-am-train-transitions [options] <nnet-in> <alignments-rspecifier> <nnet-out>\n"
        "e.g.:\n"
        " nnet3-am-train-transitions 1.nnet \"ark:gunzip -c ali.*.gz|\" 2.nnet\n";
    
    bool binary_write = true;
    bool set_priors = true; // Also set the per-pdf priors in the model.
    // 原本设置为 1e-8, 但是以前有一个问题, 有一个pdf-id在训练时不可见, 但是识别时经常出现,
    // 所以设置为 5.0e-06, 似乎是最小的可见pdf-id的先验概率.
    BaseFloat prior_floor = 5.0e-06; // The default was previously 1e-8, but
                                     // once we had problems with a pdf-id that
                                     // was not being seen in training, being
                                     // recognized all the time.  This value
                                     // seemed to be the smallest prior of the
                                     // "seen" pdf-ids in one run.

    // 最大似然估计转移更新配置
    MleTransitionUpdateConfig transition_update_config;
    MleTransitionUpdateConfig(BaseFloat floor = 0.01,
                              BaseFloat mincount = 5.0,
                              bool share_for_pdfs = false):
        floor(floor), mincount(mincount), share_for_pdfs(share_for_pdfs) {}
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("set-priors", &set_priors, "If true, also set priors in neural "
                "net (we divide by these in test time)");
    po.Register("prior-floor", &prior_floor, "When setting priors, floor for "
                "priors");
    transition_update_config.Register(&po);
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string nnet_rxfilename = po.GetArg(1),
        ali_rspecifier = po.GetArg(2),
        nnet_wxfilename = po.GetArg(3);
    
    TransitionModel trans_model;
    AmNnetSimple am_nnet;

    class AmNnetSimple {
     public:
      AmNnetSimple() { }

      AmNnetSimple(const AmNnetSimple &other):
          nnet_(other.nnet_),
          priors_(other.priors_),
          left_context_(other.left_context_),
          right_context_(other.right_context_) { }

      explicit AmNnetSimple(const Nnet &nnet):
          nnet_(nnet) { SetContext(); }

      // ............
     private:
      const AmNnetSimple &operator = (const AmNnetSimple &other); // Disallow.
      Nnet nnet_;                                                 // nnet3 结构
      Vector<BaseFloat> priors_;                                  // 保存 pdf-id 的统计出现次数得到的 先验概率
      // The following variables are derived; they are re-computed
      // when we read the network or when it is changed.
      int32 left_context_;
      int32 right_context_;
    };

    
    {
      bool binary_read;
      Input ki(nnet_rxfilename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_nnet.Read(ki.Stream(), binary_read);
    }
    
    Vector<double> transition_accs;
    // resize vector 大小为 tid_cnt + 1 统计tid 概率.
    trans_model.InitStats(&transition_accs);
    // void InitStats(Vector<double> *stats) const { stats->Resize(NumTransitionIds()+1); }
    
    int32 num_done = 0;
    SequentialInt32VectorReader ali_reader(ali_rspecifier);
    // foreach utt_ali
    for (; ! ali_reader.Done(); ali_reader.Next()) {
      const std::vector<int32> alignment(ali_reader.Value());
      // foreach frames
      for (size_t i = 0; i < alignment.size(); i++) {
        int32 tid = alignment[i];
        BaseFloat weight = 1.0;
        trans_model.Accumulate(weight, tid, &transition_accs);

        // 统计tid 总数.
        void Accumulate(BaseFloat prob, int32 trans_id, Vector<double> *stats) const {
          KALDI_ASSERT(trans_id <= NumTransitionIds());
          (*stats)(trans_id) += prob;
          // This is trivial and doesn't require class members, but leaves us more open
          // to design changes than doing it manually.
        }

        
      }
      num_done++;
    }
    KALDI_LOG << "Accumulated transition stats from " << num_done
              << " utterances.";

    {
      BaseFloat objf_impr, count;
      trans_model.MleUpdate(transition_accs, transition_update_config,  &objf_impr, &count);
      KALDI_LOG << "Transition model update: average " << (objf_impr/count)
                << " log-like improvement per frame over " << count
                << " frames.";

      void TransitionModel::MleUpdate(const Vector<double> &stats,
                                      const MleTransitionUpdateConfig &cfg,
                                      BaseFloat *objf_impr_out,
                                      BaseFloat *count_out) {
        // false
        if (cfg.share_for_pdfs) {
          MleUpdateShared(stats, cfg, objf_impr_out, count_out);
          return;
        }
        
        BaseFloat count_sum = 0.0, objf_impr_sum = 0.0;
        int32 num_skipped = 0, num_floored = 0;
        KALDI_ASSERT(stats.Dim() == NumTransitionIds()+1);

        // foreach trans-state!!!
        for (int32 tstate = 1; tstate <= NumTransitionStates(); tstate++) {
          // get trans-state's trans-indexes
          int32 n = NumTransitionIndices(tstate);
          
          KALDI_ASSERT(n>=1);
          if (n > 1) {  // no point updating if only one transition...

            // 对每个tid的统计量 分配到 表示 <trans-state, trans-index> 的count中
            Vector<double> counts(n);
            for (int32 tidx = 0; tidx < n; tidx++) {
              int32 tid = PairToTransitionId(tstate, tidx);
              counts(tidx) = stats(tid);
            }
            
            // 该状态 t-state 的所有转移总数.
            double tstate_tot = counts.Sum();
            count_sum += tstate_tot;
            
            if (tstate_tot < cfg.mincount) { num_skipped++; }
            else {
              // n trans-indexes 总数. 即 对每个 <t-state, t-index> 进行概率统计.
              Vector<BaseFloat> old_probs(n), new_probs(n);

              // 获得原始 tid 转移概率
              for (int32 tidx = 0; tidx < n; tidx++) {
                int32 tid = PairToTransitionId(tstate, tidx);
                old_probs(tidx) = new_probs(tidx) = GetTransitionProb(tid);
              }

              // 新tid 转移概率 = t-state-index总数 / t-state转移总数
              for (int32 tidx = 0; tidx < n; tidx++)
                new_probs(tidx) = counts(tidx) / tstate_tot;

              // flooring -- 保证元素 > florr
              // 正则化   -- 加和为1.
              for (int32 i = 0; i < 3; i++) {  // keep flooring+renormalizing for 3 times..
                new_probs.Scale(1.0 / new_probs.Sum());
                for (int32 tidx = 0; tidx < n; tidx++)
                  new_probs(tidx) = std::max(new_probs(tidx), cfg.floor);
              }
              
              // Compute objf change
              for (int32 tidx = 0; tidx < n; tidx++) {
                
                if (new_probs(tidx) == cfg.floor)
                  num_floored++;
                
                double objf_change = counts(tidx) * (Log(new_probs(tidx))
                                                     - Log(old_probs(tidx)));
                objf_impr_sum += objf_change;
              }

              // 更新 log_probs_(tid) 
              // Commit updated values.
              for (int32 tidx = 0; tidx < n; tidx++) {
                int32 tid = PairToTransitionId(tstate, tidx);
                log_probs_(tid) = Log(new_probs(tidx));
                
                if (log_probs_(tid) - log_probs_(tid) != 0.0)
                  KALDI_ERR << "Log probs is inf or NaN: error in update or bad stats?";
              }
            }
          }
        }
        KALDI_LOG << "TransitionModel::Update, objf change is "
                  << (objf_impr_sum / count_sum) << " per frame over " << count_sum
                  << " frames. ";
        KALDI_LOG <<  num_floored << " probabilities floored, " << num_skipped
                  << " out of " << NumTransitionStates() << " transition-states "
            "skipped due to insuffient data (it is normal to have some skipped.)";
        
        if (objf_impr_out)
          *objf_impr_out = objf_impr_sum;
        if (count_out)
          *count_out = count_sum;
        
        ComputeDerivedOfProbs();
        void TransitionModel::ComputeDerivedOfProbs() {
          // 非自环转移log概率 resize 为 state_cnt+1
          non_self_loop_log_probs_.Resize(NumTransitionStates()+1);
          // this array indexed by transition-state with nothing in zeroth element.

          // foreach t-state
          for (int32 tstate = 1; tstate <= NumTransitionStates(); tstate++) {
            // get self-loop tid
            int32 tid = SelfLoopOf(tstate);


            // tid==0 说明不正常的tid 设置 non_self_loop_log_probs_(tstate) = 0.0
            if (tid == 0) {  // no self-loop
              non_self_loop_log_probs_(tstate) = 0.0;  // log(1.0)
            }
            // 正常 self-loop tid
            else {
              BaseFloat
                  // tid为自环, 获得自环的转移概率, 剩下的就是非自环的转移概率.
                  self_loop_prob = Exp(GetTransitionLogProb(tid)),
                  non_self_loop_prob = 1.0 - self_loop_prob;
              
              if (non_self_loop_prob <= 0.0) {
                KALDI_WARN << "ComputeDerivedOfProbs(): non-self-loop prob is " << non_self_loop_prob;
                non_self_loop_prob = 1.0e-10;  // just so we can continue...
              }
              // 设置  tstate 所有非自环转移的概率和的 log.
              non_self_loop_log_probs_(tstate) = Log(non_self_loop_prob);  // will be negative.
            }
          }
        }

      }
    }

    if (set_priors) {
      KALDI_LOG << "Setting priors of pdfs in the model.";
      SetPriors(trans_model, transition_accs, prior_floor, &am_nnet);
    }
    
    {
      Output ko(nnet_wxfilename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
      am_nnet.Write(ko.Stream(), binary_write);
    }
    KALDI_LOG << "Trained transitions of neural network model and wrote it to "
              << nnet_wxfilename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
