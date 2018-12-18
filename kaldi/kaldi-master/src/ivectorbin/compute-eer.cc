// ivectorbin/compute-eer.cc

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

namespace kaldi {

/**
   ComputeEer computes the Equal Error Rate (EER) for the given scores
   and returns it as a proportion beween 0 and 1.
   If we set the threshold at x, then the target error-rate is the
   proportion of target_scores below x; and the non-target error-rate
   is the proportion of non-target scores above x.  We seek a
   threshold x for which these error rates are the same; this
   error rate is the EER.

   We compute this by iterating over the positions in target_scores: 0, 1, 2,
   and so on, and for each position consider whether the cutoff could be here.
   For each of these position we compute the corresponding position in
   nontarget_scores where the cutoff would be if the EER were the same.
   For instance, if the vectors had the same length, this would be position
   length() - 1, length() - 2, and so on.  As soon as the value at that
   position in nontarget_scores at that position is less than the value from
   target_scores, we have our EER.

   In coding this we weren't particularly careful about edge cases or
   making sure whether it's actually n + 1 instead of n.
*/

BaseFloat ComputeEer(std::vector<BaseFloat> *target_scores,
                     std::vector<BaseFloat> *nontarget_scores,
                     BaseFloat *threshold) {
  KALDI_ASSERT(!target_scores->empty() && !nontarget_scores->empty());
  // 小 -> 大 排序得分
  // eg
  // target_scores 109 个得分
  // -9 ~ +100
  // nontarget_score  103个得分
  // -100 ~ +3
  // 经过排序, 得分高低得到了, 这样
  // target_scores 最靠前的几个utt 说明识别错了,(得分这么低, 还是别为target)
  // nontarget_scores 最靠后的几个utt 说明识别错了,(得分挺高的, 但是识别为非target)
  // 这样找相同比例的 错误率
  // 就可以通过 target_score 逐个向后, 然后在 nontarget_score 相同比率 向前
  // 发现在相同错误比率 下, 得分相同了, 那么说明 达到最小等错误率.
  std::sort(target_scores->begin(), target_scores->end());
  std::sort(nontarget_scores->begin(), nontarget_scores->end());
  
  size_t
      target_position = 0,
      target_size = target_scores->size();


  for (; target_position + 1 < target_size; target_position++) {
    ssize_t nontarget_size = nontarget_scores->size(),
        nontarget_n = nontarget_size * target_position * 1.0 / target_size,
        nontarget_position = nontarget_size - 1 - nontarget_n;

    if (nontarget_position  < 0)
      nontarget_position = 0;

    if ((*nontarget_scores)[nontarget_position] < (*target_scores)[target_position])
      break;
  }


  *threshold = (*target_scores)[target_position];
  BaseFloat eer = target_position * 1.0 / target_size;
  return eer;
}


}



int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Computes Equal Error Rate\n"
        "Input is a series of lines, each with two fields.\n"
        "The first field must be a numeric score, and the second\n"
        "either the string 'target' or 'nontarget'. \n"
        "The EER will be printed to the standard output.\n"
        "\n"
        "Usage: compute-eer <scores-in>\n"
        "e.g.: compute-eer -\n";
    
    ParseOptions po(usage);
    po.Read(argc, argv);
    
    if (po.NumArgs() != 1) {
      po.PrintUsage();
      exit(1);
    }

    std::string scores_rxfilename = po.GetArg(1);

    std::vector<BaseFloat> target_scores, nontarget_scores;
    Input ki(scores_rxfilename);
    
    std::string line;
    //score_is_target 文件内是 每个utt两个line
    //line1 表示target spker的得分,以及判定是否是target
    //line2 表示nontarget spker的得分,以及判定是否是nontarget
    //score target
    //score nontarget
    while (std::getline(ki.Stream(), line)) {
      std::vector<std::string> split_line;
      SplitStringToVector(line, " \t", true, &split_line);

      BaseFloat score;
      if (split_line.size() != 2) {
        KALDI_ERR << "Invalid input line (must have two fields): "
                  << line;
      }
      if (!ConvertStringToReal(split_line[0], &score)) {
        KALDI_ERR << "Invalid input line (first field must be float): "
                  << line;
      }
      //
      if (split_line[1] == "target")
        target_scores.push_back(score);
      else if (split_line[1] == "nontarget")
        nontarget_scores.push_back(score);
      else {
        KALDI_ERR << "Invalid input line (second field must be "
                  << "'target' or 'nontarget')";
      }
    }
    if (target_scores.empty() && nontarget_scores.empty())
      KALDI_ERR << "Empty input.";
    if (target_scores.empty())
      KALDI_ERR << "No target scores seen.";
    if (nontarget_scores.empty())
      KALDI_ERR << "No non-target scores seen.";

    BaseFloat threshold;
    BaseFloat eer = ComputeEer(&target_scores, &nontarget_scores, &threshold);

    KALDI_LOG << "Equal error rate is " << (100.0 * eer)
              << "%, at threshold " << threshold;

    std::cout.precision(4);
    std::cout << (100.0 * eer);
    
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
