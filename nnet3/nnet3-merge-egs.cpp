// nnet3bin/nnet3-merge-egs.cc

// Copyright 2012-2015  Johns Hopkins University (author:  Daniel Povey)
//                2014  Vimal Manohar

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
#include "nnet3/nnet-example.h"
#include "nnet3/nnet-example-utils.h"

namespace kaldi {
namespace nnet3 {
// returns the number of indexes/frames in the NnetIo with output name
// including string "output" as part of its name in the eg.
// e.g. output-0, output-xent
int32 NumOutputIndexes(const NnetExample &eg) {
  for (size_t i = 0; i < eg.io.size(); i++)
    if (eg.io[i].name.find("output") != std::string::npos)
      return eg.io[i].indexes.size();
  return 1;  // Suppress compiler warning.
}

}
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;


    // 也是主要做的copy操作, 但是copy时, 会将很多 nnetExample对象合并成为一个, 然后构建一个minibatch 包含一个单一NnetExample.???
    // --minibatch-size=1:64 
    const char *usage =
        "This copies nnet training examples from input to output, but while doing so it\n"
        "merges many NnetExample objects into one, forming a minibatch consisting of a\n"
        "single NnetExample.\n"
        "\n"
        "Usage:  nnet3-merge-egs [options] <egs-rspecifier> <egs-wspecifier>\n"
        "e.g.\n"
        "nnet3-merge-egs --minibatch-size=512 ark:1.egs ark:- | nnet3-train-simple ... \n"
        "See also nnet3-copy-egs\n";

    ParseOptions po(usage);

    ExampleMergingConfig merging_config;
    merging_config.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string
        examples_rspecifier = po.GetArg(1),
        examples_wspecifier = po.GetArg(2);

    merging_config.ComputeDerived();

    ExampleMergingConfig(const char *default_minibatch_size = "256"):
        compress(false),
        measure_output_frames("deprecated"),
        minibatch_size(default_minibatch_size),
        // 丢弃 特定minibatch ? deprecated-反对-false
        discard_partial_minibatches("deprecated") { }

    void Register(OptionsItf *po) {
      po->Register("compress", &compress, "If true, compress the output examples "
                   "(not recommended unless you are writing to disk)");
      po->Register("measure-output-frames", &measure_output_frames, "This "
                   "value will be ignored (included for back-compatibility)");
      po->Register("discard-partial-minibatches", &discard_partial_minibatches,
                   "This value will be ignored (included for back-compatibility)");
      // 上面传入的是 1:64
      po->Register("minibatch-size", &minibatch_size,

                   // 控制minibatch大小的字符串
                   "String controlling the minibatch size.  May be just an integer, "
                   "meaning a fixed minibatch size (e.g. --minibatch-size=128). "

                   // 可能是一个list, 描述minibatch, 怎么使用?
                   // 当输入结束时 所有的minibatches都会是最大的大小,然后可以增加一些小的大小.
                   // 可以为不同大小的eg 指定不同的minibatch大小.???
                   "May be a list of ranges and values, e.g. --minibatch-size=32,64 "
                   "or --minibatch-size=16:32,64,128.  All minibatches will be of "
                   "the largest size until the end of the input is reached; "
                   "then, increasingly smaller sizes will be allowed.  Only egs "
                   "with the same structure (e.g num-frames) are merged.  You may "
                   "specify different minibatch sizes for different sizes of eg "
                   "(defined as the maximum number of Indexes on any input), in "
                   "the format "
                   "--minibatch-size='eg_size1=mb_sizes1/eg_size2=mb_sizes2', e.g. "
                   "--minibatch-size=128=64:128,256/256=32:64,128.  Egs are given "
                   "minibatch-sizes based on the specified eg-size closest to "
                   "their actual size.");
    }
    
    // nnet输入样本合并配置 计算导数?
    void ExampleMergingConfig::ComputeDerived() {
      if (measure_output_frames != "deprecated") {
        KALDI_WARN << "The --measure-output-frames option is deprecated "
            "and will be ignored.";
      }
      if (discard_partial_minibatches != "deprecated") {
        KALDI_WARN << "The --discard-partial-minibatches option is deprecated "
            "and will be ignored.";
      }

      // minibatch_size - 1:64
      std::vector<std::string> minibatch_size_split;
      // 按照'/' 分割字符串
      SplitStringToVector(minibatch_size, "/", false, &minibatch_size_split);
      // minibatch_size_split <1:64> 只有一个元素
      
      if (minibatch_size_split.empty()) {
        KALDI_ERR << "Invalid option --minibatch-size=" << minibatch_size;
      }

      // 
      // <<eg_size, int_set>>
      rules.resize(minibatch_size_split.size());
      // foreach 
      // 经过for循环
      // rules[0].eg_size = ?没有得到处理
      // rules[0].int_set.range[0].first = 1
      // rules[0].int_set.range[0].second = 64
      // rules[0].int_set.largest_size = 64
      for (size_t i = 0; i < minibatch_size_split.size(); i++) {
        int32 &eg_size = rules[i].first;
        IntSet &int_set = rules[i].second;
        // 'this_rule' will be either something like "256" or like "64-128,256"
        // (but these two only if  minibatch_size_split.size() == 1, or something with
        // an example-size specified, like "256=64-128,256"
        // 1:64
        std::string &this_rule = minibatch_size_split[i];
        // 没有 =
        if (this_rule.find('=') != std::string::npos) {
          std::vector<std::string> rule_split;  // split on '='
          SplitStringToVector(this_rule, "=", false, &rule_split);
          if (rule_split.size() != 2) {
            KALDI_ERR << "Could not parse option --minibatch-size="
                      << minibatch_size;
          }
          if (!ConvertStringToInteger(rule_split[0], &eg_size) ||
              !ParseIntSet(rule_split[1], &int_set))
            KALDI_ERR << "Could not parse option --minibatch-size="
                      << minibatch_size;

        } else {
          if (minibatch_size_split.size() != 1) {
            KALDI_ERR << "Could not parse option --minibatch-size="
                      << minibatch_size << " (all rules must have "
                      << "eg-size specified if >1 rule)";
          }



          
          if (!ParseIntSet(this_rule, &int_set))
            KALDI_ERR << "Could not parse option --minibatch-size="
                      << minibatch_size;

          // 1:64
          // int_set range[n] 每个range是一个范围 这里只有一个
          // range[0].first = 1
          // range[0].second=64
          // int_set.largest_size = 64
          bool ExampleMergingConfig::ParseIntSet(const std::string &str,
                                                 ExampleMergingConfig::IntSet *int_set) {
            std::vector<std::string> split_str;
            // <1:64> 只有一个元素
            SplitStringToVector(str, ",", false, &split_str);
            if (split_str.empty())
              return false;
            
            int_set->largest_size = 0;
            int_set->ranges.resize(split_str.size());
            // 1个元素 1:64
            for (size_t i = 0; i < split_str.size(); i++) {
              std::vector<int32> split_range;
              // 将1:64 分为 两个 1 64
              SplitStringToIntegers(split_str[i], ":", false, &split_range);

              // never
              if (split_range.size() < 1 || split_range.size() > 2 ||
                  split_range[0] > split_range.back() || split_range[0] <= 0)
                return false;

              // int_set 包含多个范围, 这里实际只有一个, 1:64
              // int_set 选择其中最大的size --- 64
              int_set->ranges[i].first = split_range[0];
              int_set->ranges[i].second = split_range.back();
              int_set->largest_size = std::max<int32>(int_set->largest_size,
                                                      split_range.back());
            }
            return true;
          }

        }
      }




      // rules 是从ComputeDerived中计算得到的, 是解析命令行参数得到的
      // 如果命令行参数 minibatch-sizes 的参数中没有= 那么 设置为0
      // 'rules' is derived from the configuration values above by ComputeDerived(),
      // and are not set directly on the command line.  'rules' is a list of pairs
      // (eg-size, int-set-of-minibatch-sizes); If no explicit eg-sizes were
      // specified on the command line (i.e. there was no '=' sign in the
      // --minibatch-size option), then we just set the int32 to 0.
      
      {
        // check that no size is repeated.
        // minibatch_size_split <1:64> 只有一个元素
        std::vector<int32> all_sizes(minibatch_size_split.size());
        for (size_t i = 0; i < minibatch_size_split.size(); i++)
          all_sizes[i] = rules[i].first;
        
        std::sort(all_sizes.begin(), all_sizes.end());
        if (!IsSortedAndUniq(all_sizes)) {
          KALDI_ERR << "Invalid --minibatch-size=" << minibatch_size
                    << " (repeated example-sizes)";
        }
      }
    }


    

    SequentialNnetExampleReader example_reader(examples_rspecifier);
    NnetExampleWriter example_writer(examples_wspecifier);





    ExampleMerger merger(merging_config, &example_writer);

    ExampleMerger::ExampleMerger(const ExampleMergingConfig &config,
                                 NnetExampleWriter *writer):
        finished_(false), num_egs_written_(0),
        config_(config), writer_(writer) { }
    

    // ================ merge 处理所有example ==============
    // 但是都是等待一个具体的minibatch_size才进行merged, 最后一次merged时,可能不到minibatch,
    // 所以后面需要一个finish里面进行 终止判断的merge,　这里的batch大小 != minibatch_size.
    
    // merge对象 读取所有的example
    for (; !example_reader.Done(); example_reader.Next()) {
      const NnetExample &cur_eg = example_reader.Value();
      merger.AcceptExample(new NnetExample(cur_eg));

      void ExampleMerger::AcceptExample(NnetExample *eg) {
        KALDI_ASSERT(!finished_);
        // 如果一个eg 具有相同结果的eg已经存在, 他不会被替换
        // 如果eg是一个新的结果, 就会构建一个key,
        // 在清空vector之前我们会删除key.
        // 这样 我们可以保证 key中的eg 会一直是vector中的first元素.
        
        // If an eg with the same structure as 'eg' is already a key in the
        // map, it won't be replaced, but if it's new it will be made
        // the key.  Also we remove the key before making the vector empty.
        // This way we ensure that the eg in the key is always the first
        // element of the vector.

        // eg_to_egs_
        // [[eg][eg][eg][eg][eg][eg]]  具有相同规格
        // [[eg][eg][eg][eg][eg]]
        // [...]
        // [[eg][eg][eg][eg][eg]]
        std::vector<NnetExample*> &vec = eg_to_egs_[eg];
       
        // typedef unordered_map<NnetExample*, std::vector<NnetExample*>,
        //                       NnetExampleStructureHasher,
        //                       NnetExampleStructureCompare> MapType;
        // MapType eg_to_egs_;
        
        vec.push_back(eg);
        int32
            // 该eg中多个input的index中最大size的size
            eg_size = GetNnetExampleSize(*eg),
            // 已经接收过的n个eg.
            num_available = vec.size();


        // 获得多个输入中 index中 size最大的 size.
        int32 GetNnetExampleSize(const NnetExample &a) {
          int32 ans = 0;
          for (size_t i = 0; i < a.io.size(); i++) {
            int32 s = a.io[i].indexes.size();
            if (s > ans)
              ans = s;
          }
          return ans;
        }


        // 返回当前eg, 应该打包的minibatch size大小.
        // 返回!=0, 则说明需要按返回数字, 将eg进行打包
        // 返回 =0, 则说明还不够打包数量, 继续循环.
        bool input_ended = false;
        int32 minibatch_size = config_.MinibatchSize(eg_size,
                                                     num_available,
                                                     input_ended);

        int32 ExampleMergingConfig::MinibatchSize(int32 size_of_eg,
                                                  int32 num_available_egs,
                                                  bool input_ended) const {
          KALDI_ASSERT(num_available_egs > 0 && size_of_eg > 0);

          
          int32 num_rules = rules.size();

          // 最接近当前eg size的rule 
          int32
              min_distance = std::numeric_limits<int32>::max(),
              closest_rule_index = 0;
          for (int32 i = 0; i < num_rules; i++) {
            // distance rules[i] 和 当前eg size 的接近程度.
            int32 distance = std::abs(size_of_eg - rules[i].first);
            if (distance < min_distance) {
              min_distance = distance;
              closest_rule_index = i;
            }
          }

          // 如果当前没输入完 eg,
          // 如果 规则中保存的大小largest_size <= 当前已经保存的eg数量 那么以largest_size 返回, 通知下面可以按largest_size打包
          // 如果 > 说明 已经保存的eg数量还太少, 还不够打一包的 返回0, 通知下面冷灰再打包.
          // 实际上就是 在 已经保存的eg数量 == largest_size时候 进行一下minibatch打包.
          // 
          // 而 当eg 输入完了,
          // 那么根据上面循环的按largest_size 进行打包, 当到达输入完成时候, 那么当前的数量一定 < largest_size
          if (!input_ended) {
            // until the input ends, we can only use the largest available
            // minibatch-size (otherwise, we could expect more later).
            int32 largest_size = rules[closest_rule_index].second.largest_size;
            if (largest_size <= num_available_egs)
              return largest_size;
            else
              return 0;
          } else {
            int32 s = rules[closest_rule_index].second.LargestValueInRange(
                num_available_egs);
            KALDI_ASSERT(s <= num_available_egs);
            return s;
          }
        }


        

        // true
        // 按照minibathch, 将
        if (minibatch_size != 0) {  // we need to write out a merged eg.
          KALDI_ASSERT(minibatch_size == num_available);

          std::vector<NnetExample*> vec_copy(vec);
          eg_to_egs_.erase(eg);

          // MergeExamples() expects a vector of NnetExample, not of pointers,
          // so use swap to create that without doing any real work.
          std::vector<NnetExample> egs_to_merge(minibatch_size);
          for (int32 i = 0; i < minibatch_size; i++) {
            egs_to_merge[i].Swap(vec_copy[i]);
            delete vec_copy[i];  // we owned those pointers.
          }
          WriteMinibatch(egs_to_merge);

          void ExampleMerger::WriteMinibatch(const std::vector<NnetExample> &egs) {
            KALDI_ASSERT(!egs.empty());
            // egs[0] 中 多个组成部分中 多个io的 indexes中 size最大的size??
            int32 eg_size = GetNnetExampleSize(egs[0]);

            // 向minibatch中写入 多个eg的信息头
            NnetExampleStructureHasher eg_hasher;
            size_t structure_hash = eg_hasher(egs[0]);
            int32 minibatch_size = egs.size();
            stats_.WroteExample(eg_size, structure_hash, minibatch_size);

            // 合并多个eg
            NnetExample merged_eg;

            // ================== 汇总多个eg ==> merged_eg ==================
            // egs - input的多个eg
            // config_.compress = True
            // merged_eg 合并后的eg
            // 将多个eg 合并成一个 merged_eg
            // 1 GetIoNames 将多个eg的所有NnetIo, 取出来 unique, 得到最终 汇总的NnetIo总数 以及对应names
            // 2 GetIoSizes 将多个eg的所有NnetIo, 按照汇总结果的样子, 增加size.
            // 3 根据上面两部, 将多个eg的所有NnetIo, 全部汇集成为 汇总NnetIo.
            //   注意 汇总前, 每个eg 是一句话的多帧数据, 所以内部的NnetIo 的 indexes的n都保持不变
            //        汇总后, 多个eg中相同的NnetIo输入, 会汇总在一起, 对应的indexes也需要修改n , n 表示来自原本的第几个eg.
            MergeExamples(egs, config_.compress, &merged_eg);
            void MergeExamples(const std::vector<NnetExample> &src,
                               bool compress,
                               NnetExample *merged_eg) {
              KALDI_ASSERT(!src.empty());
              // NnetIo 
              // 将egs中每个eg 的每个io >> io_names
              std::vector<std::string> io_names;
              GetIoNames(src, &io_names);
              // 大小是, 我们处理的所有eg的NnetIo中实际引用的几个点的 数据行数总数.??
              // the sizes are the total number of Indexes we have across all examples.
              std::vector<int32> io_sizes;
              GetIoSizes(src, io_names, &io_sizes);
              
              MergeIo(src, io_names, io_sizes, compress, merged_eg);
            }


            // ================== 至此 将汇总的merged-eg 写入文件 ====================
            std::ostringstream key;
            key << "merged-" << (num_egs_written_++) << "-" << minibatch_size;
            writer_->Write(key.str(), merged_eg);
          }

          
        }
      }



      
      

      
    }



    // =============== merge eg_to_egs_ 中到达最终时剩余的不到minibatch_size 的eg(不能浪费数据)
    
    // merge 自己打印必要的诊断信息
    // the merger itself prints the necessary diagnostics.
    merger.Finish();
    return merger.ExitStatus();
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
