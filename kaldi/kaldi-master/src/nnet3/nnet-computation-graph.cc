// nnet3/nnet-computation-graph.cc

// Copyright      2015  Johns Hopkins University (author: Daniel Povey)

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

#include <deque>
#include "nnet3/nnet-computation-graph.h"
#include "nnet3/nnet-graph.h"

namespace kaldi {
namespace nnet3 {


int32 ComputationGraph::GetCindexId(const Cindex &cindex,
                                    bool input, bool *is_new) {
  typedef unordered_map<Cindex, int32, CindexHasher> map_type;
  int32 new_index = cindexes.size();  // we'll add this if we don't find it.
  // 向 cindex_to_cindex_id_ 中插入 一个 <cindex, id(当前cindexes的数量,即id)>
  std::pair<map_type::iterator, bool> p = 
          cindex_to_cindex_id_.insert(std::pair<Cindex, int32>(cindex, new_index));
  
  // 返回 true, 则插入成功, 否则, 说明已经存在
  if (p.second == true) {  // We added something to the hash.
    *is_new = true;
    // 这个Assert没有作用, 永远会是匹配的
    // 因为 只有下面的地方,进行了 push_back()
    KALDI_ASSERT(is_input.size() == cindexes.size());
    cindexes.push_back(cindex);
    is_input.push_back(input);

    // make room for this "dependencies" entry.
    dependencies.resize(new_index + 1);
    // 返回 cindex id
    return new_index;
  } else { // We did not add anything.
    *is_new = false;
    return p.first->second;
  }
}
int32 ComputationGraph::GetCindexId(const Cindex &cindex) const {
  typedef unordered_map<Cindex, int32, CindexHasher> map_type;
  map_type::const_iterator iter = cindex_to_cindex_id_.find(cindex);
  if (iter == cindex_to_cindex_id_.end())
    return -1;
  else
    return iter->second;
}


void ComputationGraph::Renumber(int32 start_cindex_id,
                                const std::vector<bool> &keep) {
  int32 old_num_cindex_ids = cindexes.size();
  KALDI_ASSERT(keep.size() == old_num_cindex_ids - start_cindex_id);
  // count_before_renumbering is the number of cindex_ids >= start_cindex_id,
  // before renumbering.
  int32 count_before_renumbering = old_num_cindex_ids - start_cindex_id;
  std::vector<int32> old2new(count_before_renumbering, -1), new2old;
  new2old.reserve(old_num_cindex_ids);
  for (int32 j = 0; j < count_before_renumbering; j++) {
    if (keep[j]) {
      old2new[j] = new2old.size() + start_cindex_id;
      new2old.push_back(j + start_cindex_id);
    }
  }
  // count_after_renumbering is the number of cindex_ids >= start_cindex_id,
  // after renumbering.
  int32 count_after_renumbering = new2old.size(),
      new_num_cindex_ids = start_cindex_id + count_after_renumbering;
  if (count_after_renumbering == count_before_renumbering) {
    // this is an optimization for when we are not deleting any
    // cindex-ids.
    return;
  }

  for (int32 old_cindex_id = start_cindex_id;
       old_cindex_id < old_num_cindex_ids; old_cindex_id++) {
    int32 new_cindex_id = old2new[old_cindex_id - start_cindex_id];
    Cindex &cindex = cindexes[old_cindex_id];
    if (new_cindex_id == -1) {
      cindex_to_cindex_id_.erase(cindex);
    } else if (new_cindex_id != old_cindex_id) {
      cindex_to_cindex_id_[cindex] = new_cindex_id;
    }
  }

  std::vector<int32> temp;
  for (int32 c = start_cindex_id; c < new_num_cindex_ids; c++) {
    int32 d = new2old[c - start_cindex_id];
    // note: d >= c, which is why we do not overwrite data here.
    KALDI_PARANOID_ASSERT(d >= c);
    cindexes[c] = cindexes[d];
    is_input[c] = is_input[d];
    // if c == d, we need to create a temporary copy.
    const std::vector<int32> &src_dependencies =
        (c == d ? (temp = dependencies[d]) : dependencies[d]);
    std::vector<int32>::const_iterator
        iter = src_dependencies.begin(), end = src_dependencies.end();
    dependencies[c].clear();
    for (; iter != end; ++iter) {
      int32 old_dep = *iter;
      if (old_dep < start_cindex_id) {
        dependencies[c].push_back(old_dep);
      } else {
        int32 new_dep = old2new[old_dep - start_cindex_id];
        if (new_dep != -1)
          dependencies[c].push_back(new_dep);
        else
          KALDI_ERR << "Dependency on nonexistent cindex-id";
      }
    }
  }

  cindexes.resize(new_num_cindex_ids);
  is_input.resize(new_num_cindex_ids);
  dependencies.resize(new_num_cindex_ids);
}

void ComputationGraphBuilder::PrintCindexId(std::ostream &os,
                                            int32 cindex_id) const {
  KALDI_ASSERT(static_cast<size_t>(cindex_id) < graph_->cindexes.size());
  const Cindex &cindex = graph_->cindexes[cindex_id];
  const std::string &node_name = nnet_.GetNodeName(cindex.first);
  os << node_name << '(' << cindex.second.n << ", " << cindex.second.t
     << ", " << cindex.second.x << ')';
}

void ComputationGraphBuilder::ExplainWhyNotComputable(
    int32 first_cindex_id) const {
  int32 max_lines_print = 100;
  std::deque<int32> cindexes_to_explain;
  cindexes_to_explain.push_back(first_cindex_id);
  KALDI_ASSERT(graph_->cindexes.size() == graph_->dependencies.size());
  std::ostringstream os;
  os << "*** cindex ";
  PrintCindexId(os, first_cindex_id);
  os << " is not computable for the following reason: ***\n";
  for (int32 num_lines_printed = 0;
       num_lines_printed < max_lines_print && !cindexes_to_explain.empty();
       num_lines_printed++) {
    int32 cindex_id = cindexes_to_explain.front();
    cindexes_to_explain.pop_front();
    KALDI_ASSERT(static_cast<size_t>(cindex_id) < graph_->cindexes.size());
    PrintCindexId(os, cindex_id);
    os << " is " << static_cast<ComputableInfo>(
        computable_info_[cindex_id]) << ", dependencies: ";
    const std::vector<int32> dependencies = graph_->dependencies[cindex_id];
    std::vector<int32>::const_iterator iter = dependencies.begin(),
        end = dependencies.end();
    for (; iter != end; iter++) {
      int32 dep_cindex_id = *iter;
      PrintCindexId(os, dep_cindex_id);
      ComputableInfo status = static_cast<ComputableInfo>(
          computable_info_[cindex_id]);
      if (status != kComputable) {
        os << '[' << status << ']';
        cindexes_to_explain.push_back(dep_cindex_id);
      }
      if (iter+2 != end)
        os << ", ";
    }
    os << "\n";
  }
  os << "\n";
  KALDI_LOG << os.str();
}


void ComputationGraph::Print(std::ostream &os,
                             const std::vector<std::string> &node_names) {
  int32 max_cindexes_per_line = 50, max_dependencies = 5,
      num_cindexes = cindexes.size();

  std::vector<std::pair<Cindex, std::vector<Cindex> > > pairs;
  pairs.reserve(num_cindexes);
  for (int32 cindex_id = 0; cindex_id < num_cindexes; cindex_id++) {
    int32 size = dependencies[cindex_id].size();
    std::vector<Cindex> deps(size);
    for (size_t i = 0; i < size; i++)
      deps[i] = cindexes[dependencies[cindex_id][i]];
    std::sort(deps.begin(), deps.end());
    pairs.push_back(std::pair<Cindex, std::vector<Cindex> >(cindexes[cindex_id],
                                                            deps));
  }
  std::sort(pairs.begin(), pairs.end());
  int32 cur_start = 0;
  for (int32 i = 0; i < num_cindexes; i++) {
    if (pairs[i].first.first != pairs[cur_start].first.first) {
      cur_start = i;
      os << "\n";
    }
    if (i - cur_start < max_cindexes_per_line) {
      os << "[ ";
      PrintCindex(os, pairs[i].first, node_names);
      if (! is_input[GetCindexId(pairs[i].first)]) {
        // only print out dependences for cindexes that
        // were not provided as inputs.
        os << " -> ";
        for (int32 j = 0; j < pairs[i].second.size(); j++) {
          if (j < max_dependencies) {
            PrintCindex(os, pairs[i].second[j], node_names);
            if (j + 1 < pairs[i].second.size())
              os << ", ";
          } else if (j == max_dependencies) {
            os << "...";
          }
        }
      }
      os << " ] ";
    } else if (i - cur_start == max_cindexes_per_line) {
      os << "...";
    }
  }
  os << "\n";

}


// inline
void ComputationGraphBuilder::AddCindexId(int32 cindex_id,
                                          bool is_input,
                                          bool is_output) {
  // If this cindex_id has just now been added to the graph, the following
  // assert should succeed.
  KALDI_PARANOID_ASSERT(cindex_id == computable_queued_.size() &&
                        cindex_id == computable_info_.size() &&
                        cindex_id == depend_on_this_.size() &&
                        cindex_id == usable_count_.size());
  // input
  if (is_input) {
    computable_info_.push_back(kComputable);
    computable_queued_.push_back(false);
  } 
  // 非 input 
  else {
    computable_info_.push_back(kUnknown);
    // add to the queue of things for which we need to compute their computable
    // status.
    computable_queued_.push_back(false);

    //  ------------ Important
    // 这个 next_queue_ 保存已经得到的从output -> input 的路径
    // 然后在后面根据这个next_queue_, 
    // 从output开始逐渐向input, 寻路,找到完整计算路径.
    next_queue_.push_back(cindex_id);
  }


  // 为当前 cindex, 申请一个vector
  // 描述 denpen_on_this 的所有 cindex.
  depend_on_this_.push_back(std::vector<int32>());
  
  // 描述的是 cindex_id > 0 则, 该cindex 是计算路径中的一部分.
  usable_count_.push_back(is_output ? 1 : 0);
}


void ComputationGraphBuilder::AddInputs() {
  int32 num_added = 0;
  // request 内 IoSpecification --
  //                     name
  //                     indexes
  for (int32 i = 0; i < request_->inputs.size(); i++) {
    // 根据 NnetIo.name 获得 Nnet中的 node index.
    // 可以同时索引 node_names_  nodes_ 获得node-name 和 具体 node
    int32 n = nnet_.GetNodeIndex(request_->inputs[i].name);
    if (n == -1)
      KALDI_ERR << "Network has no input with name "
                << request_->inputs[i].name;

    NodeType t = nnet_.GetNode(n).node_type;
    KALDI_ASSERT((t == kInput || t == kComponent) &&
                 "Inputs to graph only allowed for Input and Component nodes.");
    // foreach 每个小样本 (left-context, 1, right-context) , 计算具体到每个点?
    for (int32 j = 0; j < request_->inputs[i].indexes.size(); j++) {
      // 为每个具体小样本构建Cindex.
      Cindex cindex(n, request_->inputs[i].indexes[j]);
      bool is_input = true, is_new;
      // GetCindexId, 查找cindex 是否已经存在.
      // 
      int32 cindex_id = graph_->GetCindexId(cindex, is_input, &is_new);
      KALDI_ASSERT(is_new && "Input index seems to be listed more than once");
      // 
      AddCindexId(cindex_id, true, false);
      num_added++;
    }
  }
  KALDI_ASSERT(num_added > 0 && "AddInputToGraph: nothing to add.");
}

void ComputationGraphBuilder::AddOutputs() {
  int32 num_added = 0;
  for (int32 i = 0; i < request_->outputs.size(); i++) {
    int32 n = nnet_.GetNodeIndex(request_->outputs[i].name);
    if (n == -1)
      KALDI_ERR << "Network has no output with name "
                << request_->outputs[i].name;
    for (int32 j = 0; j < request_->outputs[i].indexes.size(); j++) {
      Cindex cindex(n, request_->outputs[i].indexes[j]);
      bool is_input = false, is_new;
      int32 cindex_id = graph_->GetCindexId(cindex, is_input, &is_new);
      KALDI_ASSERT(is_new && "Output index seems to be listed more than once");
      AddCindexId(cindex_id, false, true);
      num_added++;
    }
  }
  if (num_added == 0) {
    KALDI_ERR << "Cannot process computation request with no outputs";
  }
  
  
  current_distance_ = 0;

  // 后续的程序 逐渐构建 计算路径的起点 ---- current_queue_.swap(next_queue_).
  // the calls to AddCindexId in this function will have added to next_queue_.
  KALDI_ASSERT(current_queue_.empty());
  current_queue_.swap(next_queue_);
}

bool ComputationGraphBuilder::AllOutputsAreComputable() const {
  char is_computable_char = static_cast<char>(kComputable);
  std::vector<char>::const_iterator iter = computable_info_.begin(),
      end = computable_info_.end();
  for (int32 cindex_id = 0; iter != end; ++iter, ++cindex_id) {
    if (*iter != is_computable_char) {  // is not computable.
      int32 network_node = graph_->cindexes[cindex_id].first;
      if (nnet_.IsOutputNode(network_node))
        return false;
    }
  }
  return true;
}

std::ostream& operator << (std::ostream &os,
                           const ComputationGraphBuilder::ComputableInfo &info) {
  switch (info) {
    case ComputationGraphBuilder::kUnknown: os << "kUnknown";
      break;
    case ComputationGraphBuilder::kComputable: os << "kComputable";
      break;
    case ComputationGraphBuilder::kNotComputable: os << "kNotComputable";
      break;
    case ComputationGraphBuilder::kWillNotCompute: os << "kWillNotCompute";
      break;
    default: os << "[invalid enum value]"; break;
  }
  return os;
}


// Prints logging info to explain why all outputs are not computable.
void ComputationGraphBuilder::ExplainWhyAllOutputsNotComputable() const {
  std::vector<int32> outputs_not_computable;
  int32 num_outputs_total = 0;

  std::vector<Cindex>::const_iterator iter = graph_->cindexes.begin(),
      end = graph_->cindexes.end();
  for (int32 cindex_id = 0; iter != end; ++iter,++cindex_id) {
    int32 network_node = iter->first;
    ComputableInfo c = static_cast<ComputableInfo>(computable_info_[cindex_id]);
    if (nnet_.IsOutputNode(network_node)) {
      num_outputs_total++;
      if (c != kComputable)
        outputs_not_computable.push_back(cindex_id);
    }
  }
  KALDI_ASSERT(!outputs_not_computable.empty() &&
               "You called this function when everything was computable.");
  int32 num_print = 10, num_not_computable = outputs_not_computable.size();
  KALDI_LOG << num_not_computable << " output cindexes out of "
            << num_outputs_total << " were not computable.";
  std::ostringstream os;
  request_->Print(os);
  KALDI_LOG << "Computation request was: " << os.str();
  if (num_not_computable > num_print)
    KALDI_LOG << "Printing the reasons for " << num_print << " of these.";
  for (int32 i = 0; i < num_not_computable && i < num_print; i++)
    ExplainWhyNotComputable(outputs_not_computable[i]);
}



// this function limits the dependencies of cindex_id "cindex_id" to just those
// which are actually used in computing it.  It also clears the dependencies
// of those cindexes that are not computable.
void ComputationGraphBuilder::PruneDependencies(int32 cindex_id) {
  ComputableInfo c = static_cast<ComputableInfo>(computable_info_[cindex_id]);
  // by the time this is called, there should be no cindexes with unknown state.
  KALDI_ASSERT(c != kUnknown);
  if (c == kNotComputable || c == kWillNotCompute) {
    // if something is not computable, there is no point
    // keeping around its dependencies.
    graph_->dependencies[cindex_id].clear();
    return;
  }
  KALDI_ASSERT(c == kComputable);
  const Cindex &cindex = graph_->cindexes[cindex_id];
  int32 node_id = cindex.first;
  const Index &index = cindex.second;
  const NetworkNode &node = nnet_.GetNode(node_id);

  std::vector<int32> &dependencies = graph_->dependencies[cindex_id];
  std::sort(dependencies.begin(), dependencies.end());
  std::vector<int32> used_cindex_ids;

  switch (node.node_type) {
    case kDescriptor: {
      const Descriptor &desc = node.descriptor;
      bool dont_care = false;  // there should be no kUnknown, and we check this
      CindexSet cindex_set(*graph_, computable_info_, dont_care);
      std::vector<Cindex> used_cindexes;
      bool ans = desc.IsComputable(index, cindex_set, &used_cindexes);
      // If the next assert fails it could be a failure in the assumption that
      // making more inputs available will never change something from not being
      // computable to being computable; or it could be a bug elsewhere.
      KALDI_ASSERT(ans);
      size_t size = used_cindexes.size();
      used_cindex_ids.resize(size);
      for (size_t i = 0; i < size; i++) {
        int32 dep_cindex_id = graph_->GetCindexId(used_cindexes[i]);
        KALDI_ASSERT(dep_cindex_id != -1);
        used_cindex_ids[i] = dep_cindex_id;
        KALDI_ASSERT(std::binary_search(dependencies.begin(),
                                        dependencies.end(),
                                        dep_cindex_id));
      }
      break;
    }
    case kComponent: {
      const Component *c = nnet_.GetComponent(node.u.component_index);
      bool dont_care = false;  // there should be no kUnknown, and we check this
      // In the line below, node_id - 1 is the index of the component-input
      // node-- the descriptor at the input to the component.  We are interested
      // in the set of inputs to the component that are computable.
      IndexSet index_set(*graph_, computable_info_, node_id - 1, dont_care);
      std::vector<Index> used_indexes;
      bool ans = c->IsComputable(request_->misc_info, index, index_set,
                                 &used_indexes);
      // If the next assert fails it could be a failure in the assumption that
      // making more inputs available will never change something from not being
      // computable to being computable; or it could be a bug elsewhere.
      KALDI_ASSERT(ans);
      size_t size = used_indexes.size();
      used_cindex_ids.resize(size);
      for (size_t i = 0; i < size; i++) {
        Cindex dep_cindex(node_id - 1, used_indexes[i]);
        int32 dep_cindex_id = graph_->GetCindexId(dep_cindex);
        KALDI_ASSERT(dep_cindex_id != -1);
        used_cindex_ids[i] = dep_cindex_id;
        KALDI_ASSERT(std::binary_search(dependencies.begin(),
                                        dependencies.end(),
                                        dep_cindex_id));
      }
      break;
    }
    case kDimRange:
      KALDI_ASSERT(dependencies.size() == 1);
      // there should be exactly one dependency and it is required, not
      // optional, so there is nothing to do here.  Return.
      return;
    case kInput:
      KALDI_ASSERT(dependencies.empty());
      // there is nothing to do; return.
      return;
    default:
      KALDI_ERR << "Invalid node type";
  }
  SortAndUniq(&used_cindex_ids);

  // the next statement modifies the graph.
  dependencies.swap(used_cindex_ids);
}

ComputationGraphBuilder::ComputationGraphBuilder(
    const Nnet &nnet,
    ComputationGraph *graph):
    nnet_(nnet), request_(NULL), graph_(graph),
    current_distance_(-1) {
  KALDI_ASSERT(graph_->cindexes.empty() &&
               "ComputationGraphBuilder initialized with nonempty graph.");
}


void ComputationGraphBuilder::Compute(const ComputationRequest &request) {
  // 注意当前是在 GraphBuilder内, 并不在 Compiler里面.
  if (request_ != NULL && graph_->segment_ends.empty()) {
    // this check is relevant to multi-segment (i.e. online) computations.
    KALDI_ERR << "You are calling things in the wrong order: should be "
              << "Compute(), Prune(), Compute(), Prune(), ...";
  }

  // ---------------------
  // segment, 是一个 request的分界位置.
  //     计算第一个request时候 == 0
  //     后续在进入这里, 就是上一个request的cindex 完全加入到计算图中了.
  // ---------------------

  int32 cur_segment_start = graph_->cindexes.size();
  request_ = &request;
  // 插入 input NnetIo<IoSpecification>
  AddInputs();
  AddOutputs();  // sets current_distance_ to 0.

  // 循环的按照Nnet网络, 完善计算路径.
  //   1 从高层 output 向 input 添加依赖 cindex
  //   2 从底层 input  向 output 通知高层cindex可计算性 
  
  // 保证没有无限递归的错误.
  // max_distance for debugging, to detect infinite recursion.
  int32 max_distance = 10000;
  while (current_distance_ < max_distance) {
    // 1 从 output -> input 查找底层依赖的方式 构建 计算路径结构 cindex路径
    BuildGraphOneIter();
    
    // only check rarely if we're running at low verbose level.
    if (GetVerboseLevel() >= 3 || RandInt(1,  (current_distance_ + 1)) == 1)
      Check(cur_segment_start);

    // 2 从 input -> output 通知高层可以计算 computable
    // TODO: come up with a scheme to delay when we call
    // UpdateAllComputableInfo().
    UpdateAllComputableInfo();

    // 3 确定没有了需要查找依赖的cindex, 完成计算路径构建
    if (current_queue_.empty()) // we're done.
      break;
  }


  if (current_distance_ == max_distance)
    KALDI_ERR << "Loop detected while building computation graph (bad "
              << "network topology?)";

  if (RandInt(1, 2 * (graph_->segment_ends.size() + 1)) == 1)
    Check(cur_segment_start);
}

// 检查当前 request 的计算依赖关系, 和 可计算性.
void ComputationGraphBuilder::Check(int32 start_cindex_id) const {
  // 
  int32 num_cindex_ids = graph_->cindexes.size();
  for (int32 cindex_id = start_cindex_id; cindex_id < num_cindex_ids;
       cindex_id += 1 + RandInt(0, num_cindex_ids / 100)) {
    { // check depend_on_this.
      std::vector<int32> depend_on_this = depend_on_this_[cindex_id];
      int32 size = depend_on_this.size();
      std::sort(depend_on_this.begin(), depend_on_this.end());
      KALDI_ASSERT(IsSortedAndUniq(depend_on_this));
      for (size_t j = 0; j < size; j++) {
        int32 other_cindex_id = depend_on_this[j];
        // make sure appears in appropriate dependencies array.
        const std::vector<int32> &dep = graph_->dependencies[other_cindex_id];
        KALDI_ASSERT(std::count(dep.begin(), dep.end(), cindex_id) == 1);
      }
    }
    
    { // check dependencies.
      std::vector<int32> dependencies = graph_->dependencies[cindex_id];
      int32 size = dependencies.size();
      std::sort(dependencies.begin(), dependencies.end());
      KALDI_ASSERT(IsSortedAndUniq(dependencies));
      for (size_t j = 0; j < size; j++) {
        int32 dep_cindex_id = dependencies[j];
        if (dep_cindex_id >= start_cindex_id) {
          // make sure appears in appropriate depend_on_this_ array.
          const std::vector<int32> &dep = depend_on_this_[dep_cindex_id];
          KALDI_ASSERT(std::count(dep.begin(), dep.end(), cindex_id) == 1);
        }
      }
    }

    {
      // check usable_count_
      int32 node_index = graph_->cindexes[cindex_id].first;
      int32 usable_count = usable_count_[cindex_id],
          usable_count_recomputed = nnet_.IsOutputNode(node_index) ? 1 : 0;
      std::vector<int32> depend_on_this = depend_on_this_[cindex_id];
      int32 size = depend_on_this.size();
      for (size_t j = 0; j < size; j++) {
        int32 other_cindex_id = depend_on_this[j];
        if (usable_count_[other_cindex_id] != 0 &&
            computable_info_[other_cindex_id] != kNotComputable)
          usable_count_recomputed++;
      }
      KALDI_ASSERT(usable_count == usable_count_recomputed);
    }
    // check computable_info_.  note: this will not be accurate
    // if the cindex_id is still queued to have dependencies added
    // (in cur_queue_ or next_queue_).
    if (computable_queue_.empty()) {
      ComputationGraphBuilder::ComputableInfo c =
          ComputeComputableInfo(cindex_id);
      // the status doesn't have to be correct if it's kWillNotCompute,
      // because these are cindex-ids that we chose not to compute
      // because we determined they would not be useful, and
      // ComputeComputableInfo() will never return this value.
      if (c != computable_info_[cindex_id] &&
          computable_info_[cindex_id] != kWillNotCompute) {
        int32 count_cur = std::count(current_queue_.begin(),
                                     current_queue_.end(), cindex_id),
            count_next = std::count(next_queue_.begin(),
                                    next_queue_.end(), cindex_id);
        // if it wasn't queued, then something is wrong.
        if (count_cur + count_next == 0)
          KALDI_ERR << "Mismatch in computable status";
      }
    }
    // check computable_queued_.
    // note, the following checks might be a bit slow.
    if (computable_queued_[cindex_id]) {
      KALDI_ASSERT(std::count(computable_queue_.begin(),
                              computable_queue_.end(),
                              cindex_id) == 1);
    } else {
      KALDI_ASSERT(std::count(computable_queue_.begin(),
                              computable_queue_.end(),
                              cindex_id) == 0);
    }
  }
}

void ComputationGraphBuilder::Prune() {
  // Since Prune() is called for each segment in turn [note: there
  // will be only 1 segment in the normal non-online case], we
  // only prune for the current, just-added segment.
  int32 start_cindex_id = (graph_->segment_ends.empty() ? 0 :
                           graph_->segment_ends.back());
  int32 num_cindex_ids = graph_->cindexes.size();
  // Prune the dependencies to just those that are used (to remove
  // optional dependencies that don't end up getting used).

  for (int32 cindex_id = start_cindex_id;
       cindex_id < num_cindex_ids; cindex_id++)
    PruneDependencies(cindex_id);
  // the following clears the elements of depend_on_this from start_cindex_id to
  // num_cindex_ids - 1, without touching the entire array.
  depend_on_this_.resize(start_cindex_id);
  depend_on_this_.resize(num_cindex_ids);
  std::vector<bool> required;
  ComputeRequiredArray(start_cindex_id, &required);

  std::vector<bool> keep(num_cindex_ids - start_cindex_id, false);
  for (int32 c = start_cindex_id; c < num_cindex_ids; c++) {
    if (required[c - start_cindex_id] || graph_->is_input[c]) {
      KALDI_ASSERT(computable_info_[c] == kComputable &&
                   "You are calling Prune when not everything is computable.");
      keep[c - start_cindex_id] = true;
    }
  }
  graph_->Renumber(start_cindex_id, keep);
  // We also need to renumber computable_info_ and usable_count_, which
  // graph_->Renumber doesn't do for us, but we can make some shortcuts.  We set
  // all computable_info_ to kComputable because actually it all was kComputable
  // (we checked when deciding what to keep); and we set the usable_count_ to 1
  // for all the cindex_ids we just added...  this is not 100% accurate
  // according to the way we defined usable_count_, but it prevents additional
  // computation since it is > 0 (notice that IncrementUsableCount and
  // DecrementUsableCount may do some work when the usable_count goes to zero or
  // from zero.  Anyway, the usable-count for these cindex_ids for those "older
  // segments" is not critical.  [this information only gets used if we process
  // additional segments as part of the compilation of an online computation.]
  int32 new_num_cindex_ids = graph_->cindexes.size();
  computable_info_.resize(start_cindex_id);
  computable_info_.resize(new_num_cindex_ids, (char)kComputable);
  usable_count_.resize(start_cindex_id);
  usable_count_.resize(new_num_cindex_ids, 1);
  // depend_on_this_ is a vector of vectors-- keeping track of the reverse of
  // the dependencies-- and I believe we won't be needing this information any
  // more past this point.
  depend_on_this_.resize(start_cindex_id);
  depend_on_this_.resize(new_num_cindex_ids);
  // computable_queued_ also shouldn't be queried past this point, but
  // I believe they should all be false at this point anyway (note that
  // we assert below that computable_queue_ is empty).
  computable_queued_.resize(new_num_cindex_ids);

  KALDI_ASSERT(computable_queue_.empty());

  // Prune 剪枝函数, 是对每个request, 构成了计算图 ComputationGraph之后才进行的
  // 而每个segment 实际上就是每个每个request的计算图部分的所有cindex的分界线, 
  // 所以在剪枝之后才会添加到 分段segment_ends 中
  graph_->segment_ends.push_back(new_num_cindex_ids);
}

//  找到当前cindex_id, 依赖的所有cindex_ids
// Add cindex_ids that this cindex_id depends on.
void ComputationGraphBuilder::AddDependencies(int32 cindex_id) {
  // 如果cindex_id 还没有安排依赖
  if (static_cast<int32>(graph_->dependencies.size()) <= cindex_id) {
    graph_->dependencies.resize(2 * cindex_id + 1);
  }

  Cindex cindex = graph_->cindexes[cindex_id];

  // find the dependencies of this cindex.
  int32 node_index = cindex.first;
  const Index &index = cindex.second;
  const NetworkNode &node = nnet_.GetNode(node_index);

  // cindex 的所有 计算依赖 
  std::vector<Cindex> input_cindexes;

  // the following switch statement sets up "input_cindexes".
  switch (node.node_type) {
    // kDescriptor 直接描述了所有依赖.
    case kDescriptor: {
      // desc describes how this node obtains its input from other nodes.
      const Descriptor &desc = node.descriptor;
      desc.GetDependencies(index, &input_cindexes);
      break;
    }
    // component , 直接依赖 node_index 的 cindex easy
    case kComponent: {
      int32 c = node.u.component_index;
      // 获得对应的 计算Component
      const Component *component = nnet_.GetComponent(c);
      // 计算Component 所需的Index.
      std::vector<Index> input_indexes;
      component->GetInputIndexes(request_->misc_info, index,
                                 &input_indexes);
      input_cindexes.resize(input_indexes.size());
      
      // cp Index 构成 Cindex
      for (size_t i = 0; i < input_indexes.size(); i++) {
        // 直接依赖 所对应的上层kDescriptor
        input_cindexes[i].first = node_index  - 1;  // preceding node
        input_cindexes[i].second = input_indexes[i];
      }
      break;
    }
    case kDimRange: {
      input_cindexes.resize(1);
      input_cindexes[0] = Cindex(node.u.node_index, index);
      break;
    }
    case kInput:
      break;  // There will be no dependencies.
    default:
      KALDI_ERR << "Invalid node type";
  }

  // 为 cindex_id 找到的所有依赖 input_cindexes
  int32 num_dependencies = input_cindexes.size();

  // reserve 语句, 确定我们下面进行的loop循环所需要的空间是可行的,避免allocation
  // this "reserve" statement is to make sure the reference
  // we declare below does not become invalid in the loop below
  // (the call to graph_->GetCindexId() could add up to
  // num_dependencies elements to the graph_->dependencies array
  // and we want to avoid allocation).
  // 申请的大一点是为了效率, 避免频繁resize.
  // the RoundUpToNearestPowerOfTwo is for efficiency, to
  // avoid too-frequent resizes.
  graph_->dependencies.reserve(RoundUpToNearestPowerOfTwo(
      graph_->dependencies.size() +  num_dependencies));
  
  // 引用当前 cindex_id 的所有依赖队列
  std::vector<int32> &this_dep = graph_->dependencies[cindex_id];
  // resize 为 找到的依赖总数
  this_dep.resize(num_dependencies);

  // 将每个依赖都添加到 graph_.cindex_to_cindex_id_中.
  for (size_t i = 0; i < num_dependencies; i++) {
    bool is_input = false, is_new;
    int32 dep_cindex_id = graph_->GetCindexId(input_cindexes[i],
                                              is_input, &is_new);
    this_dep[i] = dep_cindex_id;

    // 如果是新的cindex, 那么还要设置很多东西
    // 1 加入到next_queue_, 下次迭代就要找它的依赖
    // 2 computable_info_, 是否 kComputable, kUnknown, 
    // 3 computable_queue_,
    // 3 depend_on_this_ 为其保留一个list位置
    // 4 usable_count_ = 1
    if (is_new)
      AddCindexId(dep_cindex_id, false, false);
    // we will keep dependent's usable_count_ up to date below
  }
  // remove duplicates of dependencies.
  SortAndUniq(&this_dep);

  // 反向设置 这些依赖 depend_on_cindex_id, depend_on_this_[].push_back[cindex_id]
  // set up the "depend_on_this_" array.
  std::vector<int32>::const_iterator iter = this_dep.begin(),
      end = this_dep.end();

  // 1 设置 depend_on_cindex_id 的 denpend_on_this_
  // 2 增加 usable_count_ 计算引用计数. 
  // Populate the "depend_on_this_" array, and append the
  // usable_count_ of things we depend on (see the definition
  // of this quantity next to where it is declared).

  // Note: before calling AddDependencies() we verified the following:
  //  computable_info_[cindex_id] == kUnknown
  // and
  //  usable_count_[cindex_id] != 0
  // which ensures that the conditions to increment the dependency's
  // usable_count_ are satisfied.
  for (; iter != end; ++iter) {
    int32 dep_cindex_id = *iter;
    depend_on_this_[dep_cindex_id].push_back(cindex_id);
    IncrementUsableCount(dep_cindex_id);
  }

  // 现在, 我们已经增加了dependences, 我们将起加入到computable_queue_, 
  // 类似查找依赖使得递归判断是否可以computable.
  // Now that we've added the dependencies, we can put this into
  // the computable_queue_ to assess whether it's computable
  KALDI_ASSERT(computable_info_[cindex_id] == kUnknown && !computable_queued_[cindex_id]);

  // 认为 放到前面会比较快, push_back push_front 都可以.
  // we think it'll be faster in the next line to do push_front instead of
  // push_back; either one would be correct.
  computable_queue_.push_front(cindex_id);
  // 辨明cindex_id 已经加入判断是否可以计算队列中.
  computable_queued_[cindex_id] = true;
}


ComputationGraphBuilder::ComputableInfo
ComputationGraphBuilder::ComputeComputableInfo(int32 cindex_id)
    const {
  // 1 获得对应的 Nnet 网络中 网络节点 node_id
  const Cindex &cindex = graph_->cindexes[cindex_id];
  int32 node_id = cindex.first;
  // 2 实际数据点 ()
  const Index &index = cindex.second;
  const NetworkNode &node = nnet_.GetNode(node_id);

  switch (node.node_type) {
    case kDescriptor: {
      const Descriptor &desc = node.descriptor;
      {
        // 构建CindexSet, 实际上是根据计算图, 搜索路径的一个方法类, 并没有特殊的.
        //   1 当前计算图
        //   2 computable_info_ 可计算性信息(input 都是 kComputable, output 以及其他一开始都是 kUnknown)
        //   3 false 表明,严格确定可计算性, 不能随意认为 kUnknown == kComputable.
        CindexSet cindex_set(*graph_, computable_info_, false);
        // 1 开始时, input 还没通知到 高层cindex是否可计算, 一般都是 kUnknown, 
        // 所以返回都是false
        // 2 当到后来, input 逐渐向高层通知 可计算, 才会在这里就返回 kComputable
        if (desc.IsComputable(index, cindex_set, NULL)) {
          // it's computable even without counting kUnknown inputs as computable
          // [treat_unknown_as_computable = false] -> definitely computable.
          return kComputable;
        }
      }
      // 这里 降低限制要求, 认为 kUnknown 也可以认为是 kComputable
      // 但是这里实际判断的不是 是否可以计算, 而是 是不是不能计算
      // 一般就返回 kUnknown
      CindexSet cindex_set2(*graph_, computable_info_, true);
      if (!desc.IsComputable(index, cindex_set2, NULL)) {
        // it's not computable even when counting kUnknown inputs as
        // computable [treat_unknown_as_computable = true] -> definitely not
        // computable.
        return kNotComputable;
      }
      return kUnknown;
    }
    case kComponent: {
      // 如果 cindex 代表的是一个 kComponent-node, 那么首先找到对应的 Component
      const Component *c = nnet_.GetComponent(node.u.component_index);
      // 如果cindex 代表是一个 kComponent-node, 那么输入必然是 node_id - 1 的 kDescritpor-node.
      const int32 input_node_id = node_id - 1;

      // 这个流程和 直接计算 kDescriptor 计算可计算性 几乎一样.
      {

        IndexSet index_set(*graph_, computable_info_, input_node_id, false);
        if (c->IsComputable(request_->misc_info, index, index_set, NULL)) {
          // it's computable even without counting kUnknown inputs as computable
          // [treat_unknown_as_computable = false] -> definitely computable.
          return kComputable;
        }
      }
      IndexSet index_set2(*graph_, computable_info_, input_node_id, true);
      if (!c->IsComputable(request_->misc_info, index, index_set2, NULL)) {
        // it's not computable even when counting kUnknown inputs as computable
        // [treat_unknown_as_computable = true] -> definitely not computable.
        return kNotComputable;
      }
      return kUnknown;
    }
    case kDimRange: {
      Cindex input_cindex(node.u.node_index, index);
      int32 input_cindex_id = graph_->GetCindexId(input_cindex);
      if (input_cindex_id != -1)
        return ComputableInfo(computable_info_[input_cindex_id]);
      else
        return kUnknown;
    }
    case kInput: {
      // cindexes for input nodes that are part of the computation request will
      // have graph_->is_input[cindex_id] == true; others will have
      // graph_->is_input[cindex_id] == true.
      return graph_->is_input[cindex_id] ? kComputable : kNotComputable;
    }
    default:
      KALDI_ERR << "Invalid node type.";
      return kUnknown;  // suppress compiler warning.
  }
}

void ComputationGraphBuilder::GetComputableInfo(
    std::vector<std::vector<bool> > *computable) const {
  KALDI_ASSERT(!graph_->cindexes.empty() &&
               "You need to call this after Compute()!");
  KALDI_ASSERT(!computable_info_.empty() &&
               "You need to call this before Prune()!");
  computable->clear();
  computable->resize(request_->outputs.size());
  for (size_t i = 0; i < request_->outputs.size(); i++) {
    const IoSpecification &output = request_->outputs[i];
    int32 n = nnet_.GetNodeIndex(output.name);
    KALDI_ASSERT(n != -1);
    int32 size = output.indexes.size();
    std::vector<bool> &this_vec = (*computable)[i];
    this_vec.resize(size);
    for (size_t j = 0; j < size; j++) {
      Cindex cindex(n, output.indexes[j]);
      int32 cindex_id = graph_->GetCindexId(cindex);
      KALDI_ASSERT(cindex_id != -1);
      this_vec[j] = (computable_info_[cindex_id] == kComputable);
    }
  }
}

// 某个cindex_id , 必然存在于 computable_info_ 中.
// 只有入队, 没有出队.
void ComputationGraphBuilder::UpdateComputableInfo(int32 cindex_id) {
  // if the current computable_info_ for cindex_id value is not kUnknown, this
  // cindex_id should not have been in the queue.
  KALDI_ASSERT(static_cast<size_t>(cindex_id) < computable_info_.size());


  char &output = computable_info_[cindex_id];
  KALDI_ASSERT(output == kUnknown);
  // 计算可计算性. 
  // 1 从output 向 input 找寻依赖时加入的cindex的 都是 kUnknown
  // 2 从 input 向 output 逐渐通知计算性的 是 kComputable
  output = static_cast<char>(ComputeComputableInfo(cindex_id));

  if (output != kUnknown) {
    // 依赖于当前cindex的 当前状态为kUnknown的高层cindex 的计算状态 进行改变.
    // 如果还没在 检查可计算性队列computable_queue_ 中, 则入队,等待检查.
    // depend_on_this_[cindex_id] 因为当前的cindex 可计算性已经确定, 那么可以继续 找寻依赖它的高层cindex
    // 尝试 向上递归检查是否可计算.
    // The computable status of cindexes that depend on this cindex and whose
    // status is currently kUnknown might now change, so if they are not in the
    // computable queue, put them there.
    std::vector<int32>::const_iterator iter = depend_on_this_[cindex_id].begin(),
        end = depend_on_this_[cindex_id].end();
    for (; iter != end; ++iter) {
      int32 other_cindex_id = *iter;
      if (computable_info_[other_cindex_id] == kUnknown &&
          !computable_queued_[other_cindex_id]) {
        computable_queue_.push_back(other_cindex_id);
        computable_queued_[other_cindex_id] = true;
      }
    }
    // 如果 当前cindex的 可计算性为 kNotComputable
    // 并且 是否需要计算引用!=0 说明输出output, 需要这个cindex
    // 那么向后通知, 最后可能这个计算路径失败.
    if (output == kNotComputable && usable_count_[cindex_id] != 0) {
      // If we have just changed the computable state from kUnknown to
      // kNotComputable, then given the way the usable_count_ is defined (see
      // the declaration), this means that we must decrement the
      // usable_count_ of all cindex_ids that we depend on.
      std::vector<int32>::const_iterator
          iter = graph_->dependencies[cindex_id].begin(),
          end = graph_->dependencies[cindex_id].end();
      for (; iter != end; ++iter) {
        int32 dep_cindex_id = *iter;
        DecrementUsableCount(dep_cindex_id);
      }
    }
  }
}

void ComputationGraphBuilder::SetAsWillNotCompute(int32 cindex_id) {
  KALDI_ASSERT(usable_count_[cindex_id] == 0);
  computable_info_[cindex_id] = kWillNotCompute;
  std::vector<int32>::const_iterator iter = depend_on_this_[cindex_id].begin(),
      end = depend_on_this_[cindex_id].end();
  for (; iter != end; ++iter) {
    int32 other_cindex_id = *iter;
    if (computable_info_[other_cindex_id] == kUnknown &&
        !computable_queued_[other_cindex_id]) {
      computable_queue_.push_back(other_cindex_id);
      computable_queued_[other_cindex_id] = true;
    }
  }
}

// 逐个更新 computable_queue_ 中的cindex 的可计算性
void ComputationGraphBuilder::UpdateAllComputableInfo() {
  while (!computable_queue_.empty()) {
    int32 cindex_id = computable_queue_.front();
    computable_queue_.pop_front();
    // 在可计算队列中标记,  在=true, 
    computable_queued_[cindex_id] = false;
    // 根据什么进行更新的呢
    UpdateComputableInfo(cindex_id);
  }
}


// cindex_id 增加 计算引用计数()
void ComputationGraphBuilder::IncrementUsableCount(int32 cindex_id) {
  KALDI_PARANOID_ASSERT(static_cast<size_t>(cindex_id)<usable_count_.size());
  
  // 增加一个 计算引用计数
  // the next line post-increments the reachable count.
  if (usable_count_[cindex_id]++ == 0 &&
      computable_info_[cindex_id] != kNotComputable) {
    // 所有cindex_id的依赖cindex_id 也都递归的增加引用计数.
    // 被高层引用多的cindex 就会有更多的计算引用计数
    // 虽然是递归向下的, 但是由于是从顶层, 开始向next_queue_ 添加 cindex
    // 并且添加完一层依赖之后, 就remove, 所以正常一个cindex,的usable_count = 2
    // 1 在作为高层的cindex的依赖添加进来时, 加入next_queue_, 是设置为1
    // 2 在下一轮中从next_queue_逐个被在找其自己依赖时, 在 上面 ++, 后面就移除了, 一般==2
    std::vector<int32>::const_iterator
        iter = graph_->dependencies[cindex_id].begin(),
        end = graph_->dependencies[cindex_id].end();
    for (; iter != end; ++iter) {
      int32 dep_cindex_id = *iter;
      IncrementUsableCount(dep_cindex_id);
    }
  }
}


void ComputationGraphBuilder::DecrementUsableCount(int32 cindex_id) {
  KALDI_PARANOID_ASSERT(static_cast<size_t>(cindex_id)<usable_count_.size());
  KALDI_PARANOID_ASSERT(usable_count_[cindex_id] > 0);
  if (--usable_count_[cindex_id] == 0 &&
      computable_info_[cindex_id] != kNotComputable) {
    std::vector<int32>::const_iterator
        iter = graph_->dependencies[cindex_id].begin(),
        end = graph_->dependencies[cindex_id].end();
    for (; iter != end; ++iter) {
      int32 dep_cindex_id = *iter;
      DecrementUsableCount(dep_cindex_id);
    }
  }
}


void ComputationGraphBuilder::BuildGraphOneIter() {
  
  // 当前从output需要向input方向进行搜索路径的 cindex_id 
  while (!current_queue_.empty()) {
    int32 cindex_id = current_queue_.back();
    current_queue_.pop_back();
    KALDI_ASSERT(computable_info_[cindex_id] == kUnknown);
    
    // 判断当前的cindex_id 是否是计算路径的一部分
    //    因为是 从output 向input 寻路, 所以一直都应该 > 0
    if (usable_count_[cindex_id] == 0)
      SetAsWillNotCompute(cindex_id);
    else
      // 从 output 高层 向底层 逐步的进行查找依赖, 构建计算路径结构 NnetComputation的 cindex.
      AddDependencies(cindex_id);
  }
  current_queue_.swap(next_queue_);  // now next_queue_ will be empty.
  current_distance_++;
}

void ComputationGraphBuilder::ComputeRequiredArray(
    int32 start_cindex_id,
    std::vector<bool> *required) const {

  int32 num_cindex_ids = graph_->cindexes.size();
  KALDI_ASSERT(num_cindex_ids >= start_cindex_id);
  KALDI_ASSERT(computable_info_.size() == num_cindex_ids);
  required->clear();
  required->resize(num_cindex_ids - start_cindex_id, false);

  // would be bool, but indexing c++ bool may be slow.
  std::vector<char> is_output_node(nnet_.NumNodes());
  for (int32 n = 0; n < nnet_.NumNodes(); n++)
    is_output_node[n] = (char)(nnet_.IsOutputNode(n) ? 1 : 0);

  std::vector<int32> queue;
  for (int32 c = start_cindex_id; c < num_cindex_ids; c++) {
    // First put the output cindex_ids into the queue.
    int32 node_id = graph_->cindexes[c].first;
    if (is_output_node[node_id]) {
      (*required)[c - start_cindex_id] = true;
      queue.push_back(c);
    }
  }
  while (!queue.empty()) {
    int32 c = queue.back();
    queue.pop_back();
    const std::vector<int32> &dependencies = graph_->dependencies[c];
    std::vector<int32>::const_iterator iter = dependencies.begin(),
        end = dependencies.end();
    for (; iter != end; ++iter) {
      int32 d = *iter;
      if (d >= start_cindex_id && !(*required)[d - start_cindex_id]){
        (*required)[d - start_cindex_id] = true;
        queue.push_back(d);
      }
    }
  }
  // just check that we don't have any cindex_ids which are required but have
  // usable_count_ == 0; this would indicate a bug somewhere.
  for (int32 c = start_cindex_id; c < num_cindex_ids; c++)
    KALDI_ASSERT(!((*required)[c - start_cindex_id] &&
                   (usable_count_[c] == 0)));

}


// make our own namespace for helper functions of ComputeComputationGraph.
namespace computation_graph {


// This function adds cindex_ids corresponding to each output
// index, to the graph.
void AddOutputToGraph(const ComputationRequest &request,
                      const Nnet &nnet,
                      ComputationGraph *graph) {
  int32 num_added = 0;
  for (int32 i = 0; i < request.outputs.size(); i++) {
    int32 n = nnet.GetNodeIndex(request.outputs[i].name);
    if (n == -1)
      KALDI_ERR << "Network has no output with name "
                << request.outputs[i].name;
    for (int32 j = 0; j < request.outputs[i].indexes.size(); j++) {
      Cindex cindex(n, request.outputs[i].indexes[j]);
      bool is_input = false, is_new;
      graph->GetCindexId(cindex, is_input, &is_new);  // ignore the return value.
      KALDI_ASSERT(is_new && "Output index seems to be listed more than once");
      num_added++;
    }
  }
  KALDI_ASSERT(num_added > 0 && "AddOutputToGraph: nothing to add.");
}


// This function adds cindex_ids corresponding to each input index, to the
// graph.
void AddInputToGraph(const ComputationRequest &request,
                     const Nnet &nnet,
                     ComputationGraph *graph) {
  int32 num_added = 0;
  for (int32 i = 0; i < request.inputs.size(); i++) {
    int32 n = nnet.GetNodeIndex(request.inputs[i].name);
    if (n == -1)
      KALDI_ERR << "Network has no input with name "
                << request.inputs[i].name;
    NodeType t = nnet.GetNode(n).node_type;
    KALDI_ASSERT((t == kInput || t == kComponent) &&
                 "Inputs to graph only allowed for Input and Component nodes.");

    for (int32 j = 0; j < request.inputs[i].indexes.size(); j++) {
      Cindex cindex(n, request.inputs[i].indexes[j]);
      bool is_input = true, is_new;
      graph->GetCindexId(cindex, is_input, &is_new);  // ignore the return value.
      KALDI_ASSERT(is_new && "Input index seems to be listed more than once");
      num_added++;
    }
  }
  KALDI_ASSERT(num_added > 0 && "AddInputToGraph: nothing to add.");
}


/**
   This function outputs to dependencies_subset[c], for each cindex_id c,
   the subset of elements d of graph.dependencies[c] such that
   cindex_id_to_segment_and_epoch[d] == cindex_id_to_segment_and_epoch[c].  That is, it's
   the dependency graph of the entire computation, but removing
   links that go from one segment/epoch to another segment/epoch.  Topologically,
   'dependencies_subset' would therefore consist of a bunch of
   disconnected graphs.
*/
static void ComputeDependenciesSubset(
    const ComputationGraph &graph,
    const std::vector<int32> &cindex_id_to_segment_and_epoch,
    std::vector<std::vector<int32> > *dependencies_subset) {
  int32 num_cindex_ids = graph.cindexes.size();
  KALDI_ASSERT(cindex_id_to_segment_and_epoch.size() == num_cindex_ids);
  dependencies_subset->resize(num_cindex_ids);
  for (int32 cindex_id = 0; cindex_id < num_cindex_ids; cindex_id++) {
    int32 phase_index = cindex_id_to_segment_and_epoch[cindex_id];
    const std::vector<int32> &dependencies = graph.dependencies[cindex_id];
    std::vector<int32> &dep_subset = (*dependencies_subset)[cindex_id];
    int32 num_dep = dependencies.size();
    for (int32 i = 0; i < num_dep; i++) {
      int32 d = dependencies[i];
      if (cindex_id_to_segment_and_epoch[d] == phase_index)
        dep_subset.push_back(d);
    }
  }
}


// 这个函数计算cindex_ids 的确定epoch信息
// ComputeNnetComputationEpochs 计算一个map, 从 node-index 映射到 epoch-index
// 1 基本上, 首先被计算的node, 会有更小的epoch-index
// 2 所有一个强联通子图组件的 node 具有相同的epoch-index
// 3 在无环网络图中, 通常每个component都有自己的epoch-index(即每个component对应的cindex 都具有相同的epoch-index)
// 4 但在LSTM 这样的层component中, 每层LSTM 都有自己的epoch-index?
// 
// 完整的计算顺序, 会按照 epochs中的顺序, 我们会不关心这个函数的输出, 因为它涉及提供给输入的 cindex_id???


/// This function computes certain information about "epochs" of cindex_ids.

/// The function ComputeNnetComputationEpochs() from nnet-graph.h gives us a map
/// from the NetworkNode index to an index we call the "epoch" index:

/// 1 basically, nodes that are computed first have a lower epoch index, and
/// 2 all nodes that are part of strongly connected components(强联通组件) have the same
/// epoch index.  
/// 3 In an acyclic nnet graph each component will usually have
/// its own epoch index, 
/// 4 but in things like LSTMs, each LSTM layer (with multiple
/// components) will have its own epoch index.
///
/// The overall computation order that we compute, will respect this ordering
/// into epochs (except that outputs of nodes of type kComponent that are
/// actually provided as inputs to the network, won't be subject to these
/// limitations but will come first in the order)... we will just ignore the
/// output of this function as it concerns cindex-ids that are provided as input
/// to the network.
///



///  \param nnet [in] The neural net
///  \param graph [in] The computation graph
///  \param cindex_id_to_segment_and_epoch [out] 
//           映射 cindex_id -> segment_epoch 的map
//           这个map 会组合segment index 和 epoch index
//           A vector that maps cindex_id to
///          a number that is the same if two cindex_ids are in the same
///          segment and same epoch, and different otherwise.  This
///          number combines the segment index and the epoch index; the
///          details are not important to the calling code.
// 
///  \param epochs_per_segment [out]  
//            是计算图中的一系列cindex_ids, 通过 segment epoch 分割
//            This is a listing of all the
///           cindex_ids in the computation graph, divided up first
///           by segment and then by epoch.
// 
///  \param epoch_is_trivial [out] 
//            bool vector, epoch index 索引, 和epochs_per_segment的二级索引是相同的索引
//            如果对应的epoch 是一个但NnetworkNode, epoch_is_trivial[epoch_id] = true
//            只跟network网络结构有关.
//            A vector of bool, indexed by the epoch
///           index which is the same as the second index of
///           'epochs_per_segment', that's true if this epoch index corresponds
///           to just a single NetworkNode (and also true for epoch indexes
///           corresponding to inputs to the network, which will be the first
///           epoch of each segment).  This depends on the neural network
///           structure only.

static void ComputeEpochInfo(
    const Nnet &nnet,
    const ComputationGraph &graph,
    std::vector<int32> *cindex_id_to_segment_and_epoch,
    std::vector<std::vector<std::vector<int32 > > > *epochs_per_segment,
    std::vector<bool> *epoch_is_trivial) {



  // !!!!!!!!!!!!!!!! IMPORTANT !!!!!!!!!!!!!!!!!!!!!!!
  // 1 node_to_epochs 映射每个nnet node 到一个粗槽的计算顺序 epoch
  //   但是我们可能需要在cindex_idlevel 上计算一个更好的数序, 为了服务RNN这样的layer.

  // node_to_epoch maps each nnet node to an index >= 0 that tells us coarsely
  // what order to compute them in... but we may need to compute a finer
  // ordering at the cindex_id level in cases like RNNs.
  std::vector<int32> node_to_epoch;
  ComputeNnetComputationEpochs(nnet, &node_to_epoch);
  {
    std::ostringstream os;
    PrintIntegerVector(os, node_to_epoch);
    KALDI_VLOG(6) << "node_to_epoch: " << os.str();
  }
  // 2 所有节点对应的epoch都+=1, 因为我们要保留 0 用来表示 input node 的epoch.
  // Add one to the epoch numbering because we will be reserving
  // zero for inputs to the network, and we don't want to have to
  // prove that epoch number 0 would correspond only to inputs.
  for (int32 i = 0; i < node_to_epoch.size(); i++)
    node_to_epoch[i]++;


  // 3 some infos
  int32 num_nodes = nnet.NumNodes(),
      num_cindex_ids = graph.cindexes.size(),
      num_segments = graph.segment_ends.size(),
      // num_epoch_indexes 是最大的epoch数
      // 是所有segment共同的epoch-index, 所以下面每个segment的epoch信息都是 num_epoch_indexes个.
      //                  input-epoch(1) +      other-node epoch
      num_epoch_indexes = 1 + *std::max_element(node_to_epoch.begin(), node_to_epoch.end());
  
  // 节点数必须一致
  KALDI_ASSERT(node_to_epoch.size() == num_nodes);

  // 为每个segment都安排一个 epoch vector
  epochs_per_segment->clear();
  epochs_per_segment->resize(num_segments);

  // 4 epoch_to_num_nodes 的信息, 判断计算图是否简单.
  // 通过 node_to_epoch 计算 epoch_to_num_nodes
  // 每个epoch 实际包含的node数量(一般只有1)
  // epoch_to_num_nodes is only used so we know whether each epoch
  // index corresponds to multiple nodes; if it's just one node then we know
  // the computation is very simple and we can do an optimization.
  std::vector<int32> epoch_to_num_nodes(num_epoch_indexes, 0);
  for (int32 n = 0; n < num_nodes; n++)
    epoch_to_num_nodes[node_to_epoch[n]]++;
  // 根据 epoch_to_num_nodes 判断epoch 是否是 trivial(简单的,平常的)
  epoch_is_trivial->resize(num_epoch_indexes);
  for (int32 o = 0; o < num_epoch_indexes; o++) {
    KALDI_ASSERT(o == 0 || epoch_to_num_nodes[o] > 0);
    (*epoch_is_trivial)[o] = (epoch_to_num_nodes[o] <= 1);
  }


  // 5
  // cindex_id -> segment and epoch
  // 5.1 检查最后一个segment的划分点 是对的  == num_cindex_ids
  cindex_id_to_segment_and_epoch->resize(num_cindex_ids);
  KALDI_ASSERT(graph.segment_ends.back() == num_cindex_ids);
  // 5.2 检查每个segment
  int32 cur_segment_start = 0, cur_segment_end;
  for (int32 segment = 0; segment < num_segments; segment++) {
    cur_segment_end = graph.segment_ends[segment];
    
    // 当前segment-index 的所有epoch 信息 vector,
    std::vector<std::vector<int32> > &epochs = (*epochs_per_segment)[segment];
    epochs.resize(num_epoch_indexes);

    // 当前segment(request)的所有cindex_id
    for (int32 cindex_id = cur_segment_start; cindex_id < cur_segment_end; cindex_id++) {
      
      // 1 node_index
      int32 node_index = graph.cindexes[cindex_id].first,
      // 2 epoch index
          epoch_index = (graph.is_input[cindex_id] ? 0 : node_to_epoch[node_index]);
      // 3 获得 cindex_id -> segment's epoch_index 
      (*cindex_id_to_segment_and_epoch)[cindex_id] = epoch_index + segment * num_epoch_indexes;
      // 某个segment的epoch vector, 的具体epoch-index list上 push_back 这个 cindex_id
      epochs[epoch_index].push_back(cindex_id);
    }
    cur_segment_start = cur_segment_end;
  }
}


} // end namespace computation_graph


void ComputeComputationGraph(const Nnet &nnet,
                             const ComputationRequest &request,
                             ComputationGraph *graph) {
  using namespace computation_graph;
  // make sure graph is empty at the start.
  KALDI_ASSERT(graph->cindexes.empty());

  AddInputToGraph(request, nnet, graph);
  AddOutputToGraph(request, nnet, graph);

  // queue of cindex_ids to process.
  std::vector<int32> queue(graph->cindexes.size());
  for (int32 i = 0; i < graph->cindexes.size(); i++)
    queue.push_back(i);

  while (!queue.empty()) {
    int32 cindex_id = queue.back();
    queue.pop_back();
    if (static_cast<int32>(graph->dependencies.size()) <= cindex_id)
      graph->dependencies.resize(cindex_id + 1);

    if (graph->is_input[cindex_id])
      continue;
    Cindex cindex = graph->cindexes[cindex_id];

    // find the dependencies of this cindex.
    int32 n = cindex.first;
    const Index &index = cindex.second;
    const NetworkNode &node = nnet.GetNode(n);

    std::vector<Cindex> input_cindexes;

    // the following switch statement sets up "input_cindexes".
    switch (node.node_type) {
      case kDescriptor: {
        // desc describes how this node obtains its input from other nodes.
        const Descriptor &desc = node.descriptor;
        desc.GetDependencies(index, &input_cindexes);
        break;
      }
      case kComponent: {
        int32 c = node.u.component_index;
        const Component *component = nnet.GetComponent(c);
        std::vector<Index> input_indexes;
        component->GetInputIndexes(request.misc_info, index,
                                   &input_indexes);
        // each Component node should be preceded by a node that describes its
        // input, of type kDescriptor
        KALDI_ASSERT(nnet.GetNode(n-1).node_type ==
                     kDescriptor);

        input_cindexes.resize(input_indexes.size());
        for (size_t i = 0; i < input_indexes.size(); i++) {
          input_cindexes[i].first = n - 1;  // preceding node.
          input_cindexes[i].second = input_indexes[i];
        }
        break;
      }
      case kDimRange: {
        input_cindexes.resize(1);
        input_cindexes[0] = Cindex(node.u.node_index, index);
        break;
      }
      case kInput: default:
        // for kInput, you should have hit the "continue" statement above.
        KALDI_ERR << "Invalid node type";
    }
    std::vector<int32> &this_dep = graph->dependencies[cindex_id];

    int32 num_dependencies = input_cindexes.size();
    this_dep.resize(num_dependencies);
    for (size_t i = 0; i < num_dependencies; i++) {
      bool is_input = false, is_new;
      int32 dep_cindex_id = graph->GetCindexId(input_cindexes[i],
                                               is_input, &is_new);
      this_dep[i] = dep_cindex_id;
      if (is_new)
        queue.push_back(dep_cindex_id);
    }

    // remove duplicates of dependencies.
    SortAndUniq(&this_dep);
  }
}


static int32 SumVectorSizes(const std::vector<std::vector<int32> > &vec) {
  int32 ans = 0;
  std::vector<std::vector<int32> >::const_iterator iter = vec.begin(),
      end = vec.end();
  for (; iter != end; ++iter)
    ans += iter->size();
  return ans;
}

static int32 SumVectorSizes(const std::vector<std::vector<std::vector<int32> > > &vec) {
  int32 ans = 0;
  for (size_t i = 0; i < vec.size(); i++)
    ans += SumVectorSizes(vec[i]);
  return ans;
}


/*
  this function is called from ComputeComputationPhases; it handles the part of
  the computation from one epoch (this code was broken out to avoid that
  function being super-long).  Note: the phases are a numbered grouping of
  cindexes that say in what order we compute things, i.e. we first compute
  all the cindexes for phase 0, then for phase 1, and so on.

   @param [in] nnet       The neural net this computation is for
   @param [in] graph      The computation graph we're computing the phases for.

   @param [in] this_epoch The sorted list of the cindex_ids for this epoch; note,
                          cindex_ids are indexes into the array graph.cindexes.
                          Roughly speaking, this is a list of the cindex_ids that
                          correspond to one "layer" of the neural network, in
                          things like LSTMs, or for one part of one layer (the
                          affine component, the nonlinearity, or the splicing),
                          in things like TDNNs.
  @param [in] dependencies_subset  A subset of 'graph.dependencies' corresponding
                          just to dependencies within the same epoch (not specifically
                          this epoch; for all epochs).  In general, for a cindex_id c
                          dependencies[c] is a list of other cindex_ids d1, d2,
                          such that in order to compute c we must first compute
                          d1, d2 and so on (plus d1, d2, etc. must be from the
                          same epoch as c).
  @param [in] depends_on_subset  The graph-transpose of dependencies_subset;
                          for cindex_id c, depends_on_subset[c] is the list
                          of cindex_ids that directly depend on cindex_id c,
                          so c must be computed before them.
  @param [in] epoch_is_trivial  A bool that's true if this epoch is trivial
                          (meaning it consists of just one component)... this
                          enables a faster code path in this common case.
  @param [in,out] phase_indexes  This vector, to some elements of which this
                          function writes each time it is called, maps from
                          cindex_id to the 'phase index'.  A phase index is a
                          number identifying the phases [like coarse steps] of
                          the computation, with zero for the first phase, one
                          for the second, etc.  We work out how many phase
                          indexes have been used already by previous epochs,
                          from phases->size().  Actually, phase_indexes is
                          really just a temporary variable used by this
                          function, that we allocate outside this function for
                          efficiency.  It is initialized to -1 outside this
                          function; different invocations of this function work
                          with different non-overlapping elements of the vector.
                          @param [in,out] phases This is the output of this
                          function.  Each time we add a new phase, we append a
                          vector to *phases.  E.g. (*phases)[0] is the sorted
                          list of cindexes in the first phase of the
                          computation... and so on.  Note, this function is
                          called multiple times, and each time we add one or
                          more phases to this vector, so its size grows.
*/
static inline void ComputeComputationPhasesForEpoch(
    const Nnet &nnet,
    const ComputationGraph &graph,
    const std::vector<int32> &this_epoch,
    const std::vector<std::vector<int32> > &dependencies_subset,
    const std::vector<std::vector<int32> > &depend_on_subset,
    bool epoch_is_trivial,
    std::vector<int32> *phase_indexes,
    std::vector<std::vector<int32> > *phases) {

  // --------------------------------
  // phase(计算阶段)
  // --------------------------------

  // phases, 是一个segment 里的 phases, 
  //    基本上和epoch 应该是一样的， 不过会有一些差异
  //    phase 是 比epoch 更加准确的 对cindex的截断划分.
  // phases-0
  //          cindex_id_0
  //          cindex_id_2
  //          cindex_id_4
  // phases-1
  //          cindex_id_1
  //          cindex_id_3
  //          cindex_id_5
  // phases-2
  //          cindex_id_12
  //          cindex_id_13
  // ....
  // vector<  vector<int32>>




  std::vector<int32> this_phase, next_phase_candidates;

  if (this_epoch.empty())
    return;

  // ---------------------------------------
  // 1 将epoch 中简单依赖关系的 cindex 加入到 cur_phase
  // ---------------------------------------
  // 如果epoch 是简单的, 那么就认为epoch内的cindex 就应该一起计算
  // 直接cp 为 phase
  // 即 
  //   1 epoch 简单
  //      一个epoch 一个phases
  //   2 如果不是简单的
  //      那么会将epoch 分为多个phases
  //      将没有相同epoch-index的依赖cindex_ids 的 cindex_ids 作为第一个phases
  //      剩余的那些,在继续添加作为后续的phases.
  if (epoch_is_trivial) { // an optimization
    this_phase = this_epoch;
  } 
  // not trivial, 那么需要根据 依赖关系等 分割 epochs
  else {
    // 现将那些没有相同epoch依赖的cindex 加入进去, 先计算
    // (难道是认为这样的够简单?)
    // Start out with all elements of this epoch that have no
    // dependencies within the same epoch (i.e. those that
    // can be computed first).
    std::vector<int32>::const_iterator iter = this_epoch.begin(),
        end = this_epoch.end();
    for (; iter != end; ++iter) {
      int32 cindex_id = *iter;
      if (dependencies_subset[cindex_id].empty())
        this_phase.push_back(cindex_id);
    }
  }

  // if the next assert fails, the graph at the level of cindex_ids is not acyclic.
  KALDI_ASSERT(!this_phase.empty() &&
               "Trying to process computation with cycles");


  // ---------------------------------------
  // 2 将epoch 中剩余的 cindex 添加到下一个phases ---- next_phase
  // ---------------------------------------
  // 从刚才计算好了的 this_phase 计算出 phases 输出参数.
  while (!this_phase.empty()) {

    // ---------------------------------------------
    // phases 扩大一个位置，即新增加一个phase(计算阶段)
    // ---------------------------------------------
    // The next two lines are a more efficient version of:
    // phases->push_back(this_phase);
    phases->resize(phases->size() + 1);
    phases->back().swap(this_phase);

    // 下面的 if-语句 是一个优化
    // 如果 this_epoch 的cindex都是相同node, 那我们就可以跳过剩下的循环了
    // Note: 
    // 如果 epoch==0, 可能是来自多个不同的input的cindex_ids, 因为所有input 都是 epoch == 0
    // 但是他们还是会没有依赖, 所以也放在第一个计算, 也能跳过剩下的code
    // The next if-statement is an optimization: if for this epoch index
    // there is just one node, we can skip the rest of this loop.  Note: if
    // epoch == 0, even if there is just one node, cindex_ids from
    // multiple nodes may be put here because of the rule that cindex_ids which
    // are inputs always get epoch 0.  But it's still true that they
    // will have no dependencies, so we can still skip the code below.
    if (epoch_is_trivial)
      return;

    // 当前正在处理的 phase_index.
    int32 cur_phase_index = phases->size() - 1;

    // ---------------------------------------
    // 2 将epoch 中剩余的 cindex 添加到下一个phases ---- next_phase
    // 因为当前已经计算了一个 phase
    // 当前epoch 如果不是trivial，那么其中 还会有剩下的 cindex 可以添加到下一个phases
    // 这样将 epoch 进一步细分.

    // next_phases_candidates is a list of cindexes that we should check
    // whether they are computable now, because one of the things they depend
    // on just became computable.
    next_phase_candidates.clear();
    std::vector<int32>::const_iterator this_phase_iter = phases->back().begin(),
                                       this_phase_end = phases->back().end();
    // 当前处理的 phase 下的所有cindex
    for (; this_phase_iter != this_phase_end; ++this_phase_iter) {
      int32 c = *this_phase_iter;  // c is a cindex_id with phase cur_phase_index.
      // 安排cur_phase下的 cindex 所属 phase 标记为 cur_phase_index.
      (*phase_indexes)[c] = cur_phase_index;
      // 所有依赖于 当前cindex的 后继cindex， 都可以加入到 next_phases
      std::vector<int32>::const_iterator iter = depend_on_subset[c].begin(),
                                         end = depend_on_subset[c].end();
      for (; iter != end; ++iter) {
        int32 d = *iter;  // cindex_id that depends on c.
        next_phase_candidates.push_back(d);
      }
    }
    SortAndUniq(&next_phase_candidates);



    // note, at this point 'this_phase' will be the empty vector [see the 'swap'
    // above].
    this_phase.reserve(next_phase_candidates.size());
    // now check the candidates that might be in the next phase, and put any
    // members that we are currently able to compute into "this_phase".
    std::vector<int32>::const_iterator iter = next_phase_candidates.begin(),
        end = next_phase_candidates.end();
    for (; iter != end; ++iter) {
      int32 c = *iter;
      std::vector<int32>::const_iterator
          dep_iter = dependencies_subset[c].begin(),
          dep_end = dependencies_subset[c].end();
      for (; dep_iter != dep_end; ++dep_iter) {
        int32 d = *dep_iter;  // d is cindex_id that c depends on.
        if ((*phase_indexes)[d] < 0)  // we can't compute c yet because something we depend
          break;                      // on has not yet been computed.
      }
      if (dep_iter == dep_end) {
        // we reached the end and did not break -> all dependencies satisfied
        this_phase.push_back(c);
      }
    }
    if (!next_phase_candidates.empty() && this_phase.empty())  {
      // this should have been caught earlier so likely a code error rather than
      // a problem with user input.
      KALDI_ERR << "Your model has a type of recurrence that cannot be computed. "
                << "E.g. if x[t] depends on both x[t+1] and x[t-1]... no order "
                << "of computation will work.";
    }
  }
}

void ComputeComputationPhases(
    const Nnet &nnet,
    const ComputationGraph &graph,
    std::vector<std::vector<std::vector<int32> > > *phases_per_segment) {

  using namespace computation_graph;
  // 当前request计算图的计算节点.
  int32 num_cindex_ids = graph.cindexes.size();

  // 1 计算segment 下 对cindex 的 更粗糙分组 epoch
  // segment--- request
  //     epochs 
  //        cindexes
  std::vector<int32> cindex_id_to_segment_and_epoch;
  std::vector<std::vector<std::vector<int32 > > > epochs_per_segment;
  std::vector<bool> epoch_is_trivial;

  ComputeEpochInfo(nnet, graph, &cindex_id_to_segment_and_epoch,
                   &epochs_per_segment, &epoch_is_trivial);
  KALDI_ASSERT(SumVectorSizes(epochs_per_segment) == num_cindex_ids);


  // !!!!!!!!!!
  // epoch 实际上是描述的一个粗糙顺序
  // 所以需要 dependencies_subset, depend_on_subset 来进一步确定顺序
  // ------ phases
  // !!!!!!!!!!

  // 2 计算依赖子集 相同epoch计算次序的 dependencies_subset
  // 依赖子集 只保存  和cindex_id 的具有相同epoch的依赖cindex. 
  // 这样就能够修正一个epoch内的cindexes 的计算顺序
  // 每个 cindex 的 vector<cindex> 
  // cindexes
  //      vector<cindex_denpendces> 
  //      vector<cindex_denpendces> 
  //      vector<cindex_denpendces>
  // 这个 dependencies_subset 会是 depend_on list的一个子集

  // dependencies_subset contains just the subset of dependencies
  // of each cindex_id, that have the same epoch index as
  // cindex_id itself.  This will be used to correctly order
  // cindexes within a certain epoch (relevant for things like
  // LSTMs).
  std::vector<std::vector<int32> > dependencies_subset;
  ComputeDependenciesSubset(graph, cindex_id_to_segment_and_epoch,
                            &dependencies_subset);
  // destroy cindex_id_to_segment_and_epoch, it's no longer needed.
  { std::vector<int32> temp; temp.swap(cindex_id_to_segment_and_epoch);  }

  // 3 计算depend_on 的一个子集
  // 限制那些具有相同epoch-index 的 depend_on 的
  // depend_on_subset is a subset of the normal "depend_on" list (i.e. a list of
  // all cindex_ids that depend on the current cindex_id), limited to just those
  // cindex_ids that have the same epoch index.
  std::vector<std::vector<int32> > depend_on_subset;
  ComputeGraphTranspose(dependencies_subset, &depend_on_subset);

  // 所有epoch cnt (epoch_is_trivial 是所有epoch 的标记, 所以size 就是全部count)
  int32 num_epoch_indexes = epoch_is_trivial.size(),
      num_segments = graph.segment_ends.size();

  // "phase_indexes" is used inside ComputeComputationPhasesForEpoch.
  std::vector<int32> phase_indexes(num_cindex_ids, -1);

  // ---------------------------------------------
  // phases(计算阶段), 类似 epoch(粗糙计算阶段), 
  // 根据前面的 dependences_subset, depend_on_subset
  // 进一步详细的划分 cindex 计算流程
  // ---------------------------------------------
  phases_per_segment->clear();
  phases_per_segment->resize(num_segments);

  for (int32 segment = 0; segment < num_segments; segment++) {
    phases_per_segment->reserve(50);  // minimize unnecessary copies.  50 is
                                      // very arbitrarily chosen.
    for (int32 epoch = 0; epoch < num_epoch_indexes; epoch++)
      ComputeComputationPhasesForEpoch(nnet, graph,
                                       epochs_per_segment[segment][epoch],
                                       dependencies_subset,
                                       depend_on_subset,
                                       epoch_is_trivial[epoch],
                                       &phase_indexes,
                                       &((*phases_per_segment)[segment]));
  }


  // make sure everything was computable.  If the next assert fails it's likely
  // a bug in this function or in PruneComputataionGraph.
  KALDI_ASSERT(SumVectorSizes(*phases_per_segment) == num_cindex_ids);
}









CindexSet::CindexSet(const ComputationGraph &graph):
    graph_(graph), is_computable_(NULL) { }

CindexSet::CindexSet(const ComputationGraph &graph,
                     const std::vector<char> &is_computable,
                     bool treat_unknown_as_computable):
    graph_(graph), is_computable_(&is_computable),
    // unknown 就是kUnknown , 不认为是可以计算的.
    treat_unknown_as_computable_(treat_unknown_as_computable) { }


// 只有在 从output 向input 找寻依赖
// 与 从input 向 output通知可计算 相连接之后, 后半部分的cindex才能确定是否 kComputable.
bool CindexSet::operator () (const Cindex &cindex) const {
  int32 cindex_id = graph_.GetCindexId(cindex);
  if (cindex_id == -1) {
    return false;
  } else {
    // 实际是 computationGraph 的computable_info_
    // computable_info_ 一开始 在 AddInput, AddOutput, 时 
    // 至少添加了 kComputable的 input Index
    //          kUnknown 的 output Index.
    if (is_computable_ == NULL) {
      return true;
    } else {
      ComputationGraphBuilder::ComputableInfo
          c = static_cast<ComputationGraphBuilder::ComputableInfo>(
              ((*is_computable_)[cindex_id]));
      
      // normal: false
      if (treat_unknown_as_computable_)
        return (c == ComputationGraphBuilder::kComputable ||
                c == ComputationGraphBuilder::kUnknown);
      else
        // kComputable 才返回true
        return (c == ComputationGraphBuilder::kComputable);
    }
  }
}

IndexSet::IndexSet(const ComputationGraph &graph,
                   const std::vector<char> &is_computable,
                   int32 node_id,
                   bool treat_unknown_as_computable):
    graph_(graph), is_computable_(is_computable), node_id_(node_id),
    treat_unknown_as_computable_(treat_unknown_as_computable) { }

bool IndexSet::operator () (const Index &index) const {
  int32 cindex_id = graph_.GetCindexId(Cindex(node_id_, index));
  if (cindex_id == -1) {
    return false;
  } else {
    ComputationGraphBuilder::ComputableInfo
        c = static_cast<ComputationGraphBuilder::ComputableInfo>(
            is_computable_[cindex_id]);
    if (treat_unknown_as_computable_)
      return (c == ComputationGraphBuilder::kComputable ||
              c == ComputationGraphBuilder::kUnknown);
    else
      return (c == ComputationGraphBuilder::kComputable);
  }
}


ComputationStepsComputer::ComputationStepsComputer(
    const Nnet &nnet,
    ComputationGraph *graph,
    std::vector<std::vector<int32> > *steps,
    std::vector<std::pair<int32, int32> > *locations):
    nnet_(nnet), graph_(graph), steps_(steps), locations_(locations) {
  steps_->clear();
  locations_->clear();
  // 所有的cindexes
  int32 num_cindexes = graph_->cindexes.size();
  // leave a little space in case a few cindexes are added (unlikely
  // but could happen with dim-range nodes).
  // reserve 可以申请的最大空间(一般留一些空间余量)
  locations_->reserve(num_cindexes + num_cindexes / 10);
  // resize  申请确定空间
  locations_->resize(num_cindexes, std::pair<int32,int32>(-1, -1));
}

// request, 一个具体输入输出 batch
// phases,  描述request的segment段的phases
void ComputationStepsComputer::ComputeForSegment(
    const ComputationRequest &request,
    const std::vector<std::vector<int32> > &phases) {

  int32 this_num_phases = phases.size();
  //将比epoch稍微精细的本phase,按照内部相同的node-index 再进行划分 为 sub_phases.
  for (int32 i = 0; i < this_num_phases; i++) {
    std::vector<std::vector<Cindex> > sub_phases;
    SplitIntoSubPhases(phases[i], &sub_phases);

    // 每个phase 分成的vector<sub_phase>
    // 对每个sub_phase 安排为相同的 step-index， 
    // 这样的sub_phase 足够精细， 可以同时计算
    // 每个sub_phase 中step之间具有相同的step-index, 不同的row-index.
    for (size_t j = 0; j < sub_phases.size(); j++) {
      ProcessSubPhase(request, sub_phases[j]);
    }
  }
}

// 为sub_phase, 找对应的request(vector<IoSpecification(NnetIo)>) 中的数据Index, 
// 安排对应的Index 为一个 step
void ComputationStepsComputer::ProcessInputOrOutputStep(
    const ComputationRequest &request,
    bool is_output,
    const std::vector<Cindex> &sub_phase) {
  int32 io_node = sub_phase[0].first;
  if (is_output){
    KALDI_ASSERT(nnet_.IsOutputNode(io_node));
  } else {
    KALDI_ASSERT(nnet_.IsInputNode(io_node));
  }

  // 获得对应的node_name
  std::string node_name = nnet_.GetNodeName(io_node);

  // 遍历所有的IoSpecification(NnetIo)
  // 一个IoSpecification , 确定是 inputs, ivectors, outputs 中的一个
  // 最终所有的name 肯定都是 node_name
  const std::vector<IoSpecification> &inputs_or_outputs =
        (is_output ? request.outputs : request.inputs);
  int32 io_index = -1;
  for (size_t i = 0; i < inputs_or_outputs.size(); i++)
      if (inputs_or_outputs[i].name == node_name)
        io_index = i;
  KALDI_ASSERT(io_index >= 0);

  // 对应的所有indexes -> Cindexes
  // 并且一个input(output)的cindex一定都在一个sub_phase中.
  const std::vector<Index> &io_indexes = inputs_or_outputs[io_index].indexes;
  std::vector<Cindex> io_cindexes(io_indexes.size());
  for (size_t i = 0, size = io_cindexes.size(); i < size; i++) {
    io_cindexes[i].first = io_node;
    io_cindexes[i].second = io_indexes[i];
  }
  // 必须: 如果是input output里面， 那么io_cindexes的cindexes 必然是在同一个sub_phase里面了
  // 因为经过之前 epoch -> phase -> sub_phase, 一定已经是最优的了.
  // we expect the list of cindexes in 'io_cindexes' to be identical to
  // that in 'sub_phase' (but they don't have to be in the same order)... for now we check the size, we'll spot-check
  // that they are the same later.
  KALDI_ASSERT(io_cindexes.size() == sub_phase.size());

  // 为一个sub_phase 安排一个 step.
  // The actual output in 'steps' must be in the same order as
  int32 step_index = AddStep(io_cindexes);

  // 现在进行逐个点的check, 
  // sub_phase(step)里的cindexes 和我们刚刚添加的一致
  // 不一定完全一样的顺序， 但是要在一个固定集合.
  // Now spot-check that the cindexes in 'sub_phase' are the same as those
  // we just added.  [note: they don't have to be in the same order, but
  // they should be the same set.]
  for (size_t i = 0; i < sub_phase.size(); i += 10) {
    const Cindex &cindex = sub_phase[i];
    int32 cindex_id = graph_->GetCindexId(cindex);
    KALDI_ASSERT(cindex_id >= 0 && (*locations_)[cindex_id].first == step_index);
  }
}

int32 ComputationStepsComputer::AddStep(const std::vector<Cindex> &cindexes,
                                        bool add_if_absent) {
  // note: we can't assert that cindexes is nonempty, because it's possible for
  // input steps for GeneralComponents to be empty if they require no input
  // indexes; and because the compiler code expects component steps to be
  // preceded by component-input steps, we can't just omit these empty steps.
  // [note: a component-input step is about preparing the input for a component's
  // propagation.]
  int32 step_index = steps_->size();
  steps_->push_back(std::vector<int32>());
  std::vector<int32> &step = steps_->back();  // vector of cindex_id.
  step.resize(cindexes.size());
  size_t row_index = 0;
  std::vector<Cindex>::const_iterator iter = cindexes.begin(),
      end = cindexes.end();
  std::vector<int32>::iterator out_iter = step.begin();
  std::pair<int32, int32> *locations = &((*locations_)[0]);
  if (!add_if_absent) {
    // this version of GetCindexId will not add CindexIds, and
    // will crash if CindexIds not present in the graph are
    // encountered.
    for (; iter != end; ++iter, ++out_iter, ++row_index) {
      int32 cindex_id = graph_->GetCindexId(*iter);
      *out_iter = cindex_id;
      locations[cindex_id].first = step_index;
      locations[cindex_id].second = row_index;
    }
  } else {
    for (; iter != end; ++iter, ++out_iter, ++row_index) {
      bool is_input = false;  // only relevant if we have to add the cindex to
                              // the computation graph, which we won't for
                              // inputs (we only might for dim-range nodes
                              // and for the component-input and component
                              // steps of non-simple Components.
      bool added;
      int32 cindex_id = graph_->GetCindexId(*iter, is_input, &added);
      *out_iter = cindex_id;
      if (added) {
        KALDI_ASSERT(cindex_id == static_cast<int32>(locations_->size()));
        locations_->resize(cindex_id + 1);
        locations_->back().first = step_index;
        locations_->back().second = row_index;
        locations = &((*locations_)[0]);  // in case it was reallocated
      } else {
        locations[cindex_id].first = step_index;
        locations[cindex_id].second = row_index;
      }
    }
  }
  return step_index;
}


int32 ComputationStepsComputer::AddStep(std::vector<int32> *cindex_ids) {
  
  int32 step_index = steps_->size();
  // steps_ 中增加一个 简易step ---- vector<cindex_ids>
  // 后续会将这个简易step构成StepInfo
  steps_->push_back(std::vector<int32>());
  steps_->back().swap(*cindex_ids);

  // 对 cindex_ids 安排 对应location
  std::vector<int32>::const_iterator iter = steps_->back().begin(),
        end = steps_->back().end();
  int32 row_index = 0;
  // locations 描述 locations_ 起始内存位置， 通过 [] 方式进行索引其他位置
  std::pair<int32,int32> *locations = &((*locations_)[0]);
  size_t num_cindexes = graph_->cindexes.size();
  // 逐个cindex_id 安排具体计算位置? row_index 目的是什么
  for (; iter != end; ++iter, ++row_index) {
    int32 cindex_id = *iter;
    KALDI_ASSERT(static_cast<size_t>(cindex_id) < num_cindexes);
    locations[cindex_id].first = step_index;
    locations[cindex_id].second = row_index;
  }
  return step_index;
}

// 将 cindex_ids 里保存的 cindex_id, 取出对应的Cindex对象
// 存入 cindexes
void ComputationStepsComputer::ConvertToCindexes(
    const std::vector<int32> &cindex_ids,
    std::vector<Cindex> *cindexes) const {
  cindexes->resize(cindex_ids.size());
  size_t num_cindexes = graph_->cindexes.size();
  std::vector<int32>::const_iterator iter = cindex_ids.begin(),
      end = cindex_ids.end();
  std::vector<Cindex>::iterator out_iter = cindexes->begin();
  for (; iter != end; ++iter, ++out_iter) {
    int32 cindex_id = *iter;
    KALDI_ASSERT(static_cast<size_t>(cindex_id) < num_cindexes);
    *out_iter = graph_->cindexes[cindex_id];
  }
}


void ComputationStepsComputer::ConvertToCindexIds(
    const std::vector<Cindex> &cindexes,
    std::vector<int32> *cindex_ids) const {
  cindex_ids->resize(cindexes.size());
  std::vector<Cindex>::const_iterator iter = cindexes.begin(),
      end = cindexes.end();
  std::vector<int32>::iterator out_iter = cindex_ids->begin();
  for (; iter != end; ++iter, ++out_iter) {
    int32 cindex_id = graph_->GetCindexId(*iter);
    KALDI_ASSERT(cindex_id >= 0);
    *out_iter = cindex_id;
  }
}


// static
void ComputationStepsComputer::ConvertToIndexes(
    const std::vector<Cindex> &cindexes,
    std::vector<Index> *indexes) {
  indexes->resize(cindexes.size());
  std::vector<Cindex>::const_iterator iter = cindexes.begin(),
      end = cindexes.end();
  std::vector<Index>::iterator out_iter = indexes->begin();
  for (; iter != end; ++iter, ++out_iter)
    *out_iter = iter->second;
}

// static
void ComputationStepsComputer::ConvertToCindexes(
    const std::vector<Index> &indexes,
    int32 node_index,
    std::vector<Cindex> *cindexes) {
  KALDI_ASSERT(node_index >= 0);
  cindexes->resize(indexes.size());
  std::vector<Index>::const_iterator iter = indexes.begin(),
      end = indexes.end();
  std::vector<Cindex>::iterator out_iter = cindexes->begin();
  for (; iter != end; ++iter, ++out_iter) {
    out_iter->first = node_index;
    out_iter->second = *iter;
  }
}



// step==sub_phase
// segment -> epoch -> phase -> sub_phase
void ComputationStepsComputer::ProcessComponentStep(
    const std::vector<Cindex> &step) {
  KALDI_ASSERT(!step.empty());
  // 当前step 实际上是 sub_phase -- vector< Cindex >
  //                                      <node-id, <t, n, x>>
  // 如果是kComponent-node, 那么 node-id-1, 必然是前继的 kDescriptor-node
  int32 component_node_index = step.front().first;
  int32 component_input_index = component_node_index - 1;
  KALDI_ASSERT(nnet_.IsComponentNode(component_node_index));

  // 获得对应的NetworkNode, 和 对应的 component-index --索引到计算组建Component
  const NetworkNode &node = nnet_.GetNode(component_node_index);
  int32 c = node.u.component_index;
  const Component *component = nnet_.GetComponent(c);
  if (component->Properties() & kSimpleComponent) {
    // 如果是个简单的 kSimpleComponent
    // for simple components, the input cindexes will be the same as the
    // output ones except for the node index, so we do a shortcut that's
    // faster (no following dependencies).
    std::vector<Cindex> input_step(step.size());
    input_step.resize(step.size());
    std::vector<Cindex>::iterator iter = input_step.begin(),
        end = input_step.end();
    std::vector<Cindex>::const_iterator src = step.begin();
    for (; iter != end; ++iter,++src) {
      iter->first = component_input_index;
      iter->second = src->second;
    }
    AddStep(input_step);
    AddStep(step);
  } else {
    std::vector<int32> step_cindex_ids;
    ConvertToCindexIds(step, &step_cindex_ids);
    // to get the input cindexes we need to follow dependencies back.
    unordered_set<int32> input_cindex_ids;
    std::vector<int32>::iterator iter = step_cindex_ids.begin(),
        end = step_cindex_ids.end();
    for (; iter != end; ++iter) {
      int32 c = *iter;
      const std::vector<int32> &dependencies = graph_->dependencies[c];
      std::vector<int32>::const_iterator dep_iter = dependencies.begin(),
          dep_end = dependencies.end();
      for (; dep_iter != dep_end; ++dep_iter) {
        int32 d = *dep_iter;
        input_cindex_ids.insert(d);
      }
    }
    // Convert to Cindexes so we can sort them as Cindexes.
    std::vector<Cindex> input_step;
    input_step.reserve(input_cindex_ids.size());
    unordered_set<int32>::iterator set_iter = input_cindex_ids.begin(),
        set_end = input_cindex_ids.end();
    for (; set_iter != set_end; ++set_iter) {
      int32 c = *set_iter;
      input_step.push_back(graph_->cindexes[c]);
    }

    // sort the input cindexes.
    std::sort(input_step.begin(), input_step.end());

    if (component->Properties() & kReordersIndexes) {
      std::vector<Index> indexes, input_indexes;
      ConvertToIndexes(input_step, &input_indexes);
      ConvertToIndexes(step, &indexes);


      size_t orig_size = indexes.size() + input_indexes.size();

      // the component wants to have the opportunity to change the
      // order of these indexes from their default.
      component->ReorderIndexes(&input_indexes, &indexes);

      bool added_padding = (orig_size != indexes.size() + input_indexes.size());

      // Now convert back from indexes to cindexes (we know the
      // node-index in each case)
      std::vector<Cindex> reordered_step;
      ConvertToCindexes(indexes, component_node_index, &reordered_step);
      ConvertToCindexes(input_indexes, component_input_index, &input_step);
      // the 'added_padding' argument becomes the 'add_if_absent' arg of
      // AddStep, so it knows to expect that it might have to add new CindexIds.
      AddStep(input_step, added_padding);
      AddStep(reordered_step, added_padding);
    } else {
      AddStep(input_step);
      // it's more efficient to add the step with cindex_ids; and we have these
      // available, so we do it that way.  (in the other branch where
      // the flag kReordersIndexes was present, we couldn't do this because
      // of the reordering).
      AddStep(&step_cindex_ids);
    }
  }
}


void ComputationStepsComputer::ConvertToLocations(
    const std::vector<int32> &cindex_ids,
    std::vector<std::pair<int32, int32> > *locations) const {
  locations->resize(cindex_ids.size());
  std::vector<int32>::const_iterator iter = cindex_ids.begin(),
      end = cindex_ids.end();
  std::vector<std::pair<int32, int32> >::iterator out_iter =
      locations->begin();
  // note, locations_ and locations are different variables.
  std::pair<int32, int32> *locations_ptr = &((*locations_)[0]);
  size_t num_cindexes = locations_->size();
  for (; iter != end; ++iter, ++out_iter) {
    int32 cindex_id = *iter;
    KALDI_ASSERT(static_cast<size_t>(cindex_id) < num_cindexes);
    int32 step = locations_ptr[cindex_id].first,
        row = locations_ptr[cindex_id].second;
    KALDI_ASSERT(step >= 0);
    out_iter->first = step;
    out_iter->second = row;
  }
}

void ComputationStepsComputer::ProcessDimRangeSubPhase(
    const std::vector<Cindex> &sub_phase) {
  int32 dim_range_node = sub_phase[0].first;
  KALDI_ASSERT(nnet_.IsDimRangeNode(dim_range_node));
  const NetworkNode &node = nnet_.GetNode(dim_range_node);
  // 'input_node_index' is the node index of the component or input node
  // that this dim-range node gets its input from.
  int32 input_node_index = node.u.node_index;
  // input_cindexes will give us the cindexes of the component or input node
  // that is the input to this dim-range node
  std::vector<Cindex> input_cindexes(sub_phase);
  for (std::vector<Cindex>::iterator iter = input_cindexes.begin(),
           end = input_cindexes.end(); iter != end; ++iter)
    iter->first = input_node_index;
  std::vector<int32> input_cindex_ids;
  ConvertToCindexIds(input_cindexes, &input_cindex_ids);
  std::vector<std::pair<int32, int32> > locations;
  ConvertToLocations(input_cindex_ids, &locations);
  std::sort(locations.begin(), locations.end());
  KALDI_ASSERT(!locations.empty());
  std::vector<std::pair<int32, int32> >::const_iterator
      locations_iter = locations.begin(),
      locations_end = locations.end();
  // Each unique .first number in locations (i.e. each source step, and they
  // will all correspond to component-output or input steps) will generate one
  // 'step' of type kDimRange.  Because dim-range nodes must be contiguous
  // ranges of a source step (since they are represented as sub-matrices), for
  // each source step we work out the first and last row-index (i.e. first and
  // last .second member of locations) and use that to reconstruct the range.

  // each element of 'steps' will be (source_step, (begin_row, end_row)) so that
  // the source of the dim-range node is indexes begin_row ... end_row-1 in that
  // source step.
  std::vector<std::pair<int32, std::pair<int32, int32> > > steps;

  int32 cur_source_step = locations_iter->first,
      cur_row_begin = locations_iter->second,
      cur_row_end = cur_row_begin + 1;
  while (1) {
    ++locations_iter;
    if (locations_iter == locations_end ||
        locations_iter->first != cur_source_step) {
      // we reached the end of a run of the same step.
      std::pair<int32, std::pair<int32, int32> > this_step;
      this_step.first = cur_source_step;
      this_step.second.first = cur_row_begin;
      this_step.second.second = cur_row_end;
      steps.push_back(this_step);
      if (locations_iter != locations_end) {
        cur_source_step = locations_iter->first;
        cur_row_begin = locations_iter->second;
        cur_row_end = cur_row_begin + 1;
      } else {
        break;
      }
    } else {
      cur_row_end = locations_iter->second + 1;
    }
  }

  for (size_t i = 0; i < steps.size(); i++) {
    // iterating over different source steps, although normally
    // there will be just one.
    int32 source_step = steps[i].first,
        row_begin = steps[i].second.first,
        row_end = steps[i].second.second;
    // 'source' is just the elements of the source step that we're consuming.
    std::vector<int32> source((*steps_)[source_step].begin() + row_begin,
                              (*steps_)[source_step].begin() + row_end);
    std::vector<Cindex> cindexes;
    ConvertToCindexes(source, &cindexes);
    std::vector<Cindex>::iterator iter = cindexes.begin(),
        end = cindexes.end();
    for (; iter != end; ++iter)
      iter->first = dim_range_node;
    bool add_if_absent = true;
    // this add_if_absent says, even if cindexes were not in the graph,
    // add them.  This is possible in principle; it's to satisfy the
    // requirement that DimRangeNodes be implemented as contiguous ranges
    // of rows of component nodes or input nodes.
    AddStep(cindexes, add_if_absent);
  }
}

void ComputationStepsComputer::ProcessSubPhase(
    const ComputationRequest &request,
    const std::vector<Cindex> &sub_phase) {
  KALDI_ASSERT(!sub_phase.empty());
  int32 node_index = sub_phase[0].first;
  KALDI_ASSERT(sub_phase.back().first == node_index);
  if (nnet_.IsComponentNode(node_index)) {
    ProcessComponentStep(sub_phase);
  } else if (nnet_.IsInputNode(node_index)) {
    ProcessInputOrOutputStep(request, false, sub_phase);
  } else if (nnet_.IsOutputNode(node_index)) {
    ProcessInputOrOutputStep(request, true, sub_phase);
  } else if (nnet_.IsDimRangeNode(node_index)) {
    // this might turn out to be multiple steps, see the code.
    ProcessDimRangeSubPhase(sub_phase);
  } else if (nnet_.IsComponentInputNode(node_index)) {
    // We actually do nothing with these sub-phases, because they are processed
    // when we process the associated component's sub-phase/step.  Doing it this
    // way resolves certain problems.
    return;
  } else {
    KALDI_ERR << "Unknown node type.";
  }
}


void ComputationStepsComputer::Check() const {
  int32 num_cindexes = graph_->cindexes.size();
  KALDI_ASSERT(locations_->size() == num_cindexes);
  for (int32 c = 0; c < num_cindexes; c++) {
    int32 step = (*locations_)[c].first,
        row = (*locations_)[c].second;
    if (!(step >= 0 && row >= 0 && (*steps_)[step][row] == c)) {
      // normally the 'locations' of cindexes should be unique, so we should
      // never normally reach this point; but it's not an error to have
      // duplicates of the cindexes used for 'padding' by the ReorderIndexes()
      // function of non-simple Components.  So we check whether that's the case
      // before we die.
      if (graph_->cindexes[c].second.t != kNoTime) {
        // if this happens it will likely require some debugging by Dan.
        KALDI_ERR << "Error in computing computation steps (likely code error)";
      }
    }

  }
}

void ComputationStepsComputer::SplitIntoSubPhases(
    const std::vector<int32> &phase,
    std::vector<std::vector<Cindex> > *sub_phases) const {
  
  std::vector<Cindex> phase_cindexes;
  // 将phase 中的 cindex_id 对应的Cindex->> phase_cindexes
  ConvertToCindexes(phase, &phase_cindexes);
  KALDI_ASSERT(!phase_cindexes.empty());
  std::sort(phase_cindexes.begin(), phase_cindexes.end());

  // segment_begins 表示的是 phase里面所有Cindexes （sort 排序的）
  // 按照 node-index 顺序， 对Cindexes的一个划分， 这个划分就是 sub_phase
  // Note: segment 表示的实际是是 sub_phase, 命名有点瑕疵.
  // 'sub_phase_begins' is the indexes onto 'phase_cindexes' that
  // start a run of the same node-index
  std::vector<size_t> segment_begins;
  int32 cur_node_index = -1;
  // phase_cindexes的总数
  size_t size = phase_cindexes.size();
  for (size_t i = 0; i < size; i++) {
    if (phase_cindexes[i].first != cur_node_index) {
      cur_node_index = phase_cindexes[i].first;
      segment_begins.push_back(i);
    }
  }
  
  // 应该对 phase 进行进一步划分的总数
  size_t num_sub_phases = segment_begins.size();

  // 根据上面的 sub_phase 划分，将 cindexes 安划分，写入到 sub_phase vector中
  segment_begins.push_back(size);
  sub_phases->clear();
  sub_phases->resize(num_sub_phases);
  for (size_t i = 0; i < num_sub_phases; i++) {
    size_t this_begin = segment_begins[i],
          this_end = segment_begins[i+1];
    (*sub_phases)[i].insert((*sub_phases)[i].end(),
                            phase_cindexes.begin() + this_begin,
                            phase_cindexes.begin() + this_end);
  }
}



} // namespace nnet3
} // namespace kaldi
