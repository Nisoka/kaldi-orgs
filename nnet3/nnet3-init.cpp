// nnet3bin/nnet3-init.cc
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet3/nnet-nnet.h"
#include "hmm/transition-model.h"
#include "tree/context-dep.h"





//  ====================== nnet3 构建了 nnet3 结构
// componet , componet_names_
// nodes_   , nodes_names_
// 对应kDescritptor中descriptor的构建, kComponent 的 Component-index
//            
//  但是至今没有什么计算图的使用生成.
int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;

    const char *usage =
        "Initialize nnet3 neural network from a config file; -------- > outputs 'raw' nnet\n"
        "without associated information such as transition model and priors.\n"
        "Search for examples in scripts in /egs/wsj/s5/steps/nnet3/\n"
        "Can also be used to add layers to existing model (provide existing model\n"
        "as 1st arg)\n"
        "\n"
        "Usage:  nnet3-init [options] [<existing-model-in>] <config-in> <raw-nnet-out>\n"
        "e.g.:\n"
        " nnet3-init nnet.config 0.raw\n"
        "or: nnet3-init 1.raw nnet.config 2.raw\n"
        "See also: nnet3-copy, nnet3-info\n";

    bool binary_write = true;
    int32 srand_seed = 0;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("srand", &srand_seed, "Seed for random number generator");

    po.Read(argc, argv);
    srand(srand_seed);

    if (po.NumArgs() < 2 || po.NumArgs() > 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string raw_nnet_rxfilename = (po.NumArgs() == 3 ?
                                       po.GetArg(1) : std::string("")),
        config_rxfilename = po.GetArg(po.NumArgs() == 3 ? 2 : 1),
        raw_nnet_wxfilename = po.GetArg(po.NumArgs() == 3 ? 3 : 2);

    Nnet nnet;
    
    {
      bool binary;
      Input ki(config_rxfilename, &binary);
      KALDI_ASSERT(!binary && "Expect config file to contain text.");
      // =================== ReadConfig 读取init.config 生成nnet3结构. ===================
      nnet.ReadConfig(ki.Stream());
    }

    WriteKaldiObject(nnet, raw_nnet_wxfilename, binary_write);
    KALDI_LOG << "Initialized raw neural net and wrote it to "
              << raw_nnet_wxfilename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}




class Nnet {
 public:
  // This function can be used either to initialize a new Nnet from a config
  // file, or to add to an existing Nnet, possibly replacing certain parts of
  // it.  It will die with error if something went wrong.
  // Also see the function ReadEditConfig() in nnet-utils.h (it's made a
  // non-member because it doesn't need special access).
  void ReadConfig(std::istream &config_file);

  // the names of the components of the network.  Note, these may be distinct
  // from the network node names below (and live in a different namespace); the
  // same component may be used in multiple network nodes, to define parameter
  // sharing.
  std::vector<std::string> component_names_;

  // the components of the nnet, in arbitrary order.  The network topology is
  // defined separately, below; a given Component may appear more than once in
  // the network if necessary for parameter tying.
  std::vector<Component*> components_;

  // names of network nodes, i.e. inputs, components and outputs, used only in
  // reading and writing code.  Indexed by network-node index.  Note,
  // components' names are always listed twice, once as foo-input and once as
  // foo, because the input to a component always gets its own NetworkNode index.
  std::vector<std::string> node_names_;

  // the network nodes of the network.
  std::vector<NetworkNode> nodes_;

};



void Nnet::ReadConfig(std::istream &config_is) {

  // 保存一个nnet3所有ConfigLine 信息
  // 包括nnet3 之前存在的Component信息 Component-node信息
  // 以及新读取进来的Config file
  std::vector<std::string> lines;

  // 获得nnet3之前的Component信息等
  const bool include_dim = false;
  GetConfigLines(include_dim, &lines);

  // we'll later regenerate what we need from nodes_ and node_name_ from the
  // string representation.
  nodes_.clear();
  node_names_.clear();

  // 设置nnet3已经存在的结构信息 位置标记
  int32 num_lines_initial = lines.size();


  // 读取新加入来的Config file 的lines
  // 此时 lines包含
  // 1 nnet3 之前存在的结构 位置到 num_lines_initial 为截止
  // 2 从ConfigLines中读取 新的加入nnet3中的lines .
  ReadConfigLines(config_is, &lines);

  // 将lines 都变为 ConfigLine 结构
  std::vector<ConfigLine> config_lines(lines.size());
  ParseConfigLines(lines, &config_lines);

  // 至此, config 中的每行(component-node, key=value, key=value, ...)
  // 转化为ConfigLine, 其中Component-node 为first_token_,    data_ 保存所有的pair<key, value>
  // std::map<std::string, std::pair<std::string, bool> > data_;
  {
   
    void ParseConfigLines(const std::vector<std::string> &lines,
                          std::vector<ConfigLine> *config_lines) {
      config_lines->resize(lines.size());
      for (size_t i = 0; i < lines.size(); i++) {
        bool ret = (*config_lines)[i].ParseLine(lines[i]);
        if (!ret) {
          KALDI_ERR << "Error parsing config line: " << lines[i];
        }
      }
    }

    // 将line 生成 component-node + ConifgLine(data_ 键值对)
    bool ConfigLine::ParseLine(const std::string &line) {
      data_.clear();
      whole_line_ = line;
    
      size_t pos = 0, size = line.size();
      // 去除空格
      while (isspace(line[pos]) && pos < size) pos++;
    
      size_t first_token_start_pos = pos;

      // first get first_token_.
      // 查找space为截止点, 其中禁止出现=, 否则去掉当前位置 first_token这样就为空.
      while (!isspace(line[pos]) && pos < size) {
        if (line[pos] == '=') {
          // If the first block of non-whitespace looks like "foo-bar=...",
          // then we ignore it: there is no initial token, and FirstToken()
          // is empty.
          pos = first_token_start_pos;
          break;
        }
        pos++;
      }

      // 获得first_token_
      first_token_ = std::string(line, first_token_start_pos, pos - first_token_start_pos);

      // first_token_ 是一些类似component-node的Token标签, 实际上是一个比IsValidName定义的更加严格约束的名字.
      if (!first_token_.empty() && !IsValidName(first_token_))
        return false;

      // 循环取 first_token_中的内容, 构建具体对象Descriptor.
      while (pos < size) {
        // 去除 space
        if (isspace(line[pos])) {
          pos++;
          continue;
        }

        // OK, at this point we know that we are pointing at nonspace.
        // 找=.
        size_t next_equals_sign = line.find_first_of("=", pos);
        // 获得 = 之前的key
        std::string key(line, pos, next_equals_sign - pos);
        // check 是一个IsValidName()
        if (!IsValidName(key)) return false;

        // handle any quotes.  we support key='blah blah' or key="foo bar".
        // no escaping is supported.
        // 处理可能的 ' " value 引号. 或者没有' " value 引号,  获得value值.
        if (line[next_equals_sign+1] == '\'' || line[next_equals_sign+1] == '"') {
          char my_quote = line[next_equals_sign+1];
          size_t next_quote = line.find_first_of(my_quote, next_equals_sign + 2);
          if (next_quote == std::string::npos) {  // no matching quote was found.
            KALDI_WARN << "No matching quote for " << my_quote << " in config line '"
                       << line << "'";
            return false;
          } else {
            std::string value(line, next_equals_sign + 2,
                              next_quote - next_equals_sign - 2);
            data_.insert(std::make_pair(key, std::make_pair(value, false)));
            pos = next_quote + 1;
            continue;
          }
        } else {
          // we want to be able to parse something like "... input=Offset(a, -1) foo=bar":
          // in general, config values with spaces in them, even without quoting.
          size_t next_next_equals_sign = line.find_first_of("=", next_equals_sign + 1),
              terminating_space = size;
          // 找到下一个 =  即 key=value key=.
          if (next_next_equals_sign != std::string::npos) {  // found a later equals sign.
          
            size_t preceding_space = line.find_last_of(" \t", next_next_equals_sign);
            // terminating_space --- 分割两个key=value, 获得实际的key=value.
            if (preceding_space != std::string::npos &&
                preceding_space > next_equals_sign)
              terminating_space = preceding_space;
          }
          // 去除多余key=value后面的space
          while (isspace(line[terminating_space - 1]) && terminating_space > 0)
            terminating_space--;

          // 构建value
          std::string value(line, next_equals_sign + 1, terminating_space - (next_equals_sign + 1));
          // data_ 保存 pair<key,pair<value,false>>
          data_.insert(std::make_pair(key, std::make_pair(value, false)));
          // 循环构建 pari<key,value>
          pos = terminating_space;
        }
      }
      return true;
    }
  }

  
  // 可能会移除一些重复行 line, 有时会发生一个存在的node 或者Component 被在一个新的configfile中重新定义
  RemoveRedundantConfigLines(num_lines_initial, &config_lines);

  // 函数的输入 是一个nnet生成的config配置, 包含node信息, 可以与一个user提供的config提供.
  void Nnet::RemoveRedundantConfigLines(int32 num_lines_initial,
                                        std::vector<ConfigLine> *config_lines) {
    int32 num_lines = config_lines->size();
    unordered_map<std::string, int32, StringHasher> node_name_to_most_recent_line;
    unordered_set<std::string, StringHasher> component_names;
    typedef unordered_map<std::string, int32, StringHasher>::iterator IterType;

    // 对应每个line的flag
    std::vector<bool> to_remove(num_lines, false);
    // foreach line in lines.
    // 获得name,
    // 1 如果是component 加入到componet中
    // 2 如果是node ,与nnet 之前存在的nnet 查重, 并删除重复
    // 3 清理一下config_lines
    for (int32 line = 0; line < num_lines; line++) {
      
      ConfigLine &config_line = (*config_lines)[line];
      std::string name;
      
      // 找到line 中对应key的value值, 并将 data_ 中<key, <value,false>> 设置为 <key, <value, true>>
      if (!config_line.GetValue("name", &name))
        KALDI_ERR << "Config line has no field 'name=xxx': "
                  << config_line.WholeLine();
      
      bool ConfigLine::GetValue(const std::string &key, std::string *value) {
        KALDI_ASSERT(value != NULL);
        std::map<std::string, std::pair<std::string, bool> >::iterator it = data_.begin();
        for (; it != data_.end(); ++it) {
          if (it->first == key) {
            *value = (it->second).first;
            (it->second).second = true;
            return true;
          }
        }
        return false;
      }

      // 如果是个component, 保存到 component_names
      if (config_line.FirstToken() == "component") {
        // components 保存在自己的空间中, 不允许重复??? 但是并没有作用, 是个临时变量, 这里就是随意做了一下, 没啥用.
        if (!component_names.insert(name).second) {
          // we could not insert it because it was already there.
          KALDI_ERR << "Component name " << name
                    << " appears twice in the same config file.";
        }
      } else {
        // line 定义了 一些排序号的 network 节点 Component-node
        // 获得name 去name_name_to_most_recent_line 之前处理了的line中寻找, 禁止重复.
        // 这里的重复 是去之前存在的nnet结构中找可能的重复.
        IterType iter = node_name_to_most_recent_line.find(name);
        // if repeated, 标记 to_remove[prev_line] = true, 保留last同名layer.
        if (iter != node_name_to_most_recent_line.end()) {
          // name is repeated.
          int32 prev_line = iter->second;
          if (prev_line >= num_lines_initial) {
            // user-provided config contained repeat of node with this name.
            KALDI_ERR << "Node name " << name
                      << " appears twice in the same config file.";
          }
          KALDI_ASSERT(line >= num_lines_initial);
          to_remove[prev_line] = true;
        }
        // 将 name - line line-id, 具体的ConfigLine还是保存到
        node_name_to_most_recent_line[name] = line;
      }
    }
    // swap 通过交换方式 利用临时数组, 清理不需要内存.
    std::vector<ConfigLine> config_lines_out;
    config_lines_out.reserve(num_lines);
    for (int32 i = 0; i < num_lines; i++) {
      if (!to_remove[i])
        config_lines_out.push_back((*config_lines)[i]);
    }
    config_lines->swap(config_lines_out);
  }



  
  // 读取nnet3 已经存在的 Component 
  int32 initial_num_components = components_.size();
 
  // 两次遍历
  // ========================= 构建核心成员 =========================
  // ========================component_ component_names_  node_ node_names_的过程.
  for (int32 pass = 0; pass <= 1; pass++) {
    // foreach line  构建 component_ component_names_  node_ node_names_的过程.
    for (size_t i = 0; i < config_lines.size(); i++) {
      const std::string &first_token = config_lines[i].FirstToken();

      // ====================== 构建 component_ component_names_ 成员
      if (first_token == "component") {
        // components_.push_back(new_component);
        // component_names_.push_back(name);
        // 构建component, 并加入component_ component_names_ 中.
        if (pass == 0)
          ProcessComponentConfigLine(initial_num_components, &(config_lines[i]));
        
        // called only on pass 0 of ReadConfig.
        void Nnet::ProcessComponentConfigLine(
            int32 initial_num_components,
            ConfigLine *config) {
          
          std::string name, type;
          if (!config->GetValue("name", &name))
            KALDI_ERR << "Expected field name=<component-name> in config line: "
                      << config->WholeLine();
          
          if (!IsToken(name)) // e.g. contains a space.
            KALDI_ERR << "Component name '" << name << "' is not allowed, in line: "
                      << config->WholeLine();
          
          if (!config->GetValue("type", &type))
            KALDI_ERR << "Expected field type=<component-type> in config line: "
                      << config->WholeLine();

          // 构建一个Component
          Component *new_component = Component::NewComponentOfType(type);

          // 初始化Component的过程
          new_component->InitFromConfig(config);

          // =================== eg FixedAffineComponent =================
          // 从构建的 仿射变换 矩阵mat 中获得 参数信息
          void FixedAffineComponent::Init(const CuMatrixBase<BaseFloat> &mat) {
            KALDI_ASSERT(mat.NumCols() > 1);
            // 线性参数 取一个 sumMatrix, 除了最后一个col 全部去除 作为线性参数. 左右一col 作为bias
            linear_params_ = mat.Range(0, mat.NumRows(), 0, mat.NumCols() - 1);
            bias_params_.Resize(mat.NumRows());
            bias_params_.CopyColFromMat(mat, mat.NumCols() - 1);
          }
          void FixedAffineComponent::InitFromConfig(ConfigLine *cfl) {
            std::string filename;
            // Two forms allowed: "matrix=<rxfilename>", or "input-dim=x output-dim=y"
            // (for testing purposes only).
            if (cfl->GetValue("matrix", &filename)) {
              if (cfl->HasUnusedValues())
                KALDI_ERR << "Invalid initializer for layer of type "
                          << Type() << ": \"" << cfl->WholeLine() << "\"";

              bool binary;
              Input ki(filename, &binary);
              CuMatrix<BaseFloat> mat;
              mat.Read(ki.Stream(), binary);
              KALDI_ASSERT(mat.NumRows() != 0);
              Init(mat);
            } else {
              int32 input_dim = -1, output_dim = -1;
              if (!cfl->GetValue("input-dim", &input_dim) ||
                  !cfl->GetValue("output-dim", &output_dim) || cfl->HasUnusedValues()) {
                KALDI_ERR << "Invalid initializer for layer of type "
                          << Type() << ": \"" << cfl->WholeLine() << "\"";
              }
              CuMatrix<BaseFloat> mat(output_dim, input_dim + 1);
              mat.SetRandn();
              Init(mat);
            }
          }



          // 同名Component 需要delete
          int32 index = GetComponentIndex(name);
          if (index != -1) {  // Replacing existing component.
            // 必须index 在原本initial_num_components 之前 component_
            if (index >= initial_num_components) {
              // that index was something we added from this config.
              KALDI_ERR << "You are adding two components with the same name: '"
                        << name << "'";
            }
            delete components_[index];
            components_[index] = new_component;
          } else {
            components_.push_back(new_component);
            component_names_.push_back(name);
          }
          if (config->HasUnusedValues())
            KALDI_ERR << "Unused values '" << config->UnusedValues()
                      << " in config line: " << config->WholeLine();
        }
      }
      
      // ===== 构建 nodes_ nodes_names_ 成员, 
      // ===== 这里为所有Component-node 都增加了 两个不同类型的Node
      // ===== kDescriptor, 并产生一个新的名字, 名字为 name_input
      // ===== kComponent, name就是对应的Component名字  name
      // 其中 node_index_input name_input
      // 对应component的 kDescriptor 的node, 并且Descriptor都保存在这个node中
      // 其中 node_index       name
      // 对应component的 kComponent 的node,  并且保存了 对应的component_对象.
      
      else if (first_token == "component-node") {
        ProcessComponentNodeConfigLine(pass,  &(config_lines[i]));

        // eg:
        // component name=tdnn6.affine type=NaturalGradientAffineComponent input-dim=850 output-dim=850  max-change=0.75
        // component-node name=tdnn6.affine component=tdnn6.affine input=tdnn5.batchnorm
        // component name=tdnn6.relu type=RectifiedLinearComponent dim=850 self-repair-scale=1e-05
        // component-node name=tdnn6.relu component=tdnn6.relu input=tdnn6.affine
        void Nnet::ProcessComponentNodeConfigLine(int32 pass,
                                                  ConfigLine *config) {

          std::string name;
          if (!config->GetValue("name", &name))
            KALDI_ERR << "Expected field name=<component-name> in config line: "
                      << config->WholeLine();

          // 构造了一个 新名字 name+ _input, 这个名字应该没有出现过, 是自己新造的.
          std::string input_name = name + std::string("_input");
          int32
              // input_node_index 查找已有的自造名字, pass=0 时 应该=-1
              input_node_index = GetNodeIndex(input_name),
              // node_index 查找已有node的index, pass=0 时, 应该为-1
              node_index = GetNodeIndex(name);

          if (pass == 0) {
            KALDI_ASSERT(input_node_index == -1 && node_index == -1);
            // 现在仅仅为配置 增加 nodes_ 以及names_ 确定配置设置pass=1进行.
            nodes_.push_back(NetworkNode(kDescriptor));
            nodes_.push_back(NetworkNode(kComponent));
            node_names_.push_back(input_name);
            node_names_.push_back(name);
            return;
          } else {

            // =============  设置 node 对应的 Component ================
            // pass=1 , 上次已经为本component node 输入过了.
            KALDI_ASSERT(input_node_index != -1 && node_index == input_node_index + 1);
            std::string component_name, input_descriptor;

            // 找到 对应 本 component-node 的 component.
            if (!config->GetValue("component", &component_name))
              KALDI_ERR << "Expected component=<component-name>, in config line: "
                        << config->WholeLine();

            // 获得对应component 的 component-index.
            int32 component_index = GetComponentIndex(component_name);

            // 设置 实际nnet node 结构, 将component 对应的 index 设置给 Node.u.component_index?
            nodes_[node_index].u.component_index = component_index;

            
            // 获得 当前Line 的 inputDescriptor.
            if (!config->GetValue("input", &input_descriptor))
              KALDI_ERR << "Expected input=<input-descriptor>, in config line: "
                        << config->WholeLine();

            // 处理node 的 input Descriptor String  ==> tokens. 
            std::vector<std::string> tokens;
            if (!DescriptorTokenize(input_descriptor, &tokens))
              KALDI_ERR << "Error tokenizing descriptor in config line "
                        << config->WholeLine();

            // bool DescriptorTokenize(const std::string &input, std::vector<std::string> *tokens) {
            //   KALDI_ASSERT(tokens != NULL);
              
            //   size_t start = input.find_first_not_of(" \t"), size = input.size();
            //   tokens->clear();
            //   while (start < size) {
            //     KALDI_ASSERT(!isspace(input[start]));
            //     if (input[start] == '(' || input[start] == ')' || input[start] == ',') {
            //       tokens->push_back(std::string(input, start, 1));
            //       start = input.find_first_not_of(" \t", start + 1);
            //     } else {
            //       size_t found = input.find_first_of(" \t(),", start);
            //       KALDI_ASSERT(found != start);
            //       // 
            //       if (found == std::string::npos) {
            //         std::string str(input, start, input.size() - start);
            //         BaseFloat tmp;
            //         if (!IsValidName(str) && !ConvertStringToReal(str, &tmp)) {
            //           KALDI_WARN << "Could not tokenize line " << ErrorContext(std::string(input, start));
            //           return false;
            //         }
            //         tokens->push_back(str);
            //         break;
            //       } else {
            //         if (input[found] == '(' || input[found] == ')' || input[found] == ',') {
            //           std::string str(input, start, found - start);
            //           BaseFloat tmp;
            //           if (!IsValidName(str) && !ConvertStringToReal(str, &tmp)) {
            //             KALDI_WARN << "Could not tokenize line " << ErrorContext(std::string(input, start));
            //             return false;
            //           }
            //           tokens->push_back(str);
            //           start = found;
            //         } else {
            //           std::string str(input, start, found - start);
            //           BaseFloat tmp;
            //           if (!IsValidName(str) && !ConvertStringToReal(str, &tmp)) {
            //             KALDI_WARN << "Could not tokenize line " << ErrorContext(std::string(input, start));
            //             return false;
            //           }
            //           tokens->push_back(str);
            //           start = input.find_first_not_of(" \t", found);
            //         }
            //       }
            //     }
            //   }
            //   return true;
            // }
            
            

            // =====================  循环构建 Descriptor的过程 ==================
            // =================== 根据 Descriptor String 以及 node-names
            // =================== Parse 构建对应的Descriptor
            std::vector<std::string> node_names_temp;
            // 返回所有 type 为 Component Input dimRange 的 node-name
            GetSomeNodeNames(&node_names_temp);
            tokens.push_back("end of input");
            const std::string *next_token = &(tokens[0]);
            // 构建Descriptor的过程. Descriptor 是一个 循环内嵌的class
            if (!nodes_[input_node_index].descriptor.Parse(node_names_temp,
                                                           &next_token))
              KALDI_ERR << "Error parsing Descriptor in config line: "
                        << config->WholeLine();


            // ======================== 处理Token 构建一个Descriptor
            // ======================== Descriptor 内部保存多个 实例descriptor(SimpleSumDescriptor ForwardDescriptor等)
            bool Descriptor::Parse(const std::vector<std::string> &node_names,
                                   const std::string **next_token) {
              GeneralDescriptor *gen_desc;

              // ==================== 递归通过Parse 将Append Offset input 形成GeneralDescriptor.
              // ==================== 具体过程 见代码,
              // 最外层是一个 Append 的GeneralDescriptor.
              // 循环构建GeneralDescriptor的过程, 首先都是构造的GeneralDescriptor.
              gen_desc = GeneralDescriptor::Parse(node_names, next_token);
              

              
              // GeneralDescriptor 需要转化为 Descriptor 进行使用.
              // 内部是 多个实例用Descriptor --- SimpleSumDescriptor ForwardDescriptor 等.
              Descriptor *desc = gen_desc->ConvertToDescriptor();

              Descriptor* GeneralDescriptor::ConvertToDescriptor() {
                // 将内部的各个Descriptor都转化成为vector<sum> , 然后构建一个 Descriptor ans,
                // 将转化好的vector<sum> -> ans.parts_.
                // copy this ->> normalized.
                GeneralDescriptor *normalized = GetNormalizedDescriptor();


                // 结果就是 cp 了 this 一次,  GeneralDescriptor* cp_this = this->NormalizeAppend().
                GeneralDescriptor* GeneralDescriptor::GetNormalizedDescriptor() const {
                  GeneralDescriptor *ans = NormalizeAppend();
                  while (Normalize(ans));  // keep normalizing as long as it changes.
                  return ans;

                  GeneralDescriptor* GeneralDescriptor::NormalizeAppend() const {
                    // 获得内部的 Descriptor数量
                    int32 num_terms = NumAppendTerms();
                  
                    KALDI_ASSERT(num_terms > 0);
                    if (num_terms == 1) {
                      return GetAppendTerm(0);
                    } else {
                      // 又 构建一个kAppend 的GeneralDescriptor对象.
                      GeneralDescriptor *ans = new GeneralDescriptor(kAppend);
                      ans->descriptors_.resize(num_terms);
                      for (size_t i = 0; i < num_terms; i++) {
                        ans->descriptors_[i] = GetAppendTerm(i);
                      }
                      return ans;
                    }
                  }
                }

                
                
                std::vector<SumDescriptor*> sum_descriptors;
                if (normalized->descriptor_type_ == kAppend) {
                  
                  // 将所有 child descriptor 转化为 SumDescriptor() 
                  for (size_t i = 0; i < normalized->descriptors_.size(); i++)
                    sum_descriptors.push_back(normalized->descriptors_[i]->ConvertToSumDescriptor());

                  {  // 注释
                    // 将其他类型的Descriptor 转化为 SimpleSumDescriptor 或者 OptionalSumDescriptor 等.
                    SumDescriptor *GeneralDescriptor::ConvertToSumDescriptor() const {
                    KALDI_ASSERT(descriptor_type_ != kAppend &&
                                 "Badly normalized descriptor");
                    
                    switch (descriptor_type_) {
                      case kAppend:
                        KALDI_ERR << "Badly normalized descriptor";
                      default: {
                        // 1 会首先将this-Offset 进行转化, 转化为OffsetForwardingDescriptor 
                        //   1 将Offset内部的input, 进行生成SimpleForwardingDescriptor,
                        //     input 构建为SimpleForwardingDescriptor时需要一个 value1_, 对于kNodeName
                        //     原本是 GeneralDescriptor *ans = new GeneralDescriptor(kNodeName, i); 以node-index i为value1_
                        //     将GeneralDescriptor转化为SimpleForwardingDescriptor, 将node-index 保存到src_node_中.
                        //   2 构建Index, Offset中 的offset值 进行构建为Index(0, value1_, value2_)
                        //     value1_ 是一个time-frame 时间偏移.
                        //
                        return new SimpleSumDescriptor(this->ConvertToForwardingDescriptor());

                        // 转化为ForwardingDescriptor的过程, 实际上就是利用value1_ (Offset的 input-index)
                        ForwardingDescriptor *GeneralDescriptor::ConvertToForwardingDescriptor() const {
                          switch (this->descriptor_type_) {
                            case kNodeName: return new SimpleForwardingDescriptor(value1_);
                              // SimpleForwardingDescriptor(int32 src_node,
                              //                            BaseFloat scale = 1.0):
                              //     src_node_(src_node), scale_(scale) {
                              //   KALDI_ASSERT(src_node >= 0);
                              // }
                              
                              break;
                            case kOffset: {
                              KALDI_ASSERT(descriptors_.size() == 1 && "bad descriptor");
                              return new OffsetForwardingDescriptor(
                                  descriptors_[0]->ConvertToForwardingDescriptor(),
                                  Index(0, value1_, value2_));

                              // OffsetForwardingDescriptor class中包含的数据, 构造就是赋值
                              // private:
                              //  ForwardingDescriptor *src_;  // Owned here.
                              //  Index offset_;  // The index-offset to be added to the index.
                              //  这里的Index offset_ 十分重要, 是需要用到的偏移量, 并且具体用法还不知道呢.

                              // struct Index {
                              //   int32 n;  // member-index of minibatch, or zero.
                              //   int32 t;  // time-frame.
                              //   int32 x;  // this may come in useful in convoluational approaches.
                              //   // ... it is possible to add extra index here, if needed.
                              //   Index(): n(0), t(0), x(0) { }
                              //   Index(int32 n, int32 t, int32 x = 0): n(n), t(t), x(x) { }
                              // }
                              
                              break;
                            }
                          }
                        }
                       
                        
                      }
                    }
                    }
                  }
                  
                } else {
                  sum_descriptors.push_back(normalized->ConvertToSumDescriptor());
                }

                // 将 n 个 sum_descriptor 构建成一个 Descriptor.
                // 就是将各个SumDescriptor -> ans.parts_
                Descriptor *ans = new Descriptor(sum_descriptors);
                delete normalized;
                return ans;
              }

              // 经过这个步骤, 就没Append类型的GeneralDescriptor了, 是一个 Descriptor-parts_--vector<InstanceDescriptor>
              *this = *desc;

              delete desc;
              delete gen_desc;
              return true;
            }

            if (config->HasUnusedValues())
              KALDI_ERR << "Unused values '" << config->UnusedValues()
                        << " in config line: " << config->WholeLine();
          }
        }

      }

      else if (first_token == "input-node") {
        // eg.
        // input-node name=ivector dim=100
        // input-node name=input dim=43
        if (pass == 0)
          ProcessInputNodeConfigLine(&(config_lines[i]));

        // called only on pass 0 of ReadConfig.
        void Nnet::ProcessInputNodeConfigLine(ConfigLine *config) {
          std::string name;
          if (!config->GetValue("name", &name))
            KALDI_ERR << "Expected field name=<input-name> in config line: "
                      << config->WholeLine();
          int32 dim;
          if (!config->GetValue("dim", &dim))
            KALDI_ERR << "Expected field dim=<input-dim> in config line: "
                      << config->WholeLine();

          if (config->HasUnusedValues())
            KALDI_ERR << "Unused values '" << config->UnusedValues()
                      << " in config line: " << config->WholeLine();

          KALDI_ASSERT(GetNodeIndex(name) == -1);
          if (dim <= 0)
            KALDI_ERR << "Invalid dimension in config line: " << config->WholeLine();

          // 向nodes_ 中加入Input-node 以及 对应的dim
          // 向node_names_ 中加入name
          int32 node_index = nodes_.size();
          nodes_.push_back(NetworkNode(kInput));
          nodes_[node_index].dim = dim;
          node_names_.push_back(name);
        }
      }

      else if (first_token == "output-node") {
        ProcessOutputNodeConfigLine(pass, &(config_lines[i]));
        
        // output-node name=output input=output.log-softmax objective=linear
        void Nnet::ProcessOutputNodeConfigLine(int32 pass,
                                               ConfigLine *config) {
          std::string name;
          if (!config->GetValue("name", &name))
            KALDI_ERR << "Expected field name=<input-name> in config line: "
                      << config->WholeLine();
          // 获得对应的node 名字,从node_names_ 获得对应的index
          // 在pass=1 时node_index 会存在
          int32 node_index = GetNodeIndex(name);
          
          if (pass == 0) {
            KALDI_ASSERT(node_index == -1);
            nodes_.push_back(NetworkNode(kDescriptor));
            node_names_.push_back(name);
          } else {
            KALDI_ASSERT(node_index != -1);
            std::string input_descriptor;
            // 和 component-node 一样构建 descriptor.
            if (!config->GetValue("input", &input_descriptor))
              KALDI_ERR << "Expected input=<input-descriptor>, in config line: "
                        << config->WholeLine();
            std::vector<std::string> tokens;
            if (!DescriptorTokenize(input_descriptor, &tokens))
              KALDI_ERR << "Error tokenizing descriptor in config line "
                        << config->WholeLine();
            tokens.push_back("end of input");



            
            // if the following fails it will die.
            std::vector<std::string> node_names_temp;
            // 获得component Input DimRange node 名字.
            // 完成处理token 生成Descriptor的过程, 需要用到各个层的名字, 因为Descriptor是一个链接件, 用来链接两个Component的.
            // 或者说 Component 本身不进行组合, 而是每个 Component具有input输入, 而input输入都是通过Descriptor指定的.
            GetSomeNodeNames(&node_names_temp);
            const std::string *next_token = &(tokens[0]);
            if (!nodes_[node_index].descriptor.Parse(node_names_temp, &next_token))
              KALDI_ERR << "Error parsing descriptor (input=...) in config line "
                        << config->WholeLine();



            
            std::string objective_type;
            if (config->GetValue("objective", &objective_type)) {
              if (objective_type == "linear") {
                nodes_[node_index].u.objective_type = kLinear;
              } else if (objective_type == "quadratic") {
                nodes_[node_index].u.objective_type = kQuadratic;
              } else {
                KALDI_ERR << "Invalid objective type: " << objective_type;
              }
            } else {
              // the default objective type is linear.  This is what we use
              // for softmax objectives; the LogSoftmaxLayer is included as the
              // last layer, in this case.
              nodes_[node_index].u.objective_type = kLinear;
            }
            if (config->HasUnusedValues())
              KALDI_ERR << "Unused values '" << config->UnusedValues()
                        << " in config line: " << config->WholeLine();
          }
        }
       
      }

      else if (first_token == "dim-range-node") {
        ProcessDimRangeNodeConfigLine(pass, &(config_lines[i]));
      }

      else {
        KALDI_ERR << "Invalid config-file line ('" << first_token
                  << "' not expected): " << config_lines[i].WholeLine();
      }
    }
  }

  
  Check();

  void Nnet::Check(bool warn_for_orphans) const {
    int32
        num_nodes = nodes_.size(),
        num_input_nodes = 0,
        num_output_nodes = 0;
    KALDI_ASSERT(num_nodes != 0);

    
    // foreach node in nodes_
    for (int32 n = 0; n < num_nodes; n++) {
      
      const NetworkNode &node = nodes_[n];
      std::string node_name = node_names_[n];
      
      KALDI_ASSERT(GetNodeIndex(node_name) == n);

      // ==============  node 节点类型 -- kInput kDescriptor kComponent
      switch (node.node_type) {
        case kInput:
          KALDI_ASSERT(node.dim > 0);
          num_input_nodes++;
          break;
        case kDescriptor: {
          // 一个 component 会生成两个 node :
          // 1 kDescriptor 保存所有Descriptor, 最终通过parse内部的ConvertToDescriptor 转化为 标准Descriptor类型对象
          //               这里包含了 本 Component的input
          // 2 kComponent 保存 ????

          // 判断是否是个输出Node?
          if (IsOutputNode(n))
            num_output_nodes++;
          bool Nnet::IsOutputNode(int32 node) const {
            // 如果当前节点是kDescriptor node之后 就是最终节点, 认为是个 输出node???
            int32 size = nodes_.size();
            return (nodes_[node].node_type == kDescriptor &&  (node + 1 == size || nodes_[node + 1].node_type != kComponent));
          }
          

          std::vector<int32> node_deps;
          // 获得node descriptor的依赖关系
          node.descriptor.GetNodeDependencies(&node_deps);
          // ======= 递归的获取依赖输入, 将输入依赖 push into node_indexes
          // eg
          // 1 原本的Append 下具有多个 part_ -- Offset.
          // Offset 有两层包装 1 SimpleSumDescriptor 2 OffsetForward
          // 2 Offset 的 SimpleSumDescriptor 的src_ 保存对应的OffsetForwardingDescriptor.
          // 3 Offset --- OffsetForwarding才是实际的Descriptor, 并包含offset_.
          //   OffsetForwarding 将 对应的input -- SimpleForwardingDescriptor 作为 src_,
          // 4 input-- SimpleForwardingDescriptor -- 包含 node-index 作为最基本.

          void Descriptor::GetNodeDependencies(std::vector<int32> *node_indexes) const {
            node_indexes->clear();
            for (size_t i = 0; i < parts_.size(); i++)
              parts_[i]->GetNodeDependencies(node_indexes);

            void SimpleSumDescriptor::GetNodeDependencies(
                std::vector<int32> *node_indexes) const {
              src_->GetNodeDependencies(node_indexes);
            }
            void OffsetForwardingDescriptor::GetNodeDependencies(
                std::vector<int32> *node_indexes) const {
              src_->GetNodeDependencies(node_indexes);
            }

            void SimpleForwardingDescriptor::GetNodeDependencies(
                std::vector<int32> *node_indexes) const {
              node_indexes->push_back(src_node_);
            }

            
          }
          
          SortAndUniq(&node_deps);
         
          for (size_t i = 0; i < node_deps.size(); i++) {
            int32 src_node = node_deps[i];
            // check1 必须在0 到本节点之前,
            // check2 必须是kInput kComponent 才能作为一个Descriptor的依赖. 
            KALDI_ASSERT(src_node >= 0 && src_node < num_nodes);
            NodeType src_type = nodes_[src_node].node_type;
            if (src_type != kInput && src_type != kDimRange &&
                src_type != kComponent)
              KALDI_ERR << "Invalid source node type in Descriptor: source node "
                        << node_names_[src_node];
          }
         
          break;
        }
        case kComponent: {
          // 必须前一个node 是kDescriptor
          KALDI_ASSERT(n > 0 && nodes_[n-1].node_type == kDescriptor);

          // ================= 对应的kDescriptor =================
          const NetworkNode &src_node = nodes_[n-1];

          // 获得对应的Component
          const Component *c = GetComponent(node.u.component_index);

          // 判断 Component描述的 input_dim 必须要与 对应的
          // Descriptor-node 的src_dim 一致
          int32 src_dim,
              input_dim = c->InputDim();
              // virtual int32 InputDim() const { return linear_params_.NumCols(); }

          // 计算kComponent 对应输入的kDescriptor 的维度, kDescriptor 如果是Append
          // 就会将多个更底层的kDescriptor的Dim 获得, 并累加起来构成更大的dim.
          src_dim = src_node.Dim(*this);

          // eg 一个 Append 最终构建的Descriptor 
          int32 NetworkNode::Dim(const Nnet &nnet) const {
            int32 ans;
            switch (node_type) {
              case kInput: case kDimRange:
                ans = dim;
                break;
              case kDescriptor:
                ans = descriptor.Dim(nnet);
                // int32 Descriptor::Dim(const Nnet &nnet) const {
                //   int32 num_parts = parts_.size();
                //   int32 dim = 0;
                //   for (int32 part = 0; part < num_parts; part++)
                // ================== dim =============== 维度为多个Offset 累加
                //     dim += parts_[part]->Dim(nnet);
                //   KALDI_ASSERT(dim > 0);
                //   return dim;
                // }
                break;
              case kComponent:
                ans = nnet.GetComponent(u.component_index)->OutputDim();
                // 返回输出 向量维度
                // virtual int32 FixedAffineComponent :: OutputDim() const { return linear_params_.NumRows(); }
                
                break;
              default:
                ans = 0;  // suppress compiler warning
                KALDI_ERR << "Invalid node type.";
            }
            KALDI_ASSERT(ans > 0);
            return ans;
          }
          
          if (src_dim != input_dim) {
            KALDI_ERR << "Dimension mismatch for network-node "
                      << node_name << ": input-dim "
                      << src_dim << " versus component-input-dim "
                      << input_dim;
          }
          break;
        }
        case kDimRange: {
          int32 input_node = node.u.node_index;
          KALDI_ASSERT(input_node >= 0 && input_node < num_nodes);
          NodeType input_type = nodes_[input_node].node_type;
          if (input_type != kInput && input_type != kComponent)
            KALDI_ERR << "Invalid source node type in DimRange node: source node "
                      << node_names_[input_node];
          int32 input_dim = nodes_[input_node].Dim(*this);
          if (!(node.dim > 0 && node.dim_offset >= 0 &&
                node.dim + node.dim_offset <= input_dim)) {
            KALDI_ERR << "Invalid node dimensions for DimRange node: " << node_name
                      << ": input-dim=" << input_dim << ", dim=" << node.dim
                      << ", dim-offset=" << node.dim_offset;
          }
          break;
        }
        default:
          KALDI_ERR << "Invalid node type for node " << node_name;
      }
    }


    // check names 对应 Component
    int32 num_components = components_.size();
    for (int32 c = 0; c < num_components; c++) {
      const std::string &component_name = component_names_[c];
      KALDI_ASSERT(GetComponentIndex(component_name) == c &&
                   "Duplicate component names?");
    }
    
    KALDI_ASSERT(num_input_nodes > 0);
    KALDI_ASSERT(num_output_nodes > 0);

    // default is true
    if (warn_for_orphans) {

      // 未被用到的 componet-index
      std::vector<int32> orphans;
      FindOrphanComponents(*this, &orphans);

      // 目的是获得 未被用到的 componet-index, 但是所有对应的Component-node 都对应其Component,
      // 那么这样的话并不会出现未用到的啊.?????????????????????
      void FindOrphanComponents(const Nnet &nnet, std::vector<int32> *components) {
        int32 num_components = nnet.NumComponents(), num_nodes = nnet.NumNodes();
        std::vector<bool> is_used(num_components, false);
        // 判断ComponentNode, 对应的Component, 标记为用到的Component is_used = true.
        for (int32 i = 0; i < num_nodes; i++) {
          if (nnet.IsComponentNode(i)) {
            int32 c = nnet.GetNode(i).u.component_index;
            KALDI_ASSERT(c >= 0 && c < num_components);
            is_used[c] = true;
          }
        }
        // 将对应的 未被 用到的component-index push 输出给上层. 等待处理.
        components->clear();
        for (int32 i = 0; i < num_components; i++)
          if (!is_used[i])
            components->push_back(i);
        
      }



      
      for (size_t i = 0; i < orphans.size(); i++) {
        KALDI_WARN << "Component " << GetComponentName(orphans[i])
                   << " is never used by any node.";
      }

      FindOrphanNodes(*this, &orphans);

      // input nnet, 未被用到的 Component-index
      void FindOrphanNodes(const Nnet &nnet, std::vector<int32> *nodes) {

        std::vector<std::vector<int32> > depend_on_graph, dependency_graph;
        // depend_on_graph[i] is a list of all the nodes that depend on i.
        // ================ depend_on_graph[i] 是所有依赖于 inode 的 node的list.
        NnetToDirectedGraph(nnet, &depend_on_graph);


        // ================== 构建depend_on_graph 的依赖关系
        void NnetToDirectedGraph(const Nnet &nnet,
                                 std::vector<std::vector<int32> > *graph) {

          graph->clear();
          
          int32 num_nodes = nnet.NumNodes();
          // graph 是依赖关系图, 所有依赖index 的加入到 graph[index] list中
          graph->resize(num_nodes);

          // 处理对n node 的依赖关系
          for (int32 n = 0; n < num_nodes; n++) {

            const NetworkNode &node = nnet.GetNode(n);
            std::vector<int32> node_dependencies;
            switch (node.node_type) {
              case kInput:
                break;  // no node dependencies.
              case kDescriptor:
                // kDescriptor 的依赖 需要计算descriptor中的所有依赖.
                node.descriptor.GetNodeDependencies(&node_dependencies);
                break;
              case kComponent:
                // kComponent node 只依赖于 其对应的 kDescriptor node
                node_dependencies.push_back(n - 1);
                break;
              case kDimRange:
                node_dependencies.push_back(node.u.node_index);
                break;
              default:
                KALDI_ERR << "Invalid node type";
            }
            
            SortAndUniq(&node_dependencies);

            // =============== 构建依赖关系数组 graph[dep_n] ============
            // 对 依赖的 dep_n , 向graph[dep_n].push_back 当前节点.
            for (size_t i = 0; i < node_dependencies.size(); i++) {
              int32 dep_n = node_dependencies[i];
              KALDI_ASSERT(dep_n >= 0 && dep_n < num_nodes);
              (*graph)[dep_n].push_back(n);
            }
          }
        }

        
        

        // dependency_graph[i] 是i node 所有依赖的node list , 与depend_on_graph的反向
        ComputeGraphTranspose(depend_on_graph, &dependency_graph);

        
        // Find all nodes required to produce the outputs.
        // 找到所有需要用来生成output的 nodes
        int32 num_nodes = nnet.NumNodes();
        std::vector<bool> node_is_required(num_nodes, false);

        // 所有output node
        std::vector<int32> queue;
        for (int32 i = 0; i < num_nodes; i++) {
          if (nnet.IsOutputNode(i))
            queue.push_back(i);
        }

        // 根据outpupt-node 以及 dependency_graph, 递归寻找所有的 依赖node关系
        // 将对应 node_is_required[i] = true
        while (!queue.empty()) {
          int32 i = queue.back();
          queue.pop_back();
          if (!node_is_required[i]) {
            node_is_required[i] = true;
            for (size_t j = 0; j < dependency_graph[i].size(); j++)
              queue.push_back(dependency_graph[i][j]);
          }
        }

        // 计算所有不需要的 node
        nodes->clear();
        for (int32 i = 0; i < num_nodes; i++) {
          if (!node_is_required[i])
            nodes->push_back(i);
        }
      }

      for (size_t i = 0; i < orphans.size(); i++) {
        if (!IsComponentInputNode(orphans[i])) {
          // 没有关于 component-index 节点的警告, 因为警告会被打印对应的component
          // 重复的警告可能会让user很迷惑, 因为componentinput node 是隐性的经常对用户来说是不可见的
          KALDI_WARN << "Node " << GetNodeName(orphans[i])
                     << " is never used to compute any output.";
        }
      }
      
      bool Nnet::IsComponentInputNode(int32 node) const {
        int32 size = nodes_.size();
        // 如果是个正常ComponentNode
        return (node + 1 < size &&
                nodes_[node].node_type == kDescriptor &&
                nodes_[node+1].node_type == kComponent);
      }
      
    }
  }

}








/*
Test script:

cat <<EOF | nnet3-init --binary=false - foo.raw
component name=affine1 type=NaturalGradientAffineComponent input-dim=72 output-dim=59
component name=relu1 type=RectifiedLinearComponent dim=59
component name=final_affine type=NaturalGradientAffineComponent input-dim=59 output-dim=298
component name=logsoftmax type=SoftmaxComponent dim=298

input-node name=input dim=18
component-node name=affine1_node component=affine1 input=Append(Offset(input, -4), Offset(input, -3), Offset(input, -2), Offset(input, 0))
component-node name=nonlin1 component=relu1 input=affine1_node
component-node name=final_affine component=final_affine input=nonlin1
component-node name=output_nonlin component=logsoftmax input=final_affine
output-node name=output input=output_nonlin
EOF







cat <<EOF | nnet3-init --binary=false foo.raw -  bar.raw
component name=affine2 type=NaturalGradientAffineComponent input-dim=59 output-dim=59
component name=relu2 type=RectifiedLinearComponent dim=59
component name=final_affine type=NaturalGradientAffineComponent input-dim=59 output-dim=298
component-node name=affine2 component=affine2 input=nonlin1
component-node name=relu2 component=relu2 input=affine2
component-node name=final_affine component=final_affine input=relu2
EOF

rm foo.raw bar.raw

 */
