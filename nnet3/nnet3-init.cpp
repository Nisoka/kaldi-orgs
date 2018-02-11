// nnet3bin/nnet3-init.cc
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet3/nnet-nnet.h"
#include "hmm/transition-model.h"
#include "tree/context-dep.h"

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
    if (po.NumArgs() == 3) {
      ReadKaldiObject(raw_nnet_rxfilename, &nnet);
      KALDI_LOG << "Read raw neural net from "
                << raw_nnet_rxfilename;
    }

    {
      bool binary;
      Input ki(config_rxfilename, &binary);
      KALDI_ASSERT(!binary && "Expect config file to contain text.");
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








void Nnet::ReadConfig(std::istream &config_is) {

  std::vector<std::string> lines;
  // Write into "lines" a config file corresponding to whatever
  // nodes we currently have.  Because the numbering of nodes may
  // change, it's most convenient to convert to the text representation
  // and combine the existing and new config lines in that representation.
  const bool include_dim = false;
  GetConfigLines(include_dim, &lines);

  // we'll later regenerate what we need from nodes_ and node_name_ from the
  // string representation.
  nodes_.clear();
  node_names_.clear();

  int32 num_lines_initial = lines.size();

  
  ReadConfigLines(config_is, &lines);
  // now "lines" will have comments removed and empty lines stripped out

  std::vector<ConfigLine> config_lines(lines.size());

  ParseConfigLines(lines, &config_lines);

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
    if (line.size() == 0) return false;   // Empty line
    size_t pos = 0, size = line.size();
    while (isspace(line[pos]) && pos < size) pos++;
    if (pos == size)
      return false;  // whitespace-only line
    size_t first_token_start_pos = pos;
    // first get first_token_.
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
    first_token_ = std::string(line, first_token_start_pos, pos - first_token_start_pos);
    // first_token_ is expected to be either empty or something like
    // "component-node", which actually is a slightly more restrictive set of
    // strings than IsValidName() checks for this is a convenient way to check it.
    if (!first_token_.empty() && !IsValidName(first_token_))
      return false;

    while (pos < size) {
      if (isspace(line[pos])) {
        pos++;
        continue;
      }

      // OK, at this point we know that we are pointing at nonspace.
      size_t next_equals_sign = line.find_first_of("=", pos);
      if (next_equals_sign == pos || next_equals_sign == std::string::npos) {
        // we're looking for something like 'key=value'.  If there is no equals sign,
        // or it's not preceded by something, it's a parsing failure.
        return false;
      }
      std::string key(line, pos, next_equals_sign - pos);
      if (!IsValidName(key)) return false;

      // handle any quotes.  we support key='blah blah' or key="foo bar".
      // no escaping is supported.
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

        if (next_next_equals_sign != std::string::npos) {  // found a later equals sign.
          size_t preceding_space = line.find_last_of(" \t", next_next_equals_sign);
          if (preceding_space != std::string::npos &&
              preceding_space > next_equals_sign)
            terminating_space = preceding_space;
        }
        while (isspace(line[terminating_space - 1]) && terminating_space > 0)
          terminating_space--;

        std::string value(line, next_equals_sign + 1,
                          terminating_space - (next_equals_sign + 1));
        data_.insert(std::make_pair(key, std::make_pair(value, false)));
        pos = terminating_space;
      }
    }
    return true;
  }

  
  // the next line will possibly remove some elements from "config_lines" so no
  // node or component is doubly defined, always keeping the second repeat.
  // Things being doubly defined can happen when a previously existing node or
  // component is redefined in a new config file.
  RemoveRedundantConfigLines(num_lines_initial, &config_lines);

  int32 initial_num_components = components_.size();
  for (int32 pass = 0; pass <= 1; pass++) {
    for (size_t i = 0; i < config_lines.size(); i++) {
      const std::string &first_token = config_lines[i].FirstToken();
      if (first_token == "component") {
        if (pass == 0)
          ProcessComponentConfigLine(initial_num_components,
                                     &(config_lines[i]));
      } else if (first_token == "component-node") {
        ProcessComponentNodeConfigLine(pass,  &(config_lines[i]));
      } else if (first_token == "input-node") {
        if (pass == 0)
          ProcessInputNodeConfigLine(&(config_lines[i]));
      } else if (first_token == "output-node") {
        ProcessOutputNodeConfigLine(pass, &(config_lines[i]));
      } else if (first_token == "dim-range-node") {
        ProcessDimRangeNodeConfigLine(pass, &(config_lines[i]));
      } else {
        KALDI_ERR << "Invalid config-file line ('" << first_token
                  << "' not expected): " << config_lines[i].WholeLine();
      }
    }
  }
  Check();
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
