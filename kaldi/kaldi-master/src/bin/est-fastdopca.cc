// bin/est-pca.cc

// Copyright      2014  Johns Hopkins University  (author: Daniel Povey)

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
#include "matrix/matrix-lib.h"
using namespace kaldi;
typedef kaldi::int32 int32;

void UpdateWeightVecUseWeightedAverageSample(Vector<double>* weight_k,
                                             std::vector<Vector<double>>* vec_ivectorArk){

  Vector<double> sum;
  sum.Resize(weight_k->Dim());
  sum.SetZero();


  int32 count = vec_ivectorArk->size();
  for (int32 i = 0; i < count; i++){
      Vector<double> x(vec_ivectorArk->at(i));
      double alpha = VecVec(*weight_k, x);
      sum.AddVec(alpha, x);
    }

  sum.Scale(1.0/count);
  weight_k->CopyFromVec(sum);
}


bool check_vector_convergence(Vector<double> weight_k, Vector<double> weight_k_conv,
                              double eps = 0.0001){
  double similerRato = 0.0;
  similerRato = VecVec(weight_k, weight_k_conv);
  if(fabs(similerRato - 1) < eps)
    return true;
  else
    return false;

}


void ProjectXToLeaveSubspace(Vector<double> weight_k,
                             std::vector<Vector<double>>* vec_ivectorArk){
  int32 dim = weight_k.Dim();

  // leaveSubspace = (I - w_k w_k^T)
  Matrix<double> leaveSubspace(dim, dim);
  leaveSubspace.SetUnit();
  leaveSubspace.AddVecVec(-1.0, weight_k, weight_k);

  int32 count = vec_ivectorArk->size();
  for (int32 i = 0; i < count; i++){
      Vector<double> x(vec_ivectorArk->at(i));
      Vector<double> vec_trans(dim);
      vec_trans.AddMatVec(1.0, leaveSubspace, kNoTrans, x, 0.0);
      vec_ivectorArk->at(i).CopyFromVec(vec_trans);
    }
}

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Estimate PCA transform; dimension reduction is optional (if not specified\n"
        "we don't reduce the dimension; if you specify --normalize-variance=true,\n"
        "we normalize the (centered) covariance of the features, and if you specify\n"
        "--normalize-mean=true the mean is also normalized.  So a variety of transform\n"
        "types are supported.  Because this type of transform does not need too much\n"
        "data to estimate robustly, we don't support separate accumulator files;\n"
        "this program reads in the features directly.  For large datasets you may\n"
        "want to subset the features (see example below)\n"
        "By default the program reads in matrices (e.g. features), but with\n"
        "--read-vectors=true, can read in vectors (e.g. iVectors).\n"
        "\n"
        "Usage:  est-pca [options] (<feature-rspecifier>|<vector-rspecifier>) <pca-matrix-out>\n"
        "e.g.:\n"
        "utils/shuffle_list.pl data/train/feats.scp | head -n 5000 | sort | \\\n"
        "  est-pca --dim=50 scp:- some/dir/0.mat\n";

    bool binary = false;
    double epsilon = 0.000001;
    int32 dim = 100;

    ParseOptions po(usage);
    po.Register("binary", &binary, "Write accumulators in binary mode.");
    po.Register("dim", &dim, "Feature dimension requested (if <= 0, uses full "
                             "feature dimension");
    po.Register("epsilon", &epsilon, "The check convegence threshlod nearby 0.000001");
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
        po.PrintUsage();
        exit(1);
      }

    std::string rspecifier = po.GetArg(1),
        pca_mat_wxfilename = po.GetArg(2);

    int32 num_done = 0;

    std::vector<Vector<BaseFloat> > vec_eigenVectors;

    if (!read_vectors) {
        KALDI_ERR << "This is use for est ivector/supervector pca! not mfcc!!";
    } else {
        // read in vectors, not matrices
        SequentialBaseFloatVectorReader vec_reader(rspecifier);
        std::vector<Vector<BaseFloat> > vec_ivectorArk;
        for (; !vec_reader.Done(); vec_reader.Next()) {
            Vector<double> vec(vec_reader.Value());
            vec_ivectorArk.push_back(vec);
            num_done++;
        }

        KALDI_LOG << "First ivector is " << vec_ivectorArk[0];

        Vector<double> weight_k(dim);
        weight_k.SetRandn();
        weight_k.Scale(1.0/weight_k.Norm(2));
        Vector<double> weight_k_conv(dim);

        for (int32 k = 0; k < dim ; k++){
            // update eigenVec_k
            do{
                weight_k_conv.CopyFromVec(weight_k);
                UpdateWeightVecUseWeightedAverageSample(&weight_k, &vec_ivectorArk);
                weight_k.Scale(1.0/weight_k.Norm(2));
            }while(!check_vector_convergence(weight_k, weight_k_conv, epsilon));

            // Save the eigenVec_k
            vec_eigenVectors.push_back(weight_k);

            // project X to the leaves subspace
            ProjectXToLeaveSubspace(weight_k, &vec_ivectorArk);

            // project weight_k to the leaves subspace,
            // get the k+1 eigenVec_k's initial
            std::vector<Vector<BaseFloat> > vec_weight_k;
            vec_weight_k.push_back(weight_k);
            ProjectXToLeaveSubspace(weight_k, &vec_weight_k);
            weight_k.CopyFromVec(vec_weight_k[0]);
        }
    }

    Matrix<BaseFloat> transform(dim, vec_eigenVectors[0].Dim());
    for (int32 i = 0; i < dim; i++){
      Vector<BaseFloat> vec_eigen(vec_eigenVectors[i]);
      transform.CopyRowFromVec(vec_eigen, i);
    }

    WriteKaldiObject(transform, pca_mat_wxfilename, binary);

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


