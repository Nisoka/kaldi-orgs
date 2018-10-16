// ivectorbin/ivector-compute-lda.cc

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
#include "gmm/am-diag-gmm.h"
#include "ivector/ivector-extractor.h"
#include "util/kaldi-thread.h"

namespace kaldi {


class CovarianceStats {
 public:
  CovarianceStats(int32 dim): tot_covar_(dim),
                              between_covar_(dim),
                              num_spk_(0),
                              num_utt_(0) { }

  /// get total covariance, normalized per number of frames.
  void GetTotalCovar(SpMatrix<double> *tot_covar) const {
    KALDI_ASSERT(num_utt_ > 0);
    *tot_covar = tot_covar_;
    tot_covar->Scale(1.0 / num_utt_);
  }
  void GetWithinCovar(SpMatrix<double> *within_covar) {
    KALDI_ASSERT(num_utt_ - num_spk_ > 0);
    *within_covar = tot_covar_;
    within_covar->AddSp(-1.0, between_covar_);
    within_covar->Scale(1.0 / num_utt_);
  }

  void AccStats(const Matrix<double> &utts_of_this_spk) {
    int32 num_utts = utts_of_this_spk.NumRows();
    //total_covar = SUM_N{ (w_i - u)(w_i - u)^T}
    //            = SUM_C{ SUM_Nc{(w_i - u)(w_i - u)^T} }
    tot_covar_.AddMat2(1.0, utts_of_this_spk, kTrans, 1.0);

    Vector<double> spk_average(Dim());
    // spk_average = 1/Nc * SUM_Nc{ (wi - u) } = 1/Nc * SUM_Nc{ (u_c - u)}
    //  ---> quantion to the (u_c - u)
    spk_average.AddRowSumMat(1.0 / num_utts, utts_of_this_spk);
    // Sb = SUM_C { Nc * (u_c - u)(u_c - u)^T}
    between_covar_.AddVec2(num_utts, spk_average);
    num_utt_ += num_utts;
    num_spk_ += 1;
  }

  /// Will return Empty() if the within-class covariance matrix would be zero.
  bool SingularTotCovar() { return (num_utt_ < Dim()); }
  bool Empty() { return (num_utt_ - num_spk_ == 0); }
  std::string Info() {
    std::ostringstream ostr;
    ostr << num_spk_ << " speakers, " << num_utt_ << " utterances. ";
    return ostr.str();
  }
  int32 Dim() { return tot_covar_.NumRows(); }
  // Use default constructor and assignment operator.
  void AddStats(const CovarianceStats &other) {
    tot_covar_.AddSp(1.0, other.tot_covar_);
    between_covar_.AddSp(1.0, other.between_covar_);
    num_spk_ += other.num_spk_;
    num_utt_ += other.num_utt_;
  }
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(CovarianceStats);
  SpMatrix<double> tot_covar_;
  SpMatrix<double> between_covar_;
  int32 num_spk_;
  int32 num_utt_;
};


template<class Real>
void ComputeNormalizingTransform(const SpMatrix<Real> &covar,
                                 MatrixBase<Real> *proj) {
  int32 dim = covar.NumRows();
  TpMatrix<Real> C(dim);  // Cholesky of covar, covar = C C^T
  C.Cholesky(covar);
  C.Invert();  // The matrix that makes covar unit is C^{-1}, because
               // C^{-1} covar C^{-T} = C^{-1} C C^T C^{-T} = I.
  proj->CopyFromTp(C, kNoTrans);  // set "proj" to C^{-1}.
}

void ComputeBccAndWcc(
    const std::map<std::string, Vector<BaseFloat> *> &utt2ivector,
    const std::map<std::string, std::vector<std::string> > &cluster2utt,
    SpMatrix<double> *Bcc,
    SpMatrix<double> *Wcc) {
  KALDI_ASSERT(!utt2ivector.empty());
  int32 lda_dim = lda_out->NumRows(), dim = lda_out->NumCols();
  KALDI_ASSERT(dim == utt2ivector.begin()->second->Dim());
  KALDI_ASSERT(lda_dim > 0 && lda_dim <= dim);

  CovarianceStats stats(dim);

  std::map<std::string, std::vector<std::string> >::const_iterator iter;
  for (iter = cluster2utt.begin(); iter != cluster2utt.end(); ++iter) {
    const std::vector<std::string> &uttlist = iter->second;
    KALDI_ASSERT(!uttlist.empty());

    int32 N = uttlist.size(); // number of utterances.
    Matrix<double> utts_of_this_spk(N, dim);
    for (int32 n = 0; n < N; n++) {
      std::string utt = uttlist[n];
      KALDI_ASSERT(utt2ivector.count(utt) != 0);
      utts_of_this_spk.Row(n).CopyFromVec(
          *(utt2ivector.find(utt)->second));
    }
    stats.AccStats(utts_of_this_spk);
  }

  KALDI_LOG << "Stats have " << stats.Info();
  KALDI_ASSERT(!stats.Empty());
  KALDI_ASSERT(!stats.SingularTotCovar() &&
               "Too little data for iVector dimension.");


  SpMatrix<double> total_covar;
  stats.GetTotalCovar(&total_covar);
  SpMatrix<double> within_covar;
  stats.GetWithinCovar(&within_covar);
  Wcc->CopyFromSp(within_covar);

  SpMatrix<double> between_covar(total_covar);
  between_covar.AddSp(-1.0, within_covar);
  Bcc->CopyFromSp(between_covar);
}

void ComputeAndSubtractMean(
    std::map<std::string, Vector<BaseFloat> *> utt2ivector,
    Vector<BaseFloat> *mean_out) {
  int32 dim = utt2ivector.begin()->second->Dim();
  size_t num_ivectors = utt2ivector.size();
  Vector<double> mean(dim);
  std::map<std::string, Vector<BaseFloat> *>::iterator iter;
  for (iter = utt2ivector.begin(); iter != utt2ivector.end(); ++iter)
    mean.AddVec(1.0 / num_ivectors, *(iter->second));
  mean_out->Resize(dim);

  // mean_out -- u
  // utt2ivector -- utt - ivector-u
  mean_out->CopyFromVec(mean);
  for (iter = utt2ivector.begin(); iter != utt2ivector.end(); ++iter)
    iter->second->AddVec(-1.0, *mean_out);
}



}

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Compute the Cluster Sbcc and Swcc covariances.This can be add to the LDA Sbc,\n"
        "the file utt2cluster which use to work out the within cluster and between cluster\n"
        "covariance matrices.  \n"
        "and Outputs the Sbcc Swcc , then Combine with the LDA\n"
        "But LDA by default will normalize so that the projected\n"
        "within-class covariance is unit, but if you set --normalize-total-covariance\n"
        "to true, it will normalize the total covariance.\n"
        "So there is a question how to compute a curect Covariance"
        "Usage:  ivector-cluster-cov [options] <ivector-rspecifier> <utt2clustere-rspecifier> "
        "<lda-matrix-out>\n"
        "e.g.: \n"
        " ivector-compute-lda ark:ivectors.ark ark:utt2spk lda.mat\n";

    ParseOptions po(usage);

    BaseFloat total_covariance_factor = 0.0;
    bool binary = false;

    po.Register("total-covariance-factor", &total_covariance_factor,
                "If this is 0.0 we normalize to make the within-class covariance "
                "unit; if 1.0, the total covariance; if between, we normalize "
                "an interpolated matrix.");
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string
        ivector_rspecifier = po.GetArg(1),
        utt2cluster_rspecifier = po.GetArg(2),
        bcc_wxfilename = po.GetArg(3),
        wcc_wxfilename = po.GetArg(4);

    int32 num_done = 0, num_err = 0, dim = 0;

    SequentialBaseFloatVectorReader ivector_reader(ivector_rspecifier);
    RandomAccessTokenReader utt2cluster_reader(utt2cluster_rspecifier);

    std::map<std::string, Vector<BaseFloat> *> utt2ivector;
    std::map<std::string, std::vector<std::string> > cluster2utt;

    for (; !ivector_reader.Done(); ivector_reader.Next()) {
      std::string utt = ivector_reader.Key();
      const Vector<BaseFloat> &ivector = ivector_reader.Value();
      if (utt2ivector.count(utt) != 0) {
        KALDI_WARN << "Duplicate iVector found for utterance " << utt
                   << ", ignoring it.";
        num_err++;
        continue;
      }
      if (!utt2cluster_reader.HasKey(utt)) {
        KALDI_WARN << "utt2spk has no entry for utterance " << utt
                   << ", skipping it.";
        num_err++;
        continue;
      }
      std::string cluster = utt2cluster_reader.Value(utt);
      utt2ivector[utt] = new Vector<BaseFloat>(ivector);
      if (dim == 0) {
        dim = ivector.Dim();
      } else {
        KALDI_ASSERT(dim == ivector.Dim() && "iVector dimension mismatch");
      }
      cluster2utt[cluster].push_back(utt);
      num_done++;
    }

    KALDI_LOG << "Read " << num_done << " utterances, "
              << num_err << " with errors.";

    if (num_done == 0) {
      KALDI_ERR << "Did not read any utterances.";
    } else {
      KALDI_LOG << "Computing within-class covariance.";
    }

    Vector<BaseFloat> mean;
    ComputeAndSubtractMean(utt2ivector, &mean);
    KALDI_LOG << "2-norm of iVector mean is " << mean.Norm(2.0);


    SpMatrix<double> Bcc(dim), Wcc(dim);
    ComputeBccAndWcc(utt2ivector,
                     cluster2utt,
                     &Bcc,
                     &Wcc);


    WriteKaldiObject(Bcc, bcc_wxfilename, binary);
    WriteKaldiObject(Wcc, wcc_wxfilename, binary);



    std::map<std::string, Vector<BaseFloat> *>::iterator iter;
    for (iter = utt2ivector.begin(); iter != utt2ivector.end(); ++iter)
      delete iter->second;
    utt2ivector.clear();

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
