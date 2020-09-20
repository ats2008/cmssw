
#include "gpuVertexTrimmer.h"
#include "CUDADataFormats/Track/interface/PixelTrackHeterogeneous.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
 

 
namespace gpuVertexTrimmer {
void updateChisquareQuantile(size_t ndof, std::vector<double>& maxChi2, double track_prob_min) {
    size_t oldsize = maxChi2.size();
    for (size_t i = oldsize; i <= ndof; ++i) {
      double chi2 = TMath::ChisquareQuantile(1 - track_prob_min, i);
      maxChi2.push_back(chi2);
    }
  }

  ZVertexHeterogeneous Trimmer::make(TkSoA const* tksoa, ZVertexSoA const* VertexSoA) const {
    assert(tksoa);
    assert(VertexSoA);
    
    ZVertexHeterogeneous vertices(std::make_unique<ZVertexSoA>());
    auto& trimmedVertexSoA = *vertices;//.get();
    //auto trimmed_VtxSoA_ptr = std::make_unique<ZVertexSoA>();
    
    auto vertex_soa = *VertexSoA;
    int nv = vertex_soa.nvFinal;

    auto tsoa = *tksoa;
    auto const* quality = tsoa.qualityData();
    unsigned int maxTracks(tsoa.stride());

    double sumpt2;

    trimmedVertexSoA.nvFinal = 0;
    int vtx_id;

    double sumpt2first = 0;
    std::vector<double> track_pT2;
    std::vector<double> maxChi2_;

    unsigned int it = 0;

    for (it = 0; it < maxTracks; it++) {
      trimmedVertexSoA.idv[it] = -1;
      track_pT2.push_back(-1.0);
      auto nHits = tsoa.nHits(it);

      if (nHits == 0)
        break;  // this is a guard: maybe we need to move to nTracks...
      auto q = quality[it];
      if (q != trackQuality::loose)
        continue;  // FIXME
      if (nHits < minNumberOfHits_)
        continue;

      size_t ndof = 2 * nHits - 5;
      float chi2 = tsoa.chi2(it);

      if (track_prob_min_ >= 0. && track_prob_min_ <= 1.) {
        if (maxChi2_.size() <= (ndof))
          updateChisquareQuantile(ndof, maxChi2_, track_prob_min_);
        if (chi2 * ndof > maxChi2_[ndof])
          continue;
      }
      if (chi2 > chi2Max_)
        continue;

      float pt = tsoa.pt(it);
      if (pt < track_pT_min_)
        continue;
      if (pt > track_pT_max_)
        pt = track_pT_max_;
      track_pT2.back() = pt * pt;
    }
    maxTracks = it;

    auto nt = 0;
    for (int j = nv - 1; j >= 0; j--) {
      vtx_id = vertex_soa.sortInd[j];
      sumpt2first = 0;
      nt = 0;
      for (unsigned int k = 0; k < maxTracks; k++) {
        if (vertex_soa.idv[k] != vtx_id)
          continue;
        nt++;
        if (track_pT2[k] < 0)
          continue;
        auto pt2 = track_pT2[k];
        sumpt2first += pt2;
      }
      if (nt > 1)
        break;
    }

    trimmedVertexSoA.nvFinal = 0;
    for (int j = 0; j < nv; j++) {
      if (trimmedVertexSoA.nvFinal >= maxVtx_)
        break;
      vtx_id = vertex_soa.sortInd[j];
      sumpt2 = 0;
      nt = 0;
      for (unsigned int k = 0; k < maxTracks; k++) {
        if (vertex_soa.idv[k] != vtx_id)
          continue;
        nt++;
        if (track_pT2[k] < 0)
          continue;
        sumpt2 += track_pT2[k];
      }
      if (nt < 2)
        continue;
      if (sumpt2 >= sumpt2first * fractionSumPt2_ && sumpt2 > minSumPt2_) {
        auto newVtxId = trimmedVertexSoA.nvFinal;
        for (unsigned int k = 0; k < maxTracks; k++) {
          if (vertex_soa.idv[k] == vtx_id)
            trimmedVertexSoA.idv[k] = newVtxId;
        }

        trimmedVertexSoA.zv[newVtxId] = vertex_soa.zv[vtx_id];
        trimmedVertexSoA.wv[newVtxId] = vertex_soa.wv[vtx_id];
        trimmedVertexSoA.chi2[newVtxId] = vertex_soa.chi2[vtx_id];
        trimmedVertexSoA.sortInd[newVtxId] = newVtxId;
        trimmedVertexSoA.ndof[newVtxId] = vertex_soa.ndof[vtx_id];
        trimmedVertexSoA.ptv2[newVtxId] = vertex_soa.ptv2[vtx_id];
        trimmedVertexSoA.nvFinal++;
      }
    }
    std::cout << "\n";
 //   return ZVertexHeterogeneous(std::move(trimmed_VtxSoA_ptr));
    return vertices; 
  }
}  // namespace gpuVertexTrimmer

