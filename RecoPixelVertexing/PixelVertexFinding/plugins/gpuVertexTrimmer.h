#ifndef RecoPixelVertexing_PixelVertexFinding_src_gpuVertexTrimmer_h
#define RecoPixelVertexing_PixelVertexFinding_src_gpuVertexTrimmer_h

#include <cstddef>
#include <cstdint>

#include "CUDADataFormats/Vertex/interface/ZVertexHeterogeneous.h"
#include "TMath.h"

namespace gpuVertexTrimmer {

  using ZVertices = ZVertexSoA;
  using TkSoA = pixelTrack::TrackSoA;

 struct WorkSpaceTr {
    static constexpr uint32_t MAXTRACKS = ZVertexSoA::MAXTRACKS;
    static constexpr uint32_t MAXVTX = ZVertexSoA::MAXVTX;

    uint32_t ntrks;                        // number of "selected tracks"
    uint16_t itrk[MAXTRACKS];              // index of original track
    float chi2max[20];                     // chi2max/ndof for ndof=1,...,20 for pixel trk max(ndof)=5
    float sumPtt2[MAXVTX];                 // sum pt^2 for each new vertex
    int32_t nTracksFromVertex[MAXVTX];     // index of the vertex in trimmed collection
    int32_t newVtxIds[MAXVTX];             // index of the vertex in trimmed collection
    float maxSumPt2;
    __host__ __device__ void init() {
      ntrks = 0;
      maxSumPt2 = 0.0;
    }
  };
  
 __global__ void init(ZVertexSoA* pdata, WorkSpaceTr* pws) {
    pdata->init();
    pws->init();
  }

  class Trimmer {
  public:
    using ZVertices = ZVertexSoA;
    using TkSoA = pixelTrack::TrackSoA;

    Trimmer(int maxVtx,
            float fractionSumPt2,
            float minSumPt2,
            int minNumberOfHits,
            float track_pT_min,
            float track_pT_max,
            float track_prob_min,
            float chi2Max)
        : maxVtx_(maxVtx),
          fractionSumPt2_(fractionSumPt2),
          minSumPt2_(minSumPt2),
          minNumberOfHits_(minNumberOfHits),
          track_pT_min_(track_pT_min),
          track_pT_max_(track_pT_max),
          track_prob_min_(track_prob_min),
          chi2Max_(chi2Max) {}

    ~Trimmer() = default;
    ZVertexHeterogeneous makeAsync(cudaStream_t stream, TkSoA const* tksoa, ZVertexSoA const* VertexSoA) const;
    ZVertexHeterogeneous make(TkSoA const* tksoa, ZVertexSoA const* VertexSoA) const;

  private:
    unsigned int maxVtx_;   // max output collection size (number of accepted vertices)
    float fractionSumPt2_;  // Threshold on sumPt2 fraction of the leading vertex
    float minSumPt2_;       // min sum od pT^2 at a vertex for selection
    int minNumberOfHits_;   // min mumber of tracks in track selection TODO generalize 'loadTracks' form minNumberOfHits
    float track_pT_min_;    // min track pT for track selection
    float track_pT_max_;    // cap over track pT
    float track_prob_min_;  // min track_prob TODO figure out how to use the chi2quantile
    float chi2Max_;         // TODO check original legacy code and adapt accordingly
  };
}  // namespace gpuVertexTrimmer

#endif  // RecoPixelVertexing_PixelVertexFinding_src_gpuVertexTrimmer_h
