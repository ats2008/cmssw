#include "gpuVertexTrimmer.h"
#include "HeterogeneousCore/CUDAUtilities/interface/radixSort.h"


namespace gpuVertexTrimmer {
 
    __global__ void initWorkspace(WorkSpace* pws){
    auto idx =blockIdx.x * blockDim.x + threadIdx.x;
      if (idx < ZVertexSoA::MAXVTX) {
        pws->sumPtt2[idx] = 0.0;
        pws->nTracksFromVertex[idx] = 0;
      }
    }
    __global__ void sortByPt2(ZVertices* pdata) {

    auto& __restrict__ data = *pdata;
    uint32_t const& nvFinal = data.nvFinal;
    float* __restrict__ ptv2 = data.ptv2;
    uint16_t* __restrict__ sortInd = data.sortInd;

    if (nvFinal < 1)
      return;

    if (1 == nvFinal) {
      if (threadIdx.x == 0)
        sortInd[0] = 0;
      return;
    }
    __shared__ uint16_t sws[1024];
    
    // sort using only 16 bits
    radixSort<float, 2>(ptv2, sortInd, sws, nvFinal);
    
    }

  
  // parallel on all tracks
  __global__ void sumPt2(TkSoA const* ptracks,
                         ZVertexSoA const* pVertexSoa,
                         gpuVertexTrimmer::WorkSpace* pws,
                         float ptMin,
                         float ptMax,
                         int minHits) {
    assert(ptracks);
    assert(pVertexSoa);
    assert(pws);

    auto const& tracks = *ptracks;
    auto& vertexs = *pVertexSoa;
    auto& ws = *pws;
    auto const* quality = tracks.qualityData();

    auto first = blockIdx.x * blockDim.x + threadIdx.x;
    for (int idx = first, nt = TkSoA::stride(); idx < nt; idx += gridDim.x * blockDim.x) {
      auto nHits = tracks.nHits(idx);
      if (nHits == 0)
        break;  // this is a guard: maybe we need to move to nTracks...

      atomicAdd(&(ws.ntrks), 1);
      auto vid = vertexs.idv[idx];

      auto ndof = 2 * nHits - 5;
      if (ndof < 0)
        ndof = 0;
      if (vid < 0)
        continue;

      atomicAdd(&ws.nTracksFromVertex[vid], 1);
      if (nHits < minHits)
        continue;
      if (quality[idx] != trackQuality::loose)
        continue;

      auto chi2 = tracks.chi2(idx);

      if (chi2 > ws.chi2max[ndof])
        continue;

      auto pt = tracks.pt(idx);
      if (pt < ptMin)
        continue;
      if (pt > ptMax)
        pt = ptMax;
      atomicAdd(&ws.sumPtt2[vid], pt * pt);
    }
  }

  // parallel on all vertices
  __global__ void getPt2max(gpuVertexTrimmer::ZVertices const* oVertices,
                            gpuVertexTrimmer::WorkSpace* pws,
                            float fractionSumPt2) {
    auto& vtxSoa = *oVertices;
    auto& ws = *pws;
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto nv = vtxSoa.nvFinal;

    if (idx < nv) {
      ws.newVtxIds[idx] = -1;
      if (idx == nv - 1) {
        auto sid = vtxSoa.sortInd[idx];
        while (ws.nTracksFromVertex[sid] < 2 and idx > 0) {
          idx--;
          sid = vtxSoa.sortInd[idx];
        }
        ws.maxSumPt2 = ws.sumPtt2[sid] * fractionSumPt2;
      }
    }
  }

  // parallel on all vertices
  __global__ void vertexTrimmer(gpuVertexTrimmer::ZVertices* trimmedSoA,
                                gpuVertexTrimmer::ZVertices const* oVertices,
                                gpuVertexTrimmer::WorkSpace* pws,
                                float sumPtMin) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto& origVtxs = *oVertices;
    auto& trimmedVertices = *trimmedSoA;
    auto& ws = *pws;
    auto nv = origVtxs.nvFinal;

    if (idx < nv) {
      if (ws.nTracksFromVertex[idx] > 1 and ws.sumPtt2[idx] > sumPtMin and ws.sumPtt2[idx] > ws.maxSumPt2) {
        auto i = atomicAdd(&trimmedVertices.nvFinal, 1);
        ws.newVtxIds[idx] = i;
        trimmedVertices.zv[i] = origVtxs.zv[idx];
        trimmedVertices.wv[i] = origVtxs.wv[idx];
        trimmedVertices.chi2[i] = origVtxs.chi2[idx];
        trimmedVertices.ptv2[i] = origVtxs.ptv2[idx];
        trimmedVertices.ndof[i] = origVtxs.ndof[idx];
      }
    }
  }

  // parallel on all tracks
  __global__ void updateTrackVertexMap(gpuVertexTrimmer::ZVertices const* oVertices,
                                       gpuVertexTrimmer::ZVertices* tVertices,
                                       gpuVertexTrimmer::WorkSpace* pws) {
    assert(oVertices);
    assert(tVertices);
    assert(pws);

    auto& pVertices = *oVertices;
    auto& trimmedVertices = *tVertices;
    auto& ws = *pws;

    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < ZVertexSoA::MAXVTX)
      trimmedVertices.idv[idx] = -1;

    if (idx < ws.ntrks) {
      auto vid = pVertices.idv[idx];
      if (vid > -1)
        trimmedVertices.idv[idx] = ws.newVtxIds[vid];

    }
  }


  void updateMaxChi2(size_t ndof, float* maxChi2, float track_prob_min, float maxChi2set) {
    if (track_prob_min > 0 and track_prob_min <= 1.0) {
      for (size_t i = 1; i <= ndof; i++) {
        maxChi2[i] = TMath::ChisquareQuantile(1 - track_prob_min, i) / i;
        if (maxChi2[i] > maxChi2set)
          maxChi2[i] = maxChi2set;
      }
    } else {
      for (size_t i = 0; i <= ndof; i++) {
        maxChi2[i] = maxChi2set;
      }
    }
    maxChi2[0] = -1.0;

  }

  ZVertexHeterogeneous Trimmer::makeAsync(cudaStream_t stream, TkSoA const* tksoa, ZVertexSoA const* VertexSoA) const {
 
    ZVertexHeterogeneous vertices(cms::cuda::make_device_unique<ZVertexSoA>(stream));

    auto* trimmedVertexSoA = vertices.get();
    assert(VertexSoA);
    assert(tksoa);
    assert(trimmedVertexSoA);

    auto ws_tr = cms::cuda::make_device_unique<WorkSpace>(stream);
    auto* workspace = ws_tr.get();

    auto maxchi2valsCPU = cms::cuda::make_host_unique<float[]>(20, stream);
    updateMaxChi2(20, maxchi2valsCPU.get(), track_prob_min_, chi2Max_);
    cudaMemcpyAsync(workspace->chi2max,
                    maxchi2valsCPU.get(),
                    20 * sizeof(float),
                    cudaMemcpyHostToDevice,
                    stream);

    init<<<1, 1, 0, stream>>>(trimmedVertexSoA, ws_tr.get());
 
    auto blockSize = 128;
    auto numberOfBlocks = (ZVertexSoA::MAXVTX + blockSize - 1) / blockSize;

    initWorkspace<<<numberOfBlocks,blockSize,0,stream>>>(workspace);
    cudaCheck(cudaGetLastError());
    
    numberOfBlocks = (TkSoA::stride() + blockSize - 1) / blockSize;
    sumPt2<<<numberOfBlocks, blockSize, 0, stream>>>(
        tksoa, VertexSoA, workspace, track_pT_min_, track_pT_max_, minNumberOfHits_);
    cudaCheck(cudaGetLastError());

    numberOfBlocks = (ZVertexSoA::MAXVTX + blockSize - 1) / blockSize;
    getPt2max<<<numberOfBlocks, blockSize, 0, stream>>>(VertexSoA, workspace, fractionSumPt2_);
    cudaCheck(cudaGetLastError());

    vertexTrimmer<<<numberOfBlocks, blockSize, 0, stream>>>(trimmedVertexSoA, VertexSoA, workspace, minSumPt2_);
    cudaCheck(cudaGetLastError());

    numberOfBlocks = (TkSoA::stride() + blockSize - 1) / blockSize;
    updateTrackVertexMap<<<numberOfBlocks, blockSize, 0, stream>>>(VertexSoA, trimmedVertexSoA, workspace);
    cudaCheck(cudaGetLastError());
    
    numberOfBlocks = (ZVertexSoA::MAXVTX + blockSize - 1) / blockSize;
    sortByPt2<<<1,1024 - 256,0,stream>>>(trimmedVertexSoA);
    cudaCheck(cudaGetLastError());

    return vertices;
  }

}  // namespace gpuVertexTrimmer

