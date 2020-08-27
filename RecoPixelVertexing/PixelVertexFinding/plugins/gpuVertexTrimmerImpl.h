#include "CUDADataFormats/Track/interface/PixelTrackHeterogeneous.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "gpuVertexTrimmer.h"

namespace gpuVertexTrimmer {

  // parallel on all tracks
  // selects th
  __global__ void sumPt2(TkSoA const* ptracks,
                         ZVertexSoA const* pVertexSoa,
                         ZVertexSoA* trimmedVertexSoa,
                         gpuVertexTrimmer::WorkSpaceTr* pws,
                         float ptMin,
                         float ptMax,
                         int minHits) {
    assert(ptracks);
    assert(pVertexSoa);
    assert(trimmedVertexSoa);
    assert(pws);

    auto const& tracks = *ptracks;
    auto& vertexs = *pVertexSoa;
    auto& trimmedVertexs = *trimmedVertexSoa;
    auto& data = *pws;
    auto const* quality = tracks.qualityData();

    auto first = blockIdx.x * blockDim.x + threadIdx.x;
    for (int idx = first, nt = TkSoA::stride(); idx < nt; idx += gridDim.x * blockDim.x) {
      auto nHits = tracks.nHits(idx);
      if (nHits == 0)
        break;  // this is a guard: maybe we need to move to nTracks...

      atomicAdd(&(data.ntrks), 1);
      auto vid = vertexs.idv[idx];
      data.iv[idx] = vid;

      auto ndof = 2 * nHits - 5;
      if (ndof < 0)
        ndof = 0;
      // printf("sumpT2 kernel : checking track id %d with vid : %d ,chi2 = %f ,ndof = %d, chi2max =%f , pt = %f, ptMin =%f,pTmax = %f \n",
      //         idx,vid,tracks.chi2(idx),ndof,data.chi2max[ndof],tracks.pt(idx),ptMin,ptMax);
      if (vid < 0)
        continue;

      atomicAdd(&data.nTracksFromVertex[vid], 1);
      if (nHits < minHits)
        continue;
      if (quality[idx] != trackQuality::loose)
        continue;

      auto chi2 = tracks.chi2(idx);

      if (chi2 > data.chi2max[ndof])
        continue;

      auto pt = tracks.pt(idx);
      if (pt < ptMin)
        continue;
      if (pt > ptMax)
        pt = ptMax;
      atomicAdd(&data.sumPtt2[vid], pt * pt);
      //printf("sumpT2 kernel : adding track id %d with vid : %d ,and pT set as : %f [ tempsum =%f ]\n",idx,vid,pt,tempsum);
    }
  }

  // parallel on all vertices
  __global__ void getPt2max(gpuVertexTrimmer::ZVertices const* oVertices,
                            gpuVertexTrimmer::WorkSpaceTr* pws,
                            float fractionSumPt2) {
    auto& vtxSoa = *oVertices;
    auto& ws = *pws;
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto nv = vtxSoa.nvFinal;

    if (idx < nv) {
      ws.newVtxIds[idx] = -1;
      //     idx=vtxSoa.sortInd[idx];
      printf("in getPt2max , idx = %d , sortIdx = %d ->nv = %d  sumpT2 =%f \n",
             idx,
             vtxSoa.sortInd[idx],
             nv,
             ws.sumPtt2[idx]);
      if (idx == nv - 1) {
        auto sid = vtxSoa.sortInd[idx];
        while (ws.nTracksFromVertex[sid] < 2 and idx > 0) {
          idx--;
          sid = vtxSoa.sortInd[idx];
        }
        //           idx=vtxSoa.sortInd[idx];
        ws.maxSumPt2 = ws.sumPtt2[sid] * fractionSumPt2;
        printf(
            "in getPt2max  FOUND MAX AS : @idx = %d[%d] ,pTMax = %f [ %f ]\n ", idx, nv, ws.maxSumPt2, ws.sumPtt2[sid]);
        //printf("found sortId = %d , ws.maxSumPt2=ws.sumPtt2[%d]*fractionSumPt2 :: %f=%f * %f \n",
        //                      idx,idx,ws.maxSumPt2,ws.sumPtt2[idx],fractionSumPt2);
      }
    }
  }

  // parallel on all vertices
  __global__ void vertexTrimmer(gpuVertexTrimmer::ZVertices* trimmedSoA,
                                gpuVertexTrimmer::ZVertices const* oVertices,
                                gpuVertexTrimmer::WorkSpaceTr* pws,
                                float sumPtMin) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto& origVtxs = *oVertices;
    auto& trimmedVertices = *trimmedSoA;
    auto& ws = *pws;
    auto nv = origVtxs.nvFinal;

    if (idx < nv) {
      //        idx=origVtxs.sortInd[idx];
      printf("ntrks = %d, sumPt2 = %f , sumpTmin = %f , sumPtMax = %f \n",
             ws.nTracksFromVertex[idx],
             ws.sumPtt2[idx],
             sumPtMin,
             ws.maxSumPt2);
      if (ws.nTracksFromVertex[idx] > 1 and ws.sumPtt2[idx] > sumPtMin and ws.sumPtt2[idx] > ws.maxSumPt2) {
        auto i = atomicAdd(&trimmedVertices.nvFinal, 1);
        ws.newVtxIds[idx] = i;
        trimmedVertices.zv[i] = origVtxs.zv[idx];
        trimmedVertices.wv[i] = origVtxs.wv[idx];
        trimmedVertices.chi2[i] = origVtxs.chi2[idx];
        trimmedVertices.ptv2[i] = origVtxs.ptv2[idx];
        trimmedVertices.ndof[i] = origVtxs.ndof[idx];
        trimmedVertices.sortInd[i] = i;  //TODO
        printf("adding new i = %d old idx = %d {zv : %f ,wv : %f ,chi2 : %f ,ptv2 : %f ,ndof : %d }\n",
               i,
               idx,
               trimmedVertices.zv[i],
               trimmedVertices.wv[i],
               trimmedVertices.chi2[i],
               trimmedVertices.ptv2[i],
               trimmedVertices.ndof[i]);

        //           printf("adding new i = %d old idx = %d ptv2_original =%f , here : %f, max = %f\n",i,idx,trimmedVertices.ptv2[i],ws.sumPtt2[idx],ws.maxSumPt2);
      }
      //    else
      //    {
      //        ws.newVtxIds[idx]=-1;
      //    }
    }
  }

  // parallel on all tracks
  __global__ void updateTrackVertexMap(gpuVertexTrimmer::ZVertices const* oVertices,
                                       gpuVertexTrimmer::ZVertices* tVertices,
                                       gpuVertexTrimmer::WorkSpaceTr* pws) {
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

      // printf("updating vertex map of %d[%d], from %d to %d \n ",idx,ws.ntrks,vid,trimmedVertices.idv[idx]) ;
    }
  }

#ifdef __CUDACC__
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

    //  for (size_t i =0; i <=ndof; i++)
    //  std::cout<<" ndof = "<<i<<maxChi2[i]<<"\n";
  }

  ZVertexHeterogeneous Trimmer::makeAsync(cudaStream_t stream, TkSoA const* tksoa, ZVertexSoA const* VertexSoA) const {
    std::cout << " Starting async func \n ";
    ZVertexHeterogeneous vertices(cms::cuda::make_device_unique<ZVertexSoA>(stream));

    auto* trimmedVertexSoA = vertices.get();
    //std::cout<<"Going for asserts : ";
    assert(VertexSoA);
    //std::cout<<" VertexSoA done ";
    assert(tksoa);
    //std::cout<<" tksoa done ";
    assert(trimmedVertexSoA);
    std::cout << " trimmedVertexSoA done \n";

    auto ws_tr = cms::cuda::make_device_unique<WorkSpaceTr>(stream);
    auto* workspace = ws_tr.get();

    auto maxchi2valsCPU = cms::cuda::make_host_unique<float[]>(MAX_NDOF_EXPECTED + 1, stream);
    updateMaxChi2(MAX_NDOF_EXPECTED, maxchi2valsCPU.get(), track_prob_min_, chi2Max_);
    cudaMemcpyAsync(workspace->chi2max,
                    maxchi2valsCPU.get(),
                    (MAX_NDOF_EXPECTED + 1) * sizeof(float),
                    cudaMemcpyHostToDevice,
                    stream);

    //std::cout<<"Going to the init kernel \n"; //with numberOfBlocks = "<<numberOfBlocks<<" mN"<<minNumberOfHits_<<" c2M "<<chi2Max_<<"\n";
    init<<<1, 1, 0, stream>>>(trimmedVertexSoA, ws_tr.get());
    //std::cout<<"out of init kernel \n";
    cudaCheck(cudaGetLastError());
    std::cout << "passed checkerror of init kernel \n";

    auto blockSize = 128;

    auto numberOfBlocks = (TkSoA::stride() + blockSize - 1) / blockSize;
    //   loadTracksForTrimmer<<<numberOfBlocks, blockSize, 0, stream>>>(tksoa, trimmedVertexSoA, workspace, track_pT_min_);
    //   cudaCheck(cudaGetLastError());

    //std::cout<<"Going to the sumPt2 kernel with numberOfBlocks = "<<numberOfBlocks<<" mN"<<minNumberOfHits_<<" c2M "<<chi2Max_<<"\n";
    sumPt2<<<numberOfBlocks, blockSize, 0, stream>>>(
        tksoa, VertexSoA, trimmedVertexSoA, workspace, track_pT_min_, track_pT_max_, minNumberOfHits_);
    //std::cout<<"out of sumPt2 kernel \n";
    cudaCheck(cudaGetLastError());
    // std::cout<<"passed checkerror of sumPt2 kernel \n";

    //std::cout<<"Going to the getPt2max kernel with numberOfBlocks = "<<numberOfBlocks<<"  fractionSumPt2 = "<<fractionSumPt2_<<"\n";
    getPt2max<<<numberOfBlocks, blockSize, 0, stream>>>(VertexSoA, workspace, fractionSumPt2_);
    // std::cout<<"out of getPt2Max kernel \n";
    cudaCheck(cudaGetLastError());
    //std::cout<<"passed checkerror of getPt2max kernel \n";

    //std::cout<<"updating the number of Blocks now with VertexSoA->nvFinal - ...\n ";
    numberOfBlocks = (20 + blockSize - 1) / blockSize;
    //std::cout<<"Going to the vertextrimmer kernel with numberOfBlocks = "<<numberOfBlocks<<"\n";
    vertexTrimmer<<<numberOfBlocks, blockSize, 0, stream>>>(trimmedVertexSoA, VertexSoA, workspace, minSumPt2_);
    //std::cout<<"out of vertexTrimmer kernel \n";
    cudaCheck(cudaGetLastError());
    //std::cout<<"passed checkerror of vertextrimmer kernel \n";

    numberOfBlocks = (TkSoA::stride() + blockSize - 1) / blockSize;
    //std::cout<<"Going to the updateTrackVertexMap kernel with numberOfBlocks = "<<numberOfBlocks<<"\n";
    updateTrackVertexMap<<<numberOfBlocks, blockSize, 0, stream>>>(VertexSoA, trimmedVertexSoA, workspace);
    //std::cout<<"out of updateTrackVertexMap kernel \n";
    cudaCheck(cudaGetLastError());
    //std::cout<<"passed checkerror of updateTrackVertexMap kernel \n";
    std::cout << "exiting AsyncFuc \n";

    return vertices;
  }

#endif

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
