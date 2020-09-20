#include "gpuVertexTrimmer.h"
#include "HeterogeneousCore/CUDAUtilities/interface/radixSort.h"


namespace gpuVertexTrimmer {
 
    __global__ void initWorkspaceTr(WorkSpaceTr* pws){
    auto idx =blockIdx.x * blockDim.x + threadIdx.x;
        printf("init idx = %d [MAXVTX = %d ]\n",idx,ZVertexSoA::MAXVTX);
      if (idx < ZVertexSoA::MAXVTX) {
        pws->sumPtt2[idx] = 0.0;
        pws->nTracksFromVertex[idx] = 0;
        pws->newVtxIds[idx] = -1;
      }
    }
      // parallel on all tracks
  __global__ void sumPt2(TkSoA const* ptracks,
                         ZVertexSoA const* pVertexSoa,
                         ZVertexSoA * trimmedVertexSoa,
                         gpuVertexTrimmer::WorkSpaceTr* pws,
                         float ptMin,
                         float ptMax,
                         int minHits) {
    assert(ptracks);
    assert(pVertexSoa);
    assert(pws);

    auto const& tracks = *ptracks;
    auto& vertexs = *pVertexSoa;
    auto& trimmedVertises = *trimmedVertexSoa;
    auto& ws = *pws;
    auto const* quality = tracks.qualityData();

    auto first = blockIdx.x * blockDim.x + threadIdx.x;
 //   printf("@ @sumpT2 with idx  = %d : ZVertexSoA::MAXTRACKS = %d\n ",first, ZVertexSoA::MAXTRACKS);

    //initialization of the track-vertex map for trimmed veretices
    for (int idx = first, nt = TkSoA::stride(); idx < nt; idx += gridDim.x * blockDim.x) 
    {
        trimmedVertises.idv[idx]=-1;
    }
    for (int idx = first, nt = TkSoA::stride(); idx < nt; idx += gridDim.x * blockDim.x) {
      auto nHits = tracks.nHits(idx);
      if (nHits == 0)
        break;  // this is a guard: maybe we need to move to nTracks...

      atomicAdd(&(ws.ntrks), 1);
      auto vid = vertexs.idv[idx];

      auto ndof = 2 * nHits - 5;
      if (ndof < 0)
        ndof = 0;
   //    printf("sumpT2 kernel : checking trk id %d with vid : %d ,chi2 = %f ,ndof = %d, chi2max =%f , pt = %f, ptMin =%f,pTmax = %f \n",
   //            idx,vid,tracks.chi2(idx),ndof,ws.chi2max[ndof],tracks.pt(idx),ptMin,ptMax);
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
    printf("@ getPt2max with idx  = %d : nv = %d\n ",idx,nv);
     
    if (idx == nv - 1) {
        auto sid = vtxSoa.sortInd[idx];
        while (ws.nTracksFromVertex[sid] < 2 and idx > 0) {
          idx--;
          sid = vtxSoa.sortInd[idx];
        }
        ws.maxSumPt2 = ws.sumPtt2[sid] * fractionSumPt2;
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
    printf("@ vertexTrimmer with idx  = %d : nv = %d\n ",idx,nv);

    if (idx < nv) {
      //        idx=origVtxs.sortInd[idx];
      //printf("ntrks = %d, sumPt2 = %f , sumpTmin = %f , sumPtMax = %f \n",
      //       ws.nTracksFromVertex[idx],
      //       ws.sumPtt2[idx],
      //       sumPtMin,
      //       ws.maxSumPt2);
      if (ws.nTracksFromVertex[idx] > 1 and ws.sumPtt2[idx] > sumPtMin and ws.sumPtt2[idx] > ws.maxSumPt2) {
        auto i = atomicAdd(&trimmedVertices.nvFinal, 1);
        ws.newVtxIds[idx] = i;
        trimmedVertices.zv[i] = origVtxs.zv[idx];
        trimmedVertices.wv[i] = origVtxs.wv[idx];
        trimmedVertices.chi2[i] = origVtxs.chi2[idx];
        trimmedVertices.ptv2[i] = origVtxs.ptv2[idx];
        trimmedVertices.ndof[i] = origVtxs.ndof[idx];
     //   trimmedVertices.sortInd[i] = i;  //TODO
        //printf("adding new i = %d old idx = %d {zv : %f ,wv : %f ,chi2 : %f ,ptv2 : %f ,ndof : %d }\n",
        //       i,
        //       idx,
        //       trimmedVertices.zv[i],
        //       trimmedVertices.wv[i],
        //       trimmedVertices.chi2[i],
        //       trimmedVertices.ptv2[i],
        //       trimmedVertices.ndof[i]);

        //           printf("adding new i = %d old idx = %d ptv2_original =%f , here : %f, max = %f\n",i,idx,trimmedVertices.ptv2[i],ws.sumPtt2[idx],ws.maxSumPt2);
      }
      //    else
      //    {
      //        ws.newVtxIds[idx]=-1;
      //    }
    }
  }

  // parallel on tracks
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
    printf("@ updateTrackVertexMap with idx  = %d : ws.ntrks = %d\n ",idx,ws.ntrks);
    if (idx < ws.ntrks) {
      auto vid = pVertices.idv[idx];
      if (vid > -1)
        trimmedVertices.idv[idx] = ws.newVtxIds[vid];
      // printf("updating vertex map of %d[%d], from %d to %d \n ",idx,ws.ntrks,vid,trimmedVertices.idv[idx]) ;
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

  __global__ void printVtx(gpuVertexTrimmer::ZVertices* tVertices)
  {
    assert(tVertices);

    auto& trimmedVertices = *tVertices;

    auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < trimmedVertices.nvFinal) {
      printf("idx = %d ptt2 = %f , sortInd = %d  \n ",idx,trimmedVertices.ptv2[idx],trimmedVertices.sortInd[idx]) ;
  }
  }


    __global__ void initWorkspaceTrWrapper(WorkSpaceTr* pws,const ZVertexSoA* Verteices  ){
        auto nv= Verteices->nvFinal;
        auto blockSize=128;
        auto numberOfBlocks =(nv +blockSize -1)/blockSize;
        if(nv<blockSize) blockSize=nv;
        initWorkspaceTr<<<numberOfBlocks,blockSize,0>>>(pws);
    }

   __global__ void getPt2maxWraper(gpuVertexTrimmer::ZVertices const* oVertices,
                            gpuVertexTrimmer::WorkSpaceTr* pws,
                            float fractionSumPt2) {
        auto nv= oVertices->nvFinal;
        auto blockSize=128;
        auto numberOfBlocks =(nv +blockSize -1)/blockSize;
        if(nv<blockSize) blockSize=nv;
       getPt2maxWraper<<<numberOfBlocks,blockSize,0>>>(oVertices,pws,fractionSumPt2);
  }

  __global__ void vertexTrimmerWraper(gpuVertexTrimmer::ZVertices* trimmedSoA,
                                gpuVertexTrimmer::ZVertices const* oVertices,
                                gpuVertexTrimmer::WorkSpaceTr* pws,
                                float sumPtMin) {
        auto nv= oVertices->nvFinal;
        auto blockSize=128;
        auto numberOfBlocks =(nv +blockSize -1)/blockSize;
        if(nv<blockSize) blockSize=nv;
       vertexTrimmer<<<numberOfBlocks,blockSize,0>>>(trimmedSoA, oVertices, pws,sumPtMin);
  }

  __global__ void updateTrackVertexMapWraper(gpuVertexTrimmer::ZVertices const* oVertices,
                                       gpuVertexTrimmer::ZVertices* tVertices,
                                       gpuVertexTrimmer::WorkSpaceTr* pws) {
        auto nt= pws->ntrks;
        auto blockSize=128;
        auto numberOfBlocks =(nt +blockSize -1)/blockSize;
        if(nt<blockSize) blockSize=nt;
        updateTrackVertexMap<<<numberOfBlocks,blockSize,0>>>(oVertices,tVertices,pws);

  }

  void setMaxChi2(size_t ndof, float* maxChi2, float track_prob_min, float maxChi2set);

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

    auto maxchi2valsCPU = cms::cuda::make_host_unique<float[]>(20, stream);
    setMaxChi2(20, maxchi2valsCPU.get(), track_prob_min_, chi2Max_);
    std::cout<<"Chi2 Max Values made as \n" ;
    for(int i=0;i<8;i++)
        std::cout<<"ndof  = "<<i<<" chi2Max = "<<maxchi2valsCPU[i]<<"\n";
    cudaMemcpyAsync(workspace->chi2max,
                    maxchi2valsCPU.get(),
                    20 * sizeof(float),
                    cudaMemcpyHostToDevice,
                    stream);

    init<<<1, 1, 0, stream>>>(trimmedVertexSoA, workspace);
    initWorkspaceTrWrapper<<<1,1,0,stream>>>(workspace,VertexSoA);
    cudaCheck(cudaGetLastError());

    auto blockSize = 128;

    auto numberOfBlocks = (TkSoA::stride() + blockSize - 1) / blockSize;

    printf("@ sumpT2 launched with numberOfBlocks  = %d : blockSize = %d\n ",numberOfBlocks,blockSize );
    sumPt2<<<numberOfBlocks, blockSize, 0, stream>>>(tksoa, VertexSoA, trimmedVertexSoA, workspace, track_pT_min_, track_pT_max_, minNumberOfHits_);
    cudaCheck(cudaGetLastError());
    
    printf("@getPt2maxWraper launched\n " );
    getPt2maxWraper<<<1,1,0, stream>>>(VertexSoA, workspace, fractionSumPt2_);
    cudaCheck(cudaGetLastError());
    printf("@getPt2maxWraper sucessfully done\n " );

    printf("@vertexTrimmerWraper launched\n " );
    vertexTrimmerWraper<<<1,1,0, stream>>>(trimmedVertexSoA, VertexSoA, workspace, minSumPt2_);
    cudaCheck(cudaGetLastError());
    printf("@vertexTrimmerWraper done\n " );
   
    printf("@updateTrackVertexMapWraper launched\n " );
    updateTrackVertexMapWraper<<<1,1,0, stream>>>(VertexSoA, trimmedVertexSoA, workspace);
    cudaCheck(cudaGetLastError());
    printf("@updateTrackVertexMapWraper done\n " );
    
    printf("@sortByPt2 launched\n " );
    sortByPt2<<<1,1024 - 256,0,stream>>>(trimmedVertexSoA);
    cudaCheck(cudaGetLastError());
    printf("@sortByPt2 done\n " );
    
    printf("@peintVtx launched\n " );
    printVtx<<<numberOfBlocks,blockSize,0,stream>>>(trimmedVertexSoA);
    cudaCheck(cudaGetLastError());
    
    std::cout << "exiting AsyncFuc \n";

    return vertices;
  }
 
  void setMaxChi2(size_t ndof, float* maxChi2, float track_prob_min, float maxChi2set) {
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


}  // namespace gpuVertexTrimmer

