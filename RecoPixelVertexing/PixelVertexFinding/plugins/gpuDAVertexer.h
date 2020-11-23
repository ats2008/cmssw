#ifndef RecoPixelVertexing_PixelVertexFinding_gpuDAVertexr_h
#define RecoPixelVertexing_PixelVertexFinding_gpuDAVertexr_h

#include "CUDADataFormats/Vertex/interface/ZVertexHeterogeneous.h"

namespace gpuDAVertexer{

    using ZVertices = ZVertexSoA;
    using TkSoA = pixelTrack::TrackSoA;

    struct Workspace{
    static constexpr uint32_t MAXTRACKS = ZVertexSoA::MAXTRACKS;
    static constexpr uint32_t MAXVTX = ZVertexSoA::MAXTRACKS;

    uint32_t ntrks;
    uint16_t itrk[MAXTRACKS];
    float zt[MAXTRACKS];
    float dz2[MAXTRACKS];
    float pi[MAXTRACKS];

    float zVtx[MAXVTX];
    
    

    __host__ __device__ void init() {

            ntrks=0;

    }

    };

    class DAVertexer{

    public:
        DAVertexer( float zSplit): zSplit_(zSplit)       {}
            
        ZVertexHeterogeneous makeAsync(cudaStream_t stream,TkSoA const * tks);
        
        float get_splitSeparation() { return zSplit_ ;}

    private:
        float zSplit_;

   };
}

#endif
