#include "gpuDAVertexer.h"

namespace gpuDAVertexer{


 __global__ void init(ZVertexSoA* vtxs, Workspace* ws) {
        printf(" device function !! [ init ]  \n");
         vtxs->init();
         ws->init();
   
  }

 __global__ void load_tracks(pixelTrack::TrackSoA const* tracks_p, Workspace* ws_p )
{
   auto const& tracks = *tracks_p;
   auto& ws = *ws_p;

   auto idx=blockIdx.x*blockDim.x + threadIdx.x;
   
   for( int i=idx,nt=pixelTrack::TrackSoA::stride();idx<nt;idx+=gridDim.x*blockDim.x)
   {
    uint16_t nhits=tracks.nHits(i);
    if(nhits==0) break;
    
    auto it = atomicAdd(&ws.ntrks, 1);
    ws.zt[it]=tracks.zip(i);
    ws.pi[it]=tracks.tip(i);
    ws.dz2[it]=tracks.stateAtBS.covariance(i)(14);
    ws.itrk[it]=i;
   }

}



ZVertexHeterogeneous DAVertexer::makeAsync(cudaStream_t stream,const pixelTrack::TrackSoA* tracks )
{
   auto wSpace_ =  cms::cuda::make_device_unique<Workspace>(stream);
   auto *workspace = wSpace_.get();
   assert(workspace);
   
   ZVertexHeterogeneous vertices_(cms::cuda::make_device_unique<ZVertexSoA>(stream) );
   auto* vertices = vertices_.get();
    
   init<<<1, 1, 0>>>(vertices, workspace);
   
    auto blockSize = 128;
    auto numberOfBlocks = (pixelTrack::TrackSoA::stride()+ blockSize - 1) / blockSize;
    load_tracks<<<blockSize,numberOfBlocks,0>>>(tracks,workspace);

    return vertices_;
}


}
