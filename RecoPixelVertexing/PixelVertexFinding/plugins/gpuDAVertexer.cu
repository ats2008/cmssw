#include "gpuDAVertexer.h"

namespace gpuDAVertexer{

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
 
    ZVertexHeterogeneous vertices(cms::cuda::make_device_unique<ZVertexSoA>(stream) );

    assert(workspace);


    return vertices;
}


}
