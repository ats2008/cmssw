#include <cuda_runtime.h>
#include "CUDADataFormats/Common/interface/Product.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"

#include "gpuVertexTrimmer.h"

//
// class declaration
//
class PixelVertexCollectionTrimmerCUDA : public edm::global::EDProducer<> {
public:
  using ZTrackCUDAProduct = cms::cuda::Product<PixelTrackHeterogeneous>;
  explicit PixelVertexCollectionTrimmerCUDA(const edm::ParameterSet&);
  ~PixelVertexCollectionTrimmerCUDA() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  // void beginStream(edm::StreamID) override;
  void produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  //  void endStream() override;

  // ----------member data ---------------------------

  bool m_OnGPU;

  const edm::EDGetTokenT<ZTrackCUDAProduct> tokenGPUTrack_;
  const edm::EDGetTokenT<ZVertexCUDAProduct> tokenGPUVertex_;
  const edm::EDPutTokenT<ZVertexCUDAProduct> tokenGPUTrimmedVertex_;

  const edm::EDGetTokenT<PixelTrackHeterogeneous> tokenCPUTrack_;
  const edm::EDGetTokenT<ZVertexHeterogeneous> tokenCPUVertex_;
  const edm::EDPutTokenT<ZVertexHeterogeneous> tokenCPUTrimmedVertex_;

  const gpuVertexTrimmer::Trimmer m_gpuAlgo;
};
// constructors and destructor
PixelVertexCollectionTrimmerCUDA::PixelVertexCollectionTrimmerCUDA(const edm::ParameterSet& iConfig)
    : m_OnGPU(iConfig.getParameter<bool>("onGPU")),
      tokenGPUTrack_(m_OnGPU ? consumes<ZTrackCUDAProduct>(iConfig.getParameter<edm::InputTag>("pixelTrackSrc"))
                             : edm::EDGetTokenT<ZTrackCUDAProduct>{}),
      tokenGPUVertex_(m_OnGPU ? consumes<ZVertexCUDAProduct>(iConfig.getParameter<edm::InputTag>("pixelVertexSrc"))
                              : edm::EDGetTokenT<ZVertexCUDAProduct>{}),
      tokenGPUTrimmedVertex_(m_OnGPU ? produces<ZVertexCUDAProduct>() : edm::EDPutTokenT<ZVertexCUDAProduct>{}),

      tokenCPUTrack_(!m_OnGPU ? consumes<PixelTrackHeterogeneous>(iConfig.getParameter<edm::InputTag>("pixelTrackSrc"))
                              : edm::EDGetTokenT<PixelTrackHeterogeneous>{}),
      tokenCPUVertex_(!m_OnGPU ? consumes<ZVertexHeterogeneous>(iConfig.getParameter<edm::InputTag>("pixelVertexSrc"))
                               : edm::EDGetTokenT<ZVertexHeterogeneous>{}),
      tokenCPUTrimmedVertex_(!m_OnGPU ? produces<ZVertexHeterogeneous>() : edm::EDPutTokenT<ZVertexHeterogeneous>{}),
      m_gpuAlgo(iConfig.getParameter<unsigned int>("maxVtx"),
                iConfig.getParameter<double>("fractionSumPt2"),
                iConfig.getParameter<double>("minSumPt2"),
                iConfig.getParameter<int>("minNumberOfHits"),
                iConfig.getParameter<double>("track_pT_min"),
                iConfig.getParameter<double>("track_pT_max"),
                iConfig.getParameter<double>("track_prob_min"),
                iConfig.getParameter<double>("track_chi2_max")) {}

//
// member functions
//

// ------------ method called to produce the data  ------------

void PixelVertexCollectionTrimmerCUDA::produce(edm::StreamID streamID,
                                               edm::Event& iEvent,
                                               const edm::EventSetup& iSetup) const {
  if (m_OnGPU) {
    edm::Handle<ZTrackCUDAProduct> pTracks;
    iEvent.getByToken(tokenGPUTrack_, pTracks);
    edm::Handle<ZVertexCUDAProduct> pVertices;
    iEvent.getByToken(tokenGPUVertex_, pVertices);

    cms::cuda::ScopedContextProduce ctx{*pTracks};
    auto const* tracks = ctx.get(*pTracks).get();
    auto const* vertices = ctx.get(*pVertices).get();

    assert(vertices);
    assert(tracks);
    std::cout << "\n Going for the Async Function !! " << std::endl;
    ctx.emplace(iEvent, tokenGPUTrimmedVertex_, m_gpuAlgo.makeAsync(ctx.stream(), tracks, vertices));
  } else {
    /*
       edm::Handle<PixelTrackHeterogeneous> pTracks;
       iEvent.getByToken(tokenCPUTrack_, pTracks);
       edm::Handle<ZVertexHeterogeneous> pVertices;
       iEvent.getByToken(tokenCPUVertex_, pVertices);
       */

    auto const* tracks = iEvent.get(tokenCPUTrack_).get();
    auto const* vertices = iEvent.get(tokenCPUVertex_).get();

    std::cout << "\n calling the CPU TRIMMER !!! \n";

    iEvent.emplace(tokenCPUTrimmedVertex_, m_gpuAlgo.make(tracks, vertices));
    std::cout << "\n done with CPU TRIMMER !!! \n";
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void PixelVertexCollectionTrimmerCUDA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;

  desc.add<bool>("onGPU", true);
  desc.add<edm::InputTag>("pixelTrackSrc", edm::InputTag("caHitNtupletCUDA"));
  desc.add<edm::InputTag>("pixelVertexSrc");
  desc.add<int>("minNumberOfHits", 4);
  desc.add<unsigned int>("maxVtx", 100)->setComment("max output collection size (number of accepted vertices)");
  desc.add<double>("fractionSumPt2", 0.3)->setComment("threshold on sumPt2 fraction of the leading vertex");
  desc.add<double>("minSumPt2", 0.)->setComment("min sumPt2");
  desc.add<double>("track_pT_min", 1.0)->setComment("min track_pt");
  desc.add<double>("track_pT_max", 10.0)->setComment("max track_pt");
  desc.add<double>("track_chi2_max", 99999.0)->setComment("max track_chi2");
  desc.add<double>("track_prob_min", 1.0)->setComment("min track_prob");

  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(PixelVertexCollectionTrimmerCUDA);
