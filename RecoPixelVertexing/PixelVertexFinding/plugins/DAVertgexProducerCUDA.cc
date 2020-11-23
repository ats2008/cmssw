// -*- C++ -*-
//
// Package:    RecoVertex/DAVertgexProducerCUDA
// Class:      DAVertgexProducerCUDA
//
/**\class DAVertgexProducerCUDA DAVertgexProducerCUDA.cc RecoVertex/DAVertgexProducerCUDA/plugins/DAVertgexProducerCUDA.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Aravind Thachayath Sugunan
//         Created:  Mon, 16 Nov 2020 18:42:35 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"

#include "gpuDAVertexer.h"

//
// class declaration
//

class DAVertgexProducerCUDA : public edm::stream::EDProducer<> {
public:
    using ZTrackCUDAProduct =cms::cuda::Product<PixelTrackHeterogeneous>;
    explicit DAVertgexProducerCUDA(const edm::ParameterSet&);
  ~DAVertgexProducerCUDA() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------
  const edm::EDGetTokenT<ZTrackCUDAProduct> tockenGPUTrack_ ;
  const edm::EDPutTokenT<ZVertexCUDAProduct> tockenGPUdaVertex_ ;
  
  // why not  const defn of gpuDAVertexe::DAVertexer is not working ?
  gpuDAVertexer::DAVertexer m_daVertexer;
};

DAVertgexProducerCUDA::DAVertgexProducerCUDA(const edm::ParameterSet& iConfig) : tockenGPUTrack_(consumes<ZTrackCUDAProduct>(iConfig.getParameter<edm::InputTag>("trackSource"))),
    tockenGPUdaVertex_(produces<ZVertexCUDAProduct>()),
    m_daVertexer(iConfig.getParameter<double>("splitSeparation"))
{

}

// ------------ method called to produce the data  ------------
void DAVertgexProducerCUDA::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

    edm::Handle<ZTrackCUDAProduct> tracksHandle;
    iEvent.getByToken(tockenGPUTrack_,tracksHandle);
    
    cms::cuda::ScopedContextProduce ctx{*tracksHandle};
    auto const* tracks = ctx.get(*tracksHandle).get();

    assert(tracks);
    std::cout<<" Going to the GPU DA !! with splitSeparation =  "<<m_daVertexer.get_splitSeparation()<<std::endl; 
    
    ctx.emplace(iEvent, tockenGPUdaVertex_,m_daVertexer.makeAsync(ctx.stream(),tracks));

}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void DAVertgexProducerCUDA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("trackSource");
  desc.add<double>("splitSeparation",0.25)->setComment("split vertices if they track z r apart by ");

  descriptions.addDefault(desc);
}



//DAVertgexProducerCUDA::~DAVertgexProducerCUDA() {
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
  //
  // please remove this method altogether if it would be left empty
//}

//
// member functions
//


// ------------ method called once each stream before processing any runs, lumis or events  ------------
//void DAVertgexProducerCUDA::beginStream(edm::StreamID) {
  // please remove this method if not needed
//}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
//void DAVertgexProducerCUDA::endStream() {
  // please remove this method if not needed
//}

// ------------ method called when starting to processes a run  ------------
/*
void
DAVertgexProducerCUDA::beginRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a run  ------------
/*
void
DAVertgexProducerCUDA::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
void
DAVertgexProducerCUDA::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
DAVertgexProducerCUDA::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/



//define this as a plug-in
DEFINE_FWK_MODULE(DAVertgexProducerCUDA);
