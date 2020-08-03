#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include <vector>
#include <memory>

namespace {
  class TrackSelectorByRegion final : public edm::global::EDProducer<> {
  public:
    explicit TrackSelectorByRegion(const edm::ParameterSet& conf)
        : tracksToken_(consumes<reco::TrackCollection>(conf.getParameter<edm::InputTag>("tracks"))) {
      //      for (auto const & ir : conf.getParameter<edm::VParameterSet>("regions")) {    // can be modified here to add more regions
      edm::ParameterSet regionPSet = conf.getParameter<edm::ParameterSet>("RegionPSet");
      inputTrkRegionToken_ = consumes<edm::OwnVector<TrackingRegion>>(regionPSet.getParameter<edm::InputTag>("input"));
      etaTolerance_ = regionPSet.getParameter<double>("etaTolerance");
      phiTolerance_ = regionPSet.getParameter<double>("phiTolerance");
      dXYTolerance_ = regionPSet.getParameter<double>("dXYTolerance");
      dZTolerance_ = regionPSet.getParameter<double>("dZTolerance");
      edm::ParameterSet trackPSet = conf.getParameter<edm::ParameterSet>("TrackPSet");
      minPt_ = trackPSet.getParameter<double>("minPt");
      //      }

      produces<reco::TrackCollection>();
      produces<std::vector<bool>>();
      
      produces<std::vector<float>>("normDeltaEta").setBranchAlias("normDeltaEta");
      produces<std::vector<float>>("normDeltaPhi").setBranchAlias("normDeltaPhi");
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("tracks", edm::InputTag("hltPixelTracks"));
      edm::ParameterSetDescription tracks_par;
      tracks_par.add<double>("minPt", 0.);
      desc.add<edm::ParameterSetDescription>("TrackPSet", tracks_par);

      edm::ParameterSetDescription region_par;
      region_par.add<edm::InputTag>("input", edm::InputTag(""));
      region_par.add<double>("phiTolerance", 1.0);
      region_par.add<double>("etaTolerance", 1.0);
      region_par.add<double>("dXYTolerance", 1.0);
      region_par.add<double>("dZTolerance", 1.0);
      desc.add<edm::ParameterSetDescription>("RegionPSet", region_par);
      descriptions.add("trackSelectorByRegion", desc);
    }

  private:
    using MaskCollection = std::vector<bool>;
    
    void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const override {
    // products
    auto mask = std::make_unique<MaskCollection>();                  // mask w/ the same size of the input collection
    auto output_tracks = std::make_unique<reco::TrackCollection>();  // selected output collection
    auto delta_eta_trk = std::make_unique<std::vector<float>>();
    auto delta_phi_trk = std::make_unique<std::vector<float>>();
    size_t n_regions ;
    
    auto regionsHandle = iEvent.getHandle(inputTrkRegionToken_); 
   
    auto tracksHandle  = iEvent.getHandle(tracksToken_); 
    
    if (tracksHandle.isValid()) {
      const auto tracks = *tracksHandle;
      mask->assign(tracks.size(), false);
     if (regionsHandle.isValid()) {
        const auto& regions = *regionsHandle;

      n_regions= regions.size();
      
      delta_eta_trk->assign(tracks.size()*n_regions, 1e9);
      delta_phi_trk->assign(tracks.size()*n_regions, 1e9);

     int count=0;
        for (const auto& tmp : regions)
          if (const auto* etaPhiRegion = dynamic_cast<const TrackingRegion*>(&tmp)) 
          {
              count++;
              std::cout<<"rage count = "<<count<<"/"<<n_regions<<"\n";
              auto  amask=etaPhiRegion->trackSelection(tracks);   
              for(size_t it=0;it<amask.size();it++)
              {
                 mask->at(it) = mask->at(it) or amask.at(it);
                 std::cout<<count<<"/"<<n_regions<<" it = "<<mask->at(it) <<" ->"<< mask->at(it)<<" or "<<amask.at(it)<<std::endl;
              }
          }

      }
    
    assert(mask->size() == tracks.size());

    for(size_t it=0;it<mask->size();it++)
        if(mask->at(it))
        output_tracks->push_back(tracks[it]);
    }

    if( mask->size()>0)
      std::cout<<" Adding "<<output_tracks->size()<<" / "<<mask->size()<<" tracks (nregions = "<<n_regions<<" ) \n";
   
   iEvent.put(std::move(mask));
   iEvent.put(std::move(output_tracks));
      
   iEvent.put(std::move(delta_eta_trk),"normDeltaEta");
   iEvent.put(std::move(delta_phi_trk),"normDeltaPhi"); 
 }



    edm::EDGetTokenT<reco::TrackCollection> tracksToken_;
    edm::EDGetTokenT<edm::OwnVector<TrackingRegion>> inputTrkRegionToken_;
    float phiTolerance_;
    float etaTolerance_;
    float dXYTolerance_;
    float dZTolerance_;
    float minPt_;
  
  };

 

}  // namespace

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TrackSelectorByRegion);

