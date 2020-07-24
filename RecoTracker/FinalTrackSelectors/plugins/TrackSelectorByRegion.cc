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
      
   //   produces<std::vector<float>>("normDeltaEta").setBranchAlias("normDeltaEta");
   //   produces<std::vector<float>>("normDeltaPhi").setBranchAlias("normDeltaPhi");
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
      size_t n_regions ;
     // edm::Handle<edm::OwnVector<TrackingRegion>> regionsHandle;
     // iEvent.getByToken(inputTrkRegionToken_, regionsHandle);
      auto regionsHandle = iEvent.getHandle(inputTrkRegionToken_); 
     
     // edm::Handle<reco::TrackCollection> tracksHandle;
     // iEvent.getByToken(tracksToken_, tracksHandle);
      auto tracksHandle  = iEvent.getHandle(tracksToken_); 
      
 //     std::cout << "[TrackSelectorByRegion::produce] tracksHandle.isValid ? " << (tracksHandle.isValid() ? "YEAP" : "NOPE") << std::endl;
      if (tracksHandle.isValid()) {
        const auto tracks = *tracksHandle;
        mask->assign(tracks.size(), false);
       if (regionsHandle.isValid()) {
          const auto& regions = *regionsHandle;

          n_regions= regions.size();
          
          std::vector<float> etaMin;
          etaMin.reserve(n_regions);
          
          std::vector<float> etaMax;
          etaMax.reserve(n_regions);
         
          std::vector<float> phi0;
          phi0.reserve(n_regions);
     
          std::vector<float> phi0margin;
          phi0margin.reserve(n_regions);
          
          std::vector<float> phiMin;
          phiMin.reserve(n_regions);
          
          std::vector<float> phiMax;
          phiMax.reserve(n_regions);

          std::vector<math::XYZPoint> origin;
          origin.reserve(n_regions);
          
          std::vector<float> zBound;
          zBound.reserve(n_regions);
          
          std::vector<float> RBound;
          RBound.reserve(n_regions);
          
          std::vector<float> pTmin ;
          pTmin.reserve(n_regions);

          for (const auto& tmp : regions)
            if (const auto* etaPhiRegion = dynamic_cast<const RectangularEtaPhiTrackingRegion*>(&tmp)) {
             
              pTmin.push_back(etaPhiRegion->ptMin());
           //   if(std::abs(float(etaPhiRegion->ptMin())-2.0)>0.1) 
               std::cout<<"pTmin from region  = "<<etaPhiRegion->ptMin()<<std::endl;
              //pTmin.push_back(0.0);
              const auto& etaRange = etaPhiRegion->etaRange();
              const auto& phiMargin = etaPhiRegion->phiMargin();
              auto etamin=etaRange.min();  
              auto etamax=etaRange.max();  
              
              etaMin.push_back(0.5*((1+etaTolerance_)*etamin+(1-etaTolerance_)*etamax));
              etaMax.push_back(0.5*((1-etaTolerance_)*etamin+(1+etaTolerance_)*etamax));
            
              phi0.push_back(etaPhiRegion->phiDirection());
              phi0margin.push_back(phiMargin.right()*phiTolerance_);   // Is it perfectly okay ? is there need for left and right ?
	        
	      float phiTemp=etaPhiRegion->phiDirection()-phiMargin.left()*phiTolerance_;
	      if (phiTemp<-M_PI) 
              {
                phiTemp+=2*M_PI;
              }
	      phiMin.push_back(phiTemp);
	      
	      phiTemp=etaPhiRegion->phiDirection()+phiMargin.right()*phiTolerance_;
	      if (phiTemp>M_PI)
             {
              phiTemp-=2*M_PI;
             }
          
	      phiMax.push_back(phiTemp);
              
	      GlobalPoint gp = etaPhiRegion->origin();
              origin.push_back(math::XYZPoint(gp.x(), gp.y(), gp.z()));
              zBound.push_back(etaPhiRegion->originZBound()*dZTolerance_);
              RBound.push_back(etaPhiRegion->originRBound()*dXYTolerance_);
            
           }

          size_t it = 0;
          bool flag=false;
          for (auto const& trk : tracks) {
            const auto pt = trk.pt();
            const auto eta = trk.eta();
            const auto phi = trk.phi();

            if(flag) std::cout<<" pT = "<<pt<<std::endl;
            for (size_t k = 0; k < n_regions; k++) {
            
            if (pt < pTmin[k] ) {
              if(flag)	std::cout << " KO !!! for pt "<<pt<<" [<"<<pTmin[k]<<"]  k = "<<k<<"/"<<n_regions<< std::endl;
              continue;
            }
             
            if (std::abs(trk.dz(origin[k])) > zBound[k] ) {
               if(flag)  std::cout << " KO !! for z !! "<<"  pT = "<<pt<<" k = "<<k<<"/"<<n_regions
                                    <<" std::abs(trk.dz(origin[k])) > zBound[k] -> "
                                    <<std::abs(trk.dz(origin[k]))<<" > "<<zBound[k]<< std::endl;
                continue;
              }
           if (std::abs(trk.dxy(origin[k])) > RBound[k]) {
                if(flag)  std::cout << " KO !! for dXY"
                          <<"  pT = "<<pt<<" k = "<<k<<"/"<<n_regions<<" std::abs(trk.dxy(origin[k])): " << std::abs(trk.dxy(origin[k])) << " RBound: " << RBound[k] << std::endl;
                continue;
              }

           
      if (eta < etaMin[k] ) {
                if(flag)  std::cout << " KO !!! for eta "
              		      <<"  pT = "<<pt<<" k = "<<k<<"/"<<n_regions<<" eta : " << eta << " etaMin[k]: " << etaMin[k] << " etaMax[k]: " << etaMax[k]<<"\n";
                continue;
              }
              if (eta > etaMax[k] ) {
               	if(flag)  std::cout << " KO !!! for eta" 
      		          <<"  pT = "<<pt<<" k = "<<k<<"/"<<n_regions<< " eta : " << eta << " etaMin[k]: " << etaMin[k] << " etaMax[k]: " << etaMax[k]<<"\n" ;
                continue;
              }

              if (phiMin[k] < phiMax[k] ){
		if ( phi < phiMin[k] ) {
            if(flag)     std::cout << " KO !!! for phi "
                       <<"  pT = "<<pt<<" k = "<<k<<"/"<<n_regions<<" phi : " << phi << " phiMin[k]: " << phiMin[k] << " phiMax[k]: " << phiMax[k]<<"\n";
                continue;
              	}
              if ( phi > phiMax[k]) {
             	if(flag)  std::cout << " KO !!! for phi " 
      		             <<"  pT = "<<pt<<" k="<<k<<"/"<<n_regions<< " phi : " << phi << " phiMin[k]: " << phiMin[k] << " phiMax[k]: " << phiMax[k]<<"\n" ;
                continue;
              }
	      }
	    else  {
		if ( phi < phiMin[k] && phi > phiMax[k] ) {
              if(flag)  std::cout << " KO !!! for phi phitol etol = "<<phiTolerance_<<" , "<<etaTolerance_
                 <<"  pT = "<<pt<<" k = "<<k<<"/"<<n_regions<<" phi : " << phi << " phiMin[k]: " << phiMin[k] << " phiMax[k]: " << phiMax[k]<<"\n";
                continue;
              	}
	    }
          
	      output_tracks->push_back(trk);
              mask->at(it) = true;
              break;
            }
            it++;
          }
        }
        assert(mask->size() == tracks.size());
      }
     
     iEvent.put(std::move(mask));
     iEvent.put(std::move(output_tracks));
        
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

