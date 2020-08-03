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
<<<<<<< HEAD
      auto delta_eta_trk = std::make_unique<std::vector<float>>();
      auto delta_phi_trk = std::make_unique<std::vector<float>>();
      size_t n_regions ;
     // edm::Handle<edm::OwnVector<TrackingRegion>> regionsHandle;
     // iEvent.getByToken(inputTrkRegionToken_, regionsHandle);
      auto regionsHandle = iEvent.getHandle(inputTrkRegionToken_); 
     
     // edm::Handle<reco::TrackCollection> tracksHandle;
     // iEvent.getByToken(tracksToken_, tracksHandle);
      auto tracksHandle  = iEvent.getHandle(tracksToken_); 
      
 //     std::cout << "[TrackSelectorByRegion::produce] tracksHandle.isValid ? " << (tracksHandle.isValid() ? "YEAP" : "NOPE") << std::endl;
=======
      size_t n_regions;
      // edm::Handle<edm::OwnVector<TrackingRegion>> regionsHandle;
      // iEvent.getByToken(inputTrkRegionToken_, regionsHandle);
      auto regionsHandle = iEvent.getHandle(inputTrkRegionToken_);

      // edm::Handle<reco::TrackCollection> tracksHandle;
      // iEvent.getByToken(tracksToken_, tracksHandle);
      auto tracksHandle = iEvent.getHandle(tracksToken_);

      //     std::cout << "[TrackSelectorByRegion::produce] tracksHandle.isValid ? " << (tracksHandle.isValid() ? "YEAP" : "NOPE") << std::endl;
>>>>>>> refs/remotes/my-cmssw/TrackSelectorByRegion_dev
      if (tracksHandle.isValid()) {
        const auto & tracks = *tracksHandle;
        mask->assign(tracks.size(), false);
<<<<<<< HEAD
       if (regionsHandle.isValid()) {
          const auto& regions = *regionsHandle;

          n_regions= regions.size();
        delta_eta_trk->assign(tracks.size()*n_regions, 1e9);
        delta_phi_trk->assign(tracks.size()*n_regions, 1e9);
 
          std::vector<float> etaMin;
          etaMin.reserve(n_regions);
          
          std::vector<float> etaMax;
          etaMax.reserve(n_regions);
         
=======
        if (regionsHandle.isValid()) {
          const auto& regions = *regionsHandle;

          n_regions = regions.size();

          std::vector<float> etaMin;
          etaMin.reserve(n_regions);

          std::vector<float> etaMax;
          etaMax.reserve(n_regions);

>>>>>>> refs/remotes/my-cmssw/TrackSelectorByRegion_dev
          std::vector<float> phi0;
          phi0.reserve(n_regions);

          std::vector<float> phi0margin;
          phi0margin.reserve(n_regions);

          std::vector<float> phiMin;
          phiMin.reserve(n_regions);
<<<<<<< HEAD
          
=======

>>>>>>> refs/remotes/my-cmssw/TrackSelectorByRegion_dev
          std::vector<float> phiMax;
          phiMax.reserve(n_regions);

          std::vector<math::XYZPoint> origin;
          origin.reserve(n_regions);
<<<<<<< HEAD
          
          std::vector<float> zBound;
          zBound.reserve(n_regions);
          
          std::vector<float> RBound;
          RBound.reserve(n_regions);
          
          std::vector<float> pTmin ;
=======

          std::vector<float> zBound;
          zBound.reserve(n_regions);

          std::vector<float> RBound;
          RBound.reserve(n_regions);

          std::vector<float> pTmin;
>>>>>>> refs/remotes/my-cmssw/TrackSelectorByRegion_dev
          pTmin.reserve(n_regions);

          bool flag = false;
          
	  for (const auto& tmp : regions)
            if (const auto* etaPhiRegion = dynamic_cast<const RectangularEtaPhiTrackingRegion*>(&tmp)) {
              pTmin.push_back(etaPhiRegion->ptMin());
<<<<<<< HEAD
           //   if(std::abs(float(etaPhiRegion->ptMin())-2.0)>0.1) 
               std::cout<<"pTmin from region  = "<<etaPhiRegion->ptMin()<<std::endl;
              //pTmin.push_back(0.0);
              const auto& etaRange = etaPhiRegion->etaRange();
              const auto& phiMargin = etaPhiRegion->phiMargin();
              auto etamin=etaRange.min();  
              auto etamax=etaRange.max();  
              
              etaMin.push_back(0.5*((1+etaTolerance_)*etamin+(1-etaTolerance_)*etamax));
              etaMax.push_back(0.5*((1-etaTolerance_)*etamin+(1+etaTolerance_)*etamax));
            if(0.9<(etamax-etamin))
            {
              std::cout<<"eta : "<<0.5*((1+etaTolerance_)*etamin+(1-etaTolerance_)*etamax)<<", ";
              std::cout<<0.5*((1-etaTolerance_)*etamin+(1+etaTolerance_)*etamax)<<" <-  ";
              std::cout<<" "<<etamin<<" , "<<etamax<<"\n";
            } 
              phi0.push_back(etaPhiRegion->phiDirection());
              phi0margin.push_back(phiMargin.right()*phiTolerance_);   // Is it perfectly okay ? is there need for left and right ?
	        
	      float phiTemp=etaPhiRegion->phiDirection()-phiMargin.left()*phiTolerance_;
	      phiMin.push_back(reco::reduceRange(phiTemp));
	      
	      phiTemp=etaPhiRegion->phiDirection()+phiMargin.right()*phiTolerance_;
	      phiMax.push_back(reco::reduceRange(phiTemp));
          std::cout<<phiMargin.left()<<" ,  "<<phiMargin.right()<<std::endl;

	      GlobalPoint gp = etaPhiRegion->origin();
              origin.push_back(math::XYZPoint(gp.x(), gp.y(), gp.z()));
              zBound.push_back(etaPhiRegion->originZBound()*dZTolerance_);
            RBound.push_back(etaPhiRegion->originRBound()*dXYTolerance_);
            
           }

          size_t it = 0;
    //      if( tracks.size()>0)
    //        std::cout << "tracks: " << tracks.size() << std::endl;
          bool flag=true;
=======
              //   if(std::abs(float(etaPhiRegion->ptMin())-2.0)>0.1)
             if(flag) std::cout << "pTmin from region  = " << etaPhiRegion->ptMin() << std::endl;
              //pTmin.push_back(0.0);
              const auto& etaRange = etaPhiRegion->etaRange();
              const auto& phiMargin = etaPhiRegion->phiMargin();
              auto etamin = etaRange.min();
              auto etamax = etaRange.max();

              etaMin.push_back(0.5 * ((1 + etaTolerance_) * etamin + (1 - etaTolerance_) * etamax));
              etaMax.push_back(0.5 * ((1 - etaTolerance_) * etamin + (1 + etaTolerance_) * etamax));

              phi0.push_back(etaPhiRegion->phiDirection());
              phi0margin.push_back(phiMargin.right() *
                                   phiTolerance_);  // Is it perfectly okay ? is there need for left and right ?

              float phiTemp = etaPhiRegion->phiDirection() - phiMargin.left() * phiTolerance_;
             // if (phiTemp < -M_PI) {
             //   phiTemp += 2 * M_PI;
             // }
              phiMin.push_back(reco::reduceRange(phiTemp));

              phiTemp = etaPhiRegion->phiDirection() + phiMargin.right() * phiTolerance_;
             // if (phiTemp > M_PI) {
             //   phiTemp -= 2 * M_PI;
             // }

              phiMax.push_back(reco::reduceRange(phiTemp));

              GlobalPoint gp = etaPhiRegion->origin();
              origin.push_back(math::XYZPoint(gp.x(), gp.y(), gp.z()));
              zBound.push_back(etaPhiRegion->originZBound() * dZTolerance_);
              RBound.push_back(etaPhiRegion->originRBound() * dXYTolerance_);
            }
	else{
	    edm::LogWarning<<"region of unknown type passed to TrackSelectorByRegion";
	}


          size_t it = 0;
>>>>>>> refs/remotes/my-cmssw/TrackSelectorByRegion_dev
          for (auto const& trk : tracks) {
            const auto pt = trk.pt();
            const auto eta = trk.eta();
            const auto phi = trk.phi();

<<<<<<< HEAD
//            if ( pt<26 && pt>23) flag = true;
//            else flag =false;
            
            if(flag) std::cout<<" pT = "<<pt<<std::endl;
            for (size_t k = 0; k < n_regions; k++) {
            
           //          CALIBERATION CODE
            delta_eta_trk->at(it*n_regions+k)=(eta-0.5*(etaMax[k]+etaMin[k]));
            delta_phi_trk->at(it*n_regions+k)=(reco::deltaPhi(phi, phi0[k]));
            if (pt < pTmin[k] ) {
              if(flag)	std::cout << " KO !!! for pt "<<pt<<" [<"<<pTmin[k]<<"]  k = "<<k<<"/"<<n_regions<< std::endl;
              continue;
            }
  //   		if ( std::abs(trk.vz() - origin[k].z()) > zBound[k] ) {
//            { 		std::cout <<" k ="<<k<<"/"<<n_regions<<" std::abs(trk.vz() - origin[k].z()): " << std::abs(trk.vz() - origin[k].z()) << " zBound: " << zBound[k] << std::endl;
//              		std::cout << "std::abs(trk.dz(origin[k])): " << std::abs(trk.dz(origin[k])) << " zBound: " << zBound[k] << std::endl;
//              		std::cout << "std::abs(trk.dxy(origin[k])): " << std::abs(trk.dxy(origin[k])) << " RBound: " << RBound[k] << std::endl;
//              		std::cout << " trk   :  " <<trk.vx() <<","<<trk.vy()<<","<<trk.vz()<< std::endl;
//              		std::cout << "origin :  " <<origin[k].x() <<","<<origin[k].y()<<","<<origin[k].z()<< std::endl;
//		}
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
//              if (std::abs(reco::deltaPhi(phi, phi0[k])) > phi0margin[k] ) {
//             //	         std::cout<<"k = "<<k<<"/"<<n_regions<< " phi : " << phi << " phi0[k]: " << phi0[k] << " std::abs(reco::deltaPhi(phi,phi0[k])): " << std::abs(reco::deltaPhi(phi,phi0[k])) << " phi0margin: " << phi0margin[k] << std::endl;
//             //   	std::cout << " KO !!! for phi" << std::endl;
//                continue;
//              }
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
          
        if (std::abs(reco::deltaPhi(phi, phi0[k])) > phi0margin[k] ) {
            if(flag)  std::cout << " KOKOKO !!! for phi " 
                      <<" k = "<<k<<"/"<<n_regions<< " phi : " << phi << " phi0[k]: " << phi0[k] << " std::abs(reco::deltaPhi(phi,phi0[k])): " << std::abs(reco::deltaPhi(phi,phi0[k])) << " phi0margin: " << phi0margin[k] << std::endl;
                continue;
              }
        if(std::abs(delta_eta_trk->at(it*n_regions+k))>0.8){
        std::cout<<"KO KO KO eta-0.5*(etaMax[k]+etaMin[k])) "<<eta-0.5*(etaMax[k]+etaMin[k])<<" eta : " << eta << " etaMin[k]: " << etaMin[k] << " etaMax[k]: " << etaMax[k]<<" delta  " <<delta_eta_trk->at(it);
        delta_eta_trk->at(it*n_regions+k)=eta-0.5*(etaMax[k]+etaMin[k]);
        std::cout<<" ->  "<<delta_eta_trk->at(it*n_regions+k)<<"\n";
            }
	      output_tracks->push_back(trk);
=======
            if (flag)
              std::cout << " pT = " << pt << std::endl;
            for (size_t k = 0; k < n_regions; k++) {
              if (pt < pTmin[k]) {
                if (flag)
                  std::cout << " KO !!! for pt " << pt << " [<" << pTmin[k] << "]  k = " << k << "/" << n_regions
                            << std::endl;
                continue;
              }

              if (std::abs(trk.dz(origin[k])) > zBound[k]) {
                if (flag)
                  std::cout << " KO !! for z !! "
                            << "  pT = " << pt << " k = " << k << "/" << n_regions
                            << " std::abs(trk.dz(origin[k])) > zBound[k] -> " << std::abs(trk.dz(origin[k])) << " > "
                            << zBound[k] << std::endl;
                continue;
              }
              if (std::abs(trk.dxy(origin[k])) > RBound[k]) {
                if (flag)
                  std::cout << " KO !! for dXY"
                            << "  pT = " << pt << " k = " << k << "/" << n_regions
                            << " std::abs(trk.dxy(origin[k])): " << std::abs(trk.dxy(origin[k]))
                            << " RBound: " << RBound[k] << std::endl;
                continue;
              }

              if (eta < etaMin[k]) {
                if (flag)
                  std::cout << " KO !!! for eta "
                            << "  pT = " << pt << " k = " << k << "/" << n_regions << " eta : " << eta
                            << " etaMin[k]: " << etaMin[k] << " etaMax[k]: " << etaMax[k] << "\n";
                continue;
              }
              if (eta > etaMax[k]) {
                if (flag)
                  std::cout << " KO !!! for eta"
                            << "  pT = " << pt << " k = " << k << "/" << n_regions << " eta : " << eta
                            << " etaMin[k]: " << etaMin[k] << " etaMax[k]: " << etaMax[k] << "\n";
                continue;
              }

              if (phiMin[k] < phiMax[k]) {
                if (phi < phiMin[k]) {
                  if (flag)
                    std::cout << " KO !!! for phi "
                              << "  pT = " << pt << " k = " << k << "/" << n_regions << " phi : " << phi
                              << " phiMin[k]: " << phiMin[k] << " phiMax[k]: " << phiMax[k] << "\n";
                  continue;
                }
                if (phi > phiMax[k]) {
                  if (flag)
                    std::cout << " KO !!! for phi "
                              << "  pT = " << pt << " k=" << k << "/" << n_regions << " phi : " << phi
                              << " phiMin[k]: " << phiMin[k] << " phiMax[k]: " << phiMax[k] << "\n";
                  continue;
                }
              } else {
                if (phi < phiMin[k] && phi > phiMax[k]) {
                  if (flag)
                    std::cout << " KO !!! for phi phitol etol = " << phiTolerance_ << " , " << etaTolerance_
                              << "  pT = " << pt << " k = " << k << "/" << n_regions << " phi : " << phi
                              << " phiMin[k]: " << phiMin[k] << " phiMax[k]: " << phiMax[k] << "\n";
                  continue;
                }
              }

              output_tracks->push_back(trk);
>>>>>>> refs/remotes/my-cmssw/TrackSelectorByRegion_dev
              mask->at(it) = true;
              break;
            }
            it++;
          }
        }
        assert(mask->size() == tracks.size());
      }
<<<<<<< HEAD
//     if( mask->size()>0)
        std::cout<<" Adding "<<output_tracks->size()<<" / "<<mask->size()<<" tracks (nregions = "<<n_regions<<" ) \n";
     
     iEvent.put(std::move(mask));
     iEvent.put(std::move(output_tracks));
        
     iEvent.put(std::move(delta_eta_trk),"normDeltaEta");
     iEvent.put(std::move(delta_phi_trk),"normDeltaPhi"); 
   }
=======

      iEvent.put(std::move(mask));
      iEvent.put(std::move(output_tracks));
    }
>>>>>>> refs/remotes/my-cmssw/TrackSelectorByRegion_dev

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
