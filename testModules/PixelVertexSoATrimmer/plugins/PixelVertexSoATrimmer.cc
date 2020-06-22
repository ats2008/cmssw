// -*- C++ -*-
//
// Package: RecoPixelVertexing/PixelVertexFinding
// Class: PixelVertexSoATrimmer
//
/**\class PixelVertexSoATrimmer PixelVertexSoATrimmer.cc RecoPixelVertexing/PixelVertexFinding/plugins/PixelVertexSoATrimmer.cc

Description: [one line class summary]

Implementation:
[Notes on implementation]
*/
//
// Original Author: Riccardo Manzoni
// Created: Tue, 01 Apr 2014 10:11:16 GMT
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

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "RecoPixelVertexing/PixelVertexFinding/interface/PVClusterComparer.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "CUDADataFormats/Track/interface/PixelTrackHeterogeneous.h"
#include "CUDADataFormats/Vertex/interface/ZVertexHeterogeneous.h"


#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/FitUtils.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "TMath.h"
void updateChisquareQuantile(size_t ndof,std::vector<double> &maxChi2,double track_prob_min) ;

class PixelVertexSoATrimmer : public edm::stream::EDProducer<> {
public:
  explicit PixelVertexSoATrimmer(const edm::ParameterSet&);
  ~PixelVertexSoATrimmer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  //edm::EDGetTokenT<reco::VertexCollection> vtxToken_;
  edm::EDGetTokenT<ZVertexHeterogeneous> tokenVertex_;
  edm::EDGetTokenT<reco::BeamSpot> tokenBeamSpot_;
  edm::EDGetTokenT<PixelTrackHeterogeneous> tokenTrack_;
  edm::EDPutTokenT<ZVertexHeterogeneous> tokenSOA_;

  unsigned int maxVtx_;
  int minNumberOfHits_;
  double fractionSumPt2_;
  double minSumPt2_;
  double track_pT_min,track_pT_max,track_chi2_max,track_prob_min;
  std::unique_ptr<ZVertexSoA> trimmed_VtxSoA_ptr;
//   cms::cuda::host::unique_ptr<ZVertexSoA> trimmed_VtxSoA_ptr; 
// PVClusterComparer* pvComparer_;
};

PixelVertexSoATrimmer::PixelVertexSoATrimmer(const edm::ParameterSet& iConfig) {
  tokenVertex_  =  consumes<ZVertexHeterogeneous>(iConfig.getParameter<edm::InputTag>("src"));
  tokenTrack_   =  consumes<PixelTrackHeterogeneous>(iConfig.getParameter<edm::InputTag>("trackSrc")),
  tokenBeamSpot_=  consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpot")),
  maxVtx_ = iConfig.getParameter<unsigned int>("maxVtx");
  fractionSumPt2_ = iConfig.getParameter<double>("fractionSumPt2");
  minSumPt2_ = iConfig.getParameter<double>("minSumPt2");
  minNumberOfHits_=iConfig.getParameter<int>("minNumberOfHits");  

  track_pT_min = iConfig.getParameter<double>("track_pt_min");
  track_pT_max = iConfig.getParameter<double>("track_pt_max");
  track_chi2_max = iConfig.getParameter<double>("track_chi2_max");
  track_prob_min = iConfig.getParameter<double>("track_prob_min");

//  pvComparer_ = new PVClusterComparer(track_pt_min, track_pt_max, track_chi2_max, track_prob_min);

  tokenSOA_=produces<ZVertexHeterogeneous>();
  //produces<ZVertexHeterogeneous>();
}
PixelVertexSoATrimmer::~PixelVertexSoATrimmer() {}

// ------------ method called to produce the data ------------
void PixelVertexSoATrimmer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  trimmed_VtxSoA_ptr= std::make_unique<ZVertexSoA>();
// reading SoA s and Beamspot
  auto const &vertex_soa = *(iEvent.get(tokenVertex_).get());
  int nv=vertex_soa.nvFinal;
  math::XYZPoint vtx_position;

  edm::Handle<reco::BeamSpot> bsHandle;
  iEvent.getByToken(tokenBeamSpot_, bsHandle);
  const reco::BeamSpot &bsh = *bsHandle;
  GlobalPoint bs(bsh.x0(),bsh.y0(),bsh.z0());
  
  edm::ESHandle<MagneticField> fieldESH;
  iSetup.get<IdealMagneticFieldRecord>().get(fieldESH);

  const auto &tsoa = *iEvent.get(tokenTrack_);
  auto const *quality = tsoa.qualityData();
  auto const &fit = tsoa.stateAtBS;
//  auto const &hitIndices = tsoa.hitIndices;
   unsigned int maxTracks(tsoa.stride());


  //  edm::Handle<reco::VertexCollection> vtxs;
//  iEvent.getByToken(vtxToken_, vtxs);

  double sumpt2;
  //double sumpt2previous = -99. ;

  // this is not the logic we want, at least for now
  // if requires the sumpt2 for vtx_n to be > threshold * sumpt2 vtx_n-1
  // for (reco::VertexCollection::const_iterator vtx = vtxs->begin(); vtx != vtxs->end(); ++vtx, ++counter){
  // if (counter > maxVtx_) break ;
  // sumpt2 = PVCluster.pTSquaredSum(*vtx) ;
  // if (sumpt2 > sumpt2previous*fractionSumPt2_ && sumpt2 > minSumPt2_ ) vtxs_trim->push_back(*vtx) ;
  // else if (counter == 0 ) vtxs_trim->push_back(*vtx) ;
  // sumpt2previous = sumpt2 ;
  // }


  //ZVertexSoA trimmed_VtxSoA_ptr->*trimmed_VtxSoA_ptr;
   trimmed_VtxSoA_ptr->nvFinal=0;
  int vtx_id;
  
  double sumpt2first =0;
  std::vector<double> track_pT2;
  std::vector<double> maxChi2_;
  //std::cout<<" MAX TRACKS = " <<maxTracks<<"\n";
   unsigned int it=0;
   auto track_pT_max2=track_pT_max*track_pT_max;
   auto track_pT_min2=track_pT_min*track_pT_min;
 
   for(it=0;it<maxTracks;it++)
    {
        trimmed_VtxSoA_ptr->idv[it]=-1;
        track_pT2.push_back(-1.0);
        auto nHits = tsoa.nHits(it);
	//std::cout<<"\nDoing for track : "<<it<<" ";
        if (nHits == 0)
          break;  // this is a guard: maybe we need to move to nTracks...
        auto q = quality[it];
        if (q != trackQuality::loose)
          continue;  // FIXME
        if (nHits < minNumberOfHits_)
          continue;
        
        // mind: this values are respect the beamspot!
        
        size_t ndof=2*nHits-5;
        float chi2 = tsoa.chi2(it);
        float phi = tsoa.phi(it);
        
        if (track_prob_min >= 0. && track_prob_min <= 1.)
	{
          if(maxChi2_.size()<=(ndof)) 
            updateChisquareQuantile(ndof,maxChi2_,track_prob_min);
          if (chi2*ndof> maxChi2_[ndof]) continue;
        }
        if (chi2 >track_chi2_max ) continue;       

        Rfit::Vector5d ipar, opar;
        Rfit::Matrix5d icov, ocov;
        fit.copyToDense(ipar, icov, it);
        Rfit::transformToPerigeePlane(ipar, icov, opar, ocov);
        LocalTrajectoryParameters lpar(opar(0), opar(1), opar(2), opar(3), opar(4), 1.);

        float sp = std::sin(phi);
        float cp = std::cos(phi);
        Surface::RotationType rot(sp, -cp, 0, 0, 0, -1.f, cp, sp, 0);

        Plane impPointPlane(bs, rot);
        GlobalTrajectoryParameters gp(impPointPlane.toGlobal(lpar.position()),
                                      impPointPlane.toGlobal(lpar.momentum()),
                                      lpar.charge(),
                                      fieldESH.product());
        GlobalVector pp = gp.momentum();
        math::XYZVector mom(pp.x(), pp.y(), pp.z());
        auto pt2=mom.Perp2();
        if(pt2 < track_pT_min2 ) continue;
        if(pt2 > track_pT_max2 ) pt2=track_pT_max2;
        track_pT2.back()=pt2;
    	//std::cout<<" pT = "<<pt*pt
	}
     maxTracks=it;
    //finding sumpt2first
    
   // std::cout<<"\nSumpTFirst";
   auto nt=0;
    for(int j=nv-1;j>=0;j--)
    {
    vtx_id=vertex_soa.sortInd[j];
    sumpt2first=0;
    for(unsigned int k=0;k<maxTracks;k++)
    {
        if(vertex_soa.idv[k]!=vtx_id) continue;
  	nt++;
        if(track_pT2[k]<0) continue;
        auto pt2=track_pT2[k];
        sumpt2first+=pt2;
    }
    if(nt > 1) break;
   }
  //std::cout<<"sumptFirst"<<sumpt2first<<",";
  //std::cout<<"\nsumpt2first = "<<sumpt2first<<" , minSumPt2_ =  "<<minSumPt2_<<"  ffractionSumPt2_ = "<<fractionSumPt2_;
  trimmed_VtxSoA_ptr->nvFinal=0;
  for (int j=0; j<nv; j++)
  {
    //std::cout<<"\ntrimmed_VtxSoA_ptr->nvFinal = "<<trimmed_VtxSoA_ptr->nvFinal<<" ";
    if(trimmed_VtxSoA_ptr->nvFinal>=maxVtx_) break;
    vtx_id=vertex_soa.sortInd[j];
    sumpt2=0;
    nt=0;
    for(unsigned int k=0;k<maxTracks;k++)
    {
	
        if(vertex_soa.idv[k]!=vtx_id) continue;
	nt++;
        if(track_pT2[k]<0) continue;
        sumpt2+=track_pT2[k];
    }
    if(nt < 2) continue; 
//    std::cout<<" , sumpt2 = "<<sumpt2<<" ";
    //sumpt2 = pvComparer_->pTSquaredSum(*vtx);
     if(sumpt2 >= sumpt2first * fractionSumPt2_ && sumpt2 > minSumPt2_)
     {
        auto newVtxId=trimmed_VtxSoA_ptr->nvFinal;
        for(unsigned int k=0;k<maxTracks;k++)
        {
            if(vertex_soa.idv[k]==vtx_id) 
            trimmed_VtxSoA_ptr->idv[k]=newVtxId;
        }

        trimmed_VtxSoA_ptr->zv[newVtxId]=vertex_soa.zv[vtx_id];
        trimmed_VtxSoA_ptr->wv[newVtxId]=vertex_soa.wv[vtx_id];
        trimmed_VtxSoA_ptr->chi2[newVtxId]=vertex_soa.chi2[vtx_id];
        trimmed_VtxSoA_ptr->sortInd[newVtxId]=newVtxId;
        trimmed_VtxSoA_ptr->ndof[newVtxId]=vertex_soa.ndof[vtx_id];
        trimmed_VtxSoA_ptr->ptv2[newVtxId]=vertex_soa.ptv2[vtx_id];
//       std::cout<<trimmed_VtxSoA_ptr->zv[newVtxId]<<"<- zv ,";
//       std::cout<<trimmed_VtxSoA_ptr->wv[newVtxId]<<"<- wv ,";
//       std::cout<<trimmed_VtxSoA_ptr->chi2[newVtxId]<<" <- chi2 ,";
//       std::cout<<trimmed_VtxSoA_ptr->ptv2[newVtxId]<<"<- ptv2 ";
//       std::cout<<trimmed_VtxSoA_ptr->sortInd[newVtxId]<<"<- sortInd";
	trimmed_VtxSoA_ptr->nvFinal++;
        std::cout <<"SoA sumpt2: " << sumpt2<<","<<nt << "[" << sumpt2first << "]" << std::endl;
      }
  }
 std::cout<<"\n";
  //  std::cout<<"\nTOTAL VERITES IN TRIMMED SoA = "<<trimmed_VtxSoA_ptr->nvFinal<<"\n";
  //  std::cout << " ==> # vertices: " << vtxs_trim->size() << std::endl;
  // iEvent.put(std::move(trimmed_VtxSoA_ptr->);
 iEvent.emplace(tokenSOA_, ZVertexHeterogeneous(std::move(trimmed_VtxSoA_ptr)));
 //  iEvent.put(ZVertexHeterogeneous(std::move(&trimmed_VtxSoA_ptr->),"trimmedVtx");

}

// ------------ method fills 'descriptions' with the allowed parameters for the module ------------
void PixelVertexSoATrimmer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("beamSpot", edm::InputTag(""))->setComment("beamspot in global frame");
  desc.add<edm::InputTag>("trackSrc", edm::InputTag(""))->setComment("input Tracks in SoA");
  desc.add<edm::InputTag>("src", edm::InputTag(""))->setComment("input (pixel) vertex collection in SoA");
  desc.add<unsigned int>("maxVtx", 100)->setComment("max output collection size (number of accepted vertices)");
  desc.add<int>("minNumberOfHits", 0)->setComment("min number of hits in track canidates");
  desc.add<double>("fractionSumPt2", 0.3)->setComment("threshold on sumPt2 fraction of the leading vertex");
  desc.add<double>("minSumPt2", 0.)->setComment("min sumPt2");
  desc.add<double>("track_pt_min", 1.0)->setComment("min track_pt");
  desc.add<double>("track_pt_max", 10.0)->setComment("max track_pt");
  desc.add<double>("track_chi2_max", 99999.0)->setComment("max track_chi2");
  desc.add<double>("track_prob_min", -1.0)->setComment("min track_prob");

  descriptions.add("hltPixelVertexSoATrimmer", desc);
}

void updateChisquareQuantile(size_t ndof,std::vector<double> &maxChi2,double track_prob_min) 
{
  size_t oldsize = maxChi2.size();
  for (size_t i = oldsize; i <= ndof; ++i) 
  {
    double chi2 = TMath::ChisquareQuantile(1 - track_prob_min, i);
    maxChi2.push_back(chi2);
  }
}


//define this as a plug-in
DEFINE_FWK_MODULE(PixelVertexSoATrimmer);
