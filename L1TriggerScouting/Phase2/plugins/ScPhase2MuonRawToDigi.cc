#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/L1ScoutingRawData/interface/SDSNumbering.h"
#include "DataFormats/L1ScoutingRawData/interface/SDSRawDataCollection.h"
#include "DataFormats/L1Scouting/interface/OrbitCollection.h"
#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"
#include "DataFormats/L1TMuonPhase2/interface/L1ScoutingMuon.h"
#include "L1TriggerScouting/Phase2/interface/l1puppiUnpack.h"

class ScPhase2MuonRawToDigi : public edm::stream::EDProducer<> {
public:
  explicit ScPhase2MuonRawToDigi(const edm::ParameterSet &);
  ~ScPhase2MuonRawToDigi() override;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  //void beginStream(edm::StreamID) override;
  void produce(edm::Event &, const edm::EventSetup &) override;
  //void endStream() override;

  template <typename T>
  std::unique_ptr<OrbitCollection<T>> unpackObj(const SDSRawDataCollection &feds, std::vector<std::vector<T>> &buffer);

  //std::unique_ptr<l1Scouting::MuonSOA> unpackSOA(const SDSRawDataCollection &feds);

  edm::EDGetTokenT<SDSRawDataCollection> rawToken_;
  std::vector<unsigned int> fedIDs_;
  bool doCandidate_, doStruct_, doSOA_;

  // temporary storage
  std::vector<std::vector<l1t::PFCandidate>> candBuffer_;
  std::vector<std::vector<l1Scouting::Muon>> structBuffer_;

  //void unpackFromRaw(uint64_t data, std::vector<l1t::PFCandidate> &outBuffer);
  void unpackFromRaw(uint64_t wlo, uint32_t whi  , std::vector<l1Scouting::Muon> &outBuffer);
};

ScPhase2MuonRawToDigi::ScPhase2MuonRawToDigi(const edm::ParameterSet &iConfig)
    : rawToken_(consumes<SDSRawDataCollection>(iConfig.getParameter<edm::InputTag>("src"))),
      fedIDs_(iConfig.getParameter<std::vector<unsigned int>>("fedIDs")),
      doCandidate_(iConfig.getParameter<bool>("runCandidateUnpacker")),
      doStruct_(iConfig.getParameter<bool>("runStructUnpacker")),
      doSOA_(iConfig.getParameter<bool>("runSOAUnpacker")) {
  if (doCandidate_) {
    produces<OrbitCollection<l1t::PFCandidate>>();
    candBuffer_.resize(OrbitCollection<l1t::PFCandidate>::NBX + 1);  // FIXME magic number
  }
  if (doStruct_) {
    structBuffer_.resize(OrbitCollection<l1Scouting::Muon>::NBX + 1);  // FIXME magic number
    produces<OrbitCollection<l1Scouting::Muon>>();
  }
  if (doSOA_) {
   // produces<l1Scouting::MuonSOA>();
  }
}

ScPhase2MuonRawToDigi::~ScPhase2MuonRawToDigi(){};

void ScPhase2MuonRawToDigi::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  edm::Handle<SDSRawDataCollection> scoutingRawDataCollection;
  iEvent.getByToken(rawToken_, scoutingRawDataCollection);

  if (doCandidate_) {
    //iEvent.put(unpackObj(*scoutingRawDataCollection, candBuffer_));
  }
  if (doStruct_) {
    iEvent.put(unpackObj(*scoutingRawDataCollection, structBuffer_));
  }
  if (doSOA_) {
    //iEvent.put(unpackSOA(*scoutingRawDataCollection));
  }
}

template <typename T>
std::unique_ptr<OrbitCollection<T>> ScPhase2MuonRawToDigi::unpackObj(const SDSRawDataCollection &feds,
                                                                      std::vector<std::vector<T>> &buffer) {
  unsigned int ntot = 0;
  for (auto &fedId : fedIDs_) {
    const FEDRawData &src = feds.FEDData(fedId);
    std::cout<<"  Doing for FED : "<<fedId<<"  of size : "<<src.size()<<"\n";
    /*
    const uint64_t *begin = reinterpret_cast<const uint64_t *>(src.data());
    const uint64_t *end = reinterpret_cast<const uint64_t *>(src.data() + src.size());
    for (auto p = begin; p != end;) {
      if ((*p) == 0)
        continue;
      unsigned int bx = ((*p) >> 12) & 0xFFF;
      unsigned int nwords = (*p) & 0xFFF;
      std::cout<<" bx = "<<bx<<" | "<<"nwords : "<<nwords<<"\n";
      ++p;
    */
    const uint32_t *begin = reinterpret_cast<const uint32_t *>(src.data());
    const uint32_t *end = reinterpret_cast<const uint32_t *>(src.data() + src.size());
    for ( auto p = begin; p != end; p+=2) {
      const uint64_t *pH = reinterpret_cast<const uint64_t *>(p) ;
      if ((*pH) == 0)
        continue;
      unsigned int bx = ((*pH) >> 12) & 0xFFF;
      unsigned int nwords = (*pH) & 0xFFF;
      unsigned int nMuons = 2*nwords/3;
      std::cout<<"    > for NMuons "<<nMuons<<"  from "<<nwords<<" words |  bx = "<<bx<<"\n";
      std::cout<<"\n\n";
      ++p;++p;
      assert(bx < OrbitCollection<T>::NBX);
      std::vector<T> &outputBuffer = buffer[bx + 1];
      outputBuffer.reserve(nwords);
      
      uint64_t wlo;
      uint32_t whi;

      for (unsigned int i = 0; i < nMuons; ++i,p += 3) {
          if( i & 1  )
          {
                wlo = *reinterpret_cast<const uint64_t *>(p+1) ;
                whi = *p;
          }
          else
          {
                wlo = *reinterpret_cast<const uint64_t *>(p) ;
                whi = *(p+2);

          }
        if( (wlo==0) and (whi==0)) continue;
        unpackFromRaw(wlo,whi, outputBuffer);
        //std::cout<<"         wlo and whi  : "<<wlo<<"  , "<<whi<<"\n";
        ntot++;
      }
      
      if((nMuons%2)==1) ++p;
    }
  }
  return std::make_unique<OrbitCollection<T>>(buffer, ntot);
}
/*
void ScPhase2MuonRawToDigi::unpackFromRaw(uint64_t wlo, uint32_t whi, std::vector<l1t::PFCandidate> &outBuffer) {
  float pt, eta, phi, mass, z0 = 0, dxy = 0, puppiw = 1;
  uint16_t hwPt, hwMuonW = 1 << 8;
  int16_t pdgId, hwEta, hwPhi, hwZ0 = 0;
  int8_t hwDxy = 0;
  uint8_t pid, hwQuality;
  l1t::PFCandidate::ParticleType type;
  int charge;
  l1puppiUnpack::readshared(data, pt, eta, phi);
  l1puppiUnpack::readshared(data, hwPt, hwEta, hwPhi);
  pid = (data >> 37) & 0x7;
  l1puppiUnpack::assignpdgid(pid, pdgId);
  l1puppiUnpack::assignCMSSWPFCandidateId(pid, type);
  l1puppiUnpack::assignmass(pid, mass);
  l1puppiUnpack::assigncharge(pid, charge);
  reco::Particle::PolarLorentzVector p4(pt, eta, phi, mass);
  if (pid > 1) {
    l1puppiUnpack::readcharged(data, z0, dxy, hwQuality);
    l1puppiUnpack::readcharged(data, hwZ0, hwDxy, hwQuality);
  } else {
    l1puppiUnpack::readneutral(data, puppiw, hwQuality);
    l1puppiUnpack::readneutral(data, hwMuonW, hwQuality);
  }
  outBuffer.emplace_back(type, charge, p4, puppiw, hwPt, hwEta, hwPhi);
  if (pid > 1) {
    outBuffer.back().setZ0(z0);
    outBuffer.back().setDxy(dxy);
    outBuffer.back().setHwZ0(hwZ0);
    outBuffer.back().setHwDxy(hwDxy);
    outBuffer.back().setHwTkQuality(hwQuality);
  } else {
    outBuffer.back().setHwMuonWeight(hwMuonW);
    outBuffer.back().setHwEmID(hwQuality);
  }
  outBuffer.back().setEncodedMuon64(data);
}
*/

void ScPhase2MuonRawToDigi::unpackFromRaw(uint64_t wlo, uint32_t whi ,std::vector<l1Scouting::Muon> &outBuffer) {
  float pt, eta, phi, z0 = 0, d0 = 0,beta;
  int8_t charge;
  uint8_t quality,isolation;
  
  pt        = l1puppiUnpack::extractBitsFromW<1, 16>(wlo) * 0.03125f;
  phi       = l1puppiUnpack::extractSignedBitsFromW<17, 13>(wlo) * float(M_PI / (1 << 12));
  eta       = l1puppiUnpack::extractSignedBitsFromW<30, 14>(wlo) * float(M_PI / (1 << 12));
  z0        = l1puppiUnpack::extractSignedBitsFromW<44, 10>(wlo) * 0.05f;
  d0        = l1puppiUnpack::extractSignedBitsFromW<54, 10>(wlo) * 0.03f;
  quality   = l1puppiUnpack::extractBitsFromW<1, 8>(whi);
  isolation = l1puppiUnpack::extractBitsFromW<9, 4>(whi);
  beta      = l1puppiUnpack::extractBitsFromW<13, 4>(whi) * 0.06f;
  charge    = (whi & 1) ? -1 : +1;
  
  outBuffer.emplace_back(pt, eta, phi,  z0, d0,charge,quality,beta,isolation);
}
/*
std::unique_ptr<l1Scouting::MuonSOA> ScPhase2MuonRawToDigi::unpackSOA(const SDSRawDataCollection &feds) {
  std::vector<std::pair<const uint64_t *, const uint64_t *>> buffers;
  unsigned int sizeguess = 0;
  for (auto &fedId : fedIDs_) {
    const FEDRawData &src = feds.FEDData(fedId);
    buffers.emplace_back(reinterpret_cast<const uint64_t *>(src.data()),
                         reinterpret_cast<const uint64_t *>(src.data() + src.size()));
    sizeguess += src.size();
  }
  l1Scouting::MuonSOA ret;
  ret.bx.reserve(3564);
  ret.offsets.reserve(3564 + 1);
  for (std::vector<float> *v : {&ret.pt, &ret.eta, &ret.phi, &ret.z0, &ret.dxy, &ret.puppiw}) {
    v->resize(sizeguess);
  }
  ret.pdgId.resize(sizeguess);
  ret.quality.resize(sizeguess);
  unsigned int i0 = 0;
  for (int ibuff = 0, nbuffs = buffers.size(), lbuff = nbuffs - 1; buffers[ibuff].first != buffers[ibuff].second;
       ibuff = (ibuff == lbuff ? 0 : ibuff + 1)) {
    auto &pa = buffers[ibuff];
    while (pa.first != pa.second && *pa.first == 0) {
      pa.first++;
    }
    if (pa.first == pa.second)
      continue;
    unsigned int bx = ((*pa.first) >> 12) & 0xFFF;
    unsigned int nwords = (*pa.first) & 0xFFF;
    pa.first++;
    ret.bx.push_back(bx);
    ret.offsets.push_back(i0);
    for (unsigned int i = 0; i < nwords; ++i, ++pa.first, ++i0) {
      uint64_t data = *pa.first;
      l1puppiUnpack::readshared(data, ret.pt[i0], ret.eta[i0], ret.phi[i0]);
      uint8_t pid = (data >> 37) & 0x7;
      l1puppiUnpack::assignpdgid(pid, ret.pdgId[i0]);
      if (pid > 1) {
        l1puppiUnpack::readcharged(data, ret.z0[i0], ret.dxy[i0], ret.quality[i0]);
        ret.puppiw[i0] = 1.0f;
      } else {
        l1puppiUnpack::readneutral(data, ret.puppiw[i0], ret.quality[i0]);
        ret.dxy[i0] = 0.0f;
        ret.z0[i0] = 0.0f;
      }
    }
  }
  ret.offsets.push_back(i0);
  for (std::vector<float> *v : {&ret.pt, &ret.eta, &ret.phi, &ret.z0, &ret.dxy, &ret.puppiw}) {
    v->resize(i0);
  }
  ret.pdgId.resize(i0);
  ret.quality.resize(i0);
  auto retptr = std::make_unique<l1Scouting::MuonSOA>(std::move(ret));
  return retptr;
}
*/

void ScPhase2MuonRawToDigi::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(ScPhase2MuonRawToDigi);
