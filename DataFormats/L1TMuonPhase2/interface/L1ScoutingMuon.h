#ifndef DataFormats_L1TParticleFlow_L1ScoutingMuon_h
#define DataFormats_L1TParticleFlow_L1ScoutingMuon_h

#include <vector>
#include <utility>
#include <cstdint>
#include <Math/Vector4D.h>

namespace l1Scouting {
  class Muon {
  public:
    Muon() {}
    Muon(float pt, float eta, float phi, float z0, float d0)
        : pt_(pt), eta_(eta), phi_(phi), z0_(z0), d0_(d0) {}
    Muon(float pt,float eta,float phi, float z0,float d0,int8_t charge,uint8_t quality,float beta,uint8_t isolation)
        : pt_(pt), eta_(eta), phi_(phi), z0_(z0), d0_(d0) , charge_(charge) , quality_(quality), beta_(beta),isolation_(isolation) {}
        

    float pt() const { return pt_; }
    float eta() const { return eta_; }
    float phi() const { return phi_; }
    float z0() const { return z0_; }
    float d0() const { return d0_; }
    float beta() const { return beta_; }
     int8_t charge() const { return charge_; }
    uint8_t quality() const { return quality_; }
    uint8_t isolation() const { return isolation_; }

    void setPt(float pt) { pt_ = pt; }
    void setEta(float eta) { eta_ = eta; }
    void setPhi(float phi) { phi_ = phi; }
    void setZ0(float z0) { z0_ = z0; }
    void setD0(float d0) { d0_ = d0; }
    void setQuality(uint8_t quality) { quality_ = quality; }
    float mass() const { return 0.105 ; }

    ROOT::Math::PtEtaPhiMVector p4() const { return ROOT::Math::PtEtaPhiMVector(pt_, eta_, phi_, mass()); }

  private:
    float pt_, eta_, phi_, z0_, d0_;
    int8_t charge_;
    uint8_t  quality_;
    float beta_;
    uint8_t isolation_;

  };

/*
  struct MuonSOA {
    std::vector<uint16_t> bx;
    std::vector<uint32_t> offsets;
    std::vector<float> pt, eta, phi, z0, dxy, puppiw;
    std::vector<int16_t> pdgId;
    std::vector<uint8_t> quality;
    MuonSOA() : bx(), offsets(), pt(), eta(), phi(), z0(), dxy(), puppiw(), pdgId(), quality() {}
    MuonSOA(const MuonSOA& other) = default;
    MuonSOA(MuonSOA&& other) = default;
    MuonSOA& operator=(const MuonSOA& other) = default;
    MuonSOA& operator=(MuonSOA&& other) = default;
    void swap(MuonSOA& other) {
      using std::swap;
      swap(bx, other.bx);
      swap(offsets, other.offsets);
      swap(pt, other.pt);
      swap(eta, other.eta);
      swap(phi, other.phi);
      swap(z0, other.z0);
      swap(dxy, other.dxy);
      swap(puppiw, other.puppiw);
      swap(pdgId, other.pdgId);
      swap(quality, other.quality);
    }
  };
  inline void swap(MuonSOA& a, MuonSOA& b) { a.swap(b); }
*/  
}  // namespace l1Scouting
#endif
