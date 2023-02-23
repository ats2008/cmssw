#ifndef tdr_regionizer_ref_h
#define tdr_regionizer_ref_h

#include "L1Trigger/Phase2L1ParticleFlow/interface/regionizer/regionizer_base_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/regionizer/tdr_regionizer_elements_ref.h"

namespace edm {
  class ParameterSet;
}

namespace l1ct {
  class TDRRegionizerEmulator : public RegionizerEmulator {
  public:
    TDRRegionizerEmulator(uint32_t ntk,
                          uint32_t ncalo,
                          uint32_t nem,
                          uint32_t nmu,
                          int32_t nclocks,
                          std::vector<int32_t> bigRegionEdges,
                          bool dosort);

    // note: this one will work only in CMSSW
    TDRRegionizerEmulator(const edm::ParameterSet& iConfig);

    ~TDRRegionizerEmulator() override;

    static const unsigned int NTK_SECTORS = 9, NTK_LINKS = 1;  // max objects per sector per clock cycle
    static const unsigned int NCALO_SECTORS = 3, NCALO_LINKS = 1;
    static const unsigned int NEMCALO_SECTORS = 3, NEMCALO_LINKS = 1;
    static const unsigned int NMU_LINKS = 1;

    //assuming 96b for tracks, 64b for emcalo, calo, mu
    //all at TMUX18, per link

    // numbers of small regions

    void initSectorsAndRegions(const RegionizerDecodedInputs& in, const std::vector<PFInputRegion>& out) override;

    // TODO: implement
    void run(const RegionizerDecodedInputs& in, std::vector<PFInputRegion>& out) override;

    // link emulation from decoded inputs (for simulation)
    void fillLinks(const RegionizerDecodedInputs& in, std::vector<std::vector<l1ct::TkObjEmu>>& links);
    void fillLinks(const RegionizerDecodedInputs& in, std::vector<std::vector<l1ct::HadCaloObjEmu>>& links);
    void fillLinks(const RegionizerDecodedInputs& in, std::vector<std::vector<l1ct::EmCaloObjEmu>>& links);
    void fillLinks(const RegionizerDecodedInputs& in, std::vector<std::vector<l1ct::MuObjEmu>>& links);

    // convert links to firmware
    void toFirmware(const std::vector<l1ct::TkObjEmu>& emu, TkObj fw[NTK_SECTORS][NTK_LINKS]);
    void toFirmware(const std::vector<l1ct::HadCaloObjEmu>& emu, HadCaloObj fw[NCALO_SECTORS][NCALO_LINKS]);
    void toFirmware(const std::vector<l1ct::EmCaloObjEmu>& emu, EmCaloObj fw[NCALO_SECTORS][NCALO_LINKS]);
    void toFirmware(const std::vector<l1ct::MuObjEmu>& emu, MuObj fw[NMU_LINKS]);

  private:
    /// The nubmer of barrel big regions (boards)
    uint32_t nBigRegions_;
    /// The maximum number of objects of each type to output per small region
    uint32_t ntk_, ncalo_, nem_, nmu_;
    /// The number of clocks to receive all data of an event (TMUX18 = 162)
    int32_t nclocks_;

    /// The phi edges of the big regions (boards); one greater than the number of boards
    std::vector<int32_t> bigRegionEdges_;

    bool dosort_;

    /// The number of eta and phi small regions in a big region (board)
    uint32_t netaInBR_, nphiInBR_;
    /// The total number of small regions in barrel (not just in board)
    uint32_t nregions_;

    uint32_t MAX_TK_OBJ_;
    uint32_t MAX_EMCALO_OBJ_;
    uint32_t MAX_HADCALO_OBJ_;
    uint32_t MAX_MU_OBJ_;

    bool init_; // has initialization happened

    std::vector<tdr_regionizer::Regionizer<l1ct::TkObjEmu>> tkRegionizers_;
    std::vector<tdr_regionizer::Regionizer<l1ct::HadCaloObjEmu>> hadCaloRegionizers_;
    std::vector<tdr_regionizer::Regionizer<l1ct::EmCaloObjEmu>> emCaloRegionizers_;
    std::vector<tdr_regionizer::Regionizer<l1ct::MuObjEmu>> muRegionizers_;
  };

}  // namespace l1ct

#endif
