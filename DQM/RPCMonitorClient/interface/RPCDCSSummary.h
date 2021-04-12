#ifndef RPCMonitorClient_RPCDCSSummary_H
#define RPCMonitorClient_RPCDCSSummary_H

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include <map>

class RPCDCSSummary : public DQMEDHarvester {
public:
  /// Constructor
  RPCDCSSummary(const edm::ParameterSet &);

  /// Destructor
  ~RPCDCSSummary() override;

  // Operations

protected:
  void beginJob() override;
  void dqmEndLuminosityBlock(DQMStore::IBooker &,
                             DQMStore::IGetter &,
                             edm::LuminosityBlock const &,
                             edm::EventSetup const &) override;       //performed in the endLumi
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;  //performed in the endJob

private:
  void myBooker(DQMStore::IBooker &);
  void checkDCSbit(edm::EventSetup const &);

  bool init_;
  double defaultValue_;

  bool offlineDQM_;

  MonitorElement *DCSMap_;
  MonitorElement *totalDCSFraction;
  constexpr static int kNWheels = 5;
  MonitorElement *dcsWheelFractions[kNWheels];
  constexpr static int kNDisks = 10;
  MonitorElement *dcsDiskFractions[kNDisks];
  std::pair<int, int> FEDRange_;
  int numberOfDisks_;
  int NumberOfFeds_;
};

#endif
