#ifndef SiPixelDetId_PixelEndcapNameUpgrade_H
#define SiPixelDetId_PixelEndcapNameUpgrade_H

/** \class PixelEndcapNameUpgrade
 * Endcap Module name (as in PixelDatabase) for endcaps
 */

#include "DataFormats/SiPixelDetId/interface/PixelModuleName.h"

#include <string>
#include <iostream>
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"

class DetId;

class PixelEndcapNameUpgrade : public PixelModuleName {
public:
  enum HalfCylinder { mO = 1, mI = 2, pO = 3, pI = 4 };

  /// ctor from DetId
  PixelEndcapNameUpgrade(const DetId &);

  /// ctor for defined name
  PixelEndcapNameUpgrade(HalfCylinder part = mO, int disk = 0, int blade = 0, int pannel = 0, int plaq = 0)
      : PixelModuleName(false), thePart(part), theDisk(disk), theBlade(blade), thePannel(pannel), thePlaquette(plaq) {}

  /// ctor from name string
  PixelEndcapNameUpgrade(std::string name);

  ~PixelEndcapNameUpgrade() override {}

  /// from base class
  std::string name() const override;

  HalfCylinder halfCylinder() const { return thePart; }

  /// disk id
  int diskName() const { return theDisk; }

  /// blade id
  int bladeName() const { return theBlade; }

  /// pannel id
  int pannelName() const { return thePannel; }

  /// plaquetteId (in pannel)
  int plaquetteName() const { return thePlaquette; }

  /// module Type
  PixelModuleName::ModuleType moduleType() const override;

  /// return DetId
  PXFDetId getDetId();

  /// check equality of modules from datamemebers
  bool operator==(const PixelModuleName &) const override;
  bool operator==(const PixelEndcapNameUpgrade &other) const {
    return (thePart == other.thePart && theDisk == other.theDisk && theBlade == other.theBlade &&
            thePannel == other.thePannel && thePlaquette == other.thePlaquette);
  }

private:
  HalfCylinder thePart;
  int theDisk, theBlade, thePannel, thePlaquette;
};

std::ostream &operator<<(std::ostream &out, const PixelEndcapNameUpgrade::HalfCylinder &t);
#endif
