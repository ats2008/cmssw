import FWCore.ParameterSet.Config as cms

# helper functions
from HLTrigger.Configuration.common import *

# add one customisation function per PR
# - put the PR number into the name of the function
# - add a short comment
# for example:

# CCCTF tuning
# def customiseFor12718(process):
#     for pset in process._Process__psets.values():
#         if hasattr(pset,'ComponentType'):
#             if (pset.ComponentType == 'CkfBaseTrajectoryFilter'):
#                 if not hasattr(pset,'minGoodStripCharge'):
#                     pset.minGoodStripCharge = cms.PSet(refToPSet_ = cms.string('HLTSiStripClusterChargeCutNone'))
#     return process

def customiseHCALFor2018Input(process):
    """Customise the HLT to run on Run 2 data/MC using the old readout for the HCAL barel"""

    for producer in producers_by_type(process, "HBHEPhase1Reconstructor"):
        # switch on the QI8 processing for 2018 HCAL barrel
        producer.processQIE8 = True

    # adapt CaloTowers threshold for 2018 HCAL barrel with only one depth
    for producer in producers_by_type(process, "CaloTowersCreator"):
        producer.HBThreshold1  = 0.7
        producer.HBThreshold2  = 0.7
        producer.HBThreshold   = 0.7

    # adapt Particle Flow threshold for 2018 HCAL barrel with only one depth
    from RecoParticleFlow.PFClusterProducer.particleFlowClusterHBHE_cfi import _thresholdsHB, _thresholdsHEphase1, _seedingThresholdsHB

    logWeightDenominatorHCAL2018 = cms.VPSet(
        cms.PSet(
            depths = cms.vint32(1, 2, 3, 4),
            detector = cms.string('HCAL_BARREL1'),
            logWeightDenominator = _thresholdsHB
        ),
        cms.PSet(
            depths = cms.vint32(1, 2, 3, 4, 5, 6, 7),
            detector = cms.string('HCAL_ENDCAP'),
            logWeightDenominator = _thresholdsHEphase1
        )
    )

    for producer in producers_by_type(process, "PFRecHitProducer"):
        if producer.producers[0].name.value() == 'PFHBHERecHitCreator':
            producer.producers[0].qualityTests[0].cuts[0].threshold = _thresholdsHB

    for producer in producers_by_type(process, "PFClusterProducer"):
        if producer.seedFinder.thresholdsByDetector[0].detector.value() == 'HCAL_BARREL1':
            producer.seedFinder.thresholdsByDetector[0].seedingThresholds = _seedingThresholdsHB
            producer.initialClusteringStep.thresholdsByDetector[0].gatheringThresholds = _thresholdsHB
            producer.pfClusterBuilder.recHitEnergyNorms[0].recHitEnergyNorm = _thresholdsHB
            producer.pfClusterBuilder.positionCalc.logWeightDenominatorByDetector = logWeightDenominatorHCAL2018
            producer.pfClusterBuilder.allCellsPositionCalc.logWeightDenominatorByDetector = logWeightDenominatorHCAL2018

    for producer in producers_by_type(process, "PFMultiDepthClusterProducer"):
        producer.pfClusterBuilder.allCellsPositionCalc.logWeightDenominatorByDetector = logWeightDenominatorHCAL2018

    # done
    return process

def customiseFor2017DtUnpacking(process):
    """Adapt the HLT to run the legacy DT unpacking
    for pre2018 data/MC workflows as the default"""

    if hasattr(process,'hltMuonDTDigis'):
        process.hltMuonDTDigis = cms.EDProducer( "DTUnpackingModule",
            useStandardFEDid = cms.bool( True ),
            maxFEDid = cms.untracked.int32( 779 ),
            inputLabel = cms.InputTag( "rawDataCollector" ),
            minFEDid = cms.untracked.int32( 770 ),
            dataType = cms.string( "DDU" ),
            readOutParameters = cms.PSet(
                localDAQ = cms.untracked.bool( False ),
                debug = cms.untracked.bool( False ),
                rosParameters = cms.PSet(
                    localDAQ = cms.untracked.bool( False ),
                    debug = cms.untracked.bool( False ),
                    writeSC = cms.untracked.bool( True ),
                    readDDUIDfromDDU = cms.untracked.bool( True ),
                    readingDDU = cms.untracked.bool( True ),
                    performDataIntegrityMonitor = cms.untracked.bool( False )
                    ),
                performDataIntegrityMonitor = cms.untracked.bool( False )
                ),
            dqmOnly = cms.bool( False )
        )

    return process

def customisePixelGainForRun2Input(process):
    """Customise the HLT to run on Run 2 data/MC using the old definition of the pixel calibrations

    Up to 11.0.x, the pixel calibarations were fully specified in the configuration:
        VCaltoElectronGain      =   47
        VCaltoElectronGain_L1   =   50
        VCaltoElectronOffset    =  -60
        VCaltoElectronOffset_L1 = -670

    Starting with 11.1.x, the calibrations for Run 3 were moved to the conditions, leaving in the configuration only:
        VCaltoElectronGain      =    1
        VCaltoElectronGain_L1   =    1
        VCaltoElectronOffset    =    0
        VCaltoElectronOffset_L1 =    0

    Since the conditions for Run 2 have not been updated to the new scheme, the HLT configuration needs to be reverted.
    """
    # revert the Pixel parameters to be compatible with the Run 2 conditions
    for producer in producers_by_type(process, "SiPixelClusterProducer"):
        producer.VCaltoElectronGain = 47
        producer.VCaltoElectronGain_L1 = 50
        producer.VCaltoElectronOffset = -60
        producer.VCaltoElectronOffset_L1 = -670

    for pluginType in ["SiPixelRawToClusterCUDA", "SiPixelRawToClusterCUDAPhase1", "SiPixelRawToClusterCUDAHIonPhase1"]:
        for producer in producers_by_type(process, pluginType):
            producer.VCaltoElectronGain = 47
            producer.VCaltoElectronGain_L1 = 50
            producer.VCaltoElectronOffset = -60
            producer.VCaltoElectronOffset_L1 = -670

    return process

def customisePixelL1ClusterThresholdForRun2Input(process):
    # revert the pixel Layer 1 cluster threshold to be compatible with Run2:
    for producer in producers_by_type(process, "SiPixelClusterProducer"):
        if hasattr(producer,"ClusterThreshold_L1"):
            producer.ClusterThreshold_L1 = 2000
    for pluginType in ["SiPixelRawToClusterCUDA", "SiPixelRawToClusterCUDAPhase1", "SiPixelRawToClusterCUDAHIonPhase1"]:
        for producer in producers_by_type(process, pluginType):
            if hasattr(producer,"clusterThreshold_layer1"):
                producer.clusterThreshold_layer1 = 2000
    for producer in producers_by_type(process, "SiPixelDigisClustersFromSoA"):
        if hasattr(producer,"clusterThreshold_layer1"):
            producer.clusterThreshold_layer1 = 2000

    return process

def customiseCTPPSFor2018Input(process):
    for prod in producers_by_type(process, 'CTPPSGeometryESModule'):
        prod.isRun2 = True
    for prod in producers_by_type(process, 'CTPPSPixelRawToDigi'):
        prod.isRun3 = False

    return process

def customiseEGammaRecoFor2018Input(process):
    for prod in producers_by_type(process, 'PFECALSuperClusterProducer'):
        if hasattr(prod, 'regressionConfig'):
            prod.regressionConfig.regTrainedWithPS = cms.bool(False)

    return process

def customiseBeamSpotFor2018Input(process):
    """Customisation for the HLT BeamSpot when running on Run-2 (2018) data:
       - For Run-2 data, disable the use of the BS transient record, in order to read the BS record from SCAL.
       - Additionally, remove all instances of OnlineBeamSpotESProducer (not needed if useTransientRecord=False).
       - See CMSHLT-2271 and CMSHLT-2300 for further details.
    """
    for prod in producers_by_type(process, 'BeamSpotOnlineProducer'):
        prod.useTransientRecord = False
    onlineBeamSpotESPLabels = [prod.label_() for prod in esproducers_by_type(process, 'OnlineBeamSpotESProducer')]
    for espLabel in onlineBeamSpotESPLabels:
        delattr(process, espLabel)

    # re-introduce SCAL digis, if missing
    if not hasattr(process, 'hltScalersRawToDigi') and hasattr(process, 'HLTBeamSpot') and isinstance(process.HLTBeamSpot, cms.Sequence):

        if hasattr(process, 'hltOnlineBeamSpot'):
            process.hltOnlineBeamSpot.src = 'hltScalersRawToDigi'

        if hasattr(process, 'hltPixelTrackerHVOn'):
            process.hltPixelTrackerHVOn.DcsStatusLabel = 'hltScalersRawToDigi'

        if hasattr(process, 'hltStripTrackerHVOn'):
            process.hltStripTrackerHVOn.DcsStatusLabel = 'hltScalersRawToDigi'

        process.hltScalersRawToDigi = cms.EDProducer( "ScalersRawToDigi",
            scalersInputTag = cms.InputTag( "rawDataCollector" )
        )

        process.HLTBeamSpot.insert(0, process.hltScalersRawToDigi)

    return process

def customiseECALCalibrationsFor2018Input(process):
    """Customisation to apply the ECAL Run-2 Ultra-Legacy calibrations (CMSHLT-2339)"""
    if hasattr(process, 'GlobalTag'):
      if not hasattr(process.GlobalTag, 'toGet'):
        process.GlobalTag.toGet = cms.VPSet()
      process.GlobalTag.toGet += [
        cms.PSet(
          record = cms.string('EcalLaserAlphasRcd'),
          tag = cms.string('EcalLaserAlphas_UL_Run1_Run2_2018_lastIOV_movedTo1')
        ),
        cms.PSet(
          record = cms.string('EcalIntercalibConstantsRcd'),
          tag = cms.string('EcalIntercalibConstants_UL_Run1_Run2_2018_lastIOV_movedTo1')
        )
      ]
    else:
      print('# customiseECALCalibrationsFor2018Input -- the process.GlobalTag ESSource does not exist: no customisation applied.')

    return process

def customiseFor2018Input(process):
    """Customise the HLT to run on Run 2 data/MC"""
    process = customisePixelGainForRun2Input(process)
    process = customisePixelL1ClusterThresholdForRun2Input(process)
    process = customiseHCALFor2018Input(process)
    process = customiseCTPPSFor2018Input(process)
    process = customiseEGammaRecoFor2018Input(process)
    process = customiseBeamSpotFor2018Input(process)
    process = customiseECALCalibrationsFor2018Input(process)

    return process


def customiseForOffline(process):
    # For running HLT offline on Run-3 Data, use "(OnlineBeamSpotESProducer).timeThreshold = 1e6",
    # in order to pick the beamspot that was actually used by the HLT (instead of a "fake" beamspot).
    # These same settings can be used offline for Run-3 Data and Run-3 MC alike.
    # Note: the products of the OnlineBeamSpotESProducer are used only
    #       if the configuration uses "(BeamSpotOnlineProducer).useTransientRecord = True".
    # See CMSHLT-2271 and CMSHLT-2300 for further details.
    for prod in esproducers_by_type(process, 'OnlineBeamSpotESProducer'):
        prod.timeThreshold = int(1e6)

    # For running HLT offline and relieve the strain on Frontier so it will no longer inject a
    # transaction id which tells Frontier to add a unique "&freshkey" to many query URLs.
    # That was intended as a feature to only be used by the Online HLT, to guarantee that fresh conditions
    # from the database were loaded at each Lumi section
    # Seee CMSHLT-3123 for further details
    if hasattr(process, 'GlobalTag'):
        # Set ReconnectEachRun and RefreshEachRun to False
        process.GlobalTag.ReconnectEachRun = cms.untracked.bool(False)
        process.GlobalTag.RefreshEachRun = cms.untracked.bool(False)

        if hasattr(process.GlobalTag, 'toGet'):
            # Filter out PSet objects containing only 'record' and 'refreshTime'
            process.GlobalTag.toGet = [
                pset for pset in process.GlobalTag.toGet
                if set(pset.parameterNames_()) != {'record', 'refreshTime'}
            ]

    return process

def checkHLTfor43774(process):
    filt_types = ["HLTEgammaGenericFilter","HLTEgammaGenericQuadraticEtaFilter","HLTEgammaGenericQuadraticFilter","HLTElectronGenericFilter"]
    absAbleVar = ["DEta","deta","DetaSeed","Dphi","OneOESuperMinusOneOP","OneOESeedMinusOneOP"]
    for filt_type in filt_types:
        for filt in filters_by_type(process, filt_type):
            if filt.varTag.productInstanceLabel in absAbleVar:
                if (filt.useAbs != cms.bool(True)):
                    print('# TSG WARNING: check value of parameter "useAbs" in',filt,'(expect True but is False)!')

    return process
    
def customizeHLTfor44576(process):
    """Ensure TrackerAdditionalParametersPerDetRcd ESProducer is run when needed"""
    for esprod in esproducers_by_type(process, 'TrackerGeometricDetESModule'):
        process.load("Geometry.TrackerGeometryBuilder.TrackerAdditionalParametersPerDet_cfi")
        break
    return process

def customizeHLTfor45063(process):
    """Assigns value of MuonHLTSeedMVAClassifier mva input file, scales and mean values according to the value of isFromL1"""
    for prod in producers_by_type(process, 'MuonHLTSeedMVAClassifier'):
        if hasattr(prod, "isFromL1"):
            if (prod.isFromL1 == True):
                if hasattr(prod, "mvaFileBL1"):
                    prod.mvaFileB = prod.mvaFileBL1
                if hasattr(prod, "mvaFileEL1"):
                    prod.mvaFileE = prod.mvaFileEL1
                if hasattr(prod, "mvaScaleMeanBL1"):
                    prod.mvaScaleMeanB = prod.mvaScaleMeanBL1
                if hasattr(prod, "mvaScaleStdBL1"):
                    prod.mvaScaleStdB = prod.mvaScaleStdBL1
                if hasattr(prod, "mvaScaleMeanEL1"):
                    prod.mvaScaleMeanE = prod.mvaScaleMeanEL1
                if hasattr(prod, "mvaScaleStdEL1"):                    
                    prod.mvaScaleStdE = prod.mvaScaleStdEL1                
            else:
                if hasattr(prod, "mvaFileBL2"):
                    prod.mvaFileB = prod.mvaFileBL2
                if hasattr(prod, "mvaFileEL2"):
                    prod.mvaFileE = prod.mvaFileEL2
                if hasattr(prod, "mvaScaleMeanBL2"):
                    prod.mvaScaleMeanB = prod.mvaScaleMeanBL2
                if hasattr(prod, "mvaScaleStdBL2"):
                    prod.mvaScaleStdB = prod.mvaScaleStdBL2
                if hasattr(prod, "mvaScaleMeanEL2"):
                    prod.mvaScaleMeanE = prod.mvaScaleMeanEL2
                if hasattr(prod, "mvaScaleStdEL2"):
                    prod.mvaScaleStdE = prod.mvaScaleStdEL2
                    
    for prod in producers_by_type(process, 'MuonHLTSeedMVAClassifier'):
        delattr(prod,"mvaFileBL1")
        delattr(prod,"mvaFileEL1")
        delattr(prod,"mvaScaleMeanBL1")
        delattr(prod,"mvaScaleStdBL1")
        delattr(prod,"mvaScaleMeanEL1")
        delattr(prod,"mvaScaleStdEL1")
        delattr(prod,"mvaFileBL2")
        delattr(prod,"mvaFileEL2")
        delattr(prod,"mvaScaleMeanBL2")
        delattr(prod,"mvaScaleStdBL2")
        delattr(prod,"mvaScaleMeanEL2")
        delattr(prod,"mvaScaleStdEL2")       
                    
    return process

def customizeHLTfor45212(process):
    # Add missing parameters for Basic2DGenericPFlowPositionCalc
    for p in ["PFClusterProducer", "LegacyPFClusterProducer", "PFMultiDepthClusterProducer"]:
        for prod in producers_by_type(process, p):
            for n in ["positionCalc", "allCellsPositionCalc"]:
                if hasattr(prod, "pfClusterBuilder") and hasattr(prod.pfClusterBuilder, n) and hasattr(getattr(prod.pfClusterBuilder, n), "algoName") and (getattr(prod.pfClusterBuilder, n).algoName == "Basic2DGenericPFlowPositionCalc"):
                    if not hasattr(getattr(prod.pfClusterBuilder, n), "logWeightDenominatorByDetector"):
                        getattr(prod.pfClusterBuilder, n).logWeightDenominatorByDetector = cms.VPSet()
                    if not hasattr(getattr(prod.pfClusterBuilder, n), "timeResolutionCalcBarrel"):
                        getattr(prod.pfClusterBuilder, n).timeResolutionCalcBarrel = cms.PSet()
                    if not hasattr(getattr(prod.pfClusterBuilder, n), "timeResolutionCalcEndcap"):
                        getattr(prod.pfClusterBuilder, n).timeResolutionCalcEndcap = cms.PSet()

    # Add missing parameters for ECAL2DPositionCalcWithDepthCorr
    for prod in producers_by_type(process, 'PFClusterProducer'):
        if hasattr(prod, "pfClusterBuilder") and hasattr(prod.pfClusterBuilder, "positionCalcForConvergence") and hasattr(prod.pfClusterBuilder.positionCalcForConvergence, "algoName") and (prod.pfClusterBuilder.positionCalcForConvergence.algoName == "ECAL2DPositionCalcWithDepthCorr"):
            if not hasattr(prod.pfClusterBuilder.positionCalcForConvergence, "timeResolutionCalc"):
                prod.pfClusterBuilder.positionCalcForConvergence.timeResolutionCalc = cms.PSet()
        if hasattr(prod, "positionReCalc") and hasattr(prod.positionReCalc, "algoName") and (prod.positionReCalc.algoName == "ECAL2DPositionCalcWithDepthCorr"):
            if not hasattr(prod.positionReCalc, "timeResolutionCalc"):
                prod.positionReCalc.timeResolutionCalc = cms.PSet()

    # Add missing parameters for Basic2DGenericPFlowClusterizer
    for p in ["PFClusterProducer", "LegacyPFClusterProducer"]:
        for prod in producers_by_type(process, p):
            if hasattr(prod, "pfClusterBuilder") and hasattr(prod.pfClusterBuilder, "algoName") and (prod.pfClusterBuilder.algoName == "Basic2DGenericPFlowClusterizer"):
                if not hasattr(prod.pfClusterBuilder, "positionCalc"):
                    prod.pfClusterBuilder.positionCalc = cms.PSet()
                if not hasattr(prod.pfClusterBuilder, "allCellsPositionCalc"):
                    prod.pfClusterBuilder.allCellsPositionCalc = cms.PSet()
                if not hasattr(prod.pfClusterBuilder, "positionCalcForConvergence"):
                    prod.pfClusterBuilder.positionCalcForConvergence = cms.PSet()

    # Add missing parameters for PFMultiDepthClusterizer
    for prod in producers_by_type(process, 'PFMultiDepthClusterProducer'):
        if hasattr(prod, "pfClusterBuilder") and hasattr(prod.pfClusterBuilder, "algoName") and (prod.pfClusterBuilder.algoName == "PFMultiDepthClusterizer"):
            if not hasattr(prod.pfClusterBuilder, "positionCalc"):
                prod.pfClusterBuilder.positionCalc = cms.PSet()
            if not hasattr(prod.pfClusterBuilder, "allCellsPositionCalc"):
                 prod.pfClusterBuilder.allCellsPositionCalc = cms.PSet()

    # Add missing parameters for PFCandConnector
    for prod in producers_by_type(process, 'PFProducer'):
        if hasattr(prod, "iCfgCandConnector"):
            if not hasattr(prod.iCfgCandConnector, "dptRel_PrimaryTrack"):
                prod.iCfgCandConnector.dptRel_PrimaryTrack = cms.double(0)
            if not hasattr(prod.iCfgCandConnector, "dptRel_MergedTrack"):
                prod.iCfgCandConnector.dptRel_MergedTrack = cms.double(0)
            if not hasattr(prod.iCfgCandConnector, "ptErrorSecondary"):
                prod.iCfgCandConnector.ptErrorSecondary = cms.double(0)
            if not hasattr(prod.iCfgCandConnector, "nuclCalibFactors"):
                prod.iCfgCandConnector.nuclCalibFactors = cms.vdouble()

    # Fix types for gathering and seeding thresholds, and recHitEnergyNorm
    for prod in producers_by_type(process, 'PFClusterProducer'):
        if hasattr(prod, "initialClusteringStep") and hasattr(prod.initialClusteringStep, "thresholdsByDetector"):
            if hasattr(prod.initialClusteringStep.thresholdsByDetector, "gatheringThreshold") and isinstance(prod.initialClusteringStep.thresholdsByDetector.gatheringThreshold, cms.vdouble):
                prod.initialClusteringStep.thresholdsByDetector.gatheringThresholds = prod.initialClusteringStep.thresholdsByDetector.gatheringThreshold
                delattr(prod.initialClusteringStep.thresholdsByDetector, "gatheringThreshold")
            if hasattr(prod.initialClusteringStep.thresholdsByDetector, "gatheringThresholdPt") and isinstance(prod.initialClusteringStep.thresholdsByDetector.gatheringThresholdPt, cms.vdouble):
                prod.initialClusteringStep.thresholdsByDetector.gatheringThresholdsPt = prod.initialClusteringStep.thresholdsByDetector.gatheringThresholdPt
                delattr(prod.initialClusteringStep.thresholdsByDetector, "gatheringThresholdPt")
        if hasattr(prod, "seedFinder") and hasattr(prod.seedFinder, "thresholdsByDetector"):
            if hasattr(prod.seedFinder.thresholdsByDetector, "seedingThreshold") and isinstance(prod.seedFinder.thresholdsByDetector.seedingThreshold, cms.vdouble):
                prod.seedFinder.thresholdsByDetector.seedingThresholds = prod.seedFinder.thresholdsByDetector.seedingThreshold
                delattr(prod.seedFinder.thresholdsByDetector, "seedingThreshold")
            if hasattr(prod.seedFinder.thresholdsByDetector, "seedingThresholdPt") and isinstance(prod.seedFinder.thresholdsByDetector.seedingThresholdPt, cms.vdouble):
                prod.seedFinder.thresholdsByDetector.seedingThresholdsPt = prod.seedFinder.thresholdsByDetector.seedingThresholdPt
                delattr(prod.seedFinder.thresholdsByDetector, "seedingThresholdPt")
        for n in ["positionCalc", "allCellsPositionCalc"]:
            if hasattr(prod, "pfClusterBuilder") and hasattr(prod.pfClusterBuilder, n) and hasattr(getattr(prod.pfClusterBuilder, n), "recHitEnergyNorms") and hasattr(getattr(prod.pfClusterBuilder, n).recHitEnergyNorms, "recHitEnergyNorm") and isinstance(getattr(prod.pfClusterBuilder, n).recHitEnergyNorms.recHitEnergyNorm, cms.vdouble):
                getattr(prod.pfClusterBuilder, n).recHitEnergyNorms.recHitEnergyNorms = getattr(prod.pfClusterBuilder, n).recHitEnergyNorms.recHitEnergyNorm
                delattr(getattr(prod.pfClusterBuilder, n).recHitEnergyNorms, "recHitEnergyNorm")

    return process
            
# CMSSW version specific customizations
def customizeHLTforCMSSW(process, menuType="GRun"):

    process = customiseForOffline(process)

    # add call to action function in proper order: newest last!
    # process = customiseFor12718(process)

    process = checkHLTfor43774(process)
    process = customizeHLTfor44576(process)
    process = customizeHLTfor45063(process)
    process = customizeHLTfor45212(process)

    return process
