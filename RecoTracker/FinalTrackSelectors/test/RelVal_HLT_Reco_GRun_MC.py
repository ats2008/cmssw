# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: RelVal --step=HLT:GRun,RAW2DIGI,L1Reco,RECO --conditions=auto:run3_mc_GRun --filein=file:RelVal_Raw_GRun_MC.root --custom_conditions= --fileout=RelVal_HLT_RECO_GRun_MC.root --number=100 --mc --no_exec --datatier SIM-RAW-HLT-RECO --eventcontent=RAW --customise=HLTrigger/Configuration/CustomConfigs.L1THLT --era=Run3 --customise= --scenario=pp --python_filename=RelVal_HLT_Reco_GRun_MC.py --processName=HLT --customise_commands=process.CSCHaloData.HLTResultLabel=cms.InputTag('')
import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_cff import Run3

process = cms.Process('validation',Run3)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('HLTrigger.Configuration.HLT_GRun_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100),
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_11_1_0_pre8/RelValZMM_14/GEN-SIM-DIGI-RAW/111X_mcRun3_2021_realistic_v4-v1/20000/0AD9240A-7DB1-2C44-B53B-B0F15E9FEED9.root',
        '/store/relval/CMSSW_11_1_0_pre8/RelValZMM_14/GEN-SIM-DIGI-RAW/111X_mcRun3_2021_realistic_v4-v1/20000/0AD9240A-7DB1-2C44-B53B-B0F15E9FEED9.root',
        '/store/relval/CMSSW_11_1_0_pre8/RelValZMM_14/GEN-SIM-DIGI-RAW/111X_mcRun3_2021_realistic_v4-v1/20000/2792A13C-1439-454A-9850-443D018EDB25.root',
        '/store/relval/CMSSW_11_1_0_pre8/RelValZMM_14/GEN-SIM-DIGI-RAW/111X_mcRun3_2021_realistic_v4-v1/20000/2A081C0C-9747-4B4C-9CCC-DC2E03D58452.root',
        '/store/relval/CMSSW_11_1_0_pre8/RelValZMM_14/GEN-SIM-DIGI-RAW/111X_mcRun3_2021_realistic_v4-v1/20000/2E6E7E87-73F9-AC40-8C50-CCFA38B85ACF.root',
        '/store/relval/CMSSW_11_1_0_pre8/RelValZMM_14/GEN-SIM-DIGI-RAW/111X_mcRun3_2021_realistic_v4-v1/20000/3FE6B733-ACC3-E041-AE39-9926ADCA8651.root',
        '/store/relval/CMSSW_11_1_0_pre8/RelValZMM_14/GEN-SIM-DIGI-RAW/111X_mcRun3_2021_realistic_v4-v1/20000/5FEDB326-A145-6949-A18F-FEB479DB9005.root',
        '/store/relval/CMSSW_11_1_0_pre8/RelValZMM_14/GEN-SIM-DIGI-RAW/111X_mcRun3_2021_realistic_v4-v1/20000/7CAEC2AB-AC17-2046-890C-6E01F67AC1D0.root',
        '/store/relval/CMSSW_11_1_0_pre8/RelValZMM_14/GEN-SIM-DIGI-RAW/111X_mcRun3_2021_realistic_v4-v1/20000/BDF4FD35-C176-4A43-91C8-9AF8C7ED74DC.root',
        '/store/relval/CMSSW_11_1_0_pre8/RelValZMM_14/GEN-SIM-DIGI-RAW/111X_mcRun3_2021_realistic_v4-v1/20000/E1B44039-321A-284E-91FB-97CD10F43A48.root'
),
    secondaryFileNames = cms.untracked.vstring()
)

process.options = cms.untracked.PSet(
    FailPath = cms.untracked.vstring(),
    IgnoreCompletely = cms.untracked.vstring(),
    Rethrow = cms.untracked.vstring(),
    SkipEvent = cms.untracked.vstring(),
    allowUnscheduled = cms.obsolete.untracked.bool,
    canDeleteEarly = cms.untracked.vstring(),
    emptyRunLumiMode = cms.obsolete.untracked.string,
    eventSetup = cms.untracked.PSet(
        forceNumberOfConcurrentIOVs = cms.untracked.PSet(

        ),
        numberOfConcurrentIOVs = cms.untracked.uint32(1)
    ),
    fileMode = cms.untracked.string('FULLMERGE'),
    forceEventSetupCacheClearOnNewRun = cms.untracked.bool(False),
    makeTriggerResults = cms.obsolete.untracked.bool,
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfStreams = cms.untracked.uint32(0),
    numberOfThreads = cms.untracked.uint32(1),
    printDependencies = cms.untracked.bool(False),
    sizeOfStackForThreadsInKB = cms.optional.untracked.uint32,
    throwIfIllegalParameter = cms.untracked.bool(True),
    wantSummary = cms.untracked.bool(False)
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('RelVal nevts:100'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.RAWoutput = cms.OutputModule("PoolOutputModule",
    compressionAlgorithm = cms.untracked.string('LZMA'),
    compressionLevel = cms.untracked.int32(4),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('SIM-RAW-HLT-RECO'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('RelVal_HLT_RECO_GRun_MC.root'),
    outputCommands = process.RAWEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run3_mc_GRun', '')

#############################  Track Selector by Region   ##########################


TrackingRegionOfInteret='hltIterL3MuonPixelTracksTrackingRegions'

TrackCollectionOfInterest='hltIterL3MuonPixelTracks'


process.hltIterL3MuonPixelTracksFromGlobalViaRegion = cms.EDProducer('TrackSelectorByRegion',
  tracks = cms.InputTag('hltPixelTracks'),
  RegionPSet = cms.PSet(
      input = cms.InputTag(TrackingRegion),
      phiTolerance = cms.double(1.1),
      etaTolerance = cms.double(1.1),
      dZTolerance = cms.double(1.1),
      dXYTolerance = cms.double(1.1),

  ),
  TrackPSet = cms.PSet(
    minPt = cms.double(0),
  )
)

process.hltIterL3MuonPixelTracksFromRegionalViaRegion = cms.EDProducer('TrackSelectorByRegion',
  tracks = cms.InputTag(TrackCollectionOfInterest),
  RegionPSet = cms.PSet(
      input = cms.InputTag(TrackingRegion),
      phiTolerance = cms.double(1.1),
      etaTolerance = cms.double(1.1),
      dZTolerance = cms.double(1.1),
      dXYTolerance = cms.double(1.1),

  ),
  TrackPSet = cms.PSet(
    minPt = cms.double(0),
  )
)



########### TrackSelectorByRegion DQM_OUTPUT

process.load( "DQMServices.Core.DQMStore_cfi" )

process.dqmOutput = cms.OutputModule("DQMRootOutputModule",
    fileName = cms.untracked.string("DQMIO.root")
)

process.hltOutputTrackSelectorValidation = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "TrkSelByRegion_validationRAW.root" ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    outputCommands = cms.untracked.vstring( 
      'keep *_hltIterL3MuonPixelTracksFromRegionalViaRegion_*_*',
      'keep *_hltIterL3MuonPixelTracksFromGlobalViaRegion_*_*',
      'keep *_'+TrackCollectionOfInterest+'_*_*',
      'keep *_hltPixelTracks_*_*'
     )
    )

process.DQMOutput = cms.EndPath(process.dqmOutput + process.hltOutputTrackSelectorValidation )

##  Monitoring and MultiTrackValidator
process.load("DQMOffline.Trigger.TrackingMonitoring_cff")
process.pixelTracksMonitoringHLT.beamSpot   = cms.InputTag("hltOnlineBeamSpot")
process.pixelTracksMonitoringHLT.primaryVertex   = cms.InputTag("hltPixelVertices")
process.pixelTracksMonitoringHLT.pvNDOF   = cms.int32(1)

process.regionalPixelTracksMonitoringHLT = process.pixelTracksMonitoringHLT.clone()
process.regionalPixelTracksMonitoringHLT.FolderName       = 'HLT/Tracking/'+TrackCollectionOfInterest
process.regionalPixelTracksMonitoringHLT.TrackProducer    = TrackCollectionOfInterest
process.regionalPixelTracksMonitoringHLT.allTrackProducer = TrackCollectionOfInterest
process.regionalPixelTracksMonitoringHLT.doEffFromHitPatternVsPU   = False
process.regionalPixelTracksMonitoringHLT.doEffFromHitPatternVsBX   = False
process.regionalPixelTracksMonitoringHLT.doEffFromHitPatternVsLUMI = False

process.regionalPixelTracksFromRegionalViaRegionMonitoringHLT = process.pixelTracksMonitoringHLT.clone()
process.regionalPixelTracksFromRegionalViaRegionMonitoringHLT.FolderName       = 'HLT/Tracking/iterL3MuonPixelTracksFromRegionalViaRegion'
process.regionalPixelTracksFromRegionalViaRegionMonitoringHLT.TrackProducer    = 'hltIterL3MuonPixelTracksFromRegionalViaRegion'
process.regionalPixelTracksFromRegionalViaRegionMonitoringHLT.allTrackProducer = 'hltIterL3MuonPixelTracksFromRegionalViaRegion'
process.regionalPixelTracksFromRegionalViaRegionMonitoringHLT.doEffFromHitPatternVsPU   = False
process.regionalPixelTracksFromRegionalViaRegionMonitoringHLT.doEffFromHitPatternVsBX   = False
process.regionalPixelTracksFromRegionalViaRegionMonitoringHLT.doEffFromHitPatternVsLUMI = False

process.regionalIterL3PixelTracksFromGlobalViaRegionMonitoringHLT = process.pixelTracksMonitoringHLT.clone()
process.regionalIterL3PixelTracksFromGlobalViaRegionMonitoringHLT.FolderName       = 'HLT/Tracking/iterL3MuonPixelTracksFromGlobalViaRegion'
process.regionalIterL3PixelTracksFromGlobalViaRegionMonitoringHLT.TrackProducer    = 'hltIterL3MuonPixelTracksFromGlobalViaRegion'
process.regionalIterL3PixelTracksFromGlobalViaRegionMonitoringHLT.allTrackProducer = 'hltIterL3MuonPixelTracksFromGlobalViaRegion'
process.regionalIterL3PixelTracksFromGlobalViaRegionMonitoringHLT.doEffFromHitPatternVsPU   = False
process.regionalIterL3PixelTracksFromGlobalViaRegionMonitoringHLT.doEffFromHitPatternVsBX   = False
process.regionalIterL3PixelTracksFromGlobalViaRegionMonitoringHLT.doEffFromHitPatternVsLUMI = False

process.load("Validation.RecoTrack.HLTmultiTrackValidator_cff")
process.hltPixelTrackValidator = process.hltTrackValidator.clone()
process.hltPixelTrackValidator.cores = cms.InputTag("")
process.hltPixelTrackValidator.label = [
    "hltPixelTracks",
    "hltIterL3MuonPixelTracks",
    "hltIterL3MuonPixelTracksFromRegionalViaRegion",
    "hltIterL3MuonPixelTracksFromGlobalViaRegion",
]


process.hltTracksValidationTruth = cms.Sequence(process.hltTPClusterProducer+process.hltTrackAssociatorByHits+process.trackingParticleNumberOfLayersProducer)
process.validation = cms.Path(
    process.hltTracksValidationTruth
    *process.hltPixelTrackValidator
)


process.dqm = cms.Path(
    process.pixelTracksMonitoringHLT
    +process.regionalPixelTracksMonitoringHLT
    +process.regionalIterL3PixelTracksFromGlobalViaRegionMonitoringHLT
    +process.regionalPixelTracksFromRegionalViaRegionMonitoringHLT
    )


process.hltL1sSingleMu =cms.EDFilter("HLTL1TSeed",
    L1EGammaInputTag = cms.InputTag("hltGtStage2Digis","EGamma"),
    L1EtSumInputTag = cms.InputTag("hltGtStage2Digis","EtSum"),
    L1GlobalInputTag = cms.InputTag("hltGtStage2Digis"),
    L1JetInputTag = cms.InputTag("hltGtStage2Digis","Jet"),
    L1MuonInputTag = cms.InputTag("hltGtStage2Digis","Muon"),
    L1ObjectMapInputTag = cms.InputTag("hltGtStage2ObjectMap"),
    L1SeedsLogicalExpression = cms.string('L1_SingleMu18 OR L1_SingleMu22'),
    L1TauInputTag = cms.InputTag("hltGtStage2Digis","Tau"),
    saveTags = cms.bool(True)
)

process.hltIterL3MuonPixelTracksTrackingRegions = cms.EDProducer("MuonTrackingRegionEDProducer",
    DeltaEta = cms.double(0.2),
    DeltaPhi = cms.double(0.15),
    DeltaR = cms.double(0.025),
    DeltaZ = cms.double(24.2),
    EtaR_UpperLimit_Par1 = cms.double(0.25),
    EtaR_UpperLimit_Par2 = cms.double(0.15),
    Eta_fixed = cms.bool(True),
    Eta_min = cms.double(0.0),
    MeasurementTrackerName = cms.InputTag(""),
    OnDemand = cms.int32(-1),
    PhiR_UpperLimit_Par1 = cms.double(0.6),
    PhiR_UpperLimit_Par2 = cms.double(0.2),
    Phi_fixed = cms.bool(True),
    Phi_min = cms.double(0.0),
    Pt_fixed = cms.bool(True),
    Pt_min = cms.double(2.0),
    Rescale_Dz = cms.double(4.0),
    Rescale_eta = cms.double(3.0),
    Rescale_phi = cms.double(3.0),
    UseVertex = cms.bool(False),
    Z_fixed = cms.bool(True),
    beamSpot = cms.InputTag("hltOnlineBeamSpot"),
    input = cms.InputTag("hltL2Muons","UpdatedAtVtx"),  #cms.InputTag("hltL2SelectorForL3IO"),
    maxRegions = cms.int32(5),
    precise = cms.bool(True),
    vertexCollection = cms.InputTag("notUsed")
)


##################

process.HLTRecoPixelTracksFromGlobalViaRegionSequenceForIterL3 = cms.Sequence(process.HLTRecopixelvertexingSequence+process.hltIterL3MuonPixelTracksTrackingRegions+process.hltIterL3MuonPixelTracksFromGlobalViaRegion)

process.HLTRecoPixelTracksFromRegionalViaRegionSequenceForIterL3 = cms.Sequence(process.HLTIterL3MuonRecoPixelTracksSequence+process.hltIterL3MuonPixelTracksFromRegionalViaRegion)

process.HLTPrePixelTracksForIsoMu24 = cms.Sequence(process.HLTL2muonrecoSequence+process.HLTIterL3muonTkCandidateSequence)

## begin ->L1_Si_Mu->PrePixTrak->PixTraker->RecoPixTrakSeq->END

process.HLT_PixelTracks_IsoMu24_v0 = cms.Path(process.HLTBeginSequence+process.hltL1sSingleMu+process.HLTPrePixelTracksForIsoMu24+process.HLTIterL3MuonRecoPixelTracksSequence+process.HLTEndSequence)

process.HLT_PixelTracksFromGlobalViaRegion_IsoMu24_v0 = cms.Path(process.HLTBeginSequence+process.hltL1sSingleMu+process.HLTPrePixelTracksForIsoMu24+process.HLTRecoPixelTracksFromGlobalViaRegionSequenceForIterL3+process.HLTEndSequence)

process.HLT_PixelTracksFromRegionalViaRegion_IsoMu24_v0 = cms.Path(process.HLTBeginSequence+process.hltL1sSingleMu+process.HLTPrePixelTracksForIsoMu24+process.HLTRecoPixelTracksFromRegionalViaRegionSequenceForIterL3+process.HLTEndSequence)





# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.RawToDigi)
process.L1Reco_step = cms.Path(process.L1Reco)
process.reconstruction_step = cms.Path(process.reconstruction)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.RAWoutput_step = cms.EndPath(process.RAWoutput)

# Schedule definition

#process.schedule = cms.Schedule()
#process.schedule.extend(process.HLTSchedule)
#process.schedule.extend([process.raw2digi_step,process.L1Reco_step,process.reconstruction_step,process.endjob_step,process.RAWoutput_step])
#from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
#associatePatAlgosToolsTask(process)
#


process.schedule  = cms.Schedule(*[
                         process.HLTriggerFirstPath,
                         process.HLT_PixelTracks_IsoMu24_v0,
                         process.HLT_PixelTracksFromRegionalViaRegion_IsoMu24_v0,
                         process.HLT_PixelTracksFromGlobalViaRegion_IsoMu24_v0,
                         process.HLTriggerFinalPath,
			 process.dqm,
                         process.validation,
			 process.DQMOutput ])

process.schedule += cms.Schedule(*[  #  process.HLTriggerFinalPath,
				     #	 process.HLTAnalyzerEndpath,
				     #	 process.ScoutingPFOutput,
				     #	 process.ScoutingCaloMuonOutput,
                                     #   process.raw2digi_step,
				     #	 process.L1Reco_step,
				     #	 process.reconstruction_step,
				     	 process.endjob_step
				     #	 process.RAWoutput_step 
				]
					)



# customisation of the process.

# Automatic addition of the customisation function from HLTrigger.Configuration.CustomConfigs
from HLTrigger.Configuration.CustomConfigs import L1THLT 

#call to customisation function L1THLT imported from HLTrigger.Configuration.CustomConfigs
process = L1THLT(process)

# Automatic addition of the customisation function from HLTrigger.Configuration.customizeHLTforMC
from HLTrigger.Configuration.customizeHLTforMC import customizeHLTforMC 

#call to customisation function customizeHLTforMC imported from HLTrigger.Configuration.customizeHLTforMC
process = customizeHLTforMC(process)

# End of customisation functions

# Customisation from command line

process.CSCHaloData.HLTResultLabel=cms.InputTag('')
#Have logErrorHarvester wait for the same EDProducers to finish as those providing data for the OutputModule
from FWCore.Modules.logErrorHarvester_cff import customiseLogErrorHarvesterUsingOutputCommands
process = customiseLogErrorHarvesterUsingOutputCommands(process)

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
