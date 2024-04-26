import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

process = cms.Process("IN", eras.Phase2C17I13M9)
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.Geometry.GeometryExtended2026D88Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2026D88_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '125X_mcRun4_realistic_v2', '')

process.load('SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff')
process.load('CalibCalorimetry.CaloTPG.CaloTPGTranscoder_cfi')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('L1Trigger.TrackTrigger.TrackTrigger_cff')
process.load("L1Trigger.TrackFindingTracklet.L1HybridEmulationTracks_cff") 
process.load("L1Trigger.TrackerDTC.ProducerES_cff") 
process.load("L1Trigger.TrackerDTC.ProducerED_cff") 
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")

process.source = cms.Source("PoolSource",
    #fileNames = cms.untracked.vstring('/store/mc/Phase2Fall22DRMiniAOD/TT_TuneCP5_14TeV-powheg-pythia8/GEN-SIM-DIGI-RAW-MINIAOD/PU200_PUTP_125X_mcRun4_realistic_v2-v1/2540000/0bdf9cbd-a830-43d7-a3eb-5ae027d22be5.root'),
    fileNames  = cms.untracked.vstring('file:/afs/cern.ch/work/a/athachay/private/phase2/hf_work/emulation/TT_TuneCP5_14TeV-powheg-pythia8_GEN-SIM-DIGI-RAW-MINIAOD_PU200_PUTP_125X_mcRun4_realistic_v2-v1.root'),
    inputCommands = cms.untracked.vstring(
        'keep *',
        'drop l1tPFJets_*_*_*',
        'drop l1tPFTaus_*_*_*',
        'drop l1tTrackerMuons_*_*_*'
    ),
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(4))
process.options = cms.untracked.PSet( 
        wantSummary = cms.untracked.bool(True),
        numberOfThreads = cms.untracked.uint32(1),
        numberOfStreams = cms.untracked.uint32(1),
)

process.PFInputsTask = cms.Task(
   process.L1TLayer1TaskInputsTask,
   process.offlineBeamSpot,                                           #
   process.SimL1EmulatorTask,                                         #
)
#process.l1tLayer1.pfProducers=cms.VInputTag(cms.InputTag("l1tLayer1HF"))
process.p=cms.Path( 
        process.l1tLayer1HF
)

from L1Trigger.Phase2L1ParticleFlow.l1ctLayer1_patternWriters_cff import *
process.l1tLayer1HF.patternWriters = cms.untracked.VPSet(*hfWriterConfigs)
process.l1tLayer1HF.puAlgoParameters.debug = True 
## Calo sector defenition
#process.l1tLayer1HF.caloSectors.clear()
#process.l1tLayer1HF.caloSectors.append(
#        cms.PSet(
#            etaBoundaries = cms.vdouble(-5.5, -3.0),
#            phiSlices = cms.uint32(6),
#            phiZero = cms.double(  3.141592653589793/18  )  # offset of first wedge in HF geometry is -10 degree
#        ))
#
#process.l1tLayer1HF.caloSectors.append(
#        cms.PSet(
#            etaBoundaries = cms.vdouble(3.0, 5.5),
#            phiSlices = cms.uint32(6),
#            phiZero = cms.double(  3.141592653589793/18 ) ,
#            phiExtra = cms.double( 3.141592653589793*2/18  )
#        )
#    )
## regionizer defenition 
#process.l1tLayer1HF.regions= cms.VPSet(
#        cms.PSet( 
#            etaBoundaries = cms.vdouble(-5.5, -3.0),
#            phiSlices     = cms.uint32(6),
#            phiZero = cms.double(  3.141592653589793/18  ), # offset of first wedge in HF geometry  is -10 degree
#            etaExtra = cms.double(0.25),
#            phiExtra = cms.double( 3.141592653589793*2/18  ) # dPhi = 0.34 , motivation from Ak4 jet radius [ also strict cutoff of 0.3 in current puppi implementation ] 
#        ),
#        cms.PSet( 
#            etaBoundaries = cms.vdouble(+3.0, +5.5),
#            phiSlices     = cms.uint32(6),
#            phiZero = cms.double(  3.141592653589793/18  ), # offset of first wedge in HF geometry  is -10 degree
#            etaExtra = cms.double(0.25),
#            phiExtra = cms.double( 3.141592653589793*2/18  ) # dPhi = 0.34 , motivation from Ak4 jet radius [ also strict cutoff of 0.3 in current puppi implementation ] 
#        )
#    )
# updting the ptter file writer
#process.l1tLayer1HF.patternWriters = cms.untracked.VPSet(
#        cms.PSet(
#            eventsPerFile = cms.uint32(12),
#            fileFormat = cms.string('EMPv2'),
#            maxLinesPerOutputFile = cms.uint32(1024),
#            nOutputFramesPerBX = cms.uint32(9),
#            outputFileExtension = cms.string('txt.gz'),
#            outputFileName = cms.string('l1HFNeg-outputs'),
#            outputLinksPuppi = cms.vuint32(0,1, 2,3,4,5),
#            outputRegions = cms.vuint32(
#                0, 1, 2, 3, 4, 5
#            ),
#            partition = cms.string('HF'),
#            tmuxFactor = cms.uint32(1)
#        ),
#        cms.PSet(
#            eventsPerFile = cms.uint32(12),
#            fileFormat = cms.string('EMPv2'),
#            maxLinesPerOutputFile = cms.uint32(1024),
#            nOutputFramesPerBX = cms.uint32(9),
#            outputFileExtension = cms.string('txt.gz'),
#            outputFileName = cms.string('l1HFPos-outputs'),
#            outputLinksPuppi = cms.vuint32(0, 1, 2, 3, 4, 5),
#            outputRegions = cms.vuint32(
#                6, 7, 8, 9, 10, 11
#            ),
#            partition = cms.string('HF'),
#            tmuxFactor = cms.uint32(1)
#        )
#    )


process.p.associate(process.PFInputsTask)
process.schedule = cms.Schedule([process.p])
