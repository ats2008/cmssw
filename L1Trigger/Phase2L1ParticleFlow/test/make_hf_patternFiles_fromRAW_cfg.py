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
        #numberOfThreads = cms.untracked.uint32(4),
        #numberOfStreams = cms.untracked.uint32(4),
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


process.p.associate(process.PFInputsTask)

process.schedule = cms.Schedule([process.p])

