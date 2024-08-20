from __future__ import print_function
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
import os

options = VarParsing.VarParsing ('analysis')
options.register ('runNumber',
                  37,
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "Run Number")

options.register ('lumiNumber',
                  1,
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "Run Number")

options.register ('daqSourceMode',
                  'ScoutingPhase2', # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "DAQ source data mode")

options.register ('buBaseDir',
                  '/dev/shm/ramdisk', # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "BU base directory")

options.register ('fuBaseDir',
                  '/dev/shm/data', # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "BU base directory")

options.register ('fffBaseDir',
                  '/dev/shm', # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "FFF base directory")

options.register ('numThreads',
                  1, # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "Number of CMSSW threads")

options.register ('numFwkStreams',
                  1, # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "Number of CMSSW streams")

options.register ('puppiMode',
                  'simple', # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "puppi mode to run (simple, struct, soa)")
                 
options.register ('outMode',
                  'none', # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "puppi mode to run (none, struct, soa)")
                   
options.register ('outFile',
                  "NanoOutput.root",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "Sub lumisection number to process")


options.parseArguments()
if options.puppiMode not in ("simple", "sparse", "struct", "sparseStruct", "soa", "all", "fast"):
    raise RuntimeError("Unsupported puppiMode %r" %options.puppiMode)

cmsswbase = os.path.expandvars("$CMSSW_BASE/")

process = cms.Process("SCPU")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.maxEvents)
)

process.options = cms.untracked.PSet(
    numberOfThreads = cms.untracked.uint32(options.numThreads),
    numberOfStreams = cms.untracked.uint32(options.numFwkStreams),
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1),
    wantSummary = cms.untracked.bool(True)
)
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100

process.EvFDaqDirector = cms.Service("EvFDaqDirector",
    useFileBroker = cms.untracked.bool(False),
    fileBrokerHostFromCfg = cms.untracked.bool(True),
    fileBrokerHost = cms.untracked.string("htcp40.cern.ch"),
    runNumber = cms.untracked.uint32(options.runNumber),
    baseDir = cms.untracked.string(options.fuBaseDir),
    buBaseDir = cms.untracked.string(options.buBaseDir),
    buBaseDirsAll = cms.untracked.vstring(options.buBaseDir,),
    buBaseDirsNumStreams = cms.untracked.vint32(0),
    directorIsBU = cms.untracked.bool(False),
)

fuDir = options.fuBaseDir+("/run%06d" % options.runNumber)
buDir = options.buBaseDir+("/run%06d" % options.runNumber)
for d in fuDir, buDir, options.fuBaseDir, options.buBaseDir:
  if not os.path.isdir(d):
    os.makedirs(d)

process.source = cms.Source("DAQSource",
    testing = cms.untracked.bool(True),
    dataMode = cms.untracked.string(options.daqSourceMode),
    verifyChecksum = cms.untracked.bool(True),
    useL1EventID = cms.untracked.bool(False),
    eventChunkBlock = cms.untracked.uint32(2 * 1024),
    eventChunkSize = cms.untracked.uint32(2 * 1024),
    maxChunkSize = cms.untracked.uint32(4 * 1024),
    numBuffers = cms.untracked.uint32(4),
    maxBufferedFiles = cms.untracked.uint32(4),
    fileListMode = cms.untracked.bool(True),
    fileNames = cms.untracked.vstring(
        buDir + "/" + "run%06d_ls%04d_index%06d_ts00.raw" % (options.runNumber, options.lumiNumber, 1)
    )
)
os.system("touch " + buDir + "/" + "fu.lock")

## test pluging
scPhase2MuonRawToDigi = cms.EDProducer('ScPhase2MuonRawToDigi',
  src = cms.InputTag('rawDataCollector'),
  fedIDs = cms.vuint32(0),
  runCandidateUnpacker = cms.bool(False),
  runStructUnpacker = cms.bool(False),
  runSOAUnpacker = cms.bool(False),
)
process.scPhase2MuonRawToDigiCandidate = scPhase2MuonRawToDigi.clone(
    runCandidateUnpacker = True
)
process.scPhase2MuonRawToDigiStruct = scPhase2MuonRawToDigi.clone(
    runCandidateUnpacker = False,
    runStructUnpacker = True
)
process.scPhase2MuonRawToDigiSOA = scPhase2MuonRawToDigi.clone(
    runCandidateUnpacker = False,
    runSOAUnpacker = True
)
#process.dimuCandidate = cms.EDProducer("ScPhase2MuonDiMuDemo",
#    src = cms.InputTag("scPhase2MuonRawToDigiCandidate"),
#    runCandidate = cms.bool(True),
#    runStruct = cms.bool(False),
#    runSOA = cms.bool(False)
#)
process.dimuStruct = cms.EDProducer("ScPhase2MuonDiMuDemo",
    src = cms.InputTag("scPhase2MuonRawToDigiStruct"),
    runCandidate = cms.bool(False),
    runStruct = cms.bool(True),
    runSOA = cms.bool(False)
)
#process.dimuSOA = cms.EDProducer("ScPhase2MuonDiMuDemo",
#    src = cms.InputTag("scPhase2MuonRawToDigiSOA"),
#    runCandidate = cms.bool(False),
#    runStruct = cms.bool(False),
#    runSOA = cms.bool(True)
#)
process.scPhase2MuonStructToTable = cms.EDProducer("ScMuonToOrbitFlatTable",
    src = cms.InputTag("scPhase2MuonRawToDigiStruct"),
    name = cms.string("L1Muon"),
    doc = cms.string("L1Muon candidates from Correlator Layer 2"),
)
process.p_simple = cms.Path(
  process.scPhase2MuonRawToDigiCandidate
#  +process.dimuCandidate
)
process.p_struct = cms.Path(
  process.scPhase2MuonRawToDigiStruct
  +process.dimuStruct
  +process.scPhase2MuonStructToTable
)
process.p_soa = cms.Path(
  process.scPhase2MuonRawToDigiSOA
#  +process.dimuSOA
)
process.p_all = cms.Path(
  process.scPhase2MuonRawToDigiCandidate+
  process.scPhase2MuonRawToDigiStruct+
  process.scPhase2MuonRawToDigiSOA+
  process.scPhase2MuonStructToTable+
#  process.dimuCandidate+
  process.dimuStruct
#  process.dimuSOA
)
process.p_fast = cms.Path(
  process.scPhase2MuonRawToDigiStruct+
  process.scPhase2MuonRawToDigiSOA+
  process.scPhase2MuonStructToTable
#  process.dimuStruct+
#  process.dimuSOA
)
process.scPhase2MuonStructNanoAll = cms.OutputModule("OrbitNanoAODOutputModule",
    fileName = cms.untracked.string(options.outFile),
    outputCommands = cms.untracked.vstring("drop *", "keep l1ScoutingRun3OrbitFlatTable_scPhase2MuonStructToTable_*_*"),
    compressionLevel = cms.untracked.int32(4),
    compressionAlgorithm = cms.untracked.string("LZ4"),
)
process.scPhase2MuonStructNanoW3pi = cms.OutputModule("OrbitNanoAODOutputModule",
    fileName = cms.untracked.string(options.outFile.replace(".root","")+".dimu.root"),
    selectedBx = cms.InputTag("dimuStruct","selectedBx"),
    outputCommands = cms.untracked.vstring("drop *", 
        "keep l1ScoutingRun3OrbitFlatTable_scPhase2MuonStructToTable_*_*",
        "keep l1ScoutingRun3OrbitFlatTable_dimuStruct_*_*"
        ),
    compressionLevel = cms.untracked.int32(4),
    compressionAlgorithm = cms.untracked.string("LZ4"),
)
process.o_structAll = cms.EndPath( process.scPhase2MuonStructNanoAll )
process.o_structDiMu = cms.EndPath( process.scPhase2MuonStructNanoW3pi )

sched = [ getattr(process, "p_"+options.puppiMode) ]
if options.outMode != "none":
  sched.append(getattr(process, "o_"+options.outMode))
process.schedule = cms.Schedule(*sched)

