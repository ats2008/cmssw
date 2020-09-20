# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: RelVal --step=HLT:GRun --conditions=auto:run3_mc_GRun --filein=file:RelVal_Raw_GRun_MC.root --custom_conditions= --fileout=RelVal_HLT_GRun_MC.root --number=100 --mc --no_exec --datatier SIM-DIGI-RAW-HLTDEBUG --eventcontent=FEVTDEBUGHLT --customise=HLTrigger/Configuration/CustomConfigs.L1THLT --era=Run3 --customise= --scenario=pp --python_filename=RelVal_HLT_GRun_MC.py --processName=HLT
import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_cff import Run3

process = cms.Process('HLTreco',Run3)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('HLTrigger.Configuration.HLT_GRun_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000),
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
       'root://cms-xrd-global.cern.ch//store/relval/CMSSW_11_2_0_pre3/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/112X_mcRun3_2021_realistic_v5-v1/20000/23764A8E-FB0D-9A49-B2AB-70C4082259AE.root',
       'root://cms-xrd-global.cern.ch//store/relval/CMSSW_11_2_0_pre3/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/112X_mcRun3_2021_realistic_v5-v1/20000/79E1D875-0C92-1F46-B246-A6BCECC50065.root',
       'root://cms-xrd-global.cern.ch//store/relval/CMSSW_11_2_0_pre3/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/112X_mcRun3_2021_realistic_v5-v1/20000/888D859A-C50E-3A44-8F97-6E9EB5F65F8B.root',
       'root://cms-xrd-global.cern.ch//store/relval/CMSSW_11_2_0_pre3/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/112X_mcRun3_2021_realistic_v5-v1/20000/8D548F7C-3A0D-DB45-AA79-21012550969A.root',
       'root://cms-xrd-global.cern.ch//store/relval/CMSSW_11_2_0_pre3/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/112X_mcRun3_2021_realistic_v5-v1/20000/9301300E-BB52-AB4B-87B9-5C1A103D638A.root',
       'root://cms-xrd-global.cern.ch//store/relval/CMSSW_11_2_0_pre3/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/112X_mcRun3_2021_realistic_v5-v1/20000/A5854CAD-2254-7E41-80ED-CC2DB65D344A.root',
       'root://cms-xrd-global.cern.ch//store/relval/CMSSW_11_2_0_pre3/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/112X_mcRun3_2021_realistic_v5-v1/20000/A9166772-F2EC-6044-B09F-F18D2A28D886.root',
       'root://cms-xrd-global.cern.ch//store/relval/CMSSW_11_2_0_pre3/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/112X_mcRun3_2021_realistic_v5-v1/20000/AF1F02D9-1BEB-8849-8FC8-F9C9027702D4.root',
       'root://cms-xrd-global.cern.ch//store/relval/CMSSW_11_2_0_pre3/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/112X_mcRun3_2021_realistic_v5-v1/20000/B210FDB0-C575-8941-A6AB-171FEF417FAA.root',
       'root://cms-xrd-global.cern.ch//store/relval/CMSSW_11_2_0_pre3/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/112X_mcRun3_2021_realistic_v5-v1/20000/FDC1ABA2-EC58-FB4E-976A-671224125594.root'
        ),
    secondaryFileNames = cms.untracked.vstring()
)
#process.source = cms.Source("PoolSource",
#    fileNames = cms.untracked.vstring(
#       'root://eoscms.cern.ch//eos/cms/store/relval/CMSSW_11_2_0_pre3/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/112X_mcRun3_2021_realistic_v5-v1/20000/23764A8E-FB0D-9A49-B2AB-70C4082259AE.root',
#       'root://eoscms.cern.ch//eos/cms/store/relval/CMSSW_11_2_0_pre3/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/112X_mcRun3_2021_realistic_v5-v1/20000/79E1D875-0C92-1F46-B246-A6BCECC50065.root',
#       'root://eoscms.cern.ch//eos/cms/store/relval/CMSSW_11_2_0_pre3/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/112X_mcRun3_2021_realistic_v5-v1/20000/888D859A-C50E-3A44-8F97-6E9EB5F65F8B.root',
#       'root://eoscms.cern.ch//eos/cms/store/relval/CMSSW_11_2_0_pre3/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/112X_mcRun3_2021_realistic_v5-v1/20000/8D548F7C-3A0D-DB45-AA79-21012550969A.root',
#       'root://eoscms.cern.ch//eos/cms/store/relval/CMSSW_11_2_0_pre3/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/112X_mcRun3_2021_realistic_v5-v1/20000/9301300E-BB52-AB4B-87B9-5C1A103D638A.root',
#       'root://eoscms.cern.ch//eos/cms/store/relval/CMSSW_11_2_0_pre3/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/112X_mcRun3_2021_realistic_v5-v1/20000/A5854CAD-2254-7E41-80ED-CC2DB65D344A.root',
#       'root://eoscms.cern.ch//eos/cms/store/relval/CMSSW_11_2_0_pre3/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/112X_mcRun3_2021_realistic_v5-v1/20000/A9166772-F2EC-6044-B09F-F18D2A28D886.root',
#       'root://eoscms.cern.ch//eos/cms/store/relval/CMSSW_11_2_0_pre3/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/112X_mcRun3_2021_realistic_v5-v1/20000/AF1F02D9-1BEB-8849-8FC8-F9C9027702D4.root',
#       'root://eoscms.cern.ch//eos/cms/store/relval/CMSSW_11_2_0_pre3/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/112X_mcRun3_2021_realistic_v5-v1/20000/B210FDB0-C575-8941-A6AB-171FEF417FAA.root',
#       'root://eoscms.cern.ch//eos/cms/store/relval/CMSSW_11_2_0_pre3/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/112X_mcRun3_2021_realistic_v5-v1/20000/FDC1ABA2-EC58-FB4E-976A-671224125594.root'
#        ),
#    secondaryFileNames = cms.untracked.vstring()
#)

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
    numberOfStreams = cms.untracked.uint32(1),
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

process.FEVTDEBUGHLToutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('SIM-DIGI-RAW-HLTDEBUG'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('RelVal_HLT_GRun_MC.root'),
    outputCommands = process.FEVTDEBUGHLTEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run3_mc_GRun', '')

# Path and EndPath definitions
process.endjob_step = cms.EndPath(process.endOfProcess)
#process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)
process.FEVTDEBUGHLToutput_step = cms.EndPath()

# Schedule definition
process.schedule = cms.Schedule()
process.schedule.extend(process.HLTSchedule)
process.schedule.extend([process.endjob_step,process.FEVTDEBUGHLToutput_step])
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

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

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion

from FWCore.ParameterSet.Utilities import *
from HLTrigger.Configuration.common import *

replace_with(process.AlCa_LumiPixelsCounts_ZeroBias_v1,cms.Path())
replace_with(process.AlCa_LumiPixelsCounts_Random_v1,cms.Path())

# Patatrack Customization    
from customizeHLTforPatatrack import customise_for_Patatrack_on_gpu
process = customise_for_Patatrack_on_gpu(process)
process.hltPixelTracksHitQuadruplets.idealConditions = True

#process.MessageLogger.cerr.FwkReport.reportEvery = 20

#from HLTrigger.Configuration.customizeHLTforPatatrack import customise_for_Patatrack_on_cpu
#process = customise_for_Patatrack_on_cpu(process)


from customizeForPixelVertexTrimmerCUDA import *
customizeForValidation(process)



process.options.numberOfStreams = cms.untracked.uint32(1)
process.options.numberOfThreads = cms.untracked.uint32(1)
   
