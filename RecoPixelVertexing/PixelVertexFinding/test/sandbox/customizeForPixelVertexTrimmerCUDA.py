import FWCore.ParameterSet.Config as cms
from HLTrigger.Configuration.common import insert_modules_after
def customizeForValidation(process):
#  desc.add<bool>("onGPU", true);
#  desc.add<edm::InputTag>("pixelTrackSrc", edm::InputTag("caHitNtupletCUDA"));
#  desc.add<edm::InputTag>("pixelVertexSrc");
#  desc.add<unsigned int>("maxVtx", 100)->setComment("max output collection size (number of accepted vertices)");
#  desc.add<double>("fractionSumPt2", 0.3)->setComment("threshold on sumPt2 fraction of the leading vertex");
#  desc.add<double>("minSumPt2", 0.)->setComment("min sumPt2");
#  desc.add<double>("track_pt_min", 1.0)->setComment("min track_pt");
#  desc.add<double>("track_pt_max", 10.0)->setComment("max track_pt");
#  desc.add<double>("track_chi2_max", 99999.0)->setComment("max track_chi2");
#  desc.add<double>("track_prob_min", 1.0)->setComment("min track_prob");

    process.hltTrimmedPixelVerticesCUDAValidation= cms.EDProducer("PixelVertexCollectionTrimmerCUDA",
        onGPU = cms.bool(True),
        fractionSumPt2 = cms.double(0.3),
        maxVtx = cms.uint32(100),
        minSumPt2 = cms.double(0.0),
        pixelVertexSrc = cms.InputTag("hltPixelVerticesCUDA"),
        pixelTrackSrc = cms.InputTag("hltPixelTracksHitQuadruplets"),
        minNumberOfHits = cms.int32(0),
        track_pT_min = cms.double(1.0),
        track_pT_max =  cms.double(20.0),
        track_chi2_max =  cms.double(20.0),
        track_prob_min =  cms.double(-1.0)
   )

    process.hltTrimmedPixelVerticesSoAValidation = cms.EDProducer("PixelVertexSoAFromCUDA",
        src = cms.InputTag("hltTrimmedPixelVerticesCUDAValidation")
        )
    
    process.hltTrimmedPixelVerticesValidation = cms.EDProducer("PixelVertexProducerFromSoA",
        TrackCollection = cms.InputTag("hltPixelTracks"),
        beamSpot = cms.InputTag("hltOnlineBeamSpot"),
        src = cms.InputTag("hltTrimmedPixelVerticesSoAValidation")
       )

    process.hltTrimmingCUDAValidationSequece= cms.Sequence(
            process.hltTrimmedPixelVerticesCUDAValidation +
            process.hltTrimmedPixelVerticesSoAValidation +
            process.hltTrimmedPixelVerticesValidation 
         )
    
    insert_modules_after(process,process.hltTrimmedPixelVertices,process.hltTrimmingCUDAValidationSequece)
 

#    process.hltTrimmedPixelVerticesCPUSoAValidation= cms.EDProducer("PixelVertexSoATrimmer",
#        beamSpot = cms.InputTag("hltOnlineBeamSpot"),
#        fractionSumPt2 = cms.double(0.3),
#        maxVtx = cms.uint32(100),
#        minSumPt2 = cms.double(0.0),
#        src = cms.InputTag("hltPixelVerticesSoA"),
#        trackSrc = cms.InputTag("hltPixelTracksSoA"),
#        minNumberOfHits = cms.int32(0),
#        track_pt_min = cms.double(1.0),
#        track_pt_max =  cms.double(20.0),
#        track_chi2_max =  cms.double(20.0),
#        track_prob_min =  cms.double(-1.0)
#   )
 
    process.hltTrimmedPixelVerticesCPUSoAValidation= cms.EDProducer("PixelVertexCollectionTrimmerCUDA",
        onGPU= cms.bool(False),
        fractionSumPt2 = cms.double(0.3),
        maxVtx = cms.uint32(100),
        minSumPt2 = cms.double(0.0),
        pixelVertexSrc = cms.InputTag("hltPixelVerticesSoA"),
        pixelTrackSrc = cms.InputTag("hltPixelTracksSoA"),
        minNumberOfHits = cms.int32(0),
        track_pT_min = cms.double(1.0),
        track_pT_max =  cms.double(20.0),
        track_chi2_max =  cms.double(20.0),
        track_prob_min =  cms.double(-1.0)
   )
   
    process.hltTrimmedPixelVerticesCPUValidation = cms.EDProducer("PixelVertexProducerFromSoA",
        TrackCollection = cms.InputTag("hltPixelTracks"),
        beamSpot = cms.InputTag("hltOnlineBeamSpot"),
        src = cms.InputTag("hltTrimmedPixelVerticesCPUSoAValidation")
       )
    process.hltTrimmingCPUValidationSequece= cms.Sequence(
            process.hltTrimmedPixelVerticesCPUSoAValidation +
            process.hltTrimmedPixelVerticesCPUValidation 
         )
    
    insert_modules_after(process,process.hltTrimmedPixelVertices,process.hltTrimmingCPUValidationSequece)
 

    process.hltTrimmingCUDAValidation = cms.OutputModule( "PoolOutputModule",
            fileName = cms.untracked.string("RawCUDAVertexValidation.root"),
            fastCloning = cms.untracked.bool( False ),
            dataset = cms.untracked.PSet(
                      filterName = cms.untracked.string( "" ),
                      dataTier = cms.untracked.string( "RAW" )
                     ),
            outputCommands = cms.untracked.vstring(
               'keep *_hltPixelTracks_*_*',
               'keep *_hltPixelVertices_*_*',
               'keep *_hltTrimmedPixelVertices_*_*',
               'keep *_hltTrimmedPixelVerticesValidation_*_*',
               'keep *_hltTrimmedPixelVerticesCPUValidation_*_*',
              )
            )
    process.ValidationOutput = cms.EndPath(process.hltTrimmingCUDAValidation)
    process.schedule.extend([process.ValidationOutput])

    return process
