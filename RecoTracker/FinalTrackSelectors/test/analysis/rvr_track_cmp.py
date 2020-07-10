import DataFormats.FWLite as fwlite
from ROOT import TFile, TTree, TH1D, TCanvas
from array import array
import numpy as np

events = fwlite.Events("file:TrkSelByRegion_validationRAW.root")

mask_ = fwlite.Handle("std::vector<bool>") 
track_RvR_ = fwlite.Handle("std::vector<reco::Track>")
track_RegionalReco_ =fwlite.Handle("std::vector<reco::Track>")
#track_GlobalReco_ = fwlite.Handle("std::vector<reco::Track>")
track_ndel_eta_=fwlite.Handle("std::vector<float>")
track_ndel_phi_=fwlite.Handle("std::vector<float>")

afile=TFile("TrkSelectorValidation.root","RECREATE")
tree_RvR=TTree("tracks_RvR","Tracks_FromRegionalViaRegion")
tree_RegionalReco=TTree("track_RegReco","Track_RegionalReco")
#tree_GlobalReco=TTree("track_GlobalReco","track_GlobalReco")

qpt  = array('d',[0])
eta  = array('d',[0])
phi  = array('d',[0])
dxy  = array('d',[0])
dz  = array('d',[0])
evtId=array('i',[0])
nRegions=array('I',[0])
ndelleta=np.zeros(200)
ndellphi=np.zeros(200)

tree_RvR.Branch("qpt",qpt,"qpt/D")
tree_RvR.Branch("eta",eta,"eta/D")
tree_RvR.Branch("phi",phi,"phi/D")
tree_RvR.Branch("dxy",dxy,"dxy/D")
tree_RvR.Branch("dz",dz,"dz/D")
tree_RvR.Branch("event_id",evtId,"event_id/i")
tree_RvR.Branch( 'nRegions', nRegions, 'nRegions/I' )
tree_RvR.Branch( 'ndelleta', ndelleta, 'ndelleta[nRegions]/D')
tree_RvR.Branch( 'ndellphi', ndellphi, 'ndellphi[nRegions]/D')


tree_RegionalReco.Branch("qpt",qpt,"qpt/D")
tree_RegionalReco.Branch("eta",eta,"eta/D")
tree_RegionalReco.Branch("phi",phi,"phi/D")
tree_RegionalReco.Branch("dxy",dxy,"dxy/D")
tree_RegionalReco.Branch("dz",dz,"dz/D")
tree_RegionalReco.Branch("event_id",evtId,"event_id/i")
tree_RegionalReco.Branch( 'nRegions', nRegions, 'nRegions/I' )
tree_RegionalReco.Branch( 'ndelleta', ndelleta, 'ndelleta[nRegions]/D')
tree_RegionalReco.Branch( 'ndellphi', ndellphi, 'ndellphi[nRegions]/D')


ptRegionalReco_hist = TH1D("pt_RegionalReco","pt for RegionalvReco",50,0,50.)
ptRvR_hist = TH1D("pt_RvR","pt for RvR ",50,0.,50.)
ptMissed_hist = TH1D("pt_Miss_RvR","pt for tracks not in RvR",50,0,50.)
delleta_RegionalReco_hist = TH1D("NDellEta_RegionalReco","Normalized #delta#eta (wrt. width of region) from RegionalReco",50,-2.5,2.5)
dellphi_RegionalReco_hist = TH1D("NDellPhi_RegionalReco","Normalized #delta#phi (wrt. width of region) from RegionalReco",50,-5.,5.)
delleta_RvR_hist = TH1D("NDellEta_RvR","Normalized #delta#eta(wrt. width of region) from RvR",50,-2.5,2.5)
dellphi_RvR_hist = TH1D("NDellPhi_RvR","Normalized #delta#phi(wrt. width of region) from RvR",50,-5.,5.)


print "Total Number of Events = ",events.size()

true_trk_count=0
RvR_trk_count=0

for i, event in enumerate(events):
    
    if i%250==0:
        print "At event : ",i
    event.getByLabel('hltIterL3MuonPixelTracksFromRegionalViaRegion',mask_)

    if not mask_.isValid():
        continue
    mask=mask_.product()
    n_tracks=mask.size()
    if n_tracks<1:
        continue
    event.getByLabel('hltIterL3MuonPixelTracksFromRegionalViaRegion',track_RvR_)
    event.getByLabel('hltIterL3MuonPixelTracks',track_RegionalReco_)
    event.getByLabel('hltIterL3MuonPixelTracksFromRegionalViaRegion','normDeltaEta',track_ndel_eta_)
    event.getByLabel('hltIterL3MuonPixelTracksFromRegionalViaRegion','normDeltaPhi',track_ndel_phi_)

    track_RvR=track_RvR_.product()
    track_RegionalReco=track_RegionalReco_.product()
    track_ndel_eta=track_ndel_eta_.product()
    track_ndel_phi=track_ndel_phi_.product()
    true_trk_count+=n_tracks
    RvR_trk_count+=track_RvR.size()
#   track_GlobalReco=event.getByLabel('hltIterL3MuonPixelTracksFromGlobalViaRegion',mask_)
#    print "Event", i
    nRegions[0]=track_ndel_eta.size()/n_tracks
    idx_RvR=0
    dell_idx=0
    for idx in range(n_tracks):
        ptRegionalReco_hist.Fill(track_RegionalReco[idx].pt())       
        qpt[0]=track_RegionalReco[idx].charge()/track_RegionalReco[idx].pt()
        phi[0]=track_RegionalReco[idx].phi()
        eta[0]=track_RegionalReco[idx].eta()
        dxy[0]=track_RegionalReco[idx].dxy()
        dz[0] =track_RegionalReco[idx].dz()
        
        for j in range(nRegions[0]):
            ndelleta[j]=track_ndel_eta[dell_idx]
            ndellphi[j]=track_ndel_phi[dell_idx]
            delleta_RegionalReco_hist.Fill(ndelleta[j])
            dellphi_RegionalReco_hist.Fill(ndellphi[j])
            dell_idx+=1
        evtId[0]=i
        tree_RegionalReco.Fill()
        if mask[idx]:
            ptRvR_hist.Fill(track_RvR[idx_RvR].pt())
            for j in range(nRegions[0]):
                delleta_RvR_hist.Fill(ndelleta[j])
                dellphi_RvR_hist.Fill(ndellphi[j])
 
            qpt[0]=track_RvR[idx_RvR].charge()/track_RegionalReco[idx].pt()
            phi[0]=track_RvR[idx_RvR].phi()
            eta[0]=track_RvR[idx_RvR].eta()
            dxy[0]=track_RvR[idx_RvR].dxy()
            dz[0] =track_RvR[idx_RvR].dz()
            tree_RvR.Fill()
            idx_RvR+=1
        else:
            ptMissed_hist.Fill(track_RegionalReco[idx].pt())                  
#        print "\t phi: %.3f" %track.phi(),
#        print "\t eta: %.3f" %track.eta(),
#        print "\t dxy: %.4f" %track.dxy(),
#        int "\t dz: %.4f"  %track.dz()
 

print "true tracks ",true_trk_count
print "selcted tracks ", RvR_trk_count

afile.cd()
ptMissed_hist.Write()
ptRegionalReco_hist.Write()
ptRvR_hist.Write()
delleta_RegionalReco_hist.Write()
dellphi_RegionalReco_hist.Write()
delleta_RvR_hist.Write()
dellphi_RvR_hist.Write()

tree_RegionalReco.Write()
tree_RvR.Write()

acanvas=TCanvas("acanvas","acanvas")
ptMissed_hist.Draw()
acanvas.SaveAs("ptMissed_RvR.png")
ptRegionalReco_hist.Draw()
acanvas.SaveAs("pt_RegionalReco.png")
ptRvR_hist.Draw()
acanvas.SaveAs("pt_RvR.png")

delleta_RegionalReco_hist.Draw()
acanvas.SaveAs("nDellEta_RegionalReco.png")
dellphi_RegionalReco_hist.Draw()
acanvas.SaveAs("nDellPhi_RegionalReco.png")
delleta_RvR_hist.Draw()
acanvas.SaveAs("nDellEta_RvR.png")
dellphi_RvR_hist.Draw()
acanvas.SaveAs("nDellPhi_RvR.png")

