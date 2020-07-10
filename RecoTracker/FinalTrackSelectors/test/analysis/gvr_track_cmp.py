import DataFormats.FWLite as fwlite
from ROOT import TFile, TTree, TH1D, TCanvas
from array import array
import numpy as np

events = fwlite.Events("file:../TrkSelByRegion_validationRAW.root")

mask_ = fwlite.Handle("std::vector<bool>") 
track_GvR_ = fwlite.Handle("std::vector<reco::Track>")
track_GlobalReco_ =fwlite.Handle("std::vector<reco::Track>")
#track_GlobalReco_ = fwlite.Handle("std::vector<reco::Track>")
track_ndel_eta_=fwlite.Handle("std::vector<float>")
track_ndel_phi_=fwlite.Handle("std::vector<float>")

afile=TFile("TrkSelectorValidation.root","UPDATE")
tree_GvR=TTree("tracks_GvR","Tracks_GvR")
tree_GlobalReco=TTree("track_GlobalReco","Track_GlobalReco")
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

tree_GvR.Branch("qpt",qpt,"qpt/D")
tree_GvR.Branch("eta",eta,"eta/D")
tree_GvR.Branch("phi",phi,"phi/D")
tree_GvR.Branch("dxy",dxy,"dxy/D")
tree_GvR.Branch("dz",dz,"dz/D")
tree_GvR.Branch("event_id",evtId,"event_id/i")
tree_GvR.Branch( 'nRegions', nRegions, 'nRegions/I' )
tree_GvR.Branch( 'ndelleta', ndelleta, 'ndelleta[nRegions]/D')
tree_GvR.Branch( 'ndellphi', ndellphi, 'ndellphi[nRegions]/D')


tree_GlobalReco.Branch("qpt",qpt,"qpt/D")
tree_GlobalReco.Branch("eta",eta,"eta/D")
tree_GlobalReco.Branch("phi",phi,"phi/D")
tree_GlobalReco.Branch("dxy",dxy,"dxy/D")
tree_GlobalReco.Branch("dz",dz,"dz/D")
tree_GlobalReco.Branch("event_id",evtId,"event_id/i")
tree_GlobalReco.Branch( 'nRegions', nRegions, 'nRegions/I' )
tree_GlobalReco.Branch( 'ndelleta', ndelleta, 'ndelleta[nRegions]/D')
tree_GlobalReco.Branch( 'ndellphi', ndellphi, 'ndellphi[nRegions]/D')


ptGlobalReco_hist = TH1D("pt_GlobalReco","pt for tracks from GlobalReco",50,0.,50)
ptGvR_hist = TH1D("pt_GvR","pt of Tracks rom  GvR",50,0.,50.)
ptMissed_hist = TH1D("pt_Miss_GvR","pt of Miss tracks from GvR",50,0.,50.)
delleta_GlobalReco_hist = TH1D("NDellEta_GlobalReco","Normalized #delta#eta(wrt. width of region) from GlobalReco",50,-2.5,2.5)
dellphi_GlobalReco_hist = TH1D("NDellPhi_GlobalReco","Normalized #delta#phi(wrt. width of region) from GlobalReco",50,-5.,5.)
delleta_GvR_hist = TH1D("NDellEta_RvR","Normalized #delta#eta(wrt. width of region) from GvR",50,-2.5,2.5)
dellphi_GvR_hist = TH1D("NDellPhi_RvR","Normalized #delta#phi(wrt. width of region) from GvR",50,-5.,5.)


print "Total Number of Events = ",events.size()

true_trk_count=0
GvR_trk_count=0

for i, event in enumerate(events):
    
    if i%250==0:
        print "At event : ",i
    event.getByLabel('hltIterL3MuonPixelTracksFromGlobalViaRegion',mask_)

    if not mask_.isValid():
        continue
    mask=mask_.product()
    n_tracks=mask.size()
    if n_tracks<1:
        continue
    event.getByLabel('hltIterL3MuonPixelTracksFromGlobalViaRegion',track_GvR_)
    event.getByLabel('hltPixelTracks',track_GlobalReco_)
    event.getByLabel('hltIterL3MuonPixelTracksFromGlobalViaRegion','normDeltaEta',track_ndel_eta_)
    event.getByLabel('hltIterL3MuonPixelTracksFromGlobalViaRegion','normDeltaPhi',track_ndel_phi_)

    track_GvR=track_GvR_.product()
    track_GlobalReco=track_GlobalReco_.product()
    track_ndel_eta=track_ndel_eta_.product()
    track_ndel_phi=track_ndel_phi_.product()
    true_trk_count+=n_tracks
    GvR_trk_count+=track_GvR.size()
#   track_GlobalReco=event.getByLabel('hltIterL3MuonPixelTracksFromGlobalViaRegion',mask_)
#    print "Event", i
    nRegions[0]=track_ndel_eta.size()/n_tracks
    idx_GvR=0
    dell_idx=0
    for idx in range(n_tracks):
        ptGlobalReco_hist.Fill(track_GlobalReco[idx].pt())       
        qpt[0]=track_GlobalReco[idx].charge()/track_GlobalReco[idx].pt()
        phi[0]=track_GlobalReco[idx].phi()
        eta[0]=track_GlobalReco[idx].eta()
        dxy[0]=track_GlobalReco[idx].dxy()
        dz[0] =track_GlobalReco[idx].dz()
        
        for j in range(nRegions[0]):
            ndelleta[j]=track_ndel_eta[dell_idx]
            ndellphi[j]=track_ndel_phi[dell_idx]
            delleta_GlobalReco_hist.Fill(ndelleta[j])
            dellphi_GlobalReco_hist.Fill(ndellphi[j])
            dell_idx+=1
        evtId[0]=i
        tree_GlobalReco.Fill()
        if mask[idx]:
            ptGvR_hist.Fill(track_GvR[idx_GvR].pt())
            for j in range(nRegions[0]):
                delleta_GvR_hist.Fill(ndelleta[j])
                dellphi_GvR_hist.Fill(ndellphi[j])
 
            qpt[0]=track_GvR[idx_GvR].charge()/track_GlobalReco[idx].pt()
            phi[0]=track_GvR[idx_GvR].phi()
            eta[0]=track_GvR[idx_GvR].eta()
            dxy[0]=track_GvR[idx_GvR].dxy()
            dz[0] =track_GvR[idx_GvR].dz()
            tree_GvR.Fill()
            idx_GvR+=1
        else:
            ptMissed_hist.Fill(track_GlobalReco[idx].pt())                  
#        print "\t phi: %.3f" %track.phi(),
#        print "\t eta: %.3f" %track.eta(),
#        print "\t dxy: %.4f" %track.dxy(),
#        int "\t dz: %.4f"  %track.dz()
 

print "true tracks ",true_trk_count
print "selcted tracks ", GvR_trk_count

afile.cd()
ptMissed_hist.Write()
ptGlobalReco_hist.Write()
ptGvR_hist.Write()
delleta_GlobalReco_hist.Write()
dellphi_GlobalReco_hist.Write()
delleta_GvR_hist.Write()
dellphi_GvR_hist.Write()

tree_GlobalReco.Write()
tree_GvR.Write()

acanvas=TCanvas("acanvas","acanvas")
ptMissed_hist.Draw()
acanvas.SaveAs("ptMissed_GvR.png")
ptGlobalReco_hist.Draw()
acanvas.SaveAs("pt_GlobalReco.png")
ptGvR_hist.Draw()
acanvas.SaveAs("pt_GvR.png")

delleta_GlobalReco_hist.Draw()
acanvas.SaveAs("nDellEta_GlobalReco.png")
dellphi_GlobalReco_hist.Draw()
acanvas.SaveAs("nDellPhi_GlobalReco.png")
delleta_GvR_hist.Draw()
acanvas.SaveAs("nDellEta_GvR.png")
dellphi_GvR_hist.Draw()
acanvas.SaveAs("nDellPhi_GvR.png")

