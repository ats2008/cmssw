import DataFormats.FWLite as fwlite
from ROOT import TFile, TTree, TH1D, TCanvas
from array import array
import numpy as np

def  deltaR2( eta1,phi1,eta2,phi2): 
    deta = eta1 - eta2
    dphi =abs(phi1 - phi2)
    if dphi > np.pi:
        dphi -= 2*np.pi
    return deta * deta + dphi * dphi;

tot=0
xx=0
yy=0
zz=0

def get_trackMatches(tracks_truth,tracks_set):
    M=tracks_truth.size()
    N=tracks_set.size()
    matched_track=[-1 for i in range(M)]
    matched_trackR2=[-1 for i in range(M)]
    matched_trackR2X=[-1 for i in range(M)]
    
    set_matched_track=np.zeros(N)-1
 
    deltaR2s=np.zeros(N*M)
    deltaR2sX=np.zeros(N*M)
    for idx in range(M):
        phi=tracks_truth[idx].phi()
        eta=tracks_truth[idx].eta()
        dz=tracks_truth[idx].dz()
        dxy=tracks_truth[idx].dxy()
        pt=tracks_truth[idx].pt()
        for j in range(N):
            deltaR2s[i*N+j]=deltaR2(phi,eta,tracks_set[j].phi(),tracks_set[j].eta())
            deltaR2sX[i*N+j]=deltaR2(dz,dxy,tracks_set[j].dz(),tracks_set[j].dxy())
    sorted_idx=np.argsort(deltaR2s)
    trk_found=0
    for idx in sorted_idx:
        x=int(idx/N)
        if matched_track[x]!=-1:
            continue
        y= idx % N
        if set_matched_track[y]!=-1:
            continue
        set_matched_track[y]=x
        if y<0 :    #sanity check
            print "\n\n oh ho !! problem !! \n\n"
        matched_track[x]=y
        matched_trackR2[x]=deltaR2s[idx]
        matched_trackR2X[x]=deltaR2sX[idx]
        trk_found+=1
        if trk_found==M:
            break
    return matched_track,matched_trackR2,matched_trackR2X


events = fwlite.Events("file:TrkSelByRegion_validationRAW.root")

mask_ = fwlite.Handle("std::vector<bool>") 
track_GvR_ = fwlite.Handle("std::vector<reco::Track>")
track_RegionalReco_ =fwlite.Handle("std::vector<reco::Track>")
#track_GlobalReco_ = fwlite.Handle("std::vector<reco::Track>")
RegionalReco_track_ndel_eta_=fwlite.Handle("std::vector<float>")
RegionalReco_track_ndel_phi_=fwlite.Handle("std::vector<float>")
GvR_track_ndel_eta_=fwlite.Handle("std::vector<float>")
GvR_track_ndel_phi_=fwlite.Handle("std::vector<float>")
GvR_mask_=fwlite.Handle("std::vector<bool>")
RvR_mask_=fwlite.Handle("std::vector<bool>")

afile=TFile("TrkSelectorComparison.root","RECREATE")
tree_GvR=TTree("tracks_GvR","Tracks_GvR")
tree_GvR_matched=TTree("tracks_GvR_matchedToRegReco","Tracks_GvR_matched")
tree_GvR_unmatched=TTree("tracks_GvR_gained","Tracks_GvR_unmatched")
tree_RegionalReco=TTree("tracks_RegReco","Track_RegReco")
tree_RegionalReco_missed=TTree("tracks_RegReco_missingInGvR","Track_RegReco_miss")
tree_RvR=TTree("tracks_RvR","Tracks_RvR")
tree_RvR_missed=TTree("tracks_missRvR","Tracks_RvR_miss")
#tree_GlobalReco=TTree("track_GlobalReco","track_GlobalReco")

ptt  = array('d',[0])
qpt  = array('d',[0])
eta  = array('d',[0])
phi  = array('d',[0])
dxy  = array('d',[0])
dz  = array('d',[0])
evtId=array('i',[0])
nRegions=array('I',[0])
deltar2=array('d',[0])
deltar2X=array('d',[0])
ndelleta=np.zeros(2000)
ndellphi=np.zeros(2000)


tree_RegionalReco.Branch("pt_track",ptt,"pt/D")
tree_RegionalReco.Branch("qpt",qpt,"qpt/D")
tree_RegionalReco.Branch("eta",eta,"eta/D")
tree_RegionalReco.Branch("phi",phi,"phi/D")
tree_RegionalReco.Branch("dxy",dxy,"dxy/D")
tree_RegionalReco.Branch("dz",dz,"dz/D")
tree_RegionalReco.Branch("event_id",evtId,"event_id/i")
tree_RegionalReco.Branch( 'nRegions', nRegions, 'nRegions/I' )
tree_RegionalReco.Branch( 'ndelleta', ndelleta, 'ndelleta[nRegions]/D')
tree_RegionalReco.Branch( 'ndellphi', ndellphi, 'ndellphi[nRegions]/D')

tree_RvR.Branch("pt_track",ptt,"pt/D")
tree_RvR.Branch("qpt",qpt,"qpt/D")
tree_RvR.Branch("eta",eta,"eta/D")
tree_RvR.Branch("phi",phi,"phi/D")
tree_RvR.Branch("dxy",dxy,"dxy/D")
tree_RvR.Branch("dz",dz,"dz/D")
tree_RvR.Branch("event_id",evtId,"event_id/i")
tree_RvR.Branch( 'nRegions', nRegions, 'nRegions/I' )
tree_RvR.Branch( 'ndelleta', ndelleta, 'ndelleta[nRegions]/D')
tree_RvR.Branch( 'ndellphi', ndellphi, 'ndellphi[nRegions]/D')

tree_RvR_missed.Branch("pt_track",ptt,"pt/D")
tree_RvR_missed.Branch("qpt",qpt,"qpt/D")
tree_RvR_missed.Branch("eta",eta,"eta/D")
tree_RvR_missed.Branch("phi",phi,"phi/D")
tree_RvR_missed.Branch("dxy",dxy,"dxy/D")
tree_RvR_missed.Branch("dz",dz,"dz/D")
tree_RvR_missed.Branch("event_id",evtId,"event_id/i")
tree_RvR_missed.Branch( 'nRegions', nRegions, 'nRegions/I' )
tree_RvR_missed.Branch( 'ndelleta', ndelleta, 'ndelleta[nRegions]/D')
tree_RvR_missed.Branch( 'ndellphi', ndellphi, 'ndellphi[nRegions]/D')

tree_RegionalReco_missed.Branch("pt_track",ptt,"pt/D")
tree_RegionalReco_missed.Branch("qpt",qpt,"qpt/D")
tree_RegionalReco_missed.Branch("eta",eta,"eta/D")
tree_RegionalReco_missed.Branch("phi",phi,"phi/D")
tree_RegionalReco_missed.Branch("dxy",dxy,"dxy/D")
tree_RegionalReco_missed.Branch("dz",dz,"dz/D")
tree_RegionalReco_missed.Branch("event_id",evtId,"event_id/i")
tree_RegionalReco_missed.Branch( 'nRegions', nRegions, 'nRegions/I' )
tree_RegionalReco_missed.Branch( 'ndelleta', ndelleta, 'ndelleta[nRegions]/D')
tree_RegionalReco_missed.Branch( 'ndellphi', ndellphi, 'ndellphi[nRegions]/D')

tree_GvR.Branch("pt_track",ptt,"pt/D")
tree_GvR.Branch("qpt",qpt,"qpt/D")
tree_GvR.Branch("eta",eta,"eta/D")
tree_GvR.Branch("phi",phi,"phi/D")
tree_GvR.Branch("dxy",dxy,"dxy/D")
tree_GvR.Branch("dz",dz,"dz/D")
tree_GvR.Branch("event_id",evtId,"event_id/i")
tree_GvR.Branch( 'nRegions', nRegions, 'nRegions/I' )
tree_GvR.Branch( 'ndelleta', ndelleta, 'ndelleta[nRegions]/D')
tree_GvR.Branch( 'ndellphi', ndellphi, 'ndellphi[nRegions]/D')

tree_GvR_matched.Branch("pt_track",ptt,"pt/D")
tree_GvR_matched.Branch("qpt",qpt,"qpt/D")
tree_GvR_matched.Branch("eta",eta,"eta/D")
tree_GvR_matched.Branch("phi",phi,"phi/D")
tree_GvR_matched.Branch("dxy",dxy,"dxy/D")
tree_GvR_matched.Branch("dz",dz,"dz/D")
tree_GvR_matched.Branch("event_id",evtId,"event_id/i")
tree_GvR_matched.Branch( 'nRegions', nRegions, 'nRegions/I' )
tree_GvR_matched.Branch( 'ndelleta', ndelleta, 'ndelleta[nRegions]/D')
tree_GvR_matched.Branch( 'ndellphi', ndellphi, 'ndellphi[nRegions]/D')
tree_GvR_matched.Branch( 'deltaR2',deltar2,'deltar2/D')
tree_GvR_matched.Branch( 'deltaR2DzDxy',deltar2X,'deltar2/D')

tree_GvR_unmatched.Branch("pt_track",ptt,"pt/D")
tree_GvR_unmatched.Branch("qpt",qpt,"qpt/D")
tree_GvR_unmatched.Branch("eta",eta,"eta/D")
tree_GvR_unmatched.Branch("phi",phi,"phi/D")
tree_GvR_unmatched.Branch("dxy",dxy,"dxy/D")
tree_GvR_unmatched.Branch("dz",dz,"dz/D")
tree_GvR_unmatched.Branch("event_id",evtId,"event_id/i")
tree_GvR_unmatched.Branch( 'nRegions', nRegions, 'nRegions/I' )
tree_GvR_unmatched.Branch( 'ndelleta', ndelleta, 'ndelleta[nRegions]/D')
tree_GvR_unmatched.Branch( 'ndellphi', ndellphi, 'ndellphi[nRegions]/D')

ptRegionalReco_hist = TH1D("pt_RegionalReco","pt of RegionalReco",50,0.0,25.0)
ptGvR_hist = TH1D("pt_GvR","pt of GvR",50,0.0,25.0)
ptMissed_hist = TH1D("pt_Miss_GvR","pt of Unmatched Tracks [in RegonalReco]",50,0.0,25.0)
ptMatch_hist = TH1D("pt_Match_GvR","pt of Matched tracks [GvR & RegReco]",50,0.0,25.0)
ptGained_hist = TH1D("pt_Gained_GvR","pt of additional Tracks in GvR",50,0.0,25.0)

MissedDeltaR2_hist = TH1D("missed_deltaR2","R2 for missed tracks in RegReco",200,0.,3.)
MatchedDeltaR2X_hist = TH1D("matchedDeltaR2X","R2[dz,dxy] for matched track",200,0.,3.)
MatchedDeltaR2_hist = TH1D("matchedDeltaR2","R2 for matched track",200,0.,.3)


print "Total Number of Events = ",events.size()

true_trk_count=0
GvR_trk_count=0
RvR_trk_count=0
evtCount=0
trigEvtCount=0
for i, event in enumerate(events):
    evtCount+=1
    if i%250==0:
        print "At event : ",i
    
    event.getByLabel('hltIterL3MuonPixelTracksFromGlobalViaRegion',GvR_mask_)
    event.getByLabel('hltIterL3MuonPixelTracksFromRegionalViaRegion',RvR_mask_)
    if not GvR_mask_.isValid():
        continue
    trigEvtCount+=1
    event.getByLabel('hltIterL3MuonPixelTracksFromGlobalViaRegion',track_GvR_)
    event.getByLabel('hltIterL3MuonPixelTracks',track_RegionalReco_)
    event.getByLabel('hltIterL3MuonPixelTracksFromRegionalViaRegion','normDeltaEta',RegionalReco_track_ndel_eta_)
    event.getByLabel('hltIterL3MuonPixelTracksFromRegionalViaRegion','normDeltaPhi',RegionalReco_track_ndel_phi_)
   
    event.getByLabel('hltIterL3MuonPixelTracksFromGlobalViaRegion','normDeltaEta',GvR_track_ndel_eta_)
    event.getByLabel('hltIterL3MuonPixelTracksFromGlobalViaRegion','normDeltaPhi',GvR_track_ndel_phi_)

#    if not track_GvR_.isValid():
#        continue
    GvR_mask= GvR_mask_.product()
    RvR_mask= RvR_mask_.product()
    track_GvR=track_GvR_.product()
    track_RegionalReco=track_RegionalReco_.product()
    RegionalReco_track_ndel_eta=RegionalReco_track_ndel_eta_.product()
    RegionalReco_track_ndel_phi=RegionalReco_track_ndel_phi_.product()
    GvR_track_ndel_eta=GvR_track_ndel_eta_.product()
    GvR_track_ndel_phi=GvR_track_ndel_phi_.product()
    
    matched_track_ids,matched_deltaR2,matched_deltaR2X=get_trackMatches(track_RegionalReco,track_GvR)
#    aa=0
#    bb=0
#    for idx in matched_track_ids:
#        if idx==-1:
#            aa+=1
#        zz+=1
#    for idx in range(track_GvR.size()):
#        if idx in matched_track_ids:
#            bb+=1
#    xx+=aa
#    yy+=bb
#    if aa+bb!=track_RegionalReco.size():
#        print " total true  tracks = ",zz, "[",track_RegionalReco.size(),track_GvR.size(),"]"
#        print " total  -1   tracks = ",xx,"[",aa,"]"     
#        print " total !-1   tracks = ",yy,"[",bb,"]"      
#        print " delta = ",yy-xx
#    continue
    
    deltaRCUT=1e9
    for idx in range(len(matched_track_ids)):
        if matched_deltaR2[idx]>deltaRCUT:
            matched_track_ids[idx]=-1

    n_tracks=RvR_mask.size()
    true_trk_count+=n_tracks
    GvR_trk_count+=track_GvR.size()

    evtId[0]=i
    if n_tracks>0:
        nRegions[0]=RegionalReco_track_ndel_eta.size()/n_tracks
    else:
        nRegions[0]=0
    idx_GvR=0
    dell_idx=0
    for idx in range(n_tracks):
        for j in range(nRegions[0]):
                ndelleta[j]=RegionalReco_track_ndel_eta[dell_idx]
                ndellphi[j]=RegionalReco_track_ndel_phi[dell_idx]
                dell_idx+=1
        ptRegionalReco_hist.Fill(track_RegionalReco[idx].pt())       
        qpt[0]=track_RegionalReco[idx].charge()/track_RegionalReco[idx].pt()
        ptt[0]=track_RegionalReco[idx].pt()
        phi[0]=track_RegionalReco[idx].phi()
        eta[0]=track_RegionalReco[idx].eta()
        dxy[0]=track_RegionalReco[idx].dxy()
        dz[0] =track_RegionalReco[idx].dz()
        tree_RegionalReco.Fill()
        if matched_track_ids[idx]==-1:
            tree_RegionalReco_missed.Fill()
            ptMissed_hist.Fill(track_RegionalReco[idx].pt())
            MissedDeltaR2_hist.Fill(matched_deltaR2[idx])
        if RvR_mask[idx]:
            RvR_trk_count+=1
            tree_RvR.Fill()
        else:
            tree_RvR_missed.Fill()
    dell_idx=0
    nGvR_tracks=track_GvR.size()
    nGlobalReco_tracks=GvR_mask.size()
    if nGvR_tracks!=0:
        print "nGvR_tracks ",nGvR_tracks," GvR_track_ndel_eta ",GvR_track_ndel_eta.size()," nGlobalReco_tracks ",nGlobalReco_tracks
    if nGvR_tracks>0:
#        print "GvR_track_ndel_eta.size()/GvR_mask.size() = ",GvR_track_ndel_eta.size(),"/",GvR_mask.size()
        nRegions[0]=GvR_track_ndel_eta.size()/nGlobalReco_tracks
    else:
        nRegions[0]=0
    idx=-1
    for jdx in range(nGlobalReco_tracks):
        if not GvR_mask[jdx]:
            dell_idx+=nRegions[0]
            continue
        idx+=1
        for j in range(nRegions[0]):
                ndelleta[j]=GvR_track_ndel_eta[dell_idx]
                ndellphi[j]=GvR_track_ndel_phi[dell_idx]
                dell_idx+=1
        ptGvR_hist.Fill(track_GvR[idx].pt()) 
        qpt[0]=track_GvR[idx].charge()/track_GvR[idx].pt()
        ptt[0]=track_GvR[idx].pt()
        phi[0]=track_GvR[idx].phi()
        eta[0]=track_GvR[idx].eta()
        dxy[0]=track_GvR[idx].dxy()
        dz[0]=track_GvR[idx].dz()
        tree_GvR.Fill()
        if idx in matched_track_ids:
            idxm=matched_track_ids.index(idx)
            ptMatch_hist.Fill(track_GvR[idx].pt())       
            deltar2[0]=matched_deltaR2[idxm]
            deltar2X[0]=matched_deltaR2X[idxm]
            tree_GvR_matched.Fill()
            for j in range(nRegions[0]):
                if abs(ndelleta[j])>0.8:
                    print "ndelleta[j] : ",ndelleta[j]," [ ",j," ] ",nRegions[0]
            MatchedDeltaR2_hist.Fill(deltar2[0])
            MatchedDeltaR2X_hist.Fill(deltar2X[0])
        else:
            ptGained_hist.Fill(track_GvR[idx].pt())
            tree_GvR_unmatched.Fill()

print "total number of events processed : ",evtCount
print "total number of events ggered evts : ",trigEvtCount
print "true tracks ",true_trk_count
print "selcted rvr tracks ", RvR_trk_count
print "selcted gvr tracks ", GvR_trk_count

afile.cd()
ptMissed_hist.Write()
ptRegionalReco_hist.Write()
ptGvR_hist.Write()
MatchedDeltaR2_hist.Write()
MissedDeltaR2_hist.Write()

tree_RegionalReco.Write()
tree_RegionalReco_missed.Write()
tree_GvR.Write()
tree_GvR_matched.Write()
tree_GvR_unmatched.Write()
tree_RvR.Write()
tree_RvR_missed.Write()



acanvas=TCanvas("acanvas","acanvas")
ptRegionalReco_hist.Draw()
acanvas.SaveAs("pt_RegionalReco_cmp.png")
ptGvR_hist.Draw()
acanvas.SaveAs("pt_GvR_cmp.png")
ptMissed_hist.Draw()
acanvas.SaveAs("pt_MissedFromRegReco_cmp.png")
ptMatch_hist.Draw()
acanvas.SaveAs("pt_MatchedTracks_cmp.png")
ptGained_hist.Draw()
acanvas.SaveAs("pt_GainedTracks_cmp.png")
MatchedDeltaR2_hist.Draw()
acanvas.SaveAs("MatchedDeltaR2_cmp.png")
MatchedDeltaR2X_hist.Draw()
acanvas.SaveAs("MatchedDeltaR2_dzdxy_cmp.png")
MissedDeltaR2_hist.Draw()
acanvas.SaveAs("MissedDeltaR2_cmp.png")
