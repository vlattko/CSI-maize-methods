import unittest
import numpy as np

from tracking_graph_core import fit_cycles, predict, partial_assignment, associate_visits, build_candidate_graph


def observations(points, omit=None, noise=0.0):
    rng=np.random.default_rng(71); out={}
    for v in range(40):
        if v%4==2: continue
        t=v*2*np.pi/40; out[v]=[]
        for i,(x,z,y) in enumerate(points):
            if omit and omit(i,v): continue
            out[v].append(dict(x=x*np.cos(t)+z*np.sin(t)+rng.normal(0,noise),y=y+rng.normal(0,noise),root_length=np.linalg.norm([x,z,y]),candidate_id=i))
    return out


def node(token,x,z,y,length=None):
    return dict(node_id=token,x_coefficient=[0,x,z],height=y,radius=np.hypot(x,z),root_length=length or np.linalg.norm([x,z,y]),training_rmse=.002,training_support=25,training_total_views=30)


class CycleTests(unittest.TestCase):
    def test_two_tracks_and_true_heldout_prediction(self):
        p=[(.32,.1,.83),(-.19,.24,.41)]; tracks=fit_cycles(observations(p,noise=.001))
        self.assertEqual(len(tracks),2)
        for q in tracks:
            self.assertLess(min(np.linalg.norm(np.array([q['x_coefficient'][1],q['x_coefficient'][2],q['height']])-x) for x in p),.015)
            self.assertLess(q['training_rmse'],.01)
            self.assertEqual(len(predict(q,2)),2)

    def test_occluded_track_below_old_40percent_support_is_retained(self):
        # Ten of30 training views, distributed over the cycle, remain observed.
        keep={0,3,7,11,15,19,23,27,31,35}
        tr=fit_cycles(observations([(.3,.12,.7)],omit=lambda i,v:v not in keep))
        self.assertEqual(len(tr),1)
        self.assertEqual(tr[0]['training_support'],10)

    def test_no_heldout_leakage(self):
        with self.assertRaises(ValueError): fit_cycles({2:[dict(x=0,y=1)]})

    def test_sparse_and_empty(self):
        self.assertEqual(fit_cycles({}),[])
        self.assertEqual(fit_cycles({0:[dict(x=0,y=1)]}),[])

    def test_disjoint_candidate_ownership(self):
        tracks=fit_cycles(observations([(.3,0,.8),(.3,0,.8)]))
        ownership=[(a['view'],a['candidate_id']) for t in tracks for a in t['assignments']]
        self.assertEqual(len(ownership),len(set(ownership)))


class AssociationTests(unittest.TestCase):
    def test_partial_assignment_does_not_force_distant_link(self):
        links,ua,ub,total=partial_assignment(np.array([[0.05,9],[9,9]]),gap=.75)
        self.assertEqual(links,[(0,0)]);self.assertEqual(ua,[1]);self.assertEqual(ub,[1])

    def test_empty_sides_are_retained(self):
        r=associate_visits([node('a',.3,0,.8)],[],phases=[0])
        self.assertEqual(r['accepted_links'],[])
        self.assertEqual(r['earlier_states'][0]['state'],'unobserved_after_or_unmatched')

    def test_known_phase_with_permuted_tokens(self):
        a=[node('a',.4,0,.9),node('b',0,.25,.4)]
        b=[node('y',.25,0,.4),node('x',0,-.4,.9)]
        r=associate_visits(a,b,phases=range(0,360,9))
        self.assertEqual({(e['source'],e['target']) for e in r['accepted_links']},{('a','x'),('b','y')})
        self.assertEqual(r['best_phase_degrees'],90)

    def test_identical_instances_abstain(self):
        a=[node('a',.3,0,.8),node('b',.3,0,.8)]
        b=[node('x',.3,0,.8),node('y',.3,0,.8)]
        r=associate_visits(a,b,phases=[0])
        self.assertEqual(r['accepted_links'],[])
        self.assertTrue(all(e['status']=='ambiguous' for e in r['edges']))

    def test_half_turn_phase_symmetry_is_not_forced_to_an_identity(self):
        a=[node('a',.3,0,.8),node('b',-.3,0,.8)]
        b=[node('x',.3,0,.8),node('y',-.3,0,.8)]
        r=associate_visits(a,b)
        self.assertEqual(r['accepted_links'],[])
        self.assertIn(0,r['retained_phases_degrees']);self.assertIn(180,r['retained_phases_degrees'])

    def test_unmatched_later_is_not_called_birth(self):
        r=associate_visits([], [node('new',.1,0,.3)],phases=[0])
        self.assertEqual(r['later_states'][0]['state'],'newly_observed_birth_unresolved')
        self.assertNotIn('death',str(r))

    def test_graph_retains_all_visit_slots_and_gap_candidates(self):
        sessions=[dict(visit=1,nodes=[node('v1',.3,0,.8)]),dict(visit=2,nodes=[]),dict(visit=3,nodes=[node('v3',.3,0,.8)])]
        g=build_candidate_graph('test',sessions,phases=[0])
        self.assertEqual(len(g['sessions']),3)
        self.assertEqual(sum(p['visit_gap']==1 for p in g['pairs']),2)
        gap=[e for e in g['edges'] if e['visit_gap']==2]
        self.assertTrue(gap)
        self.assertEqual(gap[0]['state'],'possible_reappearance')
        self.assertEqual(g['physical_identity_status'],'metadata_candidate_unverified')


class ObservationTests(unittest.TestCase):
    def test_path_is_source_supported_and_ends_at_terminal(self):
        from tracking_observations import extract_paths
        mask=np.zeros((100,100),bool);mask[20:91,49:52]=True
        cc=extract_paths(mask,min_length=6)
        self.assertEqual(len(cc),1)
        p=cc[0]['path_processing_px']
        self.assertEqual(p[-1],[cc[0]['terminal_x_px'],cc[0]['terminal_y_px']])
        self.assertTrue(all(mask[int(y),int(x)] for x,y in p))

    def test_border_is_censored(self):
        from tracking_observations import extract_paths
        mask=np.zeros((100,100),bool);mask[0:91,49:52]=True
        self.assertTrue(extract_paths(mask,min_length=6)[0]['censored'])

    def test_empty_mask_retains_empty_state(self):
        from tracking_observations import extract_paths
        self.assertEqual(extract_paths(np.zeros((100,100),bool)),[])

    def test_shared_gauge_point_fields_agree(self):
        from tracking_development import shared_nodes
        n=node('a',.2,.1,.7);n['point']={'x':.2,'depth':.1,'up':.7};n['known_render_point']=[1,2,3]
        s=dict(nodes=[n],height=2.,origin=np.array([1.,0,0]));a=dict(height=4.,origin=np.zeros(3))
        q=shared_nodes(s,a)[0]
        self.assertEqual(q['point'],{'x':.75,'depth':1.,'up':1.5})


if __name__=='__main__': unittest.main()
