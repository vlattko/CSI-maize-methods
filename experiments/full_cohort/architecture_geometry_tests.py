import unittest
import numpy as np
from architecture_geometry_diagnostics import project_points,carve,render_cells,select_cases

class GeometryDiagnosticsTests(unittest.TestCase):
    def test_perspective_infinite_limit(self):
        points=np.array([[.1,.4,.2],[-.3,.8,-.1]])
        c=dict(axis=80,base=110,scale=90,principal=[80,60],distance=1e10,roll=3,elevation=0)
        pp=project_points(points,3,c);c['distance']=None;ortho=project_points(points,3,c)
        self.assertTrue(np.allclose(pp,ortho,atol=1e-5))
    def test_conservative_cell_carving_preserves_center_solutions(self):
        y,x=np.mgrid[:120,:160];mask=(abs(x-80)<15)&(y>10)&(y<110);masks={v:mask for v in [0,5,10,15,20,25,30,35]}
        c=dict(axis=80,base=110,scale=100,principal=[80,60],distance=None,roll=0,elevation=0)
        a,g=carve(masks,c,32,'center');b,_=carve(masks,c,32,'conservative_cell_bbox')
        self.assertTrue(np.all(b[a]));self.assertGreaterEqual(b.sum(),a.sum())
        pred=render_cells(b,g,2,c,mask.shape);self.assertEqual(pred.shape,mask.shape);self.assertTrue(pred.any())
    def test_subset_retains_trajectories_and_balances_treatment_and_genotype(self):
        groups={}
        for treatment in ['C','P','S']:
            for genotype in ['A','B','C']:
                for visit in range(1,6):
                    key=(treatment,str(visit),genotype,'1',treatment+genotype+str(visit),'rgb');groups[key]=[dict(view=v,qc_pass=True) for v in range(40)]
        cases=select_cases(groups,15);self.assertEqual(len(cases),15)
        self.assertEqual({k[0] for k,g in cases},{'C','P','S'})
        self.assertEqual(len({k[2] for k,g in cases}),3)
    def test_declared_domain_can_retain_a_known_point_outside_default_box(self):
        c=dict(axis=200,base=180,scale=100,principal=[200,100],distance=None,roll=0,elevation=0);masks={}
        for view in range(40):
            xy=np.rint(project_points(np.array([[1.,.8,0.]]),view,c)[0]).astype(int);m=np.zeros((200,400),bool);m[xy[1]-2:xy[1]+3,xy[0]-2:xy[0]+3]=True;masks[view]=m
        a,_=carve(masks,c,49,'center');b,g=carve(masks,c,49,'center',bounds=(1.2,0,1.4))
        self.assertEqual(a.sum(),0);self.assertGreater(b.sum(),0)

if __name__=='__main__':unittest.main()
