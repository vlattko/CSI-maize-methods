import unittest
import numpy as np
from architecture_external_scaffold import terminal_point

class TerminalScaffoldTests(unittest.TestCase):
    def test_known_camera_gauge_inversion(self):
        truth=np.array([.1,-.2,.8]);gauge=dict(axis=190,base=350,height=280)
        track=dict(x_coefficient=[2/280,300/280*.1,300/280*-.2],height=(350-360+300*.8)/280)
        actual=terminal_point(track,gauge)
        self.assertTrue(np.allclose(actual,truth))

if __name__=='__main__':unittest.main()
