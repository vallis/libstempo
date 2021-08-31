import shutil
import unittest
from pathlib import Path

import libstempo as t2
import numpy as np

DATA_PATH = t2.__path__[0] + "/data/"

TMP_DIR = Path("test_output")
TMP_DIR.mkdir(exist_ok=True)


class TestDeterministicSignals(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.psr = t2.tempopulsar(
            parfile=DATA_PATH + "/J1909-3744_NANOGrav_dfg+12.par", timfile=DATA_PATH + "/J1909-3744_NANOGrav_dfg+12.tim"
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(TMP_DIR)

    def test_attrs(self):
        self.assertEqual(self.psr.nobs, 1001)
        self.assertEqual(self.psr.name, "1909-3744")
        self.assertEqual(len(self.psr.stoas), 1001)
        self.assertTrue(np.all(self.psr.stoas > 50000) and np.all(self.psr.stoas < 59000))
        self.assertTrue(np.all(self.psr.toaerrs > 0.01) and np.all(self.psr.toaerrs < 10))
        self.assertTrue(np.all(self.psr.freqs > 700) and np.all(self.psr.freqs < 4000))
        self.assertEqual(self.psr.stoas[0].dtype, np.float128)

    def test_toas(self):
        self.assertTrue(np.all(self.psr.toas() != self.psr.stoas))
        self.assertTrue(np.allclose(self.psr.toas(), self.psr.stoas, atol=1))

    def test_residuals(self):
        self.assertTrue(np.all(self.psr.residuals() > -2e-5) and np.all(self.psr.residuals() < 1.5e-5))

    def test_flags(self):
        expected = {"B", "be", "bw", "chanid", "fe", "proc", "pta", "tobs"}
        self.assertEqual(set(self.psr.flags()), expected)

    def test_radec(self):
        self.assertTrue(np.allclose(self.psr["RAJ"].val, 5.0169080674060326785))
        self.assertTrue(np.allclose(self.psr["DECJ"].val, 7.753759525058565179e-10, atol=1))

        expected = (True, True)
        tested = (self.psr["RAJ"].set, self.psr["DECJ"].set)
        self.assertEqual(tested, expected)

    def test_fitpars(self):
        expected = ("RAJ", "DECJ", "F0", "F1", "PMRA", "PMDEC", "PX", "SINI", "PB", "A1", "TASC", "EPS1", "EPS2", "M2")
        fitpars = self.psr.pars()
        self.assertEqual(fitpars[:14], expected)

        setpars = self.psr.pars(which="set")
        self.assertEqual(len(setpars), 158)

        # different versions of tempo2 define different number of parameters
        # allpars = self.psr.pars(which="all")
        # self.assertEqual(len(allpars), 4487)

    def test_fit(self):
        _ = self.psr.fit()
        fitvals = self.psr.vals()
        self.assertEqual(len(fitvals), 82)

    def test_designmatrix(self):
        dmat = self.psr.designmatrix()
        self.assertEqual(dmat.shape, (1001, 83))

    def test_save_partim(self):
        self.psr.savepar(str(TMP_DIR / "tmp.par"))
        self.psr.savetim(str(TMP_DIR / "tmp.tim"))

        self.assertTrue((TMP_DIR / "tmp.par").exists())
        self.assertTrue((TMP_DIR / "tmp.tim").exists())
