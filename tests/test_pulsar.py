import shutil
import tempfile
import unittest
from pathlib import Path

import libstempo as t2
import numpy as np

DATA_PATH = t2.__path__[0] + "/data/"

TMP_DIR = Path("test_output")
TMP_DIR.mkdir(exist_ok=True)

try:
    NP_LONG_DOUBLE_TYPE = np.float128
except AttributeError:
    NP_LONG_DOUBLE_TYPE = np.double


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
        self.assertEqual(self.psr.stoas[0].dtype, NP_LONG_DOUBLE_TYPE)

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


class TestFdjumpParameterNames(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.tempdir.name)
        self.par_counter = 0
        self.toas = np.array([53500.0, 54500.0, 55500.0], dtype=NP_LONG_DOUBLE_TYPE)

    def tearDown(self):
        self.tempdir.cleanup()

    def make_pulsar(self, fdjump_lines=None, parfile=None):
        if parfile is None:
            self.par_counter += 1
            parfile = self.temp_path / "fdjump-{0}.par".format(self.par_counter)
            base_par = Path(DATA_PATH, "J1909-3744_NANOGrav_dfg+12.par").read_text()
            parfile.write_text(base_par + "\n" + fdjump_lines)

        return t2.tempopulsar(
            parfile=str(parfile),
            toas=self.toas,
            toaerrs=1.0,
            observatory="ao",
            obsfreq=1400.0,
            dofit=False,
        )

    @staticmethod
    def fdjump_names(psr):
        return tuple(name for name in psr.pars("all") if psr[name].isfdjump)

    def test_names_use_fd_index_not_storage_slot(self):
        psr = self.make_pulsar(
            "FDJUMPDM MJD 53000 56000 0.03 1\n" "FDJUMP3 MJD 53000 56000 0.003 1\n" "FDJUMP1 MJD 53000 56000 0.01 1\n"
        )

        self.assertEqual(
            self.fdjump_names(psr),
            ("FDJUMPDM1", "FDJUMP3", "FDJUMP1"),
        )
        self.assertEqual(
            tuple(psr[name].subct for name in self.fdjump_names(psr)),
            (1, 2, 3),
        )
        self.assertAlmostEqual(psr["FDJUMPDM1"].val, 0.03)
        self.assertAlmostEqual(psr["FDJUMP3"].val, 0.003)
        self.assertAlmostEqual(psr["FDJUMP1"].val, 0.01)

    def test_repeated_index_gets_mask_occurrence_suffix(self):
        psr = self.make_pulsar("FDJUMP1 MJD 53000 54000 0.01 1\n" "FDJUMP1 MJD 54000 55000 0.02 1\n")

        self.assertEqual(self.fdjump_names(psr), ("FDJUMP1", "FDJUMP1_2"))
        self.assertNotEqual(psr["FDJUMP1"].subct, psr["FDJUMP1_2"].subct)

        psr["FDJUMP1"].val = 0.011
        psr["FDJUMP1_2"].val = 0.022
        self.assertAlmostEqual(psr["FDJUMP1"].val, 0.011)
        self.assertAlmostEqual(psr["FDJUMP1_2"].val, 0.022)

        fitpars = psr.pars()
        designmatrix = psr.designmatrix(incoffset=False)
        first = designmatrix[:, fitpars.index("FDJUMP1")]
        second = designmatrix[:, fitpars.index("FDJUMP1_2")]
        self.assertEqual(tuple(np.flatnonzero(first)), (0,))
        self.assertEqual(tuple(np.flatnonzero(second)), (1,))

        saved_par = self.temp_path / "roundtrip.par"
        psr.savepar(str(saved_par))
        reparsed = self.make_pulsar(parfile=saved_par)
        self.assertEqual(self.fdjump_names(reparsed), ("FDJUMP1", "FDJUMP1_2"))
        self.assertAlmostEqual(reparsed["FDJUMP1"].val, 0.011)
        self.assertAlmostEqual(reparsed["FDJUMP1_2"].val, 0.022)
