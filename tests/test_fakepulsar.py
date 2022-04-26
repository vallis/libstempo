import shutil
import unittest
from pathlib import Path
from astropy.time import Time

import libstempo as t2
import numpy as np

from libstempo.toasim import fakepulsar

DATA_PATH = t2.__path__[0] + "/data/"

TMP_DIR = Path("test_fake_output")
TMP_DIR.mkdir(exist_ok=True)


class TestFakePulsar(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.obstimes = np.arange(53000, 54800, 10, dtype=np.float128)
        cls.toaerr = 1e-3
        cls.freq = 1440.0
        cls.observatory = "ao"
        cls.parfile = DATA_PATH + "/J1909-3744_NANOGrav_dfg+12.par"

        # create a fake pulsar using fakepulsar
        cls.fakepsr = fakepulsar(
            parfile=cls.parfile,
            obstimes=cls.obstimes,
            toaerr=cls.toaerr,
            freq=cls.freq,
            observatory=cls.observatory,
            iters=0,
        )

        # create a fake pulsar using tempopulsar
        cls.fakepsrtp = t2.tempopulsar(
            parfile=cls.parfile,
            toas=cls.obstimes,
            toaerrs=cls.toaerr,
            observatory=cls.observatory,
            obsfreq=cls.freq,
            dofit=False,
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(TMP_DIR)

    def test_exceptions(self):
        """
        Test exceptions when creating a fake pulsar via tempopulsar.
        """

        # pass string rather than number for TOA
        with self.assertRaises(TypeError):
            t2.tempopulsar(
                parfile=self.parfile,
                toas="blah",
                toaerrs=self.toaerr,
                observatory=self.observatory,
                obsfreq=self.freq,
            )

        # pass string rather than number for TOA error
        with self.assertRaises(TypeError):
            t2.tempopulsar(
                parfile=self.parfile,
                toas=self.obstimes,
                toaerrs="blah",
                observatory=self.observatory,
                obsfreq=self.freq,
            )

        # pass integer rather than string for observatory
        with self.assertRaises(TypeError):
            t2.tempopulsar(
                parfile=self.parfile,
                toas=self.obstimes,
                toaerrs=self.toaerr,
                observatory=0,
                obsfreq=self.freq,
            )

        # pass string rather than number for obsfreq
        with self.assertRaises(TypeError):
            t2.tempopulsar(
                parfile=self.parfile,
                toas=self.obstimes,
                toaerrs=self.toaerr,
                observatory=self.observatory,
                obsfreq="blah",
            )

        # test exceptions if values are not given
        kwargs = {
            "parfile": self.parfile,
            "toas": self.obstimes,
            "toaerrs": self.toaerr,
            "observatory": self.observatory,
            "obsfreq": self.freq,
        }

        for key in ["toaerrs", "observatory", "obsfreq"]:
            copykwargs = kwargs.copy()
            copykwargs[key] = None
            with self.assertRaises(ValueError):
                t2.tempopulsar(**copykwargs)

        # test exceptions for inconsistent lengths
        for key in ["toaerrs", "observatory", "obsfreq"]:
            copykwargs = kwargs.copy()
            # set to two value list
            copykwargs[key] = [kwargs[key] for _ in range(2)]
            with self.assertRaises(ValueError):
                t2.tempopulsar(**copykwargs)

    def test_astropy_array(self):
        """
        Test passing TOAs as an astropy Time array.
        """

        times = Time(self.obstimes, format="mjd", scale="utc")

        psr = t2.tempopulsar(
            parfile=self.parfile,
            toas=times,
            toaerrs=self.toaerr,
            observatory=self.observatory,
            obsfreq=self.freq,
        )

        self.assertEqual(len(self.obstimes), psr.nobs)
        self.assertTrue(np.all(self.obstimes == self.fakepsr.stoas))
        self.assertTrue(np.all(psr.stoas == self.fakepsr.stoas))
        self.assertEqual(psr.stoas[0].dtype, np.float128)

    def test_single_values(self):
        """
        Test passing single value TOAs.
        """

        psr = t2.tempopulsar(
            parfile=self.parfile,
            toas=self.obstimes[0],
            toaerrs=self.toaerr,
            observatory=self.observatory,
            obsfreq=self.freq,
        )

        self.assertEqual(psr.nobs, 1)
        self.assertEqual(len(psr.stoas), 1)
        self.assertTrue(np.all(self.fakepsr.stoas[0] == psr.stoas[0]))
        self.assertEqual(psr.stoas[0].dtype, np.float128)

    def test_toa_errs(self):
        """
        Test TOA errors are set correctly.
        """

        self.assertTrue(np.all(self.fakepsr.toaerrs == self.toaerr))
        self.assertTrue(np.all(self.fakepsrtp.toaerrs == self.toaerr))

    def test_observatory(self):
        """
        Test observatory values are set correctly.
        """

        self.assertTrue(np.all(self.fakepsr.telescope() == str.encode(self.observatory)))
        self.assertTrue(np.all(self.fakepsrtp.telescope() == str.encode(self.observatory)))

    def test_frequency(self):
        """
        Test frequency values are set correctly.
        """

        self.assertTrue(np.all(self.fakepsr.freqs == self.freq))
        self.assertTrue(np.all(self.fakepsrtp.freqs == self.freq))

    def test_sat_parts(self):
        """
        Test SAT day and second values are set correctly.
        """

        self.assertTrue(np.all(self.fakepsr.satDay() == self.fakepsrtp.satDay()))
        self.assertTrue(np.all(self.fakepsr.satSec() == self.fakepsrtp.satSec()))

    def test_deleted(self):
        """
        Test deleted values are equivalent.
        """

        self.assertTrue(np.all(self.fakepsr.deleted == self.fakepsrtp.deleted))
        self.assertTrue(np.all(self.fakepsr.deleted == np.zeros(len(self.obstimes), dtype=np.int32)))

    def test_pulsar_params(self):
        """
        Test pulsar parameters have been read in the same in both cases.
        """

        self.assertEqual(self.fakepsr.pars("all"), self.fakepsrtp.pars("all"))

        for key in self.fakepsr.pars("all"):
            self.assertEqual(self.fakepsr[key].val, self.fakepsrtp[key].val)

    def test_fake_pulsar(self):
        """
        Test fakepulsar function vs passing inputs directly to tempopulsar.
        """

        self.assertEqual(self.fakepsrtp.nobs, len(self.obstimes))
        self.assertEqual(self.fakepsrtp.nobs, self.fakepsr.nobs)
        self.assertEqual(self.fakepsrtp.name, "1909-3744")
        self.assertEqual(self.fakepsr.name, "1909-3744")

        self.assertTrue(np.all(self.fakepsrtp.stoas == self.obstimes))
        self.assertTrue(np.all(self.fakepsrtp.stoas == self.fakepsr.stoas))
        self.assertTrue(np.all(self.fakepsrtp.toas() == self.fakepsr.toas()))

        # check residuals are the same
        self.assertTrue(np.all(self.fakepsrtp.residuals() == self.fakepsr.residuals()))
        self.assertTrue(np.all(self.fakepsrtp.phaseresiduals() == self.fakepsr.phaseresiduals()))

    def test_write_tim(self):
        """
        Test writing out the .tim file and then reading it back in.
        """

        self.fakepsr.savetim(str(TMP_DIR / "fakepsr.tim"))
        self.fakepsrtp.savetim(str(TMP_DIR / "fakepsrtp.tim"))

        self.assertTrue((TMP_DIR / "fakepsr.tim").exists())
        self.assertTrue((TMP_DIR / "fakepsrtp.tim").exists())

        t2.purgetim(str(TMP_DIR / "fakepsr.tim"))
        t2.purgetim(str(TMP_DIR / "fakepsrtp.tim"))

        newfakepsr = t2.tempopulsar(parfile=self.parfile, timfile=str(TMP_DIR / "fakepsr.tim"), dofit=False)
        newfakepsrtp = t2.tempopulsar(parfile=self.parfile, timfile=str(TMP_DIR / "fakepsrtp.tim"), dofit=False)

        self.assertEqual(newfakepsrtp.nobs, len(self.obstimes))
        self.assertEqual(newfakepsrtp.nobs, newfakepsr.nobs)
        self.assertEqual(newfakepsrtp.name, "1909-3744")
        self.assertEqual(newfakepsr.name, "1909-3744")

        self.assertTrue(np.all(newfakepsrtp.stoas == self.obstimes))
        self.assertTrue(np.all(newfakepsrtp.stoas == self.fakepsrtp.stoas))
        self.assertTrue(np.all(newfakepsrtp.toas() == self.fakepsrtp.toas()))
        self.assertTrue(np.all(newfakepsr.stoas == self.fakepsrtp.stoas))
        self.assertTrue(np.all(newfakepsr.toas() == newfakepsrtp.toas()))

        # check residuals are the same
        self.assertTrue(np.all(newfakepsrtp.residuals() == self.fakepsrtp.residuals()))
        self.assertTrue(np.all(newfakepsrtp.phaseresiduals() == self.fakepsrtp.phaseresiduals()))
        self.assertTrue(np.all(newfakepsrtp.residuals() == newfakepsr.residuals()))
