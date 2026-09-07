import numpy
import pytest
import libstempo as t2

DATA_PATH = t2.__path__[0] + "/data/"


@pytest.fixture(scope="module")
def psr():
    return t2.tempopulsar(
        parfile=DATA_PATH + "/J1909-3744_NANOGrav_dfg+12.par",
        timfile=DATA_PATH + "/J1909-3744_NANOGrav_dfg+12.tim",
    )


def test_which_values_do_not_change_existing_lists(psr):
    before_fit = psr.pars()
    before_set = psr.pars("set")
    before_all = psr.pars("all")

    psr.pars("fittable")
    psr.pars("identifiable")

    assert psr.pars() == before_fit
    assert psr.pars("set") == before_set
    assert psr.pars("all") == before_all
    assert psr["F0"].fit is True


def test_invariants(psr):
    fitted = set(psr.pars())
    fittable = set(psr.pars(which="fittable"))
    identifiable = set(psr.pars(which="identifiable"))
    sett = set(psr.pars(which="set"))
    assert fitted <= fittable <= sett
    assert identifiable <= fittable


def test_known_unfittable_set_parameters(psr):
    names = [
        "PEPOCH",
        "POSEPOCH",
        "DMEPOCH",
        "START",
        "FINISH",
        "TZRMJD",
        "TZRFRQ",
        "TRES",
        "EPHVER",
        "DMXR1_0001",
        "DMXR2_0001",
    ]
    for name in names:
        par = psr[name]
        assert par.set is True
        assert par.fittable is False
        assert par.identifiable is False


def test_known_fittable_set_parameters(psr):
    names = [
        "F0",
        "F1",
        "RAJ",
        "DECJ",
        "DM",
        "PX",
        "SINI",
        "PB",
        "A1",
        "TASC",
        "EPS1",
        "EPS2",
        "M2",
        "JUMP1",
    ]
    for name in names:
        par = psr[name]
        assert par.set is True
        assert par.fittable is True


def test_fit_flag_does_not_change_fittable(psr):
    assert psr["PEPOCH"].fittable is False
    try:
        psr["PEPOCH"].fit = True
        assert psr["PEPOCH"].fittable is False
        assert "PEPOCH" in psr.pars()  # request succeeded
        assert "PEPOCH" not in psr.pars(which="fittable")
    finally:
        psr["PEPOCH"].fit = False


def test_fittable_does_not_mutate_state(psr):
    before = {name: (psr[name].fit, psr[name].val) for name in psr.pars("set")}
    psr.pars("fittable")
    psr.pars("identifiable")
    after = {name: (psr[name].fit, psr[name].val) for name in psr.pars("set")}
    assert before == after


def test_assignment_rejected(psr):
    with pytest.raises(AttributeError):
        psr["F0"].fittable = True
    with pytest.raises(AttributeError):
        psr["F0"].identifiable = True


def test_unknown_which(psr):
    with pytest.raises(KeyError, match="fittable"):
        psr.pars(which="nope")
    assert len(psr.vals(which=["F0"])) == 1


def test_convenience_methods(psr):
    assert psr.fittable("F0") is True
    assert psr.fittable("PEPOCH") is False
    with pytest.raises(KeyError):
        psr.fittable("not_a_par")

    assert psr.identifiable("F0") is True
    assert psr.identifiable("PEPOCH") is False
    with pytest.raises(KeyError):
        psr.identifiable("not_a_par")


def test_identifiable_matches_nonzero_designmatrix_column(psr):
    M = psr.designmatrix(incoffset=True)
    names = ["Offset"] + list(psr.pars())
    for name in psr.pars():
        col = M[:, names.index(name)]
        assert psr[name].identifiable is bool(numpy.any(col != 0))
    assert psr["DM"].identifiable is True


def test_no_log_spam(psr, capfd):
    capfd.readouterr()  # drop fixture/setup noise
    psr.pars(which="fittable")
    out, err = capfd.readouterr()
    assert out == ""
    assert err == ""
    assert "No methods for fitting parameter" not in out
    assert "No methods for fitting parameter" not in err


def test_vals_which_fittable(psr):
    assert len(psr.vals(which="fittable")) == len(psr.pars(which="fittable"))
