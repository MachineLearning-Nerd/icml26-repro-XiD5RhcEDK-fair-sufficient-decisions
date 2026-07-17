import sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'repro'/'src'))
from run_fair_decisions import main

@pytest.fixture(scope='session')
def out(tmp_path_factory): return main(tmp_path_factory.mktemp('r')/'summary.json')

def test_calibrated_scores_threshold_ppv_failure(out): assert out['claim_1']['ppv_gap'] > .1
def test_calibrated_scores_threshold_for_failure(out): assert out['claim_1']['for_gap'] > .05
def test_both_paper_instances(out): assert [r['case'] for r in out['claim_2_and_3']] == ['A','B']
@pytest.mark.parametrize('case',[0,1])
def test_accuracy_rule_is_sufficient(out,case):
    r=out['claim_2_and_3'][case]['accuracy']; assert r['ppv_gap'] < 1e-12 and r['for_gap'] < 1e-12
@pytest.mark.parametrize('case',[0,1])
def test_separation_rule_is_sufficient(out,case):
    r=out['claim_2_and_3'][case]['separation']; assert r['ppv_gap'] < 1e-12 and r['for_gap'] < 1e-12
@pytest.mark.parametrize('case',[0,1])
def test_accuracy_matches_independent_optimizer(out,case): assert out['claim_2_and_3'][case]['accuracy']['objective_gap_vs_direct'] < 1e-6
@pytest.mark.parametrize('case',[0,1])
def test_separation_matches_independent_optimizer(out,case): assert out['claim_2_and_3'][case]['separation']['objective_gap_vs_direct'] < 1e-6
@pytest.mark.parametrize('case',[0,1])
def test_independent_accuracy_constraints(out,case):
    d=out['claim_2_and_3'][case]['accuracy']['direct_optimizer']; assert d['ppv_gap'] < 1e-5 and d['for_gap'] < 1e-5
@pytest.mark.parametrize('case',[0,1])
def test_independent_separation_constraints(out,case):
    d=out['claim_2_and_3'][case]['separation']['direct_optimizer']; assert d['ppv_gap'] < 1e-5 and d['for_gap'] < 1e-5
def test_deterministic_control_exhaustive(out): assert out['negative_controls']['deterministic_pairs_checked'] == 180
def test_randomization_is_necessary(out): assert out['negative_controls']['minimum_predictive_parity_gap_over_nonconstant_deterministic_pairs'] > .01
def test_threshold_control_rejected(out): assert out['negative_controls']['thresholding_rejected']
