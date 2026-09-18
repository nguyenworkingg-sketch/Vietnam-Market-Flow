import pandas as pd
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from marketflow.scoring import weighted_score


def test_rank_order():
    d=pd.DataFrame({'date':[pd.Timestamp('2026-01-01')]*3,'x':[1,2,3]})
    s=weighted_score(d,{'x':1.0})
    assert list(s.round(2)) == [33.33,66.67,100.00]
