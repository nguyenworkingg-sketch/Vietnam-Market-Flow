from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from marketflow.features import add_stock_features


def test_atr_features_scale_with_each_stocks_own_range():
    dates = pd.bdate_range('2026-01-02', periods=35)
    bench = pd.DataFrame({
        'date': dates,
        'close': [100+i*.1 for i in range(len(dates))],
    })
    rows=[]
    for ticker, half_range in [('LOW', .5), ('HIGH', 3.0)]:
        for i,d in enumerate(dates):
            close=20+i*.05
            rows.append({
                'date':d,'ticker':ticker,
                'open':close-.05,'high':close+half_range,'low':close-half_range,
                'close':close,'volume':1_000_000,'value':close*1_000_000,
            })
    out=add_stock_features(pd.DataFrame(rows),bench)
    latest=out.sort_values('date').groupby('ticker').tail(1).set_index('ticker')
    assert latest.loc['HIGH','atr20'] > latest.loc['LOW','atr20']
    assert latest.loc['HIGH','atr_pct_20'] > latest.loc['LOW','atr_pct_20']
    assert latest.loc['LOW','atr_pct_20'] > 0
