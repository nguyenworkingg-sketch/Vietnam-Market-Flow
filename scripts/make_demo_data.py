from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
out = ROOT/'sample_data'; out.mkdir(exist_ok=True)
rng = np.random.default_rng(42)
dates = pd.bdate_range('2024-01-02', periods=430)
sectors = {'AAA':'Ngân hàng','BBB':'Ngân hàng','CCC':'Điện','DDD':'Điện','EEE':'Dầu khí','FFF':'Dầu khí','GGG':'CNTT','HHH':'BĐS'}
rows=[]
for i,(ticker,sector) in enumerate(sectors.items()):
    drift = 0.00025 + i*0.00003
    if ticker in ['EEE','CCC']:
        drift += 0.00035
    rets = rng.normal(drift, 0.018, len(dates))
    if ticker in ['EEE','CCC']:
        rets[-25:] += np.linspace(0.0005,0.004,25)
    px = 20*np.exp(np.cumsum(rets))
    vol = rng.lognormal(20.2,0.4,len(dates)).astype(int)
    if ticker in ['EEE','CCC']:
        vol[-25:] = (vol[-25:]*np.linspace(1.1,2.0,25)).astype(int)
    for d,p,v in zip(dates,px,vol):
        rows.append([d,ticker,p*0.995,p*1.01,p*0.99,p,v,p*v])
pd.DataFrame(rows,columns=['date','ticker','open','high','low','close','volume','value']).to_csv(out/'prices.csv',index=False)
pd.DataFrame([{'ticker':t,'exchange':'HOSE','sector':s,'name':t} for t,s in sectors.items()]).to_csv(out/'universe.csv',index=False)
bret = rng.normal(0.0002,0.01,len(dates)); bpx=1000*np.exp(np.cumsum(bret))
pd.DataFrame({'date':dates,'open':bpx*0.998,'high':bpx*1.006,'low':bpx*0.994,'close':bpx,'volume':1_000_000}).to_csv(out/'benchmark.csv',index=False)
print(out)
