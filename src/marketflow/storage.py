from __future__ import annotations

from pathlib import Path
import sqlite3
import pandas as pd

try:
    import duckdb
except Exception:
    duckdb = None


class DuckStore:
    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    def write_table(self, name: str, df: pd.DataFrame, replace: bool = True):
        if duckdb is not None:
            con = duckdb.connect(self.path)
            con.register('tmp_df', df)
            if replace:
                con.execute(f'CREATE OR REPLACE TABLE {name} AS SELECT * FROM tmp_df')
            else:
                con.execute(f'CREATE TABLE IF NOT EXISTS {name} AS SELECT * FROM tmp_df LIMIT 0')
                con.execute(f'INSERT INTO {name} SELECT * FROM tmp_df')
            con.close()
            return
        sqlite_path = str(Path(self.path).with_suffix('.sqlite'))
        con = sqlite3.connect(sqlite_path)
        df.to_sql(name, con, if_exists='replace' if replace else 'append', index=False)
        con.close()
