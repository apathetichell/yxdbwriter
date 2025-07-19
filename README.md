# yxdbwriter
This is a python module to write yxdb (Alteryx native database files --- with cross compatability to Alteryx and the yxdb_reader module)

This repo builds upon the work in https://github.com/tlarsendataguy-yxdb/yxdb-py --- and downloading and installation of the yxdb_reader module is neccesary for testing.



# Simple usage
import pandas as pd
from yxdb_writer import YxdbWriter

df = pd.DataFrame({...})  # Any DataFrame, any schema
with YxdbWriter.from_dataframe("output.yxdb", df) as writer:
    pass


