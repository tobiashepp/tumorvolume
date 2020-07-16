import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

work_dir = Path('/mnt/qdata/raheppt1/data/tumorvolume')
organ_csv_path = work_dir/'processed/ctorgans/ctorgans_petct.csv'
tumor_csv_path = work_dir/'processed/petct/petct_analysis.csv'

df_organ = pd.read_csv(organ_csv_path, na_values='--')
df_tumor = pd.read_csv(tumor_csv_path, na_values='--')
df = pd.merge(df_organ, df_tumor, how='left', on=['key', 'project'])

import seaborn as sns
sns.set(style="darkgrid")
g = sns.jointplot("volume_tumor", "volume_liver", data=df,
                  kind="reg", truncate=False,
                  xlim=(0, 60), ylim=(0, 12),
                  color="m", height=7)


g = sns.lmplot('uptake_tumor', 'uptake_spine_mean', data=df)
g.set(xlim=(0, 0.5e6), ylim=(1.0, 3.0), xscale='log')
g.set(yscale='log')

g = sns.lmplot('volume_tumor', 'uptake_liver_mean', data=df)
g.set(xlim=(0, 0.8e6))
g.set(ylim=(0, 4.0))