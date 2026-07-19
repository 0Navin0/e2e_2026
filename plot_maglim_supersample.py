import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def get_mag(clean_flux):
    return 22.5 - 2.5 * np.log10(clean_flux)

zbin_edges = np.array([0.00, 0.20, 0.40, 0.55, 0.70, 0.85, 0.95, 1.05]) #For extended plot before z=0.2
zbins = list( zip(zbin_edges[0:-1], zbin_edges[1:]) )
zmin = np.min(zbin_edges)
zmax = np.max(zbin_edges)

for ii, (zl,zh) in enumerate(zbins):
    df = pd.read_parquet(f"sompz_file_magLim_zbin_{zl:0.2f}_{zh:0.2f}.parquet")
    print(df.phot_z.size)
    ids = np.random.random(df.phot_z.size) <=0.1
    plt.scatter(df.phot_z.values[ids], get_mag(df.flux_gold_LSST_i.values)[ids], s=1, label=f"z: [{zl:0.2f}-{zh:0.2f})")
plt.xlabel("phot_z")
plt.ylabel("mag_gold_LSST_i")
plt.legend(title="10% data points")
plt.savefig("magLim_i_mag_vs_phot_z_sample_check.png", bbox_inches="tight")
