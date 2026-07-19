# this code template is copied from /hpc/group/cosmology/nlc38/ROMAN_HLIS/sample_selection_from_Cardinal/plot_binned_deltazByOnePlusz_dist.py
import sys
from pathlib import Path
# get cardinal deep data
from access_catalog import get_clean_cardinalDeep_data
sys.path.insert(0, "/hpc/group/cosmology/nlc38/ROMAN_HLIS/merge_truth_and_main_galaxy_files/maglim_sampling_scripts")
from make_useful_plots_allDataTogether import (binned_statistic)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.rcParams['text.usetex']=True
plt.rcParams["font.family"]="New Times Roman"

if __name__=="__main__":

    outdir = Path("/hpc/group/cosmology/nlc38/ROMAN_HLIS/sample_selection_from_Cardinal/zCut_i-magCut")


    # Setup full deep data
    fdf = get_clean_cardinalDeep_data(sample_size=100_000, use_cols=['z', 'zobs'])
    fdf = fdf.loc[(fdf.zobs>=0.2) & (fdf.zobs<1.05)] #keep within maglim z-range
    zfull = fdf.z.values
    zobsfull = fdf.zobs.values
    zdiff_full = zobsfull - zfull
    zdiff_ratio_full = zdiff_full /(1+zfull)

    # Maglim data
    df = pd.read_csv(outdir/ "Cardinal_MagLim_z_Range_0.00_1.05.csv")
    df = df.loc[(df.zobs>=0.2)] #keep within MagLim z-range
    z = df.z.values
    zobs = df.zobs.values
    zdiff = zobs - z
    zdiff_ratio = zdiff /(1+z)

    zstep = 0.05
    zl = z.min()
    zh = z.max()

    # use the same binning everywhere
    bins = np.linspace(zl, zh, int((zh-zl)/zstep))
    binned_zratio, zbin_edges, _ = binned_statistic(z, np.abs(zdiff_ratio), statistic='mean', bins=bins)
    zbin_centers = (zbin_edges[:-1] + zbin_edges[1:]) / 2
    assert all(zbin_edges==bins), "unexpected!"

    #from full dataset
    fbinned_zratio, fzbin_edges, _ = binned_statistic(zfull, np.abs(zdiff_ratio_full), statistic='mean', bins=bins)
    fzbin_centers = (fzbin_edges[:-1] + fzbin_edges[1:]) / 2
    assert all(fzbin_edges==bins), "unexpected!"

    fig, ax = plt.subplots(figsize=(6, 4.8))
    label = r"Binned, mean statistic: $\sigma= |\Delta z|/(1+z_{\rm true})$"
    ax.plot(zbin_centers, binned_zratio, ".-", ms=4, label= rf"MagLim: $z_{{\rm photo}} \in [{zobs.min():0.2f},{zobs.max():0.2f})$")
    ax.plot(fzbin_centers, fbinned_zratio, ":", label= rf"All: $z_{{\rm photo}} \in [{zobsfull.min():0.2f},{zobsfull.max():0.2f})$")
    ax.legend(loc="best", 
            fontsize=10, 
            title="Cardinal Deep-field\n" + label + f"\nBin-width={zstep:0.3f}",
            title_fontsize=12
            )
    ax.set_xlabel(r"$z_{\rm true}$", fontsize=16)
    ax.set_ylabel(r"$\left<\sigma\right>$", fontsize=16)
    ax.set_ylim(0, 0.014) #avoiding to show some outliar fraction
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.001))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.002))
    ax.grid(which='both', axis='both', color='gray', linestyle='--', alpha=0.3)
    pngf = f"{outdir}/deltaz_onePlusz_maglim_deep_onlyStats_vs_fullSample.png"
    fig.savefig(pngf , dpi=200, bbox_inches="tight")
    print(f"saved {pngf}")
    plt.close(fig)

