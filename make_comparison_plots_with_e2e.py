# This code is derived from make_useful_plots_allDataTogether.py
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
plt.rcParams['text.usetex']=True
plt.rcParams["font.family"]="New Times Roman"
rng = np.random.default_rng(2026)

from scipy.stats import binned_statistic
def plot_binned_percentiles(x, y, bins=10, ax=None, color='blue', label=None, median_lw=1, extra_args_for_fill={}):
    """
    See binned_statistic doc, some helpful comments for this func are duplicated below.
    bins : int or sequence of scalars, optional
    """
    if ax is None:
        fig, ax = plt.subplots()

    # 1. Calculate the 50th percentile (Median)
    bin_means, bin_edges, _ = binned_statistic(x, y, statistic='median', bins=bins)

    # 2. Calculate the 16th and 84th percentiles (1-sigma equivalent)
    bin_16, _, _ = binned_statistic(x, y, statistic=lambda x: np.percentile(x, 16), bins=bins)
    bin_84, _, _ = binned_statistic(x, y, statistic=lambda x: np.percentile(x, 84), bins=bins)

    # 3. Get bin centers for plotting
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 4. Plot the median line
    ax.plot(bin_centers, bin_means, color=color, lw=median_lw, label=rf'{label} (Median)')

    # 5. Plot the shaded 1-sigma region
    ax.fill_between(bin_centers, bin_16, bin_84, color=color, label=rf'{label} 1$\sigma$-range', **extra_args_for_fill)

    return ax

def plot_shaded_region_betw_lines(x, constant=0.03, ax=None, color='purple', label=None):
    """
    If needed, you can also pass a non-constant value to arg `constant`. But
    this should match the size of the input arg `x`. That scenario arises when
    you don't have a constant value of delta_z/(1_ztrue). One example of where
    I need this -> I need to propagate some emperically computed value (binned
            statistic) for different ztrue bins from some arbitrary dataset
    where the reality is not like constant=0.03.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    if isinstance(constant, np.ndarray):
        strr = r"\sigma(z_{\rm true})"
    else:
        strr = f"{constant:0.2f}"

    # y = x + C(1 + x)
    y_upper = (1 + constant) * x + constant
    # y = x - C(1 + x)
    y_lower = (1 - constant) * x - constant

    # Plot the boundaries (optional, makes it look sharper)
    ax.plot(x, y_upper, color=color, linestyle='--', alpha=0.5)
    ax.plot(x, y_lower, color=color, linestyle='--', alpha=0.5)

    # Fill the region between them
    ax.fill_between(x, y_lower, y_upper, 
                    facecolor='none',
                    edgecolor=color,
                    hatch='||',
                    #hatch='\\\\',
                    label=label or rf'$|y-x| \leq {strr}(1+x)$'
                    )

    return ax

if __name__=="__main__":

    # information of OU2024 maglim mock being compared
    delta_z_by_OnePlus_z = 0.03
    base = Path("/hpc/group/cosmology/nlc38/ROMAN_HLIS")
    ou_catbase = Path(f"{str(base)}/merge_truth_and_main_galaxy_files/maglim_samples_0.03/zCut_i-magCut")
    band = "F184"
    # information of Cardinal based maglim mock being compared
    cardinalbase = f"{str(base)}/sample_selection_from_Cardinal/zCut_i-magCut"
    # location of e2e MagLim supersample
    e2ecatbase = Path("/work/nlc38/output_base")

    # redshift bins
    zbin_edges = np.array([0.20, 0.40, 0.55, 0.70, 0.85, 0.95, 1.05])
    zbins = list( zip(zbin_edges[:-1], zbin_edges[1:]) )
    zmin = np.min(zbin_edges)
    zmax = np.max(zbin_edges)

    # alldata(full maglim redshift-range) file: 
    fulldf = pd.read_csv(f"{str(ou_catbase)}/roman_{band}_maglim_zbin_{zmin:0.2f}-{zmax:0.2f}_combined.csv")
    # make sure the created catalog didn't have duplicates
    assert fulldf.galaxy_id.size == fulldf.galaxy_id.unique().size, "Go back to your sample creation code, there's a bug!"
    print(fulldf.columns)

    #full cardinal maglim data
    cardinal = pd.read_csv(f"{str(cardinalbase)}/Cardinal_MagLim_z_Range_0.00_1.05.csv")
    # since I had kept galaxies below zobs=0.2, for a separate plot
    cardinal = cardinal.loc[cardinal.zobs>=zmin]

    # downsample
    label=r"$z_{\rm photo}\in[%0.2f,%0.2f]$"%(zmin,zmax)
    r_idx = rng.choice(fulldf.photoz.size, size=500*(zbin_edges.size-1), replace=False)

    #e2e catalog
    e2edf = pd.read_parquet(
            e2ecatbase / "magLim_fluxLim_supersample_sompz.parquet", 
            columns=["phot_z", "spec_z"], 
            filters=[('phot_z', '>=', 0), ('phot_z', '<', 1.05)]
    )
    e2edf = e2edf.loc[(e2edf.phot_z>=0.2)]

    # plot scatter photoz vs redshift
    fig,ax = plt.subplots()
    r_idx = rng.choice(fulldf.photoz.size, size=int(fulldf.photoz.size/10), replace=False)
    #ax.scatter(fulldf.redshift.values[r_idx], fulldf.photoz.values[r_idx], color="#C0C0C0", s=1, alpha=0.5, label=label)
    bins = np.linspace(fulldf.redshift.values.min(), fulldf.redshift.values.max(), 20)
    ax = plot_binned_percentiles(
            cardinal.z.values, 
            cardinal.zobs.values, 
            bins=bins, ax=ax, 
            color='firebrick', 
            label=r"Cardinal Deep $z_{\rm photo}$",
            extra_args_for_fill=dict(
                alpha=0.5,
                zorder=10,
                )
    )
    ax = plot_binned_percentiles(
            fulldf.redshift.values, 
            fulldf.photoz.values, 
            bins=bins, 
            ax=ax, 
            color='blue', 
            label=r"OU2024 $z_{\rm photo}$",
            extra_args_for_fill=dict(
                alpha=0.2,
                zorder=1,
                )
    )

    ax = plot_binned_percentiles(
            e2edf.spec_z.values, 
            e2edf.phot_z.values, 
            bins=bins, 
            ax=ax, 
            color='green', 
            label=r"E2E-2026 $z_{\rm photo}$",
            extra_args_for_fill=dict(
                alpha=0.2,
                zorder=20,
                )
    )
    xx = np.linspace(bins.min(),bins.max(), 20)
    ax.plot(xx, xx, ls="--", c="k", label="y=x")
    ax.axvline(x=zmin, color='gray', linestyle='--', linewidth=1)
    ax.axvline(x=zmax, color='gray', linestyle='--', linewidth=1)
    #ax = plot_shaded_region_betw_lines(x=bins, constant=delta_z_by_OnePlus_z, ax=ax, color='#FF7F0E', label=None)
    ax.legend(loc="upper left", fontsize=12, title="MagLim selection")
    ax.set_xlabel(r"$z_{\rm true}$", fontsize=14)
    ax.set_ylabel(r"$z_{\rm photo}$", fontsize=14)
    pngf = f"truez_vs_photoz_fullSample_maglim_OU_{band}_vs_cardinal_vs_e2e2026.png"
    fig.savefig(pngf , dpi=200, bbox_inches="tight")
    print(f"saved {pngf}")
    plt.close(fig)

    # plot 3: pz hist
    step = 0.05 # my sensible binning choice
    edges = np.arange(zmin, zmax, step)
    figpz,axpz = plt.subplots()
    hist,_,_ = axpz.hist(e2edf.phot_z,  bins=edges, histtype="step", density=True, label="E2e-2026, 500sq deg", lw=2)
    hist,_,_ = axpz.hist(fulldf.photoz, bins=edges, histtype="step", density=True, label="OU2024", ls="--")
    hist,_,_ = axpz.hist(cardinal.zobs, bins=edges, histtype="step", density=True, label="Cardinal Deep")
    axpz.legend(loc="best", fontsize=14, title=label+"\n"+r"$z-$bin width=%0.3f"%step)
    axpz.set_ylabel("PDF", fontsize=14)
    axpz.set_xlabel(r"$z_{\rm photo}$", fontsize=14)
    pngf = f"zPDF_fullSample_maglim_OU_{band}_vs_cardinal_vs_e2e2026.png"
    figpz.savefig( pngf , dpi=200, bbox_inches="tight")
    print(f"saved {pngf}")
    plt.close(figpz)
