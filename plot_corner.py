import pandas as pd
import numpy as np
from corner import corner
import matplotlib.pyplot as plt
#plt.rcParams['text.usetex']=True
#plt.rcParams["font.family"]="New Times Roman"

from plot_maglim_supersample import get_mag, get_maglim_bin_edges

if __name__ == "__main__":
    plot_roman=True

    outdir = "corner_plots/"
    catalog_dir = "/work/nlc38/output_base/magLim_for_Boyan"

    zbins, zmin, zmax = get_maglim_bin_edges()

    read_cols = "spec_z phot_z pgauss_s2n g1 g2 reff pgauss_T_ratio".split(" ") + \
            [f"flux_gold_LSST_{ii}" for ii in "ugrizy"] + [f"flux_pgauss_LSST_{ii}" for ii in "ugrizy"] + \
            [f"flux_gold_{ii}" for ii in "YJH"] + [f"flux_pgauss_{ii}" for ii in "YJH"]

    lsst_cols = "spec_z phot_z pgauss_s2n g1 g2 reff pgauss_T_ratio".split(" ") + \
            [f"flux_gold_LSST_{ii}" for ii in "ugrizy"] + [f"flux_pgauss_LSST_{ii}" for ii in "ugrizy"]

    roman_cols = "spec_z phot_z pgauss_s2n g1 g2 reff pgauss_T_ratio".split(" ") + \
        [f"flux_gold_{ii}" for ii in "YJH"] + [f"flux_pgauss_{ii}" for ii in "YJH"]

    # exclude some columns as they don't have any dynamic range in e2e file
    excld = ["pgauss_T_ratio"]
    read_cols = [col for col in read_cols if col not in excld]
    lsst_cols = [col for col in lsst_cols if col not in excld]
    roman_cols = [col for col in roman_cols if col not in excld]


    for ii, (zl,zh) in enumerate(zbins):
        if plot_roman:
            cols_to_plot = roman_cols
            survey = "roman"
            # this should be matched with lsst_cols and roman_cols in correct order.
            plotlabels = [col for col in roman_cols if "flux" not in col] + \
                    [f"{col.split('_')[1]} {col.split('_')[2]}" for col in roman_cols if "flux" in col]
        else:
            cols_to_plot = lsst_cols
            survey = "lsst"
            # this should be matched with lsst_cols and roman_cols in correct order.
            plotlabels = [col for col in lsst_cols if "flux" not in col] + \
                    [f"{col.split('_')[1]} {col.split('_')[3]}" for col in lsst_cols if "flux" in col]
        print(plotlabels)
        foutname = outdir + f"corner_{survey}_magLim_{zl:0.2f}_{zh:0.2f}.png"

        flname = f"{catalog_dir}/sompz_file_magLim_zbin_{zl:0.2f}_{zh:0.2f}.parquet"
        print(flname)
        df = pd.read_parquet(flname, columns=cols_to_plot).sample(n=20000, random_state=42)

        # right now, there are no NaNs but in future, this part of  the codes
        # needs to be improved to handle and report bad values.
        print(f"Found NaN", df.isna().sum().sum())
        df = df.dropna()

        # here just directly overwriting the fluxes with magnitudes for now
        for col in [col for col in cols_to_plot if "flux" in col]:
            df.loc[:, col] = get_mag( df[col].values )
        
        fig = corner(
            df[cols_to_plot],
            bins = 50,
            smooth = 0.9,
            plot_datapoints=True,
            plot_density=True,
            # fill contours creates problems (created shaded regions in the
            # entire range) when overplotting two contours
            #fill_contours=True,
            labels=plotlabels,
            label_kwargs={"fontsize": 20},
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 16},
            title_fmt=".2f",
            color="C0",
            contourf_kwargs={'alpha': 0.3} #contour color
        )
        plt.savefig(foutname, bbox_inches="tight", dpi=200)
        plt.close()
        print("saved ", foutname)

"""
Index(['objectid', 'snr', 'g1', 'reff', 'pgauss_s2n', 'spec_z',
       'pgauss_T_ratio', 'z', 'ra', 'phot_z', 'g2', 'dec', 'flux_gold_Y',
       'flux_gold_LSST_z', 'flux_gold_LSST_r', 'flux_gold_LSST_i',
       'flux_gold_J', 'flux_gold_LSST_g', 'flux_gold_H', 'flux_gold_LSST_u',
       'flux_gold_LSST_y', 'flux_err_gold_LSST_y', 'flux_err_gold_LSST_z',
       'flux_err_gold_LSST_u', 'flux_err_gold_LSST_g', 'flux_err_gold_Y',
       'flux_err_gold_LSST_r', 'flux_err_gold_H', 'flux_err_gold_LSST_i',
       'flux_err_gold_J', 'flux_pgauss_Y', 'flux_pgauss_J',
       'flux_pgauss_LSST_r', 'flux_pgauss_H', 'flux_pgauss_LSST_y',
       'flux_pgauss_LSST_g', 'flux_pgauss_LSST_z', 'flux_pgauss_LSST_u',
       'flux_pgauss_LSST_i', 'flux_err_pgauss_LSST_g', 'flux_err_pgauss_J',
       'flux_err_pgauss_LSST_z', 'flux_err_pgauss_Y', 'flux_err_pgauss_H',
       'flux_err_pgauss_LSST_i', 'flux_err_pgauss_LSST_u',
       'flux_err_pgauss_LSST_y', 'flux_err_pgauss_LSST_r'],
      dtype='object')

"""
