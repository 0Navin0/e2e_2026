import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def get_mag(clean_flux):
    return 22.5 - 2.5 * np.log10(clean_flux)

def get_maglim_bin_edges():
    zbin_edges = np.array([0.00, 0.20, 0.40, 0.55, 0.70, 0.85, 0.95, 1.05]) #For extended plot before z=0.2
    zbins = list( zip(zbin_edges[0:-1], zbin_edges[1:]) )
    zmin = np.min(zbin_edges)
    zmax = np.max(zbin_edges)
    return zbins, zmin, zmax

if __name__=="__main__":
    snr_check_only=True

    catalog_dir = "/work/nlc38/output_base/magLim_for_Boyan"
    
    if snr_check_only:
        # Chun-Hao said he applied some SNR~18 cut on the source catalog, let's see what it looks like after MagLim sample selection
        # Actually, I've checked, the SNR ranges from ~11-220
        # infact, even more than top 75 percentile galaxies have SNR>48.159977
        # both columns snr and pgauss_s2n are same for now!
        """
        /work/nlc38/e2e_2026/diffsky_cat/e2e_catalog_noshear.parquet
                         snr     pgauss_s2n
        count  100000.000000  100000.000000
        mean      110.364707     110.364707
        std        67.908987      67.908987
        min        10.395516      10.395516
        25%        48.159977      48.159977
        50%        99.517939      99.517939
        75%       176.921880     176.921880
        max       217.147241     217.147241
        """
        for ii, (zl,zh) in enumerate(zbins):
            flname = f"{catalog_dir}/sompz_file_magLim_zbin_{zl:0.2f}_{zh:0.2f}.parquet"
            print(flname)
            df = pd.read_parquet(flname, columns=["snr", "phot_z", "spec_z", "z"])
            print(df.describe())


        """
So, all the galaxies in magLim at this point, are all very high SNR object. I'm not sure if this high SNR makes sense!
Trend is, as you go to higher redshifts, some low snr objects start to also enter, but even they are about SNR of 150.

/work/nlc38/output_base/magLim_for_Boyan/sompz_file_magLim_zbin_0.00_0.20.parquet
                snr        phot_z        spec_z             z
count  57349.000000  57349.000000  57349.000000  57349.000000
mean     217.136482      0.141843      0.140932      0.140932
std        0.010854      0.038017      0.035834      0.035834
min      216.839093      0.000000      0.014957      0.014957
25%      217.133496      0.110000      0.116480      0.116480
50%      217.139656      0.140000      0.146558      0.146558
75%      217.143286      0.170000      0.168856      0.168856
max      217.147114      0.190000      0.257891      0.257891

/work/nlc38/output_base/magLim_for_Boyan/sompz_file_magLim_zbin_0.20_0.40.parquet
                 snr         phot_z         spec_z              z
count  132790.000000  132790.000000  132790.000000  132790.000000
mean      217.121223       0.292130       0.292360       0.292360
std         0.026226       0.058347       0.059600       0.059600
min       216.693501       0.200000       0.110877       0.110877
25%       217.113426       0.250000       0.242884       0.242884
50%       217.129114       0.290000       0.290122       0.290122
75%       217.138244       0.350000       0.342081       0.342081
max       217.146948       0.390000       0.446122       0.446122

/work/nlc38/output_base/magLim_for_Boyan/sompz_file_magLim_zbin_0.40_0.55.parquet
                snr        phot_z        spec_z             z
count  89487.000000  89487.000000  89487.000000  89487.000000
mean     217.049544      0.473253      0.471719      0.471719
std        0.094867      0.040199      0.041468      0.041468
min      215.405365      0.400000      0.308156      0.308156
25%      217.026195      0.440000      0.437068      0.437068
50%      217.076867      0.470000      0.470333      0.470333
75%      217.107322      0.510000      0.507138      0.507138
max      217.145689      0.540000      0.651792      0.651792

/work/nlc38/output_base/magLim_for_Boyan/sompz_file_magLim_zbin_0.55_0.70.parquet
                 snr         phot_z         spec_z              z
count  100477.000000  100477.000000  100477.000000  100477.000000
mean      216.857690       0.618165       0.617081       0.617081
std         0.291360       0.043356       0.045030       0.045030
min       212.083302       0.550000       0.036607       0.036607
25%       216.797019       0.580000       0.578697       0.578697
50%       216.945746       0.620000       0.615941       0.615941
75%       217.032118       0.660000       0.654604       0.654604
max       217.143217       0.690000       0.741645       0.741645

/work/nlc38/output_base/magLim_for_Boyan/sompz_file_magLim_zbin_0.70_0.85.parquet
                snr        phot_z        spec_z             z
count  99701.000000  99701.000000  99701.000000  99701.000000
mean     216.308593      0.767960      0.767918      0.767918
std        0.888021      0.041578      0.044607      0.044607
min      204.939591      0.700000      0.248103      0.248103
25%      216.147532      0.740000      0.732096      0.732096
50%      216.609975      0.770000      0.767600      0.767600
75%      216.843352      0.800000      0.804738      0.804738
max      217.138176      0.840000      1.016153      1.016153

/work/nlc38/output_base/magLim_for_Boyan/sompz_file_magLim_zbin_0.85_0.95.parquet
                snr        phot_z        spec_z             z
count  71212.000000  71212.000000  71212.000000  71212.000000
mean     214.909225      0.896005      0.894719      0.894719
std        2.381029      0.027976      0.032600      0.032600
min      184.289364      0.850000      0.076078      0.076078
25%      214.238788      0.870000      0.868850      0.868850
50%      215.804121      0.900000      0.893791      0.893791
75%      216.445084      0.920000      0.919619      0.919619
max      217.131863      0.940000      1.047556      1.047556

/work/nlc38/output_base/magLim_for_Boyan/sompz_file_magLim_zbin_0.95_1.05.parquet
                snr        phot_z        spec_z             z
count  70273.000000  70273.000000  70273.000000  70273.000000
mean     212.949714      0.992594      0.993114      0.993114
std        4.489902      0.027722      0.032499      0.032499
min      166.748849      0.950000      0.024517      0.024517
25%      211.517695      0.970000      0.968466      0.968466
50%      214.670527      0.990000      0.993661      0.993661
75%      215.944486      1.020000      1.018346      1.018346
max      217.130633      1.040000      1.220036      1.220036
        """

    else:

        for ii, (zl,zh) in enumerate(zbins):
            flname = f"{catalog_dir}/sompz_file_magLim_zbin_{zl:0.2f}_{zh:0.2f}.parquet"
            print(flname)
            df = pd.read_parquet(flname)
            print(df.phot_z.size)
            ids = np.random.random(df.phot_z.size) <=0.1
            plt.scatter(df.phot_z.values[ids], get_mag(df.flux_gold_LSST_i.values)[ids], s=1, label=f"z: [{zl:0.2f}-{zh:0.2f})")
        plt.xlabel("phot_z")
        plt.ylabel("mag_gold_LSST_i")
        plt.legend(title="10% data points")
        plt.savefig("magLim_i_mag_vs_phot_z_sample_check.png", bbox_inches="tight")
