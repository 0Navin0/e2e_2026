# this code is completely copied from ../verification/make_OU2024_samples_from_NERSC_data.py
from access_catalog import get_e2e_diffsky_MagLim_FluxLim_superset_data
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import numpy as np

#sys.path.insert(0, f"{base}/ROMAN_HLIS/merge_truth_and_main_galaxy_files/maglim_sampling_scripts")
#from maglim_sample import maglim_mag_cut, fluxLim_mag_cut

if __name__=="__main__":
    make_sompz_file = True 
    test_on_a_small_chunk_of_data = False#True#

    if make_sompz_file:
        fname = "magLim_fluxLim_supersample_sompz.parquet"
        cols = None
        fmt = "parquet"
        outputdir = Path("/work/nlc38/output_base/magLim_for_Boyan")
    else:
        fname = "magLim_fluxLim_supersample_samp_sel.parquet"
        cols = "default"
        fmt = "pandas"
        outputdir = Path("/work/nlc38/output_base/MagLim_supersample")

        if not test_on_a_small_chunk_of_data:
            sample_size = None
        else:
            sample_size = 200_000
   
        ## work on it
        #df = get_cardinal_Deep_data(sample_size)


    zbin_edges = np.array([0.00, 0.20, 0.40, 0.55, 0.70, 0.85, 0.95, 1.05]) #For extended plot before z=0.2
    zbins = list( zip(zbin_edges[0:-1], zbin_edges[1:]) )
    zmin = np.min(zbin_edges)
    zmax = np.max(zbin_edges)

    # setting up code for sompz dataset
    photoz = ds.field("phot_z")
    def get_filter(zl, zh):
        return ((photoz >= zl) & (photoz < zh))
    for ii, (zl,zh) in enumerate(zbins):
        dat = get_e2e_diffsky_MagLim_FluxLim_superset_data(fname, cols, fmt, get_filter(zl,zh))
        table = dat.to_table()
        pq.write_table(table, outputdir / f"sompz_file_magLim_zbin_{zl:0.2f}_{zh:0.2f}.parquet")

    ## -----------------
    ## define MagLim cut
    ## -----------------
    #idx, cuts = maglim_mag_cut("i", df.MAG_I.values, df.zobs.values, zmin, zmax)
    ## output dir
    #catoutdir = outputdir / f"{cuts}"
    #catoutdir.mkdir(parents=True, exist_ok=True)
    ## save the full sample once
    #dfMagLim= df.loc[idx,:]
    #assert ( all(dfMagLim.zobs>0) and np.isfinite(dfMagLim.to_numpy()).all() )

    #fname = catoutdir/f"Cardinal_MagLim_z_Range_{zmin:0.2f}_{zmax:0.2f}.csv"
    #dfMagLim.to_csv(fname, na_rep="NaN", index=False)
    #print("saved ", fname)

    #print(f"Number of galaxies in bins of redshifts, MagLim samples:")
    #sample_list = [f'{zl:0.2f}-{zh:0.2f}' for zl,zh in zbins]
    #count_list = [( (dfMagLim.zobs>=zl) & (dfMagLim.zobs<zh) ).sum() for zl,zh in zbins]
    #print(np.c_[sample_list, count_list])

    ## ------------------------------
    ##Now get the flux-limited sample
    ## ------------------------------
    #idx, cuts = fluxLim_mag_cut("i", df.MAG_I.values, df.zobs.values, zmin, zmax)
    #dfFluxLim= df.loc[idx,:]
    #assert ( all(dfFluxLim.zobs>0) and np.isfinite(dfFluxLim.to_numpy()).all() )

    #fname = catoutdir/f"Cardinal_fluxLim_z_Range_{zmin:0.2f}_{zmax:0.2f}.csv"
    #dfFluxLim.to_csv(fname, na_rep="NaN", index=False)
    #print("saved ", fname)

    #print(f"Number of galaxies in bins of redshifts, FlumLim samples:")
    #sample_list= [f'{zl:0.2f}-{zh:0.2f}' for zl,zh in zbins]
    #count_list = [( (dfFluxLim.zobs>=zl) & (dfFluxLim.zobs<zh) ).sum() for zl,zh in zbins]
    #print(np.c_[sample_list, count_list])

