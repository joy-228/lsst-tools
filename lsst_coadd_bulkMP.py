# %%
'''
For the Rubin Science Platform at data.lsst.cloud.
Data Release: Data Preview 2
Container Size: large

Originally developed by Andrés A. Plazas Malagón and the Rubin Community Science team with input from Nate Lust for DP1.
Newer code developed for DP2 with input from Yusra AlSayyad, Jim Bosch, and the Rubin Community Science team.
Learning objective: Create a deep_coadd cutouts in bulk.
'''

import os, re 
import multiprocessing
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
from astropy.io import fits
import pandas as pd
import astropy.wcs as wcs
from astropy.coordinates import SkyCoord
from astropy import units as u

from lsst.rsp import RSPDiscovery
from lsst.daf.butler import Butler
import lsst.geom as geom

butler = Butler("dp2", collections="dp2") #Instantiate the butler.
assert butler is not None

discovery = RSPDiscovery("dp2")
service = discovery.get_tap_client()
assert service is not None

skymap = butler.get("skyMap") #Define the skymap.
username = os.environ.get('USER')
cutout_dir = '/home' / username / f'lsst_science/cutouts'
print(cutout_dir)

bands = {'u':mcm.magma,'g':mcm.cividis,'r':mcm.viridis,'i':mcm.inferno,'z':mcm.Greys_r}
npix = 144
box_size_deg = 0.5 * npix * 0.2 / 3600  # Define a box of half-width 0.5*npix around the target, then search for all patches within twice that size (2 * `box_size_deg`).

def query_atlas():
    """
    Query the TAP service to retrieve the necessary information about the relevant patches.

    Returns:
    -------
    atlas: A pandas DataFrame containing the patch information.
    """
    # Query the TAP service to retrieve the necessary information about the relevant patches.
    query = """SELECT lsst_patch, lsst_tract, s_dec, s_ra, s_region FROM dp2.CoaddPatches"""
    job = service.submit_job(query)
    job.run()
    job.wait(phases=['COMPLETED', 'ERROR'])
    print('Job phase is', job.phase)
    if job.phase == 'ERROR':
        job.raise_if_error()
    assert job.phase == 'COMPLETED'

    atlas = service.search(query).to_table().to_pandas()
    job.delete()
    del query
    
    return atlas

def match_tractpatch(pos):
    """
    Match the given positions to the corresponding tracts and patches in the skymap.

    Parameters:
    ----------
    pos: A list of `SpherePoint` objects representing the positions to match.

    Returns:
    -------
    A numpy array containing the matched tract and patch indices for each position.
    """
    tp = []
    for ii, p in pos.iterrows():
        point = geom.SpherePoint(p['ra'] * geom.degrees, p['dec'] * geom.degrees)
        tract = skymap.findTract(point)
        patch = tract.findPatch(point)
        
        tp.append((tract.tract_id, patch.getSequentialIndex()))
    return np.array(tp)


def load_bigcoadd(s_ra, s_dec):
    '''
    Retrieve the coadd images and their corresponding data IDs for each filter band at the specified sky coordinates.

    Parameters:
    ----------
    s_ra: Right Ascension of the object in degrees.

    s_dec: Declination of the object in degrees.

    Returns:
    -------

    coadd_get: A dictionary containing the coadd images for each filter band.

    '''
    coadd_get = {}
    for jj,filt in enumerate(bands):
                query = f"patch.region OVERLAPS POINT({s_ra}, {s_dec}) and band in ('{filt}')"
        
                try: qry_ds = butler.query_datasets("deep_coadd", where=query,)
                except: continue
        
                #did_get[filt] = qry_ds[0].dataId
                coadd_get[filt] =  butler.get(qry_ds[0])

    return coadd_get

    

def cutout_over_patch(ll, inpatch, coadd_get, variance_required=False):
    '''
    Generate cutouts for each source in the given patch using the provided coadd images.

    '''
    for ii, src in inpatch.iterrows():
        point = SkyCoord(ra=src['ra']*u.degree,dec=src['dec']*u.degree, frame='icrs')
        filename = cutout_dir / f'cutout_{src["objid"]}.npy'

        try: small_bbox = coadd_get['g'].bbox_from_sky_circle(point, box_size_deg*u.degree, clip=False)
        except: continue

        print(ii)

        #fig,ax=plt.subplots(1,3,figsize=(30,8))

        img_array = []
        coadds = {}
        for jj,filt in enumerate(bands):
            
                try: coadds[filt] = coadd_get[filt][small_bbox] 
                except: break

                if variance_required: 
                    img_array.append([coadds[filt].image.array,coadds[filt].variance.array])
                else:
                    img_array.append([coadds[filt].image.array])

                #m, s = np.mean(img_array[jj][0]), np.std(img_array[jj][0])
                #ax[jj].imshow(img_array[jj][0], interpolation='nearest', cmap=bands[filt], vmin=m-s, vmax=m+s, origin='lower')
                #del m,s
        #plt.show()

        np.save(filename, np.array(img_array))
        del coadds, img_array, small_bbox

def loop_over_patches(ll, up, sample, patches_table, variance_required=False):
    # For each unique patch, retrieve the corresponding coadd data, and generate cutouts for each source in that patch.

    inpatch = sample[(sample['tract']==up['tract'])&(sample['patch']==up['patch'])]
      
    tp_link = patches_table[(patches_table['lsst_tract']==up['tract'])&(patches_table['lsst_patch']==up['patch'])]
    s_ra,s_dec = (tp_link['s_ra'].values)[0],(tp_link['s_dec'].values)[0]
    print(ll, s_ra, s_dec, 'tract number:', up['tract'],'patch number:', up['patch'])

    coadd_get = load_bigcoadd(s_ra, s_dec)
    cutout_over_patch(ll, inpatch, coadd_get, variance_required=variance_required)

    del inpatch, coadd_get, did_get, tp_link
    
if __name__ == '__main__':
    """
    Main execution block: process galaxy sample and generate ugriz cutouts for each galaxy in the sample. The cutouts are saved as .npy files for further analysis.
    """

    # Load the sample of galaxies from a CSV file containing their coordinates and pre-determined patch and tract information.
    samplefile = 'v1sdwarf_list.csv'
    sample = pd.read_csv(samplefile, dtype={'objid': str})
    #print('Shape of sample dataframe:',sample0.shape)
    #sample = sample0.sample(n=5000, random_state=5)
    #print(sample.shape)

     # Check if the patches_table.csv file exists; if not, query the TAP service to create it.
    if os.path.exists('patches_table.csv'):
            patches_table = pd.read_csv('patches_table.csv')
    else:
            patches_table = query_atlas()
            patches_table.to_csv('patches_table.csv', index=False)

    # Check if the 'patch' column exists in the sample dataframe; if not, compute the corresponding tract and patch for each galaxy using the skymap.
    if 'patch' not in sample.columns: 
        tp = match_tractpatch(sample[['ra','dec']]) 
        sample['tract'],sample['patch'] = tp[:,0], tp[:,1]
        sample.to_csv(samplefile, index=False)

        del tp

    existings = [re.search(r'\d+',f).group() for f in os.listdir(cutout_dir) if os.path.isfile(os.path.join(cutout_dir, f))]
    sources = (sample[sample['objid'].isin(existings)].drop_duplicates(subset='objid', keep='first')).reset_index(drop=True)
    print(len(existings), sources.shape)
    
    # Select unique patches from the tract to avoid redundant processing.
    unique_patch = pd.concat([sources['patch'],sources['tract']]).unique()
    print(unique_patch)
    print('No. of unique patches:',len(unique_patch))

    patch_input = [(ll,up) for ll,up in enumerate(unique_patch)]
    run_wrapper = partial(loop_over_patch, sample=sources, patches_table=patches_table, variance_required=False)

    with multiprocessing.Pool(4) as pool:

        pool.starmap(run_wrapper, patch_input)
