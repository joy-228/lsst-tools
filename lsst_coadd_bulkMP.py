'''
Code developed to create deep_coadd cutouts in bulk for a sample of galaxies. 
The code queries the TAP service to retrieve patch information, matches galaxy positions to tracts and patches, 
loads coadd images, and generates cutouts for each source in the specified bands. 
The cutouts are saved as .npy files for further analysis.
Data Release: Data Preview 2
Container Size: large

Author: Joy Bhattacharyya with inputs from Yusra AlSayyad, Jim Bosch, and the Rubin Community Science team.
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
cutout_dir = '/home/' + username + f'/lsst_science/cutouts_cosmos/'
print(cutout_dir)

bands = {'g':mcm.cividis,'r':mcm.viridis,'i':mcm.inferno,'z':mcm.Greys_r}
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
        filename = cutout_dir + f'cutout_{src["objid"]}.npy'

        flag = False
        try: small_bbox = coadd_get['g'].bbox_from_sky_circle(point, box_size_deg*u.degree, clip=False)
        except: flag = True

        print(ii,', objid:',src["objid"])

        #fig,ax=plt.subplots(1,3,figsize=(30,8))

        img_array = []
        coadds = {}
        for jj,filt in enumerate(bands):
            
                try: coadds[filt] = coadd_get[filt][small_bbox] 
                except: 
                    flag = True
                    break

                if variance_required: 
                    img_array.append([coadds[filt].image.array,coadds[filt].variance.array])
                else:
                    img_array.append([coadds[filt].image.array])

                #m, s = np.mean(img_array[jj][0]), np.std(img_array[jj][0])
                #ax[jj].imshow(img_array[jj][0], interpolation='nearest', cmap=bands[filt], vmin=m-s, vmax=m+s, origin='lower')
                #del m,s
        #plt.show()

        if not flag: np.save(filename, np.array(img_array))
        elif flag: print('Cutout failed for objid:',src["objid"])
        del coadds, img_array, small_bbox

def loop_over_patches(ll, up, sample,  variance_required=False):
    '''
    Loop over each unique patch and generate cutouts for all sources within that patch.
    '''
    inpatch = sample[(sample['tract']==up['tract'])&(sample['patch']==up['patch'])]    
    print(ll, 'tract number:', up['tract'],'patch number:', up['patch'],'patch sources:', inpatch.shape[0])

    # If there are sources in the patch, load the coadd data and generate cutouts for each source.
    if not inpatch.empty:
        
        coadd_get = load_bigcoadd(up['s_ra'], up['s_dec'])
        print('Patch coadd loaded')
        
        cutout_over_patch(ll, inpatch, coadd_get, variance_required=variance_required)
        print('Patch sources cutout saved')
    
        del inpatch, coadd_get
    
if __name__ == '__main__':
    """
    Main execution block: process galaxy sample and generate ugriz cutouts for each galaxy in the sample. The cutouts are saved as .npy files for further analysis.
    """

    # Load the sample of galaxies from a CSV file containing their coordinates and pre-determined patch and tract information.
    samplefile = 'cosmos_search.csv'
    sample = pd.read_csv(samplefile, dtype={'objid': str})
    #print('Shape of sample dataframe:',sample0.shape)
    #sample = sample0.sample(n=500, random_state=5)
    #print(sample.shape)
    assert 'ra' in sample.columns and 'dec' in sample.columns, "Sample must contain 'ra' and 'dec' columns."

     # Check if the patches_table.csv file exists; if not, query the TAP service to create it.
    if os.path.exists('patches_table.csv'):
            patches_table = pd.read_csv('patches_table.csv')
    else:
            patches_table = query_atlas()
            patches_table = patches_table.rename(columns={'lsst_patch': 'patch','lsst_tract': 'tract'})
            patches_table.to_csv('patches_table.csv', index=False)

    # Check if the 'patch' column exists in the sample dataframe; if not, compute the corresponding tract and patch for each galaxy using the skymap.
    if 'patch' not in sample.columns: 
        tp = match_tractpatch(sample[['ra','dec']]) 
        sample['tract'],sample['patch'] = tp[:,0], tp[:,1]
        sample.to_csv(samplefile, index=False)

        del tp

    # Identify existing cutouts in the output directory to avoid redundant processing. Only process galaxies that do not already have cutouts saved.
    existings = [re.search(r'\d+',f).group() for f in os.listdir(cutout_dir) if os.path.isfile(os.path.join(cutout_dir, f))]
    sources = (sample[~sample['objid'].isin(existings)]).reset_index(drop=True)
    print('Exisiting cutouts in directory:',len(existings),'Cutouts to be extracted:', sources.shape[0])

    # Select unique patches from the tract table.
    unique_tp = sample[['patch', 'tract']].drop_duplicates()
    unique_tp_table = unique_tp.merge(patches_table,how='left',on=['tract','patch'])
    print('No. of unique patches:',unique_tp_table.shape[0])

    del unique_tp, patches_table

    # Prepare the input for multiprocessing: create a list of tuples containing the index and corresponding patch information for each unique patch.
    patch_input = [(ll,up) for ll,up in unique_tp_table.iterrows()]

    # Use multiprocessing to process each patch in parallel, generating cutouts for all sources within each patch. The `loop_over_patches` function is called for each patch, and the results are saved as .npy files in the specified output directory.
    run_wrapper = partial(loop_over_patches, sample=sources,  variance_required=False)

with multiprocessing.Pool(processes=2, maxtasksperchild=1) as pool:
        pool.starmap(run_wrapper, patch_input)
