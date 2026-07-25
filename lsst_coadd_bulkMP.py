# %%
'''
For the Rubin Science Platform at data.lsst.cloud.
Data Release: Data Preview 1
Container Size: large

Originally developed by Andrés A. Plazas Malagón and the Rubin Community Science team with input from Nate Lust. 
Learning objective: Create a large custom `deep_coadd` cutout.
#Packages: `lsst.daf.butler`, `lsst.ip.diffim.GetTemplateTask`.
'''

import os
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

from lsst.rsp import get_tap_service
from lsst.daf.butler import Butler
import lsst.afw.display as afwDisplay
from lsst.geom import Box2I, Point2I, SpherePoint, degrees, Point2D
from lsst.sphgeom import Box as SphBox
from lsst.ip.diffim import GetTemplateTask
from lsst.pipe.tasks.makeDiscreteSkyMap import MakeDiscreteSkyMapTask

butler = Butler("dp1", collections="LSSTComCam/DP1") #Instantiate the butler.
assert butler is not None

service = get_tap_service("tap")
assert service is not None

skymap = butler.get("skyMap") #Define the skymap.
username = os.environ.get('USER')
bands = {'u':mcm.magma,'g':mcm.cividis,'r':mcm.viridis,'i':mcm.inferno,'z':mcm.Greys_r}
box_size_deg = 0.004 #Define a box size of 0.1 degrees around the BCG, then search for all patches within twice that size (2 * `box_size_deg`).

def gen_smallbox(new_wcs,ra,dec):
    '''
    Generate a small bounding box around the given location in pixel coordinates using the WCS of the coadd image.

    Parameters:
    ----------

    new_wcs: WCS object of the coadd image.

    ra: Right Ascension of the object in degrees.

    dec: Declination of the object in degrees.

    Returns:
    -------

    Box2I: A bounding box in pixel coordinates around the object location.
    '''
    return  Box2I(
        Point2I(new_wcs.skyToPixel(SpherePoint(ra - box_size_deg,
                                               dec - box_size_deg, degrees))),
        Point2I(new_wcs.skyToPixel(SpherePoint(ra + box_size_deg,
                                               dec + box_size_deg, degrees))),
        )     

def gen_wcsbox(pos_cont,point):
    '''
    Generate a new WCS and bounding box for the given position container and point.

    Parameters:
    ----------
    pos_cont: A list containing the WCS and bounding box of the coadd image.

    point: A SpherePoint object representing the location of the object in the sky.

    Returns:
    -------
    new_wcs: The new WCS object for the coadd image.

    small_box: A bounding box in pixel coordinates around the object location.
    '''
    cd_wcs = pos_cont[0][0]
    discrete_skymap_config = MakeDiscreteSkyMapTask().ConfigClass() #Make a skymap that contains all input data.
    discrete_skymap_config.skyMap.pixelScale = cd_wcs.getPixelScale(Point2D(point)).asArcseconds() #Set the plate scale at the BCG location in the new skymap.
    discrete_skymap_config.skyMap.projection = 'TAN' #Set the WCS projection type. The WCS of `deep_coadd` images are `TAN` projections([see the DP1 documentation]

    new_skymap = MakeDiscreteSkyMapTask(config=discrete_skymap_config).run(pos_cont).skyMap #Compute the new skymap.
    new_wcs = new_skymap[0].wcs     #Get the new WCS from the single tract discrete skymap.
    small_box = gen_smallbox(new_wcs,(point.getRa()).asDegrees(),(point.getDec()).asDegrees())

    return new_wcs,small_box
    

def reconstitute_fits(uname,ra,dec,objid,ll):
    img_arr = [[],[],[]]
    
    for ii, key in enumerate(bands):    
            fnm = '/home/' + uname + '/lsst_science/cutouts/temp_{}{}.fits'.format(key,ll)
            hdul = fits.open(fnm)
            w = wcs.WCS(hdul[0].header)
            pos_coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
            wc = [int(i) for i in w.world_to_pixel(pos_coord)]
            if ii==0:    img_hdr = [hdul[j].header for j in range(1,4)]
            for j in range(3): img_arr[j].append((hdul[j+1].data)[wc[1]-64:wc[1]+64,wc[0]-64:wc[0]+64])
        
            hdul.close()
            os.remove(fnm)
    #fnm = '/home/' + uname + '/lsst_science/cutouts/cutout_{:.6f}_{:.6f}.fits'.format(ra,dec)
    #fits.writeto(fnm, np.array(img_arr[0]), header=img_hdr[0],overwrite=True)
    #for j in [1,2]: fits.append(fnm, np.array(img_arr[j]), img_hdr[j]) 

    fnm = '/home/' + uname + f'/lsst_science/cutouts/cutout_{int(objid)}.npy'
    data_uint8 = np.array(img_arr[0])

    np.save(fnm, data_uint8)


def get_coadd_basics(s_ra, s_dec):
    '''
    Retrieve the coadd images and their corresponding data IDs for each filter band at the specified sky coordinates.

    Parameters:
    ----------
    s_ra: Right Ascension of the object in degrees.

    s_dec: Declination of the object in degrees.

    Returns:
    -------

    coadd_get: A dictionary containing the coadd images for each filter band.

    did_get: A dictionary containing the data IDs for each filter band.

    position_container: A list containing the WCS and bounding box of the coadd image for the 'g' band.
    '''
    coadd_get,did_get = {},{}
    for jj,filt in enumerate(bands):
            query = f"patch.region OVERLAPS POINT({s_ra}, {s_dec}) and band in ('{filt}')"
            try:
                qry_ds = butler.query_datasets("deep_coadd",where=query)
            except: continue                     
            #band_container = sorted_data.setdefault(did["band"], list())
            #band_container.append()
        
            did_get[filt] = qry_ds[0].dataId
            coadd_get[filt] = butler.get(qry_ds[0])

            patchInfo = skymap[did_get[filt]["tract"]][did_get[filt]["patch"]]
            if filt=='g': position_container = [(patchInfo.wcs, patchInfo.outer_bbox)]
    return coadd_get,did_get,position_container


def get_coadd_cutouts(ll, inpatch, coadd_get, did_get, position_container):
    for ii, source in inpatch.iterrows():
        ra,dec = source['ra'], source['dec']
        point = SpherePoint(ra * degrees,dec * degrees)
        #tract_number,patch_number = tract.tract_id,patch.getSequentialIndex()
        tract_number,patch_number = int(source['tract']),int(source['patch'])

        new_wcs,small_box = gen_wcsbox(position_container,point)
    
        #fig,ax=plt.subplots(1,5,figsize=(30,8))
    
        for jj,filt in enumerate(bands):
            #coadd = butler.get(mapping[tract_number][0]) #Get the coadd from the data reference.
            #cd_wcs = coadd.getWcs()  #Get the WCS from the coadd.
       
            #Create a new coadd image in the band defined above that is a combination of the inputs contained within the defined box. 
            #Note that the run method modifies the bounding box in place, so a copy is made.
            getTemplateTask = GetTemplateTask()
            new_image = getTemplateTask.run(
            coaddExposures={tract_number:[coadd_get[filt]]},
            bbox=Box2I(small_box),wcs=new_wcs,
            dataIds={tract_number:[did_get[filt]]},
            physical_filter=filt,)
        
            fnm = '/home/' + username + '/lsst_science/cutouts/temp_{}{}.fits'.format(filt,ll)
            afwDisplay.writeFitsImage(fnm,  new_image.template.maskedImage,  wcs=new_image.template.wcs)
        
            img_array = new_image.template.image.array
            #m, s = np.mean(img_array), np.std(img_array)
    
            del img_array, fnm, new_image
        reconstitute_fits(username,ra,dec,source['objid'],ll)
        #plt.show()

def run_cutouts(ll, up, sample, results, ):
 
    # For each unique patch, retrieve the corresponding coadd data, and generate cutouts for each source in that patch.

    inpatch = sample[sample['patch']==up]
    print('inpatch shape:',inpatch.shape)
    tp_link = results[(results['lsst_tract']==sample['tract'][0])&(results['lsst_patch']==up)]
    s_ra, s_dec = (tp_link['s_ra'].values)[0], (tp_link['s_dec'].values)[0]
    print(ll, s_ra, s_dec, 'patch number:',up)


    coadd_get, did_get, position_container = get_coadd_basics(s_ra, s_dec)
    get_coadd_cutouts(ll, inpatch, coadd_get, did_get, position_container)

    del inpatch, coadd_get, did_get, position_container

    print(ll, 'Tract')
    
if __name__ == '__main__':
    """
    Main execution block: process galaxy sample and generate ugriz cutouts for each galaxy in the sample. The cutouts are saved as .npy files for further analysis.
    """

    # Load the sample of galaxies from a CSV file containing their coordinates and pre-determined patch and tract information.
    sample0 = pd.read_csv('v1sdwarf_list.csv')
    #print('Shape of sample dataframe:',sample0.shape)

    sample = sample0.sample(n=5000, random_state=5)
    print(sample.shape)

    # Query the TAP service to retrieve the necessary information about the relevant patches.
    query = "SELECT lsst_tract,lsst_patch,s_ra,s_dec, s_region " \
            "FROM dp1.CoaddPatches " 
    results = service.search(query).to_table().to_pandas()

    # Select unique patches from the tract to avoid redundant processing.
    unique_patch = sample['patch'].unique()
    print('No. of unique patches:',len(unique_patch))

    patch_input = [(ll,up) for ll,up in enumerate(unique_patch)]

    run_wrapper = partial(run_cutouts, sample=sample, results=results)

    with multiprocessing.Pool(4) as pool:

        pool.starmap(run_wrapper, patch_input)

# %%
'''
        SIGMA_TO_FWHM = 2.0*np.sqrt(2.0*np.log(2.0))
    
        center_pix_new = new_wcs.skyToPixel(point)
        point_new = Point2D(center_pix_new)
        newimage_psf = new_image.template.getPsf()
        psf_sigma_new = newimage_psf.computeShape(point_new).getDeterminantRadius()
    
        print(f"newimage FWHM = {psf_sigma_new * SIGMA_TO_FWHM: .3f} pixels")
        '''

