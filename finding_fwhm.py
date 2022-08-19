#!/usr/bin/env python3
# finding_fwhm.py
# computes FWHM from DIMM data
# references:
# Tokovinin, A, 2002, "From Differential Image Motion to Seeing"
# Sarazin, M, Roddier, F, 1989, "The ESO differential image motion monitor"

import sys
import numpy as np
from astropy.io import fits
from astropy.time import Time, TimeDelta
from scipy.ndimage import gaussian_filter

if len(sys.argv) < 3:
    print('Usage: finding_maxima.py <base filename> <number files> <start index>')
    sys.exit(  )

px2r = .0029/48/25.4 # pixels to radians for f/6 8" mirror with ASI290
#px2r = .0024/48/25.4 # pixels to radians for f/6 8" mirror with ASI183
r2as = 180 * 3600 / np.pi
px2as = px2r * r2as
print('Assuming ',px2as,' arcsecond units')
cbdim = 5 #centroid box dims
print('centroids computed on ',px2as*(2*cbdim+1),' arcsecond wide squares')
lam=.0005 # mm following Tokovinin
print('wavelength = ',lam*10**6,' nm')
d = 5.5*25.4
print('subaperture separation: ',d,' mm')
D = 2*25.4
print('subaperture diameter: ',D,' mm')

basefilename = sys.argv[1]
files4analysis = int(sys.argv[2])
startfile = 0
if len(sys.argv) >= 4:
  startfile = int(sys.argv[3])
unsat_indx = np.ones(files4analysis,dtype=np.bool8)
centroid1 = np.zeros((2,files4analysis))
centroid2 = np.zeros((2,files4analysis))
for i in range(files4analysis):
  # format integer i into 3 characters padded with leading zeros if necessary
  hdul = fits.open(basefilename + '{:0>3}'.format(startfile+i+1) + '.fits') 
  #print(hdul.info())
  imagedata = hdul[0].data
  if i == 0:
      tb = Time(hdul[0].header['DATE-OBS'])
      print('UTC start date of observation:\n',tb)
  if i == files4analysis - 1:
      te = Time(hdul[0].header['DATE-OBS'])
      print('UTC end date of observation:\n',te)
  hdul.close()

  imagedata = np.transpose(imagedata)
  imagedata = imagedata/16-1 
  imagedata = imagedata.astype(int)
  max_bright = np.amax(imagedata)
  ind = np.unravel_index(np.argmax(imagedata, axis=None), imagedata.shape)
  if max_bright == 4093:
    box = np.zeros((3,3),dtype=int)
    box = imagedata[ind[0]-1:ind[0]+2,ind[1]-1:ind[1]+2]
    if (np.sum(box)-4093)/8 > 2000:
        unsat_indx[i] = 0
        print('saturated px frame',i)
        continue
    else:
        print('hot pixel in frame ',i+1,' at ',ind)
        sys.exit()
  #print('brightest pixel: ',max_bright)
  blrimagedata = gaussian_filter(imagedata,sigma=1,output='double')
  blrind = np.unravel_index(np.argmax(blrimagedata, axis=None), blrimagedata.shape)
  if blrind[0] < 2*cbdim or blrind[1] < 2*cbdim or blrind[0] > imagedata.shape[0] - 2*cbdim or blrind[1] > imagedata.shape[1]-2*cbdim:
      unsat_indx[i] = 0
      print('1st maximum too close to window, frame ',i)
      continue
  #compute centroid around brightest pixel
  submtrx = imagedata[blrind[0]-cbdim:blrind[0]+cbdim+1,blrind[1]-cbdim:blrind[1]+cbdim+1]
  blrsubmtrx = blrimagedata[blrind[0]-cbdim:blrind[0]+cbdim+1,blrind[1]-cbdim:blrind[1]+cbdim+1]
  #print(blrsubmtrx)
  snr = np.amax(blrsubmtrx)/np.amin(blrsubmtrx)
  if snr < 1.5:
      unsat_indx[i] = 0
      print('insufficient SNR submatrix 1, frame ',i,': ',snr)
      #print(blrsubmtrx)
      continue
  centroid1[0,i] = np.ones(cbdim*2+1)@submtrx@np.arange(-cbdim,cbdim+1)/np.sum(submtrx)+blrind[0]
  centroid1[1,i] = np.arange(-cbdim,cbdim+1)@submtrx@np.ones(cbdim*2+1)/np.sum(submtrx)+blrind[1]

  blrsubmtrx[:,:] = 0
  blrind = np.unravel_index(np.argmax(blrimagedata, axis=None), blrimagedata.shape)
  #print(blrind)
  if blrind[0] < 2*cbdim or blrind[1] < 2*cbdim or blrind[0] > imagedata.shape[0] - 2*cbdim or blrind[1] > imagedata.shape[1]-2*cbdim:
      unsat_indx[i] = 0
      print('2nd maximum too close to window, frame ',i)
      continue
  submtrx = imagedata[blrind[0]-cbdim:blrind[0]+cbdim+1,blrind[1]-cbdim:blrind[1]+cbdim+1]
  blrsubmtrx = blrimagedata[blrind[0]-cbdim:blrind[0]+cbdim+1,blrind[1]-cbdim:blrind[1]+cbdim+1]
  blrsubmtrx = blrimagedata[blrind[0]-cbdim:blrind[0]+cbdim+1,blrind[1]-cbdim:blrind[1]+cbdim+1]
  snr = np.amax(blrsubmtrx)/np.amin(blrsubmtrx)
  if snr < 1.5:
      unsat_indx[i] = 0
      print('insufficient SNR submatrix 2, frame ',i,': ',snr)
      #print(blrsubmtrx)
      continue
  centroid2[0,i] = np.ones(cbdim*2+1)@submtrx@np.arange(-cbdim,cbdim+1)/np.sum(submtrx)+blrind[0]
  centroid2[1,i] = np.arange(-cbdim,cbdim+1)@submtrx@np.ones(cbdim*2+1)/np.sum(submtrx)+blrind[1]
#print('1st centroid matrix:\n',centroid1)
#print('2nd centroid matrix:\n',centroid2)
diffmtrx = centroid2-centroid1
diffmtrx *= px2r
diffmtrx=diffmtrx[:,unsat_indx]
print('unsaturated exposures:\n',np.sum(unsat_indx))
x_sgn_mtrx=diffmtrx[0,:] < 0
diffmtrx[:,x_sgn_mtrx]*=-1
#print('difference matrix:\n',diffmtrx)
slope = diffmtrx[1,:]/diffmtrx[0,:]
#print('vector of slopes of lines joining centroids:\n',slope)
avg_slope=np.mean(slope)
alpha=np.arctan(avg_slope)
#alpha=np.arctan(diffmtrx[1,0]/diffmtrx[0,0])
print('average slope angle: ',alpha*180/np.pi,' degrees')
rot_mtrx=np.zeros((2,2))
rot_mtrx[0,0]=np.cos(alpha)
rot_mtrx[1,1]=rot_mtrx[0,0]
rot_mtrx[0,1]=np.sin(alpha)
rot_mtrx[1,0]=-rot_mtrx[0,1]
#print('rotation matrix:\n',rot_mtrx)
#print('1st diff vector: ',diffmtrx[:,0])
#print('rotation of 1st diff vector:\n',rot_mtrx@diffmtrx[:,0])
diffmtrx=rot_mtrx@diffmtrx
#print('rotated diff matrix:\n',diffmtrx)
vl,corr1,corr2,vt = np.ravel(np.cov(diffmtrx))
print('correlation coefficient:\n',corr1/np.sqrt(vl*vt))
b=d/D
kl = .34*(1-.57/b**(1/3) - .04/b**(7/3)) # G-tilt constants from Tokovinin 2002
kt = .34*(1-.855/b**(1/3) + .03/b**(7/3))
print('proportionality constants:\nLongitudinal: ',kl,'; Transverse: ',kt)
Fl = .98*(D/lam)**.2*(vl/kl)**.6
Ft = .98*(D/lam)**.2*(vt/kt)**.6
dt = te - tb
print('Length of observation run: ',dt.sec,' seconds')
print('FWHM from longitudinal variance: ',Fl*r2as,' arcseconds')
print('FWHM from transverse variance: ',Ft*r2as,' arcseconds')
