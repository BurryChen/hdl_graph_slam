#!/usr/bin/python
import tf
import numpy as np

import matplotlib.pyplot as plt
from pylab import *

odom_dir=sys.argv[1]
#seq=sys.argv[2]
seqs=['00','01','02','03','04','05','06','07','08','09','10']
#seqs=['00','01','04','10']
for seq in seqs:
 file_odom=odom_dir+'/data/KITTI_'+seq+'_odom.txt'
 file_gt='/home/whu/data/loam_KITTI/gt/'+seq+'.txt'
 odom = np.loadtxt(file_odom)   
 gt = np.loadtxt(file_gt)   
 error=odom-gt

 dx = [x[3] for x in error]  
 dy = [x[7] for x in error] 
 dz = [x[11] for x in error] 
 #for i in range(len(error)):
 #    print(error[i][3],error[i][7],error[i][11])

 plt.figure(seq)
 plt.plot(dx,"g")
 plt.plot(dy,"b")
 plt.plot(dz,"r")
 plt.title('seq '+seq+' error :g-x-right,b-y-down,r-z-front')
 fig=odom_dir+'/errors/'+'error_frame_'+seq+'.png'
 savefig(fig)
 print fig 
#plt.show()

