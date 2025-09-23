import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def MADfilter(df, column, ws, k=4.5):
    median = df[column].rolling(ws, center=True).median()
    MAD = np.abs(df[column] - median).rolling(ws, center=True).median()
    return df[column] < MAD*k+median

def connected_segments(valids):
    """
    Assumes valids==True is the segments you want to find, False values splits the segments.
    """
    arr = np.ones(valids.size)*-1
    c = 0
    i = 0
    while i < valids.size-1:
        while valids[i] and i < valids.size-1:
            arr[i] = c
            i += 1
        if valids[i-1] and not valids[i]:
            c += 1
        i += 1
    arr[-1] = arr[-2]
    return(arr)

def find_peaks2(df1, label='dW', ws=150):
    valid1 = MADfilter(df1, label, ws)
    valid1 = binary_dilation(np.invert(valid1),iterations=1)[2*ws:-2*ws]
    arr = connected_segments(valid1)
    df1 = df1.iloc[2*ws:-2*ws]
    df1 = df1.assign(group=arr)
    df1 = df1[df1.group>=0]
    peaks = df1.loc[:,(label,'group')].groupby('group').idxmax().values
    return(peaks.squeeze())


import argparse
parser = argparse.ArgumentParser(
                    prog='JPCP analysis Script',
                    description='Calculate effective joint spring constant',
                    epilog='The end')
parser.add_argument('path', nargs='?', type=str, default=r"example_data.csv", 
                    help="Path to folder or filename with exports. If left empty it plots an example.")
parser.add_argument('-i','--interval',dest='interval',type=int,nargs=2,help=' e.g. "-i 340 1005" interval to average over')
args = parser.parse_args()

dX = 0.2 # distance interval [meters] to run the analysis in.
g = 9.81 # Gravity constant
plot = True # Whether to call plt.show() that opens an interactive plot.
exp_shp = False # Whether to export ESRI shapefile
x_column = 'Chainage [m]'

path = Path(args.path)
df = pd.read_csv(path, index_col=0)
if args.interval:
    df = df.loc[(df[x_column] >= args.interval[0]) & (df[x_column] <= args.interval[1])]

lst=['DeflVel' in el for el in df.columns]
dvel_inds = df.columns[np.array(lst)]
df['dV'] = (df[dvel_inds[6]] - df[dvel_inds[7]]) * 1e-3

# This is my quick and dirty integration of delta-V (speed) to get delta-W (deflection)
res = df.iloc[1,0] - df.iloc[0,0]
df['dV'] = df['dV'].rolling(round(dX/res), center=True).mean()
dfm = df

dfm['dW'] = dfm['dV'] * dX

###################################################
# Detect peaks and extract a sub-table with these #
###################################################

## The method as I understood it from the article using a scipy function:
## It does not work so well on the data I tested, so maybe I am missing something!?
# dd = 50e-6
# lead_in = 50
# threshs = dfm['dW'].rolling(int(lead_in/dX), center=True).mean().values+dd
# peaks1, _ = find_peaks(dfm['dW'].values, distance=3, height=threshs)

## Here is my own peak detection algorithm
peaks1 = find_peaks2(dfm)

dfp = dfm.loc[peaks1,:]

###################################################
# Calculate the effective joint spring constant   #
###################################################
dfp['beta-eff [MPa]'] = dfp['Strain Gauge Right [kg]'] * g / (dfp['dW']*1e6)
dfp.to_csv(path.name+"_joints.csv")

###################################################
# Save a plot of the peaks detected               #
###################################################

f, ax = plt.subplots(1,1,figsize=(20,10))
ax.plot(dfm[x_column].values, dfm['dW']*1e6)
ax.plot(dfp[x_column], dfp['dW']*1e6, '*')

ax.set_title(path.name)
ax.set_xlabel(x_column)
ax.set_ylabel('delta W [µm]')
f.savefig(path.name+"_sci-tsd.pdf")
if plot:
    plt.show()

###################################################
# Export to ESRI Shapefile                        #
###################################################
if exp_shp:
    gdf = gpd.GeoDataFrame(dfp,geometry=gpd.points_from_xy(dfp['Longitude [dd.dddd]'], dfp['Latitude [dd.dddd]']))
    gdf=gdf.set_crs(4326)
    gdf.to_file(path.name+"_map")
