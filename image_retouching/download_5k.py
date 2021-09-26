import os
import sys
from urllib.request import urlretrieve

CURRENT_PATH=".../fivek_dataset/"
os.chdir(CURRENT_PATH)

img_lst=[]

with open('filesAdobe.txt', 'r') as f:
    for line in f.readlines():
        img_lst.append(line.rstrip("\n").replace(" ", "%20"))

with open('filesAdobeMIT.txt', 'r') as f:
    for line in f.readlines():
        img_lst.append(line.rstrip("\n").replace(" ", "%20"))

def cbk(a,b,c):
    per=100.0*a*b/c
    if per>100:
        per=100
    sys.stdout.write("progress: %.2f%%   \r" % (per))
    sys.stdout.flush()

for idx, i in enumerate(img_lst):
    # if idx < 3110:
    #     continue
    URL='https://data.csail.mit.edu/graphics/fivek/img/tiff16_c/'+i+'.tif'
    print('Downloading '+i+':')
    urlretrieve(URL, '.../fivek_c/'+i+'.tif', cbk)
