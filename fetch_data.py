import urllib.request
import zipfile
import os

try:
    os.remove('./traffic-signs-data.zip')
except OSError:
    pass
print('Beginning data download...')

url = 'https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip'  
urllib.request.urlretrieve(url, './traffic-signs-data.zip')
print('Data downloaded')
print('Unzipping')

z = zipfile.ZipFile('./traffic-signs-data.zip', 'r')
z.extractall('./data')
z.close()

try:
    os.remove('./traffic-signs-data.zip')
except OSError:
    pass

print('Done')