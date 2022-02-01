import requests
from bs4 import BeautifulSoup
import pandas as pd
import os 

from .justdoit import available

def download_hitran(directory):
    """
    Downloads index of refraction data from hitran 

    Parameters
    ----------
    directory : str 
        Directory where raw files should go 
    """
    species_in_virga = available()
    
    hitran_url = 'https://hitran.org/data/Aerosols-2016/ascii/exoplanets/'

    soup = BeautifulSoup(requests.get(hitran_url).text)

    downloaded=[]

    for a in soup.find_all('a'):
        if '.dat' in a['href']: 
            to_get = os.path.join(hitran_url,a['href'])
            r = requests.get(to_get, allow_redirects=True)
            open(os.path.join(directory,a['href']), 'wb').write(r.content)
            downloaded +=[to_get]
    print('Downloaded {0} Files from HITRAN to {1}'.format(len(downloaded), directory))