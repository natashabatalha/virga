import os
import pandas as pd
import numpy as np
import glob
from . import justdoit as jdi
from . import justplotit as jpi


def configure_ior_choices():
    return {
    "Al2O3":{
        "inhouse_file":"Al2O3_LXMIE.dat",
        "reference":"\\cite{Koike1995Al2O3,Begemann1997Al2O3}",
        "url":"https://raw.githubusercontent.com/exoclime/LX-MIE/master/compilation/Al2O3.dat", 
        "usr_note":"Data source from LX-MIE, Kitzmann & Heng (2018) \\cite{Kitzmann2018optical}",
        "pandas_kwargs":
            {
                "sep":"\s+", 
                "skiprows":3, 
                "names":['um','n','k']
            }
        },  
    "CH4":{
        "inhouse_file":"CH4(l)_Martonchik.csv",
        "reference":"\\cite{Martonchik1994CH4}",
        "url":"https://www.osapublishing.org/ao/abstract.cfm?uri=ao-33-36-8306", 
        "usr_note":"Data stripped from table. Temperature used=90 K. Do not use micron column as PDF version of paper did not go to high enough decimal precision.",
        "pandas_kwargs":
            {
                "skiprows":2, 
                "names":['cm-1','n112','k112','n','k','um']
            }
        }, 
    "CaTiO3":{
        "inhouse_file":"CaTiO3_LXMIE.dat",
        "reference":"\\cite{Posch2003TiOs,ueda1998CaTiO3}",
        "url":"https://raw.githubusercontent.com/exoclime/LX-MIE/master/compilation/CaTiO3.dat", 
        "usr_note":"Data source from LX-MIE, Kitzmann & Heng (2018) \\cite{Kitzmann2018optical}.",
        "pandas_kwargs":
            {
                "sep":"\s+", 
                "skiprows":3, 
                "names":['um','n','k']
            }
        },
    "Cr":{
        "inhouse_file":"Cr_LXMIE.dat",
        "reference":"\\cite{palik1991Vol2,Rakic1998Optical}",
        "url":"https://raw.githubusercontent.com/exoclime/LX-MIE/master/compilation/Cr.dat", 
        "usr_note":"Data source from LX-MIE, Kitzmann & Heng (2018) \\cite{Kitzmann2018optical}. Original Palik Vol 2 (1991) data can be found in Part 2, subpart 1",
        "pandas_kwargs":
            {
                "sep":"\s+", 
                "skiprows":3, 
                "names":['um','n','k']
            }
        },
    "Fe":{
        "inhouse_file":"Fe_Palik.dat",
        "reference":"\\cite{palik1991Vol2}",
        "url":"https://raw.githubusercontent.com/exoclime/LX-MIE/master/compilation/Fe.dat", 
        "usr_note":"Data stripped from Palik Vol 2 (1991). Data can be found in Part 2, subpart 1",
        "pandas_kwargs":
            {
                "sep":"\s+", 
                "skiprows":3, 
                "names":['um','n','k']
            }
        },
    "H2O":{
        "inhouse_file":"H2O_warren.dat",
        "reference":"\\cite{warren2008H2O}",
        "hitran2020":"ascii/single_files/warren_ice.dat",
        "url":"https://hitran.org/data/Aerosols/Aerosols-2020/", 
        "usr_note":"Taken from HITRAN 2020, see folder: ascii/single_files/warren_ice.dat",
        "pandas_kwargs":
            {
                "sep":"\s+", 
                "skiprows":12, 
                "names":['cm-1','um','n','k']
            }        
    },
    "KCl":{
        "inhouse_file":"KCl_Querry.dat",
        "reference":"\\cite{querry1987optical}",
        "hitran2020":"ascii/exoplanets/querry_kcl.dat",
        "url":"https://hitran.org/data/Aerosols/Aerosols-2020/", 
        "usr_note":"Taken from HITRAN 2020, see folder: ascii/exoplanets/querry_kcl.dat",
        "pandas_kwargs":
            {
                "sep":"\s+", 
                "skiprows":12, 
                "names":['cm-1','um','n','k','nerr','kerr']
            },
        "extra":{
            "url":"https://raw.githubusercontent.com/exoclime/LX-MIE/master/compilation/KCl.dat", 
            "usr_note":"0.1-0.2 micron data used to anchor edge interpolation of Querry 1987 data. Therefore the data doesn't actually show up in Virga. Data source from LX-MIE, Kitzmann & Heng (2018) \cite{Kitzmann2018optical}. Original Palik Vol 1 (1985).",
            "pandas_kwargs":
                {
                    "sep":"\s+", 
                    "skiprows":3, 
                    "names":['um','n','k']
                }            
        }
    },
    "Mg2SiO4":{
        "inhouse_file":"Mg2SiO4_Jager.dat",
        "reference":"\\cite{jager2003mg2sio4}",
        "url":"https://hitran.org/data/Aerosols/Aerosols-2020/", 
        "hitran2020":"ascii/exoplanets/jager_mg2sio4.dat",
        "usr_note":"Taken from HITRAN 2020, see folder: ascii/exoplanets/jager_mg2sio4.dat",
        "pandas_kwargs":
            {
                "sep":"\s+", 
                "skiprows":13, 
                "names":['cm-1','um','n','k']
            }        
    },    
    "MgSiO3":{
        "inhouse_file":"MgSiO3_LXMIE.dat",
        "reference":"\\cite{jager2003mg2sio4}",
        "url":"https://raw.githubusercontent.com/exoclime/LX-MIE/master/compilation/MgSiO3_amorph_sol-gel.dat", 
        "usr_note":"Data source from LX-MIE, Kitzmann & Heng (2018) \\cite{Kitzmann2018optical}.",
        "pandas_kwargs":
            {
                "sep":"\s+", 
                "skiprows":3, 
                "names":['um','n','k']
            }
    }, 
    "MnS":{
        "inhouse_file":"MnS_LXMIE.dat",
        "reference":"\\cite{Huffman1967MnS,Montaner1979Sulfur}",
        "url":"https://raw.githubusercontent.com/exoclime/LX-MIE/master/compilation/MnS.dat", 
        "usr_note":"Data source from LX-MIE, Kitzmann & Heng (2018) \\cite{Kitzmann2018optical}.",
        "pandas_kwargs":
            {
                "sep":"\s+", 
                "skiprows":3, 
                "names":['um','n','k']
            }
        },    
    "NH3":{
        "inhouse_file":"NH3_Martonchik.dat",
        "reference":"\\cite{Martonchik1984NH3}",
        "url":"https://opg.optica.org/ao/fulltext.cfm?uri=ao-23-4-541&id=27288#articleTables", 
        "usr_note":"Data stripped from table.",
        "pandas_kwargs":
            {
                "skiprows":8, 
                "sep":"\s+",
                "names":['cm-1','n','k']
            }
        },
    "Na2S":{
        "inhouse_file":"Na2S_LXMIE.dat",
        "reference":"\\cite{Khachai2009Na2S,Montaner1979Sulfur}",
        "url":"https://raw.githubusercontent.com/exoclime/LX-MIE/master/compilation/Na2S.dat", 
        "usr_note":"Data source from LX-MIE, Kitzmann & Heng (2018) \\cite{Kitzmann2018optical}. ",
        "pandas_kwargs":
            {
                "sep":"\s+", 
                "skiprows":3, 
                "names":['um','n','k']
            }
        },    
    "TiO2":{
        "inhouse_file":"TiO2_anatase_LXMIE.dat",
        "reference":"\\cite{Zeidler2011TiO2,Posch2003TiOs,Siefke2016TiO2}",
        "url":"https://raw.githubusercontent.com/exoclime/LX-MIE/master/compilation/TiO2_anatase.dat", 
        "usr_note":"Data source from LX-MIE, Kitzmann & Heng (2018) \\cite{Kitzmann2018optical}. ",
        "pandas_kwargs":
            {
                "sep":"\s+", 
                "skiprows":3, 
                "names":['um','n','k']
            }
        }, 
    "ZnS":{
        "inhouse_file":"ZnS_Querry.dat",
        "reference":"\\cite{querry1987optical}",
        "hitran2020":"ascii/exoplanets/querry_zns.dat",
        "url":"https://hitran.org/data/Aerosols/Aerosols-2020/", 
        "usr_note":"Taken from HITRAN 2020, see folder: ascii/exoplanets/querry_zns.dat",
        "pandas_kwargs":
            {
                "sep":"\s+", 
                "skiprows":12, 
                "names":['cm-1','um','n','k','nerr','kerr']
            }
        }, 
    "SiO2":{
        "inhouse_file":"SiO2_alpha.dat",
        "reference":"\\cite{Zeidler2013SiO2,Philipp1985}",
         "url":"https://raw.githubusercontent.com/exoclime/LX-MIE/master/compilation/SiO2_alpha.dat", 
        "usr_note":"Data source from LX-MIE, Kitzmann & Heng (2018) \\cite{Kitzmann2018optical}",
        "pandas_kwargs":
            {
                "sep":"\s+", 
                "skiprows":3, 
                "names":['um','n','k']
            }
            },

    "H2SO4":{
        "inhouse_file":"h2so4T293.biermann",
        "reference":"\\cite{querry1987optical}",
        "hitran2020":"ascii/biermann_h2so4/h2so4T293.biermann",
        "url":"https://hitran.org/data/Aerosols/Aerosols-2020/", 
        "usr_note":"Taken from HITRAN 2020, see folder: ascii/biermann_h2so4/h2so4T293.biermann",
        "pandas_kwargs":
            {
                "skiprows":18, 
                "header":None, 
                "names":['cm-1','um','00%',   '10%',   '20%',   '30%',   '40%',   '45%',   '50%',
                          '57%',   '60%',   '64%',   '75%'  , '80%'],
                "sep":"\s+",
                "stacked":[8191, 8194,'00%']
            }
        }
}


def create_ior_db(hitran_dir, output_dir=None,virga_dir=None, lxmie_dir=None,
                new_micron_grid = None, ior_config='default' ): 
    """
    Function uses raw data from either HITRAN, LX-MIE repository, or Virga 
    aggregated data. It interpolates data onto the original 196 grid. 

    Creates a figure automatially for each of the species. 

    Parameters
    ----------
    hitran_dir : str 
        Pointer to untarred HITRAN 2020 dirctory which you can download from here: 
        https://hitran.org/data/Aerosols/Aerosols-2020/
    output_dir : str , optional
        Output directory to save all the files as "molecule_name.refrind"
    virga_dir : str ,optional
        Directory of the current optical properties that are in virga or are from 
        Zenodo. This is only needed if you are comparing to old data.
    lxmie_dir : str, optional   
        For files that are configured to grab from LX-MIE, the will get them from 
        the online distribution. If you want to plot all the LX-MIE data in cases 
        where we are *not* using LX-MIE data, then you can clone the LX-MIE repository and 
        point to it here. https://github.com/exoclime/LX-MIE
    new_micron_grid : array, optional
        Input a diffrent array. If nothing is added it pulls from the historical 196 grid used 
        in Ackerman & Marley 2001. 
    ior_config : str or dict
        (optional) This allows users to modify their ior choices from the config file before 
        running. optiosn are "default" which uses the defaults listed in configure_ior_choices
        Alternatively, users can update their parameters and then pass the full dictionary 
        to this file. 
    """
    if isinstance(ior_config, str):
        if 'default' in ior_config: 
            config = configure_ior_choices()
        else:
            raise Exception('The only string choice for ior_config is default. Please select default or input a dict')
    elif isinstance(ior_config, dict):
            config = ior_config
    else: 
        raise Exception('Not an allowable input for ior_config. Must be either string type defualt or modified dict from configure_ior_choices')

    files_hitran = glob.glob(os.path.join(hitran_dir,'ascii','exoplanets','*.dat'))
    avail = list(config.keys())
    if not isinstance(lxmie_dir,type(None)):
        files_lxmie = glob.glob(os.path.join(lxmie_dir,"*.dat"))
    else: 
        files_lxmie = None

    fn = [jpi.figure(width=200,height=200, x_axis_type='log', 
                     y_axis_type='linear',x_range=[0.3,150],y_range=[0,7],
                     title=avail[i]) for i in range(len(avail))]
    fk = [jpi.figure(width=200,height=200, x_axis_type='log', 
                     y_axis_type='log',x_range=[0.3,150],y_range=[1e-4,1e2],
                     title=avail[i]) for i in range(len(avail))]

    colors = jpi.colpals.turbo(len(avail)+10)[9:]#jpi.colpals.Set3[12] + [jpi.colpals.Set2[7][-1] ]
    bo_h   =True
    bo_kit =True
    bo_vir =True
    for i, imol in enumerate(avail):
        # get origina data if the user wants it
        if not isinstance(virga_dir,type(None)):
            w, n, k = jdi.get_refrind(imol, virga_dir )
        else:  
            #otherwise get the 196 grid 
            if isinstance(new_micron_grid,type(None)):
                filename = os.path.join(os.path.dirname(__file__), "reference",'196.csv')
                w = pd.read_csv(filename)['Wavelength(um)'].values
            else: 
                w = new_micron_grid
            n = None 
            k = None 


        #kitzman read from files we have and define as readin which will be used to interpolate
        if 'LX-MIE' in config[imol]['url']: 
            readin = pd.read_csv(config[imol]['url'],
                **config[imol]['pandas_kwargs']
            )
            if 'cm-1' in readin.keys(): 
                readin['um'] = 1e4/readin['cm-1']
            else: 
                readin['cm-1'] = 1e4/readin['um']
            readin = readin.sort_values(by='cm-1')
            datasets = {'n':readin,'k':readin}
            fn[i].line(readin['um'], readin['n'], color='grey',line_width=6)
            fk[i].line(readin['um'], readin['k'], color='grey',line_width=6)

            #grab the hitran data, if it exists 
            hitfile =[ihit for ihit in files_hitran if imol.lower() in ihit]
            #make sure it is the molecule Fe and not something else like FeO
            if imol=='Fe': hitfile =[ihit for ihit in files_hitran if imol.lower()+'.' in ihit]
            #using anatase AB for Tio2
            if imol=='TiO2': hitfile = [hitfile[-1]]
            if imol=='SiO2': hitfile = ['/'.join(hitfile[0].split('/')[0:-1])+ '/zeidler_sio2_928k_ab.dat']
            if len(hitfile)>0: 
                skiprows = 5 
                tryread=True
                #little hack to read in hitran even though there are different numbers of skipped rows
                while tryread: 
                    try: 
                        
                        read_hit = pd.read_csv(os.path.join(hitfile[0]), sep='\s+', skiprows=skiprows)
                        x,y=1e4/read_hit['cm-1'], read_hit['real']
                        fn[i].circle(x,y, color='black',size=9)
                        x,y =1e4/read_hit['cm-1'], read_hit['imaginary']
                        fk[i].circle(x,y, color='black',size=9) 
                        tryread=False
                    except: 
                        skiprows+=1
        else: 
            if not isinstance(files_lxmie,type(None)):
                #else grab from kitzman code and define as readinlx just for plotting
                kitz = [ikit for ikit in files_lxmie if imol in ikit ]
                if imol=='Fe':kitz = [ikit for ikit in files_lxmie if imol+'.' in ikit ]
                if imol=='MgSiO3':kitz = [ikit for ikit in files_lxmie if 'MgSiO3_amorph_sol-gel' in ikit ]
                for ikit in kitz: 
                    readin_LX=pd.read_csv(ikit,sep='\s+', skiprows=3, names=['um','n','k'])
                    fn[i].line(readin_LX['um'], readin_LX['n'], color='grey',line_width=6)
                    fk[i].line(readin_LX['um'], readin_LX['k'], color='grey',line_width=6) 


                    
        if 'LX-MIE' not in config[imol]['url']: 
        #hitran or other raw non-processed data 
            if 'hitran2020' in config[imol].keys(): 
                filename = os.path.join(hitran_dir,config[imol]['hitran2020'])
            else: 
                raw_dir = os.path.join(os.path.dirname(__file__), "reference")
                filename = os.path.join(raw_dir,'raw_iors',config[imol]['inhouse_file'])
            

            #some hitran files are read in stacked (e.g. img, followed by real)
            if 'stacked' in config[imol]['pandas_kwargs'].keys(): 
                stacked = True
                real_ind = config[imol]['pandas_kwargs']['stacked'][0]
                img_ind = config[imol]['pandas_kwargs']['stacked'][1]
                col_name = config[imol]['pandas_kwargs']['stacked'][2]
                #delete as it does not belong in pandas kwargs
                del config[imol]['pandas_kwargs']['stacked']
            else: stacked=False

            readin = pd.read_csv(filename,
                **config[imol]['pandas_kwargs']
            )

            if stacked: 
                real_df= pd.DataFrame(columns=['um','cm-1','n']) 
                img_df = pd.DataFrame(columns=['um','cm-1','k'])
                real = readin.loc[0:real_ind,:].astype(float)
                real=real.loc[real['um']!=0]
                img = readin.loc[img_ind:,:].astype(float) 
                img = img.loc[img['um']!=0]              
                
                real_df['um'] = real['um'].values 
                real_df['cm-1'] = real['cm-1'].values 
                real_df['n'] = real[col_name].values 

                img_df['um'] = img['um'].values 
                img_df['cm-1'] = img['cm-1'].values 
                img_df['k'] = img[col_name].values 
            else : 
                real_df = readin.loc[:,
                        [ii for  ii in readin.keys() if ii in ['um','cm-1','n']]]
                img_df = readin.loc[:,
                        [ii for  ii in readin.keys() if ii in ['um','cm-1','k']]]

            datasets = {'n':real_df,'k':img_df}
            for idf in datasets.keys():
                if 'cm-1' in datasets[idf].keys(): 
                    datasets[idf]['um'] = 1e4/datasets[idf]['cm-1']
                else: 
                    datasets[idf]['cm-1'] = 1e4/datasets[idf]['um']

                datasets[idf] = datasets[idf].sort_values(by='cm-1')

            fn[i].circle(datasets['n']['um'], datasets['n']['n'], color='black',size=9)
            fk[i].circle(datasets['k']['um'], datasets['k']['k'], color='black',size=9)
            
        
        #plot old data if the user asks for it
        if not isinstance(virga_dir,type(None)):
            fn[i].line(w, n, color=colors[i],line_width=3, line_dash='dashed')
        if not isinstance(virga_dir,type(None)):
            fk[i].line(w, k, color=colors[i],line_width=3, line_dash='dashed')
        
        #new interpolated 
        old_cm_n = datasets['n']['cm-1'].values
        old_cm_k = datasets['k']['cm-1'].values
        old_n = datasets['n']['n'].values
        old_k =  datasets['k']['k'].values
        new_cm = 1e4/w
        
        #remove zeros 
        old_cm_k = old_cm_k[old_k != 0]
        old_k = old_k[old_k != 0] 
        old_cm_n = old_cm_n[old_n != 0]
        old_n = old_n[old_n != 0] 
        
        #specialized post-processing
        #fix zero data of KCl
        if imol == 'KCl':
            #heng data that we want to add to the beginning of KCl
            kitz_data_w = 0.15 #microns
            readin_LX = pd.read_csv(config['KCl']['extra']['url'], 
                                   **config['KCl']['extra']['pandas_kwargs'])
            kitzdf = readin_LX.loc[readin_LX['um']<kitz_data_w].sort_values(by='um',ascending=False)
            kitz_cm = 1e4/kitzdf['um'].values
            kitz_k = kitzdf['k'].values
            old_cm_k = np.concatenate((old_cm_k,kitz_cm))
            old_k = np.concatenate((old_k,kitz_k))
            
        #finally interpolate    
        int_n = np.interp(new_cm, old_cm_n,old_n)
        int_k = np.interp(new_cm, old_cm_k,old_k)
        fn[i].line(w,int_n, color=colors[i],line_width=3)
        fk[i].line(w, int_k, color=colors[i],line_width=3)    
        
        
        
        
        for ip in [fn[i], fk[i]]:
            ip.xgrid.grid_line_alpha=0
            ip.ygrid.grid_line_alpha=0
            ip.outline_line_alpha=0

        if not isinstance(output_dir ,type(None)):
            print("Saving", os.path.join(output_dir,imol+'.refrind'))
            df=pd.DataFrame({'micron':w,
             'real':int_n,
             'imaginary':int_k})
            df.index.name='index'
            df.to_csv(os.path.join(output_dir,imol+'.refrind'),index=False)        
            
    return jpi.gridplot([[ifn,ifk] for ifn, ifk in zip(fn,fk)])

