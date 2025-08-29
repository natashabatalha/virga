from bokeh.palettes import viridis,magma
from bokeh.models import ColumnDataSource, Label, LabelSet,CustomJS
from bokeh.layouts import column,row,gridplot
from bokeh.plotting import figure, show
from bokeh.models import LinearColorMapper, LogTicker,BasicTicker, ColorBar,LogColorMapper,Legend
from bokeh.palettes import magma as colfun1
from bokeh.palettes import Colorblind8
from bokeh.palettes import viridis as colfun2
from bokeh.palettes import gray as colfun3
from bokeh.palettes import Colorblind8
import bokeh.palettes as colpals
from bokeh.io import output_notebook 
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter # used to display axes labels as 0.1 instead of 10^-1 power notation
from matplotlib.ticker import FormatStrFormatter
from statistics import mean
from scipy.signal import savgol_filter
from matplotlib.legend_handler import HandlerPathCollection
import matplotlib
import pandas as pd
import sys
from .root_functions import calculate_k0, find_N_mon_and_r_mon

from . import justdoit as pyeddy

def pt(out, with_condensation=True,return_condensation=False, **kwargs):
    """
    Plot PT profiles with condensation curves. 
    Thick lines in this pt plot signify those gases that 
    were turned on in the run, NOT those that were recommended. 
    
    Parameters
    ----------
    out : dict 
        Dictionary output from pyeddy run 
    with_condensation : bool 
        Plots condensation curves of gases. Also plots those that were turned on in the 
        calculation with line_width=5. All others are set to 1. 
    return_condensation : bool 
        If true, it returns list of condenation temps for each gas, pressure grid, 
        and a list of the gas names
    **kwargs : kwargs 
        Kwargs for bokeh.figure() 

    Returns
    -------
    if return_condensation: fig, cond temperature curves, ,cond pressure grid , name of all available gases
    else: fig
    """
    kwargs['height'] = kwargs.get('plot_height',kwargs.get('height',300))
    kwargs['width'] = kwargs.get('plot_width', kwargs.get('width',600))
    if 'plot_width' in kwargs.keys() : kwargs.pop('plot_width')
    if 'plot_height' in kwargs.keys() : kwargs.pop('plot_height')
    kwargs['x_axis_label'] = kwargs.get('x_axis_label','Temperature (K)')
    kwargs['y_axis_label'] = kwargs.get('y_axis_label','Pressure (bars)')
    kwargs['y_axis_type'] = kwargs.get('y_axis_type','log')
    

    ngas = len(out['condensibles'])
    temperature = out['temperature']
    pressure = out['pressure']
    condensibles = out['condensibles']
    mh, mmw = out['scalar_inputs']['mh'], out['scalar_inputs']['mmw']

    kwargs['y_range'] = kwargs.get('y_range',[np.max(pressure), np.min(pressure)])

    fig = figure(**kwargs)

    if with_condensation:
        all_gases = pyeddy.available()
        cond_ts = []
        recommend = []
        line_width = []

        for gas_name in all_gases: #case sensitive names
            #grab p,t from eddysed
            cond_p,t = pyeddy.condensation_t(gas_name, mh, mmw)
            cond_ts +=[t]
            if gas_name in condensibles:
                line_width += [5]
            else: 
                line_width += [1]

        cols = magma(len(all_gases))
        for i in range(len(all_gases)):
            fig.line(cond_ts[i],cond_p, legend_label=all_gases[i],color=cols[i],line_width=line_width[i] )

    fig.line(temperature,pressure, legend_label='User',color='black',line_width=5,line_dash='dashed')

    plot_format(fig)
    if return_condensation: 
        return fig, cond_ts,cond_p,all_gases
    else : 
        return fig


def radii(out,gas=None,at_pressure = 1e-3, compare=False, legend=None, 
        p1w=300, p1h=300, p2w=300, p2h=300, color_indx=0):
    """
    Plots the particle radii profile along with the distribution, at a certain pressure. 

    Parameters
    ----------
    out : dict 
        Dictionary output from pyeddy run 
    pressure : float,optional 
        Pressure level to plot full distribution (bars)
    
    """
    if type(out)==dict:
        out=[out]

    lines = ['solid','dashed','dashdot']
    #lines = ['solid']*3
    if compare:
        lines = ['solid']*len(out)
        legend_it = []
    for j in range(len(out)):
        #compute initial distributions   
        r_g = out[j]['mean_particle_r']
        pressure = out[j]['pressure']

        nl = find_nearest_1d(pressure,at_pressure)

        rmin = out[j]['scalar_inputs']['rmin']
        rmax = out[j]['scalar_inputs']['rmax']
        nrad = out[j]['scalar_inputs']['nrad']
        log_radii = out[j]['scalar_inputs']['log_radii']
        if(log_radii>0):
            logspace=True # convert scalar variable into boolean for get_r_grid
        else:
            logspace=False # convert scalar variable into boolean for get_r_grid
        sig = out[j]['scalar_inputs']['sig']
        ndz = out[j]['column_density']
        #determine which condensed
        which_condensed = [False]*ndz.shape[1]

        # only consider specified gas
        if gas is not None:
            indx = out[j]['condensibles'].index(gas)
            ndz_gas = np.unique(ndz[:,indx])
            ndz_gas = ndz_gas[ndz_gas>0]
            if len(ndz_gas)>0 : 
                which_condensed[indx] = True
            else:
                print(condensate + " did not condense. Choose another condensate.")
                import sys; sys.exit()

        # consider all gases
        else:
            for i in range(ndz.shape[1]):
                ndz_gas = np.unique(ndz[:,i])
                ndz_gas = ndz_gas[ndz_gas>0]
                if len(ndz_gas)>0 : which_condensed[i] = True


        #take only those that condensed 
        N = ndz[:, which_condensed]
        r_g = r_g[:,which_condensed]
        gas_name = list(np.array(out[j]['condensibles'])[which_condensed])
        r, bin_min, bin_max, dr = pyeddy.get_r_grid(r_min = rmin, r_max=rmax, n_radii = nrad, log_space=logspace)

        # different colours for different dicts
        if (gas is not None) or compare:
            length = len(out)
        # different colours for different gases
        else:
            length = len(gas_name) 

        color = magma(length)
        color = Colorblind8[color_indx:color_indx+length]

        #initial radii profiles
        df_r_g = {i:r_g[:, gas_name.index(i)] for i in gas_name}
        df_r_g['pressure'] = pressure


        dndr = {}
        for i in gas_name:
            dndr[i]= N[nl,gas_name.index(i)]/r/np.sqrt(2*np.pi)*np.log(sig)*np.exp(
                        - np.log(r/(r_g[nl,gas_name.index(i)]*1e-4))**2/(2*np.log(sig)**2)) #1e-4 is microns to cm
        dndr['r'] = r*1e4 #convert to microns

        if j==0:
            p1 = figure(width=p1w, height=p1h, title="Select Pressure Level", 
                x_axis_type='log',y_axis_type='log',y_axis_label='Pressure (bars)',x_axis_label='Mean Particle Radius (um)',
                y_range=[np.max(pressure),np.min(pressure)])

            p2 = figure(width=p2w, height=p2h, title="Particle Distribution at %.0e bars" %at_pressure,
                         x_axis_type='log', y_axis_type='log', y_axis_label='dn/dr (cm-3)',
                         x_axis_label='Particle Radius (um)', 
                         y_range=[np.max([np.min(dndr[i]), 1e-50]), np.max(dndr[i])])

        #add to r_g for that plot 
        df_r_g['average'] = [pressure[nl]]*len(pressure)
        df_r_g['horizontal'] = np.linspace(np.amin(r_g[r_g>0]), np.amax(r_g) , len(pressure))
            

        s1 = ColumnDataSource(data=dict(df_r_g))
        s2 = ColumnDataSource(data=dict(dndr))
        for i in gas_name:
            if gas is not None or compare: indx = j
            else: indx = gas_name.index(i)
            f = p1.line(i, 'pressure', source=s1, alpha=1,color=color[np.mod(indx, len(color))] ,line_width=4,legend_label=i,line_dash=lines[j])
            p2.line('r', i, source=s2,color=color[np.mod(indx, len(color))] ,line_width=4,line_dash=lines[j])

            if compare:
                legend_it.append((legend[j], [f]))

    # r_g = out['droplet_eff_r']
    # pressure = out['pressure']

    # nl = find_nearest_1d(pressure,at_pressure)

    # rmin = out['scalar_inputs']['rmin']
    # nrad = out['scalar_inputs']['nrad']
    # sig = out['scalar_inputs']['sig']
    # ndz = out['column_density']
    # #determine which condensed
    # which_condensed = [False]*ndz.shape[1]
    # for i in range(ndz.shape[1]):
    #     ndz_gas = np.unique(ndz[:,i])
    #     ndz_gas = ndz_gas[ndz_gas>0]
    #     if len(ndz_gas)>0 : which_condensed[i] = True

    # color = magma(len(which_condensed))

    # #take only those that condensed 
    # N = ndz[:, which_condensed]
    # r_g = r_g[:,which_condensed]
    # gas_name = list(np.array(out['condensibles'])[which_condensed])
    # r, rup, dr = pyeddy.get_r_grid(r_min = rmin, n_radii = nrad)

    # #initial radii profiles
    # df_r_g = {i:r_g[:, gas_name.index(i)] for i in gas_name}
    # df_r_g['pressure'] = pressure


    # #add to r_g for that plot 
    # df_r_g['average'] = [pressure[nl]]*len(pressure)
    # df_r_g['horizontal'] = np.linspace(np.amin(r_g[r_g>0]), np.amax(r_g) , len(pressure))

    p1.line('horizontal', 'average', source=s1, color='black',line_width=3,line_dash='dashed')
    p1.legend.location = 'bottom_left'
    #plot_format(p1)
    #plot_format(p2)
    if compare:
        legend = Legend(items=legend_it, location=(0, 0))
        legend.click_policy="mute"
        p1.add_layout(legend, 'right')   
    return row(p1, p2), dndr

def opd_by_gas(out, gas = None, color = magma, compare=False, legend=None, **kwargs):
    """
    Optical depth for conservative geometric scatteres separated by gas.
    E.g. [Fig 7 in Morley+2012](https://arxiv.org/pdf/1206.4313.pdf)  
    
    Parameters
    ----------
    out : dict 
        Dictionary output from pyeddy run
    color : method
        Method from bokeh.palletes. e.g. bokeh.palletes.virids, bokeh.palletes.magma
    **kwargs : kwargs 
        Kwargs for bokeh.figure() 
    """

    kwargs['height'] = kwargs.get('plot_height',kwargs.get('height',300))
    kwargs['width'] = kwargs.get('plot_width', kwargs.get('width',400))
    if 'plot_width' in kwargs.keys() : kwargs.pop('plot_width')
    if 'plot_height' in kwargs.keys() : kwargs.pop('plot_height')
    kwargs['x_axis_label'] = kwargs.get('x_axis_label','Column Optical Depth')
    kwargs['y_axis_label'] = kwargs.get('y_axis_label','Pressure (bars)')
    kwargs['y_axis_type'] = kwargs.get('y_axis_type','log')
    kwargs['x_axis_type'] = kwargs.get('x_axis_type','log')    

    if type(out)==dict:
        out=[out]

    condensibles = out[0]['condensibles']
    if gas is not None: 
        indx = condensibles.index(gas)
        length = len(out)
    else:
        length = len(condensibles)
    ngas = len(condensibles)
    col = color(length)
    col = Colorblind8[:length]
    lines = ['solid','dashed','dotdash','dashdot']

    legend_it = []
    for j in range(len(out)):
        pressure = out[j]['pressure']
        opd_by_gas = out[j]['opd_by_gas']

        if j == 0:
            kwargs['y_range'] = kwargs.get('y_range',[np.max(pressure), np.min(pressure)])
            kwargs['x_range'] = kwargs.get('x_range',[np.max([1e-6, np.min(opd_by_gas*0.9)]), 
                                                np.max(opd_by_gas*1.1)])
            fig = figure(**kwargs)

        if compare:
            if gas is not None:
                f = fig.line(opd_by_gas[:,indx], pressure, line_width=4, legend_label = condensibles[indx], color=Colorblind8[np.mod(j, len(Colorblind8))])
            else:
                for i in range(ngas):
                    f = fig.line(opd_by_gas[:,i], pressure,line_width=4, legend_label = condensibles[i], color=Colorblind8[np.mod(j, len(Colorblind8))], line_dash=lines[i])
            legend_it.append((legend[j], [f]))
        else:
            if gas is not None:
                fig.line(opd_by_gas[:,indx], pressure, line_width=4, legend_label = condensibles[indx], color=col[j], line_dash=lines[j])
            else:
                for i in range(ngas):
                    fig.line(opd_by_gas[:,i], pressure,line_width=4, legend_label = condensibles[i], color=col[i], line_dash=lines[j])

    if compare:
        legend = Legend(items=legend_it, location=(0, 0))
        legend.click_policy="mute"
        fig.add_layout(legend, 'right')   
    #plot_format(fig)
    return fig

def condensate_mmr(out, gas=None, compare=False, legend=None, **kwargs):
    """
    Condensate mean mass mixing ratio
    
    Parameters
    ----------
    out : dict 
        Dictionary output from pyeddy run
    color : method
        Method from bokeh.palletes. e.g. bokeh.palletes.virids, bokeh.palletes.magma
    **kwargs : kwargs 
        Kwargs for bokeh.figure() 
    """
    kwargs['height'] = kwargs.get('plot_height',kwargs.get('height',300))
    kwargs['width'] = kwargs.get('plot_width', kwargs.get('width',400))
    if 'plot_width' in kwargs.keys() : kwargs.pop('plot_width')
    if 'plot_height' in kwargs.keys() : kwargs.pop('plot_height')
    kwargs['x_axis_label'] = kwargs.get('x_axis_label','Condensate MMR')
    kwargs['y_axis_label'] = kwargs.get('y_axis_label','Pressure (bars)')
    kwargs['y_axis_type'] = kwargs.get('y_axis_type','log')
    kwargs['x_axis_type'] = kwargs.get('x_axis_type','log')    

    if type(out)==dict:
        out=[out]

    condensibles = out[0]['condensibles']
    if gas is not None :
        indx = condensibles.index(gas)
        length = len(out)
    else:
        length = len(condensibles)
    ngas = len(condensibles)
    col = magma(length)
    lines = ['solid','dashed','dotdash','dashdot']

    legend_it = []
    for j in range(len(out)):
        pressure = out[j]['pressure']
        cond_mmr = out[j]['condensate_mmr']

        if j == 0:
            kwargs['y_range'] = kwargs.get('y_range',[np.max(pressure), np.min(pressure)])
            kwargs['x_range'] = kwargs.get('x_range',[np.max([1e-9, np.min(cond_mmr*0.9)]), 
                                                    np.max(cond_mmr*1.1)])
            fig = figure(**kwargs)

        if compare:
            if gas is not None:
                f = fig.line(cond_mmr[:,i], pressure, line_width=4, legend_label = condensibles[indx], color=Colorblind8[np.mod(j, len(Colorblind8))])
            else:
                for i in range(ngas):
                    label = condensibles[i] + ' ' + legend[j]
                    f = fig.line(cond_mmr[:,i], pressure, line_width=4, legend_label = condensibles[i], color=Colorblind8[np.mod(j, len(Colorblind8))], line_dash=lines[i])
            legend_it.append((legend[j], [f]))

        else:
            if gas is not None:
                fig.line(cond_mmr[:,indx], pressure, line_width=4, legend_label = condensibles[indx], color=Colorblind8[np.mod(j, len(Colorblind8))], line_dash=lines[j])
            else:
                for i in range(ngas):
                    fig.line(cond_mmr[:,i], pressure, line_width=4, legend_label = condensibles[i], color=col[i], line_dash=lines[j])

    if compare:
        legend = Legend(items=legend_it, location=(0, 0))
        legend.click_policy="mute"
        fig.add_layout(legend, 'right')   

    plot_format(fig)
    return fig

def all_optics(out):
    """
    Maps of the wavelength dependent single scattering albedo 
    and cloud opacity and asymmetry parameter as a function of altitude. 

    Parameters
    ----------
    out : dict 
        Dictionary output from pyeddy run 

    Returns
    -------
    Three bokeh plots with the single scattering, optical depth, and assymetry maps
    """
    #get into DataFrame format
    dat01 = pyeddy.picaso_format(out['opd_per_layer'],out['single_scattering'],
                      out['asymmetry'])

    nwno=len(out['wave'])
    nlayer=len(out['pressure'])
    pressure=out['pressure']

    pressure_label = 'Pressure (Bars)'

    wavelength_label = 'Wavelength (um)'
    wavelength = out['wave']

    cols = colfun1(200)
    color_mapper = LinearColorMapper(palette=cols, low=0, high=1)

    #PLOT W0
    scat01 = np.flip(np.reshape(dat01['w0'].values,(nlayer,nwno)),0)
    xr, yr = scat01.shape
    f01a = figure(x_range=[0, yr], y_range=[0,xr],
                           x_axis_label=wavelength_label, y_axis_label=pressure_label,
                           title="Single Scattering Albedo",
                          width=300, height=300)


    f01a.image(image=[scat01],  color_mapper=color_mapper, x=0,y=0,dh=xr,dw =yr )

    color_bar = ColorBar(color_mapper=color_mapper, #ticker=LogTicker(),
                       label_standoff=12, border_line_color=None, location=(0,0))

    f01a.add_layout(color_bar, 'left')


    #PLOT OPD
    scat01 = np.flip(np.reshape(dat01['opd'].values,(nlayer,nwno)),0)

    xr, yr = scat01.shape
    cols = colfun2(200)[::-1]
    color_mapper = LogColorMapper(palette=cols, low=1e-3, high=10)


    f01 = figure(x_range=[0, yr], y_range=[0,xr],
                           x_axis_label=wavelength_label, y_axis_label=pressure_label,
                           title="Cloud Optical Depth Per Layer",
                          width=320, height=300)

    f01.image(image=[scat01],  color_mapper=color_mapper, x=0,y=0,dh=xr,dw =yr )

    color_bar = ColorBar(color_mapper=color_mapper, ticker=LogTicker(),
                       label_standoff=12, border_line_color=None, location=(0,0))
    f01.add_layout(color_bar, 'left')

    #PLOT G0
    scat01 = np.flip(np.reshape(dat01['g0'].values,(nlayer,nwno)),0)

    xr, yr = scat01.shape
    cols = colfun3(200)[::-1]
    color_mapper = LinearColorMapper(palette=cols, low=0, high=1)


    f01b = figure(x_range=[0, yr], y_range=[0,xr],
                           x_axis_label=wavelength_label, y_axis_label=pressure_label,
                           title="Asymmetry Parameter",
                          width=300, height=300)

    f01b.image(image=[scat01],  color_mapper=color_mapper, x=0,y=0,dh=xr,dw =yr )

    color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(),
                       label_standoff=12, border_line_color=None, location=(0,0))
    f01b.add_layout(color_bar, 'left')

    #CHANGE X AND Y AXIS TO BE PHYSICAL UNITS 
    #indexes for pressure plot 
    if (pressure is not None):
        pressure = ["{:.1E}".format(i) for i in pressure[::-1]] #flip since we are also flipping matrices
        npres = len(pressure)
        ipres = np.array(range(npres))
        #set how many we actually want to put on the figure 
        #hard code ten on each.. 
        ipres = ipres[::int(npres/10)]
        pressure = pressure[::int(npres/10)]
        #create dictionary for tick marks 
        ptick = {int(i):j for i,j in zip(ipres,pressure)}
        for i in [f01a, f01, f01b]:
            i.yaxis.ticker = ipres
            i.yaxis.major_label_overrides = ptick
    if (wavelength is not None):
        wave = ["{:.2F}".format(i) for i in wavelength]
        nwave = len(wave)
        iwave = np.array(range(nwave))
        iwave = iwave[::int(nwave/10)]
        wave = wave[::int(nwave/10)]
        wtick = {int(i):j for i,j in zip(iwave,wave)}
        for i in [f01a, f01, f01b]:
            i.xaxis.ticker = iwave
            i.xaxis.major_label_overrides = wtick       

    return gridplot([[f01a, f01,f01b]])

def all_optics_1d(out, wave_range, return_output = False,legend=None,
    colors = colpals.Colorblind8, **kwargs):
    """
    Plots 1d profiles of optical depth per layer, single scattering, and 
    asymmetry averaged over the user input wave_range. 

    Parameters
    ----------
    out : list or dict 
        Either a list of output dictionaries or a single dictionary output
        from .compute(as_dict=True)
    wave_range : list 
        min and max wavelength in microns 
    return_output : bool 
        Default is just to return a figure but you can also 
        return all the 1d profiles 
    legend : bool 
        Default is none. Legend for each component of out 
    **kwargs : keyword arguments
        Key word arguments will be supplied to each bokeh figure function
    """
    kwargs['height'] = kwargs.get('plot_height',kwargs.get('height',300))
    kwargs['width'] = kwargs.get('plot_width', kwargs.get('width',300))
    if 'plot_width' in kwargs.keys() : kwargs.pop('plot_width')
    if 'plot_height' in kwargs.keys() : kwargs.pop('plot_height')
    kwargs['y_axis_type'] = kwargs.get('y_axis_type','log')

    if not isinstance(out, list):
        out = [out]

    pressure = out[0]['pressure']

    kwargs['y_range'] = kwargs.get('y_range',[max(pressure),min(pressure)])     

    ssa = figure(x_axis_label='Single Scattering Albedo',**kwargs)

    g0 = figure(x_axis_label='Asymmetry',**kwargs)

    opd = figure(x_axis_label='Optical Depth',y_axis_label='Pressure (bars)',
        x_axis_type='log',**kwargs)

    
    for i,results in enumerate(out): 
        inds = np.where((results['wave']>wave_range[0]) & 
            (results['wave']<wave_range[1]))
        opd_dat = np.mean(results['opd_per_layer'][:,inds],axis=2)[:,0]
        opd.line(opd_dat, 
                 results['pressure'], color=colors[np.mod(i, len(colors))],line_width=3)
        g0_dat = np.mean(results['asymmetry'][:,inds],axis=2)[:,0]
        g0.line(g0_dat, 
                 results['pressure'], color=colors[np.mod(i, len(colors))],line_width=3)
        
        if isinstance(legend, type(None)):
            ssa_dat = np.mean(results['single_scattering'][:,inds],axis=2)[:,0]
            ssa.line(ssa_dat, 
                 results['pressure'], color=colors[np.mod(i, len(colors))],line_width=3)
        else:
            ssa_dat = np.mean(results['single_scattering'][:,inds],axis=2)[:,0]
            ssa.line(ssa_dat, 
                 results['pressure'], color=colors[np.mod(i, len(colors))],line_width=3,
                 legend_label=legend[i])
            ssa.legend.location='top_left'

    if return_output:   
        return gridplot([[opd,ssa,g0]]), [opd_dat,ssa_dat,g0_dat]
    else:   
        return gridplot([[opd,ssa,g0]])

def find_nearest_1d(array,value):
    #small program to find the nearest neighbor in a matrix
    ar , iar ,ic = np.unique(array,return_index=True,return_counts=True)
    idx = (np.abs(ar-value)).argmin(axis=0)
    if ic[idx]>1: 
        idx = iar[idx] + (ic[idx]-1)
    else: 
        idx = iar[idx]
    return idx

def pressure_fig(**kwargs):
    kwargs['y_range'] = kwargs.get('y_range',[1e2,1e-6])
    kwargs['height'] = kwargs.get('plot_height',kwargs.get('height',400))
    kwargs['width'] = kwargs.get('plot_width', kwargs.get('width',600))
    if 'plot_width' in kwargs.keys() : kwargs.pop('plot_width')
    if 'plot_height' in kwargs.keys() : kwargs.pop('plot_height')
    kwargs['x_axis_label'] = kwargs.get('x_axis_label','Temperature (K)')
    kwargs['y_axis_label'] = kwargs.get('y_axis_label','Pressure (bars)')
    kwargs['y_axis_type'] = kwargs.get('y_axis_type','log')        
    fig = figure(**kwargs)
    return fig

def plot_format(df):
    """Function to reformat plots"""
    df.xaxis.axis_label_text_font='times'
    df.yaxis.axis_label_text_font='times'
    df.xaxis.major_label_text_font_size='14pt'
    df.yaxis.major_label_text_font_size='14pt'
    df.xaxis.axis_label_text_font_size='14pt'
    df.yaxis.axis_label_text_font_size='14pt'
    df.xaxis.major_label_text_font='times'
    df.yaxis.major_label_text_font='times'
    df.xaxis.axis_label_text_font_style = 'bold'
    df.yaxis.axis_label_text_font_style = 'bold'

def plot_fsed(pressure, z, scale_height, alpha, beta, epsilon=1e-2, pres_alpha=None, **kwargs):

    kwargs['height'] = kwargs.get('plot_height',kwargs.get('height',300))
    kwargs['width'] = kwargs.get('plot_width', kwargs.get('width',700))
    if 'plot_width' in kwargs.keys() : kwargs.pop('plot_width')
    if 'plot_height' in kwargs.keys() : kwargs.pop('plot_height')
    kwargs['x_axis_label'] = kwargs.get('x_axis_label','fsed')
    kwargs['y_axis_label'] = kwargs.get('y_axis_label','Pressure (bars)')
    kwargs['x_axis_type'] = kwargs.get('x_axis_type','log')
    kwargs['y_axis_type'] = kwargs.get('y_axis_type','log')

    if type(alpha) is int:
        alpha = [alpha]
    if type(beta) is int:
        beta = [beta]

    if pres_alpha is None:
        zstar = max(z)
    else:
        indx = find_nearest_1d(pressure, pres_alpha)
        zstar = z[indx]
    indx = find_nearest_1d(pressure, 1)
    H = scale_height[indx]

    cols = Colorblind8[:len(alpha)*len(beta)]
    lines = ['solid','dashed','dotted','dotdash','dashdot']
    kwargs['y_range'] = kwargs.get('y_range',[np.max(pressure), np.min(pressure)])
    fig = figure(**kwargs)
    
    for i in range(len(alpha)):
        for j in range(len(beta)):
            lab = "alpha=%g" %alpha[i] + ", beta=%g" %beta[j]
            col = Colorblind8[np.mod(i+len(alpha)*j, 8)]
            line = lines[np.mod(j, 5)]
            fsed = alpha[i] * np.exp((z-zstar)/(6*beta[j]*H)) + epsilon
            fig.line(fsed, pressure, legend_label=lab, color=col, line_width=5, line_dash='solid')#line)

    fig.legend.location = "bottom_right"

    return fig

def fsed_from_output(out,labels,y_axis='pressure',color_indx=0,cld_bounds=False,gas_indx=None,**kwargs):

    if type(out)==dict:
        out=[out]

    kwargs['height'] = kwargs.get('plot_height',kwargs.get('height',300))
    kwargs['width'] = kwargs.get('plot_width', kwargs.get('width',700))
    if 'plot_width' in kwargs.keys() : kwargs.pop('plot_width')
    if 'plot_height' in kwargs.keys() : kwargs.pop('plot_height')
    kwargs['x_axis_label'] = kwargs.get('x_axis_label','fsed')
    kwargs['y_axis_label'] = kwargs.get('y_axis_label','Pressure (bars)')
    kwargs['x_axis_type'] = kwargs.get('x_axis_type','log')
    kwargs['y_axis_type'] = kwargs.get('y_axis_type','log')
    #kwargs['x_range'] = kwargs.get('x_range', [1e-2, 2e1])
    if gas_indx is not None:
        title = 'Condensible = ' + str(out[0]['condensibles'][gas_indx])
        kwargs['title'] = kwargs.get('title', title)


    cols = Colorblind8[color_indx:color_indx+len(out)]
    pressure = out[0]['pressure']
    kwargs['y_range'] = kwargs.get('y_range',[np.max(pressure), np.min(pressure)])
    fig = figure(**kwargs)

    min_id=[]; max_id=[]
    for i in range(len(out)):
        if gas_indx is None: x = out[i]['fsed']
        else: x = out[i]['fsed'][:,gas_indx]

        if y_axis == 'pressure':
            y = out[i]['pressure']
        elif y_axis == 'z':
            y = out[i]['altitude']
        col = Colorblind8[np.mod(i+color_indx, 8)]
        fig.line(x, y, legend_label=labels[i], color=col, line_width=5)
        if cld_bounds and gas_indx is not None:
            low_clds = []; high_clds = []
            ndz = out[i]['column_density'][:,gas_indx]
            nonzero = np.where(ndz>1e-3)[0]
            min_id.append(nonzero[0])
            max_id.append(nonzero[-1])
        
    if cld_bounds and gas_indx is not None:
        xmin = kwargs['x_range'][0]
        xmax = kwargs['x_range'][1]
        x1 = np.linspace(xmin,xmax,10)
        y1 = x1*0+y[min(min_id)]
        y2 = x1*0+y[max(max_id)]
        fig.line(x1, y1, color='black',line_width=5, line_dash='dashed')
        fig.line(x1, y2, color='black',line_width=5, line_dash='dashed')


    if labels is not None:
        fig.legend.location = "bottom_left"
    plot_format(fig)
    return fig

def aggregates_optical_properties(aggregate, mieff_dir, d_f_list, min_wavelength, max_wavelength):

    '''
    Code to plot the optical properties calculated in the .mieff files. for all radii and fractal dimensions analysed.

    Plots a 3D graph of Q_ext against wavelength (um) for all fractal dimensions (third axis).

    Set filtered = True if you want a smoother curve (using a savgol filter) for easier comparison of different fractals.

    Parameters
    ----------
    aggregate: string
        chemical species used in model 
    mieff_dir: string
        filepath to directory where .mieff files are stored
    d_f_list: array
        list of fractal dimensions analysed
    min_wavelength: float
        minimum wavelength to plot
    max_wavelength: float
        maximum wavelength to plot

    Returns
    -------
    3D graph of Q_ext against wavelength (um) for all fractal dimensions (third axis).

    '''

    # choose a colormap (based on the number of shapes to analyse)
    colors = plt.cm.winter_r(np.linspace(0,1,len(d_f_list)))


    # -------- SPHERES -------- 

    # read in the dataset for spherical particles
    mieff_col_names= ['A', 'B', 'C', 'D'] # assign arbitrary column names for reading in the data (all the columns have different numbers of rows)
    mieff_data = pd.read_csv(f"{mieff_dir}/{aggregate}.mieff", names=mieff_col_names, header=None, sep=r'\s+')

    num_wavelengths = int(mieff_data['A'][0]) # extract number of wavelengths from header
    num_radii = int(mieff_data['B'][0]) # extract number of radii from header

    # initialise lists
    wavelength_list_spheres=[]
    Q_ext_list_spheres=[]

    # Go through data for one radius size at a time, extracting the Q_ext and wavelength values and storing them in their respective lists
    for r in range(num_radii):

        wavelengths=[] # reset lists of wavelength and Q_ext each time we move onto a new radius
        Q_ext=[]
        
        for i in range(num_wavelengths):
            if (mieff_data['A'][r*num_wavelengths+r+i+2]<=max_wavelength) and (mieff_data['A'][r*num_wavelengths+r+i+2]>=min_wavelength): # only add this wavelength and Q_ext pair to our lists if it's smaller than the user-defined maximum
                wavelengths.append(mieff_data['A'][r*num_wavelengths+r+i+2]) # find the individual wavelengths at this radius
                Q_ext.append(mieff_data['C'][r*num_wavelengths+r+i+2]) # find the individual Q_ext values at each wavelength and at this radius (here the index 'r*num_wavelengths+r+i+2' basically filters the .mieff file to return it to a simple 2D array of "Wavelength, Qsca, Qext, g_asymmetry" for one particular particle size)

        wavelength_list_spheres.append(wavelengths)
        Q_ext_list_spheres.append(Q_ext)


    # record the radii used in the spherical .mieff file (not needed for the graph, but useful to print alongside it)
    radii=[]
    for r in range(num_radii):
        radii.append(10000.0*mieff_data['A'][r*num_wavelengths+r+1]) # convert all radii from cm (default for the .mieff file) to um as we record them


    # -------- AGGREGATES -------- 

    wavelength_list_of_lists=[]
    Q_ext_list_of_lists=[]

    for j in range(len(d_f_list)):

        # read in each dataset for aggregate particles one fractal dimension at a time
        mieff_data = pd.read_csv(f"{mieff_dir}/{aggregate}_aggregates_Df_{d_f_list[j]:.6f}.mieff", names=mieff_col_names, header=None, sep=r'\s+')

        # reset lists each fractal dimension
        wavelength_list=[]
        Q_ext_list=[]

        # Go through data for one radius size at a time, extracting the Q_ext and wavelength values and storing them in their respective lists
        for r in range(num_radii):

            wavelengths=[] # reset lists of wavelength and Q_ext each time we move onto a new radius
            Q_ext=[]
            
            for i in range(num_wavelengths):
                if (mieff_data['A'][r*num_wavelengths+r+i+2]<=max_wavelength) and (mieff_data['A'][r*num_wavelengths+r+i+2]>=min_wavelength): # only add this wavelength and Q_ext pair to our lists if it's smaller than the user-defined maximum
                    wavelengths.append(mieff_data['A'][r*num_wavelengths+r+i+2]) # find the individual wavelengths at this radius
                    Q_ext.append(mieff_data['C'][r*num_wavelengths+r+i+2]) # find the individual Q_ext values at each wavelength and at this radius (here the index 'r*num_wavelengths+r+i+2' basically filters the .mieff file to return it to a simple 2D array of "Wavelength, Qsca, Qext, g_asymmetry" for one particular particle size)

            wavelength_list.append(wavelengths)
            Q_ext_list.append(Q_ext)

        # save the lists of wavelengths and Q_ext for each fractal dimension
        wavelength_list_of_lists.append(wavelength_list)
        Q_ext_list_of_lists.append(Q_ext_list)


    # Create 3D scatterplot figure
    fig = plt.figure() 
    fig.set_facecolor('w') # make area around plot white in case user is in dark mode
    ax = plt.axes(projection ="3d")

    radii_colours = plt.cm.winter(np.linspace(0,1,len(wavelength_list_spheres))) # plot each radius in a different colour, from blue (small) to green (large)

    # SPHERES: plot each Q_ext vs wavelength graph, one radius (each list element) at a time 
    for i in range(len(wavelength_list_spheres)):
        ax.plot(wavelength_list_spheres[i], np.full(len(wavelength_list_spheres[i]),3), Q_ext_list_spheres[i], color = radii_colours[i], linewidth=0.5) # plot Q_ext for spheres at d_f = 3 (the np.full is just to make the array the same length, but the value for d_f is constant every time we run this line)

    # AGGREGATES: plot each Q_ext vs wavelength graph, one radius (each list element) and one fractal dimension at a time 
    for j in range(len(d_f_list)):
        for i in range(len(wavelength_list_spheres)):
            ax.plot(wavelength_list_of_lists[j][i], np.full(len(wavelength_list_of_lists[j][i]),d_f_list[j]), Q_ext_list_of_lists[j][i], color = radii_colours[i], linewidth=0.5) # plot Q_ext for each aggregate at a constant d_f value (the np.full is just to make the array the same length, but the value for d_f is constant every time we run this line)


    # set axes titles for 3D scatterplot
    ax.set_xlabel('$\lambda$ ($\mu$m)', fontsize=20)
    ax.set_ylabel('$d_f$', fontsize=20)
    ax.set_zlabel('$Q_{ext}$', fontsize=20)

    plt.close() # prevents duplicate plots from appearing

    return fig


def aggregates_wavelength_radius_grid(clouds_from_virga_spheres, clouds_from_virga_fractals, aggregate, d_f_list, mieff_dir, colors=None):

    '''

    Code to plot the wavelength-radius grid that was used the create the .mieff files, and then overplot the actual radii
    for each pressure layer for each fractal dimension, so that we can check that the radii we are using actually have
    unique optical properties that were calculated at a reasonable resolution.

    Parameters
    ----------
    clouds_from_virga_spheres: dict
        output from virga (spherical model)
    clouds_from_virga_fractals: dict
        array of dicts with output from virga (all of the virga models for each fractal dimension)
    aggregate: string
        chemical species used in model 
    d_f_list: array
        list of fractal dimensions analysed
    mieff_dir: string
        filepath to directory where .mieff files are stored
    colors: array (optional)
        list of colors to use (in order: spheres, 1.2, 1.6, 2.0, 2.4, 2.8)

    Returns
    -------
    Wavelength-radius grid from .mieff files, overplotted with radii for each pressure layer for each fractal dimension.

    '''

    # if colors = None, choose a colormap (based on the number of shapes to analyse).
    if(colors is None):
        color_array = plt.cm.viridis(np.linspace(0,1,len(d_f_list)+1))
    else:
        color_array = colors # or use colors provided by user


    # -------- SPHERES -------- 

    # read in the dataset for spherical particles
    mieff_col_names= ['A', 'B', 'C', 'D'] # assign arbitrary column names for reading in the data (all the columns have different numbers of rows)
    mieff_data = pd.read_csv(f"{mieff_dir}/{aggregate}.mieff", names=mieff_col_names, header=None, sep=r'\s+')

    num_wavelengths = int(mieff_data['A'][0]) # extract number of wavelengths from header
    num_radii = int(mieff_data['B'][0]) # extract number of radii from header

    # initialise lists
    list_of_sphere_wavelengths=[]
    list_of_sphere_radii=[]

    # record the actual wavelengths used in the spherical .mieff file
    for i in range(num_wavelengths):
        list_of_sphere_wavelengths.append(mieff_data['A'][i+2])

    # record the actual radii used in the spherical .mieff file
    for i in range(num_radii):
        list_of_sphere_radii.append(10000.0*mieff_data['A'][i*num_wavelengths+i+1]) # convert all radii from cm (default for the .mieff file) to um as we record them

    print(f'For the {aggregate}.mieff file (spheres):')
    print(f'\n\t Wavelengths are between {list_of_sphere_wavelengths[num_wavelengths-1]:.3f} --> {list_of_sphere_wavelengths[0]:.3f} um in {len(list_of_sphere_wavelengths)} intervals.')
    print(f'\t Radii are between {list_of_sphere_radii[0]} --> {list_of_sphere_radii[num_radii-1]} um in {len(list_of_sphere_radii)} intervals.')

    #print(list_of_sphere_radii)

    # load the data for the particle radius and column_density values for spheres
    column_density_data_spheres = clouds_from_virga_spheres['column_density'][(clouds_from_virga_spheres['column_density'] != 0.0).all(1)] # find the rows of data where we have clouds (where column_density was not equal to 0.0)
    radius_data_spheres = clouds_from_virga_spheres['mean_particle_r'].flatten(order='C')[(clouds_from_virga_spheres['column_density'] != 0.0).all(1)] # find the rows of data where we have clouds (where column_density was not equal to 0.0)



    # -------- AGGREGATES -------- 

    radius_data_aggregates_list=[] # initialise list to hold all of the radius values (in the cloud only) each fractal dimension
    column_density_data_aggregates_list=[] # initialise list to hold all of the column_density values (in the cloud only) each fractal dimension

    for j in range(len(d_f_list)):

        # ---- SAFETY CHECK -----

        # first, check that we used the same wavelength-radius grid as the spherical model for each fractal dimension (we don't need to use the data from these .mieff files again otherwise -- this is just a safety check)
        mieff_data = pd.read_csv(f"{mieff_dir}/{aggregate}_aggregates_Df_{d_f_list[j]:.6f}.mieff", names=mieff_col_names, header=None, sep=r'\s+') # load data

        # extract number of wavelenths and radii used in the grid
        num_wavelengths_aggregate = int(mieff_data['A'][0]) # extract number of wavelengths from header
        num_radii_aggregate = int(mieff_data['B'][0]) # extract number of radii from header

        # initialise lists
        list_of_aggregate_wavelengths=[]
        list_of_aggregate_radii=[]

        # record the actual wavelengths used in the aggregate .mieff file
        for i in range(num_wavelengths_aggregate):
            list_of_aggregate_wavelengths.append(mieff_data['A'][i+2])

        # record the actual radii used in the aggregate .mieff file
        for i in range(num_radii_aggregate):
            list_of_aggregate_radii.append(10000.0*mieff_data['A'][i*num_wavelengths_aggregate+i+1]) # convert all radii from cm (default for the .mieff file) to um as we record them

        # check wavelength grid is the same as for spheres

        for i in range(len(list_of_aggregate_wavelengths)):
            if(abs(list_of_aggregate_wavelengths[i] - list_of_sphere_wavelengths[i])>0.000001): # if the list of wavelengths is not equal to the version used in the spherical .mieff file, to 6 decimal places
                print('\n\n WARNING: Wavelength grid is not the same (to 6 dp)!') # print a warning
                print(f"\n Spherical version: ({num_wavelengths} wavelengths) does not match grid for aggregate {d_f_list[j]:.6f} ({int(mieff_data['A'][0])} wavelengths):\n") # explain where the error is
                print(f"\n The exact error occurs for SPHERE wavelength: {list_of_sphere_wavelengths[i]} um, where the AGGREGATE version is: {list_of_aggregate_wavelengths[i]} um.")
                print('\n Exiting code.\n')
                sys.exit() # exit the code

        # check radius grid is the same as for spheres
        for i in range(len(list_of_aggregate_radii)):
            if(abs(list_of_aggregate_radii[i] - list_of_sphere_radii[i])>0.000001): # if the list of radii is not equal to the version used in the spherical .mieff file, to 6 decimal places
                print('\n\n WARNING: Radius grid is not the same (to 6 dp)!') # print a warning
                print(f"\n Spherical version: ({num_radii} radii) does not match grid for aggregate {d_f_list[j]:.6f} ({int(mieff_data['B'][0])} radii):\n") # explain where the error is
                print(f"\n The exact error occurs for SPHERE radius: {list_of_sphere_wavelengths[i]} um, where the AGGREGATE version is: {list_of_aggregate_wavelengths[i]} um.")
                print('\n Exiting code.\n')
                sys.exit() # exit the code'''

        # ---- SAFETY CHECK COMPLETE -----


        # load the cloud data for the particle radius and column_density values for each aggregate and find the pressures that had clouds
        reduced_column_density_data = clouds_from_virga_fractals[j]['column_density'][(clouds_from_virga_fractals[j]['column_density'] != 0.0).all(1)] # find the rows of data where we have clouds (where column_density was not equal to 0.0)
        reduced_radius_data = clouds_from_virga_fractals[j]['mean_particle_r'].flatten(order='C')[(clouds_from_virga_fractals[j]['column_density'] != 0.0).all(1)] # find the rows of data where we have clouds (where column_density was not equal to 0.0)
        reduced_pressure_data = clouds_from_virga_fractals[j]['pressure'][(clouds_from_virga_fractals[j]['column_density'] != 0.0).all(1)] # find the rows of data where we have clouds (where column_density was not equal to 0.0) - we only need one of these, as all aggregates are on the same pressure grid (so no need to add to a list below)

        column_density_data_aggregates_list.append(reduced_column_density_data) # add this data to a list (one for each fractal dimension)
        radius_data_aggregates_list.append(reduced_radius_data) # add this data to a list (one for each fractal dimension)
        

    # check whether any of the particle radii are outside of the wavelength-radius grid
    warning_flag=0
    for j in range(len(radius_data_aggregates_list)):
        for i in range(len(radius_data_aggregates_list[j])):
            if (radius_data_aggregates_list[j][i] < list_of_sphere_radii[0]) or (radius_data_aggregates_list[j][i] > list_of_sphere_radii[num_radii-1]): # if the radius for one of the fractal dimensions is less than the minimum radius or larger than the maximum for the grid (which we have already checked is the same as the one in the spherical version)...
                print(f"WARNING: {d_f_list[j]} has a radius off the grid ({radius_data_aggregates_list[j][i]:.6f} um, at pressure {reduced_pressure_data[i]:.6f} bar).\n") #...print a warning and highlight which fractal dimension and radius caused it
                warning_flag=1
    
    if(warning_flag==0):
        print("\nGood news - all aggregates are within the grid!")

    # Create scatterplot figure
    fig,ax = plt.subplots()
    fig.set_facecolor('w') # make area around plot white in case user is in dark mode

    ax2 = ax.twinx() # make second y-axis (on right - column_density)

    # make wavelength-radius grid    
    ax.vlines(list_of_sphere_radii, list_of_sphere_wavelengths[num_wavelengths-1], list_of_sphere_wavelengths[0],  colors='lightgrey', linewidth=0.5) # add vertical lines for each radius in grid

    edge_of_grid_color = 'red' # choose colour for box marking outside of grid
    ax.vlines(list_of_sphere_radii[0], list_of_sphere_wavelengths[num_wavelengths-1], list_of_sphere_wavelengths[0],  colors=edge_of_grid_color)
    ax.vlines(list_of_sphere_radii[num_radii-1], list_of_sphere_wavelengths[num_wavelengths-1], list_of_sphere_wavelengths[0],  colors=edge_of_grid_color)
    ax.hlines(list_of_sphere_wavelengths[0], list_of_sphere_radii[0], list_of_sphere_radii[num_radii-1], colors=edge_of_grid_color)
    ax.hlines(list_of_sphere_wavelengths[num_wavelengths-1], list_of_sphere_radii[0], list_of_sphere_radii[num_radii-1], colors=edge_of_grid_color)


    marker_size = 15 # set marker size

    # SPHERES: add scatter plot of radii and 'column_density' for each pressure layer of cloud
    ax2.scatter(radius_data_spheres, column_density_data_spheres, color = color_array[0], label= f'Spheres', s=marker_size)

    # AGGREGATES: add scatter plot of radii and 'column_density' for each pressure layer of cloud
    for j in range(len(d_f_list)):
        ax2.scatter(radius_data_aggregates_list[j], column_density_data_aggregates_list[j], color = color_array[j+1], label= f'{d_f_list[j]:.1f}', s=marker_size)

    # make log-log plot
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax2.set_yscale('log')

    # set axes titles
    ax.set_xlabel(r'Compact particle $r$ ($\mu$m)', fontsize=23)
    ax.set_ylabel(r'$\lambda$ ($\mu$m)', fontsize=23)
    ax2.set_ylabel(r'Column Density in layer ($\mathrm{cm}^{-2}$)', fontsize=23)

    # set tick font size
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=20)
    ax2.tick_params(axis='both', which='minor', labelsize=20)
        
    #plt.legend(loc='lower left', bbox_to_anchor=(0.1, 0.1), prop={'size': 18})

    plt.close() # prevents duplicate plots from appearing

    return fig
    

def aggregates_pressure_vs_number_density(clouds_from_virga_spheres, clouds_from_virga_fractals, d_f_list, colors=None, min_pressure=None, max_pressure=None):

    '''

    Code to plot the particle densities at each radius and at each pressure layer for aggregates of each fractal dimension.
    This basically allows you to see the relative numbers and radii of the particles that form in each layer, for each fractal dimension.

    Parameters
    ----------
    clouds_from_virga_spheres: dict
        output from virga (spherical model)
    clouds_from_virga_fractals: dict
        array of dicts with output from virga (all of the virga models for each fractal dimension)
    d_f_list: array
        list of fractal dimensions analysed
    colors: array (optional)
        list of colors to use (in order: spheres, 1.2, 1.6, 2.0, 2.4, 2.8)
    min_pressure: float (optional)
        minimum pressure (bars) to plot on y-axis
    max_pressure: float (optional)
        maximum pressure (bars) to plot on y-axis

    Returns
    -------
    Figure with the particle densities at each radius and at each pressure layer for aggregates of each fractal dimension

    '''

    
    # if colors = None, choose a colormap (based on the number of shapes to analyse).
    if(colors is None):
        color_array = plt.cm.viridis(np.linspace(0,1,len(d_f_list)+1))
    else:
        color_array = colors # or use colors provided by user

    # set maximum and minimum marker sizes (on the screen) - these are arbitrary, and we can scale them to whatever looks good visually
    max_marker_size = 1000
    min_marker_size = 1

    # initialise a list to hold all of the data
    radius_data_list=[]

    # read in all datasets into a single list
    for i in range(len(d_f_list)):
        radius_data_list.extend(clouds_from_virga_fractals[i]['mean_particle_r'].flatten(order='C')) # add each fractal
    radius_data_list.extend(clouds_from_virga_spheres['mean_particle_r'].flatten(order='C')) # add spheres

    # find th min/max values that we will need to display on-screen
    global_min_r = min(n for n in radius_data_list if n>0) # record the minimum particle size (that is still above zero)
    global_max_r = max(radius_data_list) # record the maximum particle size

    #print(f'\nFINAL RESULTS: Min = {global_min_r}, Max = {global_max_r}')

    # calculate min and max r2 values because these relate linearly to the marker sizes, which are areas
    min_r_squared = global_min_r**2
    max_r_squared = global_max_r**2

    # scale the min/max radii so that they correspond with the min/max marker sizes that we want to show on screen
    marker_size_per_square_radii = (max_marker_size - min_marker_size) / (max_r_squared - min_r_squared)

    # Create scatterplot figure
    fig,ax = plt.subplots()
    fig.set_facecolor('w') # make area around plot white in case user is in dark mode

    # plot values for each fractal dimension d_f
    for i in range(len(d_f_list)):
        # create a list of marker sizes for the radii in this fractal dimension
        square_radii = clouds_from_virga_fractals[i]['mean_particle_r'].flatten(order='C')**2 # first, simply square the radii
        marker_sizes = min_marker_size + marker_size_per_square_radii * (square_radii - min_r_squared) # use derived equation to find the marker area that would give accurately scaled radii on screen
        # make scatter plot 
        legend_handle=ax.scatter(clouds_from_virga_fractals[i]['column_density'], clouds_from_virga_fractals[i]['pressure'], color = color_array[i+1], s=marker_sizes, label= f'{d_f_list[i]:.1f}') # legend handle is just there to make the legend markers all the same size (see it's use below)

    # plot values for spheres
    # create a list of marker sizes for the radii for spheres
    square_radii = clouds_from_virga_spheres['mean_particle_r'].flatten(order='C')**2 # first, simply square the radii
    marker_sizes = min_marker_size + marker_size_per_square_radii * (square_radii - min_r_squared) # use derived equation to find the marker area that would give accurately scaled radii on screen

    ax.scatter(clouds_from_virga_spheres['column_density'], clouds_from_virga_spheres['pressure'], color = color_array[0], s=marker_sizes, label= f'Spheres')

    # make log-log plot
    ax.set_yscale('log')
    ax.set_xscale('log')

    if (min_pressure is not None) and (max_pressure is not None):
        # if provided, set limits on y-axis (to begin at the cloud deck and above)
        ax.set_ylim(min_pressure, max_pressure)

    # invert the y-axis so that pressure decreases as you go upwards (basically representing altitude)
    plt.gca().invert_yaxis()

    # set axes titles
    ax.set_xlabel(r'Column Density in layer ($\mathrm{cm}^{-2}$)', fontsize=20)
    ax.set_ylabel(r'$P$ (bar)', fontsize=20)

    # set tick font size
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)

        
    # produce nice equal marker sizes in the legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(list(reversed(handles)), list(reversed(labels)), title='$d_f$', title_fontsize=12, fontsize=12, loc='lower right')
    marker_size = 36
    def update_prop(handle, orig):
        handle.update_from(orig)
        handle.set_sizes([marker_size])
    plt.legend(handler_map={type(legend_handle): HandlerPathCollection(update_func=update_prop)})
    
    plt.close() # prevents duplicate plots from appearing

    return fig

def aggregates_pressure_vs_radius(clouds_from_virga_spheres, clouds_from_virga_fractals, d_f_list, colors=None, width=3, min_pressure=None, max_pressure=None):
    
    '''
    
    Code to plot the particle radius at each pressure layer for aggregates of each fractal dimension.
    Designed originally for diagnostic tests as part for VIRGA development by MGL.

    Parameters
    ----------
    clouds_from_virga_spheres: dict
        output from virga (spherical model)
    clouds_from_virga_fractals: dict
        array of dicts with output from virga (all of the virga models for each fractal dimension)
    d_f_list: array
        list of fractal dimensions analysed
    colors: array (optional)
        list of colors to use (in order: spheres, 1.2, 1.6, 2.0, 2.4, 2.8)
    width: float (optional)
        width of lines in plot (default = 3)
    min_pressure: float (optional)
        minimum pressure (bars) to plot on y-axis
    max_pressure: float (optional)
        maximum pressure (bars) to plot on y-axis
    
    Returns
    -------
    Figure plotting the particle radius at each pressure layer for aggregates of each fractal dimension.

    '''

    # if colors = None, choose a colormap (based on the number of shapes to analyse).
    if(colors is None):
        color_array = plt.cm.viridis(np.linspace(0,1,len(d_f_list)+1))
    else:
        color_array = colors # or use colors provided by user
        
    # Create scatterplot figure
    fig,ax = plt.subplots()
    fig.set_facecolor('w') # make area around plot white in case user is in dark mode

    # plot values for each fractal dimension d_f
    for i in range(len(d_f_list)):
        ax.plot(clouds_from_virga_fractals[i]['mean_particle_r'].flatten(order='C'), clouds_from_virga_fractals[i]['pressure'], color = color_array[i+1], label= f'{d_f_list[i]:.1f}', linewidth=width)

    # plot values for spheres
    ax.plot(clouds_from_virga_spheres['mean_particle_r'].flatten(order='C'), clouds_from_virga_spheres['pressure'], color = color_array[0], label= f'Spheres', linewidth=width)

    # make log-log plot
    ax.set_yscale('log')
    ax.set_xscale('log')

    if (min_pressure is not None) and (max_pressure is not None):
        # if provided, set limits on y-axis (to begin at the cloud deck and above)
        ax.set_ylim(min_pressure, max_pressure)

    # invert the y-axis so that pressure decreases as you go upwards (basically representing altitude)
    plt.gca().invert_yaxis()

    # set axes titles
    ax.set_xlabel(r'Compact particle $r$ ($\mu$m)', fontsize=20)
    ax.set_ylabel(r'$P$ (bar)', fontsize=20)

    # set tick font size
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)

    plt.legend(prop={'size': 12})

    plt.close() # prevents duplicate plots from appearing

    return fig
