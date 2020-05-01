from bokeh.palettes import viridis,magma
from bokeh.models import ColumnDataSource, Label, LabelSet,CustomJS
from bokeh.layouts import column,row
from bokeh.plotting import figure, show
from bokeh.models import LinearColorMapper, LogTicker,BasicTicker, ColorBar,LogColorMapper,Legend
from bokeh.palettes import magma as colfun1
from bokeh.palettes import viridis as colfun2
from bokeh.palettes import gray as colfun3

import astropy.units as u
import numpy as np

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

    kwargs['plot_height'] = kwargs.get('plot_height',300)
    kwargs['plot_width'] = kwargs.get('plot_width',600)
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


def radii(out,gas=None,at_pressure = 1e-3):
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

    lines = ['solid','dotdash','dashdot']
    for j in range(len(out)):
        #compute initial distributions   
        r_g = out[j]['mean_particle_r']
        pressure = out[j]['pressure']

        nl = find_nearest_1d(pressure,at_pressure)

        rmin = out[j]['scalar_inputs']['rmin']
        nrad = out[j]['scalar_inputs']['nrad']
        sig = out[j]['scalar_inputs']['sig']
        ndz = out[j]['column_density']
        #determine which condensed
        which_condensed = [False]*ndz.shape[1]

        # only conisider specified gas
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
        r, rup, dr = pyeddy.get_r_grid(r_min = rmin, n_radii = nrad)

        # different colours for different dicts
        if gas is not None:
            length = len(out)
        # different colours for different gases
        else:
            length = len(gas_name) 

        color = magma(length)

        #initial radii profiles
        df_r_g = {i:r_g[:, gas_name.index(i)] for i in gas_name}
        df_r_g['pressure'] = pressure


        dndr = {}
        for i in gas_name:
            dndr[i]= N[nl,gas_name.index(i)]/r/np.sqrt(2*np.pi)*np.log(sig)*np.exp(
                        - np.log(r/(r_g[nl,gas_name.index(i)]*1e-4))**2/(2*np.log(sig)**2)) #1e-4 is microns to cm
        dndr['r'] = r*1e4 #convert to microns

        if j==0:
            p1 = figure(plot_width=300, plot_height=300, title="Select Pressure Level", 
                x_axis_type='log',y_axis_type='log',y_axis_label='Pressure (bars)',x_axis_label='Mean Particle Radius (um)',
                y_range=[np.max(pressure),np.min(pressure)])

            p2 = figure(plot_width=275, plot_height=300, title="Particle Distribution",
                         x_axis_type='log', y_axis_type='log', y_axis_label='dn/dr (cm-3)',x_axis_label='Particle Radius (um)')

        #add to r_g for that plot 
        df_r_g['average'] = [pressure[nl]]*len(pressure)
        df_r_g['horizontal'] = np.linspace(np.amin(r_g[r_g>0]), np.amax(r_g) , len(pressure))
            

        s1 = ColumnDataSource(data=dict(df_r_g))
        s2 = ColumnDataSource(data=dict(dndr))
        for i in gas_name:
            if gas is not None: indx = j
            else: indx = gas_name.index(i)
            p1.line(i, 'pressure', source=s1, alpha=1,color=color[indx] ,line_width=4,legend_label=i,line_dash=lines[j])
            p2.line('r', i, source=s2,color=color[indx] ,line_width=4,line_dash=lines[j])
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
    return row(p1, p2), dndr

def opd_by_gas(out, gas = None, color = magma, **kwargs):
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

    kwargs['plot_height'] = kwargs.get('plot_height',300)
    kwargs['plot_width'] = kwargs.get('plot_width',400)
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
    lines = ['solid','dashed','dotdash','dashdot']

    for j in range(len(out)):
        pressure = out[j]['pressure']
        opd_by_gas = out[j]['opd_by_gas']

        if j == 0:
            kwargs['y_range'] = kwargs.get('y_range',[np.max(pressure), np.min(pressure)])
            kwargs['x_range'] = kwargs.get('x_range',[np.max([1e-6, np.min(opd_by_gas*0.9)]), 
                                                np.max(opd_by_gas*1.1)])
            fig = figure(**kwargs)

        if gas is not None:
            fig.line(opd_by_gas[:,indx], pressure, line_width=4, legend_label = condensibles[indx], color=col[j], line_dash=lines[j])
        else:
            for i in range(ngas):
                fig.line(opd_by_gas[:,i], pressure,line_width=4, legend_label = condensibles[i], color=col[i], line_dash=lines[j])

    plot_format(fig)
    return fig

def condensate_mmr(out, gas=None, color = magma, **kwargs):
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

    kwargs['plot_height'] = kwargs.get('plot_height',300)
    kwargs['plot_width'] = kwargs.get('plot_width',400)
    kwargs['x_axis_label'] = kwargs.get('x_axis_label','Condensate MMR')
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
    lines = ['solid','dashed','dotdash','dashdot']

    for j in range(len(out)):
        pressure = out[j]['pressure']
        cond_mmr = out[j]['condensate_mmr']

        if j == 0:
            kwargs['y_range'] = kwargs.get('y_range',[np.max(pressure), np.min(pressure)])
            kwargs['x_range'] = kwargs.get('x_range',[np.max([1e-9, np.min(cond_mmr*0.9)]), 
                                                    np.max(cond_mmr*1.1)])
            fig = figure(**kwargs)

        if gas is not None:
            fig.line(cond_mmr[:,indx], pressure, line_width=4, legend_label = condensibles[indx], color=col[j], line_dash=lines[j])
        else:
            for i in range(ngas):
                fig.line(cond_mmr[:,i], pressure, line_width=4, legend_label = condensibles[i], color=col[i], line_dash=lines[j])

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
                          plot_width=300, plot_height=300)


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
                          plot_width=320, plot_height=300)

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
                          plot_width=300, plot_height=300)

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

    return row(f01a, f01,f01b)

def find_nearest_1d(array,value):
    #small program to find the nearest neighbor in a matrix
    ar , iar ,ic = np.unique(array,return_index=True,return_counts=True)
    idx = (np.abs(ar-value)).argmin(axis=0)
    if ic[idx]>1: 
        idx = iar[idx] + (ic[idx]-1)
    else: 
        idx = iar[idx]
    return idx

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
