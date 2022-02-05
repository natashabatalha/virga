from bokeh.palettes import viridis,magma
from bokeh.models import ColumnDataSource, Label, LabelSet,CustomJS
from bokeh.layouts import column,row
from bokeh.plotting import figure, show
from bokeh.models import LinearColorMapper, LogTicker,BasicTicker, ColorBar,LogColorMapper,Legend
from bokeh.palettes import magma as colfun1
from bokeh.palettes import viridis as colfun2
from bokeh.palettes import gray as colfun3
from bokeh.palettes import plasma as colfun4
from bokeh.palettes import Colorblind8
 
import astropy.units as u
import numpy as np

from . import justdoit as pyeddy

def plot_pt(out, labels, lines, **kwargs):

    kwargs['plot_height'] = kwargs.get('plot_height',300)
    kwargs['plot_width'] = kwargs.get('plot_width',600)
    kwargs['x_axis_label'] = kwargs.get('x_axis_label','Temperature (K)')
    kwargs['y_axis_label'] = kwargs.get('y_axis_label','Pressure (bars)')
    kwargs['x_axis_type'] = kwargs.get('x_axis_type','log')
    kwargs['y_axis_type'] = kwargs.get('y_axis_type','log')
    

    cols = viridis(len(out))
    ngas = len(out[0]['condensibles'])
    mh, mmw = out[0]['scalar_inputs']['mh'], out[0]['scalar_inputs']['mmw']
    condensibles = out[0]['condensibles']
    pressure = out[0]['pressure']
    kwargs['y_range'] = kwargs.get('y_range',[np.max(pressure), np.min(pressure)])
    fig = figure(**kwargs)
    for i in range(len(out)):
        temperature = out[i]['temperature']
        pressure = out[i]['pressure']

        fig.line(temperature, pressure, legend_label=labels[i], color=cols[i],line_width=5, line_dash=lines[i])

    plot_format(fig)
    return fig

def plot_cumsum(out,labels,lines,**kwargs):

    kwargs['plot_height'] = kwargs.get('plot_height',300)
    kwargs['plot_width'] = kwargs.get('plot_width',600)
    kwargs['y_axis_label'] = kwargs.get('y_axis_label','Pressure (bars)')
    kwargs['x_axis_type'] = kwargs.get('x_axis_type','log')
    kwargs['y_axis_type'] = kwargs.get('y_axis_type','log')

    cols = viridis(len(out))
    pressure = out[0]['pressure']
    kwargs['y_range'] = kwargs.get('y_range',[np.max(pressure), np.min(pressure)])
    fig = figure(**kwargs)
    for i in range(len(out)):  
        x = np.cumsum(out[i]["opd_per_layer"][:,40])
        pressure = out[i]['pressure']

        fig.line(x, pressure, legend_label=labels[i], color=cols[i],line_width=5, line_dash=lines[i])

    fig.legend.location = "bottom_left"
    plot_format(fig)
    return fig

def plot_output(out,attribute,attribute_label,gas,labels,lines,legend_on=True,
                    color_indx=0,**kwargs):

    condensibles = out[0]['condensibles']
    kwargs['plot_height'] = kwargs.get('plot_height',400)
    kwargs['plot_width'] = kwargs.get('plot_width',350)
    kwargs['x_axis_label'] = kwargs.get('x_axis_label',attribute_label)
    kwargs['y_axis_label'] = kwargs.get('y_axis_label','Pressure (bars)')
    kwargs['x_axis_type'] = kwargs.get('x_axis_type','log')
    kwargs['y_axis_type'] = kwargs.get('y_axis_type','log')
    kwargs['x_range'] = kwargs.get('x_range', [1e-2, 1e4])

    cols = viridis(len(out))
    cols = Colorblind8[color_indx:color_indx+len(out)]
    pressure = out[0]['pressure']
    kwargs['y_range'] = kwargs.get('y_range',[np.max(pressure), np.min(pressure)])
    fig = figure(**kwargs)
    for i in range(len(out)):
        indx = out[i]['condensibles'].index(gas)
        x = out[i][attribute][:,indx]
        if attribute is "column_density":
            x = out[i][attribute][:,indx]/out[i]["layer_thickness"]
        pressure = out[i]['pressure']

        fig.line(x, pressure, legend_label=labels[i], color=cols[i],line_width=5, line_dash=lines[i])

    if legend_on:
        fig.legend.location = "bottom_left"
    plot_format(fig)
    return fig

def opd_by_gas(out,color = magma, **kwargs):
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

    ngas = len(out['condensibles'])
    temperature = out['temperature']
    pressure = out['pressure']
    condensibles = out['condensibles']
    opd_by_gas = out['opd_by_gas']

    kwargs['y_range'] = kwargs.get('y_range',[np.max(pressure), np.min(pressure)])
    kwargs['x_range'] = kwargs.get('x_range',[np.max([1e-6, np.min(opd_by_gas*0.9)]), 
                                            np.max(opd_by_gas*1.1)])
    fig = figure(**kwargs)
    col = color(ngas)
    for i in range(ngas):
        fig.line(opd_by_gas[:,i], pressure,line_width=4,legend_label = condensibles[i],color=col[i])

    plot_format(fig)
    return fig

def condensate_mmr(out,color = viridis, **kwargs):
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

    ngas = len(out[0]['condensibles'])
    temperature = out['temperature']
    pressure = out['pressure']
    condensibles = out['condensibles']
    cond_mmr = out['condensate_mmr']

    kwargs['y_range'] = kwargs.get('y_range',[np.max(pressure), np.min(pressure)])
    kwargs['x_range'] = kwargs.get('x_range',[np.max([1e-9, np.min(cond_mmr*0.9)]), 
                                            np.max(cond_mmr*1.1)])
    fig = figure(**kwargs)
    col = color(ngas)
    for i in range(ngas):
        fig.line(cond_mmr[:,i], pressure,line_width=4,legend_label = condensibles[i],color=col[i])

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
    dat01 = pyeddy.picaso_format(out['opd_per_layer'],
                      out['asymmetry'],out['single_scattering'])

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
                           title="Assymetry Parameter",
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


def plot_fsed(out,labels,y_axis='pressure',color_indx=0,cld_bounds=False,gas_indx=None,**kwargs):

    kwargs['plot_height'] = kwargs.get('plot_height',400)
    kwargs['plot_width'] = kwargs.get('plot_width',700)
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

        if y_axis is 'pressure':
            y = out[i]['pressure']
        elif y_axis is 'z':
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


    fig.legend.location = "bottom_left"
    plot_format(fig)
    return fig

