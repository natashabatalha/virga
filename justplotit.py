from bokeh.plotting import figure
import pvaps 
import gas_properties
from bokeh.palettes import viridis
from bokeh.models import ColumnDataSource, Label, LabelSet
import numpy as np

def pt_pvaps(atmo):
    mh = atmo.mh
    condensibles = atmo.condensibles

    temp = atmo.temperature
    pressure = atmo.pressure

    pt_fig = figure(y_range=[np.max(pressure),np.min(pressure)], plot_height=400, plot_width=400,
             y_axis_type='log',x_axis_label='Temperature (K)', y_axis_label='Pressure (bars)',
             x_range=[np.min(temp),np.max(temp)])

    colors = viridis(len(condensibles))
    cond_dict = {}
    for igas in condensibles:
        get_pvap = getattr(pvaps, igas)
        if igas == 'Mg2SiO4':
            cond_dict[igas] = get_pvap(temp, pressure, mh=np.log10(mh))
        else:
            cond_dict[igas] = get_pvap(temp, mh=np.log10(mh))
    labels = condensibles
    x=[]
    for i,c in zip(labels, colors):
        p_where = np.where((cond_dict[i] < np.max(pressure)) 
                        & (cond_dict[i] > np.min(pressure)))
        x += [np.mean(temp[p_where])-0.5*np.std(temp[p_where])]
    x, labels = (list(t) for t in zip(*sorted(zip(x, labels))))

    for i,c in zip(labels, colors):
        pt_fig.line(temp,cond_dict[i] , line_width=3, color=c, line_dash='dashed')


    #some logic to create staircase y values for the labels 
    y = np.array(range(len(labels)))-4.5
    yind=[]
    for i in range(len(labels)): 
        yind += [y[np.mod(i, len(y))]]

    source = ColumnDataSource(data=dict(x=x, y=10**np.array(yind),
             text=labels, color =colors))

    conv = LabelSet(x='x', y='y', #x_units='screen', y_units='screen',
             text='text', source=source, text_color='color')
    pt_fig.add_layout(conv)

    #fianlly plot pt 
    pt_fig.line(temp, pressure, line_width=3, color='black')

    return pt_fig

    


