'''

A suite of tools written by MGL to analyse spectra and other diagnostics produced by atmospheric aggregates in VIRGA.

'''

# import key modules
import matplotlib.pyplot as plt
import csv
import pandas as pd
import sys
import numpy as np
from matplotlib.ticker import FuncFormatter # used to display axes labels as 0.1 instead of 10^-1 power notation
from matplotlib.ticker import FormatStrFormatter
from statistics import mean
from scipy.signal import savgol_filter
from matplotlib.legend_handler import HandlerPathCollection
import matplotlib
matplotlib.use('qt5Agg') 
from .root_functions import calculate_k0, find_N_mon_and_r_mon



def plot_spectra(spectra_path, d_f_list, width=2, filtered=False):
        
    '''

    Code to plot the transmission spectra for aggregates of each fractal dimension.

    Set filtered = True if you want a smoother curve (using a savgol filter) for easier comparison of different fractals.

    INPUTS:

    -   spectra_path: (string)
            filepath to directory that contains .csv files for each aggregate (+ sphere) models, each with 2-columns (wavelength in um, transit depth in %) 
    -   d_f_list: (array)
            list of fractal dimensions analysed
    -   width: (float - optional)
            width of lines in plot (default = 2)
    -   filtered: (boolean - optional)
            if you want a smoother curve (using a savgol filter) for easier comparison of different fractals, set equal to True (default=False)   
    
    '''

    # choose a colormap (based on the number of shapes to analyse)
    colors = plt.cm.winter_r(np.linspace(0,1,len(d_f_list)))

    fig,ax = plt.subplots()
    fig.set_facecolor('w') # make area around plot in case user is in dark mode

    # plot clear spectrum
    spectra_data= pd.read_csv(f"{spectra_path}/clear.txt", header=None) # load data

    if(filtered==True):
        smoothed_y_axis = savgol_filter(spectra_data[1], 25, 5) # smooth the y-axis to make it look nicer for the plot
        ax.plot(spectra_data[0].values, smoothed_y_axis, color = 'k', label= r'No condensate', linewidth=width)
    else:
        ax.plot(spectra_data[0].values, spectra_data[1].values, color = 'k', label= r'No condensate', linewidth=width)

    # plot spheres
    spectra_data= pd.read_csv(f"{spectra_path}/spheres.txt", header=None) # load data
    if(filtered==True):
        smoothed_y_axis = savgol_filter(spectra_data[1], 25, 5) # smooth the y-axis to make it look nicer for the plot
        ax.plot(spectra_data[0].values, smoothed_y_axis, color = 'm', label= r'spheres', linewidth=width)
    else:
        ax.plot(spectra_data[0].values, spectra_data[1].values, color = 'm', label= r'spheres', linewidth=width)


    for i in range(len(d_f_list)):
        
        # read spectrum for each d_f value
        spectra_data= pd.read_csv(f"{spectra_path}/{d_f_list[i]:.1f}.txt", header=None)
        if(filtered==True):
            smoothed_y_axis = savgol_filter(spectra_data[1], 25, 5) # smooth the y-axis to make it look nicer for the plot
            ax.plot(spectra_data[0].values, smoothed_y_axis, color = colors[i], label= f'{d_f_list[i]:.1f}', linewidth=width)
        else:
            ax.plot(spectra_data[0].values, spectra_data[1].values, color = colors[i], label= f'{d_f_list[i]:.1f}', linewidth=width)

    
    # set axes titles
    ax.set_xlabel(r'$\lambda (\mu m)$', fontsize=23)
    ax.set_ylabel(r'Transit depth (%)', fontsize=23)
    
    ax.set_xscale('log') # log the wavelength axis

    # set tick font size
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)
        
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels), title='$d_f$', title_fontsize=15, fontsize=15, loc='lower right')

    return fig

def plot_emission_spectra(spectra_path, d_f_list, width=2, filtered=False):
        
    '''

    Code to plot the emission spectra for aggregates of each fractal dimension.

    Set filtered = True if you want a smoother curve (using a savgol filter) for easier comparison of different fractals.

    INPUTS:

    -   spectra_path: (string)
            filepath to directory that contains .csv files for each aggregate (+ sphere) models, each with 2-columns (wavelength in um, transit depth in %) 
    -   d_f_list: (array)
            list of fractal dimensions analysed
    -   width: (float - optional)
            width of lines in plot (default = 2)
    -   filtered (boolean - optional)
            if you want a smoother curve (using a savgol filter) for easier comparison of different fractals, set equal to True (default=False)

    '''

    # choose a colormap (based on the number of shapes to analyse)
    colors = plt.cm.winter_r(np.linspace(0,1,len(d_f_list)))

    fig,ax = plt.subplots()
    fig.set_facecolor('w') # make area around plot in case user is in dark mode

    # plot clear spectrum
    spectra_data= pd.read_csv(f"{spectra_path}/clear.txt", header=None) # load data

    if(filtered==True):
        smoothed_y_axis = savgol_filter(spectra_data[1], 25, 5) # smooth the y-axis to make it look nicer for the plot
        ax.plot(spectra_data[0].values, smoothed_y_axis, color='k', label= r'No condensate', linewidth=width)
    else:
        ax.plot(spectra_data[0].values, spectra_data[1].values, color='k', label= r'No condensate', linewidth=width)

    # plot spheres
    spectra_data= pd.read_csv(f"{spectra_path}/spheres.txt", header=None) # load data
    if(filtered==True):
        smoothed_y_axis = savgol_filter(spectra_data[1], 25, 5) # smooth the y-axis to make it look nicer for the plot
        ax.plot(spectra_data[0].values, smoothed_y_axis, color = 'm', label= r'spheres', linewidth=width)
    else:
        ax.plot(spectra_data[0].values, spectra_data[1].values, color = 'm', label= r'spheres', linewidth=width)


    for i in range(len(d_f_list)):
        
        # read spectrum for each d_f value
        spectra_data= pd.read_csv(f"{spectra_path}/{d_f_list[i]:.1f}.txt", header=None)
        if(filtered==True):
            smoothed_y_axis = savgol_filter(spectra_data[1], 25, 5) # smooth the y-axis to make it look nicer for the plot
            ax.plot(spectra_data[0].values, smoothed_y_axis, color = colors[i], label= f'{d_f_list[i]:.1f}', linewidth=width)
        else:
            ax.plot(spectra_data[0].values, spectra_data[1].values, color = colors[i], label= f'{d_f_list[i]:.1f}', linewidth=width)


    # set axes titles
    ax.set_xlabel(r'$\lambda (\mu m)$', fontsize=23)
    ax.set_ylabel(r'Emission Flux (erg $cm^{-3}$ $s^{-1}$)', fontsize=18)
    
    ax.set_yscale('log') # log the flux axis
    ax.set_xscale('log') # log the wavelength axis
     

    # set tick font size
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)
        
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels), title='$d_f$', title_fontsize=12, fontsize=12, loc='upper right')

    return fig






def plot_optical_properties(aggregate, mieff_dir, d_f_list, min_wavelength, max_wavelength):

    '''
    Code to plot the optical properties calculated in the .mieff files. for all radii and fractal dimensions analysed.

    Plots a 3D graph of Q_ext against wavelength (um) for all fractal dimensions (third axis).

    Set filtered = True if you want a smoother curve (using a savgol filter) for easier comparison of different fractals.

    INPUTS:

    -   aggregate: (string)
            chemical species used in model 
    -   mieff_dir: (string)
            filepath to directory where .mieff files are stored
    -   d_f_list: (array)
            list of fractal dimensions analysed
    -   min_wavelength; (float)
            minimum wavelength to plot
    -   max_wavelength; (float)
            maximum wavelength to plot

    '''

    # choose a colormap (based on the number of shapes to analyse)
    colors = plt.cm.winter_r(np.linspace(0,1,len(d_f_list)))


    # -------- SPHERES -------- 

    # read in the dataset for spherical particles
    mieff_col_names= ['A', 'B', 'C', 'D'] # assign arbitrary column names for reading in the data (all the columns have different numbers of rows)
    mieff_data = pd.read_csv(f"{mieff_dir}/{aggregate}.mieff", names=mieff_col_names, header=None, sep='\s+')

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
        mieff_data = pd.read_csv(f"{mieff_dir}/{aggregate}_aggregates_Df_{d_f_list[j]:.6f}.mieff", names=mieff_col_names, header=None, sep='\s+')

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
    fig.set_facecolor('w') # make area around plot in case user is in dark mode
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

    return fig


def plot_refractive_index(aggregate, refrind_dir, min_wavelength, max_wavelength):
        
    ''' 
    
    Plot refractive index of chosen aggregate (as a function of wavelength)

    -   aggregate: (string)
            chemical species used in model 
    -   refrind_dir: (string)
            filepath to directory where .refrind files are stored
    -   min_wavelength: (float)
            minimum wavelength to plot
    -   max_wavelength: (float)
            maximum wavelength to plot
    
    '''

    # Create scatterplot figure
    fig,ax = plt.subplots()
    fig.set_facecolor('w') # make area around plot in case user is in dark mode

    # read in the .refrind dataset
    refrind_col_names= ['index', 'wavelength', 'n', 'k'] # assign arbitrary column names for reading in the data (all the columns have different numbers of rows)
    refrind_data = pd.read_csv(f"{refrind_dir}/{aggregate}.refrind", names=refrind_col_names, header=None, sep='\s+')
    
    ax.plot(refrind_data['wavelength'].values, refrind_data['n'].values, color = 'm', label= '$n$', linewidth= 3) # plot n
    ax.plot(refrind_data['wavelength'].values, refrind_data['k'].values, color = 'm', label= '$k$', linewidth= 3, linestyle='dashed') # plot k

    # make log-log plot
    ax.set_yscale('log')
    #ax.set_xscale('log')

    ax.set_xlim(min_wavelength, max_wavelength)
    
    # set axes titles
    ax.set_xlabel(r'$\lambda$ ($\mu$m)', fontsize=20)
    ax.set_ylabel(r'Refractive Index', fontsize=20)

    # set tick font size
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)

    plt.legend(prop={'size': 14})

    return fig
        



def plot_wavelength_radius_grid(radii_path, aggregate, d_f_list, mieff_dir):

    '''

    Code to plot the wavelength-radius grid that was used the create the .mieff files, and then overplot the actual radii
    for each pressure layer for each fractal dimension, so that we can check that the radii we are using actually have
    unique optical properties that were calculated at a reasonable resolution.

    INPUTS:

    -   radii_path: (string)
            filepath to directory that contains .csv files for each aggregate (+ sphere) models, each with 4-columns with outputs from virga (pressure in bars, radii in um, number density in cm^-2, MMR) 
    -   aggregate: (string)
            chemical species used in model 
    -   d_f_list: (array)
            list of fractal dimensions analysed
    -   mieff_dir: (string)
            filepath to directory where .mieff files are stored

    '''

    # choose a colormap (based on the number of shapes to analyse)
    colors = plt.cm.winter_r(np.linspace(0,1,len(d_f_list)))


    # -------- SPHERES -------- 

    # read in the dataset for spherical particles
    mieff_col_names= ['A', 'B', 'C', 'D'] # assign arbitrary column names for reading in the data (all the columns have different numbers of rows)
    mieff_data = pd.read_csv(f"{mieff_dir}/{aggregate}.mieff", names=mieff_col_names, header=None, sep='\s+')

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
    radii_col_names= ['pressure', 'radius', 'column_density', 'mmr'] # assign column names for new dataset
    raw_radius_data = pd.read_csv(f"{radii_path}/aggregate_radii_spheres.txt", names=radii_col_names, header=None) # read in whole dataset
    radius_data_spheres = raw_radius_data[(raw_radius_data != 0.0).all(1)] # find the rows of data where we have clouds (where column_density was not equal to 0.0)



    # -------- AGGREGATES -------- 

    radius_data_aggregates_list=[] # initialise list to hold all of the 'pressure', 'radius' and 'column_density' values (in the cloud only) each fractal dimension

    for j in range(len(d_f_list)):

        # ---- SAFETY CHECK -----

        # first, check that we used the same wavelength-radius grid as the spherical model for each fractal dimension (we don't need to use the data from these .mieff files again otherwise -- this is just a safety check)
        mieff_data = pd.read_csv(f"{mieff_dir}/{aggregate}_aggregates_Df_{d_f_list[j]:.6f}.mieff", names=mieff_col_names, header=None, sep='\s+') # load data

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


        # load the cloud data for the particle radius and column_density values for each aggregate
        raw_radius_data = pd.read_csv(f"{radii_path}/aggregate_radii_{d_f_list[j]:.1f}.txt", names=radii_col_names, header=None) # read in whole dataset
        reduced_data= raw_radius_data[(raw_radius_data != 0.0).all(1)] # find the rows of data where we have clouds (where column_density was not equal to 0.0)
        radius_data_aggregates_list.append(reduced_data.reset_index(drop=True)) # reset the index for the cloud rows, and add this dataframe to a list (one for each fractal dimension)


    # check whether any of the particle radii are outside of the wavelength-radius grid
    warning_flag=0
    for j in range(len(radius_data_aggregates_list)):
        for i in range(len(radius_data_aggregates_list[j])):
            if (radius_data_aggregates_list[j]['radius'][i] < list_of_sphere_radii[0]) or (radius_data_aggregates_list[j]['radius'][i] > list_of_sphere_radii[num_radii-1]): # if the radius for one of the fractal dimensions is less than the minimum radius or larger than the maximum for the grid (which we have already checked is the same as the one in the spherical version)...
                print(f"WARNING: {d_f_list[j]} has a radius off the grid ({radius_data_aggregates_list[j]['radius'][i]:.6f} um, at pressure {radius_data_aggregates_list[j]['pressure'][i]:.6f} bar).\n") #...print a warning and highlight which fractal dimension and radius caused it
                warning_flag=1
    
    if(warning_flag==0):
        print("\nGood news - all aggregates are within the grid!")

    # Create scatterplot figure
    fig,ax = plt.subplots()
    fig.set_facecolor('w') # make area around plot in case user is in dark mode

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
    ax2.scatter(radius_data_spheres['radius'], radius_data_spheres['column_density'], color = 'm', label= f'Spheres', s=marker_size)

    # AGGREGATES: add scatter plot of radii and 'column_density' for each pressure layer of cloud
    for j in range(len(d_f_list)):
        ax2.scatter(radius_data_aggregates_list[j]['radius'], radius_data_aggregates_list[j]['column_density'], color = colors[j], label= f'{d_f_list[j]:.1f}', s=marker_size)

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

    return fig
    

def plot_pressure_vs_number_density(radii_path, d_f_list, min_pressure=None, max_pressure=None):

    '''

    Code to plot the particle densities at each radius and at each pressure layer for aggregates of each fractal dimension.
    This basically allows you to see the relative numbers and radii of the particles that form in each layer, for each fractal dimension.

    INPUTS:

    -   radii_path: (string)
            filepath to directory that contains .csv files for each aggregate (+ sphere) models, each with 4-columns with outputs from virga (pressure in bars, radii in um, number density in cm^-2, MMR) 
    -   d_f_list: (array)
            list of fractal dimensions analysed
    -   min_pressure: (float)
            minimum pressure (bars) to plot on y-axis
    -   max_pressure: (float)
            maximum pressure (bars) to plot on y-axis

    '''

    
    # choose a colormap (based on the number of shapes to analyse)
    colors = plt.cm.winter_r(np.linspace(0,1,len(d_f_list)))

    # set maximum and minimum marker sizes (on the screen) - these are arbitrary, and we can scale them to whatever looks good visually
    max_marker_size = 1000
    min_marker_size = 1

    # initialise a list to hold all of the dataframes
    radius_data_list=[]

    # assign column names for reading in data
    radii_col_names = ['pressure', 'radius', 'column_density', 'mmr']

    # read in all datasets
    for i in range(len(d_f_list)):
        radius_data_list.append(pd.read_csv(f"{radii_path}/aggregate_radii_{d_f_list[i]:.1f}.txt", header=None, names=radii_col_names))
        
    # go through each fractal dimension [i] and calculate the minimum and maximum radius for each one. Record only the global min/max.
    global_max_r=0 # intialise a running maximum radius
    global_min_r=1000000 # intialise a running minimum radius
    for i in range(len(d_f_list)):
        this_min_r = radius_data_list[i]['radius'][(radius_data_list[i]['radius'] >= 0.0000001)].min() # finds the minimum radius for each fractal dimension (but as long as it is larger than 0)
        this_max_r = radius_data_list[i]['radius'].max() # finds the maximum radius for each fractal dimension
        #print(f'For {d_f_list[i]:.1f}: Min = {this_min_r}, Max = {this_max_r}')

        if(this_min_r<global_min_r):
            global_min_r = this_min_r # if the new lowest, set new min record

        if(this_max_r>global_max_r):
            global_max_r = this_max_r # if the new highest, set new max record



    # repeat for spheres

    # read in spherical datasets
    radius_data_spheres= pd.read_csv(f"{radii_path}/aggregate_radii_spheres.txt", header=None, names=radii_col_names)
        
    # calculate the minimum and maximum radius. Record only the global min/max if less/more than for any of the fractal dimension values
    this_min_r = radius_data_spheres['radius'][(radius_data_spheres['radius'] >= 0.0000001)].min() # finds the minimum radius for spheres (but as long as it is larger than 0)
    this_max_r = radius_data_spheres['radius'].max() # finds the maximum radius for spheres
    #print(f'For spheres: Min = {this_min_r}, Max = {this_max_r}')
    if(this_min_r<global_min_r):
        global_min_r = this_min_r # if the new lowest, set new min record

    if(this_max_r>global_max_r):
        global_max_r = this_max_r # if the new highest, set new max record


    #print(f'\nFINAL RESULTS: Min = {global_min_r}, Max = {global_max_r}')

    # calculate min and max r2 values because these relate linearly to the marker sizes, which are areas
    min_r_squared = global_min_r**2
    max_r_squared = global_max_r**2

    # scale the min/max radii so that they correspond with the min/max marker sizes that we want to show on screen
    marker_size_per_square_radii = (max_marker_size - min_marker_size) / (max_r_squared - min_r_squared)

    # Create scatterplot figure
    fig,ax = plt.subplots()
    fig.set_facecolor('w') # make area around plot in case user is in dark mode

    # plot values for each fractal dimension d_f
    for i in range(len(d_f_list)):
        # create a list of marker sizes for the radii in this fractal dimension
        square_radii = radius_data_list[i]['radius']**2 # first, simply square the radii
        marker_sizes = min_marker_size + marker_size_per_square_radii * (square_radii - min_r_squared) # use derived equation to find the marker area that would give accurately scaled radii on screen
        # make scatter plot 
        legend_handle=ax.scatter(radius_data_list[i]['column_density'], radius_data_list[i]['pressure'], color = colors[i], s=marker_sizes, label= f'{d_f_list[i]:.1f}') # legend handle is just there to make the legend markers all the same size (see it's use below)

    # plot values for spheres
    # create a list of marker sizes for the radii for spheres
    square_radii = radius_data_spheres['radius']**2 # first, simply square the radii
    marker_sizes = min_marker_size + marker_size_per_square_radii * (square_radii - min_r_squared) # use derived equation to find the marker area that would give accurately scaled radii on screen

    ax.scatter(radius_data_spheres['column_density'], radius_data_spheres['pressure'], color = 'm', s=marker_sizes, label= f'Spheres')

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
    ax.legend(reversed(handles), reversed(labels), title='$d_f$', title_fontsize=12, fontsize=12, loc='lower right')
    marker_size = 36
    def update_prop(handle, orig):
        handle.update_from(orig)
        handle.set_sizes([marker_size])
    plt.legend(handler_map={type(legend_handle): HandlerPathCollection(update_func=update_prop)})

    return fig

def plot_pressure_vs_radius(radii_path, d_f_list, min_pressure=None, max_pressure=None):
    
    '''
    
    Code to plot the particle radius at each pressure layer for aggregates of each fractal dimension.
    Designed originally for diagnostic tests as part for VIRGA development by MGL.

    INPUTS:

    -   radii_path: (string)
            filepath to directory that contains .csv files for each aggregate (+ sphere) models, each with 4-columns with outputs from virga (pressure in bars, radii in um, number density in cm^-2, MMR) 
    -   d_f_list: (array)
            list of fractal dimensions analysed
    -   min_pressure: (float)
            minimum pressure (bars) to plot on y-axis
    -   max_pressure: (float)
            maximum pressure (bars) to plot on y-axis

    '''

    # choose a colormap (based on the number of shapes to analyse)
    #colors = plt.cm.winter_r(np.linspace(0,1,len(d_f_list)))
    colors = plt.cm.winter_r(np.linspace(0,1,len(d_f_list)))

    # initialise a list to hold all of the dataframes
    radius_data_list=[]

    # assign column names for reading in data
    radii_col_names = ['pressure', 'radius', 'column_density', 'mmr']

    # read in all fractal datasets
    for i in range(len(d_f_list)):
        radius_data_list.append(pd.read_csv(f"{radii_path}/aggregate_radii_{d_f_list[i]:.1f}.txt", header=None, names=radii_col_names))

    # read in spherical datasets
    radius_data_spheres= pd.read_csv(f"{radii_path}/aggregate_radii_spheres.txt", header=None, names=radii_col_names)

    # Create scatterplot figure
    fig,ax = plt.subplots()
    fig.set_facecolor('w') # make area around plot in case user is in dark mode

    # plot values for each fractal dimension d_f
    for i in range(len(d_f_list)):
        ax.plot(radius_data_list[i]['radius'].values, radius_data_list[i]['pressure'].values, color = colors[i], label= f'{d_f_list[i]:.1f}')

    # plot values for spheres
    ax.plot(radius_data_spheres['radius'].values, radius_data_spheres['pressure'].values, color = 'm', label= f'Spheres')

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

    plt.legend()

    return fig

def convert_rg_to_R_gyr(radii_path, Df=None, N_mon=None, r_mon=None, k0=0):

    '''

    Function to convert compact mean particle radii (r_g) for aggregates into their equivalent R_gyr values (in the
    same way that they are used in the v_fall equation), using the 'growth model' outlined in Fig 2. Moran & Lodge (2025).
    This function exists because calculating R_gyr is not trivial, because particles can change their number of monomers
    and monomer radii at different radii sizes.
    
    User should input exactly the same aggregate variables (with same units) as when they call VIRGA. Most importantly:
      
        - EITHER N_mon OR r_mon should be provided, (and only one of them).
        - If r_mon is provided it needs to be in cm.

    INPUTS:

    -   radii_path: (string)
            filepath to directory that contains .csv files for each aggregate (+ sphere) models, each with 4-columns with outputs from virga (pressure in bars, radii in um, number density in cm^-2, MMR) 
    
    -   Df: (float)
            fractal dimension of aggregates
   
    -   N_mon: (int - optional) - EITHER THIS OR r_mon SHOULD BE INCLUDED
            Number of monomers
    
    -   r_mon: (float - optional) - EITHER THIS OR N_mon SHOULD BE INCLUDED
            monomer radius (in cm) IMPORTANT: input in cm like with regular VIRGA calls
   
    -   k0: (float - optional)
            fractal prefactor, either prescribed by user or calculated using Eq. 14 + 15 of Moran & Lodge (2025)


    RETURNS:

    - compact_r_g_values: (in um) the compact equivalent spherical radii of the aggregate at each pressure layer, matching the pressure layers in the radius file and virga model
    - r_gyr_grid: (in um) the radius of gyration of the aggregate at each pressure layer, matching the pressure layers in the radius file and virga model

    '''

    # assign column names for reading in data
    radii_col_names = ['pressure', 'radius', 'column_density', 'mmr']

    # read in fractal datasets
    fractal_dataset=pd.read_csv(f"{radii_path}/aggregate_radii_{Df:.1f}.txt", header=None, names=radii_col_names)

    compact_r_g_values = fractal_dataset['radius'].values # retrieve the compact spherical radii of the aggregates (r_g, in um) from the file for this fractal dimension

    nrad = len(compact_r_g_values) # find number of radii analysed
    r_gyr_grid = np.zeros(nrad) # initialise array of the same length to store the R_gyr values

    # determine whether or not we need to calculate N_mon for each radius
    if N_mon is not None:
        N_mon_prescribed=1 # N_mon is a fixed value, prescribed by the user
        original_N_mon = N_mon # record the original set value, in case we need to reduce it temporarily for the smallest particles in the grid
    else:
        N_mon_prescribed=0 # N_mon needs to be calculated for each new radius
        original_N_mon = 0 # original number of monomers not set

    # determine whether or not we need to calculate k0 for each radius
    if k0>0:
        k_0_prescribed=1 # k0 is prescribed by the user
    else:
        k_0_prescribed=0 # k0 needs to be calculated for each new radius

    for irad in range(nrad):

        rr = compact_r_g_values[irad] # select compact spherical equivalent radius (in um)
        
        # calculate N_mon and r_mon for this particular radius
        rr_cm = rr*1e-4 # convert rr (from um to cm) before inputting to function
        N_mon, r_mon = find_N_mon_and_r_mon(N_mon_prescribed, rr_cm, original_N_mon, r_mon) # function works with everything in cm
        r_mon_micron = r_mon * 1e4 # convert r_mon back from cm into um

        # if radius is smaller than r_mon, use single spheres
        if rr<= r_mon_micron:
            r_gyr_grid[irad] = rr

        # otherwise, calculate and use the radius of gyration
        else:
            # if fractal prefactor k0 is left unprescribed by user, calculate it here
            if(k_0_prescribed==0):
                k0 = calculate_k0(N_mon, Df)

            R_gyr = (N_mon/k0)**(1/Df) * r_mon_micron # calculate radius of gyration (in um)
            r_gyr_grid[irad] = R_gyr # set the new grid as the R_gyr values (converted from the spherical compact-particle equivalent radii)

    return compact_r_g_values, r_gyr_grid # return both the compact r_g values and the equivalent R_gyr values in um