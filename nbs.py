# In[1]: Packages

#Import packages
import bct
import pandas as pd
from bids import BIDSLayout
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
import plotje
# In[2]: Data acquisition

# Create directories of your choice
data_dir        = './bids/'
conn_dir        = './bids/derivatives/connectivity/'
file_dir        = './datasets/'
fig_dir         = './figures/analysis/nbs/100/'
analysis_dir    = './analysis/nbs_results/100/'

# Grab bids structure
layout = BIDSLayout(data_dir)
# Add connectivity directory
layout.add_derivatives(conn_dir)

# Read and generate dataset of interest
dataset = pd.read_csv(file_dir + 'b2b_dataset_20230626.csv', sep='\t')
data = {'id'        : dataset['mr_brain'], 
        'group'     : dataset['Group']}

# Read MR-files
files = layout.derivatives['connectivity'].get(extension='.tsv', space = 'MNI152NLin2009cAsym', atlas = 'HOSubCortSchaefer2018100Parcels7Networks')
conn_list = list(np.zeros(len(files))) 
control         = {}
surgery         = {}
conservative    = {}
for file in files:
    entities    = dict(file.entities) 
    file_df     = pd.read_csv(file.path, sep='\t', index_col=[0])
    idx         = np.where(data['id'] == int(entities['subject']))
    idx         = int(idx[0])
    group       = data['group'][idx]
    
    #Group allocation
    if group == 'Surgery':
        surgery[entities['subject']]        = file_df
    elif group =='Control':
        control[entities['subject']]        = file_df
    elif group =='Conservative':
        conservative[entities['subject']]   = file_df
             

#Generate variables of restructured data
surg_nodes                  = np.zeros((len(surgery),       len(file_df),len(file_df)))
cons_nodes                  = np.zeros((len(conservative),  len(file_df),len(file_df)))
control_nodes               = np.zeros((len(control),       len(file_df),len(file_df)))

surgery_transformed         = {}
conservative_transformed    = {}
control_transformed         = {}

#Fischer transformation of data
for index,data in enumerate(surgery):
    surg_arr                            = np.array(surgery[data])
    surg_arr[surg_arr==1]               = np.nan
    surg_arr                            = np.arctanh(surg_arr)    
    surg_nodes[index,:,:]               = surg_arr
    surgery_transformed[str(data)]      = surg_arr
    
for index, data in enumerate(control):
    control_arr                         = np.array(control[data])
    control_arr[control_arr==1]         = np.nan
    control_arr                         = np.arctanh(control_arr)    
    control_nodes[index,:,:]            = control_arr
    control_transformed[str(data)]      = control_arr

for index, data in enumerate(conservative):
    cons_arr                            = np.array(conservative[data])
    cons_arr[cons_arr==1]               = np.nan
    cons_arr                            = np.arctanh(cons_arr)    
    cons_nodes[index,:,:]               = cons_arr
    conservative_transformed[str(data)] = cons_arr

# In[3]: NBS


#Transpose data for NBS compatibility
nbs_control_nodes       = control_nodes.transpose()
nbs_surgery_nodes       = surg_nodes.transpose()
nbs_conservative_nodes  = cons_nodes.transpose()

#Number of permutations
kx = 10000
#T-value for NBS 
t = 3

#NBS
nbs_surgcont_t3_p, nbs_surgcont_adj_t3, nbs_surgcont_null_t3 = bct.nbs_bct(nbs_surgery_nodes,nbs_control_nodes,     t,k=kx)
nbs_surgcons_t3_p, nbs_surgcons_adj_t3, nbs_surgcons_null_t3 = bct.nbs_bct(nbs_surgery_nodes,nbs_conservative_nodes,t,k=kx)
nbs_conscont_t3_p, nbs_conscont_adj_t3, nbs_conscont_null_t3 = bct.nbs_bct(nbs_conservative_nodes,nbs_control_nodes,t,k=kx)

#Save relevant analyses in corresponding folders
pd.DataFrame(nbs_surgcont_t3_p).to_csv(analysis_dir+'surgery_control/surgery_control_p_values_t3.csv')
pd.DataFrame(nbs_surgcont_adj_t3).to_csv(analysis_dir+'surgery_control/surgery_control_adjacency_matrix_t3.csv')
pd.DataFrame(nbs_surgcont_null_t3).to_csv(analysis_dir+'/surgery_control/surgery_control_null_t3.csv')

pd.DataFrame(nbs_surgcons_t3_p).to_csv(analysis_dir+'surgery_conservative/surgery_conservative_p_values_t3.csv')
pd.DataFrame(nbs_surgcons_adj_t3).to_csv(analysis_dir+'surgery_conservative/surgery_conservative_adjacency_matrix_t3.csv')
pd.DataFrame(nbs_surgcons_null_t3).to_csv(analysis_dir+'/surgery_conservative/surgery_conservative_null_t3.csv')

pd.DataFrame(nbs_conscont_t3_p).to_csv(analysis_dir+'conservative_control/conservative_control_p_values_t3.csv')
pd.DataFrame(nbs_conscont_adj_t3).to_csv(analysis_dir+'conservative_control/conservative_control_adjacency_matrix_t3.csv')
pd.DataFrame(nbs_conscont_null_t3).to_csv(analysis_dir+'/conservative_control/conservative_null_t3.csv')



#Verify significance, message if not significant
sig_idx_surgcont_t3_p = np.where(nbs_surgcont_t3_p < 0.05)
try:
    sig_idx_surgcont_t3_p = int(sig_idx_surgcont_t3_p[0])+1 # +1 because idx 0 would not match the value 0 (no adj) in the adjacency matrix
except:
    print('No significant NBS could be found between Surgery and Controls (p >= 0.05)')

sig_idx_surgcons_t3_p = np.where(nbs_surgcons_t3_p < 0.05)
try: 
    sig_idx_surgcons_t3_p = int(sig_idx_surgcons_t3_p[0])+1 # +1 because idx 0 would not match the value 0 (no adj) in the adjacency matrix
except:
    print('No significant NBS could be found between Surgery and Conservative (p >= 0.05)') 

sig_idx_conscont_t3_p = np.where(nbs_conscont_t3_p < 0.05)
try:
    sig_idx_conscont_t3_p = int(sig_idx_conscont_t3_p[0])+1 # +1 because idx 0 would not match the value 0 (no adj) in the adjacency matrix
except:
    print('No significant NBS could be found between Controls and Conservative (p >= 0.05)')


# In[4]: Illustration tools

#Create indices for each corresponding Network (Yeo7 + HOSPA + Cerebellum)

vis_index           = np.hstack([np.arange(0,9),     np.arange(50,58)])
sommot_index        = np.hstack([np.arange(9,15),   np.arange(58,66)])
dorsatt_index       = np.hstack([np.arange(15,23),   np.arange(66,73)]) 
salvent_index       = np.hstack([np.arange(23,30),  np.arange(73,78)])
limbic_index        = np.hstack([np.arange(30,33), np.arange(78,80)])
cont_index          = np.hstack([np.arange(33,37), np.arange(80,89)])
default_index       = np.hstack([np.arange(37,50), np.arange(89,100)])
subcort_index       = np.hstack([np.arange(100,112)])
cerebellum_index    = np.hstack([np.arange(112,123)])

network_indices     =   {'vis'      : vis_index, 
                         'sommot'   : sommot_index, 
                         'dorsatt'  : dorsatt_index, 
                         'salvent'  : salvent_index, 
                         'limbic'   : limbic_index, 
                         'cont'     : cont_index, 
                         'default'  : default_index,
                         'subcort'  : subcort_index, 
                         'cereb'    : cerebellum_index}

#Visualization setup on parcel-level

network_color={}
for i in vis_index:    
    network_color[i] = 'purple'
for i in sommot_index:
    network_color[i] = 'blue'
for i in dorsatt_index:
    network_color[i] = 'green'
for i in salvent_index:
    network_color[i] = 'violet'
for i in limbic_index:
    network_color[i] = 'gray'
for i in cont_index: 
    network_color[i] = 'orange'
for i in default_index:
    network_color[i] = 'red'
for i in subcort_index:
    network_color[i] = 'brown'
for i in cerebellum_index:
    network_color[i] = 'olive' 
network_color_full = []
for i in sorted(network_color):
    network_color_full.append(network_color[i])

#Create specific legend for used networks

legend_elements = [Patch(facecolor = 'purple',  edgecolor = None, label = 'Vis'),
                   Patch(facecolor = 'blue',    edgecolor = None, label = 'S.M'),
                   Patch(facecolor = 'green',   edgecolor = None, label = 'D.A'),
                   Patch(facecolor = 'violet',  edgecolor = None, label = 'V.A'),
                   Patch(facecolor = 'gray',    edgecolor = None, label = 'Lim'),
                   Patch(facecolor = 'orange',  edgecolor = None, label = 'F.P'),
                   Patch(facecolor = 'red',     edgecolor = None, label = 'D.M'),
                   Patch(facecolor = 'brown',   edgecolor = None, label = 'Sub'),
                   Patch(facecolor = 'olive',   edgecolor = None, label = 'Cer')]

#Design custom line to illustrate location of used networks

line_parcel_x = np.zeros(len(network_color_full)+1)
line_parcel_y = np.zeros(len(network_color_full)+1)
for i in range(len(line_parcel_x)):
    line_parcel_x[i] = i-0.5
x_points_parcel_x = np.array([line_parcel_x, line_parcel_y]).T.reshape(-1, 1, 2)
segments_parcel_x = np.concatenate([x_points_parcel_x[:-1], x_points_parcel_x[1:]], axis=1)

counter = 0
line_parcel_x = np.zeros(len(network_color_full)+1)
line_parcel_y = np.zeros(len(network_color_full)+1)
for i in (range(len(line_parcel_y))):
    line_parcel_y[counter] = i-0.5
    counter = counter+1
y_points_parcel_y = np.array([line_parcel_x, line_parcel_y]).T.reshape(-1, 1, 2)
segments_parcel_y = np.concatenate([y_points_parcel_y[:-1], y_points_parcel_y[1:]], axis=1)

# In[5]: Illustrate potential findings

#Plot significant findings

# Surgery vs Conservative
try:
                    # Remove the upper half of the correlation matrix
    mask            = np.triu(nbs_surgcons_adj_t3)
    masked_matrix   = np.ma.array(nbs_surgcons_adj_t3, mask=mask)

    fig, ax         = plt.subplots(1,figsize=(10,5))
    im              = ax.imshow(masked_matrix== sig_idx_surgcons_t3_p, cmap='Greys', vmin = 0, vmax = 1)
    
    # Replace axes with network color codes 
    ax.tick_params  (top=False,
                     bottom=False,
                     left=False,
                     right=False,
                     labelleft=False,
                     labelbottom=False)

    lc_parcel_x    = LineCollection(segments_parcel_x,colors=network_color_full, linewidth=1.4,
                                    transform=ax.get_xaxis_transform(), clip_on=False )
    ax.add_collection(lc_parcel_x)
    
    lc_parcel_y = LineCollection(segments_parcel_y, colors=network_color_full, linewidth=1.4,
                                 transform=ax.get_yaxis_transform(), clip_on=False)
    ax.add_collection(lc_parcel_y)
    
    # Add custom legend
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(-0.4,0.75))
    
    # Aesthetics
    plotje.styler(ax, 
                  title = 'Significant NBS for Surgery vs. Conservative (k = '+str(kx) +')', 
                  leftaxis=None, 
                  bottomaxis = None, 
                  xlabel='Parcels', 
                  ylabel = 'Parcels')

    # Save figure    
    plt.savefig(fig_dir +'masked_surgery_vs_conservative_t3.png', dpi= 300)
except:
    print()
    
  
# Repeat for Surgery vs Controls    
try:
    mask            = np.triu(nbs_surgcont_adj_t3)
    masked_matrix   = np.ma.array(nbs_surgcont_adj_t3, mask=mask)
    fig, ax         = plt.subplots(1,figsize=(10,5))
    im              = ax.imshow(masked_matrix== sig_idx_surgcont_t3_p, cmap='Greys', vmin = 0, vmax = 1)
    
    ax.tick_params  (top=False,
                     bottom=False,
                     left=False,
                     right=False,
                     labelleft=False,
                     labelbottom=False)

    lc_parcel_x    = LineCollection(segments_parcel_x,colors=network_color_full, linewidth=1.4,
                                    transform=ax.get_xaxis_transform(), clip_on=False )
    ax.add_collection(lc_parcel_x)
    
    lc_parcel_y = LineCollection(segments_parcel_y, colors=network_color_full, linewidth=1.4,
                                 transform=ax.get_yaxis_transform(), clip_on=False)
    ax.add_collection(lc_parcel_y)
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(-0.4,0.75))
    
    plotje.styler(ax,
                  title = 'Significant NBS for Surgery vs. Control (k = '+str(kx) +')',  
                  leftaxis=None, 
                  bottomaxis = None, 
                  xlabel='Parcels', 
                  ylabel = 'Parcels')
    
    plt.savefig(fig_dir +'masked_surgery_vs_control_t3.png', dpi= 300)
except:
    print()

# Repeat for Conservative vs Controls 
try:
    mask            = np.triu(nbs_conscont_adj_t3)
    masked_matrix   = np.ma.array(nbs_conscont_adj_t3, mask=mask)
    fig, ax         = plt.subplots(1,figsize=(10,5))
    im              = ax.imshow(masked_matrix== sig_idx_conscont_t3_p, cmap='Greys', vmin = 0, vmax = 1)
    
    ax.tick_params  (top=False,
                     bottom=False,
                     left=False,
                     right=False,
                     labelleft=False,
                     labelbottom=False)

    lc_parcel_x    = LineCollection(segments_parcel_x,colors=network_color_full, linewidth=1.4,
                                    transform=ax.get_xaxis_transform(), clip_on=False )
    ax.add_collection(lc_parcel_x)
    
    lc_parcel_y = LineCollection(segments_parcel_y, colors=network_color_full, linewidth=1.4,
                                 transform=ax.get_yaxis_transform(), clip_on=False)
    ax.add_collection(lc_parcel_y)
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(-0.4,0.75))
    
    plotje.styler(ax,
                  title = 'Significant NBS for Conservative vs Control (k = '+str(kx) +')', 
                  leftaxis=None, 
                  bottomaxis = None, 
                  xlabel='Parcels', 
                  ylabel = 'Parcels')
    
    plt.savefig(fig_dir +'masked_conservative_vs_control_t3.png', dpi= 300)
except:
    print()
