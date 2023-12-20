# In[1]: Packages
import pandas as pd
from bids import BIDSLayout
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
import plotje
import scipy
from multipy.fdr import lsu
#In[2]: Data acquisition

#Specify used paths
data_dir        = './bids/'
conn_dir        = './bids/derivatives/connectivity/'
file_dir        = './datasets/'
fig_dir         = './figures/publication/'
analysis_dir    = './analysis/nbs_results/100/'

# Grab bids structure
layout = BIDSLayout(data_dir)
# Add connectivity directory
layout.add_derivatives(conn_dir)

# Read dataset of interest
dataset = pd.read_csv(file_dir + 'b2b_dataset_20230626.csv', sep='\t')

#Load significant findings from NBS analysis (see NBS script)
nbs_surgcons_adj_t3 = np.array(pd.read_csv(analysis_dir + 'surgery_conservative/surgery_conservative_adjacency_matrix_t3.csv',index_col=0))
nbs_surgcons_t3_p = pd.read_csv(analysis_dir + 'surgery_conservative/surgery_conservative_p_values_t3.csv',index_col=0)

sig_idx_surgcons_t3_p = np.where(nbs_surgcons_t3_p < 0.05)
sig_idx_surgcons_t3_p = int(sig_idx_surgcons_t3_p[0])+1 # +1 because idx 0 would not match the value 0 (no adj) in the adjacency matrix
    
# Read MR-files
files = layout.derivatives['connectivity'].get(extension='.tsv', space = 'MNI152NLin2009cAsym', atlas = 'HOSubCortSchaefer2018100Parcels7Networks')
conn_list = list(np.zeros(len(files))) 
control_df      = []
surg_df         = []
cons_df         = []
surg_names      = []
control_names   = []
cons_names      = []

for file in files:
    entities = dict(file.entities) 
    file_df = pd.read_csv(file.path, sep='\t', index_col=[0])
    idx = np.where(dataset['mr_brain'] == int(entities['subject']))
    idx = int(idx[0]) # <-- Make sure index is an int
    group = dataset['Group'][idx]
       
    if group == 'Surgery':
        surg_df.append(file_df)
        surg_names.append(entities['subject'])
    elif group =='Control':
        control_df.append(file_df)
        control_names.append(entities['subject'])
    elif group =='Conservative':
        cons_df.append(file_df)
        cons_names.append(entities['subject'])
     
#Fischer transform data             
surg_nodes      = np.zeros((len(surg_df),   len(file_df),len(file_df)))
cons_nodes      = np.zeros((len(cons_df),   len(file_df),len(file_df)))
control_nodes   = np.zeros((len(control_df),len(file_df),len(file_df)))

for i in range(len(surg_df)):
    surg_arr                    = np.array(surg_df[i])
    surg_arr[surg_arr==1]       = np.nan
    surg_arr                    = np.arctanh(surg_arr)    
    surg_nodes[i,:,:]           = surg_arr
    
for i in range(len(control_df)):
    control_arr                 = np.array(control_df[i])
    control_arr[control_arr==1] = np.nan
    control_arr                 = np.arctanh(control_arr)    
    control_nodes[i,:,:]        = control_arr

for i in range(len(cons_df)):
    cons_arr                    = np.array(cons_df[i])
    cons_arr[cons_arr==1]       = np.nan
    cons_arr                    = np.arctanh(cons_arr)    
    cons_nodes[i,:,:]           = cons_arr


#Generate mean Matrices for each group

mean_nodes_surg                 =   np.mean(surg_nodes,     axis = (0))
mean_nodes_control              =   np.mean(control_nodes,  axis = (0))
mean_nodes_cons                 =   np.mean(cons_nodes,     axis = (0))

delta_nodes_surgcont            =   mean_nodes_surg - mean_nodes_control
delta_nodes_surgcons            =   mean_nodes_surg - mean_nodes_cons
delta_nodes_conscont            =   mean_nodes_cons - mean_nodes_control


#In[3]: 

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

#Sort according to networks
netsort = np.zeros(len(surg_arr))
for i, key_i in enumerate(network_indices):
    netsort[network_indices[key_i]] = i
sortindex = np.argsort(netsort)



colors =            [ 'purple',
                      'blue',
                      'green',
                      'violet',
                      'gray',
                      'orange',
                      'red',
                      'brown',
                      'olive'] 
  
labels =            [ 'Visual',
                      'Somatomotor',
                      'Dorsal attention',
                      'Ventral attention',
                      'Limbic',
                      'Frontoparietal',
                      'Default mode',
                      'Subcortical',
                      'Cerebellum']

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

sorted_network_colors = []
for i in sortindex:
    if netsort[i] == 0:
        sorted_network_colors.append('purple')
    elif netsort[i] == 1:
        sorted_network_colors.append('blue')
    elif netsort[i] == 2: 
        sorted_network_colors.append('green')
    elif netsort[i] == 3:
        sorted_network_colors.append('violet')
    elif netsort[i] == 4: 
        sorted_network_colors.append('gray')
    elif netsort[i] == 5:
        sorted_network_colors.append('orange')
    elif netsort[i] == 6: 
        sorted_network_colors.append('red')
    elif netsort[i] == 7: 
        sorted_network_colors.append('brown')
    elif netsort[i] == 8:
        sorted_network_colors.append('olive')
 
#Visualization setup on network-level
network_color_net = [ 'purple',
                      'blue',
                      'green',
                      'violet',
                      'gray',
                      'orange',
                      'red',
                      'brown',
                      'olive']   
ticklabels_net =    [ 'Vis',
                      'S.M',
                      'D.A',
                      'V.A',
                      'Lim',
                      'F.P',
                      'D.M',
                      'Sub',
                      'Cer']
 
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

#Generate color-coded line to illustrate location of networks within connectivity matrix
line_parcel_x = np.zeros(len(sorted_network_colors)+1)
line_parcel_y = np.zeros(len(sorted_network_colors)+1)
for i in range(len(line_parcel_x)):
    line_parcel_x[i] = i-0.5
x_points_parcel_x = np.array([line_parcel_x, line_parcel_y]).T.reshape(-1, 1, 2)
segments_parcel_x = np.concatenate([x_points_parcel_x[:-1], x_points_parcel_x[1:]], axis=1)

counter = 0
line_parcel_x = np.zeros(len(sorted_network_colors)+1)
line_parcel_y = np.zeros(len(sorted_network_colors)+1)
for i in (range(len(line_parcel_y))):
    line_parcel_y[counter] = i-0.5
    counter = counter+1
y_points_parcel_y = np.array([line_parcel_x, line_parcel_y]).T.reshape(-1, 1, 2)
segments_parcel_y = np.concatenate([y_points_parcel_y[:-1], y_points_parcel_y[1:]], axis=1)

#In[4]: Illustrations for Figure 2

#Generate parcel-level connectivity matrices per group (Surgery)
mask            = np.triu(mean_nodes_surg[sortindex,:][:,sortindex])
masked_matrix   = np.ma.array(mean_nodes_surg[sortindex,:][:,sortindex],mask=mask)
fig, ax         = plt.subplots(1,figsize=(6,5))

im              = ax.imshow(masked_matrix, cmap='RdBu_r', vmin = -1, vmax=1)
ax.tick_params(top          = False,
               bottom       = False,
               left         = False,
               right        = False,
               labelleft    = False,
               labelbottom  = False)
lc_parcel_x     = LineCollection(segments_parcel_x,
                                 colors     = sorted_network_colors, 
                                 linewidth  = 8,
                                 transform  = ax.get_xaxis_transform(), 
                                 clip_on    = False,
                                 offsets    = (0,-5))
ax.add_collection(lc_parcel_x)
lc_parcel_y     = LineCollection(segments_parcel_y, 
                                 colors     = sorted_network_colors, 
                                 linewidth  = 8,
                                 transform  = ax.get_yaxis_transform(),
                                 clip_on    = False,
                                 offsets    = (-5,0))
ax.add_collection(lc_parcel_y)
plotje.styler(ax, colorbar  =  im, 
              colorbarlabel = 'Mean connectivity', 
              cbaroffset    =  0.4, 
              title         = 'Surgery', 
              leftaxis      =  None, 
              bottomaxis    =  None, 
              xlabel        = 'Nodes', 
              ylabel        = 'Nodes')
plt.savefig(fig_dir +'surgery_conn_parcel_100.svg', dpi= 300)

#Controls
mask            = np.triu(mean_nodes_control[sortindex,:][:,sortindex])
masked_matrix   = np.ma.array(mean_nodes_control[sortindex,:][:,sortindex],mask=mask)
fig, ax         = plt.subplots(1,figsize=(6,5))

im              = ax.imshow(masked_matrix, cmap='RdBu_r', vmin = -1, vmax=1)
ax.tick_params(top          = False,
               bottom       = False,
               left         = False,
               right        = False,
               labelleft    = False,
               labelbottom  = False)
lc_parcel_x     = LineCollection(segments_parcel_x,
                                 colors     = sorted_network_colors, 
                                 linewidth  = 8,
                                 transform  = ax.get_xaxis_transform(), 
                                 clip_on    = False, 
                                 offsets    = (0,-5))
ax.add_collection(lc_parcel_x)
lc_parcel_y     = LineCollection(segments_parcel_y, 
                                 colors     = sorted_network_colors, 
                                 linewidth  = 8,
                                 transform  = ax.get_yaxis_transform(), 
                                 clip_on    = False,
                                 offsets    = (-5,0))
ax.add_collection(lc_parcel_y)
plotje.styler(ax, colorbar  =  im, 
              colorbarlabel = 'Mean connectivity', 
              cbaroffset    =  0.4, 
              title         = 'Control', 
              leftaxis      =  None, 
              bottomaxis    =  None, 
              xlabel        = 'Nodes', 
              ylabel        = 'Nodes')
plt.savefig(fig_dir +'control_conn_parcel_100.svg', dpi= 300)

#Conservative
mask            = np.triu(mean_nodes_cons[sortindex,:][:,sortindex])
masked_matrix   = np.ma.array(mean_nodes_cons[sortindex,:][:,sortindex],mask=mask)
fig, ax         = plt.subplots(1,figsize=(6,5))

im              = ax.imshow(masked_matrix, cmap='RdBu_r', vmin = -1, vmax=1)
ax.tick_params(top          = False,
               bottom       = False,
               left         = False,
               right        = False,
               labelleft    = False,
               labelbottom  = False)
lc_parcel_x     = LineCollection(segments_parcel_x,
                                 colors     = sorted_network_colors, 
                                 linewidth  = 8,
                                 transform  = ax.get_xaxis_transform(), 
                                 clip_on    = False,
                                 offsets    = (0,-5))
ax.add_collection(lc_parcel_x)
lc_parcel_y     = LineCollection(segments_parcel_y,
                                 colors     = sorted_network_colors, 
                                 linewidth  = 8,
                                 transform  = ax.get_yaxis_transform(), 
                                 clip_on    = False,
                                 offsets    = (-5,0))
ax.add_collection(lc_parcel_y)
plotje.styler(ax, colorbar  =  im, 
              colorbarlabel = 'Mean connectivity', 
              cbaroffset    =  0.4, 
              title         = 'Conservative', 
              leftaxis      =  None, 
              bottomaxis    =  None, 
              xlabel        = 'Nodes', 
              ylabel        = 'Nodes')
plt.savefig(fig_dir +'conservative_conn_parcel_100.svg', dpi= 300)

#Delta Matrix Surgery vs. Conservative
mask            = np.triu(delta_nodes_surgcons[sortindex,:][:,sortindex])
masked_matrix   = np.ma.array(delta_nodes_surgcons[sortindex,:][:,sortindex],mask=mask)
fig, ax         = plt.subplots(1,figsize=(6,5))

im              = ax.imshow(masked_matrix, cmap='RdBu_r', vmin = -0.3, vmax=0.3)
ax.tick_params(top          = False,
               bottom       = False,
               left         = False,
               right        = False,
               labelleft    = False,
               labelbottom  = False)
lc_parcel_x     = LineCollection(segments_parcel_x,
                                 colors     = sorted_network_colors, 
                                 linewidth  = 8,
                                 transform  = ax.get_xaxis_transform(), 
                                 clip_on    = False, 
                                 offsets    = (0,-5))
ax.add_collection(lc_parcel_x)
lc_parcel_y     = LineCollection(segments_parcel_y, 
                                 colors     = sorted_network_colors, 
                                 linewidth  = 8,
                                 transform  = ax.get_yaxis_transform(), 
                                 clip_on    = False,
                                 offsets    = (-5,0))
ax.add_collection(lc_parcel_y)
ax.legend(handles=legend_elements, frameon=False, bbox_to_anchor=(1,1))
plotje.styler(ax, colorbar  =  im, 
              colorbarlabel = 'Delta connectivity', 
              cbaroffset    =  0.4, 
              title         = '(Surgery - Conservative)', 
              leftaxis      =  None, 
              bottomaxis    =  None, 
              xlabel        = 'Nodes', 
              ylabel        = 'Nodes')
plt.savefig(fig_dir +'delta_surgcons_parcel_100.svg', dpi= 300)

#NBS Cluster from Surgery vs. Conservative
matrix = nbs_surgcons_adj_t3.copy()
matrix[matrix<sig_idx_surgcons_t3_p] = 0

#generate border
for i in range(123):
    matrix[i,i] = sig_idx_surgcons_t3_p/4

#Plot significant findings from NBS

# Surgery vs Conservative
mask            = np.triu(matrix[sortindex,:][:,sortindex], k=1)
masked_matrix   = np.ma.array(matrix[sortindex,:][:,sortindex], mask=mask)
fig, ax         = plt.subplots(1,figsize=(5,5))

im              = ax.imshow(masked_matrix, cmap='Greys')        
ax.tick_params  (top            = False,
                 bottom         = False,
                 left           = False,
                 right          = False,
                 labelleft      = False,
                 labelbottom    = False)
lc_parcel_x     = LineCollection(segments_parcel_x,
                                 colors     = sorted_network_colors, 
                                 linewidth  = 8,
                                 transform  = ax.get_xaxis_transform(), 
                                 clip_on    = False, 
                                 offsets    = (0,-5))
ax.add_collection(lc_parcel_x)
lc_parcel_y     = LineCollection(segments_parcel_y, 
                                 colors     = sorted_network_colors, 
                                 linewidth  = 8,
                                 transform  = ax.get_yaxis_transform(), 
                                 clip_on    = False,
                                 offsets    = (-5,0))
ax.add_collection(lc_parcel_y)
plotje.styler(ax, 
              title         = 'Significant clusters (Surgery vs. Conservative)', 
              leftaxis      =  None, 
              bottomaxis    =  None, 
              xlabel        = 'Nodes', 
              ylabel        = 'Nodes')
plt.savefig(fig_dir +'surgery_vs_conservative_t3.svg', dpi= 300)

#Highlight proportion of edges within networks in Bar graph
adj = nbs_surgcons_adj_t3.copy()
sigmap = adj==4

sum_network = np.zeros(len(network_indices.keys()))
for i, key_i in enumerate(network_indices):
    for j, key_j in enumerate(network_indices):
        if key_i == key_j:
            sum_network[i] += np.sum(sigmap[network_indices[key_i],:][:,network_indices[key_j]])/2
        else:
            sum_network[i] += np.sum(sigmap[network_indices[key_i],:][:,network_indices[key_j]])
perc_network = sum_network/sum(sum_network)*100

fig, ax = plt.subplots(1, figsize = (5,6))

ax.bar(labels,sum_network, label = labels, color = colors)
ax.set_xticklabels(ax.get_xticks(),rotation=90)
plotje.styler(ax, 
              title         = '', 
              ylabel        = 'Number of edges in significant cluster',
              xticklabels   = labels,
              rotatexticks  = True)
plt.savefig(fig_dir + 'barplot_count.svg', dpi=300)

fig, ax = plt.subplots(1, figsize = (5,6))

ax.bar(labels,perc_network, label = labels, color = colors)
ax.set_xticklabels(ax.get_xticks(),rotation=90)
plotje.styler(ax, 
              title         = '', 
              ylabel        = 'Proportion of edges in significant cluster',
              xticklabels   = labels,
              rotatexticks  = True)
plt.savefig(fig_dir + 'barplot_perc.svg', dpi=300)

#In[5]: Figure 3 - Mean Somatomotor-specific Network connectivities

mean_network_matrix_surgcont    =   np.zeros([len(network_indices), len(network_indices)])
mean_network_matrix_surgcons    =   np.zeros([len(network_indices), len(network_indices)])
mean_network_matrix_conscont    =   np.zeros([len(network_indices), len(network_indices)])

#Method for condensing parcellations into averaged network connectivities
for i, key_i in enumerate(network_indices):
    for j, key_j in enumerate(network_indices):
        mean_network_matrix_surgcont[i,j] = np.nanmean(delta_nodes_surgcont[network_indices[key_i],:][:,network_indices[key_j]])

for i, key_i in enumerate(network_indices):
    for j, key_j in enumerate(network_indices):
        mean_network_matrix_surgcons[i,j] = np.nanmean(delta_nodes_surgcons[network_indices[key_i],:][:,network_indices[key_j]])

for i, key_i in enumerate(network_indices):
    for j, key_j in enumerate(network_indices):
        mean_network_matrix_conscont[i,j] = np.nanmean(delta_nodes_conscont[network_indices[key_i],:][:,network_indices[key_j]])

sommot_matrix_surgcons = np.zeros([1,9])
sommot_matrix_surgcons[0,:] = mean_network_matrix_surgcons[1,:]

sommot_matrix_surgcont = np.zeros([1,9])
sommot_matrix_surgcont[0,:] = mean_network_matrix_surgcont[1,:]

sommot_matrix_conscont = np.zeros([1,9])
sommot_matrix_conscont[0,:] = mean_network_matrix_conscont[1,:]
 
#Generate network-specific labels and colors 
network_color_net2 = ['blue']
ticklabels_net2 =    ['S.M'] 

#Generate color-coded lines to illustrate location of networks within connectivity matrix
line_net9_x = np.zeros(len(network_color_net)+1)
line_net9_y = np.zeros(len(network_color_net)+1)
for i in range(len(line_net9_x)):
    line_net9_x[i] = i-0.5
x_points_net9_x = np.array([line_net9_x, line_net9_y]).T.reshape(-1, 1, 2)
segments_net9_x = np.concatenate([x_points_net9_x[:-1], x_points_net9_x[1:]], axis=1)

counter = 0
line_net9_x = np.zeros(len(network_color_net2)+1)
line_net9_y = np.zeros(len(network_color_net2)+1)
for i in (range(len(line_net9_y))):
    line_net9_y[counter] = i-0.5
    counter = counter+1
y_points_net9_y = np.array([line_net9_x, line_net9_y]).T.reshape(-1, 1, 2)
segments_net9_y = np.concatenate([y_points_net9_y[:-1], y_points_net9_y[1:]], axis=1)

masked_matrix   = np.ma.array(sommot_matrix_surgcons)
fig, ax         = plt.subplots(1,figsize=(7,4))
im              = ax.imshow(masked_matrix, cmap='RdBu_r', vmin = -0.07, vmax=0.07)

ax.set_xticks(np.arange(len(network_color_net)), labels=ticklabels_net)
ax.set_yticks(np.arange(len(network_color_net2)), labels=ticklabels_net2, )

for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(), network_color_net):
    ticklabel.set_color(tickcolor)

for ticklabel, tickcolor in zip(plt.gca().get_yticklabels(), network_color_net2):
    ticklabel.set_color(tickcolor)

ax.tick_params(top=False,
               bottom=False,
               left=False,
               right=False,
               labelleft=True,
               labelbottom=True)
lc_net9_x    = LineCollection(segments_net9_x,colors=network_color_net, linewidth=2.5,
                               transform=ax.get_xaxis_transform(), clip_on=False )
ax.add_collection(lc_net9_x)

lc_net9_y = LineCollection(segments_net9_y, colors=network_color_net2, linewidth=2.5,
                    transform=ax.get_yaxis_transform(), clip_on=False)
ax.add_collection(lc_net9_y)

plotje.styler(ax, colorbar=im, 
              colorbarlabel='Connectivity', 
              cbaroffset=0.4, 
              title = 'Mean delta connectivity for (Surgery - Conservative)', 
              leftaxis=None, 
              bottomaxis = None, 
              xlabel = 'Networks')


plt.savefig(fig_dir +'delta_sommot_surgcons_network_100.svg', dpi= 300)

masked_matrix   = np.ma.array(sommot_matrix_surgcont)
fig, ax         = plt.subplots(1,figsize=(7,4))
im              = ax.imshow(masked_matrix, cmap='RdBu_r', vmin = -0.07, vmax=0.07)

ax.set_xticks(np.arange(len(network_color_net)), labels=ticklabels_net)
ax.set_yticks(np.arange(len(network_color_net2)), labels=ticklabels_net2, )

for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(), network_color_net):
    ticklabel.set_color(tickcolor)

for ticklabel, tickcolor in zip(plt.gca().get_yticklabels(), network_color_net2):
    ticklabel.set_color(tickcolor)

ax.tick_params(top=False,
               bottom=False,
               left=False,
               right=False,
               labelleft=True,
               labelbottom=True)
lc_net9_x    = LineCollection(segments_net9_x,colors=network_color_net, linewidth=2.5,
                               transform=ax.get_xaxis_transform(), clip_on=False )
ax.add_collection(lc_net9_x)

lc_net9_y = LineCollection(segments_net9_y, colors=network_color_net2, linewidth=2.5,
                    transform=ax.get_yaxis_transform(), clip_on=False)
ax.add_collection(lc_net9_y)

plotje.styler(ax, colorbar=im, 
              colorbarlabel='Connectivity', 
              cbaroffset=0.4, 
              title = 'Mean delta connectivity for (Surgery - Control)', 
              leftaxis=None, 
              bottomaxis = None, 
              xlabel = 'Networks')


plt.savefig(fig_dir +'delta_sommot_surgcont_network_100.svg', dpi= 300)

masked_matrix   = np.ma.array(sommot_matrix_conscont)
fig, ax         = plt.subplots(1,figsize=(7,4))
im              = ax.imshow(masked_matrix, cmap='RdBu_r', vmin = -0.07, vmax=0.07)

ax.set_xticks(np.arange(len(network_color_net)), labels=ticklabels_net)
ax.set_yticks(np.arange(len(network_color_net2)), labels=ticklabels_net2, )

for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(), network_color_net):
    ticklabel.set_color(tickcolor)

for ticklabel, tickcolor in zip(plt.gca().get_yticklabels(), network_color_net2):
    ticklabel.set_color(tickcolor)

ax.tick_params(top=False,
               bottom=False,
               left=False,
               right=False,
               labelleft=True,
               labelbottom=True)
lc_net9_x    = LineCollection(segments_net9_x,colors=network_color_net, linewidth=2.5,
                               transform=ax.get_xaxis_transform(), clip_on=False )
ax.add_collection(lc_net9_x)

lc_net9_y = LineCollection(segments_net9_y, colors=network_color_net2, linewidth=2.5,
                    transform=ax.get_yaxis_transform(), clip_on=False)
ax.add_collection(lc_net9_y)

plotje.styler(ax, colorbar=im, 
              colorbarlabel='Connectivity', 
              cbaroffset=0.4, 
              title = 'Mean delta connectivity for (Conservative - Control)', 
              leftaxis=None, 
              bottomaxis = None, 
              xlabel = 'Networks')


plt.savefig(fig_dir +'delta_sommot_conscont_network_100.svg', dpi= 300)

#In[6]: Figure 4 - Seed-based analysis

#Left seed  = 15, with idx[0]. Hence, left seed becomes 14.
#Right seed = 66, with idx[0]. Hence, right seed becomes 65        
seed_surg_lh = surg_nodes[:,14,:].transpose()
seed_surg_rh = surg_nodes[:,65,:].transpose()
seed_cont_lh = control_nodes[:,14,:].transpose()
seed_cont_rh = control_nodes[:,65,:].transpose()
seed_cons_lh = cons_nodes[:,14,:].transpose()
seed_cons_rh = cons_nodes[:,65,:].transpose()     
 

t_val_surgcons_lh          = np.zeros(len(seed_surg_lh))
p_val_surgcons_raw_lh      = np.zeros(len(seed_surg_lh))
t_val_surgcons_rh          = np.zeros(len(seed_surg_lh))
p_val_surgcons_raw_rh      = np.zeros(len(seed_surg_lh))

t_val_surgcont_lh          = np.zeros(len(seed_surg_lh))
p_val_surgcont_raw_lh      = np.zeros(len(seed_surg_lh))
t_val_surgcont_rh          = np.zeros(len(seed_surg_lh))
p_val_surgcont_raw_rh      = np.zeros(len(seed_surg_lh))

t_val_conscont_lh          = np.zeros(len(seed_surg_lh))
p_val_conscont_raw_lh      = np.zeros(len(seed_surg_lh))
t_val_conscont_rh          = np.zeros(len(seed_surg_lh))
p_val_conscont_raw_rh      = np.zeros(len(seed_surg_lh))

for i in range(len(seed_surg_lh)):
    #Surgery vs. Conservative
    t_val_surgcons_lh[i] , p_val_surgcons_raw_lh[i] = scipy.stats.ttest_ind(seed_surg_lh[i], seed_cons_lh[i])
    t_val_surgcons_rh[i] , p_val_surgcons_raw_rh[i] = scipy.stats.ttest_ind(seed_surg_rh[i], seed_cons_rh[i])
    #Surgery vs. Control
    t_val_surgcont_lh[i] , p_val_surgcont_raw_lh[i] = scipy.stats.ttest_ind(seed_surg_lh[i], seed_cont_lh[i])
    t_val_surgcont_rh[i] , p_val_surgcont_raw_rh[i] = scipy.stats.ttest_ind(seed_surg_rh[i], seed_cont_rh[i])    
    #Conservative vs. Control    
    t_val_conscont_lh[i] , p_val_conscont_raw_lh[i] = scipy.stats.ttest_ind(seed_cons_lh[i], seed_cont_lh[i])
    t_val_conscont_rh[i] , p_val_conscont_raw_rh[i] = scipy.stats.ttest_ind(seed_cons_rh[i], seed_cont_rh[i]) 
    
#In[7]: Figure 4 - Plot network colorbar with significant arrows
#For brain illlustration, see "seedbrain.py"

#Surger vs. Conservative (LH)

p_surgcons_adj_lh = p_val_surgcons_raw_lh.copy()

sig_p = lsu(p_val_surgcons_raw_lh, q=0.05)
sig_idx = np.where(sig_p==True)

arr = np.array(range(123))
res = np.delete(arr, sig_idx)
p_surgcons_adj_lh[res] = 1
p_surgcons_adj_lh[sig_idx] = 0
 
colors=[]
for i in p_surgcons_adj_lh[sortindex]:
    if i>0:
        colors.append('white')
    else:
        colors.append('black')

fig, ax         = plt.subplots(1,figsize=(11,2))
im              = plt.scatter(range(123),p_surgcons_adj_lh[sortindex],color = colors, marker = 'v')
    
ax.tick_params  (top=False,
                     bottom=False,
                     left=False,
                     right=False,
                     labelleft=False,
                     labelbottom=False)


lc_parcel_x    = LineCollection(segments_parcel_x,
                                colors      = sorted_network_colors, 
                                linewidth   = 8,
                                transform   = ax.get_xaxis_transform(), 
                                clip_on     = False, 
                                offsets     = (0,-3))
ax.add_collection(lc_parcel_x)     
lgd  = ax.legend(handles=legend_elements, loc='lower center', ncol= 9, frameon = False, bbox_to_anchor=(0.5,-0.4))
    
plotje.styler(ax, 
              title = 'Significant nodes from LH hip/trunk-area seed (Surgery vs. Conservative)', 
              leftaxis=None, 
              bottomaxis = None, 
              xlabel='Nodes')

plt.savefig(fig_dir+'lhseed_surgcons.svg', bbox_extra_artists=(lgd,), bbox_inches='tight',dpi=300)

#Surgery vs. Conservative (RH)

p_surgcons_adj_rh = p_val_surgcons_raw_rh.copy()

sig_p = lsu(p_val_surgcons_raw_rh, q=0.05)
sig_idx = np.where(sig_p==True)

arr = np.array(range(123))
res = np.delete(arr, sig_idx)
p_surgcons_adj_rh[res] = 1
p_surgcons_adj_rh[sig_idx] = 0

colors=[]
for i in p_surgcons_adj_rh[sortindex]:
    if i>0:
        colors.append('white')
    else:
        colors.append('black')
 
fig, ax         = plt.subplots(1,figsize=(11,2))
im              = plt.scatter(range(123),p_surgcons_adj_rh[sortindex],color = colors, marker = 'v')
    
ax.tick_params  (top=False,
                     bottom=False,
                     left=False,
                     right=False,
                     labelleft=False,
                     labelbottom=False)


lc_parcel_x    = LineCollection(segments_parcel_x,
                                colors      = sorted_network_colors, 
                                linewidth   = 8,
                                transform   = ax.get_xaxis_transform(), 
                                clip_on     = False, 
                                offsets     = (0,-3))
ax.add_collection(lc_parcel_x)
    
lgd  = ax.legend(handles=legend_elements, loc='lower center', ncol= 9, frameon = False, bbox_to_anchor=(0.5,-0.4))
    
plotje.styler(ax, 
              title = 'Significant nodes from RH hip/trunk-area seed (Surgery vs. Conservative)', 
              leftaxis=None, 
              bottomaxis = None, 
              xlabel='Nodes')

plt.savefig(fig_dir+'rhseed_surgcons.svg', bbox_extra_artists=(lgd,), bbox_inches='tight',dpi=300)

#Surgery vs. Control (LH)
p_surgcont_adj_lh = p_val_surgcont_raw_lh.copy()

sig_p = lsu(p_val_surgcont_raw_lh, q=0.05)
sig_idx = np.where(sig_p==True)

arr = np.array(range(123))
res = np.delete(arr, sig_idx)
p_surgcont_adj_lh[res] = 1
p_surgcont_adj_lh[sig_idx] = 0

colors=[]
for i in p_surgcont_adj_lh[sortindex]:
    if i>0:
        colors.append('white')
    else:
        colors.append('black') 
fig, ax         = plt.subplots(1,figsize=(11,2))
im              = plt.scatter(range(123),p_surgcont_adj_lh[sortindex],color=colors, marker = 'v')
    
ax.tick_params  (top=False,
                     bottom=False,
                     left=False,
                     right=False,
                     labelleft=False,
                     labelbottom=False)


lc_parcel_x    = LineCollection(segments_parcel_x,
                                colors      = sorted_network_colors, 
                                linewidth   = 8,
                                transform   = ax.get_xaxis_transform(), 
                                clip_on     = False, 
                                offsets     = (0,-3))
ax.add_collection(lc_parcel_x)  
   
lgd  = ax.legend(handles=legend_elements, loc='lower center', ncol= 9, frameon = False, bbox_to_anchor=(0.5,-0.4))
    
plotje.styler(ax, 
              title = 'Significant nodes from LH hip/trunk-area seed (Surgery vs. Control)', 
              leftaxis=None, 
              bottomaxis = None, 
              xlabel='Nodes')

plt.savefig(fig_dir+'lhseed_surgcont.svg', bbox_extra_artists=(lgd,), bbox_inches='tight',dpi=300)

#Surgery vs. Control (RH)
p_surgcont_adj_rh = p_val_surgcont_raw_rh.copy()

sig_p = lsu(p_val_surgcont_raw_rh, q=0.05)
sig_idx = np.where(sig_p==True)

arr = np.array(range(123))
res = np.delete(arr, sig_idx)
p_surgcont_adj_rh[res] = 1
p_surgcont_adj_rh[sig_idx] = 0

colors=[]
for i in p_surgcont_adj_rh[sortindex]:
    if i>0:
        colors.append('white')
    else:
        colors.append('black')
fig, ax         = plt.subplots(1,figsize=(11,2))
im              = plt.scatter(range(123),p_surgcont_adj_rh[sortindex],color = colors, marker = 'v')
    
ax.tick_params  (top=False,
                     bottom=False,
                     left=False,
                     right=False,
                     labelleft=False,
                     labelbottom=False)


lc_parcel_x    = LineCollection(segments_parcel_x,
                                colors      = sorted_network_colors, 
                                linewidth   = 8,
                                transform   = ax.get_xaxis_transform(), 
                                clip_on     = False, 
                                offsets     = (0,-3))
ax.add_collection(lc_parcel_x)
   
lgd  = ax.legend(handles=legend_elements, loc='lower center', ncol= 9, frameon = False, bbox_to_anchor=(0.5,-0.4))
    
plotje.styler(ax, 
              title = 'Significant nodes from RH hip/trunk-area seed (Surgery vs. Control)', 
              leftaxis=None, 
              bottomaxis = None, 
              xlabel='Nodes')

plt.savefig(fig_dir+'rhseed_surgcont.svg', bbox_extra_artists=(lgd,), bbox_inches='tight',dpi=300)

#Conservative vs. Control (LH)
p_conscont_adj_lh = p_val_conscont_raw_lh.copy()

sig_p = lsu(p_val_conscont_raw_lh, q=0.05)
sig_idx = np.where(sig_p==True)

arr = np.array(range(123))
res = np.delete(arr, sig_idx)
p_conscont_adj_lh[res] = 1
p_conscont_adj_lh[sig_idx] = 0
 
colors=[]
for i in p_conscont_adj_lh[sortindex]:
    if i>0:
        colors.append('white')
    else:
        colors.append('black')

fig, ax         = plt.subplots(1,figsize=(11,2))
im              = plt.scatter(range(123),p_conscont_adj_lh[sortindex],color = colors, marker = 'v')
    
ax.tick_params  (top=False,
                     bottom=False,
                     left=False,
                     right=False,
                     labelleft=False,
                     labelbottom=False)


lc_parcel_x    = LineCollection(segments_parcel_x,
                                colors      = sorted_network_colors, 
                                linewidth   = 8,
                                transform   = ax.get_xaxis_transform(), 
                                clip_on     = False, 
                                offsets     = (0,-3))
ax.add_collection(lc_parcel_x)
   
lgd  = ax.legend(handles=legend_elements, loc='lower center', ncol= 9, frameon = False, bbox_to_anchor=(0.5,-0.4))
    
plotje.styler(ax, 
              title = 'Significant nodes from LH hip/trunk-area seed (Conservative vs. Control)', 
              leftaxis=None, 
              bottomaxis = None, 
              xlabel='Nodes')

plt.savefig(fig_dir+'lhseed_conscont.svg', bbox_extra_artists=(lgd,), bbox_inches='tight',dpi=300)

#Conservative vs. Control (RH)


p_conscont_adj_rh = p_val_conscont_raw_rh.copy()

sig_p = lsu(p_val_conscont_raw_rh, q=0.05)
sig_idx = np.where(sig_p==True)

arr = np.array(range(123))
res = np.delete(arr, sig_idx)
p_conscont_adj_rh[res] = 1
p_conscont_adj_rh[sig_idx] = 0

colors=[]
for i in p_conscont_adj_rh[sortindex]:
    if i>0:
        colors.append('white')
    else:
        colors.append('black')
 
fig, ax         = plt.subplots(1,figsize=(11,2))
im              = plt.scatter(range(123),p_conscont_adj_rh[sortindex],color = colors, marker = 'v')
    
ax.tick_params  (top=False,
                     bottom=False,
                     left=False,
                     right=False,
                     labelleft=False,
                     labelbottom=False)


lc_parcel_x    = LineCollection(segments_parcel_x,
                                colors      = sorted_network_colors, 
                                linewidth   = 8,
                                transform   = ax.get_xaxis_transform(), 
                                clip_on     = False, 
                                offsets     = (0,-3))
ax.add_collection(lc_parcel_x)  
   
lgd  = ax.legend(handles=legend_elements, loc='lower center', ncol= 9, frameon = False, bbox_to_anchor=(0.5,-0.4))
    
plotje.styler(ax, 
              title = 'Significant nodes from RH hip/trunk-area seed (Conservative vs. Control)', 
              leftaxis=None, 
              bottomaxis = None, 
              xlabel='Nodes')
plt.savefig(fig_dir+'rhseed_conscont.svg', bbox_extra_artists=(lgd,), bbox_inches='tight',dpi=300)