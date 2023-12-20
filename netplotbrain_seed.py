# In[1]:

import matplotlib.colors as colors
import templateflow.api as tf
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from nilearn import plotting
from nilearn import datasets, surface
import scipy
from multipy.fdr import lsu
from bids import BIDSLayout
import netplotbrain

data_dir        = './bids/'
conn_dir        = './bids/derivatives/connectivity/'
file_dir        = './datasets/'
fig_dir         = './figures/publication/'
analysis_dir    = './analysis/nbs_results/100/'
map_dir         = './atlases/'

# Grab bids structure
layout = BIDSLayout(data_dir)
# Add connectivity directory
layout.add_derivatives(conn_dir)

# Read MR-files
files = layout.derivatives['connectivity'].get(extension='.tsv', space = 'MNI152NLin2009cAsym', atlas = 'HOSubCortSchaefer2018100Parcels7Networks')
conn_list = list(np.zeros(len(files))) 
counter = 0
fil_mrnumber    = []
control_df      = []
surg_df         = []
cons_df         = []
all_list        = {}
surg_names      = []
control_names   = []
cons_names      = []
all_names       = {}

dataset = pd.read_csv(file_dir + 'b2b_dataset_20230626.csv', sep='\t')
atlas   = pd.read_csv(map_dir  + 'surfacemap.tsv'          , sep='\t')

for file in files:
    entities = dict(file.entities) 
    file_df = pd.read_csv(file.path, sep='\t', index_col=[0])
    idx = np.where(dataset['mr_brain'] == int(entities['subject']))
    idx = int(idx[0])
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


#Left seed = 15, with idx[0]. Hence, left seed becomes 14.
#Right seed = 66 with idx[0]. Hence, right seed becomes 65        
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

p_surgcons_adj_lh = p_val_surgcons_raw_lh.copy()
sig_p = lsu(p_val_surgcons_raw_lh, q=0.05)
sig_idx = np.where(sig_p==True)
surgcons_sigs_lh = atlas['name'][sig_idx[0]]
sig_sub = atlas['name'][sig_idx[0][10]]


arr = np.array(range(123))
res = np.delete(arr, sig_idx)
p_surgcons_adj_lh[res] = 0
p_surgcons_adj_lh[sig_idx] = 1    

p_surgcons_adj_rh = p_val_surgcons_raw_rh.copy()

sig_p = lsu(p_val_surgcons_raw_rh, q=0.05)
sig_idx = np.where(sig_p==True)
surgcons_sigs_rh = atlas['name'][sig_idx[0]]

arr = np.array(range(123))
res = np.delete(arr, sig_idx)
p_surgcons_adj_rh[res] = 0
p_surgcons_adj_rh[sig_idx] = 1

p_surgcont_adj_lh = p_val_surgcont_raw_lh.copy()

sig_p = lsu(p_val_surgcont_raw_lh, q=0.05)
sig_idx = np.where(sig_p==True)
surgcont_sigs_lh = atlas['name'][sig_idx[0]]
sig_cereb = atlas['name'][sig_idx[0][9]]

arr = np.array(range(123))
res = np.delete(arr, sig_idx)
p_surgcont_adj_lh[res] = 0
p_surgcont_adj_lh[sig_idx] = 1


p_surgcont_adj_rh = p_val_surgcont_raw_rh.copy()

sig_p = lsu(p_val_surgcont_raw_rh, q=0.05)
sig_idx = np.where(sig_p==True)

arr = np.array(range(123))
res = np.delete(arr, sig_idx)
p_surgcont_adj_rh[res] = 0
p_surgcont_adj_rh[sig_idx] = 1


p_surgcont_adj_lh = p_surgcont_adj_lh[0:100]
p_surgcont_adj_rh = p_surgcont_adj_rh[0:100]

p_surgcons_adj_lh = p_surgcons_adj_lh[0:100]
p_surgcons_adj_rh = p_surgcons_adj_rh[0:100]


# In[1]:

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
netsort = np.zeros(len(surg_arr))
for i, key_i in enumerate(network_indices):
    netsort[network_indices[key_i]] = i
    
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




highlight_surgcons_lh = {}
for i, key_i in enumerate(surgcons_sigs_lh):
                          highlight_surgcons_lh[i] = atlas[atlas['name'] == key_i]
                          
highlight = np.zeros(len(highlight_surgcons_lh))
for i in highlight_surgcons_lh:
    highlight[i] = highlight_surgcons_lh[i].index[0]
nodeinfo = np.zeros(len(atlas))
for i, key_i in enumerate(highlight):
    nodeinfo[int(highlight[i-1])] = 1


node_df = pd.DataFrame({'sign': nodeinfo, 'color': network_color_full})
plt.figure()
netplotbrain.plot(template='MNI152NLin2009cAsym',
                  nodes = map_dir + 'surfacemap.nii.gz',
                  nodes_df = node_df,
                  node_type='parcels',
                  node_color='color',
                  node_alpha = 0.0,
                  highlight_nodes = 'sign',
                  highlight_level=0.95,
                  view='LSP',
                  title = None,
                  figdpi = 300)
plt.savefig(fig_dir + 'netplot_surgcons_lhseed_3.png', dpi = 300)




highlight_surgcons_rh = {}
for i, key_i in enumerate(surgcons_sigs_rh):
                          highlight_surgcons_rh[i] = atlas[atlas['name'] == key_i]
                          
highlight = np.zeros(len(highlight_surgcons_rh))
for i in highlight_surgcons_rh:
    highlight[i] = highlight_surgcons_rh[i].index[0]   
nodeinfo = np.zeros(len(atlas))
for i, key_i in enumerate(highlight):
    nodeinfo[int(highlight[i-1])] = 1


node_df = pd.DataFrame({'sign': nodeinfo, 'color': network_color_full})
plt.figure()
netplotbrain.plot(template='MNI152NLin2009cAsym',
                  nodes = map_dir + 'surfacemap.nii.gz',
                  nodes_df = node_df,
                  node_type='parcels',
                  node_color='color',
                  node_alpha = 0.0,
                  highlight_nodes = 'sign',
                  highlight_level=0.95,
                  view='LSP',
                  title = None,
                  figdpi = 300)
plt.savefig(fig_dir + 'netplot_surgcons_rhseed_3.png', dpi = 300)



highlight_surgcont_lh = {}
for i, key_i in enumerate(surgcont_sigs_lh):
                          highlight_surgcont_lh[i] = atlas[atlas['name'] == key_i]
                          
highlight = np.zeros(len(highlight_surgcont_lh))
for i in highlight_surgcont_lh:
    highlight[i] = highlight_surgcont_lh[i].index[0]
nodeinfo = np.zeros(len(atlas))
for i, key_i in enumerate(highlight):
    nodeinfo[int(highlight[i-1])] = 1


node_df = pd.DataFrame({'sign': nodeinfo, 'color': network_color_full})
plt.figure()
netplotbrain.plot(template='MNI152NLin2009cAsym',
                  nodes = map_dir + 'surfacemap.nii.gz',
                  nodes_df = node_df,
                  node_type='parcels',
                  node_color='color',
                  node_alpha = 0,
                  highlight_nodes = 'sign',
                  highlight_level=0.95,
                  view='LSP',
                  title = None,
                  figdpi = 300)
plt.savefig(fig_dir + 'netplot_surgcont_lhseed_3.png', dpi = 300)



surgcont_sigs_rh = []
highlight_surgcont_rh = {}
for i, key_i in enumerate(surgcont_sigs_rh):
                          highlight_surgcont_rh[i] = atlas[atlas['name'] == key_i]
                          
highlight = np.zeros(len(highlight_surgcont_rh))
for i in highlight_surgcont_rh:
    highlight[i] = highlight_surgcont_rh[i].index[0]   
nodeinfo = np.zeros(len(atlas))
for i, key_i in enumerate(highlight):
    nodeinfo[int(highlight[i-1])] = 1






node_df = pd.DataFrame({'sign': nodeinfo, 'color': network_color_full})
plt.figure()
netplotbrain.plot(template='MNI152NLin2009cAsym',
                  nodes = map_dir + 'surfacemap.nii.gz',
                  nodes_df = node_df,
                  node_type='parcels',
                  node_color='color',
                  node_alpha = 0.0,
                  highlight_nodes = 'sign',
                  highlight_level=0.95,
                  view='LSP',
                  title = None,
                  figdpi = 300)
plt.savefig(fig_dir + 'netplot_surgcont_rhseed_3.png', dpi = 300)



plt.figure()
netplotbrain.plot(template='MNI152NLin2009cAsym',
                  nodes = map_dir + 'surfacemap.nii.gz',
                  nodes_df = node_df,
                  node_type='parcels',
                  node_color='color',
                  view='LSP',
                  title = None,
                  figdpi = 300)
plt.savefig(fig_dir + 'netplot_allnets_2.png', dpi = 300)
