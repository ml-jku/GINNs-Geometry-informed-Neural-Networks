# Explanation of config parameters

## 1 General
### 1.1 Global parameters (usually left constant)
- `seed`: 17
  - set the global seed to make results reproducible
- `level_set`: 0
  - the levelset to choose for the shape
- `ny`: 1 
  - output is a scalar
- `use_x_and_z_arg`: True  
  - used when creating the stateless model
- `reset_zlatents_every_n_epochs`: 1  
  - uncomment to set; not set means the latents are kept fixed
- `z_sample_interval`: [-0.1, 0.1]
  - range for sampling latent variables
- `z_sample_method`: 'uniform'  
  - sampling distribution of latent variables; other options: 'normal', 'w0-informed', 'linspace'

### 1.2 General
- `problem`: 'simjeb'
  - specifies the problem being addressed
- `nx`: 3 
  - input is 3-dimensional
- `nz`: 1
  - dimension of the latent code
- `batch_size`: 1
  - number of training samples per batch
- `plot_n_shapes`: 1
  - number of shapes to plot; cannot exceed `batch_size`

## 2 Model
- `model`: 'cond_siren'
  - specifies the model type to use
- `layers`: [4, 256, 256, 256, 256, 256, 1]
  - architecture of the model following the SIREN paper
- `w0`: 1.0
  - SIREN's default frequency base
- `w0_initial`: 8.0
  - initial value of `w0` at the start of training; this is crucial to control the smoothness of the shape

## 3 Training
### 3.1 Outer optimization loop
- `max_epochs`: 2000
  - maximum number of training epochs
- `lr`: 0.001
  - learning rate for the optimizer

### 3.2 Scheduler
- `use_scheduler`: True
  - whether to use a learning rate scheduler
- `scheduler_gamma`: 0.5
  - multiplicative factor of learning rate decay
- `decay_steps`: 1000
  - number of steps between learning rate decay

### 3.3 Gradient clipping
- `grad_clipping_on`: True
  - enables gradient clipping
- `grad_clip`: 0.5
  - max norm of the gradients
- `auto_clip_on`: True
  - enable automatic gradient clipping
- `auto_clip_percentile`: 0.8
  - percentile for automatic clipping
- `auto_clip_hist_len`: 50
  - history length to consider for automatic clipping
- `auto_clip_min_len`: 50
  - minimum history length required to perform clipping

## 4 Constraints

### 4.1 Points to sample
- `dreg_sample_from`: 'exterior'
  - specifies from where to sample the points which are used to push the OUTSIDE values to <0; it could also be 'on envelope' which would
- `n_points_domain`: 2048
  - used for eikonal loss computation
- `n_points_envelope`: 16384
  - points sampled from the envelope
- `n_points_interfaces`: 4096
  - points at the interfaces
- `n_points_normals`: 4096
  - normal points for calculations
- `n_points_obstacles`: 128
  - points around obstacles (only if present)
- `n_points_find_surface`: 4096
  - points used to find the surface
- `n_points_find_cps`: 4096
  - points used to find critical points

### 4.2 Loss-balancing
#### 4.2.1 Local
- `lambda_itf`: 1
  - weight for interface loss
- `lambda_env`: 0.1
  - weight for design region loss
- `lambda_obst`: 0.0
  - weight for obstacle loss (disabled by for simjeb)
- `lambda_normal`: 1.0e-6
  - weight for normal vector loss
#### 4.2.2 Global
- `lambda_eikonal`: 1.0e-9
  - weight for eikonal equation loss
- `lambda_scc`: 1.0e-2
  - weight for connectedness loss; scc stands for single-connected component
- `lambda_div`: 1.0e-8
  - weight for diversity loss
- `lambda_curv`: 0
  - weight for curvature loss

### 4.3 CONNECTEDNESS

#### 4.3.1 Finding critical points
- `lr_find_cps`: 0.01
  - learning rate for finding critical points
- `max_iter_find_cps`: 1000
  - maximum iterations to find critical points
  
The computation of critical points is computationally expensive. Therefore an option is to adjust the found critical points from the previous iterations.
- `recalc_cps_every_n_epochs`: 20
  - epochs until 
- `lr_adjust_cps`: 0.001
  - adjustment learning rate; much smaller than `lr_find_cps`

#### 4.3.2 Flow from saddle points to minima/maxima
- `n_iter_saddle_descent`: 1000
  - iterations for descent from saddle points
- `lr_saddle_descent`: 0.03
  - learning rate for saddle point descent
- `perturbation_saddle_descent`: 0.01
  - perturbation magnitude for descent
- `stop_grad_mag_saddle_descent`: 1.0e-5
  - stopping criterion based on gradient magnitude
- `check_interval_grad_mag_saddle_descent`: 5
  - interval for checking if the gradient magnitude is 

#### 4.3.3 Clustering
- `dbscan.y_x_mag`: 1.0e-5
  - filter threshold for DBSCAN based on gradient magnitude
- `dbscan.eps`: 0.05
  - spatial extent for DBSCAN clustering

#### 4.3.4 Graph algorithm
- `graph_algo_inv_dist_eps`: 0.1
  - epsilon value for inverse distance in graph algorithms
- `remove_penalty_points_outside_envelope`: True
  - remove penalty points that fall outside the envelope

### 4.4 Diversity

#### 4.4.1 Distance in function space
- `recompute_surface_pts_every_n_epochs`: 10
  - epochs between recomputations of surface points
- `find_surface_pts_n_iter`: 200
  - iterations to find surface points
- `lr_find_surface`: 0.01
  - learning rate for finding surface points
- `find_surface_pts_prec_eps`: 1.0e-6
  - precision epsilon for finding surface points
- `reweigh_surface_pts_close_to_interface`: True
  - enable reweighing of points close to the interface
- `reweigh_surface_pts_close_to_interface_cutoff`: 0.2
  - cutoff distance for interface proximity reweighing; points which are farer away to the interface than this threshold will have weight 1
- `reweigh_surface_pts_close_to_interface_power`: 2.0
  - exponent for distance weighting near interfaces

### 4.5 Curvature
- `strain_curvature_clip_max`: 1.0e+3
  - maximum allowable strain in curvature calculations

## 5 Logging

### 5.1 Multi-processing
- `num_workers`: 8
  - number of worker processes for parallel computation on the CPU; this is mostly used to create plots to not block the training

### 5.2 Wandb
- `wandb_experiment_name`: 'simjeb_cond_siren'
  - name of the Weights & Biases experiment
- `model_save_path`: 'path/to/saved_models'
  - directory to save models
- `load_model`: False
  - flag to load model from a file
- `model_load_path`: '/path/to/model-4ym4lhok.pth'
  - path to the model file to load
- `model_load_wandb_id`: 'cdfgpgqe'
  - Weights & Biases ID of the model to load
- `save_every_n_epochs`: 10
  - frequency of saving the last model checkpoint; for a specific model, the previous checkpoint will be overwritten

### 5.3 Timer
- `timer_print`: False
  - disable printing timing information
- `timer_accumulate`: True
  - accumulate timing information to print at the end

## 6 Plotting
- `plot_every_n_epochs`: 50
  - frequency of plotting results
- `fig_path`: 'img/simjeb/cond_siren'
  - directory for saving figures
- `fig_size`: [12, 7]
  - size of the figures (width, height)
- `fig_show`: False
  - whether to display figures
- `fig_save`: False
  - whether to save figures to the filesystem
- `fig_wandb`: True
  - whether to log figures to Weights & Biases
- `force_saving_shape`: False
  - force saving of shapes; this is used only to save the shape even is the general fig_save is disabled
- `show_colorbar`: True
  - whether to show color bar on plots

### 6.1 Meshing
- `mc_resolution`: 128
  - resolution for the marching cubes algorithm to create the mesh

### 6.2 Specific plots

#### 6.2.1 General plots
- `plot_shape`: True
  - whether to plot the shape

#### 6.2.2 Surface points plots
- `plot_surface_points`: True
  - plot points sampled on the surface
- `plot_surface_descent`: True
  - plot the descent process from points to the surface

#### 6.2.3 Plots to debug connectedness

##### 6.2.3.1 Finding Critical Points
- `plot_start_points_to_find_cps`: True
  - plot initial points for critical point search
- `plot_cps`: True
  - plot critical points found
- `plot_cp_descent_refinement`: True
  - plot refinement process in critical point descent
- `plot_cp_descent_recomputation`: True
  - plot recomputation steps in critical point descent
- `plot_cp_candidates`: True
  - plot critical point candidates

##### 6.2.3.2 Characterize Critical Points and perturb 
- `plot_grad_mag_hist`: True
  - plot histogram of gradient magnitudes to check threshold settings
- `plot_characterized_cps`: True
  - plot characterized critical points
- `plot_perturbed_sps`: False
  - plot perturbed saddle points
- `plot_saddle_descent`: True
  - plot the descent from saddle points

##### 6.2.3.3 Build Surface-Net graph
- `plot_edges`: True
  - plot graph edges
- `plot_surfnet_graph`: True
  - plot the Surface-Net graph

##### 6.2.3.4 Build augmented Surface-Net graph and find penalty points
- `plot_intermediate_wd_graph`: True
  - plot the intermediate weighted graph
- `plot_sub_and_neighbor_wd_graph`: True
  - plot subgraphs and their neighbors
- `plot_penalty_points`: True
  - plot penalty points
- `plot_wd_graph`: True
  - plot the weighted graph
- `plot_penalty_points_only_in_envelope`: True
  - plot penalty points only within the envelope
