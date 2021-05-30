# Evolution Evaluation


`target` is always mean camera.
`start` is the "wiggled" mean camera:

The number of wiggled parameters is set to 7, which is 46% of all 15 parameters. Every parameter has a chance to get selected for variation.

The amount of wiggling is specified as follows:

Parameter	| Description	| Near	| Medium	| Far	|
|:---	|:---:	|:---:	|:---:	|:---:	|
fu	| focal u	| ±10	| ±50	| ±100	|
fv	| focal v	| ±10	| ±50	| ±100	|
cx	| principal point offset x [px]	| ±10	| ±50	| ±100	|
cy	| principal point offset y [px]	| ±10	| ±50	| ±100	|
tx	| translation in x [m]	| ±0.1	| ±0.5	| ±1.0	|
ty	| translation in y [m]	| ±0.1	| ±0.5	| ±2.0	|
tz	| translation in z [m]	| ±0.1	| ±0.5	| ±1.0	|
rx	| rotation in x [rad]	| ±0.01	| ±0.1	| ±0.2	|
ry	| rotation in y [rad]	| ±0.01	| ±0.1	| ±0.2	|
rz	| rotation in z [rad]	| ±0.01	| ±0.1	| ±0.2	|
d0	| distortion parameter	| ±0.1	| ±0.25	| ±0.5	|
d1	| distortion parameter	| ±0.1	| ±0.25	| ±0.5	|
d2	| distortion parameter	| ±0.025	| ±0.1	| ±0.2	|
d3	| distortion parameter	| ±0.01	| ±0.05	| ±0.1	|
d4	| distortion parameter	| ±0.5	| ±1.5	| ±3.0	|

## [INPUT] VARIABLES


### Start camera specification 
Given a randomly chosen wiggle vector for a specified distance type, the start is defined by:

	['s_fu', 's_fv', 's_cx', 's_cy', 's_tx', 's_ty', 's_tz', 's_rx', 's_ry', 's_rz', 's_d0', 's_d1', 's_d2', 's_d3', 's_d4']

### Target camera specification
The (in a real world application unknown) target values are then:

	['t_fu', 't_fv', 't_cx', 't_cy', 't_tx', 't_ty', 't_tz', 't_rx', 't_ry', 't_rz' 't_d0', 't_d1', 't_d2', 't_d3', 't_d4']

> Pleas Note that this is  <ins>always</ins> the mean squash camera in **our case**!

### Evolution Algorithm related variables
| Variable	| Description	| Variants	|
|:---	|:---	|:---	|
| population_fn	| Strategy for creating the initial population	| BoundedUniformPopulation	|
|	|	| ValueUniformPopulation	|
| fitness_fn	|	| DistanceMap [L1|L2]	|
|	|	| DistanceMapWithPunishment [L1|L2]	|
| selection_fn	|	| RouletteWheel	|
|	|	| Tournament	|
|	|	| Random	|
| crossover_fn	|	| SinglePoint	|
|	|	| TwoPoint	|
| mutation_fn	|	| BoundedUniformMutation	|
|	|	| BoundedDistributionBasedMutation	|
| termination_fn	|	| MaxIteration	|
|	|	| NoImprovement	|
|	|	| FitnessReached	|
|	|	| And	|
|	|	| Or	|

### Noise related variables
| Variable	| Description	| Variants	|
|:---	| :---	| :---	|
| noise_type	| 	| no_noise	|
| 	| 	| salt	|
| 	| 	| hlines	|
| 	| 	| straight_grid	|
| 	| 	| angled_grid	|


## [OUTPUT] RESULTS

### Best camera
The best fitted camera result is then 

	'r_fu', 'r_fv', 'r_cx', 'r_cy', 'r_tx', 'r_ty', 'r_tz', 'r_rx', 'r_ry', 'r_rz', 'r_d0', 'r_d1', 'r_d2', 'r_d3', 'r_d4'

### Derived metrics

| Metric	| Description	|
| :---	| :---	|
| `best_fitness`	| Best achieved fitness value	|
| `generations`	| Number of generations to achieve the best fitness value	|
| `mean_geometry_reproj_error_col`	| Mean reprojection error in X for **fitting geometry** points	|
| `mean_geometry_reproj_error_row`	| Mean reprojection error in Y for **fitting geometry** points	|
| `mean_dense_reproj_error_col`	| Mean reprojection error in X for **dense sampled (16 each axis) geometry**  points	|
| `mean_dense_reproj_error_row`	| Mean reprojection error in Y for **dense sampled (16 each axis) geometry** points	|
| `mean_y0_reproj_error_col`	| Mean reprojection error in X for **Y=0 sampled (16 each axis) Plane geometry** points	|
| `mean_y0_reproj_error_row`	| Mean reprojection error in Y for **Y=0 sampled (16 each axis) Plane geometry** points	|


----
```python
# Things to report

# NLOPT
# a) ✅ StartCamera:
#    ['s_fu', 's_fv', 's_cx', 's_cy', 's_tx', 's_ty', 's_tz', 's_rx', 's_ry', 's_rz', 's_d0', 's_d1', 's_d2', 's_d3', 's_d4']
# b) ✅ TargetCamera
#    ['t_fu', 't_fv', 't_cx', 't_cy', 't_tx', 't_ty', 't_tz', 't_rx', 't_ry', 't_rz', 't_d0', 't_d1', 't_d2', 't_d3', 't_d4']
# c) ✅ ResultCamera:
#    ['r_fu', 'r_fv', 'r_cx', 'r_cy', 'r_tx', 'r_ty', 'r_tz', 'r_rx', 'r_ry', 'r_rz', 'r_d0', 'r_d1', 'r_d2', 'r_d3', 'r_d4']
# d) ✅ NoiseType including the corresponding value
# e) ⭕ StrategyBundle (Not applicable)
# f) ✅ Optimizer Type
# h) ✅ best_fitness
# h) ⭕ generations (Not applicable)
# i) ✅ mean_geometry_reproj_error_col
# j) ✅ mean_geometry_reproj_error_row
# k) ✅ mean_dense_reproj_error_col
# j) ✅ mean_dense_reproj_error_row
# m) ✅ mean_y0_reproj_error_col
# n) ✅ mean_y0_reproj_error_row


# EVO
# region [TMP]
# Things to report
# a) ✅ StartCamera:
#    ['s_fu', 's_fv', 's_cx', 's_cy', 's_tx', 's_ty', 's_tz', 's_rx', 's_ry', 's_rz', 's_d0', 's_d1', 's_d2', 's_d3', 's_d4']
# b) ✅ TargetCamera
#    ['t_fu', 't_fv', 't_cx', 't_cy', 't_tx', 't_ty', 't_tz', 't_rx', 't_ry', 't_rz', 't_d0', 't_d1', 't_d2', 't_d3', 't_d4']
# c) ✅ ResultCamera:
#    ['r_fu', 'r_fv', 'r_cx', 'r_cy', 'r_tx', 'r_ty', 'r_tz', 'r_rx', 'r_ry', 'r_rz', 'r_d0', 'r_d1', 'r_d2', 'r_d3', 'r_d4']
# d) ✅ NoiseType including the corresponding value
# e) ✅ StrategyBundle
# f) ⭕ Optimizer Type
# g) ✅ best_fitness (Not applicable)
# h) ✅ generations (Not applicable)
# i) ✅ mean_geometry_reproj_error_col
# j) ✅ mean_geometry_reproj_error_row
# k) ✅ mean_dense_reproj_error_col
# l) ✅ mean_dense_reproj_error_row
# m) ✅ mean_y0_reproj_error_col
# n) ✅ mean_y0_reproj_error_row
```