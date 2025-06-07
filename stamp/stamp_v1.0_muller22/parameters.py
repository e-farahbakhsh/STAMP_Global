parameters = {
    
    'plate_model_name': 'muller2022',
    
    'timespan': {
    'min': 0,
    'max': 540
    },
    
    'temporal_resolution': 1,
    'num_unlabelled': 10,
    'grid_resolution': 0.5,
    'buffer_distance': 6,
    
    'plate_model_dir': 'plate_model',
    'inputs_dir': 'inputs',
    'outputs_dir': 'outputs',
    'buffer_zones_dir': 'buffer_zones',
    'ml_dir': 'ml',
    'prob_grids_dir': 'prob_grids',
    'prob_maps_dir': 'prob_maps',
    
    'subduction_data_filename': 'subduction_data.csv',
    'deposit_coords_filename': 'deposit_coords_weight.csv',
    'deposit_recon_coords_filename': 'deposit_recon_coords.csv',
    'deposit_recon_coords_all_filename': 'deposit_recon_coords_all.csv',
    'unlabelled_coords_filename': 'unlabelled_coords.csv',
    'combined_coords_filename': 'combined_coords.csv',
    'target_coords_filename': 'target_coords.csv',
    'deposit_data_filename': 'deposit_data.csv',
    'unlabelled_data_filename': 'unlabelled_data.csv',
    'target_data_filename': 'target_data.csv',
    'features_imputed_weight_label_filename': 'features_imputed_weight_label.csv',
    'corr_filename': 'corr.csv',
    'Xy_train_original_filename': 'Xy_train_original.csv',
    'Xy_pos_test_original_filename': 'Xy_pos_test_original.csv',
    'Xy_train_filename': 'Xy_train.csv',
    'Xy_pos_test_filename': 'Xy_pos_test.csv',
    'Xy_train_new_filename': 'Xy_train_new.csv',
    'Xy_rf_train_filename': 'Xy_rf_train.csv',
    'Xy_rf_test_filename': 'Xy_rf_test.csv',
    'importances_filename': 'importances.csv',
    'target_prob_filename': 'target_prob.csv',
    
    'robust_scaler_filename': 'robust_scaler.pkl',
    'model_pub_filename': 'model_pub.pkl',
    'model_rf_filename': 'model_rf.pkl',
    
    'columns_to_drop_deposit': [
        'present_lon',
        'present_lat',
        'age (Ma)',
        'weight',
        'label',
        'plate_id',
        'lon',
        'lat',
        'overriding_plate_id',
        'subducting_plate_ID',
        'trench_plate_ID',
        ],
    
    'columns_to_drop_unlabelled': [
        'present_lon',
        'present_lat',
        'age (Ma)',
        'weight',
        'label',
        'lon',
        'lat',
        'overriding_plate_id',
        'subducting_plate_ID',
        'trench_plate_ID',
        ],
    
    'columns_to_drop_target': [
        'present_lon',
        'present_lat',
        'age (Ma)',
        'lon',
        'lat',
        'subducting_plate_ID',
        'trench_plate_ID',
        ]
    
    }
