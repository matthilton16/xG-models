import os
import logging
import pandas as pd
import numpy as np
import pickle 

from src.features.data_processing import DataProcessor
from src.features.feature_engineering import FeatureEngineer
from src.models.gp_xg_models import VGPXGClassifier, SVGPXGClassifier
from src.models.value_players import PlayerValueCalculator
from src.viz.plot_utils import XGDistributionPlots, XGPitchMapPlots

# --- Configuration ---
MAIN_PY_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = MAIN_PY_DIR 

# Input data paths
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')

# Output paths
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
FEATURES_DIR = os.path.join(BASE_DIR, 'data', 'features')
SAVED_MODELS_DIR = os.path.join(BASE_DIR, 'models', 'gp')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
PLAYER_VALUES_RESULTS_DIR = os.path.join(RESULTS_DIR, 'player-values')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# --- Logging Setup ---
log_file_path = os.path.join(LOGS_DIR, 'pipeline.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler() 
    ]
)
logger = logging.getLogger(__name__)

# --- Main Pipeline Function ---
def run_pipeline():
    logger.info("Starting xG model pipeline...")

    
    # 1. Data Processing
    logger.info("=== Stage 1: Data Processing ===")
    data_processor = DataProcessor(RAW_DATA_DIR, PROCESSED_DATA_DIR)
    
    try:
        logger.info("Running DataProcessor: create consolidated_events file...")
        data_processor._consolidate_event_json_files()
        logger.info("Running DataProcessor: create shot_event_df...")
        shot_event_df = data_processor.create_shot_event_df()
        logger.info("Running DataProcessor: create shot_tracking_df...")
        shot_tracking_df = data_processor.create_shot_tracking_df()
        logger.info("Running DataProcessor: create goalkeepers_df...")
        goalkeepers_df = data_processor._create_goalkeepers_df()
        logger.info("Running DataProcessor: create enriched_tracking_df...")
        enriched_tracking_df = data_processor._enrich_tracking_data(
            tracking_df=shot_tracking_df.copy(), 
            event_df=shot_event_df.copy(),
            goalkeepers_df=goalkeepers_df
        )
        logger.info("Filtering data for GK and shooter presence...")
        filtered_shot_event_df, filtered_tracking_df_for_norm = data_processor._filter_data_for_gk_and_shooter(
            event_df=shot_event_df.copy(),
            tracking_df=enriched_tracking_df.copy()
        )
        logger.info(f"After filtering: shot_event_df shape: {filtered_shot_event_df.shape}, tracking_df shape: {filtered_tracking_df_for_norm.shape}")

        logger.info("Running DataProcessor: create normalized_tracking_df...")
        normalized_tracking_df = data_processor._normalize_tracking_coordinates(
            tracking_df=filtered_tracking_df_for_norm.copy(),
            event_df=filtered_shot_event_df.copy()
        )
        
        
        logger.info("DataProcessor methods executed.")
    except Exception as e:
        logger.error(f"Error during DataProcessor execution: {e}", exc_info=True)
        logger.error("Please ensure DataProcessor methods are correctly implemented and raw data is available.")
        raise e
    
   
    # 2. Feature Engineering
    logger.info("=== Stage 2: Feature Engineering ===")
    # generate features
    feature_engineer = FeatureEngineer(pitch_length_m=100.0, pitch_width_m=100.0)
    features_df, features_names = feature_engineer.generate_features(event_df=filtered_shot_event_df.copy(), 
                                                             tracking_df=normalized_tracking_df.copy())
    if features_df.empty:
        logger.error("Feature generation resulted in an empty DataFrame. Exiting.")
        return
    logger.info(f"Generated features_df with shape: {features_df.shape}. Features: {features_names}")
    features_df.to_csv(os.path.join(FEATURES_DIR, 'model_features_unscaled.csv'), index=False)

    logger.info("Performing train/test split and scaling...")
    X_train_scaled, X_test_scaled, _, \
    y_train, y_test, _, scaler = \
        feature_engineer.split_and_scale_data(
            features_df, 
            features_names, 
            train_test_only=True
        )
    
    logger.info(f"Data split and scaled (Train/Test only). Shapes: X_train_scaled: {X_train_scaled.shape}, y_train: {y_train.shape}, "
                f"X_test_scaled: {X_test_scaled.shape}, y_test: {y_test.shape}")
    
    # Save the scaler
    scaler_path = os.path.join(FEATURES_DIR, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler saved to {scaler_path}")

    # get input dimension
    input_dim = X_train_scaled.shape[1]


    # 3. VGP Model Training and Prediction
    logger.info("=== Stage 3: VGP Model Training & Prediction ===")
    vgp_model = VGPXGClassifier(input_dim=input_dim)
    vgp_model.train(X_train_scaled, y_train.values, training_iterations=100) # .values for y_train Series, reduced iterations
    vgp_model_path = os.path.join(SAVED_MODELS_DIR, 'vgp_xg_classifier')
    vgp_model.save(vgp_model_path)
    logger.info(f"VGP model saved to {vgp_model_path}")
    vgp_model.get_ard_lengthscales(feature_names=features_names)

    # Predict latent mean and variance for ALL shots for player valuation
    X_all_features = features_df[features_names]
    X_all_scaled = scaler.transform(X_all_features) # Use the same scaler
    
    f_mean_all_vgp, f_var_all_vgp = vgp_model.predict_latent_mean_and_variance(X_all_scaled)
    # Transform latent mean to probability using the likelihood's invlink
    xg_mean_for_valuation_vgp = vgp_model.likelihood.invlink(f_mean_all_vgp).numpy().flatten()
    xg_variance_for_valuation_vgp = f_var_all_vgp.flatten() 
    logger.info(f"VGP predictions for valuation: xG_mean shape {xg_mean_for_valuation_vgp.shape}, xG_var shape {xg_variance_for_valuation_vgp.shape}")


    # 4. SVGP Model Training
    logger.info("=== Stage 4: SVGP Model Training ===")
    # Choose num_inducing_points based on data size, e.g., min(200, X_train_scaled.shape[0] // 5)
    num_ip = min(200, X_train_scaled.shape[0] // 2) if X_train_scaled.shape[0] > 10 else X_train_scaled.shape[0]
    if num_ip == 0 and X_train_scaled.shape[0] > 0 : num_ip = X_train_scaled.shape[0] # ensure num_ip > 0 if data exists
    if num_ip == 0:
        logger.warning("Cannot train SVGP, not enough data for inducing points after split.")
    else:
        svgp_model = SVGPXGClassifier(input_dim=input_dim, num_inducing_points=num_ip)
        svgp_model.train(X_train_scaled, y_train.values, training_iterations=100) 
        svgp_model_path = os.path.join(SAVED_MODELS_DIR, 'svgp_xg_classifier')
        svgp_model.save(svgp_model_path)
        logger.info(f"SVGP model saved to {svgp_model_path}")
        svgp_model.get_ard_lengthscales(feature_names=features_names)
   

    #  5. Player Valuation (using VGP predictions)
    logger.info("=== Stage 5: Player Valuation (using VGP) ===")
    # Ensure 'player' and 'goal' columns are present in model_vars_df for valuation
    if 'player' not in features_df.columns or 'goal' not in features_df.columns:
        logger.error("'player' or 'goal' column missing in model_vars_df. Cannot proceed with player valuation.")
        return
 
    player_valuation_input_df = features_df.copy()
    player_valuation_input_df['xG_mean'] = xg_mean_for_valuation_vgp
    player_valuation_input_df['xG_variance'] = xg_variance_for_valuation_vgp
    logger.info(f"Prepared player valuation input DataFrame with shape: {player_valuation_input_df.shape}")

    player_calculator = PlayerValueCalculator(player_valuation_input_df)

    min_shots_val = 10 # Threshold for player valuation summary
    player_summary = player_calculator.calculate_player_summary(min_shots_threshold=min_shots_val)
    if not player_summary.empty:
        player_summary.to_csv(os.path.join(PLAYER_VALUES_RESULTS_DIR, 'vgp_player_summary.csv'), index=False)
        logger.info(f"VGP Player summary saved. Top 3 players:\n{player_summary.head(3)}")
        player_calculator.plot_player_average_value(player_summary, top_n=15, show_plot=False, 
                                                       save_path=os.path.join(PLOTS_DIR, 'vgp_player_value.png'))
    else:
        logger.warning(f"Player summary (VGP) was empty for min_shots_threshold={min_shots_val}.")

    soft_weighted_values = player_calculator.calculate_soft_weighted_value()
    if not soft_weighted_values.empty:
        soft_weighted_values.to_csv(os.path.join(PLAYER_VALUES_RESULTS_DIR, 'vgp_soft_weighted_values.csv'), index=False)
        logger.info(f"VGP Soft-weighted player values saved. Top 3 players:\n{soft_weighted_values.head(3)}")

    #  6. Summary Plots
    xg_dist_plotter = XGDistributionPlots()
    xg_dist_plotter.plot_xg_distribution(xg_mean_for_valuation_vgp, xg_variance_for_valuation_vgp, 
                                           show_plot=False, save_path=os.path.join(PLOTS_DIR, 'vgp_xg_distribution.png'))
    xg_pitch_map_plotter = XGPitchMapPlots()
    xg_pitch_map_plotter.plot_pitch_map(x_coords=player_valuation_input_df['x0_event'], 
                                        y_coords=player_valuation_input_df['y0_event'], 
                                        intensity_values=xg_mean_for_valuation_vgp,
                                        colorbar_label='xG Mean',
                                        show_plot=False, save_path=os.path.join(PLOTS_DIR, 'vgp_xg_mean_pitch_map.png'))
    
    xg_pitch_map_plotter.plot_pitch_map(x_coords=player_valuation_input_df['x0_event'], 
                                        y_coords=player_valuation_input_df['y0_event'], 
                                        intensity_values=xg_variance_for_valuation_vgp,
                                        colorbar_label='xG Variance',
                                        show_plot=False, save_path=os.path.join(PLOTS_DIR, 'vgp_xg_variance_pitch_map.png'))


    logger.info("VGP xG distribution plot saved.")

    logger.info("Pipeline finished successfully.")

if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:
        logger.critical(f"Pipeline failed with error: {e}", exc_info=True)
