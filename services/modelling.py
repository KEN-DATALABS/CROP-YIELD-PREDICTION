from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
import numpy as np
import pandas as pd
import random

# Refactored run_all_transfer_experiments
def run_all_transfer_experiments(models_config, X_train_final, y_train_final, X_adapt, y_adapt, X_test_unseen, y_test_unseen, X_test_unseen_final, y_test_unseen_final, sample_weights=None):
    results = []
    transfer_models = {}
    
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    for model_name, source_model, baseline_model in models_config:
        # No-transfer baseline
        baseline_model.fit(X_train_final, y_train_final, sample_weight=sample_weights)
        y_pred_no_transfer = baseline_model.predict(X_test_unseen)
        no_transfer_r2 = round(r2_score(y_test_unseen, y_pred_no_transfer), 4)
        no_transfer_mae = round(mean_absolute_error(y_test_unseen, y_pred_no_transfer), 2)
        no_transfer_mape = round(mean_absolute_percentage_error(y_test_unseen, y_pred_no_transfer) * 100, 2)
        
        # Transfer learning: Use source model predictions as a feature
        X_adapt_transfer = X_adapt.copy()
        X_test_transfer = X_test_unseen_final.copy()
        X_adapt_transfer['source_pred'] = source_model.predict(X_adapt)
        X_test_transfer['source_pred'] = source_model.predict(X_test_unseen_final)
        
        # Train transfer model (no sample weights for transfer step, as per run_transfer_experiment)
        transfer_model = baseline_model  # Use the same fitted model instance
        transfer_model.fit(X_adapt_transfer, y_adapt)  # No sample_weight here
        y_pred_transfer = transfer_model.predict(X_test_transfer)
        transfer_r2 = round(r2_score(y_test_unseen_final, y_pred_transfer), 4)
        transfer_mae = round(mean_absolute_error(y_test_unseen_final, y_pred_transfer), 2)
        transfer_mape = round(mean_absolute_percentage_error(y_test_unseen_final, y_pred_transfer) * 100, 2)
        
        # Calculate improvements
        r2_improvement_pp = (transfer_r2 - no_transfer_r2) * 100
        r2_improvement_pct = ((transfer_r2 - no_transfer_r2) / abs(no_transfer_r2)) * 100 if no_transfer_r2 != 0 else 0
        mae_improvement = no_transfer_mae - transfer_mae
        mae_reduction_pct = (mae_improvement / no_transfer_mae) * 100 if no_transfer_mae != 0 else 0
        mape_improvement = no_transfer_mape - transfer_mape
        mape_reduction_pct = (mape_improvement / no_transfer_mape) * 100 if no_transfer_mape != 0 else 0
        
        results.append({
            'Model': model_name,
            'No Transfer R²': no_transfer_r2,
            'Transfer R²': transfer_r2,
            'R² Improvement (pp)': round(r2_improvement_pp, 2),
            'R² Improvement (%)': round(r2_improvement_pct, 2),
            'No Transfer MAE': no_transfer_mae,
            'Transfer MAE': transfer_mae,
            'MAE Improvement': round(mae_improvement, 2),
            'MAE Reduction (%)': round(mae_reduction_pct, 2),
            'No Transfer MAPE (%)': no_transfer_mape,
            'Transfer MAPE (%)': transfer_mape,
            'MAPE Improvement (%)': round(mape_improvement, 2),
            'MAPE Reduction (%)': round(mape_reduction_pct, 2)
        })
        transfer_models[model_name] = transfer_model
    
    return pd.DataFrame(results), transfer_models
