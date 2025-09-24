
## Audio Engagement Duration Prediction
- The goal was to predict how long a user would listen to an audio episode, measured in minutes.

## What I did 
- Built an XGBoost regression model with a 7-fold ensemble to improve performance.
- Engineered features using: Label encoding and target encoding , Non-linear transformations, Multi-column combinations to capture listening patterns
- Tuned hyperparameters with Optuna and evaluated using K-Fold cross-validation.
- Achieved a Root Mean Squared Error (RMSE) of 11.753 minutes.
## Tech Stack
- Python, XGBoost, Optuna, scikit-learn, pandas, numpy