# COMP 540 Project: Prediction of Oil and Gas Production Performance in Unconventional Reservoirs of the Permian Basin

## Building Final Model
To run a cross validation of the best model, use demo.ipynb. This imports the pre-processed dataset and runs a cross validation of a LightGBM model with the best selected hyperparameters. The average RMSE, normalized RMSE, and adjusted $R^2$ will be printed. 

Dependencies can be installed using the environment.yml file. The filtered and imputed dataset used to train and test the model is stored in data/imputed.csv.

## Code Guide
- Dataset preprocessing is found in data_preparation.ipynb
- Exploratory data analysis is found in filtered_data_exploration.ipynb and initial_data_exploration.ipynb.
- Unsupervised learning analysis is found in clustering.ipynb
- Code for the standard regression models is found in base_models.ipynb
- Code for the neural network is found in neural_network.ipynb.
- Code for the spatial regression model is found in spatial_model.ipynb
- Code to run a cross-validation for spatial and non-spatial models is found in util.py
- Code to analyze the best model and create relevant graphs is found model_analysis.ipynb
- Demo for the best model is found in demo.ipynb.
- Additional code for experiments with another variant of the spatial random forest algorithm in R is in the r_spatial_regression folder.

## Data Guide
- Initial unfiltered dataset is found Delaware_Wells.csv
- Dataset after first round of filtering features is found in filtered_1.csv
- Dataset after second round of filtering features is found in filtered_2.csv
- Final filtered and imputed dataset is found in imputed.csv