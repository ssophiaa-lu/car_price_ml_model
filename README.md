# car_price_ml_model

This project trains and evaluates a machine learning model to predict car selling prices using the CarDekho dataset from Kaggle. It demonstrates a full ML workflow: data loading, preprocessing, model training, and performance evaluation.

Key steps in cars.py:
1) Load dataset (cardekho.csv) with pandas
2) Define features & target
  Target: selling_price
  Features: numerical columns such as year, km_driven, seats, etc.
  Dropped: text/categorical fields (name, fuel, seller_type, transmission, owner, max_power)
3) Train/test split (80/20)
4) Model: RandomForestRegressor (scikit-learn) with fixed random_state for reproducibility
5) Evaluation metrics:
  Mean Absolute Error (MAE)
  Mean Squared Error (MSE)
  R^2 Score (variance explained)
  Custom accuracy: % of predictions within ±10% of actual selling price
Outputs: Metrics are printed to the console when you run the script.
  
---

## Installation
This project requires `pandas`, `numpy`, and `scikit-learn`.  
  > Install with `pip install pandas numpy scikit-learn`.
> 


## License
Copyright © 2025 Sophia Lu
