# Fraud Detection Project

Using an artificial fraud dataset, this project aims to produce a classifier capable of working with severe class imbalances whilst, critically, maintaining the model transparency necessary for such a use case. The project will begin with a conventional and comprehensive approach to modelling before concluding with a brief exploration of Bayesian approaches in  PyMC. Through the inclusion of Bayesian modelling the project hopes to gain from the incorporation of prior knowledge and the ability to express the level of uncertainty in the model's parameters. Regarding prior elicitation, this project will rely on the base modelling to inform the process but, in reality, expertise could be leveraged, which seems an attractive prospect in the case of fraud detection.   

## Project Motivation

- Find fraud and the challenge of extreme class imbalance a very interesting scenario and wanted to test my ability to address this problem.
- Wanted a project that would require me to demonstrate a range of skills.
- Keen to extend my practical experience of implementing Bayesian approaches in PyMC.

## Learning Points

- Handling extreme target class imbalances.
- Prior elicitation via insights from base modelling. 
- Bayesian modelling in PyMC.

## Project Breakdown

### Initial analysis

- Reduced dataset significantly to avoid prohibitive compute.
- Checked level of class imbalance and that minority class still present after dataset reduction.
<img src="https://github.com/mrmarq1/fraud-detection/assets/126958930/92bcc3dc-e376-44fe-bc7e-5e1eae79c554" width="300">
  
- Converted 'object' dtypes to 'string' to enable subsequent handling. 

### Data encoding

- One-hot encoded 'type' as no ordinality observed - avoided n-1 encoding as planned to use tree-based algorithms and was concerned about feature visibility.
- Removed the 'C' prefix from 'nameOrig'.
- Removed letter prefixes from 'nameDest' and one-hot encoded them so information retained.

### EDA

#### Univariate analysis
- Continuous features: Applied Yeo-Johnson to selected variables due to an extreme right skew. Power transformation handled outliers and exposes Gaussian-esque distributions in spite of continued zero inflation.
- Discrete features: largely balanced payment destinations, a predominance of short 1 hr transactions and a strong skew in favour of 'payment' for the 'type' encodings.

#### Bivariate analysis
- Continuous features: no discriminatory patterns with respect to fraud cases; linear relationships indicated between 'oldbalanceOrg' and 'newbalanceOrig'.
- Discrete features: fraud cases associated with all lengths of transactions ('step') but specific to the payment types 'type_CASH_OUT' and 'type_TRANSFER', and the destination type 'nameDestLabel_C'.

#### Multivariate analysis
- Parallel Coordinates analysis:
  - Most relevant features: 'newbalanceOrig','oldbalanceOrg','type_TRANSFER', 'type_CASH_OUT', 'newDestLabel_C'.
  - All fraud cases result in lowest range of 'newbalanceOrig' regardless of starting account balance and are associated with payment destinations of type 'customer'.
  - Pattern 1: Some fraud cases fully discriminated by payments to 'customer' destinations with starting and ending recipient account balances of approx. <500k and 1M+ respectively.
<img src="https://github.com/mrmarq1/fraud-detection/assets/126958930/2f4f5d63-b082-4e51-a2da-3c3e5f9a4903" width="300">

  - Pattern 2: Majority of fraud cases observed at lowest values for 'oldbalanceOrg' as well as 'newbalanceOrig'; however, additional features unable to discriminate fraud from non-fraud cases in this range.        
<img src="https://github.com/mrmarq1/fraud-detection/assets/126958930/7813552d-4875-4010-ae80-82509cb9f2ee" width="300">

#### Modelling and tuning
- Performance metric for Optuna tuning: Precision-recall (PR) chosen due to the extreme class imbalance (0.87% minority) and desire to gain insight into model's ability to predict fraud cases as threshold values are changed.
- SMOTE: Optuna employed initially to optimise transformation based on PR score.
- Initial modelling: Using default parameter values, assessed performance of models (logistic regression, SVC, decision tree, random forest, xgboost, catboost and lightgbm).
- Most performant models: random forest, xgboost, lightgbm and catboost.
- Hyperparameter tuning of most perfromant models: Optuna used again to assess performance of ensemble models over a wide range of hyperparameter values.
- Final model: Random Forest model consistently most performant with greatest and most harmonious metrics, including PR score.
<img src="https://github.com/mrmarq1/fraud-detection/assets/126958930/9cddc86a-9c24-4bbb-99c6-0ad1827e631b" width="300">


#### Model analysis
- Random Forest feature importances calculated and SHAP analysis used to gain even greater insight into relationships between feature space and target.
<img src="https://github.com/mrmarq1/fraud-detection/assets/126958930/bc7626fa-1b67-4ea3-b1b7-5ef62c5d62e3" width="300">

- Both analyses emphasised importnace of pre and post balance metrics along with destination type and type_CASH_OUT in terms of fraud cases, as per initial insights from the parallel coordinates analysis.

#### Bayesian modelling
- As Random Forest model not inherently probabilistic, leveraged a GLM approach. A linear predictor was engineered from the dot product between the vector, beta, of the previously calculated feature importances and the design matrix, X_array. Subsequently, the inverse logit function was applied providing the required probabilities for the likelihood function. Together with the fraud occurrence prior, set at a conservative 1%, the feature posteriors were generated.
- Using both sets of feature importances, 'oldbalanceOrg', 'type_CASH_OUT', 'nameDestLabel_C' and 'type_TRANSFER' were consistently identified as positively associated with the occurrence of fraud.





