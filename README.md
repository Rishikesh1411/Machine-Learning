# 📚 The ML Learning Lab

<div align="center">

<!-- TODO: Add project logo (e.g., a relevant ML/education icon) -->

[![GitHub stars](https://img.shields.io/github/stars/Rishikesh1411/The-ML-Learning-Lab-?style=for-the-badge)](https://github.com/Rishikesh1411/The-ML-Learning-Lab-/stargazers)

[![GitHub forks](https://img.shields.io/github/forks/Rishikesh1411/The-ML-Learning-Lab-?style=for-the-badge)](https://github.com/Rishikesh1411/The-ML-Learning-Lab-/network)

[![GitHub issues](https://img.shields.io/github/issues/Rishikesh1411/The-ML-Learning-Lab-?style=for-the-badge)](https://github.com/Rishikesh1411/The-ML-Learning-Lab-/issues)

[![GitHub license](https://img.shields.io/github/license/Rishikesh1411/The-ML-Learning-Lab-?style=for-the-badge)](LICENSE)

**A Comprehensive Hands-on Journey into Machine Learning Concepts and Algorithms through Jupyter Notebooks.**

</div>

## 📖 Overview

The ML Learning Lab is an extensive collection of Jupyter Notebooks designed for aspiring data scientists and machine learning engineers. It provides a practical, step-by-step approach to understanding fundamental and advanced machine learning concepts, from data preprocessing and exploratory data analysis (EDA) to implementing various regression, classification, clustering, and ensemble learning algorithms. Each notebook serves as a self-contained lesson, offering code examples, theoretical explanations, and visualizations to foster a deep understanding of the topics.

## ✨ Key Learning Modules & Features

This repository covers a wide array of topics crucial for machine learning mastery:

### Data Preprocessing & Feature Engineering
-   **Data Loading & Manipulation**: Working with CSV, JSON, and SQL data formats.
-   **Exploratory Data Analysis (EDA)**: Univariate, bivariate, and multivariate analysis for data understanding, including advanced techniques like Pandas Profiling.
-   **Feature Scaling**: Implementation of Standardization (Z-score) and Normalization (Min-Max Scaling) to bring features to a comparable scale.
-   **Categorical Encoding**: Techniques such as Ordinal Encoding, Label Encoding, and One-Hot Encoding for converting categorical data into numerical formats.
-   **Discretization & Binning**: Methods like KBinsDiscretization and Binarisation for transforming continuous numerical features into discrete bins.
-   **Handling Mixed Data Types** and **Date-Time Data**: Strategies for processing heterogeneous data and time-series information.
-   **Missing Value Imputation**:
    -   Complete Case Analysis (CCA) for dropping rows with missing data.
    -   Mean, Median, and Most Frequent (Mode) Imputation for numerical and categorical features.
    -   Arbitrary Value Imputation and Missing Category Imputation.
    -   Random Sample Imputation and K-Nearest Neighbors (KNN) Imputation for more sophisticated approaches.
    -   Iterative Imputer (MICE algorithm) and Random Indicator Imputation.
    -   Automated selection of optimal imputation methods.
-   **Outlier Detection & Removal**: Techniques using Z-score, IQR method, and Percentile Capping to identify and mitigate the impact of outliers.
-   **Feature Transformation**: Application of Function Transformer and Power Transformer for adjusting data distributions.
-   **Column Transformer**: A powerful tool for applying different transformations to specific columns within a dataset.
-   **Feature Extraction**: Detailed explanation and application of Principal Component Analysis (PCA), demonstrated with step-by-step examples and its use on datasets like MNIST.

### Machine Learning Algorithms
-   **Linear Regression**: Fundamentals of Simple and Multiple Linear Regression, including a scratch implementation and visualization of best-fit lines.
-   **Polynomial Regression**: Expanding linear models to capture non-linear relationships.
-   **Regularization Techniques**: In-depth coverage of Ridge, Lasso, and Elastic Net regression, with explanations from scratch, gradient descent perspective, and practical Scikit-learn implementations.
-   **Logistic Regression**: Understanding the Perceptron Trick, Sigmoid function, and extending to Polynomial Logistic Regression for binary and multi-class classification.
-   **Softmax Regression**: An extension of logistic regression for multi-class classification problems.
-   **Decision Trees**: Comprehensive exploration of Decision Trees, including hyperparameter tuning for optimization and advanced visualization using the `dtreeviz` library.
-   **Support Vector Machines (SVM)**: A detailed demo covering the core concepts of SVM and the power of the Kernel Trick for non-linear decision boundaries.
-   **Naive Bayes Classifier**: Introduction to probabilistic classification using Naive Bayes.

### Ensemble Learning
-   **Bagging**: Demonstrations of Bagging Classifiers and Bagging Regressors for improving model stability and accuracy.
-   **Random Forest**: Detailed learning tools, practical demonstrations, comparative analysis with Bagging, understanding the Out-of-Bag (OOB) Score, and calculation of Feature Importance.
-   **Boosting**:
    -   **AdaBoost**: Demos and hyperparameter tuning for adaptive boosting.
    -   **Gradient Boosting**: Step-by-step implementation and classification demonstrations.
-   **Stacking & Blending**: Advanced ensemble methods for combining multiple models to achieve superior predictive performance.
-   **Voting Regressor**: Combining predictions from multiple regression models.

### Clustering Algorithms
-   **K-Means Clustering**: Implementation using Scikit-learn and detailed explanation of the algorithm's mechanics.
-   **DBSCAN**: Density-Based Spatial Clustering of Applications with Noise, demonstrated with datasets like Iris for identifying clusters of varying shapes and sizes.
-   **Hierarchical Clustering**: Understanding agglomerative and divisive clustering approaches.

### Optimization Techniques
-   **Gradient Descent**: Step-by-step explanations, 3D visualizations, and animations illustrating the optimization process for model parameters.
-   **Batch Gradient Descent**: Updating model parameters using the entire dataset.
-   **Mini-Batch Gradient Descent**: A balance between Batch and Stochastic Gradient Descent for efficient training.
-   **Stochastic Gradient Descent**: Animated demonstrations and implementation for updating parameters with single data points.

### Data Acquisition
-   **Web Scraping**: Techniques for programmatically gathering data from websites.
-   **Fetching Data with APIs**: Interacting with web services to retrieve data.

### Model Evaluation
-   **Regression Metrics**: Comprehensive understanding of metrics such as R-squared, Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).
-   **Classification Metrics**: Evaluation using Accuracy, Precision, Recall, F1-score, and Confusion Matrix.

## 🛠️ Tech Stack

This learning lab primarily utilizes Python and its rich ecosystem of data science libraries, offering a robust environment for machine learning experimentation and education.

**Core Technologies:**
-   ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
-   ![Jupyter Notebook](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
-   ![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black) (Implied OS for many data science environments)

**Key Libraries:**
-   ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white) (Data manipulation and analysis)
-   ![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white) (Numerical computing)
-   ![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white) (Machine learning algorithms)
-   ![Matplotlib](https://img.shields.io/badge/Matplotlib-FF7200?style=for-the-badge&logo=matplotlib&logoColor=white) (Data visualization)
-   ![Seaborn](https://img.shields.io/badge/Seaborn-30A8D0?style=for-the-badge&logo=seaborn&logoColor=white) (Statistical data visualization)
-   ![Plotly](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white) (Interactive visualizations, especially for 3D animations)
-   ![dtreeviz](https://img.shields.io/badge/dtreeviz-2E8B57?style=for-the-badge) (Decision tree visualization)
-   ![Requests](https://img.shields.io/badge/Requests-105193?style=for-the-badge&logo=python-requests&logoColor=white) (HTTP library for APIs)
-   ![BeautifulSoup](https://img.shields.io/badge/BeautifulSoup-0D1117?style=for-the-badge&logo=html5&logoColor=white) (Web scraping)
-   ![IPyWidgets](https://img.shields.io/badge/IPyWidgets-4285F4?style=for-the-badge&logo=jupyter&logoColor=white) (Interactive controls for notebooks)

## 🚀 Quick Start

To set up and explore the notebooks in this repository, follow these steps:

### Prerequisites
-   **Python 3.x** (version 3.8 or higher recommended)
-   **pip** (Python package installer, usually comes with Python)
-   **Git** (for cloning the repository)

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/Rishikesh1411/The-ML-Learning-Lab-.git
    cd The-ML-Learning-Lab-
    ```

2.  **Create a virtual environment** (highly recommended for dependency management)
    ```bash
    python -m venv venv
    # Activate the virtual environment:
    # On Windows:
    # .\venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```

3.  **Install dependencies**
    Install the core data science libraries and tools required for the notebooks.
    ```bash
    pip install jupyterlab pandas numpy scikit-learn matplotlib seaborn plotly dtreeviz requests beautifulsoup4 ipywidgets
    ```
    *(Note: `dtreeviz` typically requires `graphviz` to be installed on your system for rendering decision tree visualizations. If you encounter errors related to `dot` or `graphviz`, you may need to install it separately via your system's package manager, e.g., `sudo apt-get install graphviz` on Debian/Ubuntu, `brew install graphviz` on macOS.)*

4.  **Start Jupyter Notebook or Jupyter Lab**
    Once all dependencies are installed, you can launch the Jupyter environment:
    ```bash
    jupyter notebook
    # or, for the newer interface:
    jupyter lab
    ```

5.  **Open a notebook**
    Your web browser will automatically open to the Jupyter interface (usually `http://localhost:8888`). From there, navigate to any `.ipynb` file in the repository's root directory and click on it to open and start interacting with the learning modules.

## 📁 Project Structure

The repository maintains a flat structure, with each Jupyter Notebook file (`.ipynb`) serving as an individual learning module dedicated to a specific machine learning concept or algorithm.

```
The-ML-Learning-Lab-/
├── 14.working_with_csv.ipynb
├── 19_understanding_your_data.py.ipynb
├── 20_Eda_using_univeriate_analysis.ipynb
├── 21_Eda_using_biveriate_analysis_multiveriate.py.ipynb
├── 22_pandas_profiling.ipynb
├── 26ordinal_label__encode_categorical_data.ipynb
├── 32_KBinsDESCRETIZATION_BINNING.ipynb
├── 36.arbitary_value_imputation.ipynb
├── 56_gradient-descent-3d.ipynb
├── 56_gradient-descent-animation(both-m-and-b).ipynb
├── 56_gradient-descent-animation(onlyb).ipynb
├── Batch_gradient_descent.ipynb
├── DBSCAN_IRIS.ipynb
├── LICENSE
├── Lec_101_AGABOOST_HYPERPARAMETER.ipynb
├── Lec_104_k_mean_clustring_sklearn.ipynb
├── Lec_105_k_mean_CLUSTRING.ipynb
├── Lec_106_Gradient_boosting_step_by_step.ipynb
├── Lec_108_gradient_boosting_classifiaction_demo.ipynb
├── Lec_89_bagging_classifier.ipynb
├── Lec_90_bagging_regressoor.ipynb
├── Lec_91_Random_Forest_Learning_tool.ipynb
├── Lec_92_Random_Forest_Demo.ipynb
├── Lec_93_Baggiing_vs_Random_Forest.ipynb
├── Lec_95_Code_Example_Random_forest.ipynb
├── Lec_96_OOB_Score_Demo.ipynb
├── Lec_97_How_Feature_importance_calculated_dt_rf.ipynb
├── README.md
├── classification_placement_data.ipynb
├── day18_gather_with_web_scraping.ipynb
├── day24 standarisation.ipynb
├── day25 normalization.ipynb
├── day27_one_hot_encoding.ipynb
├── day28_column_transform.ipynb
├── day30_function_transformer.ipynb
├── day31_power_transformer.ipynb
├── day39_k_nearest_neighbour(knn)_to_hancle_multivariate_missing_value.ipynb
├── day_32Binarisation.ipynb
├── day_33_handling_with_mixed_data.ipynb
├── day_34_handle_with_date_time_data.ipynb
├── day_35_cca_handle_mising_value_remove_row.ipynb
├── day_36_handling_missing_value_numerical_data_mean_median_imputation.ipynb
├── day_37_most_frequent_value_imputation_mode.ipynb
├── day_37missing-category-imputation.ipynb
├── day_38_automatically-select-imputer-parameters.ipynb
├── day_38_automatically_select_imputation_method_or_parameters.ipynb
├── day_38_random_indicator.ipynb
├── day_38_random_sample_imputation_on_numerical_data.ipynb
├── day_40I_iterative_im puter_mice_algorithmstep-by-step.ipynb
├── day_42_outlier_detection_removal_using_z_score_standardization.ipynb
├── day_43_outlier_detection_removal_using_iqr_method.ipynb
├── day_44_percentile_to_handle_remove_outliers_from_dataset.ipynb
├── day_48_pca_step_by_step _feature_extraction.ipynb
├── day_49_pca_on_mnist_dataset.ipynb
├── dbscan_campusx.ipynb
├── fetching_with_api.ipynb
├── lec_100_adaboost_demo.ipynb
├── lec_109_stacking_blending.ipynb
├── lec_110_hirerchical_clusting.ipynb
├── lec_112_linear-regression-assumptions.ipynb
├── lec_116_kernal_trick_svm.ipynb
├── lec_116_svm_demo.ipynb
├── lec_125_Naive_Bays_example.ipynb
├── lec_50_linear_regression_ml_algorithm.ipynb
├── lec_51_linear_regressiopn_best_curve_fit_line.ipynb
├── lec_53_multiple_linear_regression.ipynb
├── lec_55_multiple_linear_regression_code_from_scratch_step_by_step.ipynb
├── lec_56_gradient_descent_class_step_by_step.ipynb
├── lec_56_gradient_descent_step_by_step.ipynb
├── lec_60_polynomial_regression.ipynb
├── lec_62_regularisation_sklearn.ipynb
├── lec_63_ridge_regression_from_scratch_m_and_b.ipynb
├── lec_63_ridge_regulization_for_multiple_independent_veriable.ipynb
├── lec_64_ridge _regression_gradient_descent.ipynb
├── lec_65_5_key_understanding_ridge_regression_full_explanation.ipynb
├── lec_66_lasso_regression_sklearn.ipynb
├── lec_66_lasso_regression_sklearn_deep.ipynb
├── lec_68_elstic_net_regularisation.ipynb
├── lec_70_perceptron-trick_campusx_logistic.ipynb
├── lec_71_perceptron-trick-sigmoid.ipynb
├── lec_74_gradient-descent.ipynb
├── lec_75_classificatio_metrics.ipynb
├── lec_77_softmax-demo.ipynb
├── lec_78_polynomial-logistic-regression.ipynb
├── lec_78_polynomial_logistic_regression.ipynb
├── lec_81_decision_tree_hyperparameter.ipynb
├── lec_82_Decision Tree Classification Demo.ipynb
├── lec_83_dtreeviz_lib.ipynb
├── lec_87_voting_regressor.ipynb
├── lec_no_52_regression_metrices.ipynb
├── mini-batch-gradient-descent-from-scratch.ipynb
├── stochastic-gradient-descent-animation.ipynb
├── stochastic_gd.ipynb
└── working_with_json_sql.ipynb
```

## 📚 Resources & Learning Path

Each Jupyter Notebook in this repository is meticulously crafted to be self-explanatory, guiding you through the concepts with clear explanations and code examples. It is recommended to approach the topics systematically, ideally starting with data preprocessing fundamentals before delving into machine learning algorithms and advanced techniques.

## 🤝 Contributing

We welcome contributions to expand and enhance this learning lab! If you have additional notebooks, improvements to existing content, bug fixes, or new ideas, please consider contributing:

1.  **Fork the repository**.
2.  Create a new branch for your feature or fix (e.g., `feature/new-topic` or `fix/bug-in-notebook`).
3.  Add your changes, ensuring notebooks are clean, well-commented, and follow a consistent style.
4.  Commit your changes and push them to your forked repository.
5.  Open a Pull Request to the `main` branch of this repository with a clear description of your contributions.

## 📄 License

This project is licensed under the [MIT License](LICENSE) - see the [LICENSE](LICENSE) file for full details.

## 🙏 Acknowledgments

-   This learning lab is built upon the incredible work of the Python data science community and its extensive libraries.
-   Inspired by various educational resources, courses, and tutorials in the field of Machine Learning.
-   Special thanks to [Rishikesh1411](https://github.com/Rishikesh1411) for curating and maintaining this comprehensive collection of notebooks.

## 📞 Support & Contact

If you encounter any issues, have questions, or require support, please feel free to:

-   Report issues or suggest improvements via [GitHub Issues](https://github.com/Rishikesh1411/The-ML-Learning-Lab-/issues).

---

<div align="center">

**⭐ Star this repo if you find it helpful for your ML journey!**

</div>
```

