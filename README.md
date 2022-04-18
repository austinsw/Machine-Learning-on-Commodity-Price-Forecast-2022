# Machine-Learning-on-Commodity-Price-Forecast

There are total 8 programming files in the repository. The final submitting files are ML5_Forecasting_20220409.ipynb and Webapp_20220410.ipynb. The others are past versions and prelimianry tests of models, only serving the purposes of proof of concepts and workability, therefore only completed to a certain extent.

The files in this repository listed in descending chronological order:
1. Webapp_20220410.ipynb
2. ML5_Forecasting_20220409.ipynb
3. ML4_20220402.ipynb
4. ML3_3models.ipynb
5. ML2_gold_test.ipynb
6. SVR Price Predict.ipynb
7. RNN-2.ipynb
8. RNN-1.ipynb

Unless otherwise specified, to run the programs, one could download the files and open them on Google Colab. Then click Runtime > Run all.

## ML5

ML5_Forecasting_20220409.ipynb is the final machine learning model built. 

The imported libraries versions are:
- Python version: 3.7.13 
- NumPy version: 1.21.5
- Pandas version: 1.3.5
- Matplotlib version: 3.2.2
- Seaborn version: 0.11.2
- Scikit-learn version: 1.0.2
- XGBoost version: 0.90

It includes pd.read from IndexMundi to pull commodity price data online. Data transformation was done and some graphs and charts were produced to display to statistical information of the dataset, including the correlation heatmap, density distriubtion, and boxplot. DecisionTreeRegressor, RandomForestRegressor, XGBRegressor and LinearRegression were tested using GridSearchCV to determine the model with the best accuracy score. **Under the "Price Predictions by Combined Dataset" section, one can manually change the target into oil_price, coal_price, gas_price, sugar_price, ore_price, or copper_price to forecast the designated commodity.** The latest 4 months are shielded from the machine so it can be used to compare against the predicted results.

## Web App

Webapp_20220410.ipynb is created by the request of this project to be an interactive interface and web app to display the forecast results. The web app is created by streamlit. After running thy ipynb file, at the last output section, one could click the url link ended with **.loca.lt**, and it would open a new page. 

<img width="445" alt="loca lt" src="https://user-images.githubusercontent.com/42607409/163803961-89933665-ccdc-4915-9992-aed562d228c2.png">

The new page is a local tunnel. To actually access the web app, further click **Click to Continue**.

<img width="142" alt="click to continue2" src="https://user-images.githubusercontent.com/42607409/163804286-c3d86610-c3fe-4c87-81b0-598b1a2b4e48.png">

Then the web app is accessed. One could select the commodity and the duration of months to be predicted. A rangeslider is available below the chart to allow changing duration of view and better visualization. The predicted results are actual predictions of the future for real world application, unlike ML5 which used historical data for data analysis.

## ML4

ML4_20220402.ipynb is the previous version of the ML5 model. Unlike ML5, it lacks the acutal forecast section, and only used the model to predict historical data. All 6 commodities predictions are incorporated into the code, and manully changing of the target is unncessary.

## ML3

ML3_3models.ipynb is similar to ML4, with data analysis part and GridSearchCV function to locate the best model. However due to some errors with the continuous features, the results are unreliable. Also, instead of pulling the data from the Internet, it used a pre-downloaded and modified dataset: CombinedGoldDataCleaned.csv. In this dataset, we tried to incorporate political stability index into the gold dataset, as features of our machine learning model. This notebook was originally run on Kaggle, hence in the project directory of *gold-data-test*. However, to be run on Google Colab, simply put *CombinedGoldDataCleaned.csv* in the runtime file and change the line of code from `gold_value = pd.read_csv("../input/gold-data-test/CombinedGoldDataCleaned.csv")` to `gold_value = pd.read_csv("CombinedGoldDataCleaned.csv")`.

## ML2

ML2_gold_test.ipynb is an attempt in adding political stability index as feature into the dataset. It contains less analysis than ML3 and only a random forest regressor, instead of testing amongst 3 regression models. It was also originall run on Kaggle, like ML3. To run on Google Colab, repeat a similar procedure as above in changing the pd.read_csv() directory.

## SVR

SVR Price Predict.ipynb marked the begining of using regression to predict price in our project. It tried to fit RBF model, linear model and polynomial model the price curve. Although the RBF model seems to be a better fit than the rest, one obvious flaw was that it was using the entire historical price data to predict this exact range of data. It was a highly questionable approach in real world applicatino of forecasting future price. To run the program, put *gold-360.csv* to the project directory. Also, this *gold-360.csv* is retrieved from [IndexMundi](https://www.indexmundi.com/commodities/?commodity=gold&months=360).

## RNN

RNN-1.ipynb and RNN-2.ipynb are two similar programs, based on RNN (recurrent neural network) and LSTM (long short-term memory). *gold-360.csv* was first manually separated into training data (*gold-360train.csv*) and testing data (*gold-360test.csv*). 5 LSTM layers were connected to make a fully connected neural network. RNN-1 and RNN-2 differs by the batch size number in transformation of X. The previous is of batch size = 6 and the latter has a batch size of 12. The programs tried to use the previous 348 months to predict the latets 12 months. It was argued that this might not be a good approach to test the model accuracy since it only tested the latest values. RNN-1 and RNN-2 were originally run on Jupyter Notebook but they were tested and could be run on Google Colab as well.

The imported libraries versions in the Jupyter Notebook are:
- Python 3.7.11
- conda 4.11.0
- sklearn 1.0.1
- keras 2.3.1
- tensorflow 2.0.0

