from flask import Flask, render_template, abort

app = Flask(__name__)

@app.route("/")
def main():
    return render_template('Login.html')


@app.route('/prova/<int:task_id>', methods=['GET'])
def get_task(task_id):
   !wget https://raw.githubusercontent.com/fmardero/FuturoLavoro/master/sales_data/db2.0.csv

    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    df = pd.read_csv('db2.0.csv')
    df.rename(columns={'data': 'date', 'somma_polli': 'sales', 'feste': 'holidays', 'fenomeni': 'weather'}, inplace=True)
    df.date = pd.to_datetime(df.date)

    df.info()

    df.head()

    """## Cleaning Days"""

    df.dayname.value_counts()

    """Delate Mondays"""

    df.drop(index=df.query("dayname == 'lunedì'").index, inplace=True)

    df.dayname.value_counts()

    """## AutoCorrelogram"""

    from statsmodels.graphics.tsaplots import plot_acf

    fig, ax = plt.subplots(figsize=(12, 6))
    plot_acf(df.sales.values, lags=65, ax=ax)
    plt.title('Sales Autocorrelation', fontsize=15)
    plt.show()

    """## Sales PDF"""

    sns.distplot(df.sales, rug=False)

    sns.boxplot(df.sales)

    df.query('sales < 10')

    """### Fist 100 days after opening"""

    df.iloc[:21]

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df.iloc[:100], x='date', y='sales')
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.scatter(x=df.loc[:100, 'date'], y=df.loc[:100, 'sales'])
    plt.show()

    """* Fist 20 days after opening: OK (not anomalies)
    * Days: 2015-04-05/06 (index=82,83) anomalies doe to Easter?
    """

    df.loc[80:85]

    df.query("holidays == 'Pasqua'")

    """**Easter is a disaster for sales**"""

    df.loc[439].name

    for date in df.loc[[439, 616, 632, 852], 'date']:
        print(date)

    plt.figure(figsize=(18, 6))
    sns.lineplot(data=df.loc[400:860], x='date', y='sales')
    for date in df.loc[[439, 616, 632, 852], 'date']:
        plt.vlines(x=date, ymin=0, ymax=60, color='r', lw=1.5, ls='--')
    plt.show()

    """### Bollinger Bands"""

    ma_list = []
    std_list = []
    window = [1, 5, 6, 7]

    for idx in range(len(df)):
        if idx > 6:
            ma_val = np.mean([df.iloc[idx-k].sales for k in window])
            std_val = np.std([df.iloc[idx-k].sales for k in window])
        else:
            ma_val = np.nan
            std_val = np.nan
        ma_list.append(ma_val)
        std_list.append(std_val)

    df['ma_4'] = ma_list
    df['ma_std'] = std_list

    df['ma-3std'] = df.ma_4 - 2*df.ma_std
    df['ma+3std'] = df.ma_4 + 2*df.ma_std

    plt.figure(figsize=(18, 6))
    sns.lineplot(data=df.loc[400:860], x='date', y='sales', lw=0.5, ls='--')
    sns.lineplot(data=df.loc[400:860], x='date', y='ma_4')
    sns.lineplot(data=df.loc[400:860], x='date', y='ma-3std', color='red')
    sns.lineplot(data=df.loc[400:860], x='date', y='ma+3std', color='red')
    plt.show()

    df.tail()

    """## Percentage Variations"""

    df['sales_change'] = df.sales.pct_change(periods=1)

    sns.distplot(df.sales_change.dropna(), rug=False)

    df.query('sales_change > 1.5')

    df.loc[[671, 672, 852, 853]]

    """# Dataset"""

    dataset = df[['sales', 
                't_media',
                't_min',
                'umidità_percentuale', 
                'juventus',
                'sales_change',
                'weather',
                'holidays']].copy()

    """## Day - Month"""

    def cyclical_feature_encoding(feature, max_val):
        tmp = 2 * np.pi * feature / float(max_val)
        return pd.Series(np.sin(tmp), name=f'{feature.name}_sin'), pd.Series(np.cos(tmp), name=f'{feature.name}_cos')


    dataset['day_sin'], dataset['day_cos'] = cyclical_feature_encoding(df.date.dt.dayofweek, max_val=6)
    dataset['month_sin'], dataset['month_cos'] = cyclical_feature_encoding(df.date.dt.month, max_val=12)

    plt.figure(figsize=(6, 6))
    sns.scatterplot(x='day_sin', y='day_cos', data=dataset)
    plt.show()

    plt.figure(figsize=(6, 6))
    sns.scatterplot(x='month_sin', y='month_cos', data=dataset)
    plt.show()

    """## Teams"""

    dataset.juventus = dataset.juventus.apply(lambda x: 0 if x == '0' else 1)

    """## Lag"""

    lag_window = [1, 5, 6, 7, 11, 12, 13]
    lag_columns = ['sales', 'weather', 'juventus', 'holidays']

    for lag in lag_window:
        for col in lag_columns:
            dataset[f'lag{lag}_{col}'] = dataset[col].shift(lag)

    dataset = dataset.iloc[max(lag_window):]

    """## Holidays"""

    dataset.holidays.unique()

    dataset = pd.concat([dataset, pd.get_dummies(dataset.holidays, dummy_na=False)], axis=1)
    dataset.drop(columns=['holidays'], inplace=True)

    for lag in lag_window:
        col = f'lag{lag}_holidays'
        dataset = pd.concat([dataset, pd.get_dummies(dataset[col], dummy_na=False, prefix=f'lag{lag}')], axis=1)
        dataset.drop(columns=[col], inplace=True)

    dataset.columns.tolist()

    """## Weather"""

    df.weather.unique()

    dataset = pd.concat([dataset, pd.get_dummies(dataset.weather, drop_first=True, dummy_na=False)], axis=1)
    dataset.drop(columns=['weather'], inplace=True)

    for lag in lag_window:
        col = f'lag{lag}_weather'
        dataset = pd.concat([dataset, pd.get_dummies(dataset[col], drop_first=True, dummy_na=False, prefix=f'lag{lag}')], axis=1)
        dataset.drop(columns=[col], inplace=True)

    """## Check Dataset has no empty columns"""

    with pd.option_context('display.max_rows', -1, 'display.max_columns', 5):
        print(dataset.isnull().any(axis=0))

    """# Basic Model - Linear Regression"""

    from sklearn.linear_model import Ridge
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score, RandomizedSearchCV

    TEST_SIZE = 0.15

    X = dataset.drop(columns=['sales', 'sales_change'])
    y = dataset.sales

    split_idx = dataset.iloc[int(len(dataset)*(1-TEST_SIZE))].name

    X_train, X_test = X.loc[:split_idx], X.loc[split_idx:]
    y_train, y_test = y.loc[:split_idx], y.loc[split_idx:]

    sample_weights = [np.exp(t/float(len(y_train))) for t in range(len(y_train))]

    plt.scatter([t for t in range(len(y_train))], sample_weights)

    model = Ridge()

    hyperparameters = {'alpha': np.linspace(start=0., stop=50., num=50)}

    random_search = RandomizedSearchCV(estimator=model,
                                    param_distributions=hyperparameters,
                                    n_iter=50,
                                    cv=TimeSeriesSplit(n_splits=10),
                                    n_jobs=-1)

    random_search.fit(X_train, y_train, sample_weight=sample_weights)

    random_search.cv_results_['mean_test_score']

    random_search.best_params_

    from sklearn.model_selection import cross_val_score

    cross_val_score(estimator=model, 
                    X=X_train, 
                    y=y_train, 
                    cv=TimeSeriesSplit(n_splits=10),
                    n_jobs=-1)

    best_model = random_search.best_estimator_

    pred = np.round(best_model.predict(X_test))

    """**Observed vs Predicted**"""

    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_test, y=pred)
    plt.plot([min(y_test), max(y_test)], [min(pred), max(pred)], lw=1.5, color='red', ls='--')
    plt.xlabel('Observed Sales')
    plt.ylabel('Predicted Sales')
    plt.title('Observed vs Predicted')
    plt.show()

    """### Metrics

    **Coefficient of Determination**
    """

    from sklearn.metrics import r2_score

    r2_score(y_true=y_test, y_pred=pred)

    """**Average Percentage Error**"""

    ((y_test - pred)/y_test).abs().mean()

    """## Time-Series Observed vs Prediction

    **All time-series**
    """

    fig, ax = plt.subplots(figsize=(24, 6))
    sns.lineplot(x='date', y='sales', data=df.loc[:split_idx], color='blue', alpha=0.6, label='training', ax=ax)
    sns.lineplot(x='date', y='sales', data=df.loc[split_idx:], color='red', alpha=0.8, label='obs_test', ax=ax)
    sns.lineplot(x=df.loc[split_idx:, 'date'], y=pred, color='black', label='pred_test', ax=ax)
    plt.show()

    """**Test time-series**"""

    fig, ax = plt.subplots(figsize=(18, 6))
    sns.lineplot(x='date', y='sales', data=df.loc[split_idx:], color='red', alpha=0.8, label='obs_test', ax=ax)
    sns.lineplot(x=df.loc[split_idx:, 'date'], y=pred, color='black', label='pred_test', ax=ax)
    plt.show()

    """## Prediction Error"""

    results = pd.DataFrame({'date': df.loc[split_idx:, 'date'], 'y_true': y_test.values, 'y_pred': pred})

    results.head()

    results['residuals'] = results.y_true - results.y_pred
    results['abs_res'] = results.residuals.abs()
    results['mae'] = results.abs_res / results.y_true

    results.sort_values(by=['abs_res'], ascending=False, inplace=True)

    results = pd.merge(left=results, right=df, on='date')

    sns.distplot(results.residuals)
    plt.show()

    sns.boxplot(results.residuals)
    plt.show()

    sns.scatterplot(x='dayname', y='residuals', data=results)
    plt.axhline(0, color='black', lw=2.5, ls='--')
    plt.axhline(-10, color='black', lw=1.5, ls='--')
    plt.axhline(10, color='black', lw=1.5, ls='--')
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.barplot(x='sales', y='residuals', hue='dayname', data=results)
    plt.axhline(0, color='red', lw=2, ls='--')
    plt.show()

    results.holidays.fillna('#', inplace=True)
    sns.scatterplot(x='holidays', y='residuals', data=results)
    plt.axhline(0, color='black', lw=2.5, ls='--')
    plt.axhline(-10, color='black', lw=1.5, ls='--')
    plt.axhline(10, color='black', lw=1.5, ls='--')
    plt.xticks(rotation=90)
    plt.show()

    """### Worst Predictions"""

    worst_pred = results.query('mae > 0.10')

    worst_pred
    return 

if __name__ == "__main__":
    app.run(debug = True)