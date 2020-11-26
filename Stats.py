import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from math import sqrt
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')
from datetime import timedelta


class RegressionModel:

    def __init__(self, data_file="/Users/kyleweber/Desktop/Python Scripts/Ankle Model/Epoch15SVMData.xlsx"):

        self.data = pd.read_excel(data_file)
        self.slr_model = None
        self.flr_model = None
        self.mlr_model = None

        self.df_rmse = pd.DataFrame([], index=["Linear", "Mixed", "MixedRandomSlope"], columns=["RMSE"])

        self.normalize_data()

    def normalize_data(self):

        self.data["Speed_Normalized"] = preprocessing.scale(self.data["Speed"])
        self.data["Counts_Normalized"] = preprocessing.scale(self.data["Counts"])

    def scale_counts(self, factor=10):

        self.data["Counts_Scaled"] = self.data["Counts"] / factor

    def slr(self, iv, dv, plot_relationship=False, plot_residuals=True):

        # Create simple linear regression model
        self.slr_model = LinearRegression(fit_intercept=True)
        y = self.data[dv]
        x = self.data[iv]

        self.slr_model.fit(x[:, np.newaxis], y)

        xfit = np.linspace(-4, 4, 1000)
        yfit = self.slr_model.predict(xfit[:, np.newaxis])

        if plot_relationship:
            sns.lmplot(x=iv, y=dv, data=self.data, height=7, aspect=1.25)
            plt.plot(xfit, yfit)
            plt.ylabel(dv)
            plt.xlabel(iv)
            plt.title("{} = {} • {} + {}".format(dv, round(self.slr_model.coef_[0], 5), iv,
                                               round(self.slr_model.intercept_, 5)))
            plt.subplots_adjust(left=.095, right=.95, top=.9, bottom=.15)
            plt.xlim(-100, max(self.data["Counts"])*1.1)

        if plot_residuals:
            from yellowbrick.regressor import ResidualsPlot

            # Instantiate the linear model and visualizer
            visualizer = ResidualsPlot(model=self.slr_model)

            visualizer.fit(x[:, np.newaxis], y)  # Fit the training data to the model
            visualizer.poof()

        print("Simple Linear Regression\n{} = {} • {} + {}".format(dv, round(self.slr_model.coef_[0], 5), iv,
                                                                   round(self.slr_model.intercept_, 5)))

        # Predicts RMSE
        y_predict = self.slr_model.predict(x.values.reshape(-1, 1))
        rmse = sqrt(((y - y_predict) ** 2).values.mean())

        self.df_rmse.loc["Linear"] = round(rmse, 5)
        print("\n", self.df_rmse)

    def mlr_rand_int(self, iv, dv, mixed_effect, show_plot=False):
        """Mixed effects model where groups have same slope but different intercepts."""

        self.flr_model = smf.mixedlm("{} ~ {}".format(dv, iv), self.data, groups=self.data[mixed_effect])
        fdf = self.flr_model.fit()
        flr_summary = fdf.summary()
        print(flr_summary)

        # Checking residuals
        df_perf = pd.DataFrame()
        df_perf['Residuals'] = fdf.resid.values
        df_perf[iv] = self.data[iv]
        df_perf["Predicted"] = fdf.fittedvalues

        if show_plot:
            sns.lmplot(x="Predicted", y="Residuals", data=df_perf)
            plt.title("Random Intercept: {} ~ {}".format(dv, iv))
            plt.subplots_adjust(top=.921)
            plt.xlabel("Predicted {}".format(dv))
            plt.xlim(0, max(df_perf["Predicted"])*1.05)

        y = self.data[dv]
        x = self.data[iv]

        y_predict = fdf.fittedvalues
        rmse = sqrt(((y - y_predict) ** 2).values.mean())
        self.df_rmse.loc["Mixed"] = round(rmse, 5)
        print(self.df_rmse)

    def mlr_rand_slope(self, iv, dv, mixed_effect, show_plot=True):
        """Mixed effect where each group has own slope and own intercept"""

        mm = smf.mixedlm("{} ~ {}".format(dv, iv), self.data,
                         groups=self.data[mixed_effect], re_formula="~{}".format(iv))
        mdf = mm.fit()
        print(mdf.summary())

        df_perf = pd.DataFrame()
        df_perf['Residuals'] = mdf.resid.values
        df_perf[iv] = self.data[iv]
        df_perf["Predicted"] = mdf.fittedvalues

        y = self.data[dv]
        x = self.data[iv]

        y_predict = mdf.fittedvalues
        rmse = sqrt(((y - y_predict) ** 2).values.mean())
        self.df_rmse.loc["MixedRandomSlope"] = round(rmse, 5)

        if show_plot:

            sns.lmplot(x="Predicted", y="Residuals", data=df_perf)
            plt.title("Random Intercept and Slope: {} ~ {}".format(dv, iv))
            plt.subplots_adjust(top=.921)
            plt.xlabel("Predicted {}".format(dv))
            plt.xlim(0, max(df_perf["Predicted"])*1.05)


# data = RegressionModel()
# data.scale_counts(100)
# data.slr(iv="Counts", dv="Speed", plot_residuals=False, plot_relationship=True)
# data.mlr_rand_int(iv="Counts_Scaled", dv="Speed", mixed_effect='ID', show_plot=False)
# data.mlr_rand_slope(iv='Counts_Scaled', dv='Speed', mixed_effect='ID', show_plot=False)

class Data:

    def __init__(self, timestamp_file=None, accelerometer_file=None):

        self.df_walks = pd.read_excel(timestamp_file)
        self.df_walks = self.df_walks.loc[self.df_walks["ID"] ==
                                                  int(accelerometer_file.split(".")[0].split("/")[-1].split("_")[2])]
        self.df_walks["Start"] = pd.to_datetime(self.df_walks["Start"])
        self.df_walks["End"] = pd.to_datetime(self.df_walks["End"])

        self.df_epoch = pd.read_csv(accelerometer_file, usecols=["Timestamp", "AnkleAVM", "AnkleSVM"])
        self.df_epoch["Timestamp"] = pd.to_datetime(self.df_epoch["Timestamp"])

        # ============================================= RUNS METHODS ==================================================
        self.mark_walks()

        self.df_epoch_crop = self.crop_df()

    def crop_df(self):

        df = self.df_epoch.loc[(self.df_epoch["Timestamp"] >=
                                self.df_walks.iloc[0]["Start"] + timedelta(seconds=-120)) &
                               (self.df_epoch["Timestamp"] <=
                                self.df_walks.iloc[-1]["End"] + timedelta(seconds=120))]

        return df

    def mark_walks(self):

        walk_list = np.zeros(self.df_epoch.shape[0])

        for row in self.df_walks.itertuples():
            data = self.df_epoch.loc[(self.df_epoch["Timestamp"] >= row.Start) &
                                          (self.df_epoch["Timestamp"] <= row.End)]

            walk_list[data.index[0]:data.index[-1]] = [row.Walk for i in range(data.index[-1] - data.index[0])]

        self.df_epoch["WalkMarker"] = walk_list

    def calculate_means(self):

        svm = []
        avm = []
        for walk in [1, 2, 3, 4, 5]:
            df = self.df_epoch.loc[self.df_epoch["WalkMarker"] == walk]
            svm.append(df["AnkleSVM"].sum())
            avm.append(df["AnkleAVM"].mean())

        self.df_walks["SVM"] = svm
        self.df_walks["AVM"] = avm

    def show_boxplot(self, accel_var):

        self.df_epoch.loc[self.df_epoch["WalkMarker"] > 0].boxplot(by="WalkMarker", column=[accel_var])
        plt.xlabel("Walk Number")
        plt.ylabel("Counts")


x = Data(timestamp_file="/Users/kyleweber/Desktop/TreadmillWalkTimes.xlsx",
         accelerometer_file="/Users/kyleweber/Desktop/OND07_WTL_3032_EpochedAccelerometer.csv")
x.calculate_means()
x.show_boxplot("AnkleSVM")