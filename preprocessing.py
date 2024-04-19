import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sweetviz as sw
import sklearn.impute as impute
from sklearn.preprocessing import MinMaxScaler
from constant import Config
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from pydantic import BaseModel, Field
from typing import List


class ModelInfo(BaseModel):
    model_name: str = Field(description="Name of model training")
    model_path: str = Field(description="Directory of model")
    categorical_tranform: str = Field(
        default=None, description="Directory of model fitting"
    )
    numerical_tranform: str = Field(
        default=None, description="Directory of model fitting"
    )
    numerical_scaler: str = Field(default=None, description="Directory of model scaler")
    target_scaler: str = Field(default=False, description="Directory of model scaler")
    numerical_feature: List[str,] = Field(default=None)
    categorical_feature: List[str,] = Field(default=None)


class Precessing:
    def __init__(
        self,
        # file_path: str = None,
        data_all: pd.DataFrame,
        target_name: str,
    ):
        # if not file_path and not data:
        #     raise Exception("stupid input")
        data_all = data_all[data_all[target_name].notnull()]
        self.data = data_all.drop(columns=target_name)
        self.target = data_all[target_name]

    # @classmethod
    # def read_file(self, file_path: str, data) -> pd.DataFrame:
    #     if data:
    #         return self.data
    #     if file_path.endswith(".csv"):
    #         self.data = pd.read_csv(file_path)
    #     return self.data

    def defind_categorical_numerical_data(self, is_visualization=False):
        categorical_feature = self.data.select_dtypes(include=["object"]).columns
        data_categorical = self.data[categorical_feature]
        numerical_feature = self.data.select_dtypes(
            include=["int64", "float64"]
        ).columns
        data_numeric = self.data[numerical_feature]
        print(data_categorical.info())
        print(data_numeric.info())
        if is_visualization:
            response = sw.analyze(self.data)
            response.show_html("response.html")

    def handle_missing_value(self):
        data_copy = self.data.copy()
        analyst_columns = []
        for column in self.data.columns:
            feature = data_copy[column]
            if len(feature.dropna()) / len(data_copy) > 0.7:
                analyst_columns.append(column)
        data_copy = self.data[analyst_columns]
        categorical_feature = data_copy.select_dtypes(include=["object"])
        float_feature = data_copy.select_dtypes(include=["float64"])
        int_feature = data_copy.select_dtypes(include=["int64"])
        # impute Categorical value
        categorical_impute = impute.SimpleImputer(
            strategy=Config.categorical_impute_method
        )
        data_impute_categorical = categorical_impute.fit(categorical_feature)
        categorical_data = data_impute_categorical.transform(categorical_feature)
        categorical_data = pd.DataFrame(
            categorical_data, columns=categorical_feature.columns
        )

        # impute float value
        float_impute = impute.SimpleImputer(strategy=Config.numerical_impute_method)

        data_impute_numeric = float_impute.fit(float_feature)
        float_data = data_impute_numeric.fit_transform(float_feature)
        float_data = pd.DataFrame(float_data, columns=float_feature.columns)

        # impute int value
        int_impute = impute.SimpleImputer(strategy=Config.categorical_impute_method)
        int_tranform_fit = int_impute.fit(int_feature)
        int_data = int_tranform_fit.fit_transform(int_feature)
        int_data = pd.DataFrame(int_data, columns=int_feature.columns)
        data_clean_missing_value = pd.concat(
            [categorical_data, float_feature, int_data], axis=1
        )
        self.data_clean_missing_value = data_clean_missing_value
        self.categorical_feature = categorical_data.select_dtypes(include=["object"])
        self.float_feature = float_data.select_dtypes(include=["float64"])
        self.int_feature = int_data.select_dtypes(include=["int64"])
        self.numeric_feature = pd.concat([self.float_feature, self.int_feature], axis=1)
        return data_clean_missing_value

    def handle_target(self):
        print(1)
        target_feature = self.target
        skew = target_feature.skew()
        if abs(skew) < 0.5:
            fig, ax = plt.subplots(figsize=(6, 20))
            ax.set_title(f"Histogram of pricesale, the histogram chart is skewness")
        else:
            fig, ax = plt.subplots(1, 2, figsize=(20, 6))
            pd.plotting.hist_series(target_feature, bins=50, ax=ax[0])
            ax[0].set_title("Normal Histogram")
            # Add your second plot here if needed
            pd.plotting.hist_series(np.log(target_feature), bins=50, ax=ax[1])
            ax[1].set_title("Normal scaling")
        plt.show()

    def feature_engine(self):
        # log
        skew = self.target.skew()
        if abs(skew) > 0.5:
            self.target = pd.Series(np.log(self.target), name=self.target.name)

        # Product and square feature
        self.engine = PolynomialFeatures(degree=2, include_bias=False)
        self.engine.fit(self.numeric_feature)
        self.feature_engine_numerical = self.engine.transform(self.numeric_feature)
        return

    def tranform_data(self):
        # Categorical tranform
        categorical_tranform = OneHotEncoder(drop="first")
        categorical_tranform.fit(self.categorical_feature)
        self.feature_engine_categorical = categorical_tranform.transform(
            self.categorical_feature
        ).toarray()

        # numerical tranform
        scaler_feature_engine = MinMaxScaler(feature_range=(0, 1))
        self.scaler_feature_engine = scaler_feature_engine.fit(
            self.feature_engine_numerical
        )
        self.feature_engine_numerical_tranform = self.scaler_feature_engine.transform(
            self.feature_engine_numerical
        )
        return self.feature_engine_categorical, self.feature_engine_numerical_tranform

    def pipeline_runing(self):
        print("defind_categorical_numerical_data")
        self.defind_categorical_numerical_data()
        print("handle_missing_value")
        self.handle_missing_value()
        print("handle_target")
        self.handle_target()
        print("feature_engine")
        self.feature_engine()
        print("tranform_data")
        self.tranform_data()


if __name__ == "__main__":
    data = pd.read_csv("train.csv").iloc[:, 1:]
    precessing = Precessing(data_all=data, target_name="SalePrice")
    # precessing.defind_categorical_numerical_data(is_visualization = True)
    print(precessing.pipeline_runing())
    print(precessing.feature_engine_categorical)
