from collections import defaultdict
from datetime import datetime

import networkx as nx
import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.tree import DecisionTreeRegressor


class NetworkRegressionModel(BaseEstimator, TransformerMixin):
    def __init__(self, model=None):
        """Initialise the regression model"""
        self.model = model if model else DecisionTreeRegressor()

    def fit(self, features: pd.DataFrame, target: pd.Series):
        """Fit the regression model to the training data"""
        self.model.fit(features, target)
        return self

    def predict(self, features: pd.DataFrame) -> ndarray:
        """Predict the target variable using the trained model"""
        return self.model.predict(features)

    def score(self, features: pd.DataFrame, target: pd.Series) -> float:
        """Return the coefficient of determination RRMSE of the prediction"""

        def rmse(y_true, y_pred):
            return np.sqrt(mean_squared_error(y_true, y_pred))

        def rrmse(y_true, y_pred):
            rmse_score = rmse(y_true, y_pred)
            data_range = y_true.max() - y_true.min()
            data_range = np.finfo(float).eps if data_range == 0 else data_range
            return rmse_score / data_range

        return rrmse(target, self.predict(features))

    def extract_features_and_target(
        self, G: nx.MultiDiGraph, target_year: int
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Generate features and target variable for the regression model"""
        trades_df = self.network_to_df(G)
        year_col_index = (
            trades_df.columns.get_loc(str(target_year) + "-01-01")
            if str(target_year) + "-01-01" in trades_df.columns
            else trades_df.shape[1]
        )
        feature_cols = trades_df.columns[
            2:year_col_index
        ]  # Exclude the target year to prevent data leakage
        past_5_cols = feature_cols[-5:]  # Past 5 years
        features_list = []  # Store the features for each row
        total_trade_per_year = trades_df.iloc[:, 2:].sum(axis=0)

        for _, row in trades_df.iterrows():
            importer = row["ImporterCountry"]
            exporter = row["ExporterCountry"]
            trade_volumes = row[feature_cols].astype("float64").dropna()
            features = {"importer": importer, "exporter": exporter}

            features["cumulative_volume"] = row[feature_cols].sum()

            if len(feature_cols) > 1:
                features["volume_change_rate"] = (
                    row[feature_cols[-1]] - row[feature_cols[-2]]
                ) / row[feature_cols[-2]]
            else:
                features["volume_change_rate"] = 0

            if total_trade_per_year[str(target_year - 1) + "-01-01"]:
                features["relative_importance"] = (
                    row[str(target_year - 1) + "-01-01"]
                    / total_trade_per_year[str(target_year - 1) + "-01-01"]
                )
            else:
                features["relative_importance"] = 0

            features["last_year_value"] = row[str(target_year - 1) + "-01-01"]

            max_decay = len(past_5_cols)
            for i, col in enumerate(past_5_cols):
                years_ago = max_decay - i
                decay_weight = (
                    years_ago / max_decay
                )  # Linear decay based on how many years ago
                if i == 0:
                    continue
                prev_col = past_5_cols[i - 1]
                features[f"weighted_difference_{years_ago}y_ago"] = decay_weight * (
                    trade_volumes[col] - trade_volumes[prev_col]
                )
                features[f"weighted_lagged_volume_{years_ago}y_ago"] = (
                    decay_weight * trade_volumes[prev_col]
                )

            if len(trade_volumes) > 1:
                first_year_vol = trade_volumes.iloc[0]
                last_year_vol = trade_volumes.iloc[-2]
                years = len(trade_volumes.index) - 1
                features["average_growth_rate"] = (
                    (last_year_vol / first_year_vol) ** (1 / years) - 1
                    if years > 0
                    else 0
                )
            else:
                features["average_growth_rate"] = 0

            features_list.append(features)

        features_df = pd.DataFrame(features_list)
        features_df["target"] = trades_df[str(target_year) + "-01-01"]

        features_df = pd.get_dummies(
            features_df, columns=["importer", "exporter"]
        )  # One-hot encoding

        target = features_df.pop("target") if "target" in features_df.columns else None

        return features_df, target

    def append_prediction_to_network(
        self, G: nx.MultiDiGraph, year: int, prediction: pd.Series
    ) -> nx.MultiDiGraph:
        """Append the predicted trade volumes to the network for a given year"""
        df = self.network_to_df(G)
        df[str(year) + "-01-01"] = prediction
        return self.df_to_network(df)

    def custom_cross_validate(
        self, G: nx.MultiDiGraph, start_year: int, end_year: int, cv=5
    ) -> ndarray:
        """Custom cross-validation to handle temporal data"""
        scores = []
        available_years = list(range(start_year, end_year))
        years_per_fold = len(available_years) // cv

        for fold in range(cv):
            training_end_year = start_year + (fold + 1) * years_per_fold
            validation_year = training_end_year + 1

            if validation_year > end_year:
                print(f"Not enough data to form fold {fold + 1}. Stopping cv.")
                break

            X, y = self.extract_features_and_target(G, target_year=training_end_year)

            validation_indices = X["year"] == validation_year
            X_train, y_train = X[~validation_indices], y[~validation_indices]
            X_valid, y_valid = X[validation_indices], y[validation_indices]

            self.fit(X_train, y_train)
            score = self.score(X_valid, y_valid)
            scores.append(score)

        return np.mean(scores)

    @staticmethod
    def calculate_cumulative_trade(df, year_col):
        """Calculate the cumulative trade volume"""
        cum_sum = df[year_col].expanding().sum()
        return cum_sum.iloc[-1] if not cum_sum.empty else 0

    @staticmethod
    def calculate_trade_change_rate(df, year_col):
        """Calculate the rate of change in trade volume"""
        return df[year_col].pct_change().iloc[-1] if len(df[year_col]) > 1 else 0

    @staticmethod
    def calculate_relative_importance(df, year_col, total_trade):
        """Calculate the relative trade importance"""
        return df[year_col].iloc[-1] / total_trade if total_trade else 0

    @staticmethod
    def network_to_df(G: nx.MultiDiGraph) -> pd.DataFrame:
        """Convert MultiDiGraph to DataFrame"""
        # Collect trade volumes by date for each importer-exprter pair
        trade_data = defaultdict(lambda: defaultdict(float))
        for u, v, attr in G.edges(data=True):
            date = attr["date"].strftime("%Y-%m-%d")
            trade_data[(u, v)][date] += attr.get("weight", 0)
        # Create df
        trade_df = pd.DataFrame(
            [
                {"ImporterCountry": importer, "ExporterCountry": exporter, **volumes}
                for (exporter, importer), volumes in trade_data.items()
            ]
        )
        return trade_df.sort_values(["ImporterCountry", "ExporterCountry"]).reset_index(
            drop=True
        )

    @staticmethod
    def create_static_snapshot(G: nx.MultiDiGraph, year: int) -> nx.DiGraph:
        """Generate a static snapshot of the network for a given year"""
        snapshot_edges = [
            (u, v, data)
            for u, v, data in G.edges(data=True)
            if data["date"].year == year
        ]
        snapshot = nx.DiGraph()
        snapshot.add_edges_from(snapshot_edges)
        return snapshot

    @staticmethod
    def cut_network_by_year(G: nx.MultiDiGraph, year: int) -> nx.MultiDiGraph:
        """Cut the network by removing edges that are not from the given year"""
        G_copy = G.copy()
        edges_to_remove = [
            (u, v, key, data)
            for u, v, key, data in G_copy.edges(data=True, keys=True)
            if data["date"].year > year
        ]
        G_copy.remove_edges_from(edges_to_remove)
        return G_copy

    @staticmethod
    def df_to_network(df: pd.DataFrame) -> nx.MultiDiGraph:
        G = nx.MultiDiGraph()
        for _, row in df.iterrows():
            importer = row["ImporterCountry"]
            exporter = row["ExporterCountry"]
            G.add_node(importer)
            G.add_node(exporter)
            for date_col in df.columns[2:]:
                date = datetime.strptime(date_col, "%Y-%m-%d")
                volume = row[date_col]
                if pd.notna(volume) and volume != 0:
                    G.add_edge(exporter, importer, date=date, weight=volume)
        return G
