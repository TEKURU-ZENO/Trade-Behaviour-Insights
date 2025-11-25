import optuna
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .sequence_dataset import AccountSequenceDataset


# ================================================================
#   OPTUNA + LIGHTGBM (FIXED: UNIVERSAL CALLBACK EARLY STOPPING)
# ================================================================
def optuna_lgbcv(X, y, groups, cat_features=None, n_trials=40):
    def objective(trial):

        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": "gbdt",

            # Optuna search
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 0, 7),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 300),
        }

        gkf = GroupKFold(n_splits=3)
        aucs = []

        for tr_idx, val_idx in gkf.split(X, y, groups=groups):
            X_train, X_valid = X.iloc[tr_idx], X.iloc[val_idx]
            y_train, y_valid = y.iloc[tr_idx], y.iloc[val_idx]

            dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
            dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)

            # -----------------------------------------------------------
            # FIXED: Use LightGBM CALLBACKS (works on ALL versions)
            # -----------------------------------------------------------
            bst = lgb.train(
                params,
                dtrain,
                valid_sets=[dvalid],
                num_boost_round=2000,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                ]
            )

            pred = bst.predict(X_valid)
            auc = roc_auc_score(y_valid, pred)
            aucs.append(auc)

        return np.mean(aucs)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    return study



# ================================================================
#   LSTM SEQUENCE CLASSIFIER
# ================================================================
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(hidden, 2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.drop(out[:, -1, :])
        return self.fc(out)



# ================================================================
#   LSTM TRAINING PIPELINE (IMPROVED)
# ================================================================
def train_lstm(df, features, seq_len=20, batch_size=128, epochs=10, lr=1e-3, device="cpu"):
    df = df.copy()
    df["target"] = (df["closedpnl"] > 0).astype(int)

    dataset = AccountSequenceDataset(df, features, seq_len)

    # Split by sequence index
    idx = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42)

    train_loader = DataLoader(
        [dataset[i] for i in train_idx],
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        [dataset[i] for i in val_idx],
        batch_size=batch_size,
        shuffle=False
    )

    model = LSTMClassifier(input_size=len(features)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        total = 0

        for Xb, yb in train_loader:
            Xb = Xb.float().to(device)
            yb = yb.to(device)

            pred = model(Xb)
            loss = loss_fn(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(yb)
            total += len(yb)

        # ================== VALIDATION =====================
        model.eval()
        ys, ps = [], []

        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.float().to(device)
                pred = torch.softmax(model(Xb), dim=1)[:, 1].cpu().numpy()

                ys.extend(yb.numpy().tolist())
                ps.extend(pred.tolist())

        auc = roc_auc_score(ys, ps) if len(set(ys)) > 1 else None

        print(f"Epoch {epoch} "
              f"TrainLoss={train_loss/total:.4f} "
              f"ValAUC={auc}")

    return model
