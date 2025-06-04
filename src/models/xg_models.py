import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, f1_score, roc_auc_score, brier_score_loss, roc_curve
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
import random
import os

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

class PerformanceEvaluator:
    def __init__(self):
        pass

    def print_classification_metrics(self, y_true, y_pred_classes, y_pred_proba, model_name="Model"):
        print(f"--- {model_name} Performance ---")
        print(f'Accuracy: {accuracy_score(y_true, y_pred_classes)*100:.1f}%')
        if y_pred_proba is not None:
            print(f"Log loss: {log_loss(y_true, y_pred_proba):.3f}")
            print(f"ROC AUC: {roc_auc_score(y_true, y_pred_proba):.3f}")
            print(f"Brier score: {brier_score_loss(y_true, y_pred_proba):.4f}")
        print(f"F1 Score: {f1_score(y_true, y_pred_classes):.3f}")
        print("------------------------------------")

    def plot_training_history(self, history, model_name="MLP"):
        fig, axs = plt.subplots(2, figsize=(10,12))
        fig.suptitle(f'{model_name} Training Curves', fontsize=16)

        if 'accuracy' in history.history and 'val_accuracy' in history.history:
            axs[0].plot(history.history['accuracy'], label='Train Accuracy')   
            axs[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
            axs[0].set_title("Accuracy at each epoch")
            axs[0].set_xlabel("Epoch")
            axs[0].set_ylabel("Accuracy")
            axs[0].legend()
        else:
            axs[0].text(0.5, 0.5, 'Accuracy data not available', ha='center', va='center')
            axs[0].set_title("Accuracy")

        if 'loss' in history.history and 'val_loss' in history.history:
            axs[1].plot(history.history['loss'], label='Train Loss')   
            axs[1].plot(history.history['val_loss'], label='Validation Loss')
            axs[1].set_title("Loss at each epoch")
            axs[1].set_xlabel("Epoch")
            axs[1].set_ylabel("Binary Cross-Entropy")
            axs[1].legend()
        else:
            axs[1].text(0.5, 0.5, 'Loss data not available', ha='center', va='center')
            axs[1].set_title("Loss")
            
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def plot_roc_curve(self, y_true, y_pred_proba, model_name="Model"):
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)

        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, label= f"{model_name} ROC-AUC = {auc:.4f}")
        plt.plot([0, 1], [0, 1], color='black', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} ROC Curve')
        plt.legend()
        plt.show()

    def print_goal_shot_counts(self, y_pred_classes_test, y_test_true, X_test_len, y_train_true=None, model_name="Model"):
        print(f"--- {model_name} Goal/Shot Counts ---")
        predicted_goals_test = np.count_nonzero(y_pred_classes_test == 1)
        print(f"Predicted Goals (Test): {predicted_goals_test} in {X_test_len} shots ({predicted_goals_test/X_test_len*100:.3f}%)")

        actual_goals_test = np.count_nonzero(y_test_true == 1)
        print(f"Actual Goals (Test): {actual_goals_test}")

        if y_train_true is not None:
            actual_goals_train = np.count_nonzero(y_train_true == 1)
            print(f"Actual Goals (Train): {actual_goals_train}")
        print("------------------------------------")


class LogisticRegressionModel:
    def __init__(self, random_state=42):
        self.model = LogisticRegression(random_state=random_state, solver='liblinear') 
        self.evaluator = PerformanceEvaluator()

    def train(self, X_train, y_train):
        print("Training Logistic Regression model...")
        self.model.fit(X_train, y_train)
        print("Training complete.")

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)[:, 1] # Probability of positive class

    def evaluate(self, X_test, y_test, X_train=None, y_train=None):
        y_pred_classes = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        self.evaluator.print_classification_metrics(y_test, y_pred_classes, y_pred_proba, "Logistic Regression")
        self.evaluator.plot_roc_curve(y_test, y_pred_proba, "Logistic Regression")
        self.evaluator.print_goal_shot_counts(y_pred_classes, y_test, len(X_test), y_train_true=y_train, model_name="Logistic Regression")


class MLPModel:
    def __init__(self, input_shape, random_state=42):
        # Ensure seeds are set if an instance is created without global seeding
        np.random.seed(random_state)
        random.seed(random_state)
        tf.random.set_seed(random_state)
        
        self.input_shape = input_shape
        self.model = self._build_model()
        self.history = None
        self.evaluator = PerformanceEvaluator()
        self.checkpoint_dir = './model_checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _build_model(self):
        model = Sequential([
            Dense(64, kernel_regularizer='l2', input_shape=self.input_shape),
            BatchNormalization(),
            LeakyReLU(),
            Dropout(0.5),
            
            Dense(32, kernel_regularizer='l2'),
            BatchNormalization(),
            LeakyReLU(),
            Dropout(0.3),
            
            Dense(16, kernel_regularizer='l2'),
            BatchNormalization(),
            LeakyReLU(),
            
            Dense(1, activation='sigmoid'),
        ])
        opt = Adam(learning_rate=0.001)
        model.compile(optimizer=opt, loss="binary_crossentropy", metrics=['accuracy'])
        print("Built MLP model.")
        model.summary()
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, verbose=1):
        print("Training MLP model...")
        
        callbacks = [
            EarlyStopping(min_delta=1e-5, patience=20, mode="min", monitor="val_loss", restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001),
            ModelCheckpoint(filepath=os.path.join(self.checkpoint_dir, 'mlp_best_model.h5'), save_best_only=True, monitor='val_loss', mode='min'),
            TensorBoard(log_dir='./logs/mlp')
        ]
        
        self.history = self.model.fit(X_train, y_train, 
                                     validation_data=(X_val, y_val), 
                                     epochs=epochs, 
                                     verbose=verbose, 
                                     batch_size=batch_size, 
                                     callbacks=callbacks)
        print("Training complete.")
        return self.history

    def predict(self, X_test):
        return self.model.predict(X_test).flatten() # Flatten for consistency if single output

    def predict_classes(self, X_test, threshold=0.5):
        return (self.predict(X_test) > threshold).astype(int)

    def load_best_weights(self):
        weights_path = os.path.join(self.checkpoint_dir, 'mlp_best_model.h5')
        if os.path.exists(weights_path):
            print(f"Loading best weights from {weights_path}")
            self.model.load_weights(weights_path)
        else:
            print(f"Warning: No weights file found at {weights_path}. Model using initial/last trained weights.")
            
    def get_history(self):
        return self.history

    def evaluate(self, X_test, y_test, X_train=None, y_train=None, load_best_weights_first=True):
        if load_best_weights_first:
            self.load_best_weights()
        
        y_pred_proba = self.predict(X_test)
        y_pred_classes = self.predict_classes(X_test)
        
        self.evaluator.print_classification_metrics(y_test, y_pred_classes, y_pred_proba, "MLP")
        self.evaluator.plot_roc_curve(y_test, y_pred_proba, "MLP")
        self.evaluator.print_goal_shot_counts(y_pred_classes, y_test, len(X_test), y_train_true=y_train, model_name="MLP")
        if self.history:
            self.evaluator.plot_training_history(self.history, "MLP")


class XGBoostModel:
    def __init__(self, params=None, random_state=42):
        self.params = params if params else {}
        self.params.setdefault('random_state', random_state)
        self.model = XGBClassifier(**self.params)
        self.evaluator = PerformanceEvaluator()

    def train(self, X_train, y_train, fit_params=None):
        print("Training XGBoost model...")
        if fit_params is None:
            fit_params = {}
        self.model.fit(X_train, y_train, **fit_params)
        print("Training complete.")

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)[:, 1] # Probability of positive class

    def hyperparameter_search(self, X_train, y_train, param_grid=None, n_iter=10, cv=5, scoring='roc_auc', random_state=42):
        print("Starting XGBoost hyperparameter search...")
        if param_grid is None: # Default grid from script
            param_grid = {
                'learning_rate': [0.01, 0.1, 0.001],
                'max_depth': [3, 4, 5, 7, 12, 15],
                'min_child_weight': [1, 2, 3, 5, 10],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'n_estimators' : [50, 100, 200, 500, 700]
            }
        
        search = RandomizedSearchCV(estimator=XGBClassifier(random_state=self.params.get('random_state', random_state)), 
                                    param_distributions=param_grid, 
                                    n_iter=n_iter, 
                                    scoring=scoring, 
                                    cv=cv, 
                                    verbose=2, 
                                    random_state=random_state)
        search.fit(X_train, y_train)
        
        print("Best parameters found:", search.best_params_)
        self.params.update(search.best_params_)
        self.model = XGBClassifier(**self.params) # Re-initialize model with best params
        print("XGBoost model updated with best parameters from search.")
        return search.best_params_

    def evaluate(self, X_test, y_test, X_train=None, y_train=None):
        y_pred_classes = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        self.evaluator.print_classification_metrics(y_test, y_pred_classes, y_pred_proba, "XGBoost")
        self.evaluator.plot_roc_curve(y_test, y_pred_proba, "XGBoost")
        self.evaluator.print_goal_shot_counts(y_pred_classes, y_test, len(X_test), y_train_true=y_train, model_name="XGBoost")


