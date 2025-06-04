import tensorflow as tf
import tensorflow_probability as tfp
import gpflow
import numpy as np
from gpflow.utilities import print_summary
from gpflow.config import default_float

gpflow.config.set_default_summary_fmt("plain")

class VGPXGClassifier:
    """
    Variational Gaussian Process (VGP) Classifier
    """
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.kernel = self._create_kernel()
        self.likelihood = gpflow.likelihoods.Bernoulli()
        self.model = None
        print(f"VGPXGClassifier initialized for input_dim: {input_dim}")

    def _create_kernel(self):
        dtype = default_float()
        # Kernel
        lengthscales_init_tf = tf.ones(self.input_dim, dtype=dtype)
        kernel = gpflow.kernels.Matern52(variance=1.0, lengthscales=lengthscales_init_tf)
        
        # Custom priors
        concentration_prior = tf.constant(2.0, dtype=dtype)
        rate_prior = tf.constant(0.5, dtype=dtype)
        
        kernel.variance.prior = tfp.distributions.Gamma(concentration=concentration_prior, rate=rate_prior)

        if self.input_dim == 1:
            kernel.lengthscales.prior = tfp.distributions.Gamma(concentration=concentration_prior, rate=rate_prior)
        else:
            kernel.lengthscales.prior = tfp.distributions.Gamma(concentration=concentration_prior, rate=rate_prior)
            print(f"Matern52 Kernel for VGP created. Variance prior set. Lengthscales (shape: {kernel.lengthscales.shape}) prior set for all dimensions.")

        return kernel

    def train(self, X_train_raw, y_train_raw, training_iterations=300):
        print("Training VGPXGClassifier...")
        X_train = X_train_raw.astype(default_float())
        y_train = y_train_raw.astype(default_float()).reshape(-1, 1)

        self.model = gpflow.models.VGP(data=(X_train, y_train), likelihood=self.likelihood, kernel=self.kernel)
        print("Initial VGP model summary:")
        print_summary(self.model)

        @tf.function
        def training_loss_fn():
            return self.model.training_loss()

        print(f"Optimizing VGP model for {training_iterations} iterations using L-BFGS-B...")
        optimizer = gpflow.optimizers.Scipy()
        opt_result = optimizer.minimize(
            training_loss_fn,
            variables=self.model.trainable_variables,
            options=dict(maxiter=training_iterations, disp=True),
            method="L-BFGS-B"
        )
        print(f"L-BFGS-B optimization result: {opt_result}")
        print("VGP training complete.")
        print("Final VGP model summary:")
        print_summary(self.model)

    def predict_proba(self, X_test_raw):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        X_test = X_test_raw.astype(default_float())
        mean_pred, _ = self.model.predict_y(X_test)
        return mean_pred.numpy()

    def save(self, model_path: str):
        if self.model is None:
            raise ValueError("Model has not been trained yet. Cannot save.")
        
        self.model.predict_y_compiled = tf.function(
            lambda x: self.model.predict_y(x),
            input_signature=[tf.TensorSpec(shape=[None, self.input_dim], dtype=default_float())]
        )
        tf.saved_model.save(self.model, model_path)
        print(f"VGP model saved to {model_path}")

    @classmethod
    def load(cls, model_path: str):
        print(f"Loading VGP model from {model_path}...")
        loaded_tf_model = tf.saved_model.load(model_path)
        print("VGP TensorFlow SavedModel loaded.")
        return loaded_tf_model

    def predict_latent_mean_and_variance(self, X_raw):
        """Predicts the mean and variance of the latent function f."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        X_converted = X_raw.astype(default_float())
        f_mean, f_var = self.model.predict_f(X_converted)
        return f_mean.numpy(), f_var.numpy()

    def get_ard_lengthscales(self, feature_names=None):
        if self.model is None:
            raise ValueError("Model has not been trained or loaded.")
        if not hasattr(self.model.kernel, 'lengthscales') or self.model.kernel.lengthscales is None:
            print("Kernel does not have lengthscales attribute (e.g., not ARD or wrong kernel type or not initialized properly).")
            return None
            
        lengthscales_vals = self.model.kernel.lengthscales.numpy()
        
        if feature_names and len(feature_names) == len(lengthscales_vals):
            # Shorter lengthscale means more relevance for that dimension.
            sorted_vars = sorted(zip(feature_names, lengthscales_vals), key=lambda x: x[1], reverse=False)
            print("VGP Kernel Lengthscales (Sorted by importance - shorter is more important):")
            for name, lengthscale in sorted_vars:
                print(f"{name}: {lengthscale:.4f}")
            return dict(sorted_vars)
        else:
            print("VGP Kernel Lengthscales (raw values - shorter is more important):")
            print(lengthscales_vals)
            if feature_names:
                print("Warning: Number of feature names does not match number of lengthscales, or lengthscales not available.")
            return lengthscales_vals


class SVGPXGClassifier:
    """
    Sparse Variational Gaussian Process (SVGP) Classifier.
    """
    def __init__(self, input_dim: int, num_inducing_points: int):
        self.input_dim = input_dim
        self.num_inducing_points = num_inducing_points
        self.kernel = None
        self.likelihood = None
        self.inducing_variable = None
        self.model = None
        print(f"SVGPXGClassifier initialized for input_dim: {input_dim}, num_inducing_points: {num_inducing_points}")

    def train(self, X_train_raw, y_train_raw, training_iterations=500, learning_rate=0.01): # learning_rate is unused in L-BFGS-B
        print("Training SVGPXGClassifier...")
        X_train = X_train_raw.astype(default_float())
        y_train = y_train_raw.astype(default_float()).reshape(-1, 1)

        if X_train.shape[0] == 0:
            print("Error: Training data is empty. Cannot train SVGP model.")
            return

        # 1. Define the kernel
        # Using RBF kernel with ARD (Automatic Relevance Determination)
        self.kernel = gpflow.kernels.RBF(lengthscales=np.ones(self.input_dim, dtype=default_float()))
        print("RBF Kernel for SVGP created.")

        # 2. Define the likelihood
        self.likelihood = gpflow.likelihoods.Bernoulli()
        print("Bernoulli likelihood created.")

        # 3. Initialize inducing variables
        actual_num_inducing_points = min(self.num_inducing_points, X_train.shape[0])
        if X_train.shape[0] < self.num_inducing_points:
            print(f"Warning: num_inducing_points requested ({self.num_inducing_points}) > data points available ({X_train.shape[0]}). "
                  f"Using {actual_num_inducing_points} inducing points.")
        
        if actual_num_inducing_points == 0: 
             print("Error: Cannot use 0 inducing points. Check training data.")
             return

        # Randomly initialize the inducing point locations
        np.random.seed(42) #
        inducing_indices = np.random.choice(X_train.shape[0], actual_num_inducing_points, replace=False)
        inducing_X = X_train[inducing_indices, :].copy()
        self.inducing_variable = gpflow.inducing_variables.InducingPoints(inducing_X)
        print(f"Inducing variables initialized with {self.inducing_variable.Z.shape[0]} points.")

        # 4. Create the SVGP model
        self.model = gpflow.models.SVGP(
            kernel=self.kernel, 
            likelihood=self.likelihood, 
            inducing_variable=self.inducing_variable
        )
        print("Initial SVGP model summary:")
        print_summary(self.model)

        # 5. Optimize the model (L-BFGS-B full-batch)
        print(f"Optimizing SVGP using L-BFGS-B (full-batch) for {training_iterations} iterations...")
        
        # Ensure data is in the correct format (TensorFlow Tensors) for the loss function
        data_for_loss_fn = (tf.convert_to_tensor(X_train), tf.convert_to_tensor(y_train))

        @tf.function
        def training_loss_fn():
            # ELBO (Evidence Lower Bound) is maximized, so we minimize its negative.
            return -self.model.maximum_log_likelihood_objective(data_for_loss_fn)

        optimizer = gpflow.optimizers.Scipy()
        opt_result = optimizer.minimize(
            training_loss_fn,
            variables=self.model.trainable_variables,
            options=dict(maxiter=training_iterations, disp=True), # disp=True for iteration messages
            method="L-BFGS-B",
            compile=True # As per the example
        )
        print(f"L-BFGS-B optimization result: {opt_result.message.decode() if hasattr(opt_result.message, 'decode') else opt_result.message}")
        
        print("SVGP training complete.")
        print("Final SVGP model summary:")
        print_summary(self.model)

    def predict_proba(self, X_test_raw):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        X_test = X_test_raw.astype(default_float())
        mean_pred, _ = self.model.predict_y(X_test)
        return mean_pred.numpy()

    def save(self, model_path: str):
        if self.model is None:
            raise ValueError("Model has not been trained yet. Cannot save.")
        
        # Compile predict_f for serving as per notebook
        self.model.predict_f_compiled = tf.function(
            lambda x: self.model.predict_f(x),
            input_signature=[tf.TensorSpec(shape=[None, self.input_dim], dtype=default_float())]
        )
        tf.saved_model.save(self.model, model_path)
        print(f"SVGP model saved to {model_path}")

    @classmethod
    def load(cls, model_path: str):
        print(f"Loading SVGP model from {model_path}...")
        loaded_tf_model = tf.saved_model.load(model_path)
        print("SVGP TensorFlow SavedModel loaded.")
        return loaded_tf_model

    def predict_latent_mean_and_variance(self, X_raw):
        """Predicts the mean and variance of the latent function f."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        X_converted = X_raw.astype(default_float())
        f_mean, f_var = self.model.predict_f(X_converted)
        return f_mean.numpy(), f_var.numpy()

    def get_ard_lengthscales(self, feature_names=None):
        if self.model is None:
            raise ValueError("Model has not been trained or loaded.")
        if not hasattr(self.model.kernel, 'lengthscales') or self.model.kernel.lengthscales is None:
            print("Kernel does not have lengthscales attribute (e.g., not ARD or wrong kernel type or not initialized properly).")
            return None
            
        lengthscales_vals = self.model.kernel.lengthscales.numpy()
        
        if feature_names and len(feature_names) == len(lengthscales_vals):
            # Shorter lengthscale means more relevance for that dimension.
            sorted_vars = sorted(zip(feature_names, lengthscales_vals), key=lambda x: x[1], reverse=False)
            print("SVGP Kernel Lengthscales (Sorted by importance - shorter is more important):")
            for name, lengthscale in sorted_vars:
                print(f"{name}: {lengthscale:.4f}")
            return dict(sorted_vars)
        else:
            print("SVGP Kernel Lengthscales (raw values - shorter is more important):")
            print(lengthscales_vals)
            if feature_names:
                print("Warning: Number of feature names does not match number of lengthscales, or lengthscales not available.")
            return lengthscales_vals
