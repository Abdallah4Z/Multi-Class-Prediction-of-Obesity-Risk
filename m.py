# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, chi2


# Read train and test data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Drop ID columns
train.drop(columns='id', axis=1, inplace=True)
test.drop(columns='id', axis=1, inplace=True)

# Display initial information
print(train.describe(include='object').T)

# Handle missing values & encoding
X = train
label_encoders = {}
categorical_features = [
    "Gender", "family_history_with_overweight", "FAVC", "CAEC",
    "SMOKE", "SCC", "CALC", "MTRANS", "NObeyesdad"
]
numerical_features = [
    "Age", "Height", "Weight", "CH2O", "FAF", "TUE"
]

# Encode categorical features
for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Separate features and target variable
y = X['NObeyesdad']
X = X.drop(['NObeyesdad'], axis=1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize numerical features
scaler = MinMaxScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])


# Feature selection using Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Select the top features by Random Forest importance
feature_importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
top_features = feature_importances.nlargest(5).index
print("Top features by Random Forest importance:")
print(top_features)

# Recursive Feature Elimination (RFE) with Random Forest
rfe_selector = RFE(estimator=rf_model, n_features_to_select=5)
rfe_selector.fit(X_train, y_train)
rfe_selected_features = X_train.columns[rfe_selector.support_]
print("\nFeatures selected by RFE:")
print(rfe_selected_features)

# Chi-Square Feature Selection
X_train_nonneg = X_train.clip(lower=0)  # Ensure features are non-negative for chi-square
chi2_selector = SelectKBest(chi2, k=5)
chi2_selector.fit(X_train_nonneg, y_train)
chi2_selected_features = X_train.columns[chi2_selector.get_support()]
print("\nFeatures selected by Chi-Square:")
print(chi2_selected_features)

# Combine feature selection results
final_features = set(top_features).union(rfe_selected_features).union(chi2_selected_features)
selected_data = X_train[list(final_features)]  # Convert the set to list for indexing

# Neural network initialization helper functions
def initializeWeights(epsilon_init, L_in, L_out):
    """Initialize weights with small random values."""
    W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init
    return W


def cost_func(nn_params, input_layer_size, hidden_layer_size, output_layer_size):
    """Reshape and extract weights from the parameter vector."""
    t1 = nn_params[:hidden_layer_size * (input_layer_size + 1)].reshape(
        (hidden_layer_size, input_layer_size + 1))
    t2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape(
        (output_layer_size, hidden_layer_size + 1))
    return t1, t2


def relu(z):
    """ReLU activation function."""
    return np.maximum(0, z)


def relu_gradient(z):
    """Gradient of ReLU activation."""
    return (z > 0).astype(float)


def softmax(z):
    """Apply the softmax function."""
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def compute_cost(y_true, y_pred):
    """Cross-entropy cost function."""
    m = y_true.shape[0]
    epsilon = 1e-5  # Avoid division by zero
    cost = -np.sum(y_true * np.log(y_pred + epsilon)) / m
    return cost


def forward_propagation(X, Theta1, Theta2):
    """Perform forward propagation."""
    a1 = np.insert(X, 0, 1, axis=1)  # Add bias
    z2 = np.matmul(a1, Theta1.T)
    a2 = relu(z2)
    a2 = np.insert(a2, 0, 1, axis=1)  # Add bias
    z3 = np.matmul(a2, Theta2.T)
    a3 = softmax(z3)
    return a1, z2, a2, z3, a3


def backward_propagation(X, y, Theta1, Theta2, a2, a3):
    """
    Perform the backward pass for a simple neural network.

    Arguments:
    X -- Input features
    y -- One-hot encoded true labels
    Theta1 -- Weights between input and hidden layer
    Theta2 -- Weights between hidden and output layer
    a2 -- Activations of the hidden layer
    a3 -- Activations of the output layer

    Returns:
    grad_Theta1 -- Gradients for weights Theta1
    grad_Theta2 -- Gradients for weights Theta2
    """
    m = X.shape[0]  # Number of examples

    # Compute the error at the output layer
    delta3 = a3 - y  # Output layer error

    # Backpropagate the error to the hidden layer
    delta2 = np.matmul(delta3, Theta2[:, 1:].T) * relu_gradient(a2[:, 1:])

    # Compute gradients
    grad_Theta2 = np.matmul(delta3.T, np.hstack([np.ones((m, 1)), a2])) / m
    grad_Theta1 = np.matmul(delta2.T, np.hstack([np.ones((m, 1)), X])) / m

    return grad_Theta1, grad_Theta2



def train_nn(X_train, y_train, input_layer_size, hidden_layer_size, output_layer_size, epochs, alpha=0.01):
    """Train the neural network."""
    epsilon_init = 0.12
    Theta1 = initializeWeights(epsilon_init, input_layer_size, hidden_layer_size)
    Theta2 = initializeWeights(epsilon_init, hidden_layer_size, output_layer_size)

    y_train = np.eye(output_layer_size)[y_train]  # One-hot encode the labels

    # Training loop
    for epoch in range(epochs):
        a1, z2, a2, z3, a3 = forward_propagation(X_train, Theta1, Theta2)
        cost = compute_cost(y_train, a3)
        grad_Theta1, grad_Theta2 = backward_propagation(X_train, y_train, Theta1, Theta2, a2, a3)

        Theta1 -= alpha * grad_Theta1
        Theta2 -= alpha * grad_Theta2

        print(f"Epoch {epoch + 1}/{epochs}, Cost: {cost}")

    return Theta1, Theta2


# Model training
input_layer_size = selected_data.shape[1]
hidden_layer_size = input_layer_size * 2
output_layer_size = len(set(train['NObeyesdad']))

Theta1, Theta2 = train_nn(
    X_train=selected_data,
    y_train=y_train,
    input_layer_size=input_layer_size,
    hidden_layer_size=hidden_layer_size,
    output_layer_size=output_layer_size,
    epochs=50,
    alpha=0.01
)

