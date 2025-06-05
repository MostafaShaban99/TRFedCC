import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import scipy.io as sio
import random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib as mpl

# Set font
mpl.rcParams['font.family'] = 'Times New Roman'

# Set seed
random.seed(42)

# Load data
x_data = sio.loadmat('x_data.mat')['data']
y_data = sio.loadmat('y_data.mat')['ydata']

# Use first 12 samples
x = x_data[:, 36:45]
y = y_data[:, 36:45]
print(x[0:2, :])
# Normalize
x = (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0) + 1e-8)

# Reshape for transformer
x = np.expand_dims(x, axis=1)
print(x.shape)


# Train/validation/test split
x_trainval, x_test, y_trainval, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval, test_size=0.25, random_state=42)

# Positional Encoding
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_embedding = layers.Embedding(input_dim=sequence_length, output_dim=d_model)

    def call(self, inputs):
        positions = tf.range(start=0, limit=tf.shape(inputs)[1], delta=1)
        pos_encoding = self.pos_embedding(positions)
        return inputs + pos_encoding

# Transformer block
def transformer_encoder(inputs, num_heads, key_dim, ff_dim, dropout=0.1):
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(inputs, inputs)
    attn_output = layers.Dropout(dropout)(attn_output)
    out1 = layers.LayerNormalization(epsilon=1e-6)(inputs + attn_output)

    ffn = layers.Dense(ff_dim, activation='relu')(out1)
    ffn = layers.Dense(inputs.shape[-1])(ffn)
    ffn = layers.Dropout(dropout)(ffn)

    return layers.LayerNormalization(epsilon=1e-6)(out1 + ffn)

# Model builder
def create_transformer_model(input_shape, n_labels):
    sequence_length, feature_dim = input_shape
    inputs = layers.Input(shape=(sequence_length, feature_dim))
    x = PositionalEncoding(sequence_length, feature_dim)(inputs)

    for _ in range(2):
        x = transformer_encoder(x, num_heads=2, key_dim=16, ff_dim=32, dropout=0.1)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(n_labels, activation='sigmoid')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Build and train model
input_shape = x_train.shape[1:]
n_labels = y_train.shape[1]
model = create_transformer_model(input_shape, n_labels)

history = model.fit(
    x_train, y_train,
    epochs=250,
    batch_size=100,
    validation_data=(x_val, y_val),
    verbose=1
)

# Save training metrics to CSV
metrics_df = pd.DataFrame(history.history)
metrics_df.to_csv("training_metrics_EC4.csv", index_label="Epoch")

# Evaluate on test data
y_pred = model.predict(x_test)
y_pred_binary = (y_pred >= 0.5).astype(int)

acc = tf.keras.metrics.BinaryAccuracy(threshold=0.5)
acc.update_state(y_test, y_pred)
accuracy = acc.result().numpy()

precision_avg = precision_score(y_test, y_pred_binary, average='macro', zero_division=0)
recall_avg = recall_score(y_test, y_pred_binary, average='macro', zero_division=0)
f1_avg = f1_score(y_test, y_pred_binary, average='macro', zero_division=0)

# Print results
print(f"\nTest Accuracy: {accuracy:.4f}")
print(f"Test Precision : {precision_avg:.4f}")
print(f"Test Recall    : {recall_avg:.4f}")
print(f"Test F1 Score  : {f1_avg:.4f}")

# Per-label report
print("\nTest Classification Report (per label):")
print(classification_report(y_test, y_pred_binary, zero_division=0,
                            target_names=[f"Label {i}" for i in range(y_test.shape[1])]))


"""
# Combined Loss and Accuracy plot
plt.figure(figsize=(10, 6))
plt.plot(metrics_df['loss'], label='Train Loss', color='red')
plt.plot(metrics_df['val_loss'], label='Val Loss', color='green')
plt.plot(metrics_df['accuracy'], label='Train Accuracy', color='blue')
plt.plot(metrics_df['val_accuracy'], label='Val Accuracy', color='black')
plt.title('Training and Validation Loss & Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()





# Plot 3x3 confusion matrix subplots
from matplotlib.colors import LinearSegmentedColormap

# Define improved high-contrast custom colormaps
custom_cmaps = [
    LinearSegmentedColormap.from_list("red_map", ["#fff5f0", "#fb6a4a", "#67000d"]),
    LinearSegmentedColormap.from_list("green_map", ["#f7fcf5", "#74c476", "#00441b"]),
    LinearSegmentedColormap.from_list("blue_map", ["#f7fbff", "#6baed6", "#08306b"]),
    LinearSegmentedColormap.from_list("grey_map", ["#f7f7f7", "#bdbdbd", "#252525"])
]

fig, axes = plt.subplots(3, 3, figsize=(12, 10))
fig.suptitle('Confusion Matrices (3x3 Grid)', fontsize=16, fontweight='bold', fontname='Times New Roman')

for i in range(9):
    row, col = divmod(i, 3)
    ax = axes[row, col]
    if i < y_test.shape[1]:
        cm = confusion_matrix(y_test[:, i], y_pred_binary[:, i])
        cmap = custom_cmaps[i % len(custom_cmaps)]
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=False, ax=ax,
                    annot_kws={"size": 10}, linewidths=0.5, linecolor='black')
        ax.set_title(f'Label {i}', fontsize=12, fontname='Times New Roman')
        ax.set_xlabel('Predicted', fontsize=10, fontname='Times New Roman')
        ax.set_ylabel('True', fontsize=10, fontname='Times New Roman')

        # Draw black box around each subplot
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_edgecolor('black')
    else:
        ax.axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
"""