import os
import nibabel as nib
import numpy as np
import tensorflow as tf
import voxelmorph as vxm
import pandas as pd
import matplotlib.pyplot as plt

# GPU setup
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.set_visible_devices(physical_devices[0], 'GPU')
    print("Using GPU:", physical_devices[0])
else:
    print("No GPU detected. Using CPU.")
# ---
# Configures GPU if available. The original VoxelMorph script handles GPU via TensorFlow config,
# here we explicitly print GPU presence to verify the runtime environment.
# This helps speed up training on supported hardware (see TFG chapter 4).
# ---

# Configuration
BASE_DIR         = '/content/drive/MyDrive/TFG_Victor'
WEIGHTS_DIR      = os.path.join(BASE_DIR, 'MSE - pesos guardados')
os.makedirs(WEIGHTS_DIR, exist_ok=True)
# ---
# Sets directories adapted to Google Drive for Colab use.
# Creates directory for storing saved weights, ensuring organized output.
# ---

TRAIN_LIST_PATH  = '/content/drive/MyDrive/TFG/TFG_Victor/particiones/train_list.txt'
LOAD_MODEL_PATH  = '/content/drive/MyDrive/TFG/TFG_Victor/dense-brain-T1-3d-mse-32feat.h5'
SAVE_MODEL_FINAL = os.path.join(WEIGHTS_DIR, 'mse_perf_dti_final_weights.h5')

BATCH_SIZE       = 1
EPOCHS           = 1500
STEPS_PER_EPOCH  = 100
LEARNING_RATE    = 1e-4
LAMBDA_REG       = 0.01
PATIENCE         = 30
# ---
# Hyperparameters match those described in the memory (chapter 4.3).
# EarlyStopping patience added here to avoid wasting unnecessary compute resources,
# a practical addition not explicitly present in the original script.
# ---

# Read train list
with open(TRAIN_LIST_PATH, 'r') as f:
    train_pairs = [tuple(line.strip().split(',')) for line in f]
print(f"{len(train_pairs)} training pairs loaded. Example: {train_pairs[0]}")
# ---
# Loads list of training data pairs (DTI and perfusion volumes),
# following the dataset split generated previously (chapter 4).
# ---

# NIfTI loader
def load_nifti(path):
    raw = nib.load(path).get_fdata().astype(np.float32)
    dims = raw.shape
    try:
        i_x = dims.index(160)
        i_y = dims.index(192)
        i_z = dims.index(224)
    except ValueError:
        raise ValueError(f"Unexpected volume shape {dims} for {path}. Expected dims 160,192,224.")
    vol = np.moveaxis(raw, (i_x, i_y, i_z), (0, 1, 2))
    return np.expand_dims(vol, axis=-1)

def load_pair(dti_path, perf_path):
    return load_nifti(dti_path), load_nifti(perf_path)
# ---
# Ensures that volumes have the correct dimension order expected by the model (160x192x224x1),
# a necessary adaptation for consistency with the pretrained weights (chapter 4.3).
# ---

sample_dti, _ = load_pair(*train_pairs[0])
print("Sample DTI shape:", sample_dti.shape)
# ---
# Quick check of shape for debugging and validation purposes.
# ---

# Dataset
def data_generator():
    while True:
        for dti_path, perf_path in train_pairs:
            dti, perf = load_pair(dti_path, perf_path)
            yield (dti, perf), (perf, perf)
# ---
# Custom generator yields pairs of input (DTI as moving, perfusion as fixed) and target (perf, perf) images.
# This matches the unsupervised learning framework of VoxelMorph for image registration.
# ---

input_shape = sample_dti.shape
dataset = tf.data.Dataset.from_generator(
    data_generator,
    output_signature=(
        (tf.TensorSpec(input_shape, tf.float32),
         tf.TensorSpec(input_shape, tf.float32)),
        (tf.TensorSpec(input_shape, tf.float32),
         tf.TensorSpec(input_shape, tf.float32)),
    )
).repeat().batch(BATCH_SIZE)
# ---
# Builds a TensorFlow Dataset from the generator, repeating infinitely and batching size 1,
# balancing memory usage and training stability (chapter 4).
# ---

# Load and compile model
model = vxm.networks.VxmDense.load(LOAD_MODEL_PATH)
print("Model loaded from:", LOAD_MODEL_PATH)
# ---
# Loads official pretrained VoxelMorph Dense model for fine-tuning on breast imaging data.
# This leverages transfer learning for improved convergence (chapter 4.3).
# ---

losses   = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]
weights  = [1.0, LAMBDA_REG]
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

model.compile(optimizer=optimizer,
              loss=losses,
              loss_weights=weights)
# ---
# Uses composite loss with MSE (measures voxel-wise discrepancy) and gradient regularization (encourages smooth deformation).
# Weights balance these terms.
# This setup follows the original repo but with custom tuning for this dataset.
# ---

# Callbacks
checkpoint_fmt = os.path.join(WEIGHTS_DIR, 'mse_weights_{epoch:04d}.h5')

save_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_fmt,
    save_freq=20 * STEPS_PER_EPOCH
)
es_cb = tf.keras.callbacks.EarlyStopping(
    monitor='loss',
    patience=PATIENCE,
    restore_best_weights=True
)
# ---
# ModelCheckpoint saves weights every 20 epochs as in the original script.
# EarlyStopping is added here to halt training when loss stops improving, saving computational resources.
# ---

class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.epochs = []
        self.losses = []
        self.flow_losses = []
        self.transformer_losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.epochs.append(epoch)
        self.losses.append(logs['loss'])
        self.flow_losses.append(logs['vxm_dense_flow_loss'])
        self.transformer_losses.append(logs['vxm_dense_transformer_loss'])

        log_df = pd.DataFrame({
            'epoch': self.epochs,
            'loss': self.losses,
            'vxm_dense_flow_loss': self.flow_losses,
            'vxm_dense_transformer_loss': self.transformer_losses
        })
        log_df.to_csv(os.path.join(WEIGHTS_DIR, 'training_logs.csv'), index=False)
# ---
# Custom callback to save detailed loss history for post-training analysis,
# enabling visualization of total and component losses (chapter 5).
# ---

history_cb = LossHistory()

# Train
model.fit(
    dataset,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    callbacks=[save_cb, es_cb, history_cb],
    verbose=1
)
# ---
# Train model with specified epochs and callbacks.
# ---

# Save final model
model.save(SAVE_MODEL_FINAL)
print("Final model saved to:", SAVE_MODEL_FINAL)
# ---
# Save the final trained model weights for further inference or evaluation.
# ---

# Post training: plotting loss curves
log_path = os.path.join(WEIGHTS_DIR, 'training_logs.csv')
if os.path.exists(log_path):
    logs = pd.read_csv(log_path)

    plt.figure(figsize=(14, 6))
    plt.plot(logs['epoch'], logs['loss'], label='Total MSE Loss', color='blue')
    plt.title('Total MSE Loss during Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(WEIGHTS_DIR, 'loss_total_mse.png'))

    plt.figure(figsize=(14, 6))
    plt.plot(logs['epoch'], logs['vxm_dense_flow_loss'], label='Flow Loss', color='orange')
    plt.plot(logs['epoch'], logs['vxm_dense_transformer_loss'], label='Transformer Loss', color='green')
    plt.title('Flow and Transformer Loss during Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(WEIGHTS_DIR, 'loss_flow_transformer.png'))
# ---
# Visualizes loss evolution post-training, replicating the plots in your memory Fig. 28,
# confirming model convergence with typical deep learning loss behavior.
# ---
