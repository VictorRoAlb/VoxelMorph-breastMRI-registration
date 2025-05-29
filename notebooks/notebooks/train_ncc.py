#This script is almost identical to the one used for training with Mean Squared Error (MSE) loss.
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
# Configures GPU if available, helping speed training on compatible hardware.
# ---

# Configuration
BASE_DIR         = '/content/drive/MyDrive/TFG_Victor'
WEIGHTS_DIR      = os.path.join(BASE_DIR, 'NCC - pesos guardados')
os.makedirs(WEIGHTS_DIR, exist_ok=True)
# ---
# New directory for weights from NCC training to keep results separated.
# ---

TRAIN_LIST_PATH  = '/content/drive/MyDrive/TFG/TFG_Victor/particiones/train_list.txt'
LOAD_MODEL_PATH  = '/content/drive/MyDrive/TFG/TFG_Victor/dense-brain-T1-3d-mse-32feat.h5'  # Same pretrained model
SAVE_MODEL_FINAL = os.path.join(WEIGHTS_DIR, 'perf_dti_final_weights_NCC.h5')

BATCH_SIZE       = 1
EPOCHS           = 1500
STEPS_PER_EPOCH  = 100
LEARNING_RATE    = 1e-4
LAMBDA_REG       = 0.01
PATIENCE         = 30
# ---
# Hyperparameters match previous setup.
# NCC is more robust to intensity scale/brightness differences between modalities,
# so suitable for multimodal registration (Balakrishnan et al., 2019).
# ---

# Read train list
with open(TRAIN_LIST_PATH, 'r') as f:
    train_pairs = [tuple(line.strip().split(',')) for line in f]
print(f"{len(train_pairs)} training pairs loaded. Example: {train_pairs[0]}")
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

sample_dti, _ = load_pair(*train_pairs[0])
print("Sample DTI shape:", sample_dti.shape)
# ---

# Dataset generator
def data_generator():
    while True:
        for dti_path, perf_path in train_pairs:
            dti, perf = load_pair(dti_path, perf_path)
            yield (dti, perf), (perf, perf)
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

# Load and compile model
model = vxm.networks.VxmDense.load(LOAD_MODEL_PATH)
print("Model loaded from:", LOAD_MODEL_PATH)
# ---

losses   = [vxm.losses.NCC().loss, vxm.losses.Grad('l2').loss]
weights  = [1.0, LAMBDA_REG]
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

model.compile(optimizer=optimizer,
              loss=losses,
              loss_weights=weights)
# ---
# Loss changed to NCC for local similarity, more robust for multimodal data.
# Regularization Grad('l2') kept same with lambda=0.01.
# ---

# Callbacks: save every 10 epochs now (compared to every 20 epochs for MSE), but this can be adjusted depending on our needs.
checkpoint_fmt = os.path.join(WEIGHTS_DIR, 'ncc_weights_{epoch:04d}.h5')

save_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_fmt,
    save_freq=10 * STEPS_PER_EPOCH
)
es_cb = tf.keras.callbacks.EarlyStopping(
    monitor='loss',
    patience=PATIENCE,
    restore_best_weights=True
)
# ---
# Checkpoint saving frequency changed to every 10 epochs,
# matching description from your memory.
# EarlyStopping patience 30 as before.
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
        log_df.to_csv(os.path.join(WEIGHTS_DIR, 'training_logs_ncc.csv'), index=False)
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

# Save final model
model.save(SAVE_MODEL_FINAL)
print("Final model saved to:", SAVE_MODEL_FINAL)
# ---

# Post training: plot loss curves
log_path = os.path.join(WEIGHTS_DIR, 'training_logs_ncc.csv')
if os.path.exists(log_path):
    logs = pd.read_csv(log_path)

    plt.figure(figsize=(14, 6))
    plt.plot(logs['epoch'], logs['loss'], label='Total NCC Loss', color='blue')
    plt.title('Total NCC Loss during Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(WEIGHTS_DIR, 'loss_total_ncc.png'))

    plt.figure(figsize=(14, 6))
    plt.plot(logs['epoch'], logs['vxm_dense_flow_loss'], label='Flow Loss', color='orange')
    plt.plot(logs['epoch'], logs['vxm_dense_transformer_loss'], label='Transformer Loss', color='green')
    plt.title('Flow and Transformer Loss during Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(WEIGHTS_DIR, 'loss_flow_transformer_ncc.png'))
