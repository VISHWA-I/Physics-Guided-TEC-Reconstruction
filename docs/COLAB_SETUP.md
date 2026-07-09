# Google Colab Setup Guide

This guide explains how to run the Topside TEC Reconstruction project on Google Colab for GPU-accelerated training.

The project is fully compatible with Google Colab and Google Drive. All checkpoints, logs, TensorBoard events, and exported models are automatically saved directly to your Google Drive. This ensures that no data is lost when the Colab runtime disconnects.

## 1. Prepare Google Drive

1. Open your Google Drive.
2. Create a new folder named `TEC_Project` in the root of your Drive (`MyDrive/TEC_Project`).
3. Upload the entire project directory into this folder.
   - *Note: You do not need to upload the virtual environment (`venv`) or the `.git` folder.*
   - Ensure the structure looks like this:
     ```
     MyDrive/TEC_Project/
     ├── app/
     ├── colab/
     ├── configs/
     ├── data/
     ├── models/
     ├── src/
     ├── training/
     ├── utils/
     ├── train.py
     └── requirements_colab.txt
     ```

## 2. Start Colab Training

1. In Google Drive, navigate to `MyDrive/TEC_Project/colab/`.
2. Right-click `train_colab.ipynb` -> **Open with** -> **Google Colaboratory**.
3. In the Colab menu, go to **Runtime** -> **Change runtime type**.
4. Select **T4 GPU** or **A100 GPU** as the hardware accelerator and click Save.
5. Run the cells in order:
   - **Cell 1**: Mounts your Google Drive. You will be prompted to grant Colab access to your Drive.
   - **Cell 2**: Installs missing dependencies (`tensorboard`, `onnx`, etc.) and verifies the environment.
   - **Cell 3**: Launches TensorBoard directly inside the notebook.
   - **Cell 4**: Starts the training loop via `train.py`.

## 3. Auto-Resume and Checkpoints

Google Colab runtimes can disconnect after inactivity or time limits. The project handles this gracefully:

- During training, the model saves a `latest.pt` checkpoint to `MyDrive/TEC_Project/checkpoints/`.
- If your runtime disconnects, simply reconnect, run the setup cells again, and run the training cell.
- The `train.py` script automatically detects `latest.pt` and resumes training from the exact epoch where it left off, restoring both model weights and optimizer states.

## 4. Model Export

Once training completes (or is manually stopped and resumed with fewer epochs), the script automatically exports the model into three formats for deployment:

1. **PyTorch State Dict**: `exports/hybrid_model_epoch_X.pt`
2. **TorchScript**: `exports/hybrid_model_epoch_X_scripted.pt`
3. **ONNX**: `exports/hybrid_model_epoch_X.onnx`

These are saved directly to `MyDrive/TEC_Project/exports/`.

## 5. Troubleshooting

* **"No space left on device"**: Colab instances have limited local disk space. Ensure you are writing to `/content/drive/MyDrive/...` (which uses your Drive quota, up to 15GB free) and not `/content/...`.
* **CUDA Out of Memory**: Reduce the `batch_size` in the training cell (e.g., from 64 to 32).
* **Missing Module Error**: Make sure you ran the `pip install -r requirements_colab.txt` cell.
