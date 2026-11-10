
# Import torch terlebih dahulu agar DLL CUDA-nya tersedia.
import torch

print("=" * 50)
print("NVIDIA GPU TEST")
print("=" * 50)

print("Torch version :", torch.__version__)
print("Torch CUDA    :", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU name      :", torch.cuda.get_device_name(0))

import onnxruntime as ort

print("\nONNX RUNTIME")
print("-" * 50)

print("ORT version:", ort.__version__)
print("Providers  :", ort.get_available_providers())

# Load CUDA dan cuDNN DLL dari environment PyTorch.
ort.preload_dlls()

print("\nCUDA DLL INFORMATION")
print("-" * 50)

ort.print_debug_info()

print("\n[INFO] Diagnostic selesai.")
