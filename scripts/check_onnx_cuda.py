
import onnxruntime as ort

print("=" * 55)
print("ONNX RUNTIME CUDA DIAGNOSTIC")
print("=" * 55)

print("ORT Version :", ort.__version__)
print("ORT Path    :", ort.__file__)
print("Providers   :", ort.get_available_providers())

print("\nBUILD INFORMATION")
print("-" * 55)

if hasattr(ort, "get_build_info"):
    print(ort.get_build_info())

print("\nPRELOAD CUDA DLL")
print("-" * 55)

try:
    ort.preload_dlls()
    print("[SUCCESS] preload_dlls berhasil")
except Exception as error:
    print("[FAILED] preload_dlls:", repr(error))

print("\nDEBUG INFORMATION")
print("-" * 55)

if hasattr(ort, "print_debug_info"):
    ort.print_debug_info()
else:
    print("print_debug_info tidak tersedia")
