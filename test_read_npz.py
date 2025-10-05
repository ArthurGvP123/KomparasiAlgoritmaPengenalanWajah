import numpy as np

# Ganti 'data.npz' dengan nama file NPZ kamu
file_npz = r".\embeds\embeds_adaface_ir100.npz"

# Membaca file .npz
data = np.load(file_npz)

# Menampilkan nama-nama array yang ada di dalam file
print("Daftar array di dalam file:")
print(data.files)

print(data["probe/kevin/x0ksp4w5bqN0H3AJqjPavsv.jpg"])

# for name in data.files:
#     print(f"\nIsi array '{name}':")

# Jangan lupa menutup file jika diperlukan (meskipun dalam kasus ini tidak wajib)
data.close()
