# Repositori Algoritma untuk Embedding

Dokumen ini merangkum **repositori resmi** yang digunakan, beserta perintah **clone** untuk menyiapkan pengujian embedding.

> **Prasyarat**:  
> - Sudah menginstal **Git**  
> - (Opsional) **Git LFS** untuk repo besar: `git lfs install`  
> - Jika repo punya submodule: `git submodule update --init --recursive`

---

## Clone Semua Sekaligus

### Bash / Zsh (Linux/Mac/WSL)
```bash
# Pilih direktori kerja tempat semua repositori akan disimpan
mkdir -p repos && cd repos

# Clone cepat (shallow) agar lebih ringan. Hapus --depth=1 jika perlu riwayat penuh.
git clone --depth=1 https://github.com/HuangYG123/CurricularFace
git clone --depth=1 https://github.com/IrvingMeng/MagFace
git clone --depth=1 https://github.com/mk-minchul/AdaFace
git clone --depth=1 https://github.com/fdbtrs/ElasticFace
git clone --depth=1 https://github.com/DanJun6737/TransFace
git clone --depth=1 https://github.com/HamadYA/GhostFaceNets
git clone --depth=1 https://github.com/mk-minchul/CVLface
git clone --depth=1 https://github.com/szlbiubiubiu/LAFS_CVPR2024
git clone --depth=1 https://github.com/WakingHours-GitHub/EPL
git clone --depth=1 https://github.com/bytedance/LVFace
git clone --depth=1 https://github.com/davisking/dlib
git clone --depth=1 https://github.com/timesler/facenet-pytorch

# (Opsional) Inisialisasi submodule bila ada
for d in */ ; do
  (cd "$d" && git submodule update --init --recursive || true)
done
