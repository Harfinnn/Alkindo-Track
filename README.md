# Alkindo Track

Tracking orang & kendaraan dari CCTV (YOLO11 + ByteTrack, OpenVINO).

## Struktur folder

```
alkindo_track/
├── human_tracking.py            # Tracking lobby (1 kamera, okupansi per area)
├── human_tracking_dual_cctv.py  # Counting IN/OUT dual CCTV (orang & kendaraan)
├── requirements.txt
├── config/                      # Konfigurasi (garis hitung, ROI, tracker)
│   ├── custom_bytetrack_stable.yaml
│   ├── lobby_tracking_config.json
│   ├── people_counting_line_cam1.json
│   └── people_counting_line_cam2.json
├── models/                      # Bobot YOLO (.pt) & hasil export OpenVINO
├── data/                        # State runtime (lobby_state.json)
├── logs/
│   ├── lobby/                   # lobby_occupancy_log.csv, lobby_track_log.csv
│   └── crossing/                # crossing_YYYY-MM-DD.csv
└── venv/                        # Virtual environment (tidak di-commit)
```

## Menjalankan

```
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

python human_tracking.py
python human_tracking_dual_cctv.py
```

Semua path dihitung dari lokasi script, jadi bisa dijalankan dari folder mana pun.
