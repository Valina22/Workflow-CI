# Workflow-CI

## Heart Disease Prediction - Continuous Integration Pipeline

Repository ini merupakan implementasi Continuous Integration (CI) untuk proyek Machine Learning prediksi penyakit jantung menggunakan **MLflow Project**, **GitHub Actions**, dan **Docker Hub** sebagai bagian dari submission *Belajar Penerapan Machine Learning untuk Production (MLOps)*.

---

## Struktur Repository

```text
Workflow-CI
├── .github
│   └── workflows
│       └── ci.yml
│
├── MLProject
│   ├── modelling.py
│   ├── conda.yaml
│   ├── MLproject
│   └── dataset_preprocessing/
│
├── Dockerfile
├── README.md
└── dockerhub_link.txt
```

---

## Deskripsi Proyek

Proyek ini membangun model Machine Learning untuk memprediksi risiko penyakit jantung berdasarkan data klinis pasien.

Tahapan yang dilakukan:

1. Memuat dataset hasil preprocessing.
2. Melatih model Machine Learning.
3. Mengevaluasi performa model.
4. Mencatat eksperimen menggunakan MLflow.
5. Menjalankan proses training otomatis melalui GitHub Actions.
6. Membangun Docker Image dan mengunggahnya ke Docker Hub.

---

## Menjalankan MLflow Project

Masuk ke folder MLProject:

```bash
cd MLProject
```

Jalankan MLflow Project:

```bash
mlflow run .
```

---

## Continuous Integration (CI)

Workflow GitHub Actions akan berjalan otomatis ketika:

* Terjadi push ke branch `main`
* Terjadi push ke branch `develop`
* Workflow dijalankan secara manual melalui GitHub Actions

Tahapan workflow:

1. Lint dan validasi sintaks Python.
2. Menjalankan training model menggunakan MLflow.
3. Menyimpan artefak hasil training.
4. Membangun Docker Image.
5. Mengunggah Docker Image ke Docker Hub.

---

## Docker Hub

Docker Image dapat diakses melalui:

```text
https://hub.docker.com/r/USERNAME_DOCKERHUB/heart-disease-ml
```

Ganti `USERNAME_DOCKERHUB` dengan username Docker Hub yang digunakan.

---

## Tools dan Library

* Python 3.12
* MLflow
* Scikit-Learn
* Pandas
* NumPy
* GitHub Actions
* Docker
* Docker Hub
* DagsHub

---

## Author

**Valina Puspita Sari**

Submission MLOps - Dicoding
