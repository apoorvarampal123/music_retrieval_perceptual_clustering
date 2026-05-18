# From Sound to Similarity: Explainable Music Retrieval via Multi-Dimensional Perceptual Cluster Fingerprinting

## Overview
Built an explainable music similarity engine on 10,000 audio tracks across 10 genres. The system identifies which specific audio dimension (timbre, rhythm, harmony, energy) makes two songs similar — providing explainability that standard cosine similarity cannot offer.

## Course
Music Information Retrieval (MIR) — University of Victoria, 2026

## Tools & Technologies
- **Language:** Python
- **Audio Feature Extraction:** librosa
- **Machine Learning:** Scikit-learn (K-Means, DBSCAN, PCA)
- **Data Processing:** NumPy, Pandas
- **Visualisation:** Matplotlib

## Dataset
- 10,000 English-language tracks from the Music4All dataset
- 10 balanced genres: Electronic, Folk, Hip Hop, Jazz, Metal, Pop, Punk, Rap, Rock, Soul
- 1,000 songs per genre

## What I Built

### 1. Audio Feature Dataset
- Engineered a 110-dimensional feature dataset per track using librosa
- Features cover four perceptual dimensions:
  - **Timbre:** MFCCs, spectral centroid, bandwidth, rolloff, contrast, ZCR, RMS
  - **Rhythm:** Tempo, beat histogram, inter-beat intervals, onset strength
  - **Harmony:** Chroma CQT, Tonnetz representations
  - **Energy & Dynamics:** Mel-spectrogram statistics

### 2. Unsupervised Clustering Pipeline
- Applied auto-PCA per feature group (95% variance threshold)
- Used silhouette-optimised K-Means to find optimal clusters per dimension
- Applied DBSCAN fingerprinting with auto-estimated epsilon via k-distance elbow method
- Pipeline applied independently per genre across all 10 genres

### 3. Song Similarity Engine
- Each song assigned a cluster fingerprint across 4 perceptual dimensions
- Similarity score based on number of matching dimensions (0-4)
- System explains which specific dimension drives similarity

## Key Results
- Outperformed full 110-dimensional cosine similarity baseline by ~19 MAP points
- Discovered universal perceptual patterns across all 10 genres:
  - Energy always partitions into 3 clusters
  - Harmony and Timbre always produce binary splits
  - Rhythm is the only genre-discriminating dimension
- Metal is the most perceptually cohesive genre
- Electronic and Pop are the most sonically diverse genres

## Project Structure
DBSCAN_cluster_fingerprint_result/     — DBSCAN clustering outputs
Kmeans_clustered_samples/              — K-Means clustering results
Output_files/                          — Final output files
Song_similarity_files/                 — Similarity engine outputs
cluster_fingerprint_dbscan_results/    — Fingerprint results
combined_parquet_after_kmeans/         — Post-clustering data
demo_data/                             — Demo files
genre_specific_feature_isolation/      — Per-genre feature files

## My Contributions
- Audio feature extraction pipeline using librosa
- Per-genre clustering architecture design and implementation
- Song similarity engine development
- Cross-genre analysis and MAP evaluation against cosine baselines
- Final report writing and results interpretation
