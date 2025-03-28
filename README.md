# Alzheimer's Disease Detection using Deep Learning

This repository implements a deep learning framework for detecting Alzheimer's disease (AD) stages from MRI scans, focusing on early diagnosis using transfer learning with EfficientNetB0. The system classifies brain images into four clinical categories: nondemented, very mild, mild, and moderate dementia. Built with PyTorch and Flask, the project emphasizes reproducibility, clinical relevance, and seamless deployment.

---

### Framework Highlights  
- **Optimized Architecture**: Leverages EfficientNetB0 pretrained on ImageNet, fine-tuned for AD staging, achieving accuracy benchmarks comparable to recent studies[2][8].  
- **Data Pipeline**: Implements preprocessing (normalization, skull-stripping) and augmentation (rotation, flipping) tailored for neuroimaging[1][3].  
- **Clinical Deployment**:  
  - REST API for MRI uploads and real-time predictions  
  - Dockerized environment for consistent deployment  
- **Validation Rigor**: Evaluated using metrics like accuracy, AUC-ROC, and confusion matrices, aligning with methodologies in[2][8].  

---

### Repository Structure  
```bash
├── model/            # Model definitions (AlzheimerNet)  
├── data/             # Preprocessed MRI datasets (train/test splits)  
├── utils/            # Data loaders and image transformations  
├── api/              # Flask application for predictions  
├── tests/            # Unit and integration tests  
├── train.py          # Model training script  
├── evaluate.py       # Performance validation  
└── requirements.txt  # Dependency specifications  
```

---

### Key Objectives  
1. **Early Detection**: Enable identification of preclinical and mild cognitive impairment (MCI) stages, critical for slowing disease progression[3][6].  
2. **Research Enablement**: Provide a modular codebase for exploring neuroimaging biomarkers and model interpretability techniques like Grad-CAM.  
3. **Clinical Translation**: Bridge AI and healthcare through containerized deployment compatible with hospital IT systems[7][11].  

---

### Workflow Integration  
#### Training & Evaluation  
```bash  
# Train model with default parameters  
python train.py --epochs 50 --batch_size 32  

# Evaluate on test set  
python evaluate.py --model_path saved_models/best.pth  
```

#### API Deployment  
```python  
# Start Flask server  
cd api && gunicorn --workers 4 --bind 0.0.0.0:5000 app:app  

# Example POST request  
curl -X POST -F "file=@/path/to/mri.nii" http://localhost:5000/predict  
```

---

### Model Configuration  
| Component           | Specification                          |  
|---------------------|----------------------------------------|  
| Base Model          | EfficientNetB0 (ImageNet initialized)  |  
| Input Resolution    | 224x224x3 (resampled from 3D scans)    |  
| Loss Function       | Cross-Entropy with Class Weighting     |  
| Optimizer           | AdamW (lr=3e-4, weight_decay=1e-5)     |  

---

### Extended Applications  
- **Multi-Cohort Validation**: Compatible with ADNI, OASIS-3, and NACC datasets via standardized preprocessing[4][7].  
- **Interpretability**: Future integration of SHAP analysis to map model decisions to neuroanatomical regions[11].  

--- 

**Ethical Note**: Designed for research/educational use. Clinical adoption requires regulatory approval and validation on diverse populations.

Citations:

- [1] https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2024.1404494/full
- [2] https://pmc.ncbi.nlm.nih.gov/articles/PMC10909166/
- [3] https://pmc.ncbi.nlm.nih.gov/articles/PMC10093003/
- [4] https://www.nature.com/articles/s41598-022-20674-x
- [5] https://www.nature.com/articles/s41598-024-72321-2
- [6] https://ojs.aaai.org/index.php/AAAI/article/view/17772/17579
- [7] https://www.nature.com/articles/s41467-022-31037-5
- [8] https://pmc.ncbi.nlm.nih.gov/articles/PMC10417320/
- [9] https://pmc.ncbi.nlm.nih.gov/articles/PMC6719787/
- [10] https://pmc.ncbi.nlm.nih.gov/articles/PMC11202897/
- [11] https://github.com/vkola-lab/ncomms2022
