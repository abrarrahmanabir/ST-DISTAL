
### ST-DISTAL : Learning Spatial Transcriptomics Deconvolution via Spectral-Spatial Dual-Branch GCNs with Distributional Alignment

This repository contains the implementation of **ST-DISTAL**, a framework for accurate and biologically meaningful cell-type deconvolution in spatial transcriptomics data.

## Repository Structure

`./STDISTAL/GCN.py`
Implementation of the dual-branch GCN and the training logic. Code sections are annotated with corresponding equation numbers from the paper for clarity.

`./data/`
Contains a representative dataset.

`./output/`
Output directory for predictions and results.


## Usage

**1. Install Dependencies**
```

pip install -r requirements.txt

```
**2. Run the Model**
```
mkdir output
python run.py
```

The model output will be saved in ./output/predict_result.csv.

Each row corresponds to a spatial spot; columns represent cell types; values indicate predicted cell-type proportions.

**3. Evaluate Results**

An evaluation script is provided:
```
python eval.py
```
