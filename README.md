# SEU-TCA

SEU-TCA (Spatial Expression Utility - Transfer Component Analysis) is a computational framework for integrating spatially resolved transcriptomics (ST) data with single-cell RNA sequencing (scRNA-seq) data.

![image](https://github.com/user-attachments/assets/1b73e063-a427-461c-9f55-d8d6fc7bc99e)

---

## Algorithm Overview

The core idea of SEU-TCA is to project source and target domain data into a shared feature space where domain divergence is minimized while preserving the discriminative properties of the source domain.

### Key Steps:

1. **Feature Normalization**: 
   - The input source and target feature matrices are concatenated and normalized.

2. **Domain Discrepancy Measurement**: 
   - A Maximum Mean Discrepancy (MMD) matrix is constructed to quantify domain divergence. This matrix encourages alignment by penalizing discrepancies between domains.

3. **Dimensionality Reduction**:
   - A regularized eigenvalue problem is solved to find a projection matrix that minimizes domain discrepancy and preserves data structure.
   - Users can control the dimensionality of the projection through the `dim` parameter.

4. **Kernel Methods**:
   - SEU-TCA supports kernel-based feature mapping, including linear and RBF (Gaussian) kernels. These methods allow the algorithm to handle non-linear relationships in the data.

5. **Classification**:
   - Once features are aligned, a simple 1-Nearest Neighbor (1-NN) classifier is used to classify the target domain data based on the aligned source domain data.

---



## Kernel function selection

The SEU-TCA implementation supports three kernel types, which determine how the data is mapped into the latent space:

1. **Primal Kernel**: 
   - Directly uses the original data without mapping to a higher-dimensional space. Suitable for linearly separable data.
    
2. **Linear Kernel**: 
   - Computes pairwise dot products to capture linear relationships.
  
1. **RBF Kernel**: 
   - Maps data into a high-dimensional space to capture non-linear relationships.
   - Its performance depends on the kernel bandwidth (γ).

---



## Installation

Before using SEU-TCA, ensure that you have Python 3.8 or higher installed, along with the following dependencies:

```bash
pip install numpy scipy pandas scikit-learn matplotlib seaborn scanpy
```

### Install via PyPI
You can install SEU-TCA from PyPI:

```bash
pip install SEU_TCA
```

### Install from Source
Clone the project and install it locally:

```bash
git clone https://github.com/LinluoLab/SEU-TCA.git
cd SEU-TCA
pip install .
```

---

## Usage
Below is an example of how to apply SEU-TCA for domain alignment:
### Data preprocessing
```python
import numpy as np
from seu_tca import SEU_TCA

# scRNA-seq of human heart
sc = sc.read_h5ad("/your_path/SC.h5ad") 
print(sc)

# ST of human heart
st = sc.read_h5ad("/your_path/ST.h5ad") 
print(st)

# Identify common genes between scRNA-seq and ST datasets
glist = [i for i in sc.var_names.tolist() if i in st.var_names]
sc = sc[:,glist]
st = st[:,glist]
```
### Apply TCA for domain alignment
```python
# Extract the expression matrices
Xs = sc.X  # scRNA-seq data matrix
Xt = st.X  # ST data matrix

# Initialize SEU-TCA
seu_tca = seu_tca(kernel_type='linear', dim=10, lamb=1, gamma=1)
Xs_new, Xt_new = seu_tca.fit(Xs, Xt)

# Convert the aligned data into DataFrames with appropriate labels
st_tca = pd.DataFrame(Xt_new, 
                      index=st.obs_names, 
                      columns=['TC_' + str(i) for i in range(1, Xt_new.shape[1] + 1)])

sc_tca = pd.DataFrame(Xs_new, 
                      index=sc.obs_names, 
                      columns=['TC_' + str(i) for i in range(1, Xs_new.shape[1] + 1)])
```
### Output
```python
# Calculate the correlation matrix between scRNA-seq and ST TCA-transformed components
res = pd.DataFrame(np.corrcoef(x=st_tca, y=sc_tca, rowvar=False))

# Extract the relevant correlation submatrix (ST rows vs. scRNA-seq columns)
res = pd.DataFrame(res.iloc[len(st_tca.columns):, :len(st_tca.columns)], 
                   index=sc_tca.index, 
                   columns=st_tca.index)

print(res)
```
## Data Availability
All datasets used in the related publication are publicly available on Zenodo

## Acknowledgements
The authors are grateful to the Lin & Luo lab members for helpful discussion of this study. 
