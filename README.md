# Disease classification of Alzheimer's Disease and Parkinson’s Disease via Semi-supervised Learning

The human connectome is measured via the structural connectome (SC) which comprises of white matter connections between brain regions, and the functional connectome (FC) which comprises  correlated activations across brain regions. Both have been shown to be useful for early detection of neurodegenerative diseases and many algorithms have been proposed to extract such biomarkers from neuroimaging scans. However, a longstanding issue that has limited the effectiveness of these algorithms is the lack of large datasets, especially for rare diseases. Coupled with the high dimensionality of neuroimaging data, this makes it easy for models to overfit, rendering the discovered biomarkers ungeneralizable and inaccurate.

To address this issue, we will explore semi-supervised techniques to train these models, such that large datasets can be used in the training process to help the model generalise better. Both SC and FC can be expressed as graphs. Thus, graph neural networks are best suited to model these datasets and we will explore techniques such as manifold regularisation, graph Laplacian regularisation and dual learning to train models in a semi-supervised fashion. In the first part of the project, we will be using the Parkinson's Progression Marker Initiative (PPMI) dataset for classification of patients with Parkinson’s Disease. In the second part of the project, we will attempt to use both the Schizconnect dataset and a private dataset to diagnose Schizophrenia. Scans obtained from different sites and via different protocols cannot be trivially combined together. We will apply data harmonisation techniques such as the ComBat algorithm to remove site differences before using the datasets for modelling.

Throughout this project, the student will gain familiarity with deep learning frameworks such as Keras and PyTorch, especially the PyTorch Geometric library. Also, the student will develop an in-depth understanding of semi-supervised learning techniques. The student will be expected to be comfortable with reading and implementing algorithms from research papers and will thus pick up research skills from this experience.

## Environment Setup

1. Install packages in ``requirements.txt``.

        pip install -r requirements.txt
        conda install --file requirements.txt

2. Install ``pytorch`` following this [link](https://pytorch.org/).

3. Install ``pytorch_geometric`` following this [link](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).


