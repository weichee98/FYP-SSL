# ABIDE

[The Autism Brain Imaging Data Exchange I (ABIDE I)](http://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html) represents the first ABIDE initiative. Started as a grass roots effort, ABIDE I involved 17 international sites, sharing previously collected resting state functional magnetic resonance imaging (R-fMRI), anatomical and phenotypic datasets made available for data sharing with the broader scientific community. This effort yielded 1112 dataset, including 539 from individuals with ASD and 573 from typical controls (ages 7-64 years, median 14.7 years across groups). This aggregate was released in August 2012. Its establishment demonstrated the feasibility of aggregating resting state fMRI and structural MRI data across sites; the rate of these data use and resulting publications (see Manuscripts) have shown its utility for capturing whole brain and regional properties of the brain connectome in Autism Spectrum Disorder (ASD). In accordance with HIPAA guidelines and 1000 Functional Connectomes Project / INDI protocols, all datasets have been anonymized, with no protected health information included. Below, are the specific types of information included in ABIDE I, the data usage agreement, sign up and data download links.

## Setup Dataset Guide

1. Make sure that the path in ``setup.py`` is correct

        main_dir = "/data/data_repo/neuro_img/ABIDE"
        corr_mat_dir = os.path.join(main_dir, "fmri", "processed_ts")
        meta_csv_path = os.path.join(main_dir, "meta", "Phenotypic_V1_0b_preprocessed1.csv")

2. Run ``setup.py``

        python setup.py