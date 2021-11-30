# ADHD 200

Attention Deficit Hyperactivity Disorder (ADHD) affects at least 5-10% of school-age children and is associated with substantial lifelong impairment, with annual direct costs exceeding $36 billion/year in the US. Despite a voluminous empirical literature, the scientific community remains without a comprehensive model of the pathophysiology of ADHD. Further, the clinical community remains without objective biological tools capable of informing the diagnosis of ADHD for an individual or guiding clinicians in their decision-making regarding treatment.

The [ADHD-200 Sample](http://fcon_1000.projects.nitrc.org/indi/adhd200/) is a grassroots initiative, dedicated to accelerating the scientific community's understanding of the neural basis of ADHD through the implementation of open data-sharing and discovery-based science. Towards this goal, we are pleased to announce the unrestricted public release of 776 resting-state fMRI and anatomical datasets aggregated across 8 independent imaging sites, 491 of which were obtained from typically developing individuals and 285 in children and adolescents with ADHD (ages: 7-21 years old). Accompanying phenotypic information includes: diagnostic status, dimensional ADHD symptom measures, age, sex, intelligence quotient (IQ) and lifetime medication status. Preliminary quality control assessments (usable vs. questionable) based upon visual timeseries inspection are included for all resting state fMRI scans.

In accordance with HIPAA guidelines and 1000 Functional Connectomes Project protocols, all datasets are anonymous, with no protected health information included.

## Setup Dataset Guide

1. Make sure that the path in ``setup.py`` is correct

        main_dir = "/data/data_repo/neuro_img/ADHD-200"
        corr_mat_dir = os.path.join(main_dir, "fmri", "processed_corr_mat")
        phenotypics_path = os.path.join(main_dir, "fmri", "raw")

2. Run ``setup.py``

        python setup.py