# Salutary Labeling with Zero Human Annotation
The code for paper "Salutary Labeling with Zero Human Annotation." will be uploaded to this repository.

# Abstract
Active learning strategically selects informative unlabeled data points and queries their ground truth labels for model training. The prevailing assumption underlying this machine learning paradigm is that acquiring these ground truth labels will optimally enhance model performance. However, this assumption may not always hold true or maximize learning capacity, particularly considering the costly labor annotations required for ground truth labels. In contrast to traditional ground truth labeling, this paper proposes salutary labeling, which automatically assigns the most beneficial labels to the most informative samples without human annotation. Specifically, we utilize the influence function, a tool for estimating sample influence, to select newly added samples and assign their salutary labels by choosing the category that maximizes their positive influence. This process eliminates the need for human annotation. Extensive experiments conducted on nine benchmark datasets demonstrate the superior performance of our salutary labeling approach over traditional active learning strategies. Additionally, we provide several in-depth explorations and practical applications of large language model (LLM) fine-tuning.

# Prerequisites
- python >= 3.6.8
- pytorch ==>=1.7.0
- torchvision == >=0.5.0
- [modAL](https://github.com/modAL-python/modAL)
- [ucimlrepo](https://github.com/uci-ml-repo/ucimlrepo)
- numpy, scipy, PIL, argparse, tqdm, pandas,prettytable,scikit-learn,webcolors,matplotlib,opencv-python,numba

# Datasets
The "bank" and "diabetes" datasets are included in this code and other datasets can be loaded from the [UCI Repository](https://archive.ics.uci.edu/) with their API [ucimlrepo](https://github.com/uci-ml-repo/ucimlrepo).

# Running
Run the code for binary dataset "bank"
> python main_binary.py --dataset bank --diabetes  n

Run the code for multi-class dataset "wine_quality"
> python main_multi.py --dataset wine --diabetes  n


# Acknowledgement
This project is built on the open-source [InfDataSel](https://github.com/anshuman23/InfDataSel) and [influence function](https://github.com/Cyrus9721/Characterizing_graph_influence) implementations. Thank the authors for their excellent work.

