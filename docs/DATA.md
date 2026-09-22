# Data and preprocessing

The manuscript evaluates MMLACO-ARRW on real-world multi-label datasets. This repository does **not** redistribute third-party datasets.

## Required local format
The reproducibility materials expect a headerless CSV: first d columns are the feature representation and the next q columns are binary multi-label targets.

For Scene, the manuscript reports 2,407 instances, 294 features, 6 labels, and 40 selected features.

Before running manuscript experiments, verify that the local CSV has the same instances, feature ordering, label ordering, and preprocessing used for the reported results.

## Dataset provenance
The manuscript identifies the MULAN multi-label dataset collection as the source of the benchmark datasets:
https://mulan.sourceforge.net/datasets-mlc.html

Tsoumakas, G., Spyromitros-Xioufis, E., Vilcek, J., & Vlahavas, I. (2011). MULAN: A Java library for multi-label learning. Journal of Machine Learning Research, 12, 2411–2414.

Do not upload third-party datasets unless redistribution is permitted.
