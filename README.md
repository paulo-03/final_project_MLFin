# Stock Return Predictions Using Machine Learning

This repo will focus on **which predictors are relevant for return forecasting?**

**Data** can be easily downloaded 
***[here](https://drive.google.com/file/d/17XtSLaKEcBQFyaM40jfG_nILWIWMfL77/view?usp=share_link)***. It has already been 
merged with the stock price values using the R 
[code](https://github.com/OpenSourceAP/CrossSectionDemos/blob/main/dl_signals_add_crsp.R) proposed by Open Source Asset 
Pricing team.

**Data cleaned** can also directly being downloaded ***[here](https://drive.google.com/file/d/12HDqlbf6EJ2Sx5Ku8ORmR5RNk46IysEq/view?usp=sharing)***. We used MissForest to impute the data and retrieve 
the missing values. Please note that for quality and computaion reasons we kept only the more relevant attributes and 
records. Then the data cleaned is obviously largely smaller, i.e. 6M records and ~200 attributes vs 1.7M records and 
131 attributes, respectively data and data cleaned.

*Authors:* [Mina Attia](https://people.epfl.ch/mina.attia), [Arnaud Felber](https://people.epfl.ch/arnaud.felber), 
[Milos Novakovic](https://people.epfl.ch/milos.novakovic), [Rami Atassi](https://people.epfl.ch/rami.atassi) & 
[Paulo Ribeiro](https://people.epfl.ch/paulo.ribeirodecarvalho)