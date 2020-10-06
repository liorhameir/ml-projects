# aerial classification model, using pytorch.

classifies 17 classes from AID dataset including:
Bareland, beach, commercial, denseresident, famland, forest, industrial, mediumResidential, moutain, park, parking, playground, pond, port, river, sparse, residental, viaduct.

loss: 0.4787, Accuracy: 84%

With not to much effort this project can be generlize for many other classification tasks on differnent datasets
- main.py - maim runner
- runManager.py - taking care of one run at a time (multiple runs would be usefull in order to find there right hyperparameters)
- utils.py - useful functions - calculate (mean, std) of the dataset for normalization
- CostumDataset.py
- images_to_xml.py - creates one xml file from folder of image folders, ready for training
