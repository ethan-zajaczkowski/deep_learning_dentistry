Deep Learning paper modelling periodontal disease for the Canadian Army, supervised by Aya Mitani.

First Timer User Download Instructions
= 
To download and load the file for the first time, follow the steps below:

1. Open Pycharm.
2. Select “Clone Repository” in the top right corner.
3. Copy URL from GitHub and paste into the “URL” Box
4. On the left hand tab, under "Project", right click on "final_code", and near the bottom hover over "Mark Directory as" and select "Sources Root" (it should have a Blue Folder icon beside it).
5. Go to: deep_learning_dentistry/package_downloader.py 
6. Here, you will see a list of packages that are used in this project. To ensure all are downloaded, hover over any package with a squiggle/tilde under it's name (package names follow "import" command). Then, select "install package __name__".
7. Next, go to: "deep_learning_dentistry/data/raw". You will see the following folders:
- bleeding
- chart_general
- chart_restorative
- demographic_data
- mobility_furcation_index_mag
- pockets
- recessions
- suppuration
  
In each of the folders, put all related files into each folder. For example, in bleeding, put the following files in:
- Bleeding.xlsc
- 2023_Bleeding.xlsx
  
Note: Inside of chart_restorative, only put ChartRestorative_final.xlsx here. Don't put any other file here.

8. Then, go to: "deep_learning_dentistry/deep_learning_dentistry/data_curation/data_curator.py". Run this file by clicking the Run Icon in the top right. This will process and curate all the data from the Raw file.
9. We first want to create the dataset in wide format. Go to "deep_learning_dentistry/deep_learning_dentistry/data_curation/dataset_curator_wide.py" and run. This will produce the dataset in a wide format within code/data/full_dataset
10. Then, we want to create the dataset in long format. Go to "deep_learning_dentistry/deep_learning_dentistry/data_curation/dataset_curator_long.py" and run. This will produce the dataset in a wide format within deep_learning_dentistry/data/full_dataset. Note: This takes a while to run. In the terminal, you will see the exact research_id being examined. There are about 9000. It takes about 15 minutes to run.

General Navigation Of The Dataset
=
- deep_learning_dentistry is where all the code lies.
- deep_learning_dentistry/data_curation is where all processing and curation of the raw data into one final dataset lies.
- deep_learning_dentistry/data_analysis is where any additional variable analysis was conducted, like clinical attachment loss
- deep_learning_dentistry/deep_learning_model is where the actual model will be
