Deep Learning paper modelling periodontal disease for the Canadian Army, supervised by Aya Mitani.

First Timer User Download Instructions
= 
To download and load the file for the first time, follow the steps below:

1. Open Pycharm.
2. Select “Clone Repository” in the top right corner.
3. Copy URL from GitHub and paste into the “URL” Box
4. On the left hand tab, under "Project", right click on "final_code", and near the bottom hover over "Mark Directory as" and select "Sources Root" (it should have a Blue Folder icon beside it).
5. Go to: final_code/deep_learning_dentistry/package_downloader.py 
6. Here, you will see a list of packages that are used in this project. To ensure all are downloaded, hover over any package with a squiggle/tilde under it's name (package names follow "import" command). Then, select "install package __name__".
7. Next, go to: "final_code/data/raw". Here, paste the following data files (ensure their names are unchanged):
- Bleeding.xlsx
- ChartEndo_final.xlsx
- ChartGeneralNew_Final.xlsx
- ChartRestorative_final.xlsx
- DemographicData.xlsx
- Mobility_Furcation_Index_MAG.xlsx
- Pockets.xlsx
- Recessions.xlsx
- Suppuration.xlsx
8. Then, go to: "final_code/deep_learning_dentistry/data_curation/data_curator.py". Run this file by clicking the Run Icon in the top right. This will process and curate all the data from the Raw file.
9. To extract the final dataset, go to: "final_code/deep_learning_dentistry/data_curation/raw_data_curator_long.py" and run. This will produce the final dataset located in "final_code/data" under the name "curated_dataset.csv".

General Navigation Of The Dataset
=
- deep_learning_dentistry is where all the code lies.
- deep_learning_dentistry/data_curation is where all processing and curation of the raw data into one final dataset lies.
- deep_learning_dentistry/deep_learning_analysis/variable_analysis is where variable specific analysis is done
