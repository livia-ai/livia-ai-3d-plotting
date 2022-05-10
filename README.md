## Module: livia
The folder "livia" contains a standalone module containing the main functions of this project. 
It is structured as two classes in two separate Python files:
- **Embedding:** provides all functions necessary to generate sentence embedding given a dataframe containing text data and a selection of columns

- **Triplet:** provides all functions needed to generate triplets of the form (sample, similar, disimilar) given sentence embeddings of a dataset

## Folders and their Scripts
- **0_data_preparation**

  Contains a script that generates embeddings and one that downprojects existing embeddings to compress it

- **1_datasets_3d_plots**

  Python Script to create 3D plot of sentence embedding of a dataset with Plotly.

- **2_bar_plots_column_counts**

  Create bar plots that show value counts of one dataframe column.

- **3_clustering**

  Performs k-means clustering on n dimensional sentence embedding.

- **4_combined_dataset**

  Not yet functional. <br>
  Tries takes combined dataset and tries to analyse results
  
- **5_triplet_generation** 
  
  Generates triplets(original sample, similar sample, dissimilar sample) given a subset of the the sentence embeddings. <br>
  Also contains some functions for visualization.
 

## Some minor enhancements/questions

- [x] We don't really need the text pre-processing step for the vis, if I understand correctly
- [ ] Can we sample N records? (Right now, we're picking the first N)
- [x] Can we add the record ID to the sentence embeddings file? (So we can use it as a standalone dataset)
- [ ] It looks like the sentence embeddings correlate better with the classficiations rather than the subjects. I wonder why that's the case?

## Some examples for 'nearby' records:

- https://sammlung.wienmuseum.at/objekt/685864-1-paar-wanderschuhe/
- https://sammlung.wienmuseum.at/objekt/677318-1-paar-pumps/
- https://sammlung.wienmuseum.at/objekt/685915-1-paar-sandalen/

or:

- https://sammlung.wienmuseum.at/objekt/42062-maennliches-und-weibliches-portraet/
- https://sammlung.wienmuseum.at/objekt/46386-not-und-irrsinn/

or:

- https://sammlung.wienmuseum.at/objekt/1489623-transportrucksack-fuer-fahrradkurierezustellerinnen-des-essenslieferdienstes-foodora/
- https://sammlung.wienmuseum.at/objekt/1073228-filmprojektor-der-beogradska-banka/
- https://sammlung.wienmuseum.at/objekt/1072974-schreibmaschine-der-marke-ibm-verwendet-im-buero-der-ausfuellhilfe-des-vereins-jedinstvo-um-1975/

or: 

- https://sammlung.wienmuseum.at/objekt/675952-mantelkleid/
- https://sammlung.wienmuseum.at/objekt/686341-abendkleid/
- https://sammlung.wienmuseum.at/objekt/673796-damenzweiteiler/

## Potential next steps

A "triple sampling" script, that picks (for example) a random record, a handful of nearby records, and a handful of "far away" random records. (E.g. something like min. 45% distance of the populated vectorspace?)
