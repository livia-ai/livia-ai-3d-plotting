## Scripts and their functionalities
- **basic_3d_plotting.py**
  
  Python Script to create 3D plot of sentence embedding with Plotly. <br>
  Works now with all 3 datasets, **down projected version of all sentence embeddings(with museum-id for every sample) is on google drive.**

- **clustering.py**

  Performs k-means clustering on n dimensional sentence embedding.
  
- **triplets.py** 
  
  Generates triplets(original sample, similar sample, dissimilar sample) given a subset of the the sentence embeddings. <br>
  Also contains some functions for visualization.
  
- **combine_data.py**

  Not yet functional. <br>
  Tries to combine all 3 datasets into one big one and peform the same tasks as above.
  
- **downproject_se.py**

  Loads full sentence embedding, down-projects it to d dimensions and saves ut with the museum-id for every sample .

- **bar_plots_counts.py**

  Create bar plots that show value counts of one dataframe column.
 

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
