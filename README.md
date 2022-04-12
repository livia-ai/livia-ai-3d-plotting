Python Script to create 3D plots with Plotly: main.py

-> sentence_embeddings_wien_museum.csv is on Google Drive in folder 04_Data

## Some minor enhancements/questions

- [ ] We don't really need the text pre-processing step for the vis, if I understand correctly
- [ ] Can we sample N records? (Right now, we're picking the first N)
- [ ] Can we add the record ID to the sentence embeddings file? (So we can use it as a standalone dataset)
- [ ] It looks like the sentence embeddings correlate better with the classficiations rather than the subjects. I wonder why that's the case?

## Potential next steps

A "triple sampling" script, that picks (for example) a random record, a handful of nearby records, and a handful of "far away" random records. (E.g. something like min. 45% distance of the populated vectorspace?)

## Some examples for 'nearby' records:

https://sammlung.wienmuseum.at/objekt/685864-1-paar-wanderschuhe/
https://sammlung.wienmuseum.at/objekt/677318-1-paar-pumps/
https://sammlung.wienmuseum.at/objekt/685915-1-paar-sandalen/

or:

https://sammlung.wienmuseum.at/objekt/42062-maennliches-und-weibliches-portraet/
https://sammlung.wienmuseum.at/objekt/46386-not-und-irrsinn/

or:

https://sammlung.wienmuseum.at/objekt/1489623-transportrucksack-fuer-fahrradkurierezustellerinnen-des-essenslieferdienstes-foodora/
https://sammlung.wienmuseum.at/objekt/1073228-filmprojektor-der-beogradska-banka/
https://sammlung.wienmuseum.at/objekt/1072974-schreibmaschine-der-marke-ibm-verwendet-im-buero-der-ausfuellhilfe-des-vereins-jedinstvo-um-1975/

or: 

https://sammlung.wienmuseum.at/objekt/675952-mantelkleid/
https://sammlung.wienmuseum.at/objekt/686341-abendkleid/
https://sammlung.wienmuseum.at/objekt/673796-damenzweiteiler/
