This respository is linked with the article *Extracting event-related information from a corpus regarding soil industrial pollution* by Chuanming Dong, Philippe Gambette and Catherine Dominguès

It contains:
* `KDIR_tokenization_transformer.py`: a script to transform CamemBERT tokens to TreeTagger tokens
* `dateparser_export_to_BIEO_format.py`: a script to call the search_dates function of the dateparser library to extract dates and format them in a similar way (BIEO format) than our annotation script
* `corpus`: a folder containing an example input file, containing sentences from the BASOL database (https://www.georisques.gouv.fr/articles-risques/basol, see also https://www.data.gouv.fr/fr/datasets/base-des-sols-pollues/) for dateparser_export_to_BIEO_format.py