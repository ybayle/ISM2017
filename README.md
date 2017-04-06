# ISMIR2017
Reproducible research code for the article submitted to ISMIR 2017.

## Aim of the paper
- Propose a novel industrial musical database
- Cover Song Identification task on the before-mentioned database
- Singing Voice's Gender Classification task on the before-mentioned database

## Tree structure (Description of available files)
- The folder src/ contains python files necessary to reproduce our algorithm
- The folder data/ contains a file named `filelist.csv` that lists for each audio file:
    - the unique identifier
    - the artist name
    - the track name
    - the gender tag (female, male, females, males, mixed)
    - the language tag (en, fr, es, it, de, pt, nl)
    - a boolean indicating if features have been extracted for this audio file by:
        1. yaafe
        2. marsyas
        3. essentia
- The folder features/ contains features extracted by [bextract](http://marsyas.info/doc/manual/marsyas-user/bextract.html#bextract) from [Marsyas](http://marsyas.info/) with the following command: `bextract -mfcc -zcrs -ctd -rlf -flx -ws 1024 -as 898 -sv -fe`. The features extracted by yaafe or essentia cannot be stored on this github repository and so are available upon request for direct download.
