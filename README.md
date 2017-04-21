# ISMIR2017
Reproducible research code for the article submitted to ISMIR 2017.

## Aim of the paper
- Propose a novel industrial musical database
- Cover Song Identification task on the before-mentioned database
- Singer's Gender Classification task on the before-mentioned database

## Tree structure (Description of available files)
- The folder `src/` contains python files necessary to reproduce our algorithm
- The folder `data/` contains a file named `filelist.csv` that lists for each audio file:
    - the unique identifier
    - the artist name
    - the track name
    - the gender tag (female, male, females, males, mixed)
    - the language tag (en, fr, es, it, de, pt, nl)
    - a boolean indicating if features have been extracted for this audio file by:
        1. [YAAFE](https://github.com/Yaafe/Yaafe)
        2. [Marsyas](http://marsyas.info/)
        3. [Essentia](https://github.com/MTG/essentia/)
	4. [Vamp](http://www.vamp-plugins.org)
	5. [harmony-analyser](http://www.harmony-analyser.org)
- The folder `features/` contains features extracted by 
    - [bextract](http://marsyas.info/doc/manual/marsyas-user/bextract.html#bextract) from [Marsyas](http://marsyas.info/) with the following command: 
`bextract -mfcc -zcrs -ctd -rlf -flx -ws 1024 -as 898 -sv -fe`.
    - [harmony-analyser](http://www.harmony-analyser.org) with the following commands:
`java -jar harmony-analyser-script-jar-with-dependencies.jar -a chord_analyser:chord_complexity_distance -s .wav -t 0.07`
`java -jar harmony-analyser-script-jar-with-dependencies.jar -a chroma_analyser:complexity_difference -s .wav -t 0.07`
`java -jar harmony-analyser-script-jar-with-dependencies.jar -a chord_analyser:average_chord_complexity_distance -s .wav -t 0.07`
`java -jar harmony-analyser-script-jar-with-dependencies.jar -a chord_analyser:tps_distance -s .wav -t 0.07`

As concerns features extracted by [YAAFE](https://github.com/Yaafe/Yaafe), [Essentia](https://github.com/MTG/essentia/) and [Vamp](http://www.vamp-plugins.org), they cannot be stored on this github repository because of their inherent size and so are available upon request for direct download.
The command used for extracting features with:
- [YAAFE](https://github.com/Yaafe/Yaafe): `yaafe -r 22050 -f "mfcc: MFCC blockSize=2048 stepSize=1024" --resample -b  output_dir_features input_filename`
- [Essentia](https://github.com/MTG/essentia/): `essentia-extractors-v2.1_beta2/streaming_extractor_music input_filename output_filename`
- [Vamp](http://www.vamp-plugins.org) extracted via [harmony-analyser](http://www.harmony-analyser.org) using JNI wrapper:
`java -jar harmony-analyser-script-jar-with-dependencies.jar -a nnls-chroma:nnls-chroma -s .wav -t 0.07`
`java -jar harmony-analyser-script-jar-with-dependencies.jar -a nnls-chroma:chordino-tones -s .wav -t 0.07`
`java -jar harmony-analyser-script-jar-with-dependencies.jar -a nnls-chroma:chordino-labels -s .wav -t 0.07`
`java -jar harmony-analyser-script-jar-with-dependencies.jar -a qm-vamp-plugins:qm-keydetector -s _wav -t 0.07`
