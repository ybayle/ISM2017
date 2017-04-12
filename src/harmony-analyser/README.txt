harmony-analyser-script-jar-with-dependencies.jar
===

Move the .jar script to the folder with .wav files to analyse.

Please refer to https://github.com/lacimarsik/harmony-analyser for installation and troubleshooting.

7 analysis needed to extract all harmony features (in this order)

java -jar harmony-analyser-script-jar-with-dependencies.jar -a nnls-chroma:nnls-chroma -s .wav -t 0.07
java -jar harmony-analyser-script-jar-with-dependencies.jar -a nnls-chroma:chordino-tones -s .wav -t 0.07
java -jar harmony-analyser-script-jar-with-dependencies.jar -a nnls-chroma:chordino-labels -s .wav -t 0.07
java -jar harmony-analyser-script-jar-with-dependencies.jar -a qm-vamp-plugins:qm-keydetector -s _wav -t 0.07
java -jar harmony-analyser-script-jar-with-dependencies.jar -a chord_analyser:chord_complexity_distance -s .wav -t 0.07
java -jar harmony-analyser-script-jar-with-dependencies.jar -a chroma_analyser:complexity_difference -s .wav -t 0.07
java -jar harmony-analyser-script-jar-with-dependencies.jar -a chord_analyser:average_chord_complexity_distance -s .wav -t 0.07
java -jar harmony-analyser-script-jar-with-dependencies.jar -a chord_analyser:tps_distance -s .wav -t 0.07

Features extracted are:

1. Chroma features (Vamp)
2. Chord labels (Vamp)
3. Chord tones (Vamp)
4. Key detection (Vamp)
5. CCD Distance time series from consecutive chords (harmony-analyser)
6. Average CCD Distance per song (harmony-analyser)
7. Chroma vector distance (complexity difference) time series (harmony-analyser)
8. TPS Distance time series from consecutive chords (harmony-analyser)
