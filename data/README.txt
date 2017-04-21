filelist.csv

Contains 2975 Recisio cover songs (karaoke versions of known songs), with:

- tags: gender, language, back vocal
- Information whether yaafe, marsyas, essentia, vamp and harmony-analyser features were extracted successfully 
- Information whether we have the origin song (MP3 format, original title from which the karaoke was made)

More information

- 614 songs are (feat_yaafe = 1 AND feat_marsyas = 1 AND feat_essentia = 1 AND feat_vamp = 1 AND feat_ha = 1)
- 54 songs are (origin = 1)
- to choose origin songs we have used a filter:

artist NOT IN ('christmas-carol', 'traditional', 'nursery-rhyme', 'comptine', 'happy-birthday-songs', 'mexican-traditional')
