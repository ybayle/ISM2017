filelist.csv

Contains 2975 cover songs (karaoke versions of known songs), with:

- tags: gender, language, back vocal
- Information whether yaafe, marsyas and essentia features were extracted successfully 
- Information whether we have the origin

More information

- 950 songs are (feat_yaafe = 1 AND feat_marsyas = 1 AND feat_essentia = 1)
- to choose origin we have used a filter:

artist NOT IN ('christmas-carol', 'traditional', 'nursery-rhyme', 'comptine', 'happy-birthday-songs', 'mexican-traditional')

- 77 songs are in the origin (origin <> 0)
