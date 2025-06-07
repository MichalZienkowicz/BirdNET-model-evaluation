# BirdNET-model-evaluation
Evaluation of BirdNET-Analyzer v2.1.0 model performance on the database of polish birds audio recordings


## BirdNET 
**BirdNET** is a deep learning-based system developed by the K. Lisa Yang Center for Conservation Bioacoustics at the Cornell Lab of Ornithology and the Chair of Media Informatics at Chemnitz University of Technology. Its goal is detecting and identifying bird vocalizations in audio recordings. It uses a neural network trained on thousands of bird species worldwide to return species predictions with associated confidence scores based on the audio content.

**BirdNET-Analyzer** is the command-line and Python interface for running the BirdNET model locally and performing acoustic analyses. In the project, model BirdNET-Analyzer v2.1.0 for Windows OS has been used.

You can read more about BirdNET here:
* BirdNET Website: https://birdnet.cornell.edu
* BirdNET-Analyzer GitHub repo: https://github.com/birdnet-team/BirdNET-Analyzer
* BirdNET-Analyzer Documentation: https://birdnet-team.github.io/BirdNET-Analyzer

## Data Base
Model performance was analyzed on database containing recordings gathered in Puszcza Niepołomnicka (Cracow, Poland) on May of 2024. Descriptions of audio files contained following species:
* "Sylvia atricapilla"
* "Erithacus rubecula"
* "Phylloscopus collybita"
* "Fringilla coelebs"
* "Troglodytes troglodytes"
* "Oriolus oriolus"
* "Parus major"
* "Phasianus colchicus"
* "Acrocephalus arundinaceus"
* "Turdus philomelos"
* "Phylloscopus sibilatrix"
* "Emberiza citrinella"
* "Phylloscopus trochilus"
  
Database consisted of 768kb/s .wav files, initialy stereo (convertion to mono is included in the code), of length not greater than several minutes. For every file, certain vocalizations were labeled by an ornitologist. It's important to notice, that files consisted of many more vocalizations which were unlabeled.
Additionally file with specie "Hippolais icterina" was present, but it was not used due to missing information about vocalizations timing.

It has been made sure that the BirdNET species database includes all of the species present in the labels.

## Model Evaluation
BirdNET-Analyzer allows for predicions based on 3 seconds long windows. The evaluation method proposed in this project was to compare true and predicted labels for every seconds of an audio recording. For this reason per-second true and predicted label files were created. For the true labels, even if vocalisation of given specie was present only in fraction of a second, this specie was assigned as a true label for this second, and used for model evaluation. 

If predicted label for a given second included true label, the recognition was marked as true positive (**TP**). If true label had no corresponding prediction, recognition was marked false negative (**FN**). Complete number of seconds with true labels of given specie was named number of recognitions (**NR**).

Due to multiple unlabeled vocalizations in audio files, it was **not possible to asses the number of false positives** (or true negatives). Therefore the only metric calculated was recall (Recall =   (TP / (TP + FN)). It's important to note it gives only partial information about the model performance without its complementary metric - precision.

Table 1: True labels from ornitologist
| start_time | end_time | specie                 |
|------------|----------|------------------------|
| 0.000      | 3.204    | Sylvia atricapilla     |
| 3.651      | 5.453    | Phylloscopus collybita |
| 7.132      | 8.875    | Erithacus rubecula     |

Table 2: Example one second true and predicted label files, with resulting model metrics
| time    | true label                                 | example predictions                    |
|---------|--------------------------------------------|----------------------------------------|
| 0.0-1.0 | Sylvia atricapilla                         | Sylvia atricapilla                     |
| 1.0-2.0 | Sylvia atricapilla                         | no predictions                         |
| 2.0-3.0 | Sylvia atricapilla                         | Sylvia atricapilla, Erithacus rubecula |
| 3.0-4.0 | Sylvia atricapilla, Phylloscopus collybita | Sylvia atricapilla                     |
| 4.0-5.0 | Phylloscopus collybita                     | Phylloscopus collybita                 |
| 5.0-6.0 | Phylloscopus collybita                     | Columba oenas, Phylloscopus collybita  |
| 6.0-7.0 | unspecified                                | Sylvia atricapilla, Picus viridis      |
| 7.0-8.0 | Erithacus rubecula                         | Erithacus rubecula, Picus viridis      |
| 8.0-9.0 | Erithacus rubecula                         | Erithacus rubecula                     |

Table 3: Model metrics resulting from example true labels and predictions presented in Table 2.
| Specie                 | NR (true recognitions) | TP (accurate predictions) | FN (lack of accurate prediction for true label) | Recall |
|------------------------|------------------------|---------------------------|-------------------------------------------------|--------|
| Sylvia atricapilla     | 4                      | 3                         | 1                                               | 0.75   |
| Erithacus rubecula     | 2                      | 2                         | 0                                               | 1      |
| Phylloscopus collybita | 3                      | 3                         | 0                                               | 1      |


BirdNET-Analyzer allows for changing the input parameters. Model performance was asessed using different values of 7 of them:
* **overlap:**
Value specifies how many seconds of overlap will there be between 3 seconds windows for model analysis. With deafould value of 0.0, There is no overlap, and analysis window begins where the previous is ends. increasing this value, allows for more dense windows spacing. It increases the time of calculations, as there are more windows to run analysis for, but increases chance of feeding the whole vocalization into the model window. For example when vocalization lasts from 2.0-4.0 second, without overlap increased, it would be halved and analyzed only as one second fragments in window 0.0-3.0 and 3.0-6.0.

Table 4. Example windowing with different overlap values
| Overlap     | Windowing                                   |
|-------------|---------------------------------------------|
| overlap 0.0 | -> 0.0-3.0 -> 3.0-6.0                       |
| overlap 1.0 | -> 0.0-3.0 -> 2.0-5.0 -> 4.0-7.0            |
| overlap 2.0 | -> 0.0-3.0 -> 1.0-4.0 -> 2.0-5.0 -> 3.0-6.0 |

* **min_conf:**
This value sets a threshold of confidence, below which model predictions will be rejected. With low min_conf values, the Analyzer outputs large amount of labels, of which some have low "probablilty" (confidence is not the probability itself) of being accurate. High value of min_conf, causes model to output only the most probable labels.

* **measurement metadata:**
Model allows for specifying time and location of analyzed recording, which can reduce the amount of species taken into consideration during predictions, to those characteristic to the place and time.
Model was tested with and without specifying the measurement details. The result however were exactly the same, and specifying location and time had no influance on the model performance.

## File Structure
For running the model, the folowing structure of directories should be provided (root_path is a path to the project directory in Google Drive)

```
root_path/
├── BirdNET_model_evaluation.ipynb
├── labels_scientific.xlsx
├── Data_converted/
│   ├── Data_UFT_mono/
│   │   ├── 033 A Kapturka/
│   │   │   ├── Etykieta Utworu.txt
│   │   │   └── VOC_140120-0033-Kapturka.wav
│   │   └── ... (other recording subfolders)
│   └── Data_with_missing_labels/
│       └── 046 A Zaganiacz/
│           └── VOC_140120-0046_zaganiacz.wav
```

After running the model, if analysis results are not deleted (which is an option in the Colab Notebook), fallowing file structure is expected:

```
root_path/
├── BirdNET_model_evaluation.ipynb
├── labels_scientific.xlsx
├── labels_from_files.xlsx
├── reports_summary.xlsx
├── Reports/
│   ├── report_Sylvia atricapilla_metadata1.txt
│   └── ... (other report files)
├── Examples/
│   ├── one_sec_example_predictions - 038 A Kapturka (pierwiosnek).txt
│   └── example_predictions - 038 A Kapturka (pierwiosnek).txt
├── Data_converted/
│   ├── Data_UFT_mono/
│   │   ├── 033 A Kapturka/
│   │   │   ├── Etykieta Utworu.txt
│   │   │   └── VOC_140120-0033-Kapturka.wav
│   │   └── ... (other recording subfolders)
│   ├── Data_with_missing_labels/
│   │   └── 046 A Zaganiacz/
│   │       └── VOC_140120-0046_zaganiacz.wav
│   └── Data_UFT_mono_for_processing/
│       ├── 033 A Kapturka/
│       │   ├── Etykieta Utworu.txt
│       │   ├── VOC_140120-0033-Kapturka.wav
│       │   ├── true_033 A Kapturka.txt
│       │   ├── metadata1_033 A Kapturka.txt
│       │   ├── one_sec_metadata1_033 A Kapturka.txt
│       │   ├── ... (other metadata labels)
│       │   └── ... (other metadata one_sec labels)
│       └── ... (other processed subfolders)
```

