# ACI-BENCH

##  Introduction 

This repository contains the data and source code for:

**"ACI-BENCH: a Novel Ambient Clinical Intelligence Dataset for Benchmarking Automatic Visit Note Generation". Wen-wai Yim, Yujuan Fu, Asma Ben Abacha, Neal Snider, Thomas Lin, Meliha Yetisgen. Nature Scientific Data, 2023.**

Paper: https://www.nature.com/articles/s41597-023-02487-3

Dataset: https://figshare.com/articles/dataset/aci-bench-corpus_zip/22494601?file=41498793 

```
@article{aci-bench,
  author = {Wen{-}wai Yim and
                Yujuan Fu and
                Asma {Ben Abacha} and
                Neal Snider and Thomas Lin and Meliha Yetisgen},
  title = {ACI-BENCH: a Novel Ambient Clinical Intelligence Dataset for Benchmarking Automatic Visit Note Generation},
  journal = {Nature Scientific Data},
  year = {2023}
}
```

## Data Splits (MEDIQA-CHAT & MEDIQA-SUM Challenges) 

The ACI-BENCH collection consists of full doctor-patient conversations and associated clinical notes and includes the data splits from the [MEDIQA-CHAT 2023](https://sites.google.com/view/mediqa2023/clinicalnlp-mediqa-chat-2023) and [MEDIQA-SUM 2023](https://www.imageclef.org/2023/medical/mediqa) challenges: 
```
TRAIN: 67
VALID: 20
TEST1: 40 ( MEDIQA-CHAT TASK B test set )
TEST2: 40 ( MEDIQA-CHAT TASK C test set )
TEST3: 40 ( MEDIQA-SUM TASK C test set )
```

**Speaker Tagging:** Some ACI-BENCH subsets include swapped speaker tags (e.g., [patient], [doctor]) that originate from the Automatic Speech Recognition (ASR) process. This reflects a known and common limitation of ASR-based transcripts, which we intentionally left uncorrected, as our goal was to provide a realistic dataset that captures such errors.
Automated methods (including LLM-based approaches, such as [Max's method](https://github.com/mkieffer1107/aci-bench_upload/tree/main/corrections)) can be used to identify and correct these speaker-tagging issues.

## License

The data here is published under a Creative Commons Attribution 4.0 International License (CC BY): https://creativecommons.org/licenses/by/4.0/

Please cite our Nature Scientific Data paper if you use the full dataset or any subset of ACI-BENCH. 


## Contact

    -  Asma Ben abacha (abenabacha at microsoft dot com)
     - Wen-wai Yim (yimwenwai at microsoft dot com)

