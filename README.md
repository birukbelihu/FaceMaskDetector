![GitHub Repo stars](https://img.shields.io/github/stars/BirukBelihu/FaceMaskDetector)
![GitHub forks](https://img.shields.io/github/forks/BirukBelihu/FaceMaskDetector)
![GitHub issues](https://img.shields.io/github/issues/BirukBelihu/FaceMaskDetector)
![GitHub license](https://img.shields.io/github/license/BirukBelihu/FaceMaskDetector)

# Face Mask Detector

A Simple COVID19 Face Mask Detector In A Live Camera Using Computer Vision & Deep Learning.

# Running

To Get Started With Face Mask Detector On Your Local Machine Follow This Simple Steps One By One To Get Up & Running.

Make Sure You Have [Git](https://git-scm.com/) & [Python](https://python.org) Installed On Your Machine.

```
git --version
```

```
python --version
```

# Reminder
Make Sure You're Using **Python Version 3.9-3.12**.

Clone The Repository

```
git clone https://github.com/BirukBelihu/FaceMaskDetector.git
```

Go Inside The Project

```
cd FaceMaskDetector
```

Install Required Dependencies

```
pip install -r requirements.txt
```

Run Face Mask Detector
```
python main.py
```

# Training
To Train Your Own Model Locally Run ```dataset_downloader.py```. It Will Download The Training Dataset To Your Local Machine In The Current Directory Inside dataset Folder.

```
python dataset_downloader.py
```

Once The Dataset Is Downloaded You Can Train The Model Locally By Running ```python model_trainer.py```.

```bash
python model_trainer.py
```

Or You Can Train The Model Using The Provided Jupyter Notebook In The notebook Folder.

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for more details.
