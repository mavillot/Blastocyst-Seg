# 🌟 Blastocyst Segmentation 🌟

This project focuses on segmenting blastocyst images and extracting the exact moment of the blastocyst formation from a video using deep learning models. 🐾

## 📂 Table of Contents
In the repository, you can find the following files and directories:

- [Getting Started](#-getting-started)
- [Usage](#-usage-instructions)
- [Tutorials](#-tutorials)
- [Checkpoints](#-checkpoints)
- [Cite This Work](#-cite-this-work)
- [Papers](#-papers)
- [Contact](#-contact)

## 🎯 Getting Started
To get started with this project, follow these steps::
1. **Clone the repository:**
      ```bash
   git clone https://github.com/mavillot/Blastocyst-Seg.git
2. **Navigate to the project directory:**
   ```bash
   cd Blastocyst-Seg
3. **Install the virtual environment with the required dependencies:**
   ```bash
   pip install -r requirements_torch.txt
   ```
   Only for the replicated Harun encoder:
   ```bash
   pip install -r requirements_tensorflow.txt
   ```
## 💻 Usage Instructions
Now that everything is set, we can start segmenting our blastocysts or extracting the blastocyst moment!!
- **Segmentation of a blastocyst image:**
  For segmenting a blastocyst image you will need an image. In the case you don't have any, I provide you of an image: *img.png*.
  Then run the following:
  ```bash
   python segmentation.py path_img path_weights
   ```
   path_img: path of the image you want to segment.
   path_weights: path of the weights of the model.
   This script will generate an image: *prediction.png* with the mask segmenting the blastocyst structures: ZP, TE, ICM.
- **Extracting the exact moment of the blastocyst formation:**
  ```bash
   python blasto_moment.py path_video path_weights
   ```
## 🤓 Tutorials
In the `Notebooks` directory, you’ll find detailed Jupyter notebooks that explain and demonstrate how to:
1. **Train the Model**: Step-by-step code and explanations for setting up and training our model using the dataset.
2. **Test and evaluate the Results**: Insights and metrics for analyzing model predictions, with visualizations and examples.
3. **Inference**: Guidelines on how to use the model with an image for predictions.
Each notebook is designed to help you reproduce each method and understand the workflow from data preparation to final evaluation. Just navigate to `Notebooks/` and open the corresponding notebook to get started!

## 💾 Checkpoints

| Model                | Download link                                       | 
|------------------------|---------------------------------------------------|
| Hrnet                | [Checkpoint](https://github.com/mavillot/Blastocyst-Seg/releases/download/hrnet/hrnet.pth)  |
| Unet                | [Checkpoint](https://github.com/mavillot/Blastocyst-Seg/releases/download/unet/unet.pth)  | 
| Deeplab                | [Checkpoint](https://github.com/mavillot/Blastocyst-Seg/releases/download/deeplab/deeplab.pth)  |
| Harun - Unet - TE      | [Checkpoint](https://github.com/mavillot/Blastocyst-Seg/releases/download/harun_TE/unet_TE.zip)  |
| Harun - Unet - ICM      | [Checkpoint](https://github.com/mavillot/Blastocyst-Seg/releases/download/harun_ICM/unet_ICM.zip)  |


## 📖 Cite this work
```bibtex
@inproceedings{villota-2024,
title={Image Processing and Deep Learning Methods for the Semantic Segmentation of Blastocyst Structures},
author={Villota, María and Ayensa-Jiménez, Jacobo and Doblaré, Manuel and Heras, Jónathan},
volume={14640 LNAI}, ISBN={9783031627989},
ISSN={1611-3349},
DOI={10.1007/978-3-031-62799-6_22},
booktitle={Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics)},
publisher={Springer Science and Business Media Deutschland GmbH},
year={2024},
pages={213-222}}
```

## 📑 Papers
### [Computer Vision for Automatic Identification of Blastocyst Structures and Blastocyst Formation Time in In-Vitro Fertilization.](http://dx.doi.org/10.2139/ssrn.5027594)
María Villota Miranda, Jacobo Ayensa-Jiménez, Clara Malo, Antonio Urries, Manuel Doblaré, Jónathan Heras 
### [Image Processing and Deep Learning Methods for the Semantic Segmentation of Blastocyst Structures](https://link.springer.com/chapter/10.1007/978-3-031-62799-6_22)
María Villota, Jacobo Ayensa-Jiménez, Manuel Doblaré, Jónathan Heras 
### [Segmentation of the blastocyst structures using Image Processing and Machine Learning tools.](https://www.ctresources.info/ccc/paper.html?id=9845)
María Villota, Jacobo Ayensa-Jiménez, Manuel Doblaré, Jónathan Heras 


## 📫 Contact
Feel free to reach out if you have any questions!
- Email: [mvillota@iisaragon.es](mvillota@iisaragon.es)
- GitHub: [mavillot](https://github.com/mavillot)
- Google Scholar: [María Villota](https://scholar.google.es/citations?hl=es&user=IeGlMh8AAAAJ)
- LinkedIn: [María Villota Miranda](https://www.linkedin.com/in/maria-villota-miranda/)
