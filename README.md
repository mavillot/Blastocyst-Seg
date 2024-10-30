# 🌟 Blastocyst Segmentation 🌟

This project focuses on segmenting blastocyst images and extracting the exact moment of the blastocyst formation from a video using deep learning models. 🐾

## 📂 Repository Contents
In the repository, you can find the following files and directories:
```bash
├──Blastocyst-Seg/
├──────Notebooks/
|       ├── Image_Process_ZP.ipynb
|       ├── Image_Process_TE&ICM.ipynb
|       ├── Harun_unet_TE.ipynb
|       ├── Harun_unet_ICM.ipynb
|       ├── Unet.ipynb
|       ├── HRnet.ipynb
|       └── Deeplab-resnet50.ipynb
├────── experiments/
|       ├── kfold_consistency.py
|       ├── kfold_consistency_Harun.py
|       ├── kfold_dataset.py
├────── utils/
|   
├────── requirements_torch.txt
├────── requirements_tensorflow.txt
├────── segmentation.py
├────── blasto_moment.py
├────── img.png
├────── prediction.png
└────── video.mp4
```

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

## 📫 Contact
Feel free to reach out if you have any questions!
- Email: [mvillota@iisaragon.es](mvillota@iisaragon.es)
- GitHub: [mavillot](https://github.com/mavillot)
