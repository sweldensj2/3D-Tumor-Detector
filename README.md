[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/ifbeTrPr)
# e6691-2024Spring-project Project Files and Folders Overview
This project’s goal is to explore the application of semantic segmentation models in tumor detection using medical imaging, with a focus on MRI scans. Leveraging the U-Net architecture, originally designed for biomedical image segmentation, we developed a model tailored to analyze 3D MRI data. The motivation behind this research lies in the potential to enhance medical diagnostics by automating tumor detection processes, enabling faster and more accurate assessments. By training the model on a dataset comprising 3D MRI scans and corresponding segmentation masks, we aimed to provide medical professionals with a tool for rapid evaluation of patient data. Results from the project demonstrated the original goal was accomplished and the feasibility of the approach, with the developed model accurately predicting tumor pixels and generating informative visualizations. However, differences in performance metrics compared to previous studies suggest areas for further investigation and refinement. Overall, this project represents a significant step towards harnessing computer vision technology to improve tumor detection and patient care in oncology.
![10_3D_Visualization_90p](https://github.com/ecbme6040/e6691-2024spring-project-jwss-jws2215/assets/144495665/5d3568bf-f5a8-4c8a-b4a5-fa1ccd3ea872)

## Datasets
- **Description**: Contains the BraTS2020 dataset. The .gitignore prevents it from being pushed to the repo.
- **Link**: [Dataset](https://www.med.upenn.edu/cbica/brats2020/data.html)

## Results
- **Description**: Folder containing uploaded results. The folder contains videos of prediction overlays of patients 10 and 15, the output data for MATLAB to process, and it contains images of the 3D visualizations from MATLAB of both patients 10 and 15. 

## data_inspection.ipynb
- **Description**: Notebook starting the dataset inspection. This file loads essential packages, defines paths, inspects data samples, and processes them into correct folders for training and validation. Additionally, it displays random 2D slices from each image type and prints unique values for segmentation and image modalities.

## medical_visualization_tool.ipynb
- **Description**: Notebook that visualizes predictions from the model. This notebook has a variety of functions which provide the user with the ability to visualize a patients tumor prediction. It imports a model and then performs a layer by layer analysis of the respective MRI. It creates a video output of the MRI with prediction overlay, and it also can create output for the MATLAB program visualizer3D.m to proecss and visualize. 

## model_dice_iou.ipynb
- **Description**: Notebook calculates the performance evaluation metrics on the test data.

## seg_models.py
- **Description**: Python script containing the UNet model implementation I replicated, and it also contains a copy of the implemntation from the reference paper. However the copy is unused.

## training.ipynb
- **Description**: Notebook that is able to train the U-Net models. It is able to load in previous models and continue the progress while recording the losses. Additionally it saves the best model based upon the lowest validation loss. 

## visualizer3D.m
- **Description**: MATLAB script for generating 3D visualizations of a tumor. It can create a 3D model to scroll through based upon a desired confidence level. Additionally it can create a video of the same isosurface. 

## E6691.2024Spring.JWSS.report.jws2215.pdf
- **Description**: Final Report
  
## Saved Models (Google Drive)
- **Description**: Link to saved models on Google Drive.
- **Link**: [Saved Models](https://drive.google.com/file/d/1FuDpOWiBS80hauSfiT39cOldma9NDcLW/view?usp=drive_link)

## Visualizations (Youtube)
- **Description**: Link to visualizations on Youtube.
- **Link**: [Youtube](https://www.youtube.com/@JohanWillemSchulzSweldens)

## Legacy Code
- **Description**: Contains code from the old repository, mainly the display_data.ipynb which was an initiail effort. It used a smaller 3 channel RGB 2D dataset, but it demonstrated the feasiliblty of the project. 
