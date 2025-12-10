# CSCE-689-GENERATIVE-AI
Synthetic LiDAR Generation using 6G/mm Wave Radar Data 

Github Repository for our Project on Generating LiDAR Data from Camera and Radar using multiple Transformer Architectires 

Project Resources:

1. [Final Presentation Slides](https://docs.google.com/presentation/d/1ae66qx-GZdwHUwWGrRhfW7ADfjm5HjLLOBf55-UGyfc/edit?usp=sharing)
2. [Project Milestone Report 1](https://drive.google.com/file/d/1rjbZBgzG0scCAi6dy1APfUZeEk6K-yhr/view?usp=sharing)
3. [Project Milestone Report 2](https://drive.google.com/file/d/1zHQ9h1ObD8KuTUg-_7qYw3I94qLA8tsg/view?usp=sharing)
4. [Final Project Report](https://drive.google.com/file/d/1HRVdIqcKGyETLwThAXpRSoWCS-0-rOvF/view?usp=sharing)
5. [Models](https://drive.google.com/drive/folders/1pMv__d-SF7U-KyQZ_3TSHyttXOFzManN?usp=sharing)
6. [Dataset for inference](https://drive.google.com/drive/folders/1JorH0UpoWhnsMXzaXI1MBH1M8ZyDWx-w?usp=sharing)

## Running Inference

Each model has a corresponding inference script: inference_{model_name}_model.py

### Input Requirements
Each inference script expects:

1. **Embeddings of the input image**  
   Precomputed vision encoder embeddings included in the dataset

2. **Original `.ply` LiDAR file**  
   Used for loss and metric computation 

3. **Model checkpoint**  
   Download from the “Models” Google Drive folder

