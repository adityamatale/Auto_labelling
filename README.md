# 🚀 **Auto Video Annotation System**
Manual video annotation is a slow, labor-intensive, and costly process, often leading to inconsistencies. Traditional methods struggle to scale efficiently, making it difficult to generate high-quality, large-scale annotated datasets across different domains.  
  

## 🌜 **Table of Contents**
1.  [Introduction](#1-introduction)
2.  [Video Processing Features](#2-video-processing-features)
3.  [Tech Stack](#3-tech-stack)
4.  [Installation](#4-installation)
5.  [Usage](#5-usage)
6.  [API Endpoints](#6-api-endpoints)
7.  [Project Structure](#7-project-structure)

---
  
## 📌 **1. Introduction**

### 💡 **Solution Overview**  
This **Auto Video Annotation System** leverages state-of-the-art AI models to automate video labeling, improving efficiency and scalability while maintaining accuracy.

### 🔥 **Key Features**  
✅ **Automated Annotation Pipeline** – Uses advanced AI model [(*Grounding DINO*)](https://github.com/IDEA-Research/GroundingDINO) to detect and label objects in videos, reducing manual effort.  

✅ **Scalable & Efficient System** – Built with a lightweight, high-performance FastAPI backend for seamless integration and large-scale data processing.  

✅ **Optimized Video Processing** – Implements frame segmentation [(*SAM*)](https://github.com/facebookresearch/segment-anything) and propagation [(*SAMv2*)](https://github.com/SauravMaheshkar/samv2) to minimize redundant computation while maintaining accuracy.  

✅ **Reliable Info Monitoring** – Includes (*Supervision*) for annotation handling, and robust logging for debugging and consistency.  

---

## 🎥 **2. Video Processing Features**

🎯 **Zero-shot detection → labeling → segmentation**  

🎮 **Customizable FPS** – Users can adjust the frames per second (FPS) to optimize processing time.  

⏸️ **Interrupts for annotation refinement**:  
   🔹 **STOPPROCESS** – Users can stop the segmentation at any point and retrieve the segmented video up to that point.  
   
   🔹 **STOPDETECTION** – Users can halt new object detection while continuing segmentation of previously detected objects across the video.  
   
   🔹 **Refinement Interrupt** – Users can refine tight segmentation masks by adding positive/negative click labels for specific objects.  

---

## 🛠️ **3. Tech Stack**

| 🖥️ Technology | 🔍 Purpose |
|------------|---------|
|  **Python** | Primary development language |
|  **FastAPI** | API framework for seamless integration |
|  **Supervision** | Utility for managing and visualizing annotations |
|  **Threading** | Enables parallel video processing in the background |
|  **Logging** | Capturing runtime events for debugging and monitoring |
|  **OpenCV (cv2)** | Handles video processing and frame extraction |
|  **SAM (Segment Anything Model)** | Automatic object segmentation |
|  **SAMv2** | Video propagation functionality for *MASKS* |
|  **Grounding DINO** | Zero-shot object detection and labeling |

---

## ⚙️ **4. Installation**

Follow these steps to set up the project:

```sh
# Clone the repository
git clone https://git.acldigital.com/ai-ml/autolabelling.git

# Navigate to the project directory
cd Auto_labelling/

# Run the setup script
bash setup.sh
```

---

## 🚀 **5. Usage**

### ▶️ **Run the application**
```sh
python App.py
```

### 📝 **API Calls**  
Check the `api_call.txt` file for example API request commands.

---

## 🔗 **6. API Endpoints**

| 📡 Method | 🔗 Endpoint | 📋 Description |
|--------|---------|-------------|
| ` POST` | `/upload` | Uploads a video for processing |
| ` POST` | `/process` | Starts processing |
| ` POST` | `/interrupt/{STOP}` | Interrupts the on-going process |
| ` GET` | `/status/{job_id}` | Retrieves the processing status of a video |
| ` GET` | `/download/{job_id}` | Downloads the processed video |

---

## 📂 **7. Project Structure**

```
/project-root
├── src/                # Source code
├── data/               # Dataset (if any)
├── models/             # ML models (if applicable)
├── notebooks/          # Jupyter notebooks (if applicable)
├── docs/               # Documentation
├── tests/              # Unit tests
├── requirements.txt    # Dependencies
├── README.md           # Project README
```

---

This **Auto Video Annotation System** is built for efficiency and scalability. Happy annotating! 🎥🚀

