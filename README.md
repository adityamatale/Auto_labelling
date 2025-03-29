# 🚀 **Auto Video Annotation System**

## 📜 **Table of Contents**
1. [📌 Introduction](#introduction)
2. [🎥 Video Processing Features](#video-processing-features)
3. [🛠️ Tech Stack](#tech-stack)
4. [⚙️ Installation](#installation)
5. [🚀 Usage](#usage)
6. [🔗 API Endpoints](#api-endpoints)
7. [📂 Project Structure](#project-structure)
8. [🤝 Contributing](#contributing)
9. [📜 License](#license)
10. [🙏 Acknowledgments](#acknowledgments)

---

## 📌 **1. Introduction**

### ❓ **Problem Statement**  
⏳ Manual video annotation is a slow, labor-intensive, and costly process, often leading to inconsistencies. Traditional methods struggle to scale efficiently, making it difficult to generate high-quality, large-scale annotated datasets across different domains.

### 💡 **Solution**  
This **Auto Video Annotation System** leverages state-of-the-art AI models to automate video labeling, improving efficiency and scalability while maintaining accuracy.

### 🔥 **Key Features**  
✅ **Automated Annotation Pipeline** – Uses advanced AI model *Grounding DINO* to detect and label objects in videos, reducing manual effort.  
✅ **Scalable & Efficient System** – Built with a lightweight, high-performance FastAPI backend for seamless integration and large-scale data processing.  
✅ **Optimized Video Processing** – Implements frame segmentation (*SAM*) and propagation (*SAMv2*) to minimize redundant computation while maintaining accuracy.  
✅ **Reliable Info Monitoring** – Includes *Supervision* for annotation handling, and robust logging for debugging and consistency.  

This system is designed to streamline video annotation workflows, making large-scale dataset preparation faster and more efficient. 🚀

---

## 🎥 **2. Video Processing Features**

🎯 **Zero-shot detection → labeling → segmentation**  
🎞️ **Customizable FPS** – Users can adjust the frames per second (FPS) to optimize processing time.  
⏸️ **Interrupts for annotation refinement**:  
  🔹 **STOPPROCESS** – Users can stop the segmentation at any point and retrieve the segmented video up to that point.  
  🔹 **STOPDETECTION** – Users can halt new object detection while continuing segmentation of previously detected objects across the video.  
  🔹 **Refinement Interrupt** – Users can refine tight segmentation masks by adding positive/negative click labels for specific objects.  

---

## 🛠️ **3. Tech Stack**

| 🖥️ Technology | 🔍 Purpose |
|------------|---------|
| 🐍 **Python** | Primary development language |
| ⚡ **FastAPI** | API framework for seamless integration |
| 📊 **Supervision** | Utility for managing and visualizing annotations |
| 🔄 **Threading** | Enables parallel video processing in the background |
| 📜 **Logging** | Capturing runtime events for debugging and monitoring |
| 🎥 **OpenCV (cv2)** | Handles video processing and frame extraction |
| 🧩 **SAM (Segment Anything Model)** | Automatic object segmentation |
| 🏎️ **SAMv2** | Video propagation functionality |
| 🎯 **Grounding DINO** | Zero-shot object detection and labeling |

---

## ⚙️ **4. Installation**

Follow these steps to set up the project:

```sh
# 📥 Clone the repository
git clone https://git.acldigital.com/ai-ml/autolabelling.git
```

```sh
# 📂 Navigate to the project directory
cd Auto_labelling/
```

```sh
# ⚙️ Run the setup script
bash setup.sh
```

---

## 🚀 **5. Usage**

### ▶️ **Run the application**
```sh
python App.py
```

### 📜 **API Calls**  
📄 Check the `api_call.txt` file for example API request commands.

---

## 🔗 **6. API Endpoints**

| 📡 Method | 🔗 Endpoint | 📋 Description |
|--------|---------|-------------|
| `📤 POST` | `/upload` | Uploads a video for processing |
| `📥 GET` | `/status/{job_id}` | Retrieves the processing status of a video |
| `🛑 POST` | `/stopprocess` | Stops video segmentation at the current frame |
| `🛑 POST` | `/stopdetection` | Stops new object detection but continues segmentation |
| `✏️ POST` | `/refine` | Allows the user to refine segmentation masks |

---

## 📂 **7. Project Structure**

```
/project-root
├── src/                # 🚀 Source code
├── data/               # 📂 Dataset (if any)
├── models/             # 🤖 ML models (if applicable)
├── notebooks/          # 📒 Jupyter notebooks (if applicable)
├── docs/               # 📑 Documentation
├── tests/              # ✅ Unit tests
├── requirements.txt    # 📜 Dependencies
├── README.md           # 📝 Project README
```

---

## 🤝 **8. Contributing**

We welcome contributions! 🎉 To contribute:

```sh
# 🍴 Fork the repository
git fork https://git.acldigital.com/ai-ml/autolabelling.git
```

```sh
# 🌿 Create a new branch
git checkout -b feature-xyz
```

```sh
# 💾 Make your changes and commit them
git commit -m "Added new feature"
```

```sh
# 🔀 Push your branch and open a Pull Request
git push origin feature-xyz
```

For detailed contribution guidelines, refer to `CONTRIBUTING.md`. 🛠️

---

## 📜 **9. License**

📝 This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 🙏 **10. Acknowledgments**

A huge thanks to:
- 🙌 The developers and contributors of **Grounding DINO, SAM, and FastAPI**.
- 💡 Open-source AI research communities for pushing the boundaries of video annotation.
- 🏢 **[Your Team/Company Name]** for providing resources and support.

---

This **Auto Video Annotation System** is built for efficiency and scalability. Happy annotating! 🎥🚀

