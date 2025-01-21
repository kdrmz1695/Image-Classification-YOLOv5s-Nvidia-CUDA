# Object Detection with Sequential and Parallel Processing with YOLOv5s, Nvidia - CUDA

This project implements object detection on video streams using the YOLOv5s model. It demonstrates the performance comparison between **sequential processing** and **parallel processing** to highlight the benefits of parallelism in real-time video processing.

---

## **Features**
- **Sequential Processing:** Processes video frames one by one using a single thread.
- **Parallel Processing:** Leverages multi-threading to process multiple frames simultaneously.
- **YOLOv5s Model:** Uses a pre-trained YOLOv5s model for object detection.
- **FPS Calculation:** Calculates frames per second (FPS) for performance evaluation.
- **Real-time Visualization:** Displays processed video frames with bounding boxes, labels, and confidence scores.

---

## **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/kdrmz1695/Image-Classification-YOLOv5s-Nvidia-CUDA.git
   cd Image-Classification-YOLOv5s-Nvidia-CUDA
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # For Linux/Mac
   env\Scripts\activate    # For Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## **Usage**

1. **Run Sequential Processing:**
   ```bash
   python sequential_processing.py
   ```

2. **Run Parallel Processing:**
   ```bash
   python parallel_processing.py
   ```

3. **Input Video:**
   - Default input video: `people1.mp4`
   - Replace the video path in the respective scripts to use a different input file.

---

## **Code Structure**

```plaintext
├── model.py                   # YOLOv5 model loading and configuration
├── utils.py                   # Utility functions (e.g., time measurement)
├── video_processing.py        # Centralized video frame processing logic
├── sequential_processing.py   # Sequential frame processing script
├── parallel_processing.py     # Parallel frame processing script
├── requirements.txt           # Required Python libraries
└── README.md                  # Project documentation
```

---

## **Results**
### **Performance Comparison:**
| Processing Method | FPS (Frames Per Second) | Total Time (Seconds) |
|-------------------|--------------------------|-----------------------|
| Sequential        | ~5-6 FPS                | 120 seconds           |
| Parallel          | ~20-25 FPS              | 30 seconds            |

### **Key Observations:**
- Parallel processing significantly improves performance compared to sequential processing.
- GPU acceleration further enhances speed in parallel processing.

---

## **Future Work**
- **Add GPU Optimization:** Enhance GPU utilization for better real-time performance.
- **Object Tracking:** Extend the system to include object tracking.
- **Dynamic Load Balancing:** Implement adaptive resource allocation for parallel processing.

---

## **Contributing**
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and push them to your branch.
4. Create a pull request.

---

## **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
