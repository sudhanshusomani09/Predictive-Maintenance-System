# Predictive-Maintenance-System
---
##  Phase 2: Deep Learning Approach (CNN)
While the Random Forest model performed well, it relied on manual feature engineering (Kurtosis, Skewness). To make the system more robust and automated, I implemented a **1D Convolutional Neural Network (CNN)**.

### **The Architecture**
* **Input:** Raw vibration signals (downsampled to 2048 points).
* **Layers:**
    * 2x `Conv1D` Layers (Filters: 32, 64) to extract waveform patterns automatically.
    * `MaxPooling` layers to reduce noise.
    * `Dense` layers for final classification.
* **Outcome:** The model achieved **99-100% accuracy** without needing any manual math/physics calculations. It learned to recognize the "shape" of the failure directly from the sensor data.

### **Model Comparison**
| Approach | Feature Engineering? | Accuracy | Pros |
| :--- | :--- | :--- | :--- |
| **Random Forest** | Yes (Manual) | 100% | Fast training, interpretable. |
| **1D-CNN (Deep Learning)** | No (Automatic) | 100% | No domain expertise needed, scalable to complex data. |
---
