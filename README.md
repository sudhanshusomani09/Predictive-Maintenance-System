# Predictive-Maintenance-System
# AI-Powered Predictive Maintenance for Rotating Machinery

##  Project Overview
This project leverages Machine Learning to predict catastrophic failure in industrial machinery using vibration sensor data. By analyzing raw accelerometer signals from the **NASA Bearing Dataset**, the system identifies "Health" vs. "Faulty" states with **100% accuracy** using a Random Forest classifier.

This solution demonstrates how **Industry 4.0** concepts can minimize downtime by detecting faults before they become critical.

##  Technologies Used
* **Language:** Python 3.9
* **Data Processing:** Pandas, NumPy
* **Signal Processing:** Statistical feature extraction (Kurtosis, Skewness, RMS)
* **Machine Learning:** Scikit-Learn (Random Forest Classifier)
* **Visualization:** Matplotlib, Seaborn

##  The Engineering Approach
Raw vibration data (20kHz sampling rate) is too noisy for direct analysis. I implemented a **Feature Engineering** pipeline to translate physical vibrations into statistical metrics:

1.  **Data Acquisition:** Loaded 984 sensor files from the NASA IMS dataset.
2.  **Feature Extraction:** Calculated the following for each bearing:
    * **RMS (Root Mean Square):** Measures the overall energy of the vibration.
    * **Kurtosis:** Detects "impulsive" shocks (e.g., a ball hitting a crack). High kurtosis is a key indicator of early bearing failure.
    * **Skewness:** Measures the asymmetry of the vibration signal.
3.  **Model Training:** Trained a **Random Forest Classifier** on the processed features.
4.  **Validation:** Achieved 100% accuracy on the test set, proving that statistical features effectively separate healthy and faulty states.

##  Results
| Metric | Value |
| :--- | :--- |
| **Model** | Random Forest (100 Trees) |
| **Test Accuracy** | 100% |
| **Key Indicator** | Kurtosis proved to be the strongest predictor of failure. |

##  How to Run
1.  Clone the repository.
2.  Install dependencies: `pip install pandas numpy scikit-learn matplotlib seaborn`
3.  Download the **NASA IMS Bearing Dataset** and place it in `dataset/2nd_test`.
4.  Run the Jupyter Notebook `Predictive_Maintenance_V1.ipynb`.

##  Future Scope
* Implement a **1D-Convolutional Neural Network (CNN)** to learn features directly from raw waveforms, eliminating the need for manual feature engineering.
* Deploy the model as a real-time API.
* We could have used XGBoost also but Random Forest is widely regarded as the best "first model" to try on any new tabular dataset. It is highly reliable.
Gradient Boosting is the only model that consistently beats Random Forest in accuracy competitions, but XGBoost is harder to tune, 
XGboost is more accurate and Random Forest is more reliable. 
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
---
##  Phase 3: Prognostics (Predicting Time-to-Failure)
The final phase moved beyond simple classification ("Is it broken?") to **Regression** ("When will it break?"). The goal was to estimate the **Remaining Useful Life (RUL)** of the bearing at any given moment.

### **The Approach**
Since the dataset did not have explicit "Time Left" labels, I created a custom target variable based on **Linear Degradation**:
* **Start of Test:** RUL = 100% (1.0)
* **End of Test:** RUL = 0% (0.0)

### **The Model (CNN Regressor)**
I modified the previous CNN architecture for regression:
* **Loss Function:** Changed from `Binary Crossentropy` to **Mean Squared Error (MSE)** to minimize the distance between predicted and actual time.
* **Output Layer:** Used a single neuron with **Linear Activation** (instead of Sigmoid) to output a continuous time value.

### **Results**
* **Performance:** The model achieved a **Mean Absolute Error (MAE) of ~0.13**.
* **Interpretation:** The system can predict the remaining life of the machinery with an accuracy of **±13%**.
* **Visual Proof:** The predicted degradation curve (Red) closely follows the actual physical degradation (Blue), proving the model learned the "physics of failure" effectively.

---
##  Conclusion
This project demonstrates a complete **End-to-End Predictive Maintenance Pipeline**:
1.  **Phase 1:** Established a baseline with **Random Forest** (Feature Engineering).
2.  **Phase 2:** Automated fault detection with **Deep Learning (CNN)**.
3.  **Phase 3:** Enabled predictive planning with **RUL Estimation**.
