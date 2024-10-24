# Explainable-Deep-Learning-for-Medical-Imaging-Pneumonia-Detection-with-ResNet50-and-Grad-CAM

[Open this notebook in Google Colab]([https://colab.research.google.com/drive/your_notebook_id](https://colab.research.google.com/drive/1FkwzjsXCyjErTABgkphBFRoFzqV3j2SZ?usp=sharing))

# **Introduction**

In recent years, deep learning models have demonstrated impressive performance in medical image classification tasks, providing significant benefits for early diagnosis and treatment decisions. However, their "black-box" nature often leaves healthcare professionals without clear insight into how these models make decisions, raising concerns about trust, safety, and interpretability. The purpose of this experiment is to explore the application of Explainable AI (XAI) techniques, specifically Gradient-weighted Class Activation Mapping (Grad-CAM), to uncover which regions of medical images are most influential in model predictions. By using Grad-CAM, we aim to provide visual explanations that can help bridge the gap between machine intelligence and human interpretability.

The dataset used in this experiment, titled [**Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification**](https://data.mendeley.com/datasets/rscbjbr9sj/2) , contains labeled chest X-ray images for binary classification between healthy and pneumonia-affected individuals. The ability to detect pneumonia accurately from chest X-rays is crucial in clinical settings, especially for early detection and timely intervention.

The pre-trained **ResNet50** model after transfer learning will be utilized to classify these images, and Grad-CAM will be employed to generate heatmaps that highlight the regions within the X-rays most relevant to the model’s classification decision. The experiment will focus on comparing the feature importance scores between healthy and pneumonia images to determine whether there is a significant difference in the regions highlighted by the model for each class.

The impact of this experiment lies in its potential to improve the interpretability of deep learning models used in medical image analysis. By identifying which areas of the image are most relevant to the model’s predictions, healthcare professionals can gain a deeper understanding of model decisions, increasing trust and enabling more informed clinical decision-making. Additionally, this approach may help identify any biases in model attention, ensuring that the models are focusing on clinically relevant features, such as lung regions, rather than irrelevant areas.

This experiment aims to combine the power of deep learning with transparency, ultimately contributing to the safe and reliable adoption of AI systems in healthcare.

# **Experiment Design**

## 1. Hypothesis:
- **Null Hypothesis (H0):** There is no significant difference in the feature importance highlighted by Grad-CAM between pneumonia and healthy chest X-ray images when using a pre-trained ResNet50 model.
- **Alternative Hypothesis (H1):** There is a significant difference in the feature importance highlighted by Grad-CAM between pneumonia and healthy chest X-ray images when using a pre-trained ResNet50 model.

---

## 2. Dataset:
- **Name:** Chest X-ray Images (Pneumonia)
- **Classes:**
  - **Pneumonia:** Images labeled as pneumonia.
  - **Healthy:** Images labeled as normal (healthy).
- **Sample Size:**
  - Use **30 images per class** for a balanced comparison.
- **Preprocessing:** Resize images to a consistent size (e.g., 224x224) and normalize pixel values as required by ResNet50.

---

## 3. Model:
- **Model Used:** Pre-trained ResNet50 from the ImageNet dataset.
  - Use the pre-trained version of ResNet50, leveraging transfer learning for pneumonia and healthy classification.
  
---

## 4. XAI Method:
- **Method:** Grad-CAM (Gradient-weighted Class Activation Mapping)
  - Grad-CAM will be used to generate heatmaps that highlight important regions in the chest X-rays that contribute to the classification decision (pneumonia vs. healthy).
  - The heatmaps will provide visual explanations of which regions in the images influence the model's predictions.

---

## 5. Experiment:

### a. Data Preparation:
- Load the Chest X-ray Images (Pneumonia) dataset.
- Randomly select **30 images** from each class (pneumonia and healthy).
- Resize the images to **224x224** and normalize according to ResNet50's input requirements.

### b. Model Inference & Grad-CAM Heatmap Generation:
- Pass each image through the pre-trained ResNet50 model to get predictions for pneumonia and healthy images.
- Use **Grad-CAM** to generate class-specific heatmaps for both pneumonia and healthy images.
- **Store the heatmaps** for both classes.

### c. Feature Importance Aggregation:
- For each image, **average the Grad-CAM heatmap values** across the entire image to create an overall feature importance score.
- This score will represent the importance of image regions for the model’s decision.

### d. Collect Data for Each Class:
- Collect the **aggregated feature importance scores** for each image in the pneumonia class.
- Collect the **aggregated feature importance scores** for each image in the healthy class.

---

## 6. Statistical Testing:

### a. Objective:
- Compare the aggregated feature importance scores between the two classes (pneumonia and healthy) to see if there’s a significant difference in how the model highlights important regions for each class.

### b. Statistical Test:
- Use a **t-test** to compare the means of the feature importance scores for the two groups (pneumonia vs. healthy).
- The t-test will help determine whether the difference in feature importance between the two classes is statistically significant.

### c. Threshold for Significance:
- Set a significance level (e.g., **p < 0.05**). If the p-value is below this threshold, reject the null hypothesis, indicating that there is a significant difference in feature importance between the two classes.

---

## 7. Visualization & Reporting:

- **Grad-CAM Heatmaps**: Visualize a few examples of the Grad-CAM heatmaps for both pneumonia and healthy images to show how the model focuses on different regions.
- **Feature Importance Scores**: Create a plot comparing the feature importance scores between the two classes.
- **Statistical Test Results**: Present the results of the statistical test, including the t-statistic and p-value.

# **Final Conclusion Report**

#### 1. **Analysis of Model Performance (Tailored to ResNet50 Transfer Learning)**:
The first stage of this experiment evaluated the performance of a transfer learning-based model using **ResNet50**, which was pre-trained on ImageNet, to classify chest X-ray images into two categories: **Normal** and **Pneumonia**. The model was adapted for this medical task by freezing the base (convolutional) layers and fine-tuning only the top layers for binary classification.

- **Model Performance**:
  - **Test Accuracy**: The model achieved an accuracy of **77.40%**, which reflects reasonable performance for this task.
  - **Class-wise Performance**: The model performed significantly better on the **Pneumonia** class, with a recall of **0.68** and an F1-score of **0.65**. However, it struggled to correctly classify **Normal** cases, with a recall of only **0.29**.
  
- **Challenges with Normal Class**: The model showed **poor performance** in detecting normal lungs, misclassifying a significant number of healthy images as pneumonia. This discrepancy could be attributed to the class imbalance in the dataset (390 Pneumonia cases vs. 234 Normal cases), as well as the domain gap between ImageNet images and medical X-rays.

- **Transfer Learning Insights**: While the **ResNet50 model** was effective in identifying pneumonia features, it demonstrated limitations in detecting healthy lung features, which suggests that further fine-tuning or additional training may be required to improve its overall generalization, particularly for normal cases.

#### 2. **Analysis of Grad-CAM Visualization Results**:
The second phase of the experiment involved using **Grad-CAM** to visualize the regions of the X-ray images that were most influential in the model’s decision-making process. This method provided insight into which areas the model focused on when classifying **Pneumonia** and **Normal** images.

- **Key Findings**:
  - **Focus on the Diaphragm**: Both healthy and pneumonia cases showed **strong attention on the diaphragm** (red area), which is not typically a key region for diagnosing lung conditions like pneumonia. This suggests that the model may be misinterpreting important features.
  - **Attention on the Right Blank Area**: Another surprising finding was the model’s focus on the **blank area outside the body** (yellow area), which holds no diagnostic value. This could be a result of overfitting to irrelevant patterns or artifacts in the dataset.
  - **Less Focus on the Lung Area**: The **lungs** (blue area) received less attention from the model, indicating that the model is not effectively identifying the critical lung regions needed for accurate diagnosis.

- **Transfer Learning Implications**: The reason for this misfocus may lie in the fact that the convolutional layers were **frozen** during transfer learning. As a result, the **pre-trained ResNet50 model** could not correctly adapt its feature extraction to focus on the relevant regions in medical X-ray images. This highlights the need for further fine-tuning of the base layers to better capture medically relevant features, especially in the lung regions.

#### 3. **Interpretation of the Statistical Analysis**:
The third phase of the experiment focused on statistically analyzing the **feature importance scores** generated by Grad-CAM for **Pneumonia** and **Normal** images. The goal was to determine whether there was a significant difference in how the model highlighted important regions between the two classes.

- **Summary Statistics**:
  - The **mean feature importance score** for Pneumonia images was **0.242**, slightly higher than the score for Normal images, which was **0.215**.
  - The **t-test** results showed a **t-statistic of -5.3253** and a **p-value of 0.0000**, which is far below the significance threshold (p < 0.05).

- **Hypothesis Testing**:
  - **Null Hypothesis (H₀)**: There is no significant difference in the feature importance highlighted by Grad-CAM between pneumonia and healthy chest X-ray images when using a pre-trained ResNet50 model.
  - **Alternative Hypothesis (H₁)**: There is a significant difference in the feature importance highlighted by Grad-CAM between pneumonia and healthy chest X-ray images when using a pre-trained ResNet50 model.
  
  Since the **p-value** was significantly lower than the threshold, we **rejected the null hypothesis**, indicating that there is a statistically significant difference in how the pre-trained ResNet50 model highlights important regions for Pneumonia vs. Normal images.

- **Effect Size (Cohen's d)**: The calculated **Cohen's d** was **1.3985**, which indicates a **large effect size**. This suggests that the difference in feature importance between the two classes is not only statistically significant but also substantial in magnitude.

- **Conclusion**: The model clearly behaves differently when interpreting healthy and pneumonia images, with pneumonia images receiving higher feature importance scores. This shows that the pre-trained ResNet50 model is better at focusing on pneumonia-related features, although its attention to healthy lung features remains problematic.

---

### Final Conclusion

This experiment has successfully demonstrated that using a **pre-trained ResNet50 model** for pneumonia detection on chest X-ray images, combined with **Grad-CAM visualization**, provides insights into the model’s decision-making process. However, the experiment also reveals several areas that require improvement:

- **Model Performance**: While the model performs well on Pneumonia cases, achieving a recall of **0.68**, its performance on Normal cases is significantly worse, with a recall of **0.29**. This suggests that the model is biased toward detecting pneumonia and struggles with healthy lungs.
  
- **Grad-CAM Visualizations**: The **Grad-CAM** heatmaps revealed that the model's attention is often focused on irrelevant areas (e.g., the diaphragm and blank regions) rather than the lungs themselves. This misfocus is likely due to freezing the convolutional layers during transfer learning, preventing the model from adapting to the specific task of pneumonia detection.

- **Statistical Significance**: The statistical analysis of the feature importance scores confirmed that there is a **significant difference** in how the model interprets pneumonia vs. healthy images, with the model assigning more importance to regions in pneumonia images. The large effect size (Cohen's d = 1.3985) further supports the practical relevance of this finding.

- **Recommendations**:
  - **Unfreeze and Fine-Tune**: To improve the model’s focus on relevant lung areas, unfreezing the convolutional layers during transfer learning is recommended, allowing the model to better adapt to medical imaging features.
  - **Class Balancing**: Address the class imbalance in the dataset to reduce the model's bias toward pneumonia.
  - **Further Validation**: Additional experiments with a larger and more balanced dataset will be crucial for improving the model’s generalization capabilities across both classes.

This experiment has highlighted the potential of combining deep learning and **Explainable AI (XAI)** techniques like Grad-CAM to enhance the transparency and reliability of AI systems in healthcare. However, further refinements are necessary to optimize the model for real-world clinical applications, particularly in detecting healthy lungs with greater accuracy.

