# Explainable-Deep-Learning-for-Medical-Imaging-Pneumonia-Detection-with-ResNet50-and-Grad-CAM

[Open this notebook in Google Colab](https://colab.research.google.com/drive/1FkwzjsXCyjErTABgkphBFRoFzqV3j2SZ?usp=sharing)

## **Introduction**

In recent years, deep learning models have demonstrated impressive performance in medical image classification tasks, providing significant benefits for early diagnosis and treatment decisions. However, their "black-box" nature often leaves healthcare professionals without clear insight into how these models make decisions, raising concerns about trust, safety, and interpretability. The purpose of this experiment is to explore the application of Explainable AI (XAI) techniques, specifically Gradient-weighted Class Activation Mapping (Grad-CAM), to uncover which regions of medical images are most influential in model predictions. By using Grad-CAM, we aim to provide visual explanations that can help bridge the gap between machine intelligence and human interpretability.

The dataset used in this experiment, titled [**Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification**](https://data.mendeley.com/datasets/rscbjbr9sj/2) , contains labeled chest X-ray images for binary classification between healthy and pneumonia-affected individuals. The ability to detect pneumonia accurately from chest X-rays is crucial in clinical settings, especially for early detection and timely intervention.

The pre-trained **ResNet50** model after transfer learning will be utilized to classify these images, and Grad-CAM will be employed to generate heatmaps that highlight the regions within the X-rays most relevant to the model’s classification decision. The experiment will focus on comparing the feature importance scores between healthy and pneumonia images to determine whether there is a significant difference in the regions highlighted by the model for each class.

The impact of this experiment lies in its potential to improve the interpretability of deep learning models used in medical image analysis. By identifying which areas of the image are most relevant to the model’s predictions, healthcare professionals can gain a deeper understanding of model decisions, increasing trust and enabling more informed clinical decision-making. Additionally, this approach may help identify any biases in model attention, ensuring that the models are focusing on clinically relevant features, such as lung regions, rather than irrelevant areas.

This experiment aims to combine the power of deep learning with transparency, ultimately contributing to the safe and reliable adoption of AI systems in healthcare.

![Normal](https://github.com/user-attachments/assets/ada0be67-27ac-47f8-8b1d-cfb6c4e590e2)

![Pheumonia](https://github.com/user-attachments/assets/322694a6-02bd-4bd1-a8d9-11badda5e6a3)

## **Experiment Design**

### 1. Hypothesis:
- **Null Hypothesis (H0):** There is no significant difference in the feature importance highlighted by Grad-CAM between pneumonia and healthy chest X-ray images when using a pre-trained ResNet50 model.
- **Alternative Hypothesis (H1):** There is a significant difference in the feature importance highlighted by Grad-CAM between pneumonia and healthy chest X-ray images when using a pre-trained ResNet50 model.

---

### 2. Dataset:
- **Name:** Chest X-ray Images (Pneumonia)
- **Classes:**
  - **Pneumonia:** Images labeled as pneumonia.
  - **Healthy:** Images labeled as normal (healthy).
- **Sample Size:**
  - Use **30 images per class** for a balanced comparison.
- **Preprocessing:** Resize images to a consistent size (e.g., 224x224) and normalize pixel values as required by ResNet50.

---

### 3. Model:
- **Model Used:** Pre-trained ResNet50 from the ImageNet dataset.
  - Use the pre-trained version of ResNet50, leveraging transfer learning for pneumonia and healthy classification.
  
---

### 4. XAI Method:
- **Method:** Grad-CAM (Gradient-weighted Class Activation Mapping)
  - Grad-CAM will be used to generate heatmaps that highlight important regions in the chest X-rays that contribute to the classification decision (pneumonia vs. healthy).
  - The heatmaps will provide visual explanations of which regions in the images influence the model's predictions.

---

### 5. Experiment:

#### a. Data Preparation:
- Load the Chest X-ray Images (Pneumonia) dataset.
- Randomly select **30 images** from each class (pneumonia and healthy).
- Resize the images to **224x224** and normalize according to ResNet50's input requirements.

#### b. Model Inference & Grad-CAM Heatmap Generation:
- Pass each image through the pre-trained ResNet50 model to get predictions for pneumonia and healthy images.
- Use **Grad-CAM** to generate class-specific heatmaps for both pneumonia and healthy images.
- **Store the heatmaps** for both classes.

#### c. Feature Importance Aggregation:
- For each image, **average the Grad-CAM heatmap values** across the entire image to create an overall feature importance score.
- This score will represent the importance of image regions for the model’s decision.

#### d. Collect Data for Each Class:
- Collect the **aggregated feature importance scores** for each image in the pneumonia class.
- Collect the **aggregated feature importance scores** for each image in the healthy class.

---

### 6. Statistical Testing:

#### a. Objective:
- Compare the aggregated feature importance scores between the two classes (pneumonia and healthy) to see if there’s a significant difference in how the model highlights important regions for each class.

#### b. Statistical Test:
- Use a **t-test** to compare the means of the feature importance scores for the two groups (pneumonia vs. healthy).
- The t-test will help determine whether the difference in feature importance between the two classes is statistically significant.

#### c. Threshold for Significance:
- Set a significance level (e.g., **p < 0.05**). If the p-value is below this threshold, reject the null hypothesis, indicating that there is a significant difference in feature importance between the two classes.

---

### 7. Visualization & Reporting:

- **Grad-CAM Heatmaps**: Visualize a few examples of the Grad-CAM heatmaps for both pneumonia and healthy images to show how the model focuses on different regions.
- **Feature Importance Scores**: Create a plot comparing the feature importance scores between the two classes.
- **Statistical Test Results**: Present the results of the statistical test, including the t-statistic and p-value.

## Analysis of Model Performance (Tailored to ResNet50 Transfer Learning)

### 1. Introduction
This experiment evaluates the performance of a transfer learning-based model for classifying chest X-ray images into two categories: **Normal** and **Pneumonia**. Transfer learning was employed using a **pre-trained ResNet50 model**, which was fine-tuned for this specific task. The goal of the experiment was to leverage the powerful feature extraction capabilities of ResNet50 (pre-trained on ImageNet) while adapting it to the medical imaging domain, specifically for chest X-ray classification.

ResNet50, known for its strong performance on general image classification tasks, was used as the backbone, and a fully connected layer was added on top for binary classification. The model’s performance was evaluated on a test set to assess its ability to distinguish between Pneumonia and Normal cases.

### 2. Model and Experiment Setup
- **Model Architecture**: ResNet50 (pre-trained on ImageNet) with the top layers removed, followed by a Global Average Pooling layer and a dense layer for binary classification.
- **Transfer Learning**: The base ResNet50 model was frozen, and only the top layers were fine-tuned on the chest X-ray dataset.
- **Training Dataset**: Labeled chest X-ray images, split into Normal and Pneumonia classes, were used for training and validation.
- **Batch Size**: 32
- **Epochs**: 10

The goal was to benefit from ResNet50’s strong pre-trained features and adapt those features to the domain of medical images, particularly for the detection of Pneumonia.

### 3. Evaluation Results

#### Model Accuracy and Loss:
- **Test Accuracy**: 77.40%
- **Test Loss**: 0.5110

The model achieved an accuracy of **77.40%**, reflecting reasonable performance for this medical image classification task. However, further breakdown of class-specific performance indicates the model’s tendency to perform better on Pneumonia cases than on Normal cases.

#### Classification Report:

| Class      | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| **Normal** | 0.36      | 0.29   | 0.32     | 234     |
| **Pneumonia** | 0.62   | 0.68   | 0.65     | 390     |
| **Overall Accuracy** |         |        | 0.54     | 624     |

- **Precision**: Precision for the **Normal** class is low at 0.36, indicating a higher rate of false positives when the model predicts an image as Normal. The **Pneumonia** class performs better, with a precision of 0.62, suggesting that when the model predicts Pneumonia, it is correct 62% of the time.
  
- **Recall**: The model demonstrates good recall for the **Pneumonia** class (0.68), meaning it correctly identifies 68% of actual Pneumonia cases. However, the recall for **Normal** is notably lower (0.29), indicating that many Normal cases are misclassified as Pneumonia.

- **F1-Score**: The **F1-score** for the Pneumonia class is reasonable at 0.65, but for Normal, it is low at 0.32, reflecting the model’s difficulty in handling this class.

#### Class Imbalance:
- The test set contains **390 Pneumonia cases** and **234 Normal cases**, showing a natural class imbalance, which is likely contributing to the model's poor performance on the Normal class.

### 4. Transfer Learning Insights
The use of **ResNet50** for transfer learning provided strong feature extraction capabilities for Pneumonia detection. The pre-trained ResNet50 layers captured essential features from the chest X-ray images, leading to a relatively high recall for the Pneumonia class. However, due to the imbalanced nature of the dataset and the possible domain gap between ImageNet images and medical X-rays, the model struggled with detecting Normal cases.

### 5. Analysis of Results

- **Strong Performance on Pneumonia**: The model performs well on Pneumonia cases, achieving a recall of 0.68 and an F1-score of 0.65. This shows that the pre-trained ResNet50 effectively extracted relevant features for Pneumonia detection.

- **Challenges with Normal Class**: The low precision (0.36) and recall (0.29) for Normal cases indicate that the model misclassifies a significant number of Normal images as Pneumonia. This could be due to the dataset's imbalance or insufficient feature adaptation during fine-tuning.

- **Discrepancy Between Accuracy and Class-wise Performance**: While the overall test accuracy is **77.40%**, the class-wise performance (with an F1-score of 0.54 for Normal and Pneumonia combined) reveals that the model is biased toward predicting Pneumonia, which skews the accuracy.

### 7. Conclusion
The use of transfer learning with ResNet50 yielded promising results, particularly for detecting Pneumonia in chest X-rays, with a test accuracy of **77.40%**. However, its performance on the Normal class is suboptimal, indicating a need for further refinement.

## Analysis of Grad-CAM Visualization Results:

<img width="1129" alt="Screen Shot 2024-10-23 at 11 56 42 PM" src="https://github.com/user-attachments/assets/d2f9c139-73be-445b-aa91-c1811ed5f424">

<img width="1136" alt="Screen Shot 2024-10-23 at 11 58 42 PM" src="https://github.com/user-attachments/assets/1de61a49-e57e-409c-9e70-9382bea09fbd">

1. **Focus on the Diaphragm (Red Area)**:
   - The fact that the model is focusing heavily on the diaphragm, regardless of whether the X-ray shows a healthy lung or a lung affected by pneumonia, suggests that the model may be **misinterpreting the features relevant to the classification task**.
   - The diaphragm is not typically a region of interest for diagnosing lung conditions like pneumonia. Pneumonia-related features are usually present in the lung parenchyma (the tissues of the lungs), where fluid buildup or inflammation can be detected. Therefore, the model’s focus on the diaphragm may indicate that it has not been sufficiently trained to recognize lung-specific features.
   - **I was only doing transfer learning with ResNet50 by freezing the base model, which means the convolutional layers are all frozen during my transfer learning. This means the pre-trained ResNet50 can't correctly focus its attention on the right feature region.**

2. **Attention on the Right Blank Area (Yellow Area)**:
   - The attention given to the right blank area outside the body is a red flag. This area should have no diagnostic relevance, as it contains no anatomical structures.
   - This could happen due to several reasons:
     - **Model overfitting**: The model may have overfit to some irrelevant background patterns that are consistently present in the dataset (e.g., artifacts from X-ray machines, borders around the images).
     - **Dataset artifact**: If some images in the dataset consistently have certain patterns or artifacts in this area (e.g., labels, markers, or machine artifacts), the model might incorrectly learn to associate these with the class labels.
     - **Incorrect bounding of the lung region**: It’s possible that the lung area is not properly centered or highlighted in the dataset, leading the model to misinterpret the right blank area as a region of importance.

3. **Less Focus on the Lung Area (Blue Area)**:
   - The lungs, being the primary region of interest for detecting pneumonia, are receiving relatively less attention from the model (shown in blue).
   - This suggests that the model is not effectively recognizing the key visual features in the lungs that differentiate healthy from pneumonia-affected lungs.
   - This could be due to:
     - **Lack of specific features learned for pneumonia**: The model may not have been trained well enough on pneumonia-specific features (e.g., fluid accumulation, consolidation) within the lung tissues.
     - **Need for further fine-tuning**: Additional training with a more focused loss function or data augmentation to highlight lung areas might help the model pay more attention to relevant features in the lungs.

### Conclusion:
The Grad-CAM visualization shows that the model is not focusing on the most important diagnostic regions (the lungs) and is instead focusing on irrelevant areas like the diaphragm and blank spaces. **Since the convolutional layers were frozen during transfer learning with ResNet50, the pre-trained model cannot adapt its focus to the relevant regions, such as the lungs**. This suggests that the model may require further fine-tuning, improved preprocessing, and enhanced training to better understand and focus on the features relevant to distinguishing healthy lungs from pneumonia-affected lungs.

## Interpretation of the Statistical Analysis

![stats_result](https://github.com/user-attachments/assets/b8d11117-f340-44bc-835c-6d909aaacfc8)

Feature Importance Analysis Results
==================================================

Summary Statistics:
            Metric    Normal  Pneumonia
              Mean  0.215205   0.242184
Standard Deviation  0.018152   0.020368
       Sample Size 30.000000  30.000000

Statistical Test Results:
T-statistic: -5.3253
P-value: 0.0000

Null Hypothesis: No significant difference in feature importance between classes
Alternative Hypothesis: Significant difference in feature importance between classes
Significance level (α): 0.05

Result: Reject the null hypothesis (p-value = 0.0000 < 0.05)
There is a statistically significant difference in feature importance between normal and pneumonia classes.

Effect Size:
Cohen's d: 1.3985
Effect size interpretation: large

#### 1. **Objective of the Analysis**:
The goal of this analysis was to determine whether there is a **significant difference in the Grad-CAM feature importance scores** between images of healthy lungs and those with pneumonia when using a pre-trained ResNet50 model. Specifically, the feature importance scores represent how much the model focused on different regions of the image when making predictions.

#### 2. **Summary Statistics**:
- **Mean Feature Importance**:
  - **Normal (Healthy)**: The average feature importance score for the normal (healthy) images is 0.215.
  - **Pneumonia**: The average feature importance score for the pneumonia images is 0.242.
  - This shows that, on average, the pre-trained ResNet50 Model places slightly more importance on regions in pneumonia images than in healthy images.
  
- **Standard Deviation**:
  - The standard deviations for the normal (0.018) and pneumonia (0.020) classes indicate the variation in feature importance scores within each class. The pneumonia class shows slightly higher variability in feature importance across its images.

- **Sample Size**:
  - The analysis is based on 30 images for each class (healthy and pneumonia), which is a sufficient sample size for performing the t-test.

#### 3. **T-Test Results**:
- **T-Statistic**: The t-statistic is -5.3253. This value represents the standardized difference between the two sample means (healthy vs. pneumonia feature importance scores). A negative value means that the healthy class had lower feature importance on average compared to the pneumonia class.
  
- **P-Value**: The p-value is 0.0000, which is **much smaller than the significance level of 0.05**. This result indicates that the observed difference in feature importance between the healthy and pneumonia classes is highly unlikely to be due to random chance.

#### 4. **Hypothesis Testing**:

- **Null Hypothesis (H₀)**: 
  There is no significant difference in the feature importance highlighted by Grad-CAM between pneumonia and healthy chest X-ray images when using a pre-trained ResNet50 model.
  
- **Alternative Hypothesis (H₁)**: 
  There is a significant difference in the feature importance highlighted by Grad-CAM between pneumonia and healthy chest X-ray images when using a pre-trained ResNet50 model.

- **Conclusion**:
  - Since the **p-value (0.0000) is less than the significance level (0.05)**, we **reject the null hypothesis**. This means there is a **statistically significant difference** in how the pre-trained ResNet50 model highlights important regions between the healthy and pneumonia classes.
  
#### 5. **Effect Size (Cohen’s d)**:
- **Cohen's d**: The calculated Cohen’s d is **1.3985**, which indicates a **large effect size**. 
  - **Interpretation**: A large effect size means that the difference in feature importance scores between the two classes (healthy and pneumonia) is not only statistically significant but also substantial in magnitude.
  - In practical terms, the model places a **much greater emphasis** on certain regions in pneumonia images compared to healthy images, which could suggest that the model is detecting stronger signals or abnormalities in pneumonia cases.

#### 6. **Overall Interpretation**:
- The statistical analysis reveals that the **model behaves significantly differently when interpreting healthy and pneumonia images**. The higher feature importance scores for pneumonia images suggest that the model is assigning greater importance to certain regions in pneumonia images, potentially indicating areas of inflammation or abnormalities that are not present in healthy lungs.
  
- The **large effect size** (Cohen’s d = 1.3985) further supports the practical relevance of this difference. This suggests that the model’s attention mechanism (as visualized by Grad-CAM) strongly differentiates between the two classes.

- **Implication for Model Performance**: The model’s stronger focus on pneumonia regions might indicate that it is successfully identifying features associated with pneumonia (such as consolidation or opacities) that are not present in healthy lungs. However, the difference in focus areas, as mentioned in my earlier analysis (i.e., focus on the diaphragm and irrelevant regions), might suggest that the model is not entirely optimal yet, and further fine-tuning may be necessary to improve its interpretability and performance.

### Conclusion:
There is a **statistically significant** and **substantial difference** in how the pre-trained ResNet50 model uses image regions to make predictions for pneumonia and healthy images indicated by the Grad-CAM feature importance scores, with pneumonia images receiving higher feature importance scores. This suggests the model may be successfully identifying pneumonia-related features but may still require further refinement to focus on the most relevant areas for clinical decision-making.

## **Final Conclusion Report**

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

