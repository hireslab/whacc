# Model Card for WhACC (Whisker Automatic Contact Classifier)

## Model Details
- Developed by Dr. Andrew Hires Lab at the University of Southern Claifornia
- Contributors: Phillip Maire, Samson G. King, Jonathan Andrew Cheung, Stef Walker, Samuel Andrew Hires
- Model Type: 2-stage hybrid model integrating a Convolutional Neural Net (Resnet50V2) and a gradient boosted decision tree (LightGBM)
  - Alternative architectures tested: MobileNetV3-Small, MobileNetV3-Large, Inception-v3
  - Single architecture tests: ResNet50V2 alone, 2-stage hybrid model with long short-term memory (LSTM custom variants)
- Feature Extraction: ResNet50V2 (initialized with ImageNet weights, and trained on custom data)
- Hyperparameters Optimization: Optuna
- Generalizability: Augmented images, multiple timestep integration, dropout
- Model Release Date: 2023-02-01
- Model Version: V1.0
## Intended Use
- Purpose: Classifying rodent whisker touch frames from high-speed video for neuroscience analysis (e.g., electrophysiology data correlation)
- Constraints: High-speed video inputs (ideally 1000 fps) and limited to small object interactions (approx. from 61x61 to 96x96 pixels)
## Factors
- Image clarity and contrast of whisker to background critically affect performance, with higher contrast yielding better performance

## Metrics
- Custom Metrics: Touch errors per touch (TC-error), Edge errors per touch
- Traditional Metrics: AUC, Accuracy, Sensitivity, Precision, Geometric Mean at 0.5 decision threshold
- Confidence Measurement: 95% confidence intervals with 1.96* standard error
## Data
- Custom datasets for training, validation, and holdout test for retraining validation.
## Ethical Considerations
- All procedures were approved under USC IACUC protocols 20169 and 20788 in accordance with United States national guidelines issued by Office of Laboratory Animal Welfare of the National Institute of Health.
## Caveats and Recommendations
- For best performance increasing contrast between the whisker and background and position the object orthogonal to imaging plane to reduce whisker object occlusion.
- Use retraining with sample images for optimal performance.
- For LightGBM model using a weight between 2 and 100 for new training data compared to original training data or train without original training data if new data is significantly different.