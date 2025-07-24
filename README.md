Project Title: Brain Tumor MRI Image Classification using Deep Learning

Objective:
To build an efficient and accurate deep learning-based model that classifies brain MRI images into four tumor types — glioma, meningioma, pituitary, and no tumor — using both a custom CNN and a pretrained model (MobileNetV2). The goal also includes deploying the model via a Streamlit web app for real-time prediction.

Dataset Understanding & Preprocessing:
Source: A multiclass brain MRI dataset containing labeled images across training, validation, and test splits.
Initial Exploratory Analysis:
Verified the existence of train, valid, and test folders.
Counted number of classes and images per class to check for class imbalance.
Visualized sample images to understand image quality and category representation.
Checked for:
Duplicate images using hashing techniques.
Corrupt or unreadable files using PIL validation.
Image preprocessing:
Normalized pixel values to the [0, 1] range.
Resized all images to 224x224 pixels to match the input size for pretrained models.
 
Data Augmentation:
Applied augmentations to artificially increase training diversity:
Rotation, flipping, zoom, brightness shift, and translation.
Implemented using ImageDataGenerator.

Model Building:
Custom CNN:
Built from scratch using Conv2D, MaxPooling2D, Dropout, and Dense layers.
Included BatchNormalization and Dropout to prevent overfitting.
Compiled with categorical_crossentropy and Adam optimizer.
Achieved ~69% test accuracy.

Transfer Learning (MobileNetV2):
Used pretrained MobileNetV2 with imagenet weights, excluding the top layer.
Added custom head: GlobalAveragePooling2D, Dense, and Dropout layers.
Froze base layers and trained only the top classification head.
Achieved ~79% test accuracy — significantly outperforming the custom CNN.

Model Evaluation:
Evaluated both models using:
Accuracy
Precision, Recall, F1-score
Confusion Matrix
MobileNetV2 showed better generalization and class-wise performance, especially on classes like glioma and pituitary.

Model Selection & Justification:
Final model chosen: MobileNetV2.
Reasons:
Higher accuracy.
Better precision and recall across all classes.
Efficient for deployment due to fewer trainable parameters.

Explainability:
Since CNNs process pixel data, feature importance using SHAP/LIME was not necessary.
Emphasis was on classification performance and deployment readiness.

Model Saving:
Best performing model was saved as an .h5 file (best_mobilenetv2.h5) and also exported using joblib for compatibility.

Streamlit Deployment:
Built an intuitive Streamlit web application that allows:
Uploading an MRI image.
Performing real-time prediction using the trained MobileNetV2 model.
Displaying the predicted tumor type with a confidence score.
The app is lightweight, user-friendly, and ideal for clinical demo or educational use.

Final Summary:
This end-to-end project combined deep learning, computer vision, and web deployment to deliver a robust MRI classification system. From data wrangling to model training, evaluation, and deployment — each step was executed with practical considerations, producing a scalable and deployable AI solution for medical imaging classification.

