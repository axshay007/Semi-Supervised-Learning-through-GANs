# Semi-Supervised-Learning-through-GANs

Imagine being a Computer Vision Engineer working on a proof of concept for a mobile app designed to identify melanomas in photos of moles, taken typically using cellphone cameras. Early detection of melanoma is crucial for a patient's long-term survival, making this project incredibly important.

Our goal is to develop a model for malignant versus benign image classification, specifically focusing on low-resolution images like those captured by cellphones. The challenge lies in having a dataset with only 200 training images—half labeled as melanomas and half as benign. Additionally, there's a vast number of unlabeled images, reflecting a common scenario in medical imaging where labeled data is scarce compared to the abundance of unlabeled data.

To overcome this hurdle and build a model that generalizes well on unseen data, we're leveraging semi-supervised learning. We'll integrate this with data augmentation techniques. Despite the limited labeled training data, we'll employ a DCGAN (Deep Convolutional GAN) to generate mole-like images and use it as a classifier to train on both labeled malignant and benign images. This approach ensures our model (Discriminator) generalizes effectively on unseen images, making it a robust tool for melanoma detection.
