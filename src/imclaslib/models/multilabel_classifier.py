import torch.nn as nn
import torch

class MultiLabelClassifier(nn.Module):
    def __init__(self, base_model, config):
        super().__init__()
        self.base_model = base_model
        self.num_classes = config.model_num_classes

        # Assuming base_model outputs features of size (batch_size, feature_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(config.train_dropout_prob / 100),
            nn.Linear(base_model.output_dim, self.num_classes)
        )
        self.output_dim = self.num_classes  # Set the output dimension
        self.image_size = config.model_image_size
        self.normalization_mean = config.dataset_normalization_mean
        self.normalization_std = config.dataset_normalization_std

    def forward(self, x):
        # Get the image features from the base model
        image_features = self.base_model(x)  # [batch_size, feature_dim]
        
        # Pass the image features through the classifier
        logits = self.classifier(image_features)  # [batch_size, num_classes]
        
        return logits