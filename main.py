from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import timm

from tqdm.auto import tqdm


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TEST_IMAGE_PATH = "./archive/test/paler/002.jpg"


# Create the classifier model.
class SimpleImageClassifier(nn.Module):
    def __init__(self, num_classes=53):
        super(SimpleImageClassifier, self).__init__()

        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        # Remove the last layer of model (to replace with our own)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        enet_out_size = 1280
        self.classifier = nn.Linear(enet_out_size, num_classes)

    def forward(self, x):
        # Connect these parts and return the output.
        x = self.features(x)
        output = self.classifier(x)
        return output


class FacesDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        super(FacesDataset, self).__init__()
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def classes(self):
        return self.data.classes


def preprocess_image(image_path, transform):
    '''Load and preprocess the image.'''
    image = Image.open(image_path).convert("RGB")
    return image, transform(image).unsqueeze(0)


def predict(model, image_tensor, device):
    '''Predict using the model.'''
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.cpu().numpy().flatten()


def get_images_to_tensor_transform():
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    return transform


def test_model(model, class_names):
    # Top numbers to print.
    top_n = 5
    test_image = TEST_IMAGE_PATH

    original_image, image_tensor = preprocess_image(test_image, get_images_to_tensor_transform())

    # Total of all probabilities is 1.
    probabilities = predict(model, image_tensor, DEVICE)

    # Pair with class names
    result = dict(zip(class_names, probabilities))

    # Sort & print top N
    top_indices = probabilities.argsort()[-top_n:][::-1]
    print("Top Predictions:")
    for idx in top_indices:
        print(f"{class_names[idx]}: {probabilities[idx]:.4f}")

    return result


def loss_function(model, images, labels):
    '''Do training and validation loss calculation.'''

    example_out = model(images)

    # Criterion allows you to calculate loss by measuring outputs of model against labels.
    criterion = nn.CrossEntropyLoss()

    # Adam is one of the best optimizers.
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    criterion(example_out, labels)
    transform = get_images_to_tensor_transform()

    # Create data loaders for training, validation and testing.
    train_folder = './archive/train/'
    valid_folder = './archive/valid/'
    test_folder = './archive/test/'

    train_dataset = FacesDataset(train_folder, transform=transform)
    val_dataset = FacesDataset(valid_folder, transform=transform)
    test_dataset = FacesDataset(test_folder, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # How many times the model sees all images in the training data set and does forward back pass.
    num_epochs = 8
    train_losses, val_losses = [], []

    # Set pytorch to use the gpu rather than cpu.

    model.to(DEVICE)
    for epoch in range(num_epochs):
        model.train()

        # Keeping track of loss.
        running_loss = 0.0

        # Wrap the loop in a tqdm do get progress bars :)
        for images, labels in tqdm(train_loader, desc='Training Loop', leave=True):
            # Move inputs and labels to the GPU
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            # Resets to zero gradients of all model params optimizer is tracking.
            optimizer.zero_grad()

            # Run the images through the model.
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Loss and optimizer are linked to the same graph.
            # Do back propagation to calculate new grads, then apply them to the model with optimizer step.
            # Applying updated gradients means tweaking parameters in opposite direction of their gradient.
            # As gradient of each param points in direction of the steepest increase in loss.
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Set model to the validation phase.
        model.eval()
        running_loss = 0.0

        # Dont track gradients or build computational graph (only required when doing back prop).
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validation Loop'):
                # Move inputs and labels to the GPU
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)
            val_loss = running_loss / len(val_loader.dataset)
            val_losses.append(val_loss)
        tqdm.write(f"\nEpoch {epoch + 1}/{num_epochs} - Train loss: {train_loss}, Validation loss: {val_loss}")

    return model


def train():
    '''Train the model with the training dataset.'''

    # Need to transform PIL image data into tensors of the same size, so we create this transform to plug into class.
    transform = get_images_to_tensor_transform()

    dataset = FacesDataset(data_dir='./archive/train', transform=transform)

    # Data loader to parallelize loading of images in dataset.
    # Batch size is how many examples to pull each time we iterate.
    # Shuffle loads random data when iterating (usually done when training).
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for images, labels in dataloader:
        break

    # Should print a tensor [<batch size>, channels, image width, image height]
    # Labels is a one d array of length num labels.
    print(images.shape)

    model = SimpleImageClassifier(num_classes=5)
    print(model(images))

    # Train the model and print losses.
    model = loss_function(model, images, labels)

    torch.save(model.state_dict(), './model/model.pth')


def test():
    '''Test the model on the specified test image.'''

    # Create the dataset just to get the classes.
    dataset = FacesDataset(data_dir='./archive/train', transform=get_images_to_tensor_transform())

    # Recreate the model on the GPU.
    model = SimpleImageClassifier(num_classes=5)
    state_dict = torch.load('./model/model.pth', map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE)

    # Set the model to evaluation mode (important if using batch norm/dropout layers)
    model.eval()

    # Test the model with images.
    test_model(model, dataset.classes)


#train()
test()
