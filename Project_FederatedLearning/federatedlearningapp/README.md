Federated Learning Application using Flower and PyTorch 🌸🔥
A clean and easy-to-run Federated Learning (FL) example built with Flower and PyTorch. This project trains a Convolutional Neural Network (CNN) model collaboratively across multiple virtual clients using the CIFAR-10 dataset.
⚙️ Tech Stack

Federated Learning: Flower
Deep Learning: PyTorch
Dataset: CIFAR-10 (available via Kaggle or TorchVision)

🔧 Installation

Clone the repository:git clone https://github.com/nafis979/MLDLMother.git
cd MLDLMother/Project_FederatedLearning



🗃️ Dataset Setup

Download the CIFAR-10 dataset from Kaggle. Ensure you have the Kaggle CLI configured.
Run the following commands to download and unzip the dataset:mkdir -p data/cifar10
kaggle datasets download -d ayush1220/cifar10 -p data/cifar10 --unzip



🚀 Running the Project
To start a local simulation with both server and clients, use the following command:
flwr run .

This will:

Launch 10 local clients (virtual nodes)
Train each client on its own shard of the CIFAR-10 dataset
Perform federated averaging for 3 rounds (default configuration)

🌼 Happy Federated Learning! 🔥
