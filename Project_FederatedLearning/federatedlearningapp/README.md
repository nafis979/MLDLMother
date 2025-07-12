# Federated Learning Application using Flower and PyTorch 🌸🔥

A complete and easy-to-run Federated Learning (FL) example built with **Flower** and **PyTorch**. This project trains a CNN model collaboratively across multiple virtual clients on the CIFAR-10 dataset.

---

## ⚙️ Tech Stack

- **Federated Learning**: [Flower](https://flower.dev/)
- **Deep Learning**: [PyTorch](https://pytorch.org/)
- **Data**: CIFAR-10 dataset (from Kaggle or TorchVision)

---

## 🔧 Installation

### Step 1: Clone the repository

git clone https://github.com/nafis979/MLDLMother.git
cd MLDLMOTHER/Project_FederatedLearning

🗃️ Dataset Setup

Download CIFAR-10 from Kaggle
Make sure you have the Kaggle CLI set up:
mkdir -p data/cifar10
kaggle datasets download -d ayush1220/cifar10 -p data/cifar10 --unzip

🚀 Running the Project

Local Simulation (Easy start)
Run both server and clients in one command:

flwr run .

This will start:
10 local clients (virtual nodes)
Each client trains on its own shard of CIFAR-10 data
Federated averaging for 3 rounds (default)

Happy Federated Learning! 🌼🔥
