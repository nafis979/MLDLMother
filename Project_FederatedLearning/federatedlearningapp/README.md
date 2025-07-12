# Federated Learning Application using Flower and PyTorch 🌸🔥

A complete and easy-to-run Federated Learning (FL) example built with **Flower** and **PyTorch**. This project trains a CNN model collaboratively across multiple virtual clients on the CIFAR-10 dataset.

---

## 🚩 Project Structure

Project_FederatedLearning/
├── pyproject.toml # Project configuration
├── README.md
├── federatedlearningapp/ # FL application code
│ ├── init.py
│ ├── task.py # Data loading, model definition, training & evaluation
│ ├── client_app.py # Flower client setup
│ └── server_app.py # Flower server setup
└── data/
└── cifar10/
├── train/ # CIFAR-10 training images
└── test/ # CIFAR-10 testing images


---

## ⚙️ Tech Stack

- **Federated Learning**: [Flower](https://flower.dev/)
- **Deep Learning**: [PyTorch](https://pytorch.org/)
- **Data**: CIFAR-10 dataset (from Kaggle or TorchVision)

---

## 🔧 Installation

### Step 1: Clone the repository

git clone https://github.com/yourusername/your-flower-fl-project.git
cd your-flower-fl-project

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