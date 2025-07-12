# Federated Learning Application using Flower and PyTorch 🌸🔥

A clean and easy-to-run Federated Learning (FL) example built with **Flower** and **PyTorch**.  
This project trains a Convolutional Neural Network (CNN) model collaboratively across multiple virtual clients using the **CIFAR-10** dataset.

---

## ⚙️ Tech Stack

- **Federated Learning**: [Flower](https://flower.dev)
- **Deep Learning**: [PyTorch](https://pytorch.org)
- **Dataset**: CIFAR-10 (available via Kaggle or TorchVision)

---

## 🔧 Installation

1. **Clone the repository:**

```bash
git clone https://github.com/nafis979/MLDLMother.git
cd MLDLMother/Project_FederatedLearning
pip install -e .
mkdir -p data/cifar10
kaggle datasets download -d ayush1220/cifar10 -p data/cifar10 --unzip
flwr run .
'''

Let me know if you want to include metric screenshots, results, or next-step suggestions!


