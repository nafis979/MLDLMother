# smoketest_task.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from federatedlearningapp.task import load_data, Net, test

def main():
    print("🔥 Smoke test started!")

    # 1) Load a small shard (partition 0 of 10)
    trainloader, testloader = load_data(partition_id=0, num_partitions=10)

    # 2) Build model & move to device
    net = Net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # 3) Train for one epoch, printing every 100 batches
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss().to(device)
    total_loss = 0.0

    for batch_idx, (data, target) in enumerate(trainloader):
        if batch_idx % 100 == 0:
            print(f"  batch {batch_idx}/{len(trainloader)}")
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(trainloader)
    print(f"✅ Train loss (1 epoch): {avg_train_loss:.4f}")

    # 4) Run one evaluation pass
    test_loss, test_acc = test(net, testloader, device)
    print(f"✅ Test  loss: {test_loss:.4f}, Test accuracy: {test_acc:.2%}")


if __name__ == "__main__":
    main()
