import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import time
from numpy import mean, std

def train(g, model, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_val_acc = 0
    best_test_acc = 0

    features = g.x
    labels = g.y
    train_mask = g.train_mask
    val_mask = g.val_mask
    test_mask = g.test_mask

    model.train() # changes some boolean value that lets model know its in training mode
    '''
    epochTimes is array of times for each epoch (forward and backward pass), size is number of epochs
    layerTimes is time for each layer in each forward pass, size is number of layers * number of epochs
    forwardPropTimes is time of each forward pass, size is number of epochs
    '''
    epochTimes, layerTimes, forwardPropTimes = [], [], []
    for epoch in range(epochs):
        start = time.time()
        # Forward
        logits, forwardPropLayerTimes, forwardPropTime = model.forward(g, features)

        layerTimes.extend(forwardPropLayerTimes)
        forwardPropTimes.append(forwardPropTime)
        # Prediction
        pred = logits.argmax(1)

        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epochTimes.append(time.time() - start)
        if epoch % 5 == 0:
            print(
                f"In epoch {epoch}, loss: {loss:.3f}, val acc: {val_acc:.3f} (best {best_val_acc:.3f}), test acc: {test_acc:.3f} (best {best_test_acc:.3f})"
        )
            
    print("-"*25)
    print("Average time per epoch: ", mean(epochTimes), "±", std(epochTimes))
    print("Average time of each layer: ", mean(layerTimes), "±", std(layerTimes))
    print("Average forward propogation time: ", mean(forwardPropTimes), "±", std(epochTimes))

    return [best_val_acc, test_acc, str(mean(epochTimes)) + "±" + str(std(epochTimes)), str(mean(layerTimes)) + "±" + str(std(layerTimes)), str(mean(forwardPropTimes)) + "±" + str(std(forwardPropTimes))]