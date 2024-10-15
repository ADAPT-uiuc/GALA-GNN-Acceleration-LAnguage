import torch
import time
from numpy import mean, std, sum

def train(g, model, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_val_acc = 0
    best_test_acc = 0
    
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]

    '''
    epochTimes is array of times for each epoch (forward and backward pass), size is number of epochs
    layerTimes is time for each layer in each forward pass, size is number of layers * number of epochs
    forwardPropTimes is time of each forward pass, sizse is number of epochs
    '''
    epochTimes, layerTimes, forwardPropTimes = [], [], []
    for epoch in range(epochs):
        start = time.time()
        # Forward
        logits, forwardPropLayerTimes, forwardPropTime = model.forward(g, features)

        layerTimes.extend(forwardPropLayerTimes)
        forwardPropTimes.append(forwardPropTime)
        # Compute prediction
        pred = logits.argmax(1)

        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        loss = torch.nn.functional.cross_entropy(logits[train_mask], labels[train_mask])

        # Compute accuracy on training/validation/test
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epochTimes.append(time.time() - start)
        '''
        if epoch % 5 == 0:
            print(
                f"In epoch {epoch}, loss: {loss:.3f}, val acc: {val_acc:.3f} (best {best_val_acc:.3f}), test acc: {test_acc:.3f} (best {best_test_acc:.3f})"
        )
        '''
    
    epochTimes = epochTimes[4:]
    print("-"*25)
    print("Total execution time for {} epochs: {}".format(len(epochTimes), sum(epochTimes)))
    print("Best Validation Accuracy: {} Best Test Acc, {}".format(best_val_acc, test_acc))
    print("Average time per epoch: ", mean(epochTimes), "±", std(epochTimes))
    print("Average time of each layer: ", mean(layerTimes), "±", std(layerTimes))
    print("Average forward propogation time: ", mean(forwardPropTimes), "±", std(forwardPropTimes))

    return [best_val_acc, test_acc, str(mean(epochTimes)) + "±" + str(std(epochTimes)), str(mean(layerTimes)) + "±" + str(std(layerTimes)), str(mean(forwardPropTimes)) + "±" + str(std(forwardPropTimes))]

