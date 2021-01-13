def e_lrp2(model, img_tensor):
    layers = [module for module in model.modules() if not isinstance(module, torch.nn.Sequential)
              or not isinstance(module, torch.nn.Dropout)][1:]  # Gets all layers except sequential and dropout
    L = len(layers)
    A = [img_tensor] + [img_tensor] * L  # Makes the list
    for layer in range(L):
        if layer == 6:  # This is where we need to reshape the tensor
            print(A[layer].shape)
            A[layer] = A[layer].reshape(A[layer].size(0), -1)
            print(A[layer].shape)
        A[layer + 1] = layers[layer].forward(A[layer].to(device))

    T = A[-1].cpu().detach().numpy().tolist()[0]
    index = T.index(max(T))  # The index of the highest classification
    T = torch.FloatTensor(np.abs(np.array(T)) * 0)  # Sets all values to 0 - as a tensor
    T[index] = 1  # Sets the index of the highest classification to 1, as its all we care for

    R = [None] * L + [(A[-1].cpu() * T).data + 1e-6]
    for layer in range(0, L)[::-1]:
        A[layer] = A[layer].data.requires_grad_(True)

        # Treat max-pooling layers as average pooling layers in the backward pass
        if isinstance(layers[layer], torch.nn.MaxPool2d): layers[layer] = torch.nn.AvgPool2d(2)

        if isinstance(layers[layer], torch.nn.Conv2d) or isinstance(layers[layer], torch.nn.Linear):
            # Specifies the rho function that will be applied to the weights of the layer
            if 0 < layer <= 5:  # Gamma rule (LRP-gamma)
                rho = lambda p: p + 0.25 * p.clamp(min=0)
            else:  # Basic rule (LRP-0)
                rho = lambda p: p
            print(layer)
            # Step 1: Transform the weights of the layer and executes a forward pass
            z = newlayer(layers[layer], rho).forward(A[layer]) + 1e-9
            # Step 2: Element-wise division between the relevance of the next layer and the denominator
            print(layer+1, R[layer +1].shape, z.shape)
            s = (R[layer + 1].to(device) / z).data
            # Step 3: Calculate the gradient and multiply it by the activation layer
            (z * s).sum().backward()
            c = A[layer].grad

            R[layer] = (A[layer] * c).data
            if layer == 6:  # Reshape after layer fc1
                R[layer] = torch.reshape(R[layer], (1, 64, 7, 7))
                print(R[layer].shape)
        else:
            R[layer] = R[layer + 1]

    # Return the relevance of the input layer
    return R[0]

def newlayer(layer, rho):
    layer = copy.deepcopy(layer)
    layer.weight = torch.nn.Parameter(rho(layer.weight))
    layer.bias = torch.nn.Parameter(rho(layer.bias))
    return layer