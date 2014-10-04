import ffnet, activation, ranknet, la

def train(data):
    
    # network structure
    layers = [13, 8, 1]
    activf = [activation.linear(), activation.tanh(), activation.sigmoid()] 
    net = ffnet(layers, activation)
    
    # initialize weights
    w = net.getw();
    