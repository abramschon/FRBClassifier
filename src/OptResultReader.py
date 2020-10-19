#Reads through optimisation output and summarises the best performances

def main():
    with open("data/HPO.txt", "r") as f:
        for line in f:
            print(line)
            break



class Trial:
    """Object to store the results from a hyper-parameter optimization trial

    # Arguments
        trial_no: Int. The number of theiteration in the optimisation routine 
        loss: Dict. of floats 
            {"train": [<training loss, epoch 1>, <training loss, epoch 2>, ...] , 
             "val": [<validation loss, epoch 1>, <validation loss, epoch 2>, ...]}
        accuracy: Dict. of lists of floats 
            {"train": [<training accuracy, epoch 1>, <training accuracy, epoch 2>, ...] , 
             "val": [<validation accuracy, epoch 1>, <validation accuracy, epoch 2>, ...]}
        precision:  Dict. of lists of floats 
            {"train": [<training precision, epoch 1>, <training precision, epoch 2>, ...] , 
             "val": [<validation precision, epoch 1>, <validation precision, epoch 2>, ...]}
        recall:  Dict. of lists of floats 
            {"train": [<training recall, epoch 1>, <training recall, epoch 2>, ...] , 
             "val": [<validation recall, epoch 1>, <validation recall, epoch 2>, ...]}
        activation: Str. Whether "ReLU" or "Swish" was used
        layer_1: Dict. of different types
            {"units": Int. <no. filters in convolutional layer>, 
             "f_dim": Int. <dimensions of the filter, e.g. 2 imples a 2x2 filter>
             "pool": Bool. <whether max pooling was implemented>}
        layer_2: As with layer 1
        layer_3: As with layer 1
        gapool: Bool. Whether global average pooling, or a fully-connected layer was used on the end
    """
    def __init__(self, 
                trial_no, 
                loss, 
                accuracy, 
                precision, 
                recall,
                activation, 
                layer_1, 
                layer_2, 
                layer_3,
                gapool):
        self.trial_no = trial_no
        self.loss = loss
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.activation = activation
        self.layer_1 = layer_1
        self.layer_2 = layer_2
        self.layer_3 = layer_3
        self.gapool = gapool

        def to_string():
            #outputs trial in a LaTex table format


        def best(metric):
            #returns the lowest loss or highest accuracy, etc. 

    

