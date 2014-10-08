import unittest
import reader, preprocess, activation, ffnet
import trainsg

class Test(unittest.TestCase):

    def testName(self):
        
        # read data
        data = reader.read("../data/log_1_fixed.txt", jstartline=15000, maxlines=5000)
                
        # preprocess data
        preprocess.preproc(data)
        
        # initialize model
        layers = [13, 8, 1]
        activf = [activation.linear(), activation.tanh(), activation.sigmoid()]   # activation.tanh(1.75, 3./2.),
        net = ffnet.FFNet(layers, activf)
        net.initw(0.1) 
        
        # create training options
        opts = trainsg.options()
        
        # write function
        f = open("../output/trainsg_test.txt", "w+")
        writefcn = lambda s: f.write(s)
                
        # training
        trainsg.train(data, opts, net, writefcn)
        
        # close file
        f.close()

if __name__ == "__main__":
    unittest.main()