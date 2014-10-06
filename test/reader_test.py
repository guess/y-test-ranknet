import unittest
import reader, eda, preprocess2, train

class Test(unittest.TestCase):

    n = 20000

    def testName(self):
        
        # read data
        data = reader.read("../data/log_1_fixed.txt", self.n)
        
        # check order of queries
        eda.qorder(data)
                
        # compute basic data statistics
        stats = eda.stats(data)
        
        # preprocess data
        preprocess2.preproc(data)
        
        # start training
        train.train(data)

if __name__ == "__main__":
    unittest.main()