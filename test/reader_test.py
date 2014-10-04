import unittest
import reader, eda, preprocess

class Test(unittest.TestCase):

    n = 1000

    def testName(self):
        
        # read data
        data = reader.read("../data/log_1_fixed.txt", self.n)
        
        # check order of queries
        eda.qorder(data)
                
        # compute basic data statistics
        stats = eda.stats(data)
        
        # print for each variable
        for jvar in xrange(len(data[0][0])):
            print "%d: min: %e max: %e mean: %e var: %e" % (jvar, stats[0][jvar], stats[1][jvar], stats[2][jvar], stats[3][jvar])
        
        # preprocess data
        preprocess.preproc(data)
        
        print data[1]

if __name__ == "__main__":
    unittest.main()