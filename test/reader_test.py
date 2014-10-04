import unittest
import reader

class Test(unittest.TestCase):


    def testName(self):
        
        data = reader.read("../data/log_1")
        
        print data[-1]
                


if __name__ == "__main__":
    unittest.main()