import unittest
import activation

class Test(unittest.TestCase):

    def setUp(self):
        pass
   
    def test_sigmoid(self):
        
        sigm = activation.sigmoid(-2.3, 10.4)

        x = 0.2;
        dx = 1.e-7;

        f = sigm.f(x)
        
        df_a = sigm.df(f)
        df_n = (sigm.f(x + dx) - sigm.f(x - dx)) / 2 / dx
        
        print "Numerical: %e, Analytical: %e" % (df_a, df_n)
        
        self.assertAlmostEqual(df_a, df_n, 8, "Numerical derivative is not equal to analytical")
    
    def test_tanh(self):
        
        tanh = activation.tanh(-2.7, -10.4)
        
        x = 22.2
        dx = 1.e-7
        
        f = tanh.f(x)
        
        df_a = tanh.df(f)
        df_n = (tanh.f(x + dx) - tanh.f(x - dx)) / 2 / dx
        
        print "Numerical: %e, Analytical: %e" % (df_a, df_n)
        
        self.assertAlmostEqual(df_a, df_n, 8, "Numerical derivative is not equal to analytical") 

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()