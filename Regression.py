class RegressionDemo :
    def __init__(self, a,b,c,tval):
        self.x = a #number of hours of study per day
        self.y = b # some value based on function h with w0=1, w1=2
        self.result = c # result based on O grade
        self.threshold = tval
        self.w0 = []
        self.w1 = []
        
    def h(self, w0, w1): # function h as per 18.6.1 univariate linear regression
        hresult= []
        for i in range(0 , len(self.x)):
            hresult.append(w0[i] + w1[i]*self.x[i]) 
        return hresult

    def checkthreshold(self, hresult):  # hard threshold function technique for regression
        flag = True
        actfun =[]
        for i in range(0 , len(self.x)) :            
            if (hresult[i] < self.threshold ):
                actfun.append( "no")
            else :
                actfun.append( "yes")
        #print("new hresult act fun:", actfun)

        for i in range(0 , len(self.x)) :
            if (actfun[i] != self.result[i]) :
                return False
        return True
            
    def training(self, w0, w1, alpha):
        i=1
        while i<=10 : # Max 1000 attempts           
                hresult = self.h(w0,w1)                
                #print("Attempt ", i  )
                #print("w1 :", w1 ,", hresult :" , hresult)
                if(self.checkthreshold(hresult)) :                    
                    self.w0 = w0
                    self.w1 = w1                    
                    print("In Attempt number ", i,  ", i got it! I think i have learnt enough: w0-->", self.w0, ", w1-->", self.w1)
                    break

                i = i +1      
                # Changing values of w0 and w1 to reduce error/loss using batch gradient descent learning rule given on page 720 below eqn 18.5
                for j in range(0,len(self.x)) :
                    w0[j] = w0[j] + alpha*(self.y[j] - hresult[j]) 
                    w1[j] = w1[j] + alpha*(self.y[j] - hresult[j]) *self.x[j]

        if(i>=1000):
            print("I am exhausted, tried 1000 iterations! plz change something else...")
        
        
a = [1,2,3,4,5,6]
b = [3,5,7,9,11,13]
c = ["no", "no", "no", "yes", "yes", "yes"]
p = RegressionDemo(a,b,c,9)
print("number of hours of study x=", p.x)
print("some function y=", p.y, " with threshold value :", p.threshold)
print("whether a student will get O grade =", p.result)

print("trying with w0=1, w1=3, alpha=0.01 -->")
p.training([1,1,1,1,1,1],[3,3,3,3,3,3], 0.01)


print("trying with w0=1, w1=1 , alpha=0.01 -->")
p.training([1,1,1,1,1,1],[1,1,1,1,1,1], 0.01)


print("trying with w0=1, w1=1 , alpha=0.02 -->")
p.training([1,1,1,1,1,1],[1,1,1,1,1,1], 0.02)
        
        
    
        


