class NaiveBayes:    def __init__(self, f, r):        self.features = f        self.response = r    def predict(self,custcase):        anskeys = list(self.response.keys())        ansvalues = dict.fromkeys(anskeys,0)        #print(custcase)        for key in anskeys :            ansvalues[key] = self.response[key]            for ikey, ival in custcase.items() :                ansvalues[key] = ansvalues[key] * self.features[ikey][ival][key]        print(ansvalues)                #calculating MAP        maxkey=""        maxans=-1        for ikey, ival in ansvalues.items():            if ival > maxans :                maxans= ival                maxkey = ikey        return maxkeyresponse = {"wait":0.4, "leave":0.6}features = {"reservation":{"yes" : {"wait":0.5, "leave":0.666667},"no" : {"wait":0.5, "leave":0.333333}} ,"time>30":{"yes" : {"wait":0.25, "leave":0.83333},"no" : {"wait":0.75, "leave":0.16667}}                  }naivebayers = NaiveBayes(features, response)resstatus = input("have you reserved table?")timestatus = input("Will it take more than 30 mins?")custcase = {"reservation":resstatus, "time>30":timestatus}print("Manager predicts that Customer will :" , naivebayers.predict(custcase) )