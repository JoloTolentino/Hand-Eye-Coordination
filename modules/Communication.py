import bluetooth

class ProstheticBT: 
    def __init__(self,ESP32MacAddress,BluetoothName):
        self.ESP32MacAddress = ESP32MacAddress 
        self.BlueToothName = BluetoothName
        self.Port = None 
        self.Found = True
        self.Connected = False
        self.Listening = False
        self.message = None
    
    def check(self):
        deviceList = bluetooth.discover_devices(lookup_names=True)
        for MACAddress,BTNames in deviceList:
            if self.ESP32MacAddress ==MACAddress or self.BlueToothName == BTNames:
                print("Device : ",self.BlueToothName, " with MAC address : ", self.ESP32MacAddress, " was found.")
                self.Found = True
            
    def connect(self):
        try:
            self.BTPairConnection = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
            for services in bluetooth.find_service(address=self.ESP32MacAddress):
                if  b'ESP32SPP\x00' == services['name']: 
                    self.Port = services['port']
            self.BTPairConnection.connect((self.ESP32MacAddress,self.Port))
            print("Conneced to :",self.BlueToothName)
            self.Listening =False
            self.send("<2>")
            print("Esp32 is now listening ....")
            while(self.Listening):
                self.recieve()
        except:
            print("BlueTooth Device : ",self.BlueToothName,  "Failed to Connect")
    
    def disconnect(self):
        self.BTPairConnection.close()

    def send(self,message):
        self.BTPairConnection.send(message)
        print("sending message : ", message)

    def recieve(self):
        self.data = str(self.BTPairConnection.recv(64))[2]
        print("Command recieved : ",self.data)
        if self.data: 
        		
            if self.data: 

                self.send("Predicting")
                self.message = 0
                self.Listening = False
            if self.data == 'q': 
                print("Quiting....")
                self.send("Quiting")
                self.message = 0
                self.disconnect()
                self.Listening = False
                
            if self.data == "s": 
                self.message = 1 
            else : 1
            

