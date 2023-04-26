
class Network_cal():
    def __init__(self, input_size = 256):
        
        self.input_size = input_size

    
    def Calc_convelution(self, layers=3, filter_size =32, kernel = 4, kernel_p = 2, stride = 2, stride_p = 2, padding = 1, padding_p = 0, pooling = False):
        
        encoder = []
        input_size = self.input_size
        for i,layer in enumerate(range(layers)):
            output_size = (input_size+2*padding - kernel)//stride + 1
            encoder.append(output_size)
            if pooling:
                if i+1 < layers:
                    output_size = (output_size+2*padding_p - kernel_p)//stride_p + 1
                    encoder.append(output_size)
            
            input_size = output_size
        
        decoder = encoder.copy()
        decoder.reverse()
        
        linar = encoder[-1] * filter_size
        
        return encoder, decoder, linar

        
        
    #def Calc_lin():
        
        
     #   return output_size
        
    
Nc = Network_cal()
print(Nc.Calc_convelution(layers=3, kernel = 4, kernel_p = 2, stride = 1, stride_p = 2, padding = 1, padding_p = 0, pooling = True))