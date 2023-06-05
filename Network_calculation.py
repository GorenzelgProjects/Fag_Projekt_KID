
'''Simple helper class to deal with a CNN network's calculations of the individual layers' in- and output-sizes, 
creating a list with the expected sizes of the CNN architecture with a set of given inputs. These inputs are:
number of layers, filter size, kernel size, "kernel_p"-size, stride of kernel, stride of "kernel_p", padding of kernel
, padding of "kernel_p", and if pooling should be True or False. the "_p" is for pooling layers only.'''

class Network_cal():
    def __init__(self, input_size = 256):
        
        self.input_size = input_size

    
    def Calc_convolution(self, layers=3, filter_size =32, kernel = 4, kernel_p = 2, stride = 2, stride_p = 2, padding = 1, padding_p = 0, pooling = False):
        
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
        
        for i in range(1,len(decoder)-1,2):
            decoder.pop(i)
            encoder.pop(i)


        linear = encoder[-1] * filter_size
        
        return encoder, decoder, linear

        
        
    #def Calc_lin():
        
        
     #   return output_size
        
if __name__ == "__main__":   
    Nc = Network_cal()
    print(Nc.Calc_convolution(layers=2, kernel = 3, kernel_p = 2, stride = 1, stride_p = 2, padding = 1, padding_p = 0, pooling = True))

