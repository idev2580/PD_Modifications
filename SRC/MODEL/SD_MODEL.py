import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_CHANNEL = 3       #SURE.{R|G|B}
KERNEL_NUMBER = 100     #Kernel Number specified in Bako et al,2022 was 100. We have SMALL input channels(No G-buffers), so smaller model used.
KERNEL_SIZE = 5         #Kernel for THIS MODEL
PADDING_SIZE = KERNEL_SIZE // 2
OUTPUT_KERNEL_SIZE = 11

class SureKernelPredictingNetwork(nn.Module):
    def __init__(self, cfg, input_channels=INPUT_CHANNEL, num_kernels=KERNEL_NUMBER, kernel_size=KERNEL_SIZE, output_size=OUTPUT_KERNEL_SIZE):
        """
        Kernel Predicting Network.
        Args:
            input_channels (int): Number of input channels (e.g., 3 for RGB).
            num_kernels (int): Number of kernels (feature maps) in each convolutional layer.
            kernel_size (int): Size of the convolutional kernel (e.g., 5x5).
            output_size (int): Size of the predicted kernel (e.g., 21x21).
        """
        super(SureKernelPredictingNetwork, self).__init__()
        self.cfg = cfg
        
        # Initialize convolutional layers
        self.conv_layers = nn.ModuleList()
        
        # First convolution layer
        self.conv_layers.append(
            nn.Conv2d(in_channels=input_channels, out_channels=num_kernels, kernel_size=kernel_size, padding=PADDING_SIZE) # padding=2 keeps spatial dims
        )
        '''self.conv_layers.append(
            nn.BatchNorm2d(num_kernels)
        )'''
        self.conv_layers.append(
            nn.ReLU()
        )
        
        # Intermediate convolution layers (8 layers)
        for _ in range(8):
            self.conv_layers.append(
                nn.Conv2d(in_channels=num_kernels, out_channels=num_kernels, kernel_size=kernel_size, padding=PADDING_SIZE)
            )
            '''self.conv_layers.append(
                nn.BatchNorm2d(num_kernels)
            )'''
            self.conv_layers.append(
                nn.ReLU()
            )
        
        # Final layer to output (3 x 21 x 21 = 3 x 441 channels per pixel)
        self.final_conv = nn.Conv2d(
            in_channels=num_kernels,
            out_channels= input_channels * output_size * output_size,  # Flattened kernel per pixel
            kernel_size=1  # Keep spatial resolution, no reduction
        )

        for layer in self.conv_layers:
            if(isinstance(layer, nn.Conv2d)):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)
        nn.init.xavier_uniform_(self.final_conv.weight)
        nn.init.constant_(self.final_conv.bias, 0)


        self.output_size = output_size
        self.input_channels = input_channels

    def kernel_inf(self, x):
        # Pass through convolutional layers
        for conv in self.conv_layers:
            x = conv(x)
        
        # Final convolution to produce output kernels
        x = self.final_conv(x)
        
        # Reshape to [Batch, Height, Width, 21, 21]
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, self.input_channels, self.output_size, self.output_size, height, width)  # Move channels into spatial format
        x = x.permute(0, 4, 5, 1, 2, 3)  # Reorder to [Batch, Height, Width, 3, 21, 21]
        return x
    
    def forward(self, x):
        kernels = self.kernel_inf(x)
        outputs = self.apply_kernels(x, kernels)
        return outputs
    
    def apply_kernels(self, in_tensor, kernels):
        koutputs = kernels
        # Apply this kernel into image
        #   - in_tensor.size() = [batch, 3, h, w]
        #   - kernels.size()   = [batch, h, w, 3, 21, 21]
        batch_num = in_tensor.shape[0]
        h = in_tensor.shape[2]
        w = in_tensor.shape[3]

        patches = F.unfold(in_tensor, kernel_size=(OUTPUT_KERNEL_SIZE, OUTPUT_KERNEL_SIZE), padding=(OUTPUT_KERNEL_SIZE // 2, OUTPUT_KERNEL_SIZE // 2))
        #print(patches.size())
        patches = patches.view(batch_num, 3, OUTPUT_KERNEL_SIZE, OUTPUT_KERNEL_SIZE, h, w)
        #print(patches.size())
        patches = patches.permute(0, 4, 5, 1, 2, 3) # N, H, W, C, kH, kW

        
        #print(koutputs.size())

        outputs = patches * koutputs
        outputs = outputs.sum(dim=[4,5]) # Sum over Kernel Height and Kernel Width
        outputs = outputs.view(batch_num, h, w, 3)
        return outputs
    def is_weight_nan(self):
        for name, param in self.named_parameters():
            if torch.isnan(param).any():
                print("Nan found in name{}".format(name))
                return True
        return False
    
    def weight_nan_remover(self):
        for name, param in self.named_parameters():
            if torch.isnan(param).any():
                param.data = torch.where(torch.isnan(param.data), torch.zeros_like(param.data), param.data)
        return

def save_model(model,path):
    torch.save(model.state_dict(), path)

def load_model(model, path, is_eval=False, is_cpu=False):
    model.load_state_dict(torch.load(path, weights_only=True))
    if(is_cpu):
        model.to(torch.device('cpu'))
    else:
        model.to(torch.device('cuda'))
    
    if(is_eval):
        model.eval()
    return model