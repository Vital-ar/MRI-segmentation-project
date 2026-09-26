import torch
import torch.nn as nn
from src.logging import logger





class ConvBlock(nn.Module):


    def __init__(self, in_channels: int, n_layers: int, out_channels: int, kernel_size: int):

        super().__init__()

        blocks = []
        padding = (kernel_size - 1)//2

        for i in range(n_layers):
        
            

            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding = padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )

            blocks.append(block)

            in_channels = out_channels

        self.conv_block = nn.Sequential(*blocks)


    def forward(self, x):

        return self.conv_block(x)






        

class EncoderBlock(nn.Module):


    def __init__(self, in_channels: int, out_channels: int, n_layers: int, kernel_size: int):

        super().__init__()

        
        self.conv_block = ConvBlock(in_channels, n_layers, out_channels, kernel_size)
        self.pool = nn.MaxPool2d(2,2)


    def forward(self, x):

        skip = self.conv_block(x)

        out = self.pool(skip)
        return skip, out


        




class AttentionGate(nn.Module):


    def __init__(self, inp_channels, s_channels, out_channels):

        super().__init__()

        self.Wg = nn.Sequential(
            nn.Conv2d(inp_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        self.Ws = nn.Sequential(
            nn.Conv2d(s_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        self.atten = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU()


    def forward(self, inp, skip):
        g = self.Wg(inp)       
        s = self.Ws(skip)       
        out = self.relu(g+s)
        attention = self.atten(out)   
        return s * attention        









class DecoderBlock(nn.Module):
    def __init__(self,  inp_channels: int, skip_channels: int,  out_channels: int, n_layers: int, kernel_size: int):
        super().__init__()

        self.conv = ConvBlock(inp_channels + skip_channels, n_layers, out_channels, kernel_size)
        self.upsample = nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners= False)
        self.attention = AttentionGate(inp_channels, skip_channels, out_channels)


    def forward(self, x , skip):
        x = self.upsample(x)
        out= self.attention(x, skip)

        out = torch.cat([out, x], dim = 1)
        

        return self.conv(out)





class MRIFlexAttentionUNet(nn.Module):

    def __init__(self, inp_channels: int, first_conv_out_channels: int, num_classes: int, depth: int, n_encoder_conv_layers: int, n_decoder_conv_layers: int,  kernel_sizes: list|int):

        super().__init__()

        logger.logging.info('creation of the model...')


        if type(kernel_sizes) == int:
            kernel_sizes = [kernel_sizes] * depth * 2

        
        encoder_list = []
        decoder_list = []
        #   in_channels 3, 4, 5 
        #   n_encoder_conv_layers: 2, 3
        #   n_decoder_conv_layers: 2, 3
        #   kernel_size: 3, 5
        #   first_conv_out_channels: 32, 64
        len_enc_dec = len(kernel_sizes)

        out_channels = first_conv_out_channels
        for i in range(depth):

            
            
            enc = EncoderBlock(inp_channels, out_channels, n_encoder_conv_layers, kernel_sizes[i])
            dec = DecoderBlock(out_channels*2, out_channels, out_channels, n_decoder_conv_layers, kernel_sizes[len_enc_dec -  1 - i])
            encoder_list.append(enc)
            decoder_list.append(dec)

            inp_channels = out_channels
            out_channels *= 2
            

        decoder_list.reverse()

        self.encoder_list = nn.ModuleList(encoder_list)
        self.decoder_list = nn.ModuleList(decoder_list)
        
        self.bottleneck = nn.Conv2d(inp_channels, out_channels, 3, padding = 1)
        self.head = nn.Conv2d(first_conv_out_channels, num_classes, 1)

        logger.logging.info('creation of the model compleated')


    def forward(self, x):

        skips = []

        for enc in self.encoder_list:

            skip, out = enc(x)  
            skips.append(skip)
            x = out         

        skips.reverse()

        x = self.bottleneck(x)

        for i, dec in enumerate(self.decoder_list):

            x = dec(x, skips[i]) 


        return self.head(x)







class ConvBlockGroupNorm(nn.Module):


    def __init__(self, in_channels: int, n_layers: int, out_channels: int, kernel_size: int):

        super().__init__()

        blocks = []
        padding = (kernel_size - 1)//2

        for i in range(n_layers):
        
            

            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding = padding),
                nn.GroupNorm(out_channels//8, out_channels),
                nn.ReLU()
            )

            blocks.append(block)

            in_channels = out_channels

        self.conv_block = nn.Sequential(*blocks)


    def forward(self, x):

        return self.conv_block(x)






        

class EncoderBlockGroupNorm(nn.Module):


    def __init__(self, in_channels: int, out_channels: int, n_layers: int, kernel_size: int):

        super().__init__()

        
        self.conv_block = ConvBlockGroupNorm(in_channels, n_layers, out_channels, kernel_size)
        self.pool = nn.MaxPool2d(2,2)


    def forward(self, x):

        skip = self.conv_block(x)

        out = self.pool(skip)
        return skip, out


        




class AttentionGateGroupNorm(nn.Module):


    def __init__(self, inp_channels, s_channels, out_channels):

        super().__init__()

        self.Wg = nn.Sequential(
            nn.Conv2d(inp_channels, out_channels, kernel_size=1),
            nn.GroupNorm(out_channels//8, out_channels)
        )
        self.Ws = nn.Sequential(
            nn.Conv2d(s_channels, out_channels, kernel_size=1),
            nn.GroupNorm(out_channels//8, out_channels)        )
        self.atten = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU()


    def forward(self, inp, skip):
        g = self.Wg(inp)       
        s = self.Ws(skip)       
        out = self.relu(g+s)
        attention = self.atten(out)   
        return s * attention        









class DecoderBlockGroupNorm(nn.Module):
    def __init__(self,  inp_channels: int, skip_channels: int,  out_channels: int, n_layers: int, kernel_size: int):
        super().__init__()

        self.conv = ConvBlockGroupNorm(inp_channels + skip_channels, n_layers, out_channels, kernel_size)
        self.upsample = nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners= False)
        self.attention = AttentionGateGroupNorm(inp_channels, skip_channels, out_channels)


    def forward(self, x , skip):
        x = self.upsample(x)
        out= self.attention(x, skip)

        out = torch.cat([out, x], dim = 1)
        

        return self.conv(out)





class MRIFlexAttentionUNetGroupNorm(nn.Module):

    def __init__(self, inp_channels: int, first_conv_out_channels: int, num_classes: int, depth: int, n_encoder_conv_layers: int, n_decoder_conv_layers: int,  kernel_sizes: list|int):

        super().__init__()

        logger.logging.info('creation of the model...')


        if type(kernel_sizes) == int:
            kernel_sizes = [kernel_sizes] * depth * 2

        
        encoder_list = []
        decoder_list = []
        #   in_channels 3, 4, 5 
        #   n_encoder_conv_layers: 2, 3
        #   n_decoder_conv_layers: 2, 3
        #   kernel_size: 3, 5
        #   first_conv_out_channels: 32, 64
        len_enc_dec = len(kernel_sizes)

        out_channels = first_conv_out_channels
        for i in range(depth):

            
            
            enc = EncoderBlockGroupNorm(inp_channels, out_channels, n_encoder_conv_layers, kernel_sizes[i])
            dec = DecoderBlockGroupNorm(out_channels*2, out_channels, out_channels, n_decoder_conv_layers, kernel_sizes[len_enc_dec -  1 - i])
            encoder_list.append(enc)
            decoder_list.append(dec)

            inp_channels = out_channels
            out_channels *= 2
            

        decoder_list.reverse()

        self.encoder_list = nn.ModuleList(encoder_list)
        self.decoder_list = nn.ModuleList(decoder_list)
        
        self.bottleneck = nn.Conv2d(inp_channels, out_channels, 3, padding = 1)
        self.head = nn.Conv2d(first_conv_out_channels, num_classes, 1)

        logger.logging.info('creation of the model compleated')


    def forward(self, x):

        skips = []

        for enc in self.encoder_list:

            skip, out = enc(x)  
            skips.append(skip)
            x = out         

        skips.reverse()

        x = self.bottleneck(x)

        for i, dec in enumerate(self.decoder_list):

            x = dec(x, skips[i]) 


        return self.head(x)
            