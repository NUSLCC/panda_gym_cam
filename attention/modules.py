import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import matplotlib.pyplot as plt
import cv2

def _get_out_shape(in_shape, layers, attn=False):
	#print(f"IN_SHAPE: {in_shape}")
	x = torch.randn(*in_shape).unsqueeze(0)
	if attn:
		return layers(x, x, x).squeeze(0).shape
	else:
		return layers(x).squeeze(0).shape
	
def _get_out_shape_modified(in_shape, layers, attn=False):
	x = torch.randn(*in_shape).unsqueeze(0)
	if attn:
		return layers(x, x, x)[0].squeeze(0).shape # return layers(x,x,x)[0] because layers(x,x,x) is a tuple (x, attention_weights)
	else:
		return layers(x).squeeze(0).shape

def gaussian_logprob(noise, log_std):
	"""Compute Gaussian log probability"""
	residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
	return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
	"""Apply squashing function, see appendix C from https://arxiv.org/pdf/1812.05905.pdf"""
	mu = torch.tanh(mu)
	if pi is not None:
		pi = torch.tanh(pi)
	if log_pi is not None:
		log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
	return mu, pi, log_pi


def orthogonal_init(m):
	if isinstance(m, nn.Linear):
		nn.init.orthogonal_(m.weight.data)
		if hasattr(m.bias, 'data'):
			m.bias.data.fill_(0.0)
	elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
		gain = nn.init.calculate_gain('relu')
		nn.init.orthogonal_(m.weight.data, gain)
		if hasattr(m.bias, 'data'):
			m.bias.data.fill_(0.0)


class NormalizeImg(nn.Module):
	def __init__(self, mean_zero=False):
		super().__init__()
		self.mean_zero = mean_zero

	def forward(self, x):
		if self.mean_zero:
			return x/255. - 0.5
		return x/255.


class Flatten(nn.Module):
	def __init__(self):
		super().__init__()
		
	def forward(self, x):
		return x.view(x.size(0), -1)


class Identity(nn.Module):
	def __init__(self, obs_shape=None, out_dim=None):
		super().__init__()
		self.out_shape = obs_shape
		self.out_dim = out_dim
	
	def forward(self, x):
		print(f'Identity projection input/output: {x.shape}')
		return x


class RandomShiftsAug(nn.Module):
	def __init__(self, pad):
		super().__init__()
		self.pad = pad

	def forward(self, x):
		n, c, h, w = x.size()
		assert h == w
		padding = tuple([self.pad] * 4)
		x = F.pad(x, padding, 'replicate')
		eps = 1.0 / (h + 2 * self.pad)
		arange = torch.linspace(-1.0 + eps,
								1.0 - eps,
								h + 2 * self.pad,
								device=x.device,
								dtype=x.dtype)[:h]
		arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
		base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
		base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

		shift = torch.randint(0,
							  2 * self.pad + 1,
							  size=(n, 1, 1, 2),
							  device=x.device,
							  dtype=x.dtype)
		shift *= 2.0 / (h + 2 * self.pad)

		grid = base_grid + shift
		return F.grid_sample(x,
							 grid,
							 padding_mode='zeros',
							 align_corners=False)



class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_query = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.conv_key = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.conv_value = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.in_channels = in_channels

    def forward(self, query, key, value):
        N, C, H, W = query.shape
        assert query.shape == key.shape == value.shape, "Key, query and value inputs must be of the same dimensions in this implementation"
        q = self.conv_query(query).reshape(N, C, H*W)#.permute(0, 2, 1)
        k = self.conv_key(key).reshape(N, C, H*W)#.permute(0, 2, 1)
        v = self.conv_value(value).reshape(N, C, H*W)#.permute(0, 2, 1)
        attention = k.transpose(1, 2)@q / C**0.5
        attention = attention.softmax(dim=1)
        output = v@attention
        output = output.reshape(N, C, H, W)
        return query + output # Add with query and output
	
class SelfAttentionWithWeights(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_query = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.conv_key = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.conv_value = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.in_channels = in_channels

    def forward(self, query, key, value):
        N, C, H, W = query.shape
        assert query.shape == key.shape == value.shape, "Key, query and value inputs must be of the same dimensions in this implementation"
        q = self.conv_query(query).reshape(N, C, H*W)#.permute(0, 2, 1)
        k = self.conv_key(key).reshape(N, C, H*W)#.permute(0, 2, 1)
        v = self.conv_value(value).reshape(N, C, H*W)#.permute(0, 2, 1)
        attention = k.transpose(1, 2)@q / C**0.5
        attention = attention.softmax(dim=1)
        output = v@attention
        output = output.reshape(N, C, H, W)
        print(f"Attention shape: {attention.shape}")
        print(f"Query shape: {query.shape}")
        print(f"Output shape: {output.shape}")
        return query + output, attention # Add with query and output. Attention == attn weights

class AttentionBlockWithWeights(nn.Module):
	def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, contextualReasoning=False):
		super().__init__()
		self.norm1 = norm_layer(dim)
		self.norm2 = norm_layer(dim)
		self.norm3 = norm_layer(dim)
		self.attn = SelfAttentionWithWeights(dim[0])
		self.context = contextualReasoning
		temp_shape = _get_out_shape_modified(dim, self.attn, attn=True)
		self.out_shape = _get_out_shape_modified(temp_shape, nn.Flatten())
		self.apply(orthogonal_init)
		
		self.dim = dim

	def forward(self, query, key, value):
		# print(f'Attention block Input shape: {self.dim}')
		# print(f"Query, key and value shapes: {query.shape}, {key.shape}, {value.shape}")
		# print(f'Norm shapes: {self.norm1(query).shape}, {self.norm2(key).shape}, {self.norm3(value).shape}')
		x, attention_weights = self.attn(self.norm1(query), self.norm2(key), self.norm3(value))
		if self.context:
		#	print(f'Attention block Output shape: {x.shape}')
			return x, attention_weights
		else:
			x = x.flatten(start_dim=1)
		#	print(f'Attention block Output shape: {x.shape}')
			return x, attention_weights
		
class AttentionBlock(nn.Module):
	def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, contextualReasoning=False):
		super().__init__()
		self.norm1 = norm_layer(dim)
		self.norm2 = norm_layer(dim)
		self.norm3 = norm_layer(dim)
		self.attn = SelfAttention(dim[0])
		self.context = contextualReasoning
		temp_shape = _get_out_shape(dim, self.attn, attn=True)
		self.out_shape = _get_out_shape(temp_shape, nn.Flatten())
		self.apply(orthogonal_init)
		
		self.dim = dim

	def forward(self, query, key, value):
		# print(f'Attention block Input shape: {self.dim}')
		# print(f"Query, key and value shapes: {query.shape}, {key.shape}, {value.shape}")
		# print(f'Norm shapes: {self.norm1(query).shape}, {self.norm2(key).shape}, {self.norm3(value).shape}')
		x = self.attn(self.norm1(query), self.norm2(key), self.norm3(value))
		if self.context:
		#	print(f'Attention block Output shape: {x.shape}')
			return x
		else:
			x = x.flatten(start_dim=1)
		#	print(f'Attention block Output shape: {x.shape}')
			return x


class SharedCNN(nn.Module):
	def __init__(self, obs_shape, num_layers=11, num_filters=32, mean_zero=False):
		super().__init__()
		assert len(obs_shape) == 3
		self.num_layers = num_layers
		self.num_filters = num_filters
		self.layers = [NormalizeImg(mean_zero), nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
		for _ in range(1, num_layers):
			self.layers.append(nn.ReLU())
			self.layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
		self.layers = nn.Sequential(*self.layers)
		self.out_shape = _get_out_shape(obs_shape, self.layers)
		self.apply(orthogonal_init)

	def forward(self, x):
	#	print(f'Shared CNN Input shape: {x.shape}')
	#	print(f'Shared CNN Output shape: {self.layers(x).shape}')
		return self.layers(x)


class HeadCNN(nn.Module):
	def __init__(self, in_shape, num_layers=3, num_filters=32, flatten=True):
		super().__init__()
		self.layers = []
		for _ in range(0, num_layers):
			self.layers.append(nn.ReLU())
			self.layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
		if flatten:
			self.layers.append(Flatten())
		self.layers = nn.Sequential(*self.layers)
		self.out_shape = _get_out_shape(in_shape, self.layers)
		self.apply(orthogonal_init)

	def forward(self, x):
	#	print(f'Head CNN Input shape: {x.shape}')
	#	print(f'Head CNN Output shape: {self.layers(x).shape}')
		return self.layers(x)

		

class Integrator(nn.Module):
	def __init__(self, in_shape_1, in_shape_2, num_filters=32, concatenate=True):
		super().__init__()
		self.relu = nn.ReLU()
		if concatenate:
			self.conv1 = nn.Conv2d(in_shape_1[0]+in_shape_2[0], num_filters, (1,1))
		else:
			self.conv1 = nn.Conv2d(in_shape_1[0], num_filters, (1,1))
		self.apply(orthogonal_init)

	def forward(self, x):
		x = self.conv1(self.relu(x))
		return x


class Encoder(nn.Module):
	def __init__(self, shared_cnn, head_cnn, projection, attention=None):
		super().__init__()
		self.shared_cnn = shared_cnn
		self.head_cnn = head_cnn
		self.projection = projection
		self.attention = attention
		self.out_dim = projection.out_dim

	def forward(self, x, detach=False):
		x = self.shared_cnn(x)
		x = self.head_cnn(x)
		if detach:
			x = x.detach()
		x = self.projection(x)
		return x
		
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(orthogonal_init)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

	
class MultiViewCrossAttentionEncoderModifiedORG(nn.Module):
	"""
	Input is the dual environment obs (active and static images already concatenated in core.py). Applies cross attention.
	"""
	def __init__(self, shared_cnn_1, shared_cnn_2, integrator, head_cnn, projection, attention1=None, attention2=None, mlp1=None, mlp2=None, norm1=None, norm2=None, concatenate=True):
		super().__init__()
		self.shared_cnn_1 = shared_cnn_1
		self.shared_cnn_2 = shared_cnn_2
		self.integrator = integrator
		self.head_cnn = head_cnn
		self.projection = projection
		self.relu = nn.ReLU()
		self.attention1 = attention1
		self.attention2 = attention2

		self.out_dim = projection.out_dim
		self.mlp1 = mlp1
		self.norm1 = norm1
		self.mlp2 = mlp2
		self.norm2 = norm2
		self.concatenate = concatenate

		self.out_dim = projection.out_dim

	def forward(self, x1, x2, detach=False):
     
		#print(f"x1 shape {x1.shape} and x2 shape {x2.shape}")
     
		# fig, ax = plt.subplots(1,2)
		# ax[0].imshow(x1[0].permute(1,2,0).detach().cpu().numpy())
		# ax[1].imshow(x2[0].permute(1,2,0).detach().cpu().numpy())
		# plt.show()

		x1 = self.shared_cnn_1(x1) #3rd Person
		x2 = self.shared_cnn_2(x2)

		B, C, H, W = x1.shape 
		
		x1 = self.attention1(x1, x2, x2) # Contextual reasoning on 3rd person image based on 1st person image
		x1 = self.norm1(x1)
		x1 = x1.view(B, C, -1).permute(0, 2, 1)
		x1 = self.mlp1(x1).permute(0, 2, 1).contiguous().view(B, C, H, W)

		x2 = self.attention2(x2, x1, x1) # Contextual reasoning on 1st person image based on 3rd person image
		x2 = self.norm2(x2)
		x2 = x2.view(B, C, -1).permute(0, 2, 1)
		x2 = self.mlp2(x2).permute(0, 2, 1).contiguous().view(B, C, H, W)

		if self.concatenate:
			# Concatenate features along channel dimension
			x = torch.cat((x1, x2), dim=1) 
		else:
			x = x1 + x2 

		x = self.integrator(x)
		x = self.head_cnn(x)

		if detach:
			x = x.detach()

		x = self.projection(x)
		
		return x
	

class MultiViewCrossAttentionEncoderModifiedTesting(nn.Module):
	"""
	Input is the dual environment obs (active and static images already concatenated in core.py). Applies cross attention.
	"""
	def __init__(self, shared_cnn_1, shared_cnn_2, integrator, head_cnn, projection, attention1=None, attention2=None, mlp1=None, mlp2=None, norm1=None, norm2=None, concatenate=True):
		super().__init__()
		self.shared_cnn_1 = shared_cnn_1
		self.shared_cnn_2 = shared_cnn_2
		self.integrator = integrator
		self.head_cnn = head_cnn
		self.projection = projection
		self.relu = nn.ReLU()
		self.attention1 = attention1
		self.attention2 = attention2

		self.out_dim = projection.out_dim
		self.mlp1 = mlp1
		self.norm1 = norm1
		self.mlp2 = mlp2
		self.norm2 = norm2
		self.concatenate = concatenate

		self.out_dim = projection.out_dim

	def forward(self, x1, x2, detach=False):

		org_img_x1 = x1
		org_img_x2 = x2
		
		x1 = self.shared_cnn_1(x1) #1st Person
		x2 = self.shared_cnn_2(x2) #3rd Person

		B, C, H, W = x1.shape 
		
		print(f'Org x1 shape {org_img_x1.shape}') # returns 1,3,160,90
		print(f"After CNN: x1 shape {x1.shape} and x2 shape {x2.shape}") # returns 1,32,24,59. 

	#	x1 = self.attention1(x1, x2, x2)

		x1, attention_weights_1 = self.attention1(x1, x2, x2) # Contextual reasoning on 1st person image based on 3rd person image
		
		# Query + output (x1) is 1,32,59,24, attention_weights is 1, 1416, 1416

		# Generate heatmap
		print(f"x1 shape: {x1.shape}")
		print(f"Attention weight 1 shape: {attention_weights_1.shape}") # returns 1, 1416, 1416

		attention_weights_1 = attention_weights_1.unsqueeze(0) # returns 1, 1, 1416, 1416
		print(f"Attention weights 1 before interpolation: {attention_weights_1}")

		# plt.imshow(attention_weights_1.squeeze(0).permute(1,2,0).detach().cpu().numpy(), cmap='inferno')
		# plt.title('Original attention weights 1 (1st person)')
		# plt.show()

		#attention_weights_1 = F.interpolate(attention_weights_1, size = (90,160), mode = 'nearest')  # Turn attentions of shape (1,1,1416,1416) to (1,1,90,160)
		attention_weights_1 = cv2.resize(attention_weights_1.squeeze(0).permute(1,2,0).detach().cpu().numpy(), (160, 90), interpolation=cv2.INTER_AREA)
		print(f"Attention weights 1 shape aft interpolation: {attention_weights_1.shape}")
		print(f"Attention weights 1 aft interpolation: {attention_weights_1}")
		#attention_weights_1 = attention_weights_1.squeeze(0).expand(3, -1, -1) # We want to try to make attentions into size (3, 90, 160)
		print(f"Attention weights 1 shape after: {attention_weights_1.shape}")
		heatmap = attention_weights_1
		#	heatmap = torch.sum(image, dim=1)

		print(f"Shape before showing: {org_img_x2.squeeze(0).permute(1,2,0).detach().cpu().numpy().shape}")

		fig, ax = plt.subplots(1, 2)
		ax[0].imshow(org_img_x1.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()) # Original 1st person image convert from (1, 3, 90, 160) to (90, 160, 3) --> H,W,C
		ax[1].imshow(heatmap, cmap='inferno')
		ax[0].set_title('1st person image')
		ax[1].set_title('Heatmap')
		plt.show()

		x1 = self.norm1(x1)
		x1 = x1.view(B, C, -1).permute(0, 2, 1)
		x1 = self.mlp1(x1).permute(0, 2, 1).contiguous().view(B, C, H, W)

		print(f"x1 shape before passing into attn: {x1.shape}")

		x2, attention_weights_2 = self.attention2(x2, x1, x1) # Contextual reasoning on 3rd person image based on 1st person image
		
		# Query + output (x2) is 1,32,59,24, attention_weights_2 is 1, 1416, 1416

		# Generate heatmap
		print(f"x2 shape: {x2.shape}")
		print(f"Attention weight 2 shape: {attention_weights_2.shape}") # returns 1, 1416, 1416

		attention_weights_2 = attention_weights_2.unsqueeze(0) # returns 1, 1, 1416, 1416
		print(f"Attention weights 2 before interpolation: {attention_weights_2}")

		# plt.imshow(attention_weights_2.squeeze(0).permute(1,2,0).detach().cpu().numpy(), cmap='inferno')
		# plt.title('Original attention weights 2 (3rd person)')
		# plt.show()

		#attention_weights_2 = F.interpolate(attention_weights_2, size = (90,160), mode = 'nearest')  # Turn attentions of shape (1,1,1416,1416) to (1,1,90,160)
		attention_weights_2 = cv2.resize(attention_weights_2.squeeze(0).permute(1,2,0).detach().cpu().numpy(), (160, 90), interpolation=cv2.INTER_AREA)
		print(f"Attention weights 2 shape aft interpolation: {attention_weights_2.shape}")
		print(f"Attention weights 2 aft interpolation: {attention_weights_2}")
		#attention_weights_2 = attention_weights_2.squeeze(0).expand(3, -1, -1) # We want to try to make attentions into size (3, 90, 160)
		print(f"Attention weights 2 shape after: {attention_weights_2.shape}")
		heatmap = attention_weights_2

		fig, ax = plt.subplots(1, 2)
		ax[0].imshow(org_img_x2.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()) # Original 3rd person image convert from (1, 3, 90, 160) to (90, 160, 3) --> H,W,C
		ax[1].imshow(heatmap, cmap='inferno')
		ax[0].set_title('3rd person image')
		ax[1].set_title('Heatmap')
		plt.show()
		
		x2 = self.norm2(x2)
		x2 = x2.view(B, C, -1).permute(0, 2, 1)
		x2 = self.mlp2(x2).permute(0, 2, 1).contiguous().view(B, C, H, W)

		if self.concatenate:
			# Concatenate features along channel dimension
			x = torch.cat((x1, x2), dim=1) 
			print(f"x1 shape bef self.concat: {x1.shape} and x2 shape {x2.shape}")
			print(f"x shape aft self.concat: {x.shape}")
		else:
			x = x1 + x2 

		x = self.integrator(x)
		x = self.head_cnn(x)

		if detach:
			x = x.detach()

		x = self.projection(x)
				
		return x
    
class CustomCombinedExtractorCrossAttentionORG(BaseFeaturesExtractor):
    """
    Custom feature extractor for handling multiple inputs (image + goal info). 
    Observation["observation"] is image data,
    and observation["achieved_goal"] and ["desired_goal"] are joint info.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 500):
        super(CustomCombinedExtractorCrossAttentionORG, self).__init__(observation_space, features_dim = 1)

        extractors = {}
        total_concat_size = 0
        self.output = None

        for key, subspace in observation_space.spaces.items():
            if key == "observation": 
				
                shared_cnn_1 = SharedCNN(obs_shape=(3,90,160))
                shared_cnn_2 = SharedCNN(obs_shape=(3,90,160))
                integrator = Integrator(shared_cnn_1.out_shape, shared_cnn_2.out_shape)
                head = HeadCNN(in_shape=shared_cnn_1.out_shape, flatten=True)
                mlp_hidden_dim = int(shared_cnn_1.out_shape[0] * 4)

                attention_1 = AttentionBlock(dim=shared_cnn_1.out_shape, contextualReasoning=True)
              #  print(f'SHARED CNN OUT SHAPE: {shared_cnn_1.out_shape}')
                mlp1 = Mlp(in_features=shared_cnn_1.out_shape[0], hidden_features=mlp_hidden_dim, act_layer=nn.GELU)
                norm1 = nn.LayerNorm(shared_cnn_1.out_shape)

                attention_2 = AttentionBlock(dim=shared_cnn_1.out_shape, contextualReasoning=True)
                mlp2 = Mlp(in_features=shared_cnn_1.out_shape[0], hidden_features=mlp_hidden_dim, act_layer=nn.GELU)
                norm2 = nn.LayerNorm(shared_cnn_1.out_shape)

                projection = Identity(out_dim=head.out_shape[0])

                self.output = MultiViewCrossAttentionEncoderModifiedORG(
                    shared_cnn_1 = shared_cnn_1,
                    shared_cnn_2 = shared_cnn_2,
                    integrator = integrator,
                    head_cnn = head,
                    projection = projection,
                    attention1 = attention_1,
                    attention2 = attention_2,
                    mlp1 = mlp1,
                    mlp2 = mlp2,
                    norm1 = norm1,
                    norm2 = norm2
                )

                extractors[key] = self.output
                total_concat_size += features_dim
            else:
                extractors[key] = nn.Flatten() # flatten the achieved goal and desired goal
                total_concat_size += 7
				
      #  print(extractors) # disable comment to see architecture here
        
        self.extractors = nn.ModuleDict(extractors)
	
        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            if key == "observation":
                x1 = observations[key][:, :, :90, :] # first half
                x2 = observations[key][:, :, 90:, :] # second half
                encoded_tensor_list.append(extractor(x1, x2))
                print(f"Obs extractor shape: {extractor(x1,x2).shape}")
            else:
                encoded_tensor_list.append(extractor(observations[key]))
                print(f"Other extractor shape: {extractor(observations[key]).shape}")
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return torch.cat(encoded_tensor_list, dim=1)


class CustomCombinedExtractorCrossAttentionTesting(BaseFeaturesExtractor):
    """
    Custom feature extractor for handling multiple inputs (image + goal info). 
    Observation["observation"] is image data,
    and observation["achieved_goal"] and ["desired_goal"] are joint info.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 500):
        super(CustomCombinedExtractorCrossAttention, self).__init__(observation_space, features_dim = 1)

        extractors = {}
        total_concat_size = 0
        self.output = None
		
        for key, subspace in observation_space.spaces.items():
            if key == "observation": 
				
                shared_cnn_1 = SharedCNN(obs_shape=(3,90,160))
                shared_cnn_2 = SharedCNN(obs_shape=(3,90,160))
                integrator = Integrator(shared_cnn_1.out_shape, shared_cnn_2.out_shape)
                head = HeadCNN(in_shape=shared_cnn_1.out_shape, flatten=True)
                mlp_hidden_dim = int(shared_cnn_1.out_shape[0] * 4)

                attention_1 = AttentionBlockWithWeights(dim=shared_cnn_1.out_shape, contextualReasoning=True)
              #  print(f'SHARED CNN OUT SHAPE: {shared_cnn_1.out_shape}')
                mlp1 = Mlp(in_features=shared_cnn_1.out_shape[0], hidden_features=mlp_hidden_dim, act_layer=nn.GELU)
                norm1 = nn.LayerNorm(shared_cnn_1.out_shape)

                attention_2 = AttentionBlockWithWeights(dim=shared_cnn_1.out_shape, contextualReasoning=True)
                mlp2 = Mlp(in_features=shared_cnn_1.out_shape[0], hidden_features=mlp_hidden_dim, act_layer=nn.GELU)
                norm2 = nn.LayerNorm(shared_cnn_1.out_shape)

                projection = Identity(out_dim=head.out_shape[0])

                self.output = MultiViewCrossAttentionEncoderModified(
                    shared_cnn_1 = shared_cnn_1,
                    shared_cnn_2 = shared_cnn_2,
                    integrator = integrator,
                    head_cnn = head,
                    projection = projection,
                    attention1 = attention_1,
                    attention2 = attention_2,
                    mlp1 = mlp1,
                    mlp2 = mlp2,
                    norm1 = norm1,
                    norm2 = norm2
                )

                extractors[key] = self.output
                print(f"FEATURES DIM: {features_dim}")
                total_concat_size += features_dim
            else:
                extractors[key] = nn.Flatten() # flatten the achieved goal and desired goal
                total_concat_size += 7
				
      #  print(extractors) # disable comment to see architecture here
        
        self.extractors = nn.ModuleDict(extractors)
	
        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        print(f"Self extractor items: {self.extractors.items}")
        for key, extractor in self.extractors.items():
            if key == "observation":
                print(f'KEY SHAPE: {observations[key].shape}')
                x1 = observations[key][:, :, :90, :] # first half
                x2 = observations[key][:, :, 90:, :] # second half

                fig, ax = plt.subplots(1,3)
                ax[0].imshow(observations[key].squeeze(0).permute(1, 2, 0).detach().cpu().numpy())
                ax[0].set_title('Observation image in forward pass')
                ax[1].imshow(x1.squeeze(0).permute(1, 2, 0).detach().cpu().numpy())
                ax[1].set_title('Active image in forward pass')
                ax[2].imshow(x2.squeeze(0).permute(1, 2, 0).detach().cpu().numpy())
                ax[2].set_title('Static image in forward pass')
                plt.show()

                encoded_tensor_list.append(extractor(x1, x2))
            else:
                encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return torch.cat(encoded_tensor_list, dim=1)
	
class SharedCNNPhil(nn.Module):
	def __init__(self, obs_shape):
		super().__init__()
		n_input_channels = obs_shape[0]
		self.cnn = nn.Sequential(
			NormalizeImg(mean_zero=False),
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        )
		self.out_shape = _get_out_shape(obs_shape, self.cnn)
		self.apply(orthogonal_init)

	def forward(self, x):
		return self.cnn(x)
	
class IntegratorPhil(nn.Module):
	def __init__(self, obs_shape, in_shape_1, in_shape_2, num_filters=32):
		super().__init__()
	#	print(f"Integrator in shape: {in_shape_1[0]}") # gives 64
		self.integrator = nn.Sequential(
			nn.ReLU(),
            nn.Conv2d(in_shape_1[0]+in_shape_2[0], num_filters, (1,1)),
			nn.ReLU(),
			nn.Conv2d(num_filters, num_filters, 3, stride = 1),
			nn.Flatten()
			)
	
		self.out_shape = _get_out_shape(obs_shape, self.integrator)
		self.apply(orthogonal_init)

	def forward(self, x):
		x = self.integrator(x)
		return x

class LinearLayersPhil(nn.Module):
	def __init__(self, in_shape, features_dim):
		super().__init__()

		# print(f"In_shape for Linear is: {in_shape}") # 2240
		self.linear = nn.Sequential(
			nn.Linear(in_shape[0], 256),
		)
		self.apply(orthogonal_init)

	def forward(self, x):
		x = self.linear(x)
		return x
	
class MultiViewCrossAttentionEncoderPhil(nn.Module):
	"""
	Input is the dual environment obs (active and static images already concatenated in core.py). Applies cross attention.
	"""
	def __init__(self, shared_cnn_1, shared_cnn_2, linear, integrator, attention1=None, attention2=None, mlp1=None, mlp2=None, norm1=None, norm2=None, concatenate=True):
		super().__init__()
		self.shared_cnn_1 = shared_cnn_1
		self.shared_cnn_2 = shared_cnn_2
		self.relu = nn.ReLU()
		self.attention1 = attention1
		self.attention2 = attention2
		self.integrator = integrator
		self.linear = linear

		self.mlp1 = mlp1
		self.norm1 = norm1
		self.mlp2 = mlp2
		self.norm2 = norm2
		self.concatenate = concatenate

	def forward(self, x1, x2, detach=False):
     
		#print(f"x1 shape {x1.shape} and x2 shape {x2.shape}")
     
		# fig, ax = plt.subplots(1,2)
		# ax[0].imshow(x1[0].permute(1,2,0).detach().cpu().numpy())
		# ax[1].imshow(x2[0].permute(1,2,0).detach().cpu().numpy())
		# plt.show()

		x1 = self.shared_cnn_1(x1) #1st Person
		x2 = self.shared_cnn_2(x2) #3rd Person

		B, C, H, W = x1.shape 
		
		x1 = self.attention1(x1, x2, x2) # Contextual reasoning on 3rd person image based on 1st person image
		x1 = self.norm1(x1)
		x1 = x1.view(B, C, -1).permute(0, 2, 1)
		x1 = self.mlp1(x1).permute(0, 2, 1).contiguous().view(B, C, H, W)

		x2 = self.attention2(x2, x1, x1) # Contextual reasoning on 1st person image based on 3rd person image
		x2 = self.norm2(x2)
		x2 = x2.view(B, C, -1).permute(0, 2, 1)
		x2 = self.mlp2(x2).permute(0, 2, 1).contiguous().view(B, C, H, W)

		if self.concatenate:
			# Concatenate features along channel dimension
			x = torch.cat((x1, x2), dim=1) 
		else:
			x = x1 + x2 

	#	print(f"X SHAPE AFT CONCAT: {x.shape}") # gives torch.Size([16, 128, 7, 16])

		x = self.integrator(x)
		x = self.linear(x)

		return x

class CustomCombinedExtractorCrossAttention(BaseFeaturesExtractor):
    """
    Custom feature extractor for handling multiple inputs (image + goal info). 
    Observation["observation"] is image data,
    and observation["achieved_goal"] and ["desired_goal"] are joint info.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 500):
        super(CustomCombinedExtractorCrossAttention, self).__init__(observation_space, features_dim = 1)

        extractors = {}
        total_concat_size = 0
        self.output = None
		
        for key, subspace in observation_space.spaces.items():
            if key == "observation": 
				
                shared_cnn_1 = SharedCNNPhil(obs_shape=(3,90,160))
                shared_cnn_2 = SharedCNNPhil(obs_shape=(3,90,160))
                integrator = IntegratorPhil((128, 7, 16), shared_cnn_1.out_shape, shared_cnn_2.out_shape)
                linear = LinearLayersPhil(integrator.out_shape, features_dim)
                mlp_hidden_dim = int(shared_cnn_1.out_shape[0] * 4)

                attention_1 = AttentionBlock(dim=shared_cnn_1.out_shape, contextualReasoning=True)
              #  print(f'SHARED CNN OUT SHAPE: {shared_cnn_1.out_shape}')
                mlp1 = Mlp(in_features=shared_cnn_1.out_shape[0], hidden_features=mlp_hidden_dim, act_layer=nn.GELU)
                norm1 = nn.LayerNorm(shared_cnn_1.out_shape)

                attention_2 = AttentionBlock(dim=shared_cnn_1.out_shape, contextualReasoning=True)
                mlp2 = Mlp(in_features=shared_cnn_1.out_shape[0], hidden_features=mlp_hidden_dim, act_layer=nn.GELU)
                norm2 = nn.LayerNorm(shared_cnn_1.out_shape)

                self.output = MultiViewCrossAttentionEncoderPhil(
                    shared_cnn_1 = shared_cnn_1,
                    shared_cnn_2 = shared_cnn_2,
                    attention1 = attention_1,
                    attention2 = attention_2,
					integrator=integrator,
                    mlp1 = mlp1,
                    mlp2 = mlp2,
                    norm1 = norm1,
                    norm2 = norm2,
					linear = linear
                )

                extractors[key] = self.output
                total_concat_size += features_dim
            else:
                extractors[key] = nn.Flatten() # flatten the achieved goal and desired goal
                total_concat_size += 7
				
      #  print(extractors) # disable comment to see architecture here
        
        self.extractors = nn.ModuleDict(extractors)
	
        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            if key == "observation":
            #    print(f'KEY SHAPE: {observations[key].shape}')
                x1 = observations[key][:, :, :90, :] # first half
                x2 = observations[key][:, :, 90:, :] # second half

                # fig, ax = plt.subplots(1,3)
                # ax[0].imshow(observations[key].squeeze(0).permute(1, 2, 0).detach().cpu().numpy())
                # ax[0].set_title('Observation image in forward pass')
                # ax[1].imshow(x1.squeeze(0).permute(1, 2, 0).detach().cpu().numpy())
                # ax[1].set_title('Active image in forward pass')
                # ax[2].imshow(x2.squeeze(0).permute(1, 2, 0).detach().cpu().numpy())
                # ax[2].set_title('Static image in forward pass')
                # plt.show()

                encoded_tensor_list.append(extractor(x1, x2))
             #   print(f"Obs extractor shape: {extractor(x1,x2).shape}")
            else:
                encoded_tensor_list.append(extractor(observations[key]))
             #   print(f"Other extractor shape: {extractor(observations[key]).shape}")
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return torch.cat(encoded_tensor_list, dim=1)
	
class SharedCNNNature(nn.Module):
	"""Adapted from Nature CNN: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/torch_layers.py
	"""
	def __init__(self, obs_shape):
		super().__init__()
		n_input_channels = obs_shape[0]
		self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
			nn.ReLU(),
			nn.Flatten(),
			nn.Linear(19456,256),
			nn.ReLU()
		)

	def forward(self, x):
		x = self.cnn(x)
		return x

class MultiViewCrossAttentionEncoderNature(nn.Module):
	"""
	Input is the dual environment obs (active and static images already concatenated in core.py). Applies cross attention.
	"""
	def __init__(self, shared_cnn):
		super().__init__()
		self.shared_cnn = shared_cnn

	def forward(self, x):

		x = self.shared_cnn(x) # 1st and 3rd Person Image concat together

		return x

class CustomCombinedExtractorNature(BaseFeaturesExtractor):
    """
    Custom feature extractor for handling multiple inputs (image + goal info). 
    Observation["observation"] is image data,
    and observation["achieved_goal"] and ["desired_goal"] are joint info.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 500):
        super(CustomCombinedExtractorNature, self).__init__(observation_space, features_dim = 1)

        extractors = {}
        total_concat_size = 0
        self.output = None
		
        for key, subspace in observation_space.spaces.items():
            if key == "observation": 
				
                shared_cnn= SharedCNNNature(obs_shape=(3,180,160))

                self.output = MultiViewCrossAttentionEncoderNature(
                    shared_cnn = shared_cnn
                )

                extractors[key] = self.output
                total_concat_size += features_dim
            else:
                extractors[key] = nn.Flatten() # flatten the achieved goal and desired goal
                total_concat_size += 7
        
        self.extractors = nn.ModuleDict(extractors)
	
        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return torch.cat(encoded_tensor_list, dim=1)





# class MultiViewEncoder(nn.Module):
# 	def __init__(self, shared_cnn_1, shared_cnn_2, integrator, head_cnn, projection, attention1=None, attention2=None, mlp1=None, mlp2=None, norm1=None, norm2=None, concatenate=True, contextualReasoning1=False, contextualReasoning2=False):
# 		super().__init__()
# 		self.shared_cnn_1 = shared_cnn_1
# 		self.shared_cnn_2 = shared_cnn_2
# 		self.integrator = integrator
# 		self.head_cnn = head_cnn
# 		self.projection = projection
# 		self.relu = nn.ReLU()
# 		self.contextualReasoning1 = contextualReasoning1
# 		self.contextualReasoning2 = contextualReasoning2
# 		self.attention1 = attention1
# 		self.attention2 = attention2

# 		self.mlp1 = mlp1
# 		self.norm1 = norm1
# 		self.mlp2 = mlp2
# 		self.norm2 = norm2

# 		self.out_dim = projection.out_dim
# 		self.concatenate = concatenate

# 	def forward(self, x1, x2, detach=False):
		
# 		x1 = self.shared_cnn_1(x1) #3rd Person
# 		x2 = self.shared_cnn_2(x2)

# 		B, C, H, W = x1.shape

# 		if self.contextualReasoning1:
# 			x1 = self.attention1(x1, x2, x2) # Contextual reasoning on 3rd person image based on 1st person image
# 			x1 = self.norm1(x1)
# 			x1 = x1.view(B, C, -1).permute(0, 2, 1)
# 			x1 = self.mlp1(x1).permute(0, 2, 1).contiguous().view(B, C, H, W)

# 		if self.contextualReasoning2:
# 			x2 = self.attention2(x2, x1, x1) # Contextual reasoning on 1st person image based on 3rd person image
# 			x2 = self.norm2(x2)
# 			x2 = x2.view(B, C, -1).permute(0, 2, 1)
# 			x2 = self.mlp2(x2).permute(0, 2, 1).contiguous().view(B, C, H, W)

# 		if self.concatenate:
# 			# Concatenate features along channel dimension
# 			x = torch.cat((x1, x2), dim=1) # 1, 64, 21, 21
# 		else:
# 			x = x1 + x2 # 1, 32, 21, 21

# 		x = self.integrator(x)
# 		x = self.head_cnn(x)

		
# 		if self.attention1 is not None and not self.contextualReasoning1:
# 			x = self.relu(self.attention1(x, x, x))
		
# 		if detach:
# 			x = x.detach()

# 		x = self.projection(x)
		
# 		return x

# class Actor(nn.Module):
# 	def __init__(self, out_dim, projection_dim, state_shape, action_shape, hidden_dim, hidden_dim_state, log_std_min, log_std_max):
# 		super().__init__()
# 		self.log_std_min = log_std_min
# 		self.log_std_max = log_std_max

# 		self.trunk = nn.Sequential(nn.Linear(out_dim, projection_dim),
# 								nn.LayerNorm(projection_dim), nn.Tanh())

# 		self.layers = nn.Sequential(
# 			nn.Linear(projection_dim, hidden_dim), nn.ReLU(inplace=True),
# 			nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
# 			nn.Linear(hidden_dim, 2 * action_shape[0])
# 		)

# 		if state_shape:
# 			self.state_encoder = nn.Sequential(nn.Linear(state_shape[0], hidden_dim_state),
# 											nn.ReLU(inplace=True),
# 											nn.Linear(hidden_dim_state, projection_dim),
# 											nn.LayerNorm(projection_dim), nn.Tanh())
# 		else:
# 			self.state_encoder = None
# 		self.apply(orthogonal_init)

# 	def forward(self, x, state, compute_pi=True, compute_log_pi=True):
# 		x = self.trunk(x)

# 		if self.state_encoder:
# 			x = x + self.state_encoder(state)

# 		mu, log_std = self.layers(x).chunk(2, dim=-1)
# 		log_std = torch.tanh(log_std)
# 		log_std = self.log_std_min + 0.5 * (
# 			self.log_std_max - self.log_std_min
# 		) * (log_std + 1)

# 		if compute_pi:
# 			std = log_std.exp()
# 			noise = torch.randn_like(mu)
# 			pi = mu + noise * std
# 		else:
# 			pi = None
# 			entropy = None

# 		if compute_log_pi:
# 			log_pi = gaussian_logprob(noise, log_std)
# 		else:
# 			log_pi = None

# 		mu, pi, log_pi = squash(mu, pi, log_pi)

# 		return mu, pi, log_pi, log_std


# class Critic(nn.Module):
# 	def __init__(self, out_dim, projection_dim, state_shape, action_shape, hidden_dim, hidden_dim_state):
# 		super().__init__()
# 		self.projection = nn.Sequential(nn.Linear(out_dim, projection_dim),
# 								nn.LayerNorm(projection_dim), nn.Tanh())

# 		if state_shape:
# 			self.state_encoder = nn.Sequential(nn.Linear(state_shape[0], hidden_dim_state),
# 											nn.ReLU(inplace=True),
# 											nn.Linear(hidden_dim_state, projection_dim),
# 											nn.LayerNorm(projection_dim), nn.Tanh())
# 		else:
# 			self.state_encoder = None
		
# 		self.Q1 = nn.Sequential(
# 			nn.Linear(projection_dim + action_shape[0], hidden_dim),
# 			nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
# 			nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))
# 		self.Q2 = nn.Sequential(
# 			nn.Linear(projection_dim + action_shape[0], hidden_dim),
# 			nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
# 			nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))
# 		self.apply(orthogonal_init)

# 	def forward(self, obs, state, action):
# 		obs = self.projection(obs)

# 		if self.state_encoder:
# 			obs = obs + self.state_encoder(state)

# 		h = torch.cat([obs, action], dim=-1)
# 		return self.Q1(h), self.Q2(h)
	
# class MultiViewEncoderModified(nn.Module):
# 	"""
# 	Input is the dual environment obs (active and static images already concatenated in core.py). Applies self attention.
# 	"""
# 	def __init__(self, shared_cnn, head_cnn, projection, attention):
# 		super().__init__()
# 		self.shared_cnn = shared_cnn
# 		self.head_cnn = head_cnn
# 		self.projection = projection
# 		self.relu = nn.ReLU()
# 		self.attention = attention

# 		self.out_dim = projection.out_dim

# 	def forward(self, x, detach=False):
		
# 		x = self.shared_cnn(x) # Active cam

# 		B, C, H, W = x.shape

# 		x = self.head_cnn(x)

# 		x = self.relu(self.attention(x, x, x))
		
# 		if detach:
# 			x = x.detach()

# 		x = self.projection(x)
		
# 		return x
	

# class CustomFeatureExtractor(BaseFeaturesExtractor):
#     """
#     :param observation_space: (gym.Space)
#     :param features_dim: (int) Number of features extracted.
#         This corresponds to the number of unit for the last layer.
#     """

#     def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
#         super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)
		
#         shared_cnn = SharedCNN(obs_shape=observation_space.shape)
#         head = HeadCNN(in_shape=shared_cnn.out_shape, flatten=False)
#         attention_block = AttentionBlock(dim=head.out_shape, contextualReasoning=False)
#         projection = Identity(out_dim=head.out_shape[0])
		
#         self.output = MultiViewEncoderModified(
#             shared_cnn = shared_cnn,
#             head_cnn = head,
#             projection = projection,
#             attention= attention_block
#         )
		

#     def forward(self, observations: torch.Tensor) -> torch.Tensor:
#         return self.output(observations)

# class CustomCombinedExtractor(BaseFeaturesExtractor):
#     """
#     Custom feature extractor for handling multiple inputs (image + goal info). 
#     Observation["observation"] is image data,
#     and observation["achieved_goal"] and ["desired_goal"] are joint info.
#     """

#     def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 500):
#         super(CustomCombinedExtractor, self).__init__(observation_space, features_dim = 1)

#         extractors = {}
#         total_concat_size = 0
#         self.output = None

#         for key, subspace in observation_space.spaces.items():
#             if key == "observation": 
				
#                 shared_cnn = SharedCNN(obs_shape=observation_space.spaces["observation"].shape)
#                 head = HeadCNN(in_shape=shared_cnn.out_shape, flatten=False)
#                 attention_block = AttentionBlock(dim=head.out_shape, contextualReasoning=False)
#                 projection = Identity(out_dim=head.out_shape[0])
                
#                 self.output = MultiViewEncoderModified(
#                     shared_cnn = shared_cnn,
#                     head_cnn = head,
#                     projection = projection,
#                     attention= attention_block
#                 )
				
#                 extractors[key] = self.output
#                 total_concat_size += features_dim
#             else:
#                 extractors[key] = nn.Flatten() # flatten the achieved goal and desired goal
#                 total_concat_size += 7
				
#       #  print(extractors) # disable comment to see architecture here
        
#         self.extractors = nn.ModuleDict(extractors)
	
#         # Update the features dim manually
#         self._features_dim = total_concat_size

#     def forward(self, observations) -> torch.Tensor:
#         encoded_tensor_list = []

#         # self.extractors contain nn.Modules that do all the processing.
#         for key, extractor in self.extractors.items():
#             encoded_tensor_list.append(extractor(observations[key]))
#         # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
#         return torch.cat(encoded_tensor_list, dim=1)