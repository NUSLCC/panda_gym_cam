"""
From https://github.com/YanjieZe/rl3d/tree/901da1d743fef33a59cbc19526751318d53e2565
"""

from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from rl3d import load_3d
encoder_2d = load_3d.visual_representation(ckpt_path="rl3d/videoae_co3d.tar", use_3d=False)

img_path = 'test_image.png'
org_img = Image.open(img_path).convert('RGB')
trans = transforms.Compose([transforms.ToTensor()])
img = trans(org_img)
encoded_img = encoder_2d(img.unsqueeze(0))

plt.subplot(1, 4, 1)
plt.imshow(org_img)
plt.title('Original image')
plt.subplot(1, 4, 2)
plt.imshow(encoded_img[0, 0].cpu().detach().numpy(), cmap='viridis')  # Visualize the first channel of the first (and only) image
plt.title(f'Channel 0 of feature map')
plt.subplot(1, 4, 3)
plt.imshow(encoded_img[0, 1].cpu().detach().numpy(), cmap='viridis')  
plt.title(f'Channel 1 of feature map')
plt.subplot(1, 4, 4)
plt.imshow(encoded_img[0, 2].cpu().detach().numpy(), cmap='viridis')  
plt.title(f'Channel 2 of feature map')
plt.tight_layout()
plt.show()