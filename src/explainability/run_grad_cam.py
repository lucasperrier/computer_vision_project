import torch
from src.explainability.grad_cam import GradCAM, upsample_cam_to_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) ResNet50
lit_resnet = ResNet50Module(config).to(device).eval()
x, y = next(iter(datamodule.val_dataloader()))
x = x.to(device)

cam_expl = GradCAM(lit_resnet.model, target_layer=lit_resnet.model.layer4[-1])
out = cam_expl(x)  # explains predicted class
cam224 = upsample_cam_to_image(out.cam, (224,224))
cam_expl.close()

# 2) ViT
lit_vit = VisionTransformerModule(config).to(device).eval()
cam_expl = GradCAM(lit_vit.model, target_layer=lit_vit.model.blocks[-1].norm1)
out = cam_expl(x, vit_grid=(14,14))
cam224 = upsample_cam_to_image(out.cam, (224,224))
cam_expl.close()