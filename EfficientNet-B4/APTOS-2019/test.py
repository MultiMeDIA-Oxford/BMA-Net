from timm import create_model
m = create_model('efficientnet_b4', pretrained=False)
print(m)
