model.load_state_dict(torch.load("asl_mobilenetv2_best.pth", map_location=torch.device('cpu')))
model.eval()  
