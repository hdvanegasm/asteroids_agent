import torchvision.transforms as transforms
import constants

def transform_image(observation):
    return transforms.Compose([
            transforms.ToPILImage(mode="RGB"),
            transforms.Resize((110, 84)),
            transforms.CenterCrop(84),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.4161,],[0.1688,]),
        ])(observation)
    
def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1

def process_state(cumulative_screenshot):
    if len(cumulative_screenshot) < constants.STATE_LENGTH:
        n_padding = constants.STATE_LENGTH
        