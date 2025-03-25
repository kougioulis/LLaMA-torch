import torch

def get_device():
    """
    Returns the device available to torch.
    
    Args:
    ---
       None
    
    Returns:
    ---
       - `cuda` if CUDA is currently available, else `cpu`.
    """

    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def count_parameters(model):
    """
    Returns the number of trainable parameters in the model.
    
    Args:
    ---
        model (torch.nn.Module): The model to count the parameters for.
        
    Returns:
    ---
        (int): The number of trainable parameters in the model.
    """

    return sum(p.numel() for p in model.parameters() if p.requires_grad)