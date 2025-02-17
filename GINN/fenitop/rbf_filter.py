import torch
import torch.nn.functional as F

class RBFKernelFilter:
    def __init__(self, kernel_size, sigma, dim):
        """
        Initialize the RBFKernelFilter class.

        Parameters:
        kernel_size (int): The size of the kernel.
        sigma (float): The standard deviation of the Gaussian.
        dim (int): The dimension of the kernel (2 for 2D, 3 for 3D).
        """
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.dim = dim
        self.kernel = self._create_gaussian_kernel()

    def _create_gaussian_kernel(self):
        """
        Generate a Gaussian kernel.

        Returns:
        torch.Tensor: The Gaussian kernel.
        """
        if self.dim == 2:
            coords = torch.arange(self.kernel_size) - self.kernel_size // 2
            x, y = torch.meshgrid(coords, coords)
            kernel = torch.exp(-(x**2 + y**2) / (2 * self.sigma**2))
        elif self.dim == 3:
            coords = torch.arange(self.kernel_size) - self.kernel_size // 2
            x, y, z = torch.meshgrid(coords, coords, coords)
            kernel = torch.exp(-(x**2 + y**2 + z**2) / (2 * self.sigma**2))
        else:
            raise ValueError("Dimension must be 2 or 3.")

        # Normalize the kernel
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)

    def __call__(self, rho):
        """
        Apply the RBF kernel filter to a 2D or 3D image.
        rho: [b, h, w] or [b, d, h, w] tensor

        """
        if self.dim == 2:
            if rho.dim() == 2:
                # insert batch and channel dimensions
                rho = rho.unsqueeze(0).unsqueeze(0)
            elif rho.dim() == 3:
                # insert channel dimension
                rho = rho.unsqueeze(1)
            else:
                raise ValueError("Input tensor must have 2 or 3 dimensions.")
            
            #filtered_image = F.conv2d(rho, self.kernel, padding=self.kernel_size//2)
            ones_img = torch.ones_like(rho)
            normalization = F.conv2d(ones_img, self.kernel, padding='same')
            filtered_image = F.conv2d(rho, self.kernel, padding='same')
            filtered_image = torch.div(filtered_image, normalization)
        elif self.dim == 3:
            if rho.dim() == 3:
                # insert batch and channel dimensions
                rho = rho.unsqueeze(0).unsqueeze(0)
            elif rho.dim() == 4:
                # insert channel dimension
                rho = rho.unsqueeze(1)
            else:
                raise ValueError("Input tensor must have 3 or 4 dimensions.")
            
            #filtered_image = F.conv3d(rho, self.kernel, padding=self.kernel_size//2)
            filtered_image = F.conv3d(rho, self.kernel, padding='same')
        else:
            raise ValueError("Dimension must be 2 or 3.")

        return filtered_image.squeeze(1)
