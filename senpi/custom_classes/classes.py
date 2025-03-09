import torch


# this is a global variable that tunes how fast the sigmoid operation transitions + gradient strength (tune)
k = 1

## Make a custom pytorch class that implements a differentiable comparison operation that exhibits a hard forward path and soft backwards path
## Replaces torch.where functionality
class CustomComparison(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, comparison, min_val, max_val):
        ctx.save_for_backward(x, y)
        ctx.comparison = comparison

        # Perform the comparison operation
        if comparison == '==':
            condition = (x == y)
        elif comparison == '!=':
            condition = (x != y)
        elif comparison == '<':
            condition = (x < y)
        elif comparison == '<=':
            condition = (x <= y)
        elif comparison == '>':
            condition = (x > y)
        elif comparison == '>=':
            condition = (x >= y)
        else:
            raise ValueError(f"Invalid comparison operator: {comparison}")

        # Apply torch.where to get the result
        result = torch.where(condition, max_val, min_val)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        condition = None
        weight = None
        # Determine the condition based on the stored comparison
        if ctx.comparison == '==':
            condition = (x == y)
            # For an equals operation, the soft approximation would be: sigmoid(-k*|x-y|); scaled: 2 * sigmoid(-k*|x-y|)
            g = -k * torch.abs(x - y)
            sig_g = torch.sigmoid(g)
            weight = 2 * sig_g * (1 - sig_g) * (-k * (x - y) / torch.abs(x - y))
        elif ctx.comparison == '!=':
            condition = (x != y)
            # For a ! equals operation, the soft approximation would be: 1-sigmoid(-k*|x-y|); scaled: 2 * sigmoid(k*|x-y|) - 1
            g = k * torch.abs(x - y)
            sig_g = torch.sigmoid(g)
            weight = 2 * sig_g * (1 - sig_g) * (k * (x - y) / torch.abs(x - y))
        elif ctx.comparison == '<':
            # For a < operation, the soft approximation would be: sigmoid(-k*(x-y))
            condition = (x < y)
            g = -k * (x - y)
            sig_g = torch.sigmoid(g)
            weight = sig_g * (1 - sig_g) * (-k)
        elif ctx.comparison == '<=':
            # For a <= operation, the soft approximation would be: sigmoid(-k*(x-y))
            condition = (x <= y)
            g = -k * (x - y)
            sig_g = torch.sigmoid(g)
            weight = sig_g * (1 - sig_g) * (-k)
        elif ctx.comparison == '>':
            # For a > operation, the soft approximation would be: sigmoid(k*(x-y))
            condition = (x > y)
            g = k * (x - y)
            sig_g = torch.sigmoid(g)
            weight = sig_g * (1 - sig_g) * k
        elif ctx.comparison == '>=':
            # For a >= operation, the soft approximation would be: sigmoid(k*(x-y))
            condition = (x >= y)
            g = k * (x - y)
            sig_g = torch.sigmoid(g)
            weight = sig_g * (1 - sig_g) * k
        # Initialize gradients
        grad_x = grad_y = None
        
        if condition is not None:
            # Create gradient mask
            # mask = condition.float()
            grad_x = weight * grad_output  # Gradient for x
            grad_y = -weight * grad_output  # Gradient for y, negative because of the comparison

        return grad_x, grad_y, None, None, None  # Other inputs do not have gradients
    


# develop custom poisson class with REINFORCE inspired backpropagation step (see supplement)
class PoissonReinforce(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        # perform poisson noise
        output = torch.clamp(torch.poisson(input), min=0)  # clamp physically untrue returns
        # save input and output to calculate PMF later
        ctx.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input. Gradient_output is the gradient at the output of the
        forward step (i.e. input gradient in backwards graph)
        """
        input, output = ctx.saved_tensors
        # REINFORCE weights the downstream gradient by the derivative of the log PMF (has a closed form): https://pytorch.org/docs/stable/generated/torch.nn.PoissonNLLLoss.html#torch.nn.PoissonNLLLoss
        # final factor is an approximate factorial found on: https://discuss.pytorch.org/t/factorials-that-will-work-with-cuda/131716/2
        # log_PMF = -input + torch.mul(output, torch.log(input)) - torch.log(torch.lgamma(output+1).exp()) # for user notes

        # The below gradient calculation occassionally creates incorrect responses - correct
        grad_output = grad_output * -1*(-1 + output / input)  # closed form solution for result
        grad_output = torch.nan_to_num(grad_output, nan=0.0, posinf=1.0, neginf=-1.0)  # filter unrealistic results that might occur at low SNR
        return grad_output
    

# Develop custom log class to handle invalid inputs (i.e. in physical systems no signal (0) leads to no current)
class CustomLog(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # If the input is less than 1, return 0
        output = torch.where(input < 1, torch.tensor(0.0, dtype=input.dtype, device=input.device), input.log())
        
        # Save input for the backward pass
        ctx.save_for_backward(input)
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve the input from the forward pass
        input, = ctx.saved_tensors
        
        # If the input is less than 1, the gradient should be zero
        grad_input = torch.where(input < 1, torch.tensor(0.0, dtype=input.dtype, device=input.device), grad_output / input)
        
        return grad_input

if __name__ == "__main__":
    #Test if custom log works as expected
    x = torch.tensor([0, 0.5, 1.0, 2.0, 3.0], requires_grad=True)
    y = CustomLog.apply(x)

    # gradient is 1/x for all values >=1
    y.backward(torch.ones_like(y))

    # print output
    print(f"x: {x}")
    print(f"y: {y}")
    print(f"Gradients: {x.grad}")

    

