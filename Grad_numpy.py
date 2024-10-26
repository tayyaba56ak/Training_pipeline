
#1 Design model (input, output size, forward pass)
#2 Construst loss and optimizer 
#3 Training loop 
#4 Forward pass: compute prediction 
#5 Backward pass: gradients  
# update weights 

import torch 
import torch.nn as nn 

x = torch.tensor([1 ,2, 3, 4], dtype = torch.float32)
y = torch.tensor([2,4,6,8], dtype = torch.float32)

w =torch.tensor(0.0, dtype = torch.float32 , requires_grad = True)

def forward(x):
    return  w * x 
print(f"Predicted before training: f(5) = {forward(5):.3f}")
learning_rate = 0.01 
n_iters = 100  

loss = nn.MSELoss()
optimizer = torch.optim.SGD([w], lr =learning_rate)


for epoch in range(n_iters):
    y_pred = forward(x)
    l = loss(y, y_pred) 

    l.backward() 
    optimizer.step()
    optimizer.zero() 

    if epoch % 10 == 0:
        print(f"epoch{epoch+1}:w = {w:.3f}, loss = {l:8f}")

print(f"prediction after training: f(5) = {forward(5):.3f}")        


   
   
      




