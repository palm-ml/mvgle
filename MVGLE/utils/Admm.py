import numpy as np
import torch

def admm_lasso(A, b, W, rho, lambda_, iter, gama1, gama2):
    # S = @(tau, g) max(0, g - tau) + min(0, g + tau);
    def S(tau, g):
        max_vec = g - tau
        max_vec[max_vec < 0] = 0
        min_vec = g + tau
        min_vec[min_vec > 0] = 0
        return max_vec + min_vec

    # [m,n] = size(A);
    # [~,c] = size(W);
    m, n = A.shape
    _, c = W.shape

    # I = eye(n);
    # x = zeros(n,c);
    I = torch.eye(n).cuda()
    x = torch.zeros((n, c)).cuda()

    # z_old = zeros(n,c);
    # u_old = zeros(n,c);
    # x_old=x;
    z_old = torch.zeros((n, c)).cuda()
    u_old = torch.zeros((n, c)).cuda()
    x_old = x.clone().detach()

    # e_his= gama2*norm(A*(W+x)-b,2)^2+lambda*norm(x,1);
    e_his = gama2 * torch.pow(torch.norm(A @ (W + x) - b), 2) + lambda_ * torch.norm(x, 1)

    for ii in range(0, iter):

        # % record x_s
        # x_s =x;
        # % minimize x,z,u
        # x = (gama2*A'*A+rho*I) \ (gama2*A'*b-gama2*A'*A*W+rho*z_old-u_old);
        x_s = x.clone().detach()
        # print((A.T @ A).shape)
        # print(I.shape)
        # print((A.T @ b).shape)
        # print((A.T @ A @ W).shape)
        # print(z_old.shape)
        x = torch.inverse( gama2 * (A.T @ A) + rho * I) @ (gama2 * (A.T @ b) - gama2 * (A.T @ A @ W) + rho * z_old - u_old)

        # z_new = S(lambda/rho, x+u_old/rho);
        # u_new = u_old + rho*(x-z_new);
        z_new = S(lambda_ / rho, x + u_old/rho)
        u_new = u_old + rho * (x - z_new)

        # % check stop criteria
        # e = gama2*norm(A*(W+x)-b,2)^2+lambda*norm(x,1);
        # if abs(e_his-e)<0.001
        #     break
        # end
        e = gama2 * torch.pow(torch.norm(A@(W+x)-b, 2), 2) + lambda_ * torch.norm(x, 1)
        if torch.abs(e_his - e) < 0.001:
            break

        # % update
        # e_his=e;
        # z_old = z_new;
        # u_old = u_new;
        e_his = e.clone().detach()
        z_old = z_new.clone().detach()
        u_old = u_new.clone().detach()
    
    return x_s