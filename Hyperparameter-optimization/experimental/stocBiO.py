import torch
from torch.autograd import grad 
from torch.nn import functional as F


def stocbio(params, hparams, val_data_list, args, out_f, reg_f):
        data_list, labels_list = val_data_list
        # Fy_gradient
        output = out_f(data_list[0], params)
        Fy_gradient = gradient_fy(args, labels_list[0], params, data_list[0], output)
        v_0 = torch.unsqueeze(torch.reshape(Fy_gradient, [-1]), 1).detach()

        # Hessian
        z_list = []
        output = out_f(data_list[1], params)
        Gy_gradient = gradient_gy(args, labels_list[1], params, data_list[1], hparams, output, reg_f) 

        G_gradient = torch.reshape(params[0], [-1]) - args.eta*torch.reshape(Gy_gradient, [-1])
        
        for _ in range(args.hessian_q):
            Jacobian = torch.matmul(G_gradient, v_0)
            v_new = torch.autograd.grad(Jacobian, params, retain_graph=True)[0]
            v_0 = torch.unsqueeze(torch.reshape(v_new, [-1]), 1).detach()
            z_list.append(v_0)            
        v_Q = args.eta*v_0+torch.sum(torch.stack(z_list), dim=0)

        # Gyx_gradient
        output = out_f(data_list[2], params)
        Gy_gradient = gradient_gy(args, labels_list[2], params, data_list[2], hparams, output, reg_f)
        Gy_gradient = torch.reshape(Gy_gradient, [-1])
        Gyx_gradient = torch.autograd.grad(torch.matmul(Gy_gradient, v_Q.detach()), hparams, retain_graph=True)[0]
        outer_update = -Gyx_gradient 

        return outer_update

def gradient_fy(args, labels, params, data, output):
    loss = F.cross_entropy(output, labels)
    grad = torch.autograd.grad(loss, params)[0]
    return grad

def gradient_gy(args, labels_cp, params, data, hparams, output, reg_f):
    # For MNIST data-hyper cleaning experiments
    loss = F.cross_entropy(output, labels_cp, reduction='none')
    # For NewsGroup l2reg expriments
    # loss = F.cross_entropy(output, labels_cp)
    loss_regu = reg_f(params, hparams, loss)
    grad = torch.autograd.grad(loss_regu, params, create_graph=True)[0]
    return grad
