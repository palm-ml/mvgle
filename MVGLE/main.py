import numpy as np
import torch
from utils.metrics import RankingLoss, Coverage, AveragePrecision, OneError, HammingLoss
from utils.Admm import admm_lasso
from scipy.io import loadmat
import argparse
import os
import random

def set_seed(seed):
    torch.manual_seed(seed) # ΪCPU�����������
    torch.cuda.manual_seed(seed) # Ϊ��ǰGPU�����������
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU��Ϊ����GPU�����������
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.	
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# main setting
parser = argparse.ArgumentParser(
    prog='MVGLE demo file.',
    usage='Demo with Multi-View Partial Multi-Label Learning.',
    epilog='end',
    add_help=True
)
# hyper-param
parser.add_argument('--dataset', type=str, default='emotions_new.mat')
parser.add_argument('--p', type=int, default=3)
parser.add_argument('--r', type=int, default=1)
parser.add_argument('--lambda_', type=float, default=0.01)
parser.add_argument('--gamma1', type=float, default=5)
parser.add_argument('--gamma2', type=float, default=17.79)
parser.add_argument('--lr_lbfgs', type=float, default=1e-5)
parser.add_argument('--lr_gd', type=float, default=1e-3)
parser.add_argument('--maxiter', type=int, default=300)
parser.add_argument('--maxiter_bfgs', type=int, default=300)
parser.add_argument('--folds', type=int, default=10)
# gpu
parser.add_argument('--exp', type=str, default="main_A1_1_1")
parser.add_argument('--gpu', type=str, default="0")
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()
# set gpu idx
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
# set random seed
set_seed(args.seed)
print(args)

def main():
    data_path = "datasets/"
    dataset = args.dataset
    p = args.p
    r = args.r

    lambda_ = args.lambda_
    gamma1 = args.gamma1
    gamma2 = args.gamma2
    maxiter = args.maxiter
    maxiter_bfgs = args.maxiter_bfgs
    folds = args.folds

    # Take the corresponding data from .mat
    mat_data_dict = loadmat(os.path.join(data_path, dataset))
    data = mat_data_dict["data"]
    print("data shape: ", data.shape)
    # print(data)
    target = mat_data_dict["target"]
    print("target shape: ", target.shape)
    # print(target)
    weak_target = mat_data_dict["p{}r{}_noise_target".format(str(p), str(r))]
    print("weak target shape: ", weak_target.shape)
    # print(weak_target)
    idx  = mat_data_dict["idx"]
    print("idx shape: ",  idx.shape)
    # print(idx)
    v_num = data.shape[0]
    print("view number: ", v_num)

    # normalization
    for i in range(0, v_num):
        view_data = data[i, 0]
        n_sample, n_fea = data[i, 0].shape
        print("For View {}, n_sample={} and n_fea={}. ".format(i, n_sample, n_fea))
        # min-max normalization
        # data[i, 0] = (view_data - np.min(view_data, axis=0, keepdims=True)) / (np.max(view_data, axis=0, keepdims=True) - np.min(view_data, axis=0, keepdims=True))
        # mean-std normalization
        # data[i, 0] = (view_data - np.mean(view_data, axis=0, keepdims=True)) / (np.std(view_data, axis=0, keepdims=True))

    # ten-fold
    save_list = []
    for fold in range(0, folds):
        print("Current Fold: {}".format(str(fold)))
        test_idx = idx[fold, 0][0] - 1
        train_idx = np.ones(n_sample, np.bool_)
        train_idx[test_idx] = False
        
        print("Test idx: ", test_idx)
        print("Test num: ", len(test_idx))
        test_target = target[test_idx, :]
        train_target = target[train_idx, :]

        print("Train num: ", len(train_target))

        # build train and test multi-view features
        test_data =  [ data[i, 0][test_idx, :]  for i in range(0, v_num)]
        train_data = [ data[i, 0][train_idx, :] for i in range(0, v_num)]

        # normalize
        mean_std_array = [ (np.mean(train_data[i], axis=0, keepdims=True), np.std(train_data[i], axis=0, keepdims=True)) for i in range(0, v_num)]
        train_data = [ (train_data[i] - mean_std_array[i][0]) / mean_std_array[i][1] for i in range(0, v_num) ]
        test_data =  [ (test_data[i]  - mean_std_array[i][0]) / mean_std_array[i][1] for i in range(0, v_num) ]
        
        train_data = [ np.concatenate((train_data[i], np.ones((len(train_data[i]), 1))), axis=1) for i in range(0, v_num) ]
        test_data =  [ np.concatenate((test_data[i],  np.ones((len(test_data[i]),  1))), axis=1) for i in range(0, v_num) ]

        # noisy label matrix L of train dataset
        train_weak_target = weak_target[train_idx, :]

        # core algorithm
        W, S = MVGLE(train_weak_target, train_data, lambda_, gamma1, gamma2, maxiter, maxiter_bfgs)


        test_datall = np.concatenate(test_data, axis=1)
        test_datall = torch.FloatTensor(test_datall)
        print("Test datall shape", test_datall.shape)
        Y_test = test_datall @ W
        # calculate indexes
        test_target = torch.FloatTensor(test_target)
        Y_test = torch.FloatTensor(Y_test)
        RK = RankingLoss(Y_test, test_target)
        CV = Coverage(Y_test, test_target)
        AP = AveragePrecision(Y_test, test_target)
        OE = OneError(Y_test, test_target)
        Y_bin = perform_binarization(Y_test, thres=0.44)
        HM = HammingLoss(Y_bin, test_target)
        
        print("Ranking Loss: ", RK)
        print("Coverage: ", CV)
        print("AveragePrecision: ", AP)
        print("OneError: ", OE)
        fold_save_dict = {
            "W": W,
            "Y_test": Y_test,
            "measures": {
                "HammingLoss": HM,
                "RankingLoss": RK,
                "Coverage": CV,
                "AveragePrecision": AP,
                "OneError": OE
            }
        }
        save_list.append(fold_save_dict)
        print("Current Fold: {}".format(str(fold)))
        print(fold_save_dict["measures"])
        # exit()
    
    RK_list = torch.FloatTensor([ save_list[i]["measures"]["RankingLoss"] for i in range(0, folds)])
    RK_mean, RK_std = torch.mean(RK_list), torch.std(RK_list)
    CV_list = torch.FloatTensor([ save_list[i]["measures"]["Coverage"] for i in range(0, folds)])
    CV_mean, CV_std = torch.mean(CV_list), torch.std(CV_list)
    AP_list = torch.FloatTensor([ save_list[i]["measures"]["AveragePrecision"] for i in range(0, folds)])
    AP_mean, AP_std = torch.mean(AP_list), torch.std(AP_list)
    OE_list = torch.FloatTensor([ save_list[i]["measures"]["OneError"] for i in range(0, folds)])
    OE_mean, OE_std = torch.mean(OE_list), torch.std(OE_list)
    HM_list = torch.FloatTensor([ save_list[i]["measures"]["HammingLoss"] for i in range(0, folds)])
    HM_mean, HM_std = torch.mean(HM_list), torch.std(HM_list)
    print(HM_list)
    # save
    save_dict = {
        "experiment_file": args.exp, 
        "dataset": args.dataset,
        "args": str(args),
        "ten_fold": save_list,
        "total": {
            "HammingLoss": (HM_mean, HM_std),
            "RankingLoss": (RK_mean, RK_std),
            "Coverage": (CV_mean, CV_std),
            "AveragePrecision": (AP_mean, AP_std),
            "OneError": (OE_mean, OE_std)
        }
    }
    print(args)
    print(save_dict["total"])
    save_path = "results/exp={}/ds={}/r={}_p={}/".format(
        save_dict["experiment_file"],
        args.dataset,
        args.r,
        args.p,
    )
    save_name = str(args) + ".pt"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(save_dict, save_path + save_name)

def perform_binarization(Y_test, thres=0.5):
    Y_bin = (Y_test - torch.min(Y_test, dim=1, keepdim=True)[0]) / (torch.max(Y_test, dim=1, keepdim=True)[0] - torch.min(Y_test, dim=1, keepdim=True)[0])
    Y_bin[Y_bin > thres] = 1
    Y_bin[Y_bin < thres] = 0
    return Y_bin

def binarization(d, t):
    pass

def MVGLE(train_weak_target, train_data, lambda_, gamma1, gamma2, maxiter, maxiter_bfgs):
    v_num = len(train_data)

    # full view
    train_datall = torch.FloatTensor(np.concatenate(train_data, axis=1)).cuda()
    train_weak_target = torch.FloatTensor(train_weak_target).cuda()
    # numbers
    n_num, f_num = train_datall.shape
    print("Train_datall shape: ", (n_num, f_num))
    c_num = train_weak_target.shape[-1]
    # initialize W,S,U
    W = torch.zeros((f_num, c_num)).cuda()
    W = torch.nn.Parameter(W.clone().detach())
    torch.nn.init.xavier_normal_(W)
    S = torch.zeros((f_num, c_num)).cuda()
    U = torch.zeros((n_num, n_num)).cuda()

    # initialize A_v of each view
    train_data = [ torch.FloatTensor(train_data[i]).cuda() for i in range(0, v_num)]
    A = [ initialize_A(train_data[i]) for i in range(0, v_num)]
    t = [ 1 / torch.pow(torch.norm(A[i] - U, 2), 2) for i in range(0, v_num)]
    
    tao = [ t[i] / sum(t) for i in range(0, v_num)]
    
    for i in range(0, v_num):
        U += tao[i] * A[i]

    loss1 = lambda_ * torch.pow(torch.norm(train_datall @ W - U @ train_datall @ W, 2), 2) + \
        torch.pow(torch.norm(W, 2), 2) + \
            gamma1 * torch.norm(S, 1) + \
                gamma2 * torch.pow( torch.norm((train_datall @ (W + S) - train_weak_target), 2), 2)
    # print("-" * 100)
    # print(train_datall)
    # print(train_datall @ W)
    for i in range(0, v_num):
        loss1 += torch.pow(torch.norm(train_data[i] - A[i] @ train_data[i], 2), 2) + \
            tao[i] * torch.pow(torch.norm(U - A[i], 2), 2)

    print("loss1=", loss1.item())

    for iter in range(0, maxiter):
        print("Current Iter: ", iter)
        
        parameters = torch.nn.Parameter(W.clone().detach())
        U_for_W, train_weak_target_for_W = map(lambda x: x.clone().detach(), (U, train_weak_target))
        train_datall_for_W = train_datall.clone().detach()
        non_parameters = (U_for_W, train_datall_for_W, train_weak_target_for_W, lambda_, gamma2)
        # W = L_BFGS(parameters, non_parameters, W_loss_fun, maxiter, "W")
        if args.dataset in ['emotions_new.mat', 'yeast_new.mat']:
            W = L_BFGS(parameters, non_parameters, W_loss_fun_linear_and_l2, maxiter_bfgs, "W")
        else:
            W = GD(parameters, non_parameters, W_loss_fun_linear_and_l2, maxiter_bfgs, "W")

        parameters = torch.nn.Parameter(U.clone().detach())
        train_datall_for_U = train_datall.clone().detach()
        W_for_U = W.clone().detach()
        train_data_for_U = list(map(lambda x: x.clone().detach(), train_data))
        A_for_U = list(map(lambda x: x.clone().detach(), A))
        non_parameters = (train_data_for_U, train_datall_for_U, W_for_U, A_for_U, lambda_, tao)
        U = L_BFGS(parameters, non_parameters, U_loss_fun, maxiter, "U")
        # normalize
        U = U / torch.norm(U, p=1, dim=1, keepdim=True)

        t = [ 1 / torch.pow(torch.norm(A[i] - U, 2), 2) for i in range(0, v_num)]
        tao = [ t[i] / sum(t) for i in range(0, v_num)]

        for i in range(0, v_num):
            parameters = torch.nn.Parameter(A[i].clone().detach())
            non_parameters = tuple(map(lambda x: x.clone().detach(), (train_data[i], U, tao[i])))
            A[i] = L_BFGS(parameters, non_parameters, A_loss_fun, maxiter, "A[{}]".format(i))
            # normalize
            A[i] = normalize_A_i(A[i])


        loss1 = lambda_ * torch.pow(torch.norm(train_datall @ W - U @ train_datall @ W, 2), 2) + \
                    torch.pow(torch.norm(W, 2), 2) + \
                        gamma1 * torch.norm(S, 1) + \
                            gamma2 * torch.pow( torch.norm((train_datall @ (W + S) - train_weak_target), 2), 2)

        for i in range(0, v_num):
            loss1 += torch.pow(torch.norm(train_data[i] - A[i] @ train_data[i], 2), 2) + \
            tao[i] * torch.pow(torch.norm(U - A[i], 2), 2)
            
        print("loss1=", loss1.item() / n_num)

    return W.cpu(), S.cpu()

def Parameter(x):
    x = x.clone().detach().cpu()
    x = torch.nn.Parameter(x)

    return x.cuda() 

def L_BFGS(parameters, non_parameters, loss_fun, maxiter=100, flag=""):
    optim = torch.optim.LBFGS([parameters], lr=args.lr_lbfgs, max_iter=maxiter)
    optim.zero_grad()
    def closure():
        loss = loss_fun(parameters, *non_parameters)
        loss.backward(retain_graph=True)
        return loss
    optim.step(closure=closure)
    loss = loss_fun(parameters, *non_parameters).item()
    return parameters

def GD(parameters, non_parameters, loss_fun, maxiter=100, flag=""):
    optim = torch.optim.Adam([parameters], lr=args.lr_gd)
    for _ in range(0, 100):
        optim.zero_grad()
        loss = loss_fun(parameters, *non_parameters)
        loss.backward()
        optim.step()
    return parameters

def A_loss_fun(A_i, train_data_i, U, tau_i):
    return 0.001 * (torch.pow( torch.norm(train_data_i - A_i @ train_data_i, 2), 2) + tau_i * torch.pow( torch.norm(U -A_i, 2), 2)) / (A_i.shape[0])

def U_loss_fun(U, train_data, train_datall, W, A, lambda_, tao):
    v_num = len(train_data)
    U_loss = 0
    for i in range(0, v_num):
        U_loss += tao[i] * torch.pow( torch.norm(U-A[i], 2), 2)
    
    return 0.001 * (U_loss + lambda_ * torch.pow( torch.norm(train_datall @ W - U @ train_datall @ W), 2)) / (U.shape[0])
    
 
def W_loss_fun(W, U, train_datall, train_weak_target, lambda_, gamma2):
    
    return (1/ len(train_datall)) * (lambda_*torch.pow( torch.norm(train_datall @ W - U @ train_datall @ W, 2), 2) + \
            torch.pow( torch.norm(W, 2), 2) + \
                gamma2*torch.pow( torch.norm(train_datall @ W - train_weak_target, 2), 2) )

def W_loss_fun_linear_and_l2(W, U, train_datall, train_weak_target, lambda_, gamma2):
    
    return  (1 / len(train_datall)) * (gamma2 * torch.pow( torch.norm(W, 2), 2) + \
                torch.pow( torch.norm(train_datall @ W - train_weak_target, 2), 2) )



def initialize_A(trainFea):
    num1 = 1 / torch.sqrt( torch.sum(trainFea * trainFea, dim=1))
    num1 = torch.where(torch.isnan(num1), torch.full_like(num1, 1), num1)
    num1 = torch.where(torch.isinf(num1), torch.full_like(num1, 1), num1)
    trainFea = torch.diag(num1) @ trainFea
    A_i = trainFea @ trainFea.T

    # set diagonal value to zero
    A_i = A_i - torch.diag_embed(torch.diag(A_i))
    # row normalization
    A_i = normalize_A_i(A_i)

    return A_i

def normalize_A_i(A_i):
    v = torch.sum(A_i, dim=1)
    D = torch.diag(v)
    A_i = torch.inverse(D) @ A_i
    return A_i





if __name__ == "__main__":
    main()



