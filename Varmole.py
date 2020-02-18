import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/gpfs/software/Anaconda3/lib/python3.6/site-packages')
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients
from captum.attr import LayerConductance
from captum.attr import NeuronConductance
import csv
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class GRNeQTL(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GRNeQTL, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, adj):
        return input.matmul(self.weight.t() * adj) + self.bias

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    
class Net(nn.Module):
    def __init__(self, adj, D_in, H1, H2, H3, D_out):
        super(Net, self).__init__()
        self.adj = adj
        self.GRNeQTL = GRNeQTL(D_in, H1)
        self.linear2 = torch.nn.Linear(H1, H2)
        self.linear3 = torch.nn.Linear(H2, H3)
        self.linear4 = torch.nn.Linear(H3, D_out)

    def forward(self, x):
        h1 = self.GRNeQTL(x, self.adj).relu()
        h2 = self.linear2(h1).relu()
        h3 = self.linear3(h2).relu()
        y_pred = self.linear4(h3).sigmoid()
        return y_pred
    

parser = argparse.ArgumentParser(description='''Predicts schizophrenia disease outcome of input SNP and gene expressions
                                                using Varmole, based on PyTorch Version 0.4.1
                                                Requierements: PyTorch Version 0.4.1''')
parser.add_argument('infile', metavar='F', type=str, nargs='+',
                    help='CSV file containing both SNP and gene expressions')
args = parser.parse_args()


print('Loading input SNP and gene expressions')
obs = pd.read_csv(args.infile[0], index_col=0).sort_index()
adj = pd.read_pickle('adj.pkl').sort_index()
X_test = obs.values.T
print('Succesfully loaded SNP and gene expressions')

print('Loading model...')
model = torch.load('varmole_model.pth', map_location=torch.device('cpu'))

print('Making predictions')
with torch.no_grad():
    x_tensor_test = torch.from_numpy(X_test).float().to(device)
    model.eval()
    yhat = model(x_tensor_test)
    y_hat_class = np.where(yhat.cpu().numpy()<0.5, 0, 1)

ls = list(zip(obs.columns.tolist()[:223], y_hat_class.tolist()))
df = pd.DataFrame(ls, columns = ['individualID', 'diagnosis'])

outFile = '{0}_Predictions.csv'.format(args.infile[0].split('.')[0])

print('Saving predictions to file {}'.format(outFile))
df.to_csv(outFile)


print('Interpreting SNP and TF importance...')
feature_names = list(obs.index)

model.adj = model.adj.cpu()
ig = IntegratedGradients(model.cpu())

test_input_tensor = torch.from_numpy(X_test).type(torch.FloatTensor)
attr, delta = ig.attribute(test_input_tensor, return_convergence_delta=True)
attr = attr.detach().numpy()

importances = dict(zip(feature_names, np.mean(abs(attr), axis=0)))

outFile = '{0}_FeatureImportance.csv'.format(args.infile[0].split('.')[0])

print('Saving SNP and TF importance to file {}'.format(outFile))
with open(outFile, 'w') as f:
    for key in importances.keys():
        f.write("%s,%s\n"%(key,importances[key]))
        
print('Interpreting gene importance...')
cond = LayerConductance(model, model.GRNeQTL)

cond_vals = cond.attribute(test_input_tensor)
cond_vals = cond_vals.detach().numpy()

importances_layer1 = dict(zip(adj.columns.tolist(), np.mean(abs(cond_vals), axis=0)))

outFile = '{0}_GeneImportance.csv'.format(args.infile[0].split('.')[0])

print('Saving gene importance to file {}'.format(outFile))
with open(outFile, 'w') as f:
    for key in importances_layer1.keys():
        f.write("%s,%s\n"%(key,importances_layer1[key]))
        

neuron_cond = NeuronConductance(model, model.GRNeQTL)

outFile = '{0}_ConnectionImportance.csv'.format(args.infile[0].split('.')[0])
with open(outFile, 'w') as f:
    print('Interpreting eQTL and GRN connections importance...')
    for gene in adj.columns.tolist():
        neuron_cond_vals = neuron_cond.attribute(test_input_tensor, neuron_index=adj.columns.tolist().index(gene))
        importances_neuron = dict(zip(feature_names, abs(neuron_cond_vals.mean(dim=0).detach().numpy())))
        importances_neuron = {key:val for key, val in importances_neuron.items() if val != 0}
        
        for key in importances_neuron.keys():
            f.write("%s,%s,%s\n"%(gene,key,importances_neuron[key]))

print('Succesfully saved eQTL and GRN connections importance to file {}'.format(outFile))
