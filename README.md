# Hierarchical_FL
Implementation of HierFAVG algorithm in [Client-Edge-Cloud Hierarchical Federated Learning](https://arxiv.org/abs/1905.06641) with Pytorch.

For running HierFAVG with mnist :
```
python3 hierfavg 
--dataset mnist 
--model lenet 
--num_clients 50 
--num_edges 5 
--frac 1 
--num_local_update 60 
--num_edge_aggregation 1 
--num_communication 100
--batch_size 20 
--iid 0
--edgeiid 1
--show_dis 1
--lr 0.01
--lr_decay 0.995
--lr_decay_epoch 1
--momentum 0
--weight_decay 0
```


# Comparison of Gaussian and Laplacian Mechanism :

(i) The vector-valued Laplace mechanism requires the use of L1 sensitivity, while the vector-valued Gaussian mechanism allows the use of either L1 or L2 sensitivity.

(ii) This is a major strength of the Gaussian mechanism. For applications in which L2 sensitivity is much lower than L1 sensitivity, the Gaussian mechanism allows adding much less noise.

# Laplacian  mechanism major drawbacks :
(i) It is only really good for low sensitivity queries. (usually with L1 sensitivity)

(ii) Need large epsilon values (aka privacy budget) if you are going to fire off a bunch of queries. (Large epsilon values produce results that are less accurate in order to achieve the privacy guarantee).

(iii) Provides solution to handle numeric queries, but cannot be applied to the non-numeric valued queries like “what is the most common nationality in this room”: Chinese/Indian/American…

# Gaussian mechanism  major drawbacks :
(i) It requires the use of the the relaxed (ε, δ)-differential privacy definition,

(ii) It’s less accurate than the Laplace mechanism.