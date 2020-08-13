# gcn
python train.py #0.79
python train.py --dataset enzymes --dropout 0   #0.36
# graphsage
python train.py --model_type GraphSage --hidden_dim 256 --dropout 0.6 --weight_decay 5e-4 #0.78
python train.py --model_type GraphSage --hidden_dim 256 --dataset enzymes --dropout 0 --epochs 600  #0.75
# gat
python train.py --model_type GAT --lr 0.001 --hidden_dim 64 --dropout 0.6 --weight_decay 5e-4   #0.815
python train.py --model_type GAT --dataset enzymes --dropout 0 --epochs 600 #0.61

# env setup
pip install torch-scatter==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install torch-sparse==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install torch-cluster==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install torch-spline-conv==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install torch-geometric