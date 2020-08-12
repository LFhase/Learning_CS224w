# gcn
python train.py 
python train.py --dataset enzymes
# graphsage
python train.py --model_type GraphSage --lr 0.01 --hidden_dim 256 --dropout 0.6 --weight_decay 5e-4
python train.py --model_type GraphSage --dataset enzymes
# gat

# env setup
pip install torch-scatter==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install torch-sparse==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install torch-cluster==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install torch-spline-conv==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install torch-geometric