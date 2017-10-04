# datasets

Deep Learning has a few classical datasets. But they are each from different institutions and in their own particular format. So this repo is ment to make a standard way to download and load each of these datasets.

    import datasets
    
    train_set, test_set = datasets.load_Cifar_10()
    
So far the datasets supported are:
1. Cifar 10
2. SVHN
