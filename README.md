## A Multi-Mode Modulator for Multi-Domain Few-Shot Classification (ICCV'21)

Official code implementation for
[A Multi-Mode Modulator for Multi-Domain Few-Shot Classification](https://csyanbin.github.io/papers/ICCV2021_tri-M.pdf) (ICCV 2021)

### Requirements:

    pip install tensorflow==2.4
    pip install torch torchvision
    pip install gin-config
    pip install simclr
    GPU with 16GB+ memory
    
### Set Enviroment:
    (1) Download&Process Meta-Dataset following: 
        https://github.com/google-research/meta-dataset#downloading-and-converting-datasets
        
    (2) Download&Process 3 extra datasets (Mnist, Cifar10, Cifar100) following: 
        https://github.com/cambridge-mlg/cnaps --> Installation --> 3. Install additional test datasets (MNIST, CIFAR10, CIFAR100)
        
    (3) Set the PROJECT_ROOT, META_DATASET_ROOT, and META_RECORDS_ROOT in datareader/path.py
        ulimit -n 50000
        
### Training:
    python run_triM.py --learning_rate 2e-3 --feature_adaptation MahSpecCoop -T 150000 --tasks_per_batch=16 
    
### Testing:
    python run_triM.py --learning_rate 2e-3 --feature_adaptation MahSpecCoop -T 150000 --tasks_per_batch=16 --test_model_path TEST_MODEL_CKPT_PATH --mode test --test_datasets=traffic_sign 


### Bibtex
If you use this code or results for your research, please consider citing:
````
@INPROCEEDINGS{yanbin21triM,
  title     = {A Multi-Mode Modulator for Multi-Domain Few-Shot Classification},
  author    = {Liu, Yanbin and Lee, Juho and Zhu, Linchao and Chen, Ling and Shi, Humphrey and Yang, Yi},
  booktitle = {ICCV},
  year      = {2021}
}
````

  
 
