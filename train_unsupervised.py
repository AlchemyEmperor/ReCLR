
import os
import argparse
import torch
import torch.optim
import numpy as np
import random


parser = argparse.ArgumentParser(description="Training script for the unsupervised phase via self-supervision")
parser.add_argument("--seed", default=1024, type=int,
                    help="Seed for Numpy and PyTorch. Default: -1 (None)")
parser.add_argument("--epoch_start", default=0, type=int,
                    help="Epoch to start learning from, used when resuming")
parser.add_argument("--epochs", default=200, type=int,
                    help="Total number of epochs")
parser.add_argument("--dataset", default="PBC",
                    help="Dataset: BCISC1K, PBC, Raabin-WBC"
                    )
parser.add_argument("--backbone", default="conv4",
                    help="Backbone: conv4, resnet|8|32|34|56")
parser.add_argument("--method", default="HeCLR",
                    help="Model: standard, HeCLR, ReCLR")
parser.add_argument("--data_size", default=128, type=int,
                    help="Size of the mini-batch")
parser.add_argument("--K", default=8, type=int,
                    help="Total number of augmentations (K), sed only in RelationNet")
parser.add_argument("--aggregation", default="cat",
                    help="Aggregation function used in RelationNet: sum, mean, max, cat")
parser.add_argument("--id", default="",
                    help="Additional string appended when saving the checkpoints")
parser.add_argument("--save_file", default="1_6_pretrain",
                    help="location of a save file")
parser.add_argument("--checkpoint", default="",
                    help="location of a checkpoint file, used to resume training")
parser.add_argument("--num_workers", default=4, type=int,
                    help="Number of torchvision workers used to load data (default: 8)")
parser.add_argument("--gpu", default="0", type=str,
                    help="GPU id in case of multiple GPUs")

args = parser.parse_args()

if (args.id != ""):
    header = str(args.method) + "_" + str(args.id) + "_" + str(args.dataset) + "_" + str(
        args.backbone) + "_seed_" + str(args.seed)
else:
    header = str(args.method) + "_" + str(args.dataset) + "_" + str(args.backbone) + "_seed_" + str(args.seed)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


if (args.seed >= 0):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("[INFO] Setting SEED: " + str(args.seed))
else:
    print("[INFO] Setting SEED: None")

def main():
    if (torch.cuda.is_available() == False): print("[WARNING] CUDA is not available.")
    if (args.backbone == "conv4"):
        from backbones.conv4 import Conv4

        feature_extractor = Conv4(flatten=True)
    elif (args.backbone == "resnet8"):
        from backbones.resnet_small import ResNet, BasicBlock

        feature_extractor = ResNet(BasicBlock, [1, 1, 1], channels=[16, 32, 64], flatten=True)
    elif (args.backbone == "resnet32"):
        from backbones.resnet_small import ResNet, BasicBlock

        feature_extractor = ResNet(BasicBlock, [5, 5, 5], channels=[16, 32, 64], flatten=True)
    elif (args.backbone == "resnet56"):
        from backbones.resnet_small import ResNet, BasicBlock

        feature_extractor = ResNet(BasicBlock, [9, 9, 9], channels=[16, 32, 64], flatten=True)
    elif (args.backbone == "resnet34"):
        from backbones.resnet_large import ResNet, BasicBlock

        feature_extractor = ResNet(BasicBlock, layers=[3, 4, 6, 3], zero_init_residual=False,
                                   groups=1, width_per_group=64, replace_stride_with_dilation=None,
                                   norm_layer=None)
    else:
        raise RuntimeError("[ERROR] the backbone " + str(args.backbone) + " is not supported.")

    tot_params = sum(p.numel() for p in feature_extractor.parameters() if p.requires_grad)
    print("[INFO]", str(str(args.backbone)), "loaded in memory.")
    print("[INFO] Feature size:", str(feature_extractor.feature_size))
    print("[INFO] Feature extractor TOT trainable params: " + str(tot_params))
    print("[INFO] Found " + str(torch.cuda.device_count()) + " GPU(s) available.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device type: " + str(device))


    from datamanager import DataManager

    manager = DataManager(args.seed)
    num_classes = manager.get_num_classes(args.dataset)

    if (args.method == "standard"):
        from methods.standard import StandardModel

        model = StandardModel(feature_extractor, num_classes, tot_epochs=args.epochs)
        if (args.dataset == "stl10"):
            train_transform = manager.get_train_transforms("finetune", args.dataset)
        else:
            train_transform = manager.get_train_transforms("standard", args.dataset)
        val_loader = manager.get_val_loader(args.dataset, data_size=args.data_size, num_workers=4)
        test_loader = manager.get_test_loader(args.dataset, data_size=args.data_size, num_workers=4)

        train_loader, _ = manager.get_train_loader(dataset=args.dataset,
                                                   data_type="single",
                                                   data_size=args.data_size,
                                                   train_transform=train_transform,
                                                   repeat_augmentations=None,
                                                   num_workers=args.num_workers,
                                                   drop_last=False)
    elif (args.method == "ReCLR"):
        from methods.ReCLR import Model

        model = Model(feature_extractor)
        train_transform = manager.get_train_transforms(args.method, args.dataset)
        train_loader, _ = manager.get_train_loader(dataset=args.dataset,
                                                   data_type="multi",
                                                   data_size=args.data_size,
                                                   train_transform=train_transform,
                                                   repeat_augmentations=2,
                                                   num_workers=args.num_workers,
                                                   drop_last=False)
    elif (args.method == "HeCLR"):
        from methods.HeCLR import Model

        model = Model(feature_extractor)
        train_transform = manager.get_train_transforms(args.method, args.dataset)
        train_loader, _ = manager.get_train_loader(dataset=args.dataset,
                                                   data_type="multi",
                                                   data_size=args.data_size,
                                                   train_transform=train_transform,
                                                   repeat_augmentations=2,
                                                   num_workers=args.num_workers,
                                                   drop_last=False)
    else:
        raise RuntimeError("[ERROR] the model " + str(args.method) + " is not supported.")

    model.to(device)

    if (args.checkpoint != ""):
        print("Loading checkpoint: " + str(args.checkpoint))
        model.load(args.checkpoint)
        print("Loading checkpoint: Done!")
    #-------------------------

    if not os.path.exists("./checkpoint/" + str(args.method) + "/" + str(args.dataset) + "/" + str(args.save_file)):
        os.makedirs("./checkpoint/" + str(args.method) + "/" + str(args.dataset) + "/" + str(args.save_file))
    log_file = "./checkpoint/" + str(args.method) + "/" + str(args.dataset) + "/" + str(
        args.save_file) + "/log_" + header + ".cvs"
    with open(log_file, "w") as myfile:
        myfile.write("epoch,loss,score" + "\n")  # create a new log file (it destroys the previous one)

    best_acc = 0
    best_epoch = 0

    for epoch in range(args.epoch_start, args.epochs):
        print("Epoch[{}/{}] [{}] [{}]:".format(epoch, args.epochs, args.dataset, args.method))
        loss_train, accuracy_train = model.train(epoch, train_loader)  # <-- Each model must have a "train" method
        with open(log_file, "a") as myfile:
            myfile.write(str(epoch) + "," + str(loss_train) + "," + str(accuracy_train) + "\n")
        if ((epoch + 1) % 10 == 0):
            checkpoint_path = "./checkpoint/" + str(args.method) + "/" + str(args.dataset) + "/" + str(
                args.save_file) + "/" + header + "_epoch_" + str(epoch + 1) + ".tar"
            print("[INFO] Saving in:", checkpoint_path)
            model.save(checkpoint_path)
        if epoch % 10 == 0:
            if (args.method == "standard"):
                # For the standard supervised method, it estimates now the test accuracy
                loss_val, val_acc = model.test(val_loader)
                print("Val loss: " + str(loss_val))
                print("Val accuracy: " + str(val_acc) + "%")
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_epoch = epoch
                    loss_test, accuracy_test = model.test(test_loader)
                print("Test loss: " + str(loss_test))
                print("Test accuracy: " + str(accuracy_test) + "%")
                print("Best epoch: {}".format(best_epoch))


if __name__ == "__main__": main()
