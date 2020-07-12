from alexnet import Alexnet
from preprocess import preprocess_train, preprocess_test
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import argparse

# eagerly (declared only once)
tf.compat.v1.enable_eager_execution(device_policy=tfe.DEVICE_PLACEMENT_SILENT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="cifar data path", default="../data/cifar-10-batches-py")
    parser.add_argument("--epochs", type=int, help="number of learning epoch, default is 10", default=10)
    parser.add_argument("--saving", help="wheter saving or not(each verbose iteration)", action="store_true")
    parser.add_argument("--batch_size", type=int, help="batch size(default is 32)", default=32)
    parser.add_argument("--verbose", type=int, help="verbosity cycle(default is 1 epoch)", default=1)
    parser.add_argument("--no_tqdm", help="whether to use tqdm process bar", action="store_true")
    parser.add_argument("--lr", type=float, help="learning rate, default is 0.001", default=1e-3)

    args = parser.parse_args()
    dirname = args.data

    X_train, y_train = preprocess_train(dirname)
    X_test, y_test = preprocess_test(dirname)

    device = 'gpu:0' if tfe.num_gpus() > 0 else 'cpu:0'
    alex_model = Alexnet(learning_rate=args.lr, device_name=device)
    alex_model(tf.convert_to_tensor(X_train[:1]), True)
    alex_model.summary()
    # alex_model.load() - for loading your latest saved model 

    if args.no_tqdm:
        tqdm_option = None
    else:
        tqdm_option = "normal"
    alex_model.fit(X_train, y_train, X_test, y_test, epochs=args.epochs, verbose=args.verbose,
                   batch_size=args.batch_size, saving=args.saving, tqdm_option=tqdm_option)


if __name__ == "__main__":
    main()
