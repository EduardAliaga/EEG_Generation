import argparse
from train import train
from validate import validate

def main(data_path, num_epochs, batch_size, learning_rate, model_path):
    # Train the model
    train(data_path, num_epochs, batch_size, learning_rate, model_path)
    # Validate the model
    validate(data_path, model_path, batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and validate a simple model on synthetic data.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data directory')
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and evaluation')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for the optimizer')
    parser.add_argument('--model_path', type=str, required=True, help='Path to save the trained model')

    args = parser.parse_args()
    
    main(args.data_path, args.num_epochs, args.batch_size, args.learning_rate, args.model_path)
