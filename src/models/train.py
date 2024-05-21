import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_dataloader
from models import RNNModel

def train(data_path, num_epochs, batch_size, learning_rate, model_path):
    dataloader = get_dataloader(data_path, batch_size)

    # Initialize the model, loss function, and optimizer
    input_size = 3 * 396
    hidden_size = 128
    output_size = 3 * 396
    model = RNNModel(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for x_batch, VE_batch in dataloader:
            # Flatten the batch data for the linear model
            x_batch = x_batch.view(x_batch.size(0), -1)
            VE_batch = VE_batch.view(VE_batch.size(0), -1)
            
            # Forward pass
            outputs = model(x_batch)
            loss = criterion(outputs, VE_batch)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Print loss for each epoch
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # Save the trained model
    torch.save(model.state_dict(), model_path)
    print("Training complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train a simple model on synthetic data.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data directory')
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for the optimizer')
    parser.add_argument('--model_path', type=str, required=True, help='Path to save the trained model')

    args = parser.parse_args()
    
    train(args.data_path, args.num_epochs, args.batch_size, args.learning_rate, args.model_path)

