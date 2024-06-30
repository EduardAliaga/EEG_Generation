import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from data_loader import get_dataloader
from models import RNNModel

def validate(data_path, model_path, batch_size):
    dataloader = get_dataloader(data_path, batch_size)

    # Initialize the model
    input_size = 3 * 396
    hidden_size = 128
    output_size = 3 * 396
    model = RNNModel(input_size, hidden_size, output_size)
    
    # Load the trained model weights
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Define the loss function
    criterion = nn.MSELoss()

    # Evaluation
    total_loss = 0
    predictions = []
    actuals = []
    with torch.no_grad():
        for x_batch, VE_batch in dataloader:
            # Flatten the batch data for the linear model
            x_batch = x_batch.view(x_batch.size(0), -1)
            VE_batch = VE_batch.view(VE_batch.size(0), -1)

            outputs = model(x_batch)
            loss = criterion(outputs, VE_batch)
            total_loss += loss.item()
            predictions.append(outputs)
            actuals.append(VE_batch)
        
    average_loss = total_loss / len(dataloader)
    print(f'Average Evaluation Loss: {average_loss:.4f}')

    # Concatenate all predictions and actuals for plotting
    predictions = torch.cat(predictions).cpu().numpy()
    actuals = torch.cat(actuals).cpu().numpy()

    # Plot the predictions vs the corresponding labels
    plt.figure(figsize=(10, 6))
    plt.plot(predictions[0], label='Predicted', color='b')
    plt.plot(actuals[0], label='Actual', color='r')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('Prediction vs Actual')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Validate a trained model on synthetic data.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data directory')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    validate(args.data_path, args.model_path, args.batch_size)

