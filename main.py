import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
import sys
import os

# Import attention models
from models.multihead.multihead import MultiAttention
from models.selfhead import SelfAttention


class AttentionTrainer:
    def __init__(self, model_type, embed_dim, num_heads=None, learning_rate=0.001, batch_size=32, seq_length=10):
        """
        Initialize the attention model trainer
        
        Args:
            model_type (str): 'multihead' or 'selfhead'
            embed_dim (int): Embedding dimension
            num_heads (int): Number of attention heads (only for multihead)
            learning_rate (float): Learning rate for optimizer
            batch_size (int): Batch size for training
            seq_length (int): Sequence length for training data
        """
        self.model_type = model_type
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.seq_length = seq_length
        
        # Initialize model
        self.model = self._create_model()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        print(f"Model initialized: {model_type}")
        print(f"Embedding dimension: {embed_dim}")
        if num_heads:
            print(f"Number of heads: {num_heads}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _create_model(self):
        """Create the specified attention model"""
        if self.model_type.lower() == 'multihead':
            if self.num_heads is None:
                raise ValueError("num_heads must be specified for multihead attention")
            if self.embed_dim % self.num_heads != 0:
                raise ValueError(f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})")
            return MultiAttention(emd_dim=self.embed_dim, num_heads=self.num_heads)
        
        elif self.model_type.lower() == 'selfhead':
            return SelfAttention(emd_dim=self.embed_dim)
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}. Use 'multihead' or 'selfhead'")
    
    def generate_training_data(self, num_samples=1000):
        """Generate synthetic training data"""
        # Generate random input sequences
        x = torch.randn(num_samples, self.seq_length, self.embed_dim)
        
        # For demonstration, we'll use the model to generate targets
        # In real scenarios, you'd have actual target data
        with torch.no_grad():
            y = self.model(x)
        
        return x, y
    
    def train(self, epochs=100, save_path=None):
        """Train the attention model"""
        # Generate training data
        x_train, y_train = self.generate_training_data()
        
        # Create data loader
        dataset = TensorDataset(x_train, y_train)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Training samples: {len(x_train)}")
        print(f"Batch size: {self.batch_size}")
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(batch_x)
                loss = self.criterion(output, batch_y)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
        
        print(f"Training completed! Final loss: {avg_loss:.6f}")
        
        # Save model if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'model_type': self.model_type,
                'embed_dim': self.embed_dim,
                'num_heads': self.num_heads,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'final_loss': avg_loss
            }, save_path)
            print(f"Model saved to: {save_path}")
    
    def evaluate(self, test_samples=100):
        """Evaluate the trained model"""
        self.model.eval()
        
        # Generate test data
        x_test, y_test = self.generate_training_data(test_samples)
        x_test, y_test = x_test.to(self.device), y_test.to(self.device)
        
        with torch.no_grad():
            predictions = self.model(x_test)
            loss = self.criterion(predictions, y_test)
        
        print(f"\nEvaluation Results:")
        print(f"Test loss: {loss.item():.6f}")
        print(f"Input shape: {x_test.shape}")
        print(f"Output shape: {predictions.shape}")
        
        return loss.item()


def main():
    parser = argparse.ArgumentParser(description='Train Attention Models')
    parser.add_argument('--model', type=str, choices=['multihead', 'selfhead'], 
                       default='multihead', help='Type of attention model to train')
    parser.add_argument('--embed_dim', type=int, default=64, 
                       help='Embedding dimension')
    parser.add_argument('--num_heads', type=int, default=8, 
                       help='Number of attention heads (for multihead only)')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Batch size')
    parser.add_argument('--seq_length', type=int, default=10, 
                       help='Sequence length')
    parser.add_argument('--epochs', type=int, default=100, 
                       help='Number of training epochs')
    parser.add_argument('--save_path', type=str, default=None, 
                       help='Path to save the trained model')
    parser.add_argument('--interactive', action='store_true', 
                       help='Run in interactive mode to choose parameters')
    
    args = parser.parse_args()
    
    if args.interactive:
        print("=== Attention Model Training Interface ===\n")
        
        # Model selection
        print("Available models:")
        print("1. MultiHead Attention")
        print("2. Self Attention")
        choice = input("Choose model (1 or 2): ").strip()
        
        if choice == "1":
            args.model = 'multihead'
        elif choice == "2":
            args.model = 'selfhead'
        else:
            print("Invalid choice. Using MultiHead Attention.")
            args.model = 'multihead'
        
        # Parameter input
        try:
            args.embed_dim = int(input(f"Enter embedding dimension (default: 64): ") or "64")
            if args.model == 'multihead':
                args.num_heads = int(input(f"Enter number of heads (default: 8): ") or "8")
            args.learning_rate = float(input(f"Enter learning rate (default: 0.001): ") or "0.001")
            args.batch_size = int(input(f"Enter batch size (default: 32): ") or "32")
            args.seq_length = int(input(f"Enter sequence length (default: 10): ") or "10")
            args.epochs = int(input(f"Enter number of epochs (default: 100): ") or "100")
            
            save_choice = input("Save model? (y/n, default: n): ").strip().lower()
            if save_choice == 'y':
                args.save_path = input("Enter save path (default: models/trained_model.pth): ") or "models/trained_model.pth"
        except ValueError as e:
            print(f"Invalid input: {e}")
            return
    
    # Validate parameters
    if args.model == 'multihead' and args.embed_dim % args.num_heads != 0:
        print(f"Error: embed_dim ({args.embed_dim}) must be divisible by num_heads ({args.num_heads})")
        return
    
    # Create trainer and train
    try:
        trainer = AttentionTrainer(
            model_type=args.model,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads if args.model == 'multihead' else None,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            seq_length=args.seq_length
        )
        
        trainer.train(epochs=args.epochs, save_path=args.save_path)
        trainer.evaluate()
        
    except Exception as e:
        print(f"Error during training: {e}")
        return


if __name__ == "__main__":
    main()