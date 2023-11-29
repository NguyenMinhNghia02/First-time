import joblib

def train_and_save_model():
    # Assume you have a trained model
    model = ...  # Your trained model

    # Save the model to the 'output' directory
    output_path = "/kaggle/working/output/model.pkl"
    joblib.dump(model, output_path)
    
    print(f"Model saved to: {output_path}")

if __name__ == "__main__":
    train_and_save_model()