# test.py

import random
import wandb

settings = wandb.Settings()
settings.update({"enable_job_creation": True, "save_code": True})
def run_training_run(epochs, lr):
    print(f"Training for {epochs} epochs with learning rate {lr}")

    run = wandb.init(
        # Set the project where this run will be logged
        project="job_example",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": lr,
            "epochs": epochs,
        })

    offset = random.random() / 5
    print(f"lr: {lr}")

    for epoch in range(2, epochs):
        # simulating a training run
        acc = 1 - 2 ** -epoch - random.random() / epoch - offset
        loss = 2 ** -epoch + random.random() / epoch + offset
        print(f"epoch={epoch}, acc={acc}, loss={loss}")
        wandb.log({"acc": acc, "loss": loss})

    
    #run.log_code() # log code so we can reuse this experiment


run_training_run(epochs=1, lr=0.01)