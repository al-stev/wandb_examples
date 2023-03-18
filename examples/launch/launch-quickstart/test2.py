import wandb
import os
import random
import time
import math

# SETTINGS
PROJECT = "launch-test" 
ENTITY = "allanstevenson" 
MODEL_NAME = "trained-model"
STEPS = 20
# Initialize a new W&B run to track this job
run = wandb.init(
    project=PROJECT, 
    entity=ENTITY,
    config={
            "learning_rate": 0.01 * random.random(),
            "batch_size": 128,
            "momentum": 0.1 * random.random(),
            "dropout": 0.4 * random.random()
            })
run.log_code()
  # Log metrics and checkpoints at N steps
displacement1 = random.random() * 2
displacement2 = random.random() * 4

for step in range(STEPS):
  wandb.log({
      "acc": .1 + 0.4 * (math.log(1 + step + random.random()) + 
             random.random() * run.config.learning_rate + random.random() + 
             displacement1 + random.random() * run.config.momentum),
      "val_acc": .1 + 0.5 * (math.log(1 + step + random.random()) + 
                 random.random() * run.config.learning_rate - random.random() + 
                 displacement1),
      "loss": .1 + 0.08 * (3.5 - math.log(1 + step + random.random()) + 
              random.random() * run.config.momentum + random.random() + 
              displacement2),
      "val_loss": .1 + 0.04 * (4.5 - math.log(1 + step + random.random()) + 
                  random.random() * run.config.learning_rate - random.random() + 
                  displacement2),
  })
  time.sleep(1)