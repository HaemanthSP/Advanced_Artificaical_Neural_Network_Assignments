import os
import trainer
import tester

#
# General configurations

# Create an empty dictionary to store the configurations
CONFIG = dict()

CONFIG["source_path"] = os.path.realpath("../")

CONFIG["architecture_name"] = "architecture_0"
CONFIG["model_name"] = "test_model"

CONFIG["device"] = "CPU"  # can be "CPU" or "GPU"
CONFIG["mode"] = 1  # 1 to train the model, 2 to test the model


#
# Training parameters

CONFIG["save_model"] = True
# Set the following value True to continue training of an existing model
CONFIG["continue_training"] = False

CONFIG["epochs"] = 100
CONFIG["seq_len"] = 40
CONFIG["learning_rate"] = 0.001

#
# Testing parameters

CONFIG["teacher_forcing_steps"] = 20
CONFIG["closed_loop_steps"] = 100

#
# Run configuration

# Hide the GPU(s) in case the user specified to use the CPU
if CONFIG["device"] == "CPU":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Run the training/testing
if CONFIG["mode"] == 1:
    print("Running script 'trainer.py'")
    trainer.run_training(cfg=CONFIG)
else:
    print("Running script 'tester.py'")
    tester.run_testing(cfg=CONFIG)
