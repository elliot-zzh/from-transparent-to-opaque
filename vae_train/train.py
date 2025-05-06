import gc
import torch
from vae_train.dataset import (
    train_loader,
    test_loader,
    model,
    train_sampler,
    test_sampler,
)
from vae_train.parameters import (
    device,
    accelerator,
    num_epochs,
    gradient_accumulation_steps,
    log_interval,
    hidden_layer_num,
    save_interval,
)
from vae_train.vae_model import optimizer, vae, lossf
import matplotlib.pyplot as plt


def cleanup():
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def train_vae(epochs=num_epochs, collect_data=True):
    # Prepare models, optimizer for distributed training
    vae, optimizer, train_loader, test_loader = accelerator.prepare(
        vae, optimizer, train_loader, test_loader
    )

    train_loss = []
    test_loss = []

    for epoch in range(epochs):
        # Set epoch for distributed sampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for step, (input_ids, attn_mask) in enumerate(train_loader):
            try:
                cleanup()

                input_ids = input_ids.to(device)
                attn_mask = attn_mask.to(device)

                with accelerator.accumulate(vae):
                    optimizer.zero_grad(set_to_none=True)

                    with accelerator.autocast():
                        with torch.no_grad():
                            hidden = (
                                model(
                                    input_ids=input_ids,
                                    attention_mask=attn_mask,
                                    output_hidden_states=True,
                                    return_dict=True,
                                )
                                .hidden_states[hidden_layer_num]
                                .float()
                            )
                        loss = lossf(vae(hidden), hidden)

                    accelerator.backward(loss)

                if (step + 1) % (gradient_accumulation_steps * log_interval) == 0:
                    print(
                        f"Epoch {epoch + 1}, Step {step + 1}, Train Loss: {loss.item():.3f}"
                    )

            except KeyboardInterrupt:
                cleanup()

        if (step + 1) % gradient_accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            print(f"Epoch {epoch + 1}, Step {step + 1}, Train Loss: {loss.item():.3f}")
            if collect_data:
                train_loss.append(loss.item())

        count = 0
        loss = 0
        vae.eval()
        for step, (input_ids, attn_mask) in enumerate(test_loader):
            try:
                cleanup()

                input_ids = input_ids.to(device)
                attn_mask = attn_mask.to(device)

                with torch.no_grad():
                    with torch.amp.autocast(
                        device_type=str(device), dtype=torch.float16
                    ):
                        hidden = (
                            model(
                                input_ids=input_ids,
                                attention_mask=attn_mask,
                                output_hidden_states=True,
                                return_dict=True,
                            )
                            .hidden_states[hidden_layer_num]
                            .float()
                        )
                    loss += lossf(vae(hidden), hidden)
                    count += 1

            except KeyboardInterrupt:
                cleanup()

        print(f"Epoch {epoch + 1}, Test Loss: {loss.item() / count:.3f}")

        if collect_data:
            test_loss.append(loss.item() / count)

        if (epoch + 1) % save_interval == 0:
            # Wait for all processes to finish before saving
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                # Save checkpoint
                unwrapped_vae = accelerator.unwrap_model(vae)
                torch.save(unwrapped_vae.state_dict(), f"vae_epoch{epoch + 1}.pth")
                print(f"Checkpoint saved at epoch {epoch + 1}")

    # Wait for all processes to finish before saving final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_vae = accelerator.unwrap_model(vae)
        torch.save(unwrapped_vae.state_dict(), "vae_final.pth")

    print("Training complete. Final model saved as vae_final.pth")

    if collect_data and accelerator.is_main_process:
        print("Now plotting the loss")

        plt.plot(train_loss, label="Train Loss")
        plt.plot(test_loss, label="Test Loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("VAE Training Loss")
        plt.legend()
        plt.savefig("vae_training_loss.pdf")
        plt.savefig("vae_training_loss.png")

        print("Loss plot saved as vae_training_loss.pdf and vae_training_loss.png")
