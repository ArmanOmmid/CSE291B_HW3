
import torch
import torch.nn as nn
from lstm_data import create_batch_mask

def run_epoch(
        generator,
        discriminator,
        data_loader,
        criterion,
        optimizer_generator,
        optimizer_discriminator,
        device,
        learn,
        mse_support = False,
        warm_up = False,
        mask = False
    ):
        
        if mse_support or warm_up:
            mse_criteron = nn.MSELoss()

        dataset_len = len(data_loader.dataset)
        epoch_g_loss = 0.
        epoch_d_loss = 0.
        
        for iter, real_x in enumerate(data_loader):

            with torch.set_grad_enabled(learn):

                real_x = real_x.to(device)

                batch_size, seq_length = real_x.size(0), real_x.size(1)

                if warm_up:
                    optimizer_generator.zero_grad()
                    z = torch.randn(batch_size, seq_length, generator.h_dim, device=device)
                    fake_x = generator(z)
                    g_loss = mse_criteron(fake_x, real_x)
                    g_loss.backward()
                    optimizer_generator.step()
                    with torch.no_grad():
                        epoch_g_loss += g_loss.detach().item() * batch_size
                    continue

                real_labels = torch.ones(batch_size, 1, device=device)
                fake_labels = torch.zeros(batch_size, 1, device=device)
                
                optimizer_generator.zero_grad()
                z = torch.randn(batch_size, seq_length, generator.h_dim, device=device)
                fake_x = generator(z)

                if mask:
                    real_x = real_x * create_batch_mask(real_x)
                    fake_x = fake_x * create_batch_mask(fake_x)

                optimizer_discriminator.zero_grad()
                real_loss = criterion(discriminator(real_x), real_labels)
                fake_loss = criterion(discriminator(fake_x.detach()), fake_labels)
                d_loss = (real_loss + fake_loss) / 2
                if learn:
                    d_loss.backward()
                    optimizer_discriminator.step()

                g_loss = criterion(discriminator(fake_x), real_labels)
                if learn:
                    if mse_support:
                        g_mse_loss = mse_criteron(fake_x, real_x)
                        (g_loss + g_mse_loss).backward()
                    else:
                        g_loss.backward()
                    optimizer_generator.step()

                with torch.no_grad():
                    epoch_g_loss += g_loss.detach().item() * batch_size
                    epoch_d_loss += d_loss.detach().item() * batch_size

        avg_epoch_g_loss = epoch_g_loss / dataset_len
        avg_epoch_d_loss = epoch_d_loss / dataset_len

        return avg_epoch_g_loss, avg_epoch_d_loss
