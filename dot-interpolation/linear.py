from utils import *
from model import *
import torch_directml
from train_UOT import train
from main import *

num_epochs = 200
args = {"batch_size": 512, "shuffle": True}
args = TempArgs()

train_dataloader, test_dataloader = loadMNIST(args, True, 1000)

def visualize_path(args, train_dataloader, my_autoencoder, T=None, flip_=False):
    steps = (torch.arange(T + 1, device=args.device) / T).unsqueeze(-1) if T is not None else args.steps

    with torch.no_grad():
        for image_batch in train_dataloader:

            # image_batch = train_data
            image_batch = image_batch[0].to(args.device)
            # image_batch_code = my_autoencoder.encode(image_batch)
            # mu = image_batch_code[0]
            # sigma = image_batch_code[1]
            # image_batch_recon = my_autoencoder(image_batch)

            num_b = min(args.N_Selected, image_batch.shape[0]) - 1
            image_path = []
            for i in range(num_b):

                image_path.append(image_batch[i])  # default type is torch.float32
                print(image_batch[i].unsqueeze(0).shape)
                for step in steps[1:T]:
                    # print(step)
                    # mu_i = mu[i] + step * (mu[i + 1] - mu[i])
                    # sigma_i = sigma[i] + step * (sigma[i + 1] - sigma[i])
                    # #code_sequence_i = image_batch_code[i] + step * (image_batch_code[i + 1] - image_batch_code[i])
                    #
                    # rep = my_autoencoder.reparameterize(mu_i, sigma_i)
                    image_interpolated_i = image_batch[i] + step * (image_batch[i + 1] - image_batch[i])
                    print(image_interpolated_i.shape)

                    image_path.append(image_interpolated_i.clone())

            image_path.append(image_batch[num_b])

            image_path = torch.cat(image_path)
            if flip_:
                image_path = 1 - image_path
            make_grid_show(image_path, pad_value=args.pad_value)
            return None

def visualize_compare():
    T = 9
    flip_ = False
    steps = (torch.arange(T + 1, device=args.device) / T).unsqueeze(-1) if T is not None else args.steps

    with torch.no_grad():
        for image_batch in train_dataloader:
            print(image_batch[0].shape)
            from torchvision import transforms
            from PIL import Image

            # Define a transform to convert the image to a tensor
            transform = transforms.ToTensor()

            # Apply the transform to the image
            image_batch = []
            image = Image.open("compare-1.png")
            image_tensor = transform(image)
            image_batch.append(image_tensor)
            image = Image.open("compare-2.png")
            image_tensor = transform(image)
            image_batch.append(image_tensor)
            image_batch = torch.stack(image_batch).to(args.device)
            print(image_batch.shape)

            # image_batch_code = vae.encode(image_batch)
            # mu = image_batch_code[0]
            # sigma = image_batch_code[1]
            # image_batch_recon = vae(image_batch)

            num_b = 1
            image_path = []
            for i in range(num_b):

                image_path.append(image_batch[i])  # default type is torch.float32
                for step in steps[0:T]:
                    # print(step)
                    # mu_i = mu[i] + step * (mu[i + 1] - mu[i])
                    # sigma_i = sigma[i] + step * (sigma[i + 1] - sigma[i])
                    # #code_sequence_i = image_batch_code[i] + step * (image_batch_code[i + 1] - image_batch_code[i])
                    #
                    # rep = my_autoencoder.reparameterize(mu_i, sigma_i)
                    image_interpolated_i = image_batch[i] + step * (image_batch[i + 1] - image_batch[i])
                    print(image_interpolated_i.shape)

                    image_path.append(image_interpolated_i.clone())
                image_path.append(image_batch[i + 1])

            # image_path.append(image_batch[num_b].unsqueeze(0))
            image_path = torch.cat(image_path)
            if flip_:
                image_path = 1 - image_path
            print(len(image_path))
            fig, axes = plt.subplots(num_b, T + 2, figsize=(5 * (T + 2), num_b * 1.4))
            # fig.subplots_adjust(hspace=0)
            fig.subplots_adjust(wspace=0.05)
            print(list(enumerate(axes.flat)))
            for i, ax in enumerate(axes.flat):
                ax.imshow(image_path[i].detach().cpu().numpy(), 'gray')

                ax.grid(False)
                ax.set_xticks([])
                ax.set_yticks([])

                if i % (T + 2) == 0 or i % (T + 2) == (T + 2) - 1:
                    frame_color = 'red'
                    ax.spines['top'].set_color(frame_color)
                    ax.spines['bottom'].set_color(frame_color)
                    ax.spines['left'].set_color(frame_color)
                    ax.spines['right'].set_color(frame_color)

                    frame_width = 3  # Increase the width
                    ax.spines['top'].set_linewidth(frame_width)
                    ax.spines['bottom'].set_linewidth(frame_width)
                    ax.spines['left'].set_linewidth(frame_width)
                    ax.spines['right'].set_linewidth(frame_width)
                elif i % (T + 2) == 1 or i % (T + 2) == (T + 2) - 2:
                    frame_color = 'blue'
                    ax.spines['top'].set_color(frame_color)
                    ax.spines['bottom'].set_color(frame_color)
                    ax.spines['left'].set_color(frame_color)
                    ax.spines['right'].set_color(frame_color)

                    frame_width = 3  # Increase the width
                    ax.spines['top'].set_linewidth(frame_width)
                    ax.spines['bottom'].set_linewidth(frame_width)
                    ax.spines['left'].set_linewidth(frame_width)
                    ax.spines['right'].set_linewidth(frame_width)

            plt.grid(False)
            plt.show()
            fig.savefig("test-result-3.png", bbox_inches='tight')
            break
            # make_grid_show(image_path, pad_value=args.pad_value)

visualize_compare()