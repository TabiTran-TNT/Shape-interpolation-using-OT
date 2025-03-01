from utils import *
from model import *
import torch_directml
from train_UOT import train
from main import *

num_epochs = 200
args = {"batch_size": 1024, "shuffle": True}
args = TempArgs()

# matplotlib.use('TkAgg')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

train_dataloader, test_dataloader = loadMNIST(args, True, 1000)


def visualize_path(args, train_dataloader, my_autoencoder, T=None, flip_=False):
    steps = (torch.arange(T + 1, device=args.device) / T).unsqueeze(-1) if T is not None else args.steps

    with torch.no_grad():
        for image_batch in train_dataloader:

            # image_batch = train_data
            image_batch = image_batch[0].to(args.device)
            image_batch_code = my_autoencoder.encode(image_batch)
            mu = image_batch_code[0]
            sigma = image_batch_code[1]
            image_batch_recon = my_autoencoder(image_batch)

            num_b = min(args.N_Selected, image_batch.shape[0]) - 1
            image_path = []
            for i in range(num_b):

                image_path.append(image_batch[i].unsqueeze(0))  # default type is torch.float32
                for step in steps[1:T]:
                    # print(step)
                    mu_i = mu[i] + step * (mu[i + 1] - mu[i])
                    sigma_i = sigma[i] + step * (sigma[i + 1] - sigma[i])
                    #code_sequence_i = image_batch_code[i] + step * (image_batch_code[i + 1] - image_batch_code[i])

                    rep = my_autoencoder.reparameterize(mu_i, sigma_i)
                    image_interpolated_i = my_autoencoder.decode(torch.unsqueeze(rep, 0))

                    image_path.append(image_interpolated_i.clone())

            image_path.append(image_batch[num_b].unsqueeze(0))

            image_path = torch.cat(image_path)
            if flip_:
                image_path = 1 - image_path
            make_grid_show(image_path, pad_value=args.pad_value)
            return None

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.eConv1 = nn.Conv2d(1,6,4)
        self.eConv2 = nn.Conv2d(6,12,5)
        self.ePool1 = nn.MaxPool2d(2,2)
        self.eConv3 = nn.Conv2d(12,24,5)
        self.ePool2 = nn.MaxPool2d(2,2)
        self.eF1 = nn.Linear(24*4*4,180)
        self.eMu = nn.Linear(180,180)
        self.eSigma = nn.Linear(180,180)

        self.dConvT1 = nn.ConvTranspose2d(180,200,4)
        self.dBatchNorm1 = nn.BatchNorm2d(200)
        self.dConvT2 = nn.ConvTranspose2d(200,120,6,2)
        self.dBatchNorm2 = nn.BatchNorm2d(120)
        self.dConvT3 = nn.ConvTranspose2d(120,60,6,2)
        self.dBatchNorm3 = nn.BatchNorm2d(60)
        self.dConvT4 = nn.ConvTranspose2d(60,1,5,1)

    def encode(self,x):
        x = self.eConv1(x)
        x = F.relu(x)
        x = self.eConv2(x)
        x = F.relu(x)
        x = self.ePool1(x)
        x = self.eConv3(x)
        x = F.relu(x)
        x = self.ePool2(x)
        x = x.view(x.size()[0], -1)
        x = self.eF1(x)
        mu = self.eMu(x)
        sigma = self.eSigma(x)
        return((mu,sigma))

    # From https://github.com/pytorch/examples/blob/master/vae/main.py
    def reparameterize(self,mu,sigma):
        std = torch.exp(0.5*sigma)
        eps = torch.randn_like(std)
        return (mu + eps*std)

    def decode(self,x):
        x = torch.reshape(x,(x.shape[0],180,1,1))
        x = self.dConvT1(x)
        x = self.dBatchNorm1(x)
        x = F.relu(x)
        x = self.dConvT2(x)
        x = self.dBatchNorm2(x)
        x = F.relu(x)
        x = self.dConvT3(x)
        x = self.dBatchNorm3(x)
        x = F.relu(x)
        x = self.dConvT4(x)
        x = torch.sigmoid(x)
        return(x)

    def forward(self,x):
        mu,sigma = self.encode(x)
        z = self.reparameterize(mu,sigma)
        x_gen = self.decode(z)
        return((x_gen,mu,sigma))


# From https://github.com/pytorch/examples/blob/master/vae/main.py
# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(x, x_gen, mu, sigma):
    #print(x.shape)
    #print(x_gen.shape)
    BCE = F.binary_cross_entropy(x_gen, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())

    return BCE + KLD


vae = VAE()
# vae.train()
vae.load_state_dict(torch.load("vae_1.pth", weights_only=True))
vae.eval()

# visualize_path(args, test_dataloader, vae, flip_= False, T=19)

def visualize_result():
    T = 9
    flip_ = False
    steps = (torch.arange(T + 1, device=args.device) / T).unsqueeze(-1) if T is not None else args.steps

    with torch.no_grad():
        for image_batch in train_dataloader:

            # import torch
            # from torchvision import datasets, transforms
            #
            # # Define a transform to convert the data to tensor
            # transform = transforms.ToTensor()
            #
            # # Load the MNIST training dataset
            # # train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
            #
            # image = transforms.ToPILImage()(image_batch[0][0])  # Save the image as a PNG file
            # image.save('compare-1.png')
            # image = transforms.ToPILImage()(image_batch[0][1])  # Save the image as a PNG file
            # image.save('compare-2.png')

            # image_batch = train_data
            image_batch = image_batch[0].to(args.device)
            image_batch_code = vae.encode(image_batch)
            image_batch_recon = vae(image_batch)

            # num_b = min(args.N_Selected, image_batch.shape[0]) - 1
            num_b = 5
            image_path = []
            for i in range(num_b):

                image_path.append(image_batch[i].unsqueeze(0))  # default type is torch.float32
                for step in steps[0:T]:
                    # print(step)
                    code_sequence_i = image_batch_code[i] + step * (image_batch_code[i + 1] - image_batch_code[i])
                    image_interpolated_i = vae.decode(torch.unsqueeze(code_sequence_i, 0))

                    image_path.append(image_interpolated_i.clone())
                image_path.append(image_batch[i + 1].unsqueeze(0))

            # image_path.append(image_batch[num_b].unsqueeze(0))
            image_path = torch.cat(image_path)
            if flip_:
                image_path = 1 - image_path
            fig, axes = plt.subplots(num_b, T + 2, figsize=(5 * (T + 2), num_b * 1.4))
            # fig.subplots_adjust(hspace=0)
            fig.subplots_adjust(wspace=0.05)
            print(list(enumerate(axes.flat)))
            for i, ax in enumerate(axes.flat):
                ax.imshow(np.squeeze(image_path[i].detach().cpu().numpy(), 0), 'gray')

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
            fig.savefig("main-result-1.png", bbox_inches='tight')
            break
            # make_grid_show(image_path, pad_value=args.pad_value)

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

            image_batch_code = vae.encode(image_batch)
            mu = image_batch_code[0]
            sigma = image_batch_code[1]
            image_batch_recon = vae(image_batch)

            num_b = 1
            image_path = []
            for i in range(num_b):

                image_path.append(image_batch[i].unsqueeze(0))  # default type is torch.float32
                for step in steps[0:T]:
                    # print(step)
                    mu_i = mu[i] + step * (mu[i + 1] - mu[i])
                    sigma_i = sigma[i] + step * (sigma[i + 1] - sigma[i])
                    # code_sequence_i = image_batch_code[i] + step * (image_batch_code[i + 1] - image_batch_code[i])

                    rep = vae.reparameterize(mu_i, sigma_i)
                    image_interpolated_i = vae.decode(torch.unsqueeze(rep, 0))

                    image_path.append(image_interpolated_i.clone())
                image_path.append(image_batch[i + 1].unsqueeze(0))

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
                ax.imshow(np.squeeze(image_path[i].detach().cpu().numpy(), 0), 'gray')

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
            fig.savefig("test-result-2.png", bbox_inches='tight')
            break
            # make_grid_show(image_path, pad_value=args.pad_value)

visualize_compare()

# for i, data in enumerate(train_dataloader, 0):
#     # get the inputs; data is a list of [inputs, labels]
#     images = data[0].to(device)
#     #images = data[0]
#     # zero the parameter gradients
#     outputs = vae(images)
#     loss = loss_function(images, outputs[0], outputs[1], outputs[2])
#     print(loss)
#
# vae.to(device)
#
# optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
#
# for epoch in range(1000):
#
#     running_loss = 0.0
#     for i, data in enumerate(train_dataloader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         images = data[0].to(device)
#         #images = data[0]
#         # zero the parameter gradients
#         optimizer.zero_grad()
#
#         # forward + backward + optimize
#         outputs = vae(images)
#         loss = loss_function(images, outputs[0], outputs[1], outputs[2])
#         loss.backward()
#         optimizer.step()
#
#         # print statistics
#         running_loss += loss.item()
#         if i % 500 == 499:    # print every 500 mini-batches
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 2000))
#             running_loss = 0.0
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
#
#     # PATH = 'vae_checkpoints/'
#     # torch.save(vae.state_dict(), PATH+str(epoch)+".pt")
#     # imsave("actual/" + str(epoch) + ".png", torchvision.utils.make_grid(images.cpu()))
#     # imsave("recon/" + str(epoch) + ".png",torchvision.utils.make_grid(outputs[0].detach().cpu()))
#
# torch.save(vae.state_dict(), "vae_2.pth")
# print('Finished Training')
